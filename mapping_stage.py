"""
Stage 3 — Sequential answer mapping (Groq)

Core guarantee: the LLM NEVER generates answer text. It only returns a
{start_line, end_line} integer range over a pre-numbered array of raw OCR
lines. The actual answer is produced by plain Python slicing
(`"\n".join(lines[start:end+1])`), so it is byte-for-byte identical to the
OCR output. `_verify_slice` re-derives the slice independently and asserts
equality before a record is accepted — this is the "integrity check".

Optimization — cursor-windowed sequential search:
Answers appear in the same order as questions were asked (this is an
assignment booklet, not a jumbled reference doc). So instead of searching
the *entire* answer text for every single question (O(questions x lines)
tokens), each search starts a few lines before wherever the previous
answer ended and scans forward to the end of the document. This:
  - shrinks the average prompt size roughly in half over the whole document
    (first question searches everything, last question searches almost
    nothing), which is where the real token/latency savings come from
  - structurally prevents the model from matching an earlier, already-used
    answer to a later question — the exact "wrong thing gets matched"
    failure mode called out in the requirements
  - still finds one question at a time (never batched), per spec, so the
    model's attention isn't split across multiple questions at once
A small MAPPING_LOOKBACK_LINES window before the cursor absorbs cases
where OCR line-splitting made the previous end_line slightly conservative.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from groq import Groq

from classify_stage import ClassifiedPage, Question
from config import settings
from utils import render_numbered_block, retry_with_backoff, status, text_hash, cache_get, cache_set

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set.")
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


@dataclass
class QAPair:
    question_id: str
    question: str
    answer_raw: str | None
    found: bool
    confidence: float | None
    source_pages: list[int] = field(default_factory=list)
    start_line: int | None = None
    end_line: int | None = None
    integrity_ok: bool = True


_MAPPING_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "start_line": {"type": "integer"},
        "end_line": {"type": "integer"},
        "confidence": {"type": "number"},
    },
    "required": ["found", "start_line", "end_line", "confidence"],
    "additionalProperties": False,
}

_MAPPING_SYSTEM_PROMPT = """You locate where a student's handwritten answer to a specific question begins and ends, within a numbered array of OCR'd lines.
Rules:
- You are given ONE question at a time. Find only that question's answer.
- Lines are numbered "N: <text>". Respond with start_line and end_line as the INCLUSIVE line-number range of the answer.
- Include the full answer: all sub-points, workings, and diagrentions belonging to it, but STOP before the next question's answer begins (a new question number, or a clear topic shift, marks the end).
- Do NOT include the question restatement itself if the student copied it before answering — start at the first line of actual answer content. If the student did not restate the question, start_line is simply the first content line.
- If you cannot find this question's answer in the given window, set found=false and use start_line=end_line=-1.
- Never invent or rewrite text — you are only pointing at line numbers, not producing content."""


def _build_answer_lines(classified_pages: list[ClassifiedPage]) -> tuple[list[str], dict[int, int]]:
    """
    Concatenate all "answer"-labeled pages, in page order, into one flat
    line array. Returns (lines, line_to_page) where line_to_page maps a
    line index to its source page number for traceability.
    """
    lines: list[str] = []
    line_to_page: dict[int, int] = {}
    for page in sorted(classified_pages, key=lambda p: p.page_number):
        if page.label != "answer":
            continue
        for raw_line in page.text.split("\n"):
            line_to_page[len(lines)] = page.page_number
            lines.append(raw_line)
        # A blank separator between pages avoids accidentally fusing the
        # last line of one page with the first line of the next.
        line_to_page[len(lines)] = page.page_number
        lines.append("")
    return lines, line_to_page


def _format_question(q: Question) -> str:
    marks_suffix = f" ({int(q.marks) if float(q.marks).is_integer() else q.marks})" if q.marks else ""
    return f"{q.id}. {q.text}{marks_suffix}"


def _verify_slice(lines: list[str], start: int, end: int, answer_raw: str) -> bool:
    """Independent re-derivation of the slice — the integrity check."""
    expected = "\n".join(lines[start : end + 1])
    return expected == answer_raw


def _search_one(question_text: str, lines: list[str], window_start: int, model: str) -> dict:
    numbered_block = render_numbered_block(lines, window_start, len(lines))
    cache_key = text_hash(f"{model}|{question_text}|{window_start}|{text_hash(numbered_block)}")
    cached = cache_get("mapping", cache_key)
    if cached:
        return cached

    client = _get_client()
    user_prompt = (
        f"QUESTION TO LOCATE:\n{question_text}\n\n"
        f"NUMBERED ANSWER-TEXT LINES (search starts at line {window_start}):\n{numbered_block}"
    )

    def _do():
        resp = client.chat.completions.create(
            model=model,
            temperature=settings.GROQ_TEMPERATURE,
            messages=[
                {"role": "system", "content": _MAPPING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer_span", "schema": _MAPPING_SCHEMA, "strict": True},
            },
        )
        return json.loads(resp.choices[0].message.content)

    result = retry_with_backoff(_do)
    cache_set("mapping", cache_key, result)
    return result


def map_answers(
    questions: list[Question],
    classified_pages: list[ClassifiedPage],
    status_cb: Callable[[str], None] | None = None,
) -> list[QAPair]:
    lines, line_to_page = _build_answer_lines(classified_pages)
    if not lines:
        status(status_cb, "No answer pages detected — cannot map any answers.")
        return [
            QAPair(question_id=q.id, question=_format_question(q), answer_raw=None, found=False, confidence=None)
            for q in questions
        ]

    # Sort questions by id-as-printed order is risky (string sort breaks on
    # "10" vs "2"); instead trust extraction order, which mirrors the
    # question paper's reading order — the same order answers should appear in.
    cursor = 0
    results: list[QAPair] = []
    total = len(questions)

    for idx, q in enumerate(questions, start=1):
        status(status_cb, f"Mapping answer {idx}/{total} — Q{q.id} (searching from line {cursor})…")
        window_start = max(0, cursor - settings.MAPPING_LOOKBACK_LINES)
        raw = _search_one(_format_question(q), lines, window_start, settings.GROQ_MAP_MODEL)

        found = bool(raw.get("found"))
        start_line = raw.get("start_line", -1)
        end_line = raw.get("end_line", -1)
        confidence = raw.get("confidence")

        if not found or start_line < 0 or end_line < start_line or start_line >= len(lines):
            results.append(
                QAPair(
                    question_id=q.id,
                    question=_format_question(q),
                    answer_raw=None,
                    found=False,
                    confidence=confidence,
                )
            )
            continue

        start_line = max(0, min(start_line, len(lines) - 1))
        end_line = max(start_line, min(end_line, len(lines) - 1))
        answer_raw = "\n".join(lines[start_line : end_line + 1])
        integrity_ok = _verify_slice(lines, start_line, end_line, answer_raw)
        if not integrity_ok:
            status(status_cb, f"⚠️ Integrity check failed for Q{q.id} — discarding match.")

        source_pages = sorted({line_to_page[i] for i in range(start_line, end_line + 1) if i in line_to_page})

        results.append(
            QAPair(
                question_id=q.id,
                question=_format_question(q),
                answer_raw=answer_raw.strip("\n") if integrity_ok else None,
                found=integrity_ok,
                confidence=confidence,
                source_pages=source_pages,
                start_line=start_line,
                end_line=end_line,
                integrity_ok=integrity_ok,
            )
        )
        # Sequential cursor advance — next question only searches forward.
        cursor = end_line + 1

    status(status_cb, f"Answer mapping complete — {sum(r.found for r in results)}/{total} matched.")
    return results
