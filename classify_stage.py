"""
Stage 2 — Page classification + canonical question extraction (Groq)

Two token/time optimizations live here:

1. Classification is batched. Instead of 1 LLM call per page (N calls),
   pages are grouped into chunks of CLASSIFY_BATCH_PAGES and classified in
   one call per chunk (N/batch_size calls). This is safe because
   classifying a page as question-paper/admin/answer needs only local
   context, not cross-page reasoning — batching doesn't hurt accuracy here,
   only round-trip count and fixed per-request overhead. Batches run
   concurrently (ThreadPoolExecutor) since they're independent.

2. Question extraction only ever sees pages already labeled
   "question_paper" — every admin/cover/answer page is filtered out before
   it touches this (more expensive) call, cutting prompt tokens
   substantially on a typical booklet where question-paper pages are a
   small minority.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

from groq import Groq

from config import settings
from ocr_stage import OcrPage
from utils import cache_get, cache_set, retry_with_backoff, status, text_hash

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set.")
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


PageLabel = str  # "question_paper" | "admin_cover" | "answer"


@dataclass
class ClassifiedPage:
    page_number: int
    label: PageLabel
    text: str


@dataclass
class Question:
    id: str  # e.g. "1", "1.i", "2.b" — sub-parts get their own id, never merged
    text: str
    marks: float | None
    source_page: int  # page in the question paper this was read from


_CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "page_number": {"type": "integer"},
                    "label": {
                        "type": "string",
                        "enum": ["question_paper", "admin_cover", "answer"],
                    },
                },
                "required": ["page_number", "label"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["pages"],
    "additionalProperties": False,
}

_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "marks": {"type": "number"},
                    "source_page": {"type": "integer"},
                },
                "required": ["id", "text", "marks", "source_page"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}

_CLASSIFY_SYSTEM_PROMPT = """You classify pages from a scanned university assignment/exam booklet.
Labels:
- "question_paper": contains official printed questions set by the university (the text the student must answer).
- "admin_cover": cover page, roll-number/enrollment grid, instructions, blank page, index, or any administrative page with no questions and no student answers.
- "answer": the student's own handwritten (or written) response content, including rough work.
A page can mix content — pick the label matching its DOMINANT content.
Return every page_number you were given, exactly once, in the same order."""

_QUESTION_SYSTEM_PROMPT = """You extract the canonical list of official questions from question-paper pages of a university booklet.
Rules:
- Preserve the exact question wording (do not paraphrase or summarize).
- If a question has sub-parts (i, ii, iii / a, b, c / 1, 2), emit EACH sub-part as its OWN separate entry with its own id (e.g. "3.i", "3.ii"). Never merge sub-parts into one entry.
- id should reflect the question numbering as printed (e.g. "1", "2.a", "Q5(iii)") — keep it short and use it consistently.
- marks: the mark value printed next to the question if present, else 0.
- source_page: the page_number (as given in the input) where this question's text appears.
- Ignore instructions, headers, and administrative text — only real questions."""


def _call_groq_json(system_prompt: str, user_prompt: str, schema: dict, schema_name: str, model: str) -> dict:
    client = _get_client()

    def _do():
        resp = client.chat.completions.create(
            model=model,
            temperature=settings.GROQ_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
        )
        return json.loads(resp.choices[0].message.content)

    return retry_with_backoff(_do)


def _classify_batch(pages: list[OcrPage]) -> list[ClassifiedPage]:
    key = text_hash("|".join(f"{p.page_number}:{p.text}" for p in pages))
    cached = cache_get("classify_batch", key)
    if cached:
        text_by_page = {p.page_number: p.text for p in pages}
        return [
            ClassifiedPage(page_number=r["page_number"], label=r["label"], text=text_by_page[r["page_number"]])
            for r in cached
        ]

    user_prompt = "\n\n".join(
        f"=== page_number: {p.page_number} ===\n{p.text[:4000]}" for p in pages
    )
    result = _call_groq_json(
        _CLASSIFY_SYSTEM_PROMPT, user_prompt, _CLASSIFY_SCHEMA, "page_classification", settings.GROQ_CLASSIFY_MODEL
    )
    text_by_page = {p.page_number: p.text for p in pages}
    classified = [
        ClassifiedPage(page_number=r["page_number"], label=r["label"], text=text_by_page.get(r["page_number"], ""))
        for r in result["pages"]
        if r["page_number"] in text_by_page
    ]
    cache_set("classify_batch", key, [{"page_number": c.page_number, "label": c.label} for c in classified])
    return classified


def classify_pages(
    pages: list[OcrPage], status_cb: Callable[[str], None] | None = None
) -> list[ClassifiedPage]:
    batch_size = settings.CLASSIFY_BATCH_PAGES
    batches = [pages[i : i + batch_size] for i in range(0, len(pages), batch_size)]

    status(status_cb, f"Classifying {len(pages)} pages in {len(batches)} batched calls…")
    results: list[ClassifiedPage] = []
    with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as pool:
        futures = {pool.submit(_classify_batch, b): b for b in batches}
        for fut in as_completed(futures):
            results.extend(fut.result())

    results.sort(key=lambda c: c.page_number)
    status(status_cb, "Page classification complete.")
    return results


def extract_questions(
    classified_pages: list[ClassifiedPage], status_cb: Callable[[str], None] | None = None
) -> list[Question]:
    qp_pages = [p for p in classified_pages if p.label == "question_paper"]
    if not qp_pages:
        status(status_cb, "No question-paper pages detected — nothing to extract.")
        return []

    status(status_cb, f"Extracting canonical question list from {len(qp_pages)} question-paper page(s)…")
    key = text_hash("|".join(f"{p.page_number}:{p.text}" for p in qp_pages))
    cached = cache_get("questions", key)
    if cached:
        return [Question(**q) for q in cached]

    user_prompt = "\n\n".join(f"=== page_number: {p.page_number} ===\n{p.text}" for p in qp_pages)
    result = _call_groq_json(
        _QUESTION_SYSTEM_PROMPT, user_prompt, _QUESTION_SCHEMA, "question_extraction", settings.GROQ_EXTRACT_MODEL
    )
    questions = [
        Question(id=q["id"], text=q["text"], marks=q.get("marks"), source_page=q["source_page"])
        for q in result["questions"]
    ]
    cache_set("questions", key, [q.__dict__ for q in questions])
    status(status_cb, f"Extracted {len(questions)} question(s) (sub-parts counted individually).")
    return questions
