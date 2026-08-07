"""
pipeline.py — all core logic for the Assignment PDF -> Q-A pair extractor.

Pipeline:
  PDF
   -> hybrid text extraction (PyMuPDF native text, Mistral OCR fallback for
      scanned/handwritten/garbled pages)
   -> Stage A: regex/structural parser (deterministic, multilingual markers,
      zero hallucination risk since text is sliced not generated)
   -> Stage B: Groq LLM semantic segmentation for whatever Stage A couldn't
      confidently resolve (verbatim-extraction prompt, JSON-only output)
   -> anti-hallucination verification: fuzzy-match every q/a back against the
      exact source text it came from (rapidfuzz); untraceable pairs are kept
      but flagged verified=False / low confidence, never silently trusted
   -> cross-page dedup (overlap windows can double-extract the same pair)
   -> DocumentResult (Pydantic), exportable as full JSON or minimal [{q,a}]

See README.md for the full rationale behind each design choice.
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
from pydantic import BaseModel, Field
from rapidfuzz import fuzz


# =============================================================================
# Output schema
# =============================================================================

class QAPair(BaseModel):
    id: int
    q: str                          # verbatim question text, exactly as in the PDF
    a: Optional[str] = None         # verbatim answer text, exactly as in the PDF (None if unanswered)
    page_start: int
    page_end: int
    language: Optional[str] = None
    source: str = Field(description="'regex' (deterministic) or 'llm' (Groq semantic fallback)")
    confidence: float = Field(ge=0.0, le=1.0)
    verified: bool = Field(
        description="True if q/a text was fuzzy-matched back against the raw source "
                    "text above the acceptance threshold (anti-hallucination check)."
    )

    def to_minimal(self) -> dict:
        return {"q": self.q, "a": self.a}


class DocumentResult(BaseModel):
    source_file: str
    num_pages: int
    languages_detected: List[str] = []
    qa_pairs: List[QAPair] = []
    unmatched_low_confidence: int = 0

    def to_minimal(self) -> list:
        return [p.to_minimal() for p in self.qa_pairs]


# =============================================================================
# Hybrid text extraction (native PyMuPDF + Mistral OCR fallback)
# =============================================================================
#
# Rationale: most "assignment PDFs" are digitally typed and already have a
# perfect embedded text layer -- OCR-ing them wastes money/latency and can
# introduce errors a good text layer wouldn't have. Scanned/handwritten pages
# have no usable text layer and genuinely need OCR. Mistral OCR is used here
# because it accepts PDFs directly, is strong on multilingual/mixed-script
# and handwritten documents, and returns structure-aware markdown (headers,
# lists) which the regex stage below relies on.

@dataclass
class PageText:
    page_num: int   # 1-indexed
    text: str
    method: str      # 'native' | 'ocr' | 'native_low_confidence_no_ocr_key'


_GARBAGE_RATIO_THRESHOLD = 0.35
_MIN_NATIVE_CHARS = 25


def _looks_garbled(text: str) -> bool:
    if len(text.strip()) < _MIN_NATIVE_CHARS:
        return True
    weird = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in ".,;:!?()[]{}'\"-/\\%&#@*+=<>|"))
    return (weird / max(len(text), 1)) > _GARBAGE_RATIO_THRESHOLD


def extract_native(pdf_path: str) -> List[PageText]:
    """Fast pass: pull embedded text layer per page via PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = [PageText(page_num=i + 1, text=page.get_text("text"), method="native")
             for i, page in enumerate(doc)]
    doc.close()
    return pages


def pages_needing_ocr(pages: List[PageText]) -> List[int]:
    return [p.page_num for p in pages if _looks_garbled(p.text)]


def ocr_pages_with_mistral(pdf_path: str, page_numbers: List[int], api_key: str,
                            model: str = "mistral-ocr-latest") -> dict:
    """Sends the PDF to Mistral OCR; returns {page_num: markdown_text} for requested pages."""
    if not page_numbers:
        return {}

    from mistralai import Mistral

    client = Mistral(api_key=api_key)
    with open(pdf_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    result = client.ocr.process(
        model=model,
        document={"type": "document_url", "document_url": f"data:application/pdf;base64,{b64}"},
        include_image_base64=False,
    )

    wanted = set(page_numbers)
    return {idx: page.markdown for idx, page in enumerate(result.pages, start=1) if idx in wanted}


def build_document_pages(pdf_path: str, mistral_api_key: Optional[str] = None) -> List[PageText]:
    """Returns one PageText per page: native extraction where possible, Mistral OCR where needed."""
    native_pages = extract_native(pdf_path)
    needs_ocr = pages_needing_ocr(native_pages)

    if needs_ocr and mistral_api_key:
        ocr_results = ocr_pages_with_mistral(pdf_path, needs_ocr, mistral_api_key)
        for p in native_pages:
            if p.page_num in ocr_results:
                p.text = ocr_results[p.page_num]
                p.method = "ocr"
    elif needs_ocr and not mistral_api_key:
        for p in native_pages:
            if p.page_num in needs_ocr:
                p.method = "native_low_confidence_no_ocr_key"

    return native_pages


# =============================================================================
# Stage A: regex / structural parser
# =============================================================================
#
# Multilingual question/answer markers. Extend these as new assignment
# formats are encountered -- cheapest lever for shifting more of the document
# into the free/deterministic path instead of the LLM fallback.

_Q_MARKERS = [
    r"Q(?:uestion)?\.?\s*#?\s*\d+", r"Ques\.?\s*\d+",
    r"प्रश्न\s*[\d०-९]+",     # Hindi
    r"سؤال\s*\d+",             # Arabic
    r"\d+[\.\)]\s+",           # bare "1." / "1)" numbering
]
_A_MARKERS = [
    r"A(?:ns(?:wer)?)?\.?\s*#?\s*\d*\s*[:\-\.]",
    r"उत्तर\s*[:\-]?",         # Hindi
    r"جواب\s*[:\-]?",          # Arabic/Urdu
    r"Solution\s*[:\-]",
]
_Q_PATTERN = re.compile(r"(?:^|\n)\s*(?:" + "|".join(_Q_MARKERS) + r")", re.IGNORECASE)
_A_PATTERN = re.compile(r"(?:^|\n)\s*(?:" + "|".join(_A_MARKERS) + r")", re.IGNORECASE)


@dataclass
class RawSegment:
    text: str
    page_start: int
    page_end: int


def regex_segment(pages: List[PageText]) -> Tuple[List[QAPair], List[RawSegment]]:
    """
    Deterministic Q/A segmentation. Returns (confident_pairs, leftover_segments_for_llm).
    Finds question-marker start positions; for each, looks for the next answer-marker
    before the next question-marker. Blocks with no marker at all become leftovers.
    """
    full_text = ""
    offsets = []
    for p in pages:
        offsets.append((len(full_text), p.page_num))
        full_text += p.text + "\n"

    def page_for_offset(pos: int) -> int:
        pg = offsets[0][1]
        for off, page_num in offsets:
            if off <= pos:
                pg = page_num
            else:
                break
        return pg

    q_starts = [m.start() for m in _Q_PATTERN.finditer(full_text)]
    pairs: List[QAPair] = []
    leftovers: List[RawSegment] = []

    if not q_starts:
        return pairs, []  # no structure at all -> caller chunks whole doc to LLM

    q_starts.append(len(full_text))
    qid = 1
    is_last_block = lambda i: q_starts[i + 1] == len(full_text)

    for i in range(len(q_starts) - 1):
        block = full_text[q_starts[i]:q_starts[i + 1]]
        a_match = _A_PATTERN.search(block)
        if a_match:
            q_text = block[:a_match.start()].strip()
            a_text = block[a_match.end():]

            trailing_leftover = ""
            confidence = 0.9
            if is_last_block(i):
                # KNOWN LIMITATION: the final block has no next question marker to
                # bound the answer, and PDF text extraction often loses blank-line
                # spacing between paragraphs. We cap the captured length, route
                # overflow to the LLM stage, and downgrade confidence rather than
                # trust an unbounded capture. See README "Known limitations".
                _MAX_LAST_ANSWER_CHARS = 200
                para_break = re.search(r"\n\s*\n", a_text)
                if para_break and para_break.start() <= _MAX_LAST_ANSWER_CHARS:
                    trailing_leftover = a_text[para_break.end():].strip()
                    a_text = a_text[:para_break.start()]
                elif len(a_text.strip()) > _MAX_LAST_ANSWER_CHARS:
                    trailing_leftover = a_text[_MAX_LAST_ANSWER_CHARS:].strip()
                    a_text = a_text[:_MAX_LAST_ANSWER_CHARS]
                    confidence = 0.55

            a_text = a_text.strip()
            q_text = _Q_PATTERN.sub("", q_text, count=1).strip(" :.-\n")
            if q_text and a_text:
                start_pg = page_for_offset(q_starts[i])
                end_pg = page_for_offset(q_starts[i] + a_match.end())
                pairs.append(QAPair(
                    id=qid, q=q_text, a=a_text,
                    page_start=start_pg, page_end=end_pg,
                    source="regex", confidence=confidence, verified=(confidence >= 0.8),
                ))
                qid += 1
                if trailing_leftover:
                    leftovers.append(RawSegment(
                        text=trailing_leftover,
                        page_start=end_pg, page_end=page_for_offset(q_starts[i + 1]),
                    ))
                continue

        start_pg = page_for_offset(q_starts[i])
        end_pg = page_for_offset(q_starts[i + 1])
        leftovers.append(RawSegment(text=block, page_start=start_pg, page_end=end_pg))

    return pairs, leftovers


def chunk_pages_with_overlap(pages: List[PageText], overlap_chars: int = 200) -> List[RawSegment]:
    """Splits by page but prepends a tail of the previous page so a Q/A split
    across a page boundary isn't cut apart by the chunk boundary."""
    segments = []
    prev_tail = ""
    for p in pages:
        text = (prev_tail + "\n" + p.text) if prev_tail else p.text
        segments.append(RawSegment(text=text, page_start=p.page_num, page_end=p.page_num))
        prev_tail = p.text[-overlap_chars:]
    return segments


# =============================================================================
# Stage B: Groq LLM fallback for ambiguous / unstructured segments
# =============================================================================

_LLM_SYSTEM_PROMPT = """You extract question-answer pairs from assignment documents.

STRICT RULES:
1. Output ONLY valid JSON: {"pairs": [{"q": "...", "a": "..."}, ...]}. No prose, no markdown fences.
2. "q" and "a" MUST be copied VERBATIM from the input text. Do not paraphrase, translate,
   summarize, correct spelling/grammar, or invent any content that is not literally present.
3. If a question has no visible answer in the text, set "a" to null. Never fabricate an answer.
4. Preserve the original language of the text exactly as written -- do not translate.
5. If the text contains no discernible question-answer structure, return {"pairs": []}.
6. Do not merge unrelated questions and do not split a single question into multiple pairs.
"""


def llm_segment(segment: RawSegment, groq_api_key: str,
                 model: str = "llama-3.3-70b-versatile") -> List[dict]:
    from groq import Groq

    client = Groq(api_key=groq_api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract Q/A pairs from this text:\n\n{segment.text}"},
        ],
    )
    try:
        return json.loads(resp.choices[0].message.content).get("pairs", [])
    except (json.JSONDecodeError, AttributeError):
        return []


_VERIFY_THRESHOLD = 82  # rapidfuzz partial_ratio (0-100) to count as "traceable to source"


def verify_pair(q_text: str, a_text: Optional[str], source_text: str) -> Tuple[bool, float]:
    """Confirms extracted q/a text is actually present in the source it was drawn
    from, rather than trusting the LLM's output blindly."""
    q_score = fuzz.partial_ratio(q_text, source_text)
    a_score = fuzz.partial_ratio(a_text, source_text) if a_text else 100
    verified = q_score >= _VERIFY_THRESHOLD and a_score >= _VERIFY_THRESHOLD
    return verified, round(min(q_score, a_score) / 100, 3)


def run_llm_stage(leftovers: List[RawSegment], groq_api_key: str, start_id: int,
                   model: str = "llama-3.3-70b-versatile") -> List[QAPair]:
    pairs = []
    qid = start_id
    for seg in leftovers:
        for rp in llm_segment(seg, groq_api_key, model=model):
            q_text = (rp.get("q") or "").strip()
            a_text = rp.get("a")
            a_text = a_text.strip() if isinstance(a_text, str) else None
            if not q_text:
                continue
            verified, confidence = verify_pair(q_text, a_text, seg.text)
            pairs.append(QAPair(
                id=qid, q=q_text, a=a_text,
                page_start=seg.page_start, page_end=seg.page_end,
                source="llm", confidence=confidence, verified=verified,
            ))
            qid += 1
    return pairs


# =============================================================================
# Orchestration: dedup + full pipeline
# =============================================================================

def dedupe(pairs: List[QAPair], threshold: int = 92) -> List[QAPair]:
    """Overlapping page chunks can double-extract the same Q/A; keep the
    higher-confidence copy when two pairs' questions are near-duplicates."""
    kept: List[QAPair] = []
    for p in pairs:
        is_dup = False
        for k in kept:
            if fuzz.ratio(p.q, k.q) >= threshold:
                is_dup = True
                if p.confidence > k.confidence:
                    kept.remove(k)
                    kept.append(p)
                break
        if not is_dup:
            kept.append(p)
    for i, p in enumerate(kept, start=1):
        p.id = i
    return kept


def run_pipeline(pdf_path: str, mistral_key: Optional[str], groq_key: Optional[str],
                  groq_model: str = "llama-3.3-70b-versatile") -> DocumentResult:
    """Full end-to-end pipeline: PDF path in, DocumentResult out."""
    pages = build_document_pages(pdf_path, mistral_api_key=mistral_key)

    regex_pairs, leftovers = regex_segment(pages)

    if not regex_pairs and not leftovers:
        # No structure found anywhere -> fall back to chunked whole-document LLM pass.
        leftovers = chunk_pages_with_overlap(pages) if groq_key else []

    llm_pairs: List[QAPair] = []
    if leftovers and groq_key:
        llm_pairs = run_llm_stage(leftovers, groq_key, start_id=len(regex_pairs) + 1, model=groq_model)

    all_pairs = dedupe(regex_pairs + llm_pairs)
    low_conf = sum(1 for p in all_pairs if not p.verified)

    return DocumentResult(
        source_file=pdf_path,
        num_pages=len(pages),
        qa_pairs=all_pairs,
        unmatched_low_confidence=low_conf,
    )
