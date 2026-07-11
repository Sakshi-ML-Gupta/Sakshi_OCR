"""
Orchestrator — wires Stage 1 (OCR) -> Stage 2 (classify + extract) ->
Stage 3 (sequential mapping) into one call, and serializes both the raw
OCR JSON and the final Q&A JSON for the frontend to offer as downloads.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Callable

from classify_stage import classify_pages, extract_questions
from mapping_stage import map_answers
from ocr_stage import run_ocr


def run_pipeline(
    pdf_bytes: bytes,
    filename: str = "document.pdf",
    status_cb: Callable[[str], None] | None = None,
) -> dict:
    """
    Returns a dict with:
      - ocr: {page_count, pages: [{page_number, text}]}
      - classification: [{page_number, label}]
      - questions: [{id, text, marks, source_page}]
      - qa_pairs: [{question, answer, source_pages, found, confidence, ...}]
    """
    ocr_result = run_ocr(pdf_bytes, filename=filename, status_cb=status_cb)
    classified = classify_pages(ocr_result.pages, status_cb=status_cb)
    questions = extract_questions(classified, status_cb=status_cb)
    qa_pairs = map_answers(questions, classified, status_cb=status_cb)

    return {
        "ocr": {
            "page_count": ocr_result.page_count,
            "parse_quality_score": ocr_result.parse_quality_score,
            "pages": [{"page_number": p.page_number, "text": p.text} for p in ocr_result.pages],
        },
        "classification": [{"page_number": c.page_number, "label": c.label} for c in classified],
        "questions": [asdict(q) for q in questions],
        "qa_pairs": [
            {
                "question_id": r.question_id,
                "question": r.question,
                "answer": r.answer_raw,  # raw OCR slice, untouched by the LLM
                "found": r.found,
                "confidence": r.confidence,
                "source_pages": r.source_pages,
                "start_line": r.start_line,
                "end_line": r.end_line,
                "integrity_ok": r.integrity_ok,
            }
            for r in qa_pairs
        ],
    }


def save_outputs(result: dict, out_dir: str, base_name: str) -> dict[str, str]:
    """
    Writes ocr JSON and qa_pairs JSON to disk for the Streamlit download
    buttons. Returns the two file paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    ocr_path = os.path.join(out_dir, f"{base_name}_ocr.json")
    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(result["ocr"], f, ensure_ascii=False, indent=2)

    qa_path = os.path.join(out_dir, f"{base_name}_qa_pairs.json")
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(result["qa_pairs"], f, ensure_ascii=False, indent=2)

    return {"ocr_json": ocr_path, "qa_pairs_json": qa_path}
