import os
import io
import re
import json
import fitz
import httpx
from pathlib import Path
from mistralai import Mistral

# =========================================================
# API KEY
# =========================================================

def get_api_key(name):
    try:
        import streamlit as st
        return st.secrets[name]
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv(name)


# =========================================================
# PREPROCESS PDF
# =========================================================

def preprocess_pdf(file_bytes, dpi=250):
    src_doc = fitz.open(stream=file_bytes, filetype="pdf")
    out_doc = fitz.open()
    for page in src_doc:
        pix = page.get_pixmap(dpi=dpi)
        new_page = out_doc.new_page(width=pix.width, height=pix.height)
        new_page.insert_image(new_page.rect, pixmap=pix)
    buf = io.BytesIO()
    out_doc.save(buf)
    src_doc.close()
    out_doc.close()
    buf.seek(0)
    return buf.read()


# =========================================================
# OCR — mistral-ocr-latest, pure transcription
# =========================================================

def run_ocr(file_content: bytes, file_name: str, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    api_key = get_api_key("MISTRAL_API_KEY")
    client = Mistral(api_key=api_key)

    log("Uploading to Mistral OCR...")
    uploaded = client.files.upload(
        file={"file_name": file_name, "content": file_content},
        purpose="ocr"
    )

    signed = client.files.get_signed_url(file_id=uploaded.id, expiry=1)
    log("Running OCR...")

    resp = httpx.post(
        "https://api.mistral.ai/v1/ocr",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": signed.url
            },
            "include_image_base64": False
        },
        timeout=180
    )

    if resp.status_code != 200:
        raise Exception(f"OCR error {resp.status_code}: {resp.text}")

    try:
        client.files.delete(file_id=uploaded.id)
    except Exception:
        pass

    # Collect raw text per page — exactly as OCR returned it
    pages = []
    for page in resp.json().get("pages", []):
        pages.append({
            "page_number": page.get("index", 0) + 1,
            "raw_text": page.get("markdown", "")
        })

    log(f"OCR done — {len(pages)} pages")
    return pages


# =========================================================
# BUILD OCR JSON
# =========================================================

def build_ocr_json(pages: list) -> dict:
    return {
        "total_pages": len(pages),
        "pages": [
            {
                "page_number": p["page_number"],
                "text": p["raw_text"]
            }
            for p in pages
        ]
    }


# =========================================================
# FIND QUESTION BOUNDARIES — pure regex, no LLM
# Detects lines that look like question labels:
#   "1.", "2.", "Q1", "Q.1", "(i)", "i)", "ii.", etc.
# Returns list of (label, start_line_index)
# =========================================================

QUESTION_LABEL_PATTERNS = [
    r'^\*{0,2}Q\.?\s*\d+[\.\)\:]',          # Q1. Q1) Q.1
    r'^\*{0,2}\d+[\.\)]\s',                  # 1. 2. 3)
    r'^\*{0,2}\(\s*[ivxlcdmIVXLCDM]+\s*\)',  # (i) (ii) (iv)
    r'^\*{0,2}[ivxlcdmIVXLCDM]+[\.\)]\s',    # i. ii. iii)
    r'^\*{0,2}[A-Da-d][\.\)]\s',             # a. b. c. A. B.
    r'^\*{0,2}(?:Section|SECTION|Part|PART)\s+[A-Z\d]',  # Section A
]

def looks_like_question_label(line: str) -> bool:
    line = line.strip()
    if len(line) < 2:
        return False
    for pat in QUESTION_LABEL_PATTERNS:
        if re.match(pat, line):
            return True
    return False


def find_question_boundaries(all_lines: list) -> list:
    """
    Scan all lines and return positions where questions start.
    Returns list of dicts: {label, line_index}
    """
    boundaries = []
    for i, line in enumerate(all_lines):
        if looks_like_question_label(line):
            boundaries.append({
                "label": line.strip(),
                "line_index": i
            })
    return boundaries


# =========================================================
# SLICE RAW TEXT INTO Q-A PAIRS — no LLM, no modification
# =========================================================

def slice_qa_pairs(all_lines: list, boundaries: list) -> list:
    """
    For each boundary:
      - question = the label line itself (raw)
      - answer   = all lines from (label+1) to (next label-1)
    Everything is raw OCR text, untouched.
    """
    qa_pairs = []

    for i, b in enumerate(boundaries):
        q_start = b["line_index"]

        # answer starts right after question label
        a_start = q_start + 1

        # answer ends where next question starts
        if i + 1 < len(boundaries):
            a_end = boundaries[i + 1]["line_index"]
        else:
            a_end = len(all_lines)

        question_text = all_lines[q_start]

        answer_lines = [
            all_lines[j]
            for j in range(a_start, a_end)
        ]

        answer_text = "\n".join(answer_lines).strip()

        qa_pairs.append({
            "question": question_text,
            "answer": answer_text
        })

    return qa_pairs


# =========================================================
# COMPLETE PIPELINE
# =========================================================

def process_pdf(file_input, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    # ── Read bytes ─────────────────────────────────────────
    if isinstance(file_input, (str, Path)):
        file_bytes = Path(file_input).read_bytes()
        file_name  = Path(file_input).name
    else:
        file_bytes = file_input.read()
        file_name  = getattr(file_input, "name", "document.pdf")

    # ── Preprocess ────────────────────────────────────────
    log("Preprocessing PDF...")
    processed = preprocess_pdf(file_bytes)

    # ── OCR ───────────────────────────────────────────────
    pages = run_ocr(processed, file_name, status_callback)

    # ── Build OCR JSON ────────────────────────────────────
    log("Building OCR JSON...")
    ocr_json = build_ocr_json(pages)
    log(f"Total pages: {ocr_json['total_pages']}")

    # ── Flatten all lines across all pages ────────────────
    log("Flattening lines...")
    all_lines = []
    for page in pages:
        for line in page["raw_text"].split("\n"):
            all_lines.append(line)  # keep empty lines for spacing

    # ── Find question boundaries by regex ─────────────────
    log("Finding question boundaries...")
    boundaries = find_question_boundaries(all_lines)
    log(f"Found {len(boundaries)} question boundaries")

    if not boundaries:
        raise Exception(
            "No question boundaries found.\n"
            f"First 500 chars of OCR:\n{chr(10).join(all_lines[:30])}"
        )

    # ── Slice raw text ────────────────────────────────────
    log("Slicing raw Q-A pairs...")
    qa_pairs = slice_qa_pairs(all_lines, boundaries)

    log(f"Done — {len(qa_pairs)} Q-A pairs extracted (100% raw text)")
    return ocr_json, qa_pairs