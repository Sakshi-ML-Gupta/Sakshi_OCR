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
# OCR — mistral-ocr-latest
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
# DETECT SPLIT POINT
# The question paper pages come first.
# Answer pages start when we see "TOPIC" / "DATE" /
# answer sheet headers — typical handwritten answer sheet markers.
# =========================================================

def find_answer_start_page(pages: list) -> int:
    """
    Returns the page index (0-based) where answer sheets begin.
    Heuristic: answer pages contain 'TOPIC' and 'DATE' fields
    which are printed on answer sheet templates.
    """
    for i, page in enumerate(pages):
        text = page["raw_text"]
        # Answer sheet marker: has TOPIC/DATE fields AND question-like content
        has_answer_marker = (
            re.search(r'\bTOPIC\b', text) and
            re.search(r'\bDATE\b', text)
        )
        if has_answer_marker:
            return i
    # fallback: second half
    return len(pages) // 2


# =========================================================
# EXTRACT QUESTIONS FROM QUESTION PAGES
# Reads only question paper pages, returns list of raw question strings
# =========================================================

def extract_questions_from_pages(pages: list, end_page_idx: int) -> list:
    """
    Collects all question text from pages 0..end_page_idx-1.
    A question is any line that starts with a question label pattern
    AND is long enough to be a real question (not a heading).
    """
    LABEL_PATTERNS = [
        r'^\(?[ivxIVX]+[\.\)]\s+\S',       # (i) (ii) i. ii.
        r'^\d+[\.\)]\s+\S',                  # 1. 2. 3)
        r'^Q\.?\s*\d+[\.\):\s]',             # Q1. Q1) Q.1
    ]

    questions = []
    for page in pages[:end_page_idx]:
        lines = page["raw_text"].split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) < 20:
                continue  # too short to be a real question
            for pat in LABEL_PATTERNS:
                if re.match(pat, stripped):
                    questions.append(stripped)
                    break

    # deduplicate while preserving order
    seen = set()
    unique = []
    for q in questions:
        key = re.sub(r'\s+', ' ', q.lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(q)

    return unique


# =========================================================
# EXTRACT ANSWERS FROM ANSWER PAGES
# Each answer page has the question repeated at the top,
# followed by the student's answer.
# We split on those repeated question headers.
# =========================================================

def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for fuzzy compare."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def similarity(a: str, b: str) -> float:
    """Simple word-overlap similarity between two strings."""
    wa = set(normalize(a).split())
    wb = set(normalize(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def is_noise_line(line: str) -> bool:
    """Lines that are printed on answer sheet template — not content."""
    patterns = [
        r'^\s*TOPIC\s*$',
        r'^\s*DATE\s*$',
        r'TOPIC\s*_+',
        r'DATE\s*_+',
        r'^\s*\d+\s*$',           # lone page numbers
        r'^\s*[-_]+\s*$',         # divider lines
    ]
    for p in patterns:
        if re.search(p, line, re.IGNORECASE):
            return True
    return False


def find_question_boundaries_in_answers(
    answer_lines: list,
    questions: list,
    similarity_threshold: float = 0.45
) -> list:
    """
    Scan answer page lines.
    When a line closely matches a known question, mark it as a boundary.
    Returns list of {question_text, line_index}
    """
    boundaries = []
    used_questions = set()

    for i, line in enumerate(answer_lines):
        stripped = line.strip()
        if len(stripped) < 15:
            continue

        best_score = 0
        best_q = None

        for q in questions:
            if q in used_questions:
                continue
            score = similarity(stripped, q)
            if score > best_score:
                best_score = score
                best_q = q

        if best_q and best_score >= similarity_threshold:
            boundaries.append({
                "question": best_q,
                "line_index": i
            })
            used_questions.add(best_q)

    return boundaries


# =========================================================
# SLICE RAW ANSWERS — zero LLM, pure text slicing
# =========================================================

def slice_raw_answers(answer_lines: list, boundaries: list) -> list:
    """
    For each boundary:
      question = the matched question text (from question paper)
      answer   = raw lines from (boundary line + 1) to (next boundary - 1)
                 with noise lines removed
    """
    qa_pairs = []

    for i, b in enumerate(boundaries):
        a_start = b["line_index"] + 1

        if i + 1 < len(boundaries):
            a_end = boundaries[i + 1]["line_index"]
        else:
            a_end = len(answer_lines)

        raw_lines = []
        for j in range(a_start, a_end):
            line = answer_lines[j]
            if not is_noise_line(line):
                raw_lines.append(line)

        answer_text = "\n".join(raw_lines).strip()

        qa_pairs.append({
            "question": b["question"],   # clean question from question paper
            "answer": answer_text         # raw OCR text from answer pages
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

    # ── Find where answer pages start ─────────────────────
    answer_start = find_answer_start_page(pages)
    log(f"Question pages: 1 to {answer_start} | Answer pages: {answer_start + 1} to {len(pages)}")

    # ── Extract questions from question pages ──────────────
    log("Extracting questions from question paper...")
    questions = extract_questions_from_pages(pages, answer_start)
    log(f"Found {len(questions)} questions: {[q[:60] for q in questions]}")

    if not questions:
        raise Exception(
            "No questions found in question pages.\n"
            f"First page preview:\n{pages[0]['raw_text'][:400]}"
        )

    # ── Flatten answer page lines ──────────────────────────
    log("Flattening answer pages...")
    answer_lines = []
    for page in pages[answer_start:]:
        for line in page["raw_text"].split("\n"):
            answer_lines.append(line)

    # ── Find question boundaries in answer pages ───────────
    log("Matching questions in answer pages...")
    boundaries = find_question_boundaries_in_answers(answer_lines, questions)
    log(f"Matched {len(boundaries)} question boundaries")

    if not boundaries:
        raise Exception(
            "Could not match any questions in answer pages.\n"
            f"Questions found: {questions}\n"
            f"First 20 answer lines:\n" + "\n".join(answer_lines[:20])
        )

    # ── Slice raw answers ──────────────────────────────────
    log("Slicing raw answers (no LLM)...")
    qa_pairs = slice_raw_answers(answer_lines, boundaries)

    log(f"Done — {len(qa_pairs)} Q-A pairs")
    return ocr_json, qa_pairs