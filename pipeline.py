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
# OCR
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
            {"page_number": p["page_number"], "text": p["raw_text"]}
            for p in pages
        ]
    }


# =========================================================
# DETECT WHERE ANSWER PAGES START
# =========================================================

def find_answer_start_page(pages: list) -> int:
    for i, page in enumerate(pages):
        text = page["raw_text"]
        if re.search(r'\bTOPIC\b', text) and re.search(r'\bDATE\b', text):
            return i
    return len(pages) // 2


# =========================================================
# EXTRACT QUESTIONS FROM QUESTION PAGES
# =========================================================

def extract_questions_from_pages(pages: list, end_idx: int) -> list:
    LABEL_PATTERNS = [
        r'^\(?[ivxIVX]+[\.\)]\s+\S',
        r'^\d+[\.\)]\s+\S',
        r'^Q\.?\s*\d+[\.\):\s]',
    ]
    questions = []
    for page in pages[:end_idx]:
        for line in page["raw_text"].split("\n"):
            stripped = line.strip()
            if len(stripped) < 20:
                continue
            for pat in LABEL_PATTERNS:
                if re.match(pat, stripped):
                    questions.append(stripped)
                    break

    seen = set()
    unique = []
    for q in questions:
        key = re.sub(r'\s+', ' ', q.lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


# =========================================================
# TEXT HELPERS
# =========================================================

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def similarity(a: str, b: str) -> float:
    wa = set(normalize(a).split())
    wb = set(normalize(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def strip_leading_label(text: str) -> str:
    text = re.sub(r'^(?:Ans(?:wer)?[.\s]+)', '', text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'^(?:Q\.?\s*)?\d+[.)]\s*', '', text.strip(), flags=re.IGNORECASE)
    return text.strip()


def is_noise_line(line: str) -> bool:
    patterns = [
        r'^\s*TOPIC\s*$',
        r'^\s*DATE\s*$',
        r'TOPIC\s*_+',
        r'DATE\s*_+',
        r'^\s*\d+\s*$',
        r'^\s*[-_]+\s*$',
    ]
    for p in patterns:
        if re.search(p, line, re.IGNORECASE):
            return True
    return False


# =========================================================
# FIND QUESTION BOUNDARIES IN ANSWER PAGES
#
# Two problems solved here:
#
# Problem 1 — OCR misreads roman numerals:
#   "(i)" gets OCR'd as "(Tii)", "(1i)", etc.
#   Fix: strip all non-alpha from roman numeral tokens before comparing,
#   and also try matching by the question BODY (words after the label)
#
# Problem 2 — question split across lines:
#   "1. (iii) This music hall is empty; where could"
#   "the actors have gone?..."
#   Fix: sliding window joins multiple lines before scoring
# =========================================================

# Lines that can start a question boundary:
# optional "1." prefix, then roman/number/quote
LABEL_RE = re.compile(
    r"^\s*(?:(?:Q\.?\s*)?\d+[.)]\s*)?"
    r"(?:\(?\w+[.)]\s|\"|\d+[.)]\s)"
)


def find_question_boundaries_in_answers(
    answer_lines: list,
    questions: list,
    similarity_threshold: float = 0.35,
    window: int = 5
) -> list:
    boundaries = []
    used_questions = set()
    used_line_indices = set()

    for i in range(len(answer_lines)):
        line_i = answer_lines[i].strip()

        if not LABEL_RE.match(line_i):
            continue

        for w in range(1, window + 1):
            if i + w > len(answer_lines):
                break

            combined = " ".join(
                answer_lines[i + k].strip()
                for k in range(w)
                if answer_lines[i + k].strip()
            )
            if len(combined) < 10:
                continue

            combined_clean = strip_leading_label(combined)

            best_score = 0
            best_q = None

            for q in questions:
                if q in used_questions:
                    continue

                # Score 1: full combined vs full question
                s1 = similarity(combined, q)
                # Score 2: strip label from both sides
                s2 = similarity(combined_clean, strip_leading_label(q))
                # Score 3: match just the body words (skip first token)
                # handles OCR misread of roman numerals like (i) -> (Tii)
                combined_words = normalize(combined_clean).split()
                q_words = normalize(strip_leading_label(q)).split()
                body_combined = " ".join(combined_words[1:]) if len(combined_words) > 1 else ""
                body_q = " ".join(q_words[1:]) if len(q_words) > 1 else ""
                s3 = similarity(body_combined, body_q) if body_combined and body_q else 0

                score = max(s1, s2, s3)

                if score > best_score:
                    best_score = score
                    best_q = q

            if best_q and best_score >= similarity_threshold:
                if i not in used_line_indices:
                    boundaries.append({
                        "question": best_q,
                        "line_index": i
                    })
                    used_questions.add(best_q)
                    used_line_indices.add(i)
                break

    boundaries.sort(key=lambda b: b["line_index"])
    return boundaries


# =========================================================
# FIND WHERE ACTUAL ANSWER STARTS
# After the boundary line, the student may repeat the full
# question across multiple lines before writing the answer.
# Skip all those question-continuation lines.
# The answer starts at the first line that is NOT part of the question.
# =========================================================

def find_answer_start_offset(answer_lines: list, boundary_idx: int, question: str) -> int:
    """
    Starting from boundary_idx+1, skip lines that are still
    part of the question text (high similarity to question words).
    Returns the offset from boundary_idx where the real answer begins.
    Minimum offset = 1 (always skip at least the boundary line itself).
    """
    q_words = set(normalize(question).split())
    # remove very common words that appear in both question and answer
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in',
        'to', 'and', 'that', 'this', 'it', 'he', 'she', 'they',
        'i', 'we', 'you', 'at', 'or', 'but', 'not', 'with', 'for',
        'on', 'from', 'by', 'as', 'be', 'his', 'her', 'its', 'their'
    }
    q_words = q_words - stopwords

    offset = 1
    max_skip = min(10, len(answer_lines) - boundary_idx - 1)

    for k in range(1, max_skip + 1):
        idx = boundary_idx + k
        if idx >= len(answer_lines):
            break

        line = answer_lines[idx].strip()
        if not line or is_noise_line(line):
            offset = k + 1
            continue

        line_words = set(normalize(line).split()) - stopwords
        if not line_words:
            offset = k + 1
            continue

        # how many words in this line overlap with the question
        overlap = len(line_words & q_words) / max(len(line_words), 1)

        if overlap >= 0.5:
            # still looks like question text — skip it
            offset = k + 1
        else:
            # this line looks like answer content — stop skipping
            break

    return offset


# =========================================================
# SLICE RAW ANSWERS
# =========================================================

def slice_raw_answers(answer_lines: list, boundaries: list) -> list:
    qa_pairs = []

    for i, b in enumerate(boundaries):
        # find where the actual answer content starts
        # (skip question-repetition lines at the top)
        skip = find_answer_start_offset(answer_lines, b["line_index"], b["question"])
        a_start = b["line_index"] + skip

        if i + 1 < len(boundaries):
            a_end = boundaries[i + 1]["line_index"]
        else:
            a_end = len(answer_lines)

        raw_lines = [
            answer_lines[j]
            for j in range(a_start, a_end)
            if not is_noise_line(answer_lines[j])
        ]

        qa_pairs.append({
            "question": b["question"],
            "answer": "\n".join(raw_lines).strip()
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

    if isinstance(file_input, (str, Path)):
        file_bytes = Path(file_input).read_bytes()
        file_name  = Path(file_input).name
    else:
        file_bytes = file_input.read()
        file_name  = getattr(file_input, "name", "document.pdf")

    log("Preprocessing PDF...")
    processed = preprocess_pdf(file_bytes)

    pages = run_ocr(processed, file_name, status_callback)

    log("Building OCR JSON...")
    ocr_json = build_ocr_json(pages)
    log(f"Total pages: {ocr_json['total_pages']}")

    answer_start = find_answer_start_page(pages)
    log(f"Question pages: 1-{answer_start} | Answer pages: {answer_start+1}-{len(pages)}")

    log("Extracting questions from question pages...")
    questions = extract_questions_from_pages(pages, answer_start)
    log(f"Found {len(questions)} questions: {[q[:50] for q in questions]}")

    if not questions:
        raise Exception(
            "No questions found.\n"
            f"First page preview:\n{pages[0]['raw_text'][:400]}"
        )

    log("Flattening answer pages...")
    answer_lines = []
    for page in pages[answer_start:]:
        for line in page["raw_text"].split("\n"):
            answer_lines.append(line)

    log("Matching question boundaries in answer pages...")
    boundaries = find_question_boundaries_in_answers(answer_lines, questions)
    log(f"Matched {len(boundaries)} of {len(questions)} boundaries")

    if not boundaries:
        raise Exception(
            "Could not match any questions in answer pages.\n"
            f"Questions: {questions}\n"
            f"First 20 answer lines:\n" + "\n".join(answer_lines[:20])
        )

    log("Slicing raw answers (skipping question repetitions)...")
    qa_pairs = slice_raw_answers(answer_lines, boundaries)

    log(f"Done — {len(qa_pairs)} Q-A pairs")
    return ocr_json, qa_pairs