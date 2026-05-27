import os
import io
import re
import json
import fitz

from pathlib import Path
from rapidfuzz import fuzz
from pdf2image import convert_from_bytes
import streamlit as st

from mistralai.client import MistralClient

# =========================================================
# CONFIG
# =========================================================

SIMILARITY_THRESHOLD = 72

NOISE_PATTERNS = [
    r"^TOPIC$",
    r"^DATE$",
    r"^TOPIC\s*_*$",
    r"^DATE\s*_*$",
    r"^BEGIN-\d+$",
    r"^\d+$",
]

REMOVE_LINES_CONTAINING = [
    "TOPIC",
    "DATE",
]

# =========================================================
# MISTRAL
# =========================================================

api_key = st.secrets["MISTRAL_API_KEY"]

client = MistralClient(api_key=api_key)

# =========================================================
# PREPROCESS PDF
# =========================================================

def preprocess_pdf(file_bytes):

    try:

        images = convert_from_bytes(file_bytes, dpi=300)

        pdf_bytes = io.BytesIO()

        images[0].save(
            pdf_bytes,
            format="PDF",
            save_all=True,
            append_images=images[1:]
        )

        pdf_bytes.seek(0)

        return pdf_bytes.read()

    except Exception as e:

        print("Preprocessing failed:", e)

        return file_bytes

# =========================================================
# OCR
# =========================================================

def run_ocr(file_bytes, filename):

    uploaded_file = client.files.upload(
        file={
            "file_name": filename,
            "content": file_bytes,
        },
        purpose="ocr"
    )

    signed_url = client.files.get_signed_url(
        file_id=uploaded_file.id
    )

    response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        }
    )

    return response

# =========================================================
# OCR JSON
# =========================================================

def build_ocr_json(ocr_response):

    pages = []

    for page in ocr_response.pages:

        text = page.markdown

        lines = []

        seen = set()

        for line in text.split("\n"):

            line = line.strip()

            if not line:
                continue

            if line not in seen:
                seen.add(line)
                lines.append(line)

        pages.append({
            "page_number": page.index + 1,
            "text": lines
        })

    return {
        "total_pages": len(pages),
        "pages": pages
    }

# =========================================================
# HELPERS
# =========================================================

def normalize(text):

    text = str(text).lower()

    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def clean_text(text):

    text = str(text)

    cleaned = []

    for line in text.split("\n"):

        line = line.strip()

        skip = False

        for item in REMOVE_LINES_CONTAINING:

            if item.lower() in line.lower():
                skip = True
                break

        if not skip:
            cleaned.append(line)

    text = "\n".join(cleaned)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def is_noise(line):

    if not line:
        return True

    for pattern in NOISE_PATTERNS:

        if re.match(pattern, line, re.IGNORECASE):
            return True

    return False

# =========================================================
# EXTRACT QUESTIONS
# =========================================================

def extract_questions(pages):

    questions = []

    for page in pages:

        lines = page["text"]

        for line in lines:

            line = clean_text(line)

            if is_noise(line):
                continue

            # SECTION A
            roman_match = re.match(
                r'^(?:\d+\s*\.?\s*)?\(?([ivxlcdm]+)\)?[\.\)]?\s*(.+)',
                line,
                re.IGNORECASE
            )

            if roman_match:

                roman = roman_match.group(1)
                qtext = roman_match.group(2)

                if len(qtext) > 15:

                    questions.append({
                        "id": f"A({roman.lower()})",
                        "question": qtext
                    })

                    continue

            # SECTION B
            normal_match = re.match(
                r'^(\d+)[\.\)]\s*(.+)',
                line
            )

            if normal_match:

                num = normal_match.group(1)
                qtext = normal_match.group(2)

                if len(qtext) > 15:

                    questions.append({
                        "id": f"B{num}",
                        "question": qtext
                    })

    # remove duplicates

    unique = []

    seen = set()

    for q in questions:

        key = normalize(q["question"])

        if key not in seen:

            seen.add(key)

            unique.append(q)

    return unique

# =========================================================
# MATCH QUESTION
# =========================================================

def match_question(line, questions):

    best_match = None

    best_score = 0

    line_norm = normalize(line)

    for q in questions:

        q_norm = normalize(q["question"])

        score = fuzz.partial_ratio(
            line_norm,
            q_norm
        )

        if score > best_score:

            best_score = score

            best_match = q

    if best_score >= SIMILARITY_THRESHOLD:
        return best_match

    return None

# =========================================================
# PARSE ANSWERS
# =========================================================

def parse_answers(pages, questions):

    qa_map = {}

    current_qid = None
    current_question = None
    current_answer = []

    for page in pages:

        lines = page["text"]

        for raw_line in lines:

            line = clean_text(raw_line)

            if is_noise(line):
                continue

            matched = match_question(line, questions)

            if matched:

                if current_qid:

                    qa_map[current_qid] = {
                        "question": current_question,
                        "answer": " ".join(current_answer).strip()
                    }

                current_qid = matched["id"]

                current_question = matched["question"]

                current_answer = []

                continue

            if current_qid:

                similarity = fuzz.partial_ratio(
                    normalize(line),
                    normalize(current_question)
                )

                if similarity > 90:
                    continue

                current_answer.append(line)

    if current_qid:

        qa_map[current_qid] = {
            "question": current_question,
            "answer": " ".join(current_answer).strip()
        }

    return qa_map

# =========================================================
# BUILD FINAL JSON
# =========================================================

def build_final_json(qa_map):

    qa_pairs = []

    for qid, qa in qa_map.items():

        qa_pairs.append({
            "question_id": qid,
            "question": qa["question"],
            "answer": qa["answer"]
        })

    return {
        "total_qa_pairs": len(qa_pairs),
        "qa_pairs": qa_pairs
    }

# =========================================================
# COMPLETE PIPELINE
# =========================================================

def process_pdf(uploaded_file):

    file_bytes = uploaded_file.read()

    processed_pdf = preprocess_pdf(file_bytes)

    ocr_response = run_ocr(
        processed_pdf,
        uploaded_file.name
    )

    # OCR JSON

    ocr_json = build_ocr_json(ocr_response)

    # QUESTION EXTRACTION

    questions = extract_questions(
        ocr_json["pages"]
    )

    # ANSWER PARSING

    qa_map = parse_answers(
        ocr_json["pages"],
        questions
    )

    # FINAL QA JSON

    final_json = build_final_json(qa_map)

    return ocr_json, final_json