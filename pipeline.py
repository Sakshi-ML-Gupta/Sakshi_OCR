import os
import re
import json
from pathlib import Path

import fitz
import streamlit as st

from mistralai.client import MistralClient
from rapidfuzz import fuzz

# =========================================================
# API KEY
# =========================================================

api_key = st.secrets["MISTRAL_API_KEY"]

client = MistralClient(api_key=api_key)

# =========================================================
# CONFIG
# =========================================================

SIMILARITY_THRESHOLD = 75

NOISE_PATTERNS = [
    r"^topic$",
    r"^date$",
    r"^\d+$",
    r"^page\s+\d+$",
]

REMOVE_LINES_CONTAINING = [
    "topic",
    "date",
]

# =========================================================
# NORMALIZE
# =========================================================

def normalize(text):

    text = str(text).lower()

    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = str(text)

    cleaned = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        skip = False

        for item in REMOVE_LINES_CONTAINING:

            if item.lower() in line.lower():
                skip = True
                break

        if not skip:
            cleaned.append(line)

    return "\n".join(cleaned)

# =========================================================
# NOISE FILTER
# =========================================================

def is_noise(line):

    line = clean_text(line)

    if not line:
        return True

    for pattern in NOISE_PATTERNS:

        if re.match(pattern, line, re.IGNORECASE):
            return True

    return False

# =========================================================
# OCR USING PYMUPDF
# =========================================================

def extract_pdf_text(pdf_path):

    doc = fitz.open(pdf_path)

    pages = []

    for page_num in range(len(doc)):

        page = doc.load_page(page_num)

        text = page.get_text()

        lines = []

        for line in text.split("\n"):

            line = line.strip()

            if line:
                lines.append(line)

        pages.append({
            "page_number": page_num + 1,
            "text": lines
        })

    return {
        "total_pages": len(pages),
        "pages": pages
    }

# =========================================================
# QUESTION EXTRACTION
# =========================================================

def extract_questions(pages):

    questions = []

    seen = set()

    patterns = [

        r'^(\d+)\.\s+(.+)',
        r'^(\d+)\)\s+(.+)',
        r'^q\.?\s*(\d+)\s+(.+)',
    ]

    for page in pages:

        for line in page["text"]:

            line = clean_text(line)

            if is_noise(line):
                continue

            for pattern in patterns:

                match = re.match(
                    pattern,
                    line,
                    re.IGNORECASE
                )

                if match:

                    qid = f"Q{match.group(1)}"

                    qtext = match.group(2).strip()

                    key = normalize(qtext)

                    if key not in seen and len(qtext) > 10:

                        seen.add(key)

                        questions.append({
                            "id": qid,
                            "question": qtext
                        })

    return questions

# =========================================================
# MATCH QUESTION
# =========================================================

def match_question(line, official_questions):

    line_norm = normalize(line)

    best_match = None
    best_score = 0

    for q in official_questions:

        q_norm = normalize(q["question"])

        # STRICT MATCHING
        score = fuzz.ratio(line_norm, q_norm)

        # BONUS if beginning matches
        if line_norm[:40] in q_norm:
            score += 10


# =========================================================
# PARSE ANSWERS
# =========================================================

def parse_answers(pages, questions):

    qa_map = {}

    current_qid = None

    current_question = None

    current_answer_lines = []

    for page in pages:

        for raw_line in page["text"]:

            line = clean_text(raw_line)

            if is_noise(line):
                continue

            matched = match_question(
                line,
                questions
            )

            if matched:

                if current_qid:

                    qa_map[current_qid] = {
                        "question": current_question,
                        "answer": " ".join(current_answer_lines)
                    }

                current_qid = matched["id"]

                current_question = matched["question"]

                current_answer_lines = []

                continue

            if current_qid:

                similarity = fuzz.partial_ratio(
                    normalize(line),
                    normalize(current_question)
                )

                if similarity > 90:
                    continue

                current_answer_lines.append(line)

    if current_qid:

        qa_map[current_qid] = {
            "question": current_question,
            "answer": " ".join(current_answer_lines)
        }

    return qa_map

# =========================================================
# BUILD JSON
# =========================================================

def build_json(qa_map):

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
# PROCESS PDF
# =========================================================

def process_pdf(pdf_path):

    os.makedirs("outputs", exist_ok=True)

    ocr_json = extract_pdf_text(pdf_path)

    pages = ocr_json["pages"]

    questions = extract_questions(pages)

    qa_map = parse_answers(
        pages,
        questions
    )

    final_json = build_json(qa_map)

    output_path = "outputs/final_output.json"

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(
            final_json,
            f,
            indent=4,
            ensure_ascii=False
        )

    return output_path