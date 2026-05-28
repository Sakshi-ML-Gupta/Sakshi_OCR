import os
import io
import re
import json

from rapidfuzz import fuzz
from pdf2image import convert_from_bytes
from mistralai.client import MistralClient

import streamlit as st

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
# MISTRAL CLIENT
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

def run_ocr(file_content, file_name):

    print("Uploading to Mistral...")

    uploaded_file = client.files.create(
        file=(
            file_name,
            file_content
        ),
        purpose="ocr"
    )

    print("Running OCR...")

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {
                "role": "user",
                "content": f"""
Extract ALL text from this PDF EXACTLY.

Maintain:
- line breaks
- question numbering
- section headings
- formatting

Do NOT summarize.
Return plain raw text only.
"""
            }
        ],
        document=uploaded_file.id
    )

    return response.choices[0].message.content

# =========================================================
# OCR JSON
# =========================================================

def ocr_to_clean_json(raw_text):

    lines = raw_text.split("\n")

    clean_lines = []

    seen = set()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if line not in seen:

            seen.add(line)

            clean_lines.append(line)

    pages = [
        {
            "page_number": 1,
            "text": clean_lines
        }
    ]

    return {
        "total_pages": 1,
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

        for line in page["text"]:

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

    unique = []

    seen = set()

    for q in questions:

        key = normalize(q["question"])

        if key not in seen:

            seen.add(key)

            unique.append(q)

    return unique

# =========================================================
# CLEAN ANSWER
# =========================================================

def clean_answer(answer):

    answer = clean_text(answer)

    answer = re.sub(r'\s+', ' ', answer)

    return answer.strip()

# =========================================================
# PARSE ANSWERS
# =========================================================

def parse_answers(pages, official_questions):

    qa_map = {}

    all_lines = []

    for page in pages:

        all_lines.extend(page["text"])

    question_positions = []

    # FIND QUESTION LOCATIONS

    for idx, line in enumerate(all_lines):

        line_clean = clean_text(line)

        if is_noise(line_clean):
            continue

        best_match = None

        best_score = 0

        for q in official_questions:

            score = fuzz.partial_ratio(
                normalize(line_clean),
                normalize(q["question"])
            )

            if score > best_score:

                best_score = score
                best_match = q

        if best_match and best_score >= SIMILARITY_THRESHOLD:

            question_positions.append({
                "index": idx,
                "qid": best_match["id"],
                "question": best_match["question"]
            })

    # REMOVE DUPLICATES

    filtered = []

    seen = set()

    for item in question_positions:

        if item["qid"] not in seen:

            seen.add(item["qid"])

            filtered.append(item)

    # EXTRACT ANSWERS

    for i in range(len(filtered)):

        current = filtered[i]

        start = current["index"] + 1

        if i < len(filtered) - 1:

            end = filtered[i + 1]["index"]

        else:

            end = len(all_lines)

        answer_lines = []

        for j in range(start, end):

            line = clean_text(all_lines[j])

            if is_noise(line):
                continue

            answer_lines.append(line)

        qa_map[current["qid"]] = {
            "question": current["question"],
            "answer": clean_answer(
                " ".join(answer_lines)
            )
        }

    return qa_map

# =========================================================
# FINAL JSON
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

    raw_text = run_ocr(
        processed_pdf,
        uploaded_file.name
    )

    # OCR JSON

    ocr_json = ocr_to_clean_json(raw_text)

    # QUESTIONS

    questions = extract_questions(
        ocr_json["pages"]
    )

    # ANSWERS

    qa_map = parse_answers(
        ocr_json["pages"],
        questions
    )

    # FINAL JSON

    final_json = build_final_json(qa_map)

    return ocr_json, final_json