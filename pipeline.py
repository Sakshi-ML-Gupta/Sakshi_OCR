import os
import io
import re
import json

from pdf2image import convert_from_bytes
from rapidfuzz import fuzz

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

def run_ocr(file_content, file_name):
    response = client.chat(
        model="mistral-large-latest",
        messages=[
            {
                "role": "user",
                "content": f"""
Extract all text from this PDF exactly as written.
Return plain text only.
PDF filename:
{file_name}
"""
            }
        ]
    )
    return response

# =========================================================

# OCR JSON

# =========================================================

def ocr_to_clean_json(raw_text):
    pages = raw_text.split("\n\n")

    pages_data = []

    for idx, page_text in enumerate(pages):
        lines = []
        seen = set()

        for line in page_text.split("\n"):
            line = line.strip()

            if not line:
                continue

            if line not in seen:
                seen.add(line)
                lines.append(line)

        pages_data.append({
            "page_number": idx + 1,
            "text": lines
        })

    return {
        "total_pages": len(pages_data),
        "pages": pages_data
    }

# =========================================================

# HELPERS

# =========================================================

def extract_text(page):
    return "\n".join(page["text"])

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

def clean_answer(answer):
    answer = clean_text(answer)
    answer = re.sub(r'\s+', ' ', answer)
    return answer.strip()

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

# PARSE ANSWERS

# =========================================================

def parse_answers(pages, official_questions):
    qa_map = {}

    normalized_questions = []

    for q in official_questions:
        normalized_questions.append({
            "id": q["id"],
            "question": q["question"],
            "normalized": normalize(q["question"])
        })

    full_text = ""

    for page in pages:
        page_text = extract_text(page)
        full_text += "\n" + page_text

    lines = full_text.split("\n")

    question_positions = []

    for idx, raw_line in enumerate(lines):
        line = clean_text(raw_line)

        if is_noise(line):
            continue

        line_norm = normalize(line)

        best_match = None
        best_score = 0

        for q in normalized_questions:
            score = fuzz.partial_ratio(
                line_norm,
                q["normalized"]
            )

            if score > best_score:
                best_score = score
                best_match = q

        if best_match and best_score >= SIMILARITY_THRESHOLD:
            question_positions.append({
                "line_index": idx,
                "qid": best_match["id"],
                "question": best_match["question"]
            })

    filtered_positions = []
    seen_qids = set()

    for item in question_positions:
        if item["qid"] not in seen_qids:
            filtered_positions.append(item)
            seen_qids.add(item["qid"])

    for i in range(len(filtered_positions)):
        current = filtered_positions[i]
        start_idx = current["line_index"] + 1

        if i < len(filtered_positions) - 1:
            end_idx = filtered_positions[i + 1]["line_index"]
        else:
            end_idx = len(lines)

        answer_lines = []

        for j in range(start_idx, end_idx):
            line = clean_text(lines[j])

            if is_noise(line):
                continue

            similarity = fuzz.partial_ratio(
                normalize(line),
                normalize(current["question"])
            )

            if similarity > 90:
                continue

            answer_lines.append(line)

        final_answer = clean_answer(
            " ".join(answer_lines)
        )

        qa_map[current["qid"]] = {
            "question": current["question"],
            "answer": final_answer
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

    ocr_response = run_ocr(
        processed_pdf,
        uploaded_file.name
    )

    raw_text = ocr_response.choices[0].message.content

    ocr_json = ocr_to_clean_json(raw_text)

    questions = extract_questions(
        ocr_json["pages"]
    )

    qa_map = parse_answers(
        ocr_json["pages"],
        questions
    )

    final_json = build_final_json(qa_map)

    return ocr_json, final_json