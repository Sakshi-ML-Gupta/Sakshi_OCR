import os
import io
import re
import json


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

def run_ocr(file_content: bytes, file_name: str):

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
                "content": [
                    {
                        "type": "document_url",
                        "document_url": uploaded_file.url
                    },
                    {
                        "type": "text",
                        "text": "Extract all text from this PDF exactly as it appears."
                    }
                ]
            }
        ]
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

def parse_answers(pages, official_questions):

    qa_map = {}

    # =====================================================
    # CREATE QUESTION PATTERNS
    # =====================================================

    normalized_questions = []

    for q in official_questions:

        normalized_questions.append({
            "id": q["id"],
            "question": q["question"],
            "normalized": normalize(q["question"])
        })

    # =====================================================
    # MERGE ALL PAGES
    # =====================================================

    full_text = ""

    for page in pages:

        page_text = extract_text(page)

        full_text += "\n" + page_text

    lines = full_text.split("\n")

    # =====================================================
    # FIND QUESTION POSITIONS
    # =====================================================

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

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    filtered_positions = []

    seen_qids = set()

    for item in question_positions:

        if item["qid"] not in seen_qids:

            filtered_positions.append(item)
            seen_qids.add(item["qid"])

    # =====================================================
    # EXTRACT ANSWERS BETWEEN QUESTIONS
    # =====================================================

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
# BUILD FINAL JSON
# =========================================================

def ocr_to_clean_json(ocr_response):

    text = ocr_response.choices[0].message.content

    pages = text.split("\n\n")

    pages_data = []

    for idx, page_text in enumerate(pages):

        lines = []

        seen = set()

        for line in page_text.split("\n"):

            line = line.strip()

            if line and line not in seen:

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