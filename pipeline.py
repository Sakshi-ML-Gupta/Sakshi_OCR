import os
import io
import re
import json
from pathlib import Path

import fitz
import streamlit as st

from rapidfuzz import fuzz
from mistralai.client import MistralClient

# =========================================================
# MISTRAL
# =========================================================

api_key = st.secrets["MISTRAL_API_KEY"]

client = MistralClient(api_key=api_key)

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
# PREPROCESS PDF
# =========================================================

def preprocess_pdf(file_bytes, dpi=250):

    try:

        src_doc = fitz.open(stream=file_bytes, filetype="pdf")

        out_doc = fitz.open()

        for page in src_doc:

            pix = page.get_pixmap(dpi=dpi)

            new_page = out_doc.new_page(
                width=pix.width,
                height=pix.height
            )

            new_page.insert_image(
                new_page.rect,
                pixmap=pix
            )

        pdf_bytes = io.BytesIO()

        out_doc.save(pdf_bytes)

        src_doc.close()
        out_doc.close()

        pdf_bytes.seek(0)

        return pdf_bytes.read()

    except Exception as e:

        print("Preprocessing failed:", e)

        return file_bytes

# =========================================================
# OCR
# =========================================================

def run_ocr(file_content, file_name):

    uploaded_file = client.files.create(
        file=(file_name, file_content),
        purpose="fine-tune"
    )

    signed_url = client.files.get_signed_url(
        uploaded_file.id
    )

    response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url
        }
    )

    return response

# =========================================================
# OCR JSON
# =========================================================

def build_ocr_json(ocr_response):

    pages_data = []

    for page in ocr_response.pages:

        lines = page.markdown.split("\n")

        clean_lines = []

        seen = set()

        for line in lines:

            line = line.strip()

            if not line:
                continue

            if line not in seen:

                seen.add(line)

                clean_lines.append(line)

        pages_data.append({
            "page_number": page.index + 1,
            "text": clean_lines
        })

    return {
        "total_pages": len(pages_data),
        "pages": pages_data
    }

# =========================================================
# HELPERS
# =========================================================

def extract_text(data):

    if isinstance(data, list):

        return "\n".join(data)

    return str(data)

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

        for line in page["text"]:

            line = clean_text(line)

            if is_noise(line):

                continue

            match = re.match(
                r'^(\d+|[ivxlcdm]+)[\.\)]\s*(.+)',
                line,
                re.IGNORECASE
            )

            if match:

                qid = match.group(1)

                qtext = match.group(2)

                if len(qtext) > 15:

                    questions.append({
                        "id": qid,
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

    full_text = ""

    for page in pages:

        full_text += "\n".join(page["text"]) + "\n"

    lines = full_text.split("\n")

    positions = []

    for idx, raw_line in enumerate(lines):

        line = clean_text(raw_line)

        if is_noise(line):

            continue

        best_match = None

        best_score = 0

        for q in official_questions:

            score = fuzz.partial_ratio(
                normalize(line),
                normalize(q["question"])
            )

            if score > best_score:

                best_score = score

                best_match = q

        if best_match and best_score >= SIMILARITY_THRESHOLD:

            positions.append({
                "index": idx,
                "qid": best_match["id"],
                "question": best_match["question"]
            })

    filtered = []

    seen = set()

    for item in positions:

        if item["qid"] not in seen:

            filtered.append(item)

            seen.add(item["qid"])

    for i in range(len(filtered)):

        current = filtered[i]

        start = current["index"] + 1

        if i < len(filtered) - 1:

            end = filtered[i + 1]["index"]

        else:

            end = len(lines)

        answer_lines = []

        for j in range(start, end):

            line = clean_text(lines[j])

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
# PROCESS PDF
# =========================================================

def process_pdf(uploaded_file):

    file_bytes = uploaded_file.read()

    processed_pdf = preprocess_pdf(file_bytes)

    ocr_response = run_ocr(
        processed_pdf,
        uploaded_file.name
    )

    ocr_json = build_ocr_json(ocr_response)

    questions = extract_questions(
        ocr_json["pages"]
    )

    qa_map = parse_answers(
        ocr_json["pages"],
        questions
    )

    final_json = build_final_json(qa_map)

    return ocr_json, final_json