import os
import re
import json
import requests
import streamlit as st

from pathlib import Path
from rapidfuzz import fuzz

# =========================================================
# API KEY
# =========================================================

API_KEY = st.secrets["MISTRAL_API_KEY"]

# =========================================================
# NORMALIZE
# =========================================================

def normalize(text):

    text = str(text).lower()

    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# =========================================================
# OCR USING RAW HTTP API
# =========================================================

def run_ocr(pdf_path):

    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    # =========================================
    # STEP 1 UPLOAD FILE
    # =========================================

    with open(pdf_path, "rb") as f:

        files = {
            "file": (
                Path(pdf_path).name,
                f,
                "application/pdf"
            )
        }

        data = {
            "purpose": "ocr"
        }

        upload_response = requests.post(
            "https://api.mistral.ai/v1/files",
            headers=headers,
            files=files,
            data=data
        )

    upload_json = upload_response.json()

    if "id" not in upload_json:
        raise Exception(upload_json)

    file_id = upload_json["id"]

    # =========================================
    # STEP 2 GET SIGNED URL
    # =========================================

    signed_response = requests.get(
        f"https://api.mistral.ai/v1/files/{file_id}/url",
        headers=headers
    )

    signed_json = signed_response.json()

    if "url" not in signed_json:
        raise Exception(signed_json)

    signed_url = signed_json["url"]

    # =========================================
    # STEP 3 OCR
    # =========================================

    payload = {
        "model": "mistral-ocr-latest",
        "document": {
            "type": "document_url",
            "document_url": signed_url
        }
    }

    ocr_response = requests.post(
        "https://api.mistral.ai/v1/ocr",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload
    )

    return ocr_response.json()

# =========================================================
# EXTRACT LINES
# =========================================================

def extract_lines(ocr_json):

    all_lines = []

    pages = ocr_json.get("pages", [])

    for page in pages:

        markdown = page.get("markdown", "")

        lines = markdown.split("\n")

        for line in lines:

            line = line.strip()

            if line:
                all_lines.append(line)

    return all_lines

# =========================================================
# QUESTION DETECTOR
# =========================================================

def is_question(line):

    patterns = [

        r'^\d+\.',
        r'^\d+\)',
        r'^q\s*\d+',
        r'^question\s*\d+',
        r'^\(?[ivxlcdm]+\)',
    ]

    for p in patterns:

        if re.match(p, line, re.IGNORECASE):
            return True

    return False

# =========================================================
# QA PARSER
# =========================================================

def parse_qa(lines):

    qa_pairs = []

    current_question = None

    current_answer = []

    qid = 1

    for line in lines:

        line = line.strip()

        if is_question(line):

            if current_question:

                qa_pairs.append({
                    "question_id": f"Q{qid}",
                    "question": current_question,
                    "answer": " ".join(current_answer).strip()
                })

                qid += 1

            current_question = line

            current_answer = []

        else:

            if current_question:
                current_answer.append(line)

    if current_question:

        qa_pairs.append({
            "question_id": f"Q{qid}",
            "question": current_question,
            "answer": " ".join(current_answer).strip()
        })

    return qa_pairs

# =========================================================
# MAIN PIPELINE
# =========================================================

def process_pdf(pdf_path):

    ocr_json = run_ocr(pdf_path)

    lines = extract_lines(ocr_json)

    qa_pairs = parse_qa(lines)

    final_json = {
        "total_qa_pairs": len(qa_pairs),
        "qa_pairs": qa_pairs
    }

    os.makedirs("outputs", exist_ok=True)

    output_path = "outputs/output.json"

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(
            final_json,
            f,
            ensure_ascii=False,
            indent=4
        )

    return output_path