import os
import io
import re
import json
from pathlib import Path

import streamlit as st

from mistralai.client import MistralClient
from pdf2image import convert_from_path
from rapidfuzz import fuzz

# =========================================================
# API
# =========================================================

api_key = st.secrets["MISTRAL_API_KEY"]

client = MistralClient(api_key=api_key)

# =========================================================
# CONFIG
# =========================================================

SIMILARITY_THRESHOLD = 80

# =========================================================
# PDF PREPROCESS
# =========================================================

def preprocess_pdf(pdf_path):

    with open(pdf_path, "rb") as f:
        return f.read()

# =========================================================
# OCR
# =========================================================

def run_ocr(file_content, file_name):

    uploaded_file = client.files.upload(
        file={
            "file_name": file_name,
            "content": file_content,
        },
        purpose="ocr"
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
# CLEAN
# =========================================================

def normalize(text):

    text = str(text).lower()

    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# =========================================================
# OCR TO LINES
# =========================================================

def extract_lines(ocr_response):

    lines = []

    for page in ocr_response.pages:

        page_lines = page.markdown.split("\n")

        for line in page_lines:

            line = line.strip()

            if line:

                lines.append(line)

    return lines

# =========================================================
# QUESTION DETECTOR
# =========================================================

def detect_question(line):

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
# PARSER
# =========================================================

def parse_qa(lines):

    qa_pairs = []

    current_question = None

    current_answer = []

    qid = 1

    for line in lines:

        line_clean = line.strip()

        if detect_question(line_clean):

            if current_question:

                qa_pairs.append({
                    "question_id": f"Q{qid}",
                    "question": current_question,
                    "answer": " ".join(current_answer).strip()
                })

                qid += 1

            current_question = line_clean

            current_answer = []

        else:

            if current_question:

                current_answer.append(line_clean)

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

    pdf_bytes = preprocess_pdf(pdf_path)

    ocr_response = run_ocr(
        pdf_bytes,
        Path(pdf_path).name
    )

    lines = extract_lines(ocr_response)

    qa_pairs = parse_qa(lines)

    final_json = {
        "total_qa_pairs": len(qa_pairs),
        "qa_pairs": qa_pairs
    }

    os.makedirs("outputs", exist_ok=True)

    output_path = "outputs/final.json"

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(
            final_json,
            f,
            ensure_ascii=False,
            indent=4
        )

    return output_path