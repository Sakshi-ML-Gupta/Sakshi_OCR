import os
import io
import re
import json
import tempfile
from pathlib import Path

import streamlit as st

from mistralai.client import MistralClient
from pdf2image import convert_from_bytes
from rapidfuzz import fuzz

# =========================================================
# API KEY
# =========================================================

api_key = st.secrets["MISTRAL_API_KEY"]

client = MistralClient(api_key=api_key)

# =========================================================
# CLEANERS
# =========================================================

def normalize(text):

    text = str(text).lower()

    text = re.sub(r'[“”"\'`]', '', text)

    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def clean_text(text):

    text = str(text)

    text = text.replace("\n", " ")

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# =========================================================
# OCR
# =========================================================

def preprocess_pdf(pdf_bytes):

    try:

        images = convert_from_bytes(
            pdf_bytes,
            dpi=300
        )

        pdf_buffer = io.BytesIO()

        images[0].save(
            pdf_buffer,
            format="PDF",
            save_all=True,
            append_images=images[1:]
        )

        pdf_buffer.seek(0)

        return pdf_buffer.read()

    except Exception:

        return pdf_bytes


def run_ocr(file_content, file_name):

    print("Uploading to Mistral...")

    uploaded_file = client.files.upload(
        file={
            "file_name": file_name,
            "content": file_content,
        },
        purpose="ocr"
    )

    print("Getting signed URL...")

    signed_url = client.files.get_signed_url(
        uploaded_file.id
    )

    print("Running OCR...")

    response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url
        }
    )

    return response

# =========================================================
# EXTRACT QA
# =========================================================

def extract_qa(ocr_response):

    all_lines = []

    for page in ocr_response.pages:

        lines = page.markdown.split("\n")

        for line in lines:

            line = clean_text(line)

            if line:
                all_lines.append(line)

    qa_pairs = []

    current_question = None

    current_answer = []

    for line in all_lines:

        detected = detect_question(line)

        if detected:

            if current_question:

                qa_pairs.append({
                    "question_id": current_question["question_id"],
                    "question_type": current_question["question_type"],
                    "question": current_question["question"],
                    "answer": " ".join(current_answer).strip()
                })

            current_question = detected

            current_answer = []

            continue

        if current_question:

            similarity = fuzz.partial_ratio(
                normalize(line),
                normalize(current_question["question"])
            )

            if similarity > 92:
                continue

            current_answer.append(line)

    # =====================================
    # LAST QA
    # =====================================

    if current_question:

        qa_pairs.append({
            "question_id": current_question["question_id"],
            "question_type": current_question["question_type"],
            "question": current_question["question"],
            "answer": " ".join(current_answer).strip()
        })

    return qa_pairs

# =========================================================
# MAIN PIPELINE
# =========================================================

def process_pdf(uploaded_file):

    os.makedirs("outputs", exist_ok=True)

    pdf_bytes = uploaded_file.read()

    processed_pdf = preprocess_pdf(
        pdf_bytes
    )

    ocr_response = run_ocr(
        processed_pdf,
        uploaded_file.name
    )

    qa_pairs = extract_qa(
        ocr_response
    )

    final_json = {
        "total_qa_pairs": len(qa_pairs),
        "qa_pairs": qa_pairs
    }

    output_path = (
        f"outputs/"
        f"{Path(uploaded_file.name).stem}.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_json,
            f,
            ensure_ascii=False,
            indent=4
        )

    return output_path