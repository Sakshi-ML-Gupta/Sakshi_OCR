

import os
import io
import re
import json
from pathlib import Path

import streamlit as st
from mistralai import Mistral
from pdf2image import convert_from_bytes
from rapidfuzz import fuzz


api_key = st.secrets["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)



SIMILARITY_THRESHOLD = 85

NOISE_PATTERNS = [
    r"^page\s+\d+$",
    r"^\d+$",
    r"^assignment$",
    r"^tma$",
    r"^topic$",
    r"^date$",
]

REMOVE_LINES_CONTAINING = [
    "enrolment",
    "programme",
    "course code",
    "study centre",
    "university",
    "mobile no",
    "email",
]



def normalize(text):

    text = str(text).lower()

    text = text.replace("—", "-")

    text = re.sub(r'[“”"\'`]', '', text)

    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def clean_text(text):

    text = str(text)

    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')

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

    text = "\n".join(cleaned)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def is_noise(line):

    line = clean_text(line)

    if not line:
        return True

    for pattern in NOISE_PATTERNS:

        if re.match(pattern, line, re.IGNORECASE):
            return True

    return False



def preprocess_pdf(pdf_bytes):

    try:

        images = convert_from_bytes(pdf_bytes, dpi=300)

        temp_pdf = io.BytesIO()

        images[0].save(
            temp_pdf,
            format="PDF",
            save_all=True,
            append_images=images[1:]
        )

        temp_pdf.seek(0)

        return temp_pdf.read()

    except Exception:

        return pdf_bytes


def run_ocr(file_bytes, file_name):

    uploaded_file = client.files.upload(
        file={
            "file_name": file_name,
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
            "document_url": signed_url.url
        }
    )

    return response



def extract_lines(ocr_response):

    all_lines = []

    for page in ocr_response.pages:

        page_text = page.markdown

        lines = page_text.split("\n")

        for line in lines:

            line = clean_text(line)

            if not line:
                continue

            if is_noise(line):
                continue

            all_lines.append(line)

    return all_lines



def detect_question(line):

    patterns = [

        r'^q[\.\s]*(\d+)[\)\.\-\s]+(.+)',

        r'^question[\s]*(\d+)[\)\.\-\s]+(.+)',

        r'^(\d+)[\)\.\-\s]+(.+)',

        r'^\(?([ivxlcdm]+)\)?[\)\.\-\s]+(.+)',
    ]

    for pattern in patterns:

        match = re.match(
            pattern,
            line,
            re.IGNORECASE
        )

        if match:

            qnum = match.group(1)
            qtext = match.group(2).strip()

            if len(qtext) < 8:
                continue

            if re.match(r'^[ivxlcdm]+$', qnum, re.IGNORECASE):

                qid = f"sub_{qnum.lower()}"

                qtype = "subquestion"

            else:

                qid = f"Q{qnum}"

                qtype = "main"

            return {
                "question_id": qid,
                "question_type": qtype,
                "question": qtext
            }

    return None


def detect_section(line):

    line_lower = line.lower()

    if "section a" in line_lower:
        return "Section A"

    if "section b" in line_lower:
        return "Section B"

    if "part a" in line_lower:
        return "Part A"

    if "part b" in line_lower:
        return "Part B"

    return None



def extract_qa(lines):

    qa_pairs = []

    current_question = None

    current_answer = []

    current_section = "General"

    for line in lines:

        section = detect_section(line)

        if section:
            current_section = section
            continue

        detected = detect_question(line)

        if detected:

            if current_question:

                qa_pairs.append({
                    "question_id": current_question["question_id"],
                    "section": current_section,
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



    if current_question:

        qa_pairs.append({
            "question_id": current_question["question_id"],
            "section": current_section,
            "question_type": current_question["question_type"],
            "question": current_question["question"],
            "answer": " ".join(current_answer).strip()
        })

    return qa_pairs



def process_pdf(uploaded_pdf):

    os.makedirs("outputs", exist_ok=True)

    pdf_bytes = uploaded_pdf.read()

    processed_pdf = preprocess_pdf(pdf_bytes)

    ocr_response = run_ocr(
        processed_pdf,
        uploaded_pdf.name
    )

    lines = extract_lines(ocr_response)

    qa_pairs = extract_qa(lines)

    final_json = {
        "total_qa_pairs": len(qa_pairs),
        "qa_pairs": qa_pairs
    }

    output_path = (
        f"outputs/"
        f"{Path(uploaded_pdf.name).stem}.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(
            final_json,
            f,
            ensure_ascii=False,
            indent=4
        )

    return output_path