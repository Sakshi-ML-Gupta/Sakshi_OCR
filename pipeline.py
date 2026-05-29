import os
import io
import re
import json
import fitz
from pathlib import Path
import base64
from rapidfuzz import fuzz
from mistralai import Mistral

import streamlit as st

# =========================================================
# MISTRAL
# =========================================================

api_key = st.secrets["MISTRAL_API_KEY"]



client = Mistral(
    api_key=api_key
)


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



import time

# =========================================================
# OCR
# =========================================================

import time
import base64


# =========================================================
# OCR
# =========================================================


# =========================================================
# OCR
# =========================================================

def run_ocr(file_content: bytes, file_name: str, status_callback=None):

    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    log("Starting OCR — uploading file to Mistral...")

    try:
        import io as _io

        # ============================================
        # UPLOAD FILE
        # ============================================

        uploaded_pdf = client.files.upload(
            file={
                "file_name": file_name,
                "content": file_content
            },
            purpose="ocr"
        )

        log(f"File uploaded: {uploaded_pdf.id}")

        # ============================================
        # GET SIGNED URL
        # ============================================

        signed = client.files.get_signed_url(file_id=uploaded_pdf.id)

        log("Signed URL obtained, running OCR model...")

        # ============================================
        # OCR via CHAT with document_url
        # ============================================

        response = client.chat.complete(
            model="mistral-ocr-latest",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document_url",
                            "document_url": signed.url
                        },
                        {
                            "type": "text",
                            "text": "Extract ALL text from this document exactly as it appears. Preserve structure, section headings, numbering, and formatting. Output only the raw extracted text."
                        }
                    ]
                }
            ]
        )

        log("OCR model responded")

        # ============================================
        # EXTRACT TEXT
        # ============================================

        final_text = response.choices[0].message.content

        if not final_text or not final_text.strip():
            raise Exception("OCR returned empty text")

        log(f"OCR complete — extracted {len(final_text)} characters")

        # ============================================
        # CLEANUP
        # ============================================

        try:
            client.files.delete(file_id=uploaded_pdf.id)
            log("Temp file deleted from Mistral")
        except Exception:
            pass

        return final_text

    except Exception as e:
        raise Exception(f"OCR failed: {str(e)}")

# =========================================================
# OCR TO CLEAN JSON
# =========================================================

def ocr_to_clean_json(raw_text):

    pages_data = []

    # split pseudo pages
    raw_pages = raw_text.split("\n\n")

    for idx, page_text in enumerate(raw_pages):

        lines = []

        seen = set()

        for line in page_text.split("\n"):

            line = line.strip()

            if not line:
                continue

            if line not in seen:

                seen.add(line)

                lines.append(line)

        if lines:

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

# =========================================================
# EXTRACT OFFICIAL QUESTIONS
# =========================================================

def extract_official_questions(pages):

    assignment_page = None

    # =====================================================
    # FIND PAGE CONTAINING QUESTIONS
    # =====================================================

    for page in pages:

        text = extract_text(page)

        normalized = normalize(text)

        if (
            "section a" in normalized
            or "section b" in normalized
        ):

            assignment_page = text
            break

    # fallback
    if not assignment_page:

        assignment_page = extract_text(pages[0])

    lines = assignment_page.split("\n")

    questions = []

    current_section = None

    # =====================================================
    # EXTRACT QUESTIONS
    # =====================================================

    for line in lines:

        line = clean_text(line)

        if is_noise(line):
            continue

        # SECTION A

        if "section a" in line.lower():

            current_section = "A"

            continue

        # SECTION B

        if "section b" in line.lower():

            current_section = "B"

            continue

        # =================================================
        # SECTION A QUESTIONS
        # =================================================

        if current_section == "A":

            match = re.match(
                r'^(?:\d+\s*\.?\s*)?\(?([ivxlcdm]+)\)?[\.\)]?\s*(.+)',
                line,
                re.IGNORECASE
            )

            if match:

                roman = match.group(1)

                qtext = match.group(2)

                if is_roman(roman) and len(qtext) > 15:

                    qid = f"A1({roman.lower()})"

                    questions.append({
                        "id": qid,
                        "question": qtext
                    })

        # =================================================
        # SECTION B QUESTIONS
        # =================================================

        elif current_section == "B":

            match = re.match(
                r'^(\d+)[\.\)]\s*(.+)',
                line
            )

            if match:

                num = match.group(1)

                qtext = match.group(2)

                if len(qtext) > 10:

                    questions.append({
                        "id": f"B{num}",
                        "question": qtext
                    })

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    unique_questions = []

    seen = set()

    for q in questions:

        key = normalize(q["question"])

        if key not in seen:

            seen.add(key)

            unique_questions.append(q)

    return unique_questions
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

# =========================================================
# COMPLETE PIPELINE
# =========================================================

# =========================================================
# COMPLETE PIPELINE
# =========================================================

# =========================================================
# COMPLETE PIPELINE
# =========================================================

def process_pdf(uploaded_file, status_callback=None):

    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    log("Reading uploaded file...")
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name

    log("Preprocessing PDF (rasterizing pages)...")
    processed_bytes = preprocess_pdf(file_bytes)

    log("Running OCR...")
    raw_text = run_ocr(processed_bytes, file_name, status_callback=status_callback)

    log("Building OCR JSON...")
    ocr_json = ocr_to_clean_json(raw_text)

    pages = ocr_json["pages"]

    log("Extracting official questions...")
    official_questions = extract_official_questions(pages)
    log(f"Found {len(official_questions)} questions")

    log("Mapping answers...")
    qa_map = parse_answers(pages, official_questions)

    log("Building final JSON...")
    final_json = build_json(qa_map)

    log(f"Done — {final_json['total_qa_pairs']} QA pairs extracted")
    return ocr_json, final_json