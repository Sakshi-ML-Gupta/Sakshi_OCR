import os
import io
import re
import json
import fitz
import base64
from pathlib import Path
from rapidfuzz import fuzz

# =========================================================
# MISTRAL CLIENT
# =========================================================

try:
    import streamlit as st
    api_key = st.secrets["MISTRAL_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")

from mistralai import Mistral
client = Mistral(api_key=api_key)

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
    r"^TOPIC\s+DATE$",
]

REMOVE_LINES_CONTAINING = [
    "TOPIC",
    "DATE",
]

# =========================================================
# PREPROCESS PDF
# =========================================================

def preprocess_pdf(file_input, dpi=300):
    """Accepts file path (str) or bytes."""

    try:
        if isinstance(file_input, (str, Path)):
            file_bytes = Path(file_input).read_bytes()
        else:
            file_bytes = file_input

        src_doc = fitz.open(stream=file_bytes, filetype="pdf")
        out_doc = fitz.open()

        for page in src_doc:
            pix = page.get_pixmap(dpi=dpi)
            new_page = out_doc.new_page(
                width=pix.width,
                height=pix.height
            )
            new_page.insert_image(new_page.rect, pixmap=pix)

        pdf_bytes = io.BytesIO()
        out_doc.save(pdf_bytes)
        src_doc.close()
        out_doc.close()
        pdf_bytes.seek(0)

        return pdf_bytes.read()

    except Exception as e:
        print(f"Preprocessing failed: {e}")
        if isinstance(file_input, (str, Path)):
            return Path(file_input).read_bytes()
        return file_input

# =========================================================
# OCR
# =========================================================

def run_ocr(file_content: bytes, file_name: str, status_callback=None):

    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    log("Starting OCR — converting pages to images...")

    try:
        src_doc = fitz.open(stream=file_content, filetype="pdf")
        all_text = []

        for page_num, page in enumerate(src_doc):

            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            log(f"OCR processing page {page_num + 1} of {len(src_doc)}...")

            response = client.chat.complete(
                model="pixtral-12b-2409",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Extract ALL text from this image exactly as it appears. "
                                    "Preserve structure, section headings, numbering and formatting. "
                                    "Output only the raw extracted text, nothing else."
                                )
                            }
                        ]
                    }
                ]
            )

            page_text = response.choices[0].message.content
            if page_text:
                all_text.append(page_text)

        src_doc.close()

        final_text = "\n\n".join(all_text)

        if not final_text.strip():
            raise Exception("OCR returned empty text")

        log(f"OCR complete — {len(final_text)} characters extracted")
        return final_text

    except Exception as e:
        raise Exception(f"OCR failed: {str(e)}")

# =========================================================
# OCR TO CLEAN JSON
# =========================================================

def ocr_to_clean_json(raw_text: str):

    pages_data = []
    raw_pages = raw_text.split("\n\n")

    for idx, page_text in enumerate(raw_pages):

        seen = set()
        clean_lines = []

        for line in page_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line not in seen:
                seen.add(line)
                clean_lines.append(line)

        if clean_lines:
            pages_data.append({
                "page_number": idx + 1,
                "text": clean_lines
            })

    return {
        "total_pages": len(pages_data),
        "pages": pages_data
    }

# =========================================================
# TEXT HELPERS
# =========================================================

def extract_text(data):

    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        return "\n".join([extract_text(x) for x in data])
    if isinstance(data, dict):
        parts = []
        for key in ["text", "markdown", "content", "value"]:
            if key in data:
                parts.append(extract_text(data[key]))
        if not parts:
            for v in data.values():
                parts.append(extract_text(v))
        return "\n".join(parts)
    return str(data)

def normalize(text):
    text = str(text).lower()
    text = text.replace("—", "-")
    text = re.sub(r'["""\'`]', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_text(text):
    text = str(text)
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("—", "-")
    cleaned_lines = []
    for line in text.split("\n"):
        line_strip = line.strip()
        skip = any(
            item.lower() in line_strip.lower()
            for item in REMOVE_LINES_CONTAINING
        )
        if not skip:
            cleaned_lines.append(line_strip)
    text = "\n".join(cleaned_lines)
    return re.sub(r'\s+', ' ', text).strip()

def is_noise(line):
    line = clean_text(line)
    if not line:
        return True
    return any(
        re.match(pattern, line, re.IGNORECASE)
        for pattern in NOISE_PATTERNS
    )

def is_roman(token):
    romans = [
        "i","ii","iii","iv","v","vi","vii",
        "viii","ix","x","xi","xii","xiii","xiv","xv"
    ]
    return token.lower().strip() in romans

# =========================================================
# EXTRACT OFFICIAL QUESTIONS
# =========================================================

def extract_official_questions(pages):

    # =========================================================
    # STEP 1: Find pages containing Section A and Section B
    # They may be on different pages
    # =========================================================

    section_a_text = None
    section_b_text = None

    for page in pages:

        text = extract_text(page)
        normalized = normalize(text)

        if "section a" in normalized and section_a_text is None:
            section_a_text = text

        if "section b" in normalized and section_b_text is None:
            section_b_text = text

    # fallback: if both on same page
    if section_a_text is None and section_b_text is None:

        # try to find any page with question-like content
        all_text = "\n".join(extract_text(p) for p in pages)
        normalized_all = normalize(all_text)

        if "section a" not in normalized_all and "section b" not in normalized_all:
            raise Exception(
                f"Assignment page (Section A + B) not found.\n"
                f"Debug — first 500 chars of OCR output:\n"
                f"{all_text[:500]}"
            )

        section_a_text = all_text
        section_b_text = all_text

    # if one is missing, try using all pages combined
    combined = "\n".join(extract_text(p) for p in pages)

    if section_a_text is None:
        section_a_text = combined

    if section_b_text is None:
        section_b_text = combined

    # =========================================================
    # STEP 2: Parse combined text in one pass
    # =========================================================

    full_text = combined
    lines = full_text.split("\n")

    questions = []
    current_section = None

    for line in lines:

        line = clean_text(line)

        if is_noise(line):
            continue

        # flexible section detection — handles bold markdown (**Section A**)
        clean_line = re.sub(r'[*_#]', '', line).strip()

        if re.search(r'section\s*a\b', clean_line, re.IGNORECASE):
            current_section = "A"
            continue

        if re.search(r'section\s*b\b', clean_line, re.IGNORECASE):
            current_section = "B"
            continue

        # =====================================================
        # SECTION A — roman numeral questions
        # =====================================================

        if current_section == "A":

            match = re.match(
                r'^(?:\d+\s*\.?\s*)?\(?([ivxlcdm]+)\)?[\.\):\s]\s*(.+)',
                line,
                re.IGNORECASE
            )

            if match:
                roman = match.group(1)
                qtext = match.group(2).strip()

                if is_roman(roman) and len(qtext) > 10:
                    questions.append({
                        "id": f"A1({roman.lower()})",
                        "question": qtext
                    })

        # =====================================================
        # SECTION B — numbered questions
        # =====================================================

        elif current_section == "B":

            match = re.match(
                r'^(\d+)[\.\)\s]\s*(.+)',
                line
            )

            if match:
                num = match.group(1)
                qtext = match.group(2).strip()

                if len(qtext) > 10:
                    questions.append({
                        "id": f"B{num}",
                        "question": qtext
                    })

    # =========================================================
    # STEP 3: Deduplicate
    # =========================================================

    seen = set()
    unique = []

    for q in questions:
        key = normalize(q["question"])
        if key not in seen:
            seen.add(key)
            unique.append(q)

    if not unique:
        raise Exception(
            f"Sections found but no questions extracted.\n"
            f"Debug — first 800 chars of OCR:\n"
            f"{combined[:800]}"
        )

    return unique

# =========================================================
# PARSE ANSWERS
# =========================================================

def parse_answers(pages, official_questions):

    qa_map = {}
    current_qid = None
    current_question = None
    current_answer_lines = []

    for page in pages:

        page_text = extract_text(page)

        for raw_line in page_text.split("\n"):

            line = clean_text(raw_line)

            if is_noise(line):
                continue

            # check if line matches a question
            best_match = None
            best_score = 0
            for q in official_questions:
                score = fuzz.partial_ratio(
                    normalize(line), normalize(q["question"])
                )
                if score > best_score:
                    best_score = score
                    best_match = q

            if best_match and best_score >= SIMILARITY_THRESHOLD:

                if best_match["id"] != current_qid:

                    # save previous
                    if current_qid:
                        qa_map[current_qid] = {
                            "question": current_question,
                            "answer": clean_answer(" ".join(current_answer_lines))
                        }

                    current_qid = best_match["id"]
                    current_question = best_match["question"]
                    current_answer_lines = []
                    continue

            if current_qid:
                if fuzz.partial_ratio(normalize(line), normalize(current_question)) > 90:
                    continue
                current_answer_lines.append(line)

    # save last question
    if current_qid:
        qa_map[current_qid] = {
            "question": current_question,
            "answer": clean_answer(" ".join(current_answer_lines))
        }

    return qa_map

def clean_answer(answer):
    answer = clean_text(answer)
    answer = re.sub(r'TOPIC\s*_*\s*DATE\s*_*\s*', ' ', answer, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', answer).strip()

# =========================================================
# BUILD JSON
# =========================================================

def build_json(qa_map):
    return {
        "total_qa_pairs": len(qa_map),
        "qa_pairs": [
            {
                "question_id": qid,
                "question": qa["question"],
                "answer": qa["answer"]
            }
            for qid, qa in qa_map.items()
        ]
    }

# =========================================================
# COMPLETE PIPELINE
# =========================================================

def process_pdf(file_input, status_callback=None):
    """
    Accepts either:
    - a file path string (local usage)
    - a Streamlit UploadedFile object (cloud usage)
    """

    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    # ── read bytes ──────────────────────────────────────
    if isinstance(file_input, (str, Path)):
        file_bytes = Path(file_input).read_bytes()
        file_name = Path(file_input).name
    else:
        # Streamlit UploadedFile
        file_bytes = file_input.read()
        file_name = file_input.name

    log("Preprocessing PDF...")
    processed_bytes = preprocess_pdf(file_bytes)

    log("Running OCR...")
    raw_text = run_ocr(processed_bytes, file_name, status_callback=status_callback)

    log("Building OCR JSON...")
    ocr_json = ocr_to_clean_json(raw_text)

    pages = ocr_json["pages"]

    log("Extracting questions...")
    official_questions = extract_official_questions(pages)
    log(f"Found {len(official_questions)} questions")

    log("Mapping answers...")
    qa_map = parse_answers(pages, official_questions)

    log("Building final output...")
    final_json = build_json(qa_map)

    log(f"Done — {final_json['total_qa_pairs']} QA pairs extracted")
    return ocr_json, final_json