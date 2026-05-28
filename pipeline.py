import os
import io
import re
import json
import fitz
import base64

from rapidfuzz import fuzz
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
    r"^TOPIC\s+DATE$",
]

REMOVE_LINES_CONTAINING = [
    "TOPIC",
    "DATE",
]

# =========================================================
# MISTRAL CLIENT
# =========================================================

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise Exception("MISTRAL_API_KEY not found")

client = MistralClient(api_key=api_key)

# =========================================================
# PREPROCESS PDF
# =========================================================

def preprocess_pdf(file_bytes: bytes, dpi: int = 300) -> bytes:

    print("Preprocessing PDF...")

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

        print("Converted to image-based PDF")

        return pdf_bytes.read()

    except Exception as e:

        print("Preprocessing failed:", e)

        return file_bytes

# =========================================================
# OCR
# =========================================================

def run_ocr(file_content: bytes, file_name: str):

    print("Running OCR...")

    base64_pdf = base64.b64encode(
        file_content
    ).decode("utf-8")

    response = client.chat.complete(

        model="mistral-large-latest",

        messages=[
            {
                "role": "user",
                "content": [

                    {
                        "type": "text",
                        "text": """
Extract ALL text from this PDF exactly as written.

Rules:
1. Preserve line breaks
2. Preserve question numbering
3. Preserve sections
4. Do not summarize
5. Return only extracted text
"""
                    },

                    {
                        "type": "document",
                        "document": {
                            "name": file_name,
                            "data": base64_pdf
                        }
                    }
                ]
            }
        ]
    )

    return response.choices[0].message.content

# =========================================================
# OCR TO JSON
# =========================================================

def ocr_to_clean_json(raw_text):

    pages_data = []

    chunks = raw_text.split("\n\n")

    for idx, chunk in enumerate(chunks):

        lines = chunk.split("\n")

        seen = set()

        clean_lines = []

        for line in lines:

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
# HELPERS
# =========================================================

def extract_text(data):

    if data is None:
        return ""

    if isinstance(data, str):
        return data

    if isinstance(data, list):

        return "\n".join([
            extract_text(x)
            for x in data
        ])

    if isinstance(data, dict):

        parts = []

        preferred_keys = [
            "text",
            "markdown",
            "content",
            "value"
        ]

        for key in preferred_keys:

            if key in data:

                parts.append(
                    extract_text(data[key])
                )

        if not parts:

            for v in data.values():

                parts.append(
                    extract_text(v)
                )

        return "\n".join(parts)

    return str(data)

# =========================================================
# NORMALIZE
# =========================================================

def normalize(text):

    text = str(text)

    text = text.lower()

    text = text.replace("—", "-")

    text = re.sub(r'[“”"\'`]', '', text)

    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = str(text)

    text = text.replace("\u201c", '"')

    text = text.replace("\u201d", '"')

    text = text.replace("—", "-")

    cleaned_lines = []

    for line in text.split("\n"):

        line_strip = line.strip()

        skip = False

        for item in REMOVE_LINES_CONTAINING:

            if item.lower() in line_strip.lower():

                skip = True

                break

        if not skip:

            cleaned_lines.append(line_strip)

    text = "\n".join(cleaned_lines)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# =========================================================
# NOISE FILTER
# =========================================================

def is_noise(line):

    line = clean_text(line)

    if not line:
        return True

    for pattern in NOISE_PATTERNS:

        if re.match(pattern, line, re.IGNORECASE):
            return True

    return False

# =========================================================
# CLEAN ANSWER
# =========================================================

def clean_answer(answer):

    answer = clean_text(answer)

    answer = re.sub(
        r'TOPIC\s*_*\s*DATE\s*_*\s*',
        ' ',
        answer,
        flags=re.IGNORECASE
    )

    answer = re.sub(r'\s+', ' ', answer)

    return answer.strip()

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
# BUILD FINAL JSON
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

    # QA EXTRACTION

    qa_map = parse_answers(
        ocr_json["pages"],
        questions
    )

    # FINAL JSON

    final_json = build_json(qa_map)

    return ocr_json, final_json