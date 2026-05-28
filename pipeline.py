import os
import io
import re
import json
import fitz  # PyMuPDF
from pathlib import Path
from dotenv import load_dotenv
# CORRECTED IMPORT for mistralai >= 1.0.0
from mistralai.client import MistralClient
from rapidfuzz import fuzz
import streamlit as st

# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

# Initialize Mistral Client (Corrected for v1.0+)
try:
    api_key = st.secrets["MISTRAL_API_KEY"]
except (AttributeError, KeyError):
    api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    st.error("Mistral API Key not found.")
    st.stop()

# CORRECTED INITIALIZATION
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
    r"^TOPIC\s+DATE$",
]

REMOVE_LINES_CONTAINING = [
    "TOPIC",
    "DATE",
]

# =========================================================
# PREPROCESS PDF (Using PyMuPDF)
# =========================================================

def preprocess_pdf(file_bytes: bytes, dpi: int = 300) -> bytes:
    """
    Rasterizes the PDF (converts pages to images) to ensure clean OCR.
    Uses PyMuPDF (fitz) to avoid external dependencies like poppler.
    """
    print("Preprocessing PDF...")
    try:
        src_doc = fitz.open(stream=file_bytes, filetype="pdf")
        out_doc = fitz.open()

        for page in src_doc:
            # Render page to image
            pix = page.get_pixmap(dpi=dpi)
            # Create new PDF page with the image
            new_page = out_doc.new_page(width=pix.width, height=pix.height)
            new_page.insert_image(new_page.rect, pixmap=pix)

        pdf_bytes = io.BytesIO()
        out_doc.save(pdf_bytes)
        
        src_doc.close()
        out_doc.close()
        
        pdf_bytes.seek(0)
        print("Converted to image-based PDF successfully.")
        return pdf_bytes.read()

    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return file_bytes

# =========================================================
# OCR (Fixed for Mistral 1.0+)
# =========================================================

def run_ocr(file_content: bytes, file_name: str):
    print("Running OCR directly on file bytes...")
    
    # MISTRAL 1.0+ SDK:
    # Use 'process_ocr' to upload and process in one step.
    response = client.ocr.process_ocr(
        file={
            "file_name": file_name,
            "content": file_content,
        },
        model="mistral-ocr-latest",
    )

    return response

# =========================================================
# OCR TO CLEAN JSON
# =========================================================

def ocr_to_clean_json(ocr_response):
    pages_data = []

    # Handle object or dict response
    pages = ocr_response.pages if hasattr(ocr_response, 'pages') else ocr_response.get('pages', [])

    for page in pages:
        markdown_content = page.markdown if hasattr(page, 'markdown') else page.get('markdown', '')
        
        lines = markdown_content.split("\n")
        seen = set()
        clean_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line not in seen:
                seen.add(line)
                clean_lines.append(line)

        page_idx = page.index if hasattr(page, 'index') else page.get('index', len(pages_data))

        pages_data.append({
            "page_number": page_idx + 1,
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
        preferred_keys = ["text", "markdown", "content", "value"]
        for key in preferred_keys:
            if key in data:
                parts.append(extract_text(data[key]))
        if not parts:
            for v in data.values():
                parts.append(extract_text(v))
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
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("—", "-")
    
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
# ROMAN CHECK
# =========================================================

def is_roman(token):
    token = token.lower().strip()
    romans = [
        "i", "ii", "iii", "iv", "v",
        "vi", "vii", "viii", "ix", "x",
        "xi", "xii", "xiii", "xiv", "xv"
    ]
    return token in romans

# =========================================================
# EXTRACT OFFICIAL QUESTIONS
# =========================================================

def extract_official_questions(pages):
    assignment_page = None
    
    for page in pages:
        text = extract_text(page)
        normalized = normalize(text)
        if "section a" in normalized and "section b" in normalized:
            assignment_page = text
            break

    if not assignment_page:
        print("Warning: Assignment page not found.")
        return []

    lines = assignment_page.split("\n")
    questions = []
    current_section = None

    for line in lines:
        line = clean_text(line)
        if is_noise(line):
            continue

        if "Section A" in line:
            current_section = "A"
            continue
        elif "Section B" in line:
            current_section = "B"
            continue

        # SECTION A
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

        # SECTION B
        elif current_section == "B":
            match = re.match(
                r'^(\d+)\.\s*(.+)',
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

    unique_questions = []
    seen = set()
    for q in questions:
        key = normalize(q["question"])
        if key not in seen:
            seen.add(key)
            unique_questions.append(q)

    return unique_questions

# =========================================================
# QUESTION MATCHER
# =========================================================

def match_question(line, official_questions):
    line_norm = normalize(line)
    best_match = None
    best_score = 0

    for q in official_questions:
        q_norm = normalize(q["question"])
        score = fuzz.partial_ratio(line_norm, q_norm)
        if score > best_score:
            best_score = score
            best_match = q

    if best_score >= SIMILARITY_THRESHOLD:
        return best_match
    return None

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
# PARSE ANSWERS
# =========================================================

def parse_answers(pages, official_questions):
    qa_map = {}
    current_qid = None
    current_question = None
    current_answer_lines = []

    for page in pages:
        page_text = extract_text(page)
        lines = page_text.split("\n")

        for raw_line in lines:
            line = clean_text(raw_line)
            if is_noise(line):
                continue

            matched = match_question(line, official_questions)

            if matched:
                new_qid = matched["id"]
                if new_qid != current_qid:
                    if current_qid:
                        qa_map[current_qid] = {
                            "question": current_question,
                            "answer": clean_answer(" ".join(current_answer_lines))
                        }
                    
                    current_qid = new_qid
                    current_question = matched["question"]
                    current_answer_lines = []
                    continue

            if current_qid:
                similarity = fuzz.partial_ratio(normalize(line), normalize(current_question))
                if similarity > 90:
                    continue
                current_answer_lines.append(line)

    if current_qid:
        qa_map[current_qid] = {
            "question": current_question,
            "answer": clean_answer(" ".join(current_answer_lines))
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
    """
    Main entry point for Streamlit.
    """
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name

    # 1. Preprocess (Rasterize)
    processed_bytes = preprocess_pdf(file_bytes)

    # 2. OCR
    ocr_result = run_ocr(processed_bytes, file_name)

    # 3. Convert OCR result to JSON
    ocr_json = ocr_to_clean_json(ocr_result)

    # 4. Extract Questions
    pages = ocr_json["pages"]
    official_questions = extract_official_questions(pages)

    # 5. Parse Answers
    qa_map = parse_answers(pages, official_questions)

    # 6. Build Final Output
    final_json = build_json(qa_map)

    return ocr_json, final_json