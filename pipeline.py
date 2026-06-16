import os
import io
import re
import json
import fitz
import base64
import httpx
from pathlib import Path
from mistralai import Mistral

# =========================================================
# API KEY
# =========================================================

def get_api_key(name):
    try:
        import streamlit as st
        return st.secrets[name]
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv(name)


# =========================================================
# PREPROCESS PDF
# =========================================================

def preprocess_pdf(file_bytes, dpi=250):
    src_doc = fitz.open(stream=file_bytes, filetype="pdf")
    out_doc = fitz.open()
    for page in src_doc:
        pix = page.get_pixmap(dpi=dpi)
        new_page = out_doc.new_page(width=pix.width, height=pix.height)
        new_page.insert_image(new_page.rect, pixmap=pix)
    buf = io.BytesIO()
    out_doc.save(buf)
    src_doc.close()
    out_doc.close()
    buf.seek(0)
    return buf.read()


# =========================================================
# OCR
# =========================================================

def run_ocr(file_content: bytes, file_name: str, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    api_key = get_api_key("MISTRAL_API_KEY")
    client  = Mistral(api_key=api_key)

    log("Uploading to Mistral OCR...")
    uploaded = client.files.upload(
        file={"file_name": file_name, "content": file_content},
        purpose="ocr"
    )
    signed = client.files.get_signed_url(file_id=uploaded.id, expiry=1)
    log("Running OCR...")

    resp = httpx.post(
        "https://api.mistral.ai/v1/ocr",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": signed.url
            },
            "include_image_base64": False
        },
        timeout=180
    )

    if resp.status_code != 200:
        raise Exception(f"OCR error {resp.status_code}: {resp.text}")

    try:
        client.files.delete(file_id=uploaded.id)
    except Exception:
        pass

    pages = []
    for page in resp.json().get("pages", []):
        pages.append({
            "page_number": page.get("index", 0) + 1,
            "raw_text":    page.get("markdown", "")
        })

    log(f"OCR done — {len(pages)} pages")
    return pages


# =========================================================
# BUILD OCR JSON
# =========================================================

def build_ocr_json(pages: list) -> dict:
    return {
        "total_pages": len(pages),
        "pages": [
            {"page_number": p["page_number"], "text": p["raw_text"]}
            for p in pages
        ]
    }


# =========================================================
# REFERENCE BOOK — page by page, base64 inline
# =========================================================

def ocr_page_base64(page_b64: str, api_key: str) -> str:
    resp = httpx.post(
        "https://api.mistral.ai/v1/ocr",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{page_b64}"
            },
            "include_image_base64": False
        },
        timeout=120
    )
    if resp.status_code == 200:
        ocr_pages = resp.json().get("pages", [])
        return ocr_pages[0].get("markdown", "") if ocr_pages else ""
    raise Exception(f"OCR error {resp.status_code}: {resp.text[:200]}")


def process_reference(file_input, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    if isinstance(file_input, (str, Path)):
        file_bytes = Path(file_input).read_bytes()
    else:
        file_bytes = file_input.read()

    api_key     = get_api_key("MISTRAL_API_KEY")
    src_doc     = fitz.open(stream=file_bytes, filetype="pdf")
    total_pages = len(src_doc)
    src_doc.close()

    log(f"Reference book: {total_pages} page(s)")
    pages_output = []

    for page_num in range(total_pages):
        log(f"Page {page_num + 1} of {total_pages}...")
        try:
            src_doc  = fitz.open(stream=file_bytes, filetype="pdf")
            one_page = fitz.open()
            one_page.insert_pdf(src_doc, from_page=page_num, to_page=page_num)
            src_doc.close()

            pix      = one_page[0].get_pixmap(dpi=200)
            out_doc  = fitz.open()
            new_page = out_doc.new_page(width=pix.width, height=pix.height)
            new_page.insert_image(new_page.rect, pixmap=pix)

            buf      = io.BytesIO()
            out_doc.save(buf)
            page_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            one_page.close()
            out_doc.close()
            buf.close()

            text = ocr_page_base64(page_b64, api_key)
        except Exception as e:
            log(f"  Page {page_num + 1} failed: {e}")
            text = ""

        pages_output.append({"page_number": page_num + 1, "raw_text": text})

    log(f"Reference OCR complete — {total_pages} pages")
    return build_ocr_json(pages_output)


# =========================================================
# DETECT QUESTION PAPER PAGE
# The page that contains the printed list of questions
# Identified by having numbered questions >= 5 items
# =========================================================

def find_question_paper_page(pages: list) -> int:
    """
    Returns index (0-based) of the page that is the question paper.
    Looks for a page with 5+ numbered question lines.
    """
    Q_LINE = re.compile(r'^\s*\d+[\.\)]\s+.{20,}')

    best_idx   = -1
    best_count = 0

    for i, page in enumerate(pages):
        count = sum(
            1 for line in page["raw_text"].split("\n")
            if Q_LINE.match(line.strip())
        )
        if count > best_count:
            best_count = count
            best_idx   = i

    return best_idx if best_count >= 3 else -1


# =========================================================
# EXTRACT OFFICIAL QUESTIONS FROM QUESTION PAPER PAGE
# Returns list of question strings in order
# =========================================================

def extract_official_questions(page_text: str) -> list:
    """
    Extracts numbered questions from the question paper page.
    Handles multi-line questions by joining continuation lines.
    """
    lines     = page_text.split("\n")
    questions = []
    current   = None

    # Matches: "1. text", "2. text", etc.
    Q_START = re.compile(r'^\s*(\d+)[\.\)]\s+(.+)')
    # Section headers to skip
    SKIP    = re.compile(r'^#+\s|^भाग|^PART|^\s*$')

    for line in lines:
        stripped = line.strip()
        if not stripped or SKIP.match(stripped):
            if current:
                questions.append(current.strip())
                current = None
            continue

        m = Q_START.match(stripped)
        if m:
            if current:
                questions.append(current.strip())
            current = stripped
        elif current:
            # continuation of previous question
            current += " " + stripped

    if current:
        questions.append(current.strip())

    return questions


# =========================================================
# FIND ANSWER BLOCKS IN ANSWER PAGES
#
# Two strategies tried in order:
#
# Strategy A — उत्तर / Ans marker based:
#   Scan all answer page lines for "उत्तर", "Ans-", "उत्तर:"
#   Each marker = start of a new answer block
#   Text from marker to next marker = answer
#
# Strategy B — TOPIC/DATE split (English answer sheets):
#   If strategy A finds nothing, fall back to the original
#   question-paper-split approach
# =========================================================

# Answer start markers
ANS_RE = re.compile(
    r'^(?:'
    r'उत्तर\s*[\-\:\s]'        # उत्तर - / उत्तर:
    r'|Ans\.?\s*[\-\:\→]'       # Ans- / Ans:
    r'|Answer\s*[\-\:\→]'       # Answer:
    r')'
)

# Lines to skip (headers/footers)
NOISE_RE = re.compile(
    r'(?:Teacher\'?s?\s*Signature'
    r'|PAGE\s*NO'
    r'|^\s*DATE'
    r'|Neel?\s*Kamal'
    r'|Neal?\s*Kamal'
    r'|TAKMA\s*SINAN'
    r'|Facebook\'?s?\s*Signature'
    r'|Tancher\'?s?\s*Signature)',
    re.IGNORECASE
)

def is_noise(line: str) -> bool:
    return bool(NOISE_RE.search(line))


def extract_answers_by_uttar(answer_lines: list, status_callback=None) -> list:
    """
    Finds उत्तर / Ans markers in the answer lines.
    Returns list of raw answer strings in order found.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    # Find positions of all उत्तर markers
    positions = []
    for i, line in enumerate(answer_lines):
        if ANS_RE.match(line.strip()):
            positions.append(i)

    log(f"Found {len(positions)} उत्तर/Ans markers")

    if not positions:
        return []

    answers = []
    for idx, pos in enumerate(positions):
        end = positions[idx + 1] if idx + 1 < len(positions) else len(answer_lines)

        # Include the उत्तर line itself (it has answer content after the marker)
        raw_lines = []
        for j in range(pos, end):
            line = answer_lines[j]
            if is_noise(line):
                continue
            raw_lines.append(line.strip())

        answer_text = " ".join(l for l in raw_lines if l).strip()

        # Strip the "उत्तर -" / "Ans-" prefix from the first line
        answer_text = ANS_RE.sub("", answer_text, count=1).strip()
        answer_text = re.sub(r'^[\-\:\→\s]+', '', answer_text).strip()

        answers.append(answer_text)

    return answers


# =========================================================
# SPLIT FORMAT — TOPIC/DATE answer sheets (English format)
# =========================================================

def find_answer_start_page(pages: list) -> int:
    for i, page in enumerate(pages):
        text = page["raw_text"]
        if re.search(r'\bTOPIC\b', text) and re.search(r'\bDATE\b', text):
            return i
    return -1


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def similarity(a: str, b: str) -> float:
    wa = set(normalize(a).split())
    wb = set(normalize(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def strip_leading_label(text: str) -> str:
    text = re.sub(r'^(?:Ans(?:wer)?[.\s]+)', '', text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'^(?:Q\.?\s*)?\d+[.)]\s*', '', text.strip(), flags=re.IGNORECASE)
    return text.strip()


def strip_markdown(line: str) -> str:
    return re.sub(r'^\s*[-*+]\s+', '', line)


STOPWORDS = {
    'the','a','an','is','are','was','were','of','in','to','and','that',
    'this','it','he','she','they','i','we','you','at','or','but','not',
    'with','for','on','from','by','as','be','his','her','its','their'
}

LABEL_RE = re.compile(
    r"^\s*(?:(?:Q\.?\s*)?\d+[.)]\s*)?"
    r"(?:\(?\w+[.)]\s|\"|\d+[.)]\s)"
)


def find_question_boundaries_in_answers(answer_lines, questions,
                                         similarity_threshold=0.35, window=5):
    boundaries = []
    used_questions = set()
    used_line_indices = set()

    for i in range(len(answer_lines)):
        line_i = strip_markdown(answer_lines[i].strip())
        if not LABEL_RE.match(line_i):
            continue

        for w in range(1, window + 1):
            if i + w > len(answer_lines):
                break
            combined = " ".join(
                strip_markdown(answer_lines[i + k].strip())
                for k in range(w) if answer_lines[i + k].strip()
            )
            if len(combined) < 10:
                continue
            combined_clean = strip_leading_label(combined)

            best_score = 0
            best_q = None
            for q in questions:
                if q in used_questions:
                    continue
                s1 = similarity(combined, q)
                s2 = similarity(combined_clean, strip_leading_label(q))
                cw = normalize(combined_clean).split()
                qw = normalize(strip_leading_label(q)).split()
                bc = " ".join(cw[1:]) if len(cw) > 1 else ""
                bq = " ".join(qw[1:]) if len(qw) > 1 else ""
                s3 = similarity(bc, bq) if bc and bq else 0
                score = max(s1, s2, s3)
                if score > best_score:
                    best_score = score
                    best_q = q

            if best_q and best_score >= similarity_threshold:
                if i not in used_line_indices:
                    boundaries.append({"question": best_q, "line_index": i})
                    used_questions.add(best_q)
                    used_line_indices.add(i)
                break

    boundaries.sort(key=lambda b: b["line_index"])
    return boundaries


def is_quote_question(question: str) -> bool:
    return bool(re.match(r'^\(?[ivxIVX]+[.)]\s', question.strip()))


def find_answer_start_offset(answer_lines, boundary_idx, question):
    if not is_quote_question(question):
        return 1
    q_words = set(normalize(question).split()) - STOPWORDS
    offset = 1
    for k in range(1, min(8, len(answer_lines) - boundary_idx - 1) + 1):
        idx = boundary_idx + k
        if idx >= len(answer_lines):
            break
        line = answer_lines[idx].strip()
        if not line:
            offset = k + 1
            continue
        line_words = set(normalize(line).split()) - STOPWORDS
        if not line_words:
            offset = k + 1
            continue
        if len(line_words & q_words) / max(len(line_words), 1) >= 0.55:
            offset = k + 1
        else:
            break
    return offset


def slice_raw_answers(answer_lines, boundaries):
    qa_pairs = []
    for i, b in enumerate(boundaries):
        skip    = find_answer_start_offset(answer_lines, b["line_index"], b["question"])
        a_start = b["line_index"] + skip
        a_end   = boundaries[i + 1]["line_index"] if i + 1 < len(boundaries) else len(answer_lines)
        raw     = [answer_lines[j] for j in range(a_start, a_end) if answer_lines[j].strip()]
        qa_pairs.append({
            "question": b["question"],
            "answer":   " ".join(raw).strip()
        })
    return qa_pairs


# =========================================================
# COMPLETE PIPELINE
# =========================================================

def process_pdf(file_input, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    if isinstance(file_input, (str, Path)):
        file_bytes = Path(file_input).read_bytes()
        file_name  = Path(file_input).name
    else:
        file_bytes = file_input.read()
        file_name  = getattr(file_input, "name", "document.pdf")

    # Step 1: Preprocess
    log("Preprocessing PDF...")
    processed = preprocess_pdf(file_bytes)

    # Step 2: OCR
    pages = run_ocr(processed, file_name, status_callback)

    # Step 3: Build OCR JSON
    log("Building OCR JSON...")
    ocr_json = build_ocr_json(pages)
    log(f"Total pages: {ocr_json['total_pages']}")

    # Step 4: Find question paper page
    qp_idx = find_question_paper_page(pages)
    log(f"Question paper detected on page: {qp_idx + 1 if qp_idx >= 0 else 'not found'}")

    if qp_idx >= 0:
        official_questions = extract_official_questions(pages[qp_idx]["raw_text"])
        log(f"Official questions extracted: {len(official_questions)}")
    else:
        official_questions = []
        log("No question paper page found — will use answer markers only")

    # Step 5: Get answer pages (everything after question paper)
    answer_start = qp_idx + 1 if qp_idx >= 0 else 0
    answer_pages = pages[answer_start:]

    # Step 6: Flatten answer page lines
    answer_lines = []
    for page in answer_pages:
        for line in page["raw_text"].split("\n"):
            answer_lines.append(line)

    # Step 7A: Try उत्तर/Ans marker based extraction first
    log("Trying उत्तर/Ans marker extraction...")
    raw_answers = extract_answers_by_uttar(answer_lines, status_callback)

    if raw_answers:
        log(f"Found {len(raw_answers)} answer blocks via markers")
        qa_pairs = []
        for idx, answer in enumerate(raw_answers):
            if idx < len(official_questions):
                question = official_questions[idx]
            else:
                question = f"Q{idx + 1}"
            qa_pairs.append({
                "question": question,
                "answer":   answer
            })
        log(f"Done — {len(qa_pairs)} Q-A pairs")
        return ocr_json, qa_pairs

    # Step 7B: Fall back to TOPIC/DATE split format (English answer sheets)
    log("No उत्तर markers found — trying TOPIC/DATE split format...")
    split_idx = find_answer_start_page(pages)

    if split_idx >= 0 and official_questions:
        log(f"Split format: answer sheets start at page {split_idx + 1}")
        split_lines = []
        for page in pages[split_idx:]:
            for line in page["raw_text"].split("\n"):
                split_lines.append(line)

        boundaries = find_question_boundaries_in_answers(split_lines, official_questions)
        log(f"Matched {len(boundaries)} boundaries")

        if boundaries:
            qa_pairs = slice_raw_answers(split_lines, boundaries)
            log(f"Done — {len(qa_pairs)} Q-A pairs")
            return ocr_json, qa_pairs

    # Step 7C: Last resort — return OCR text as single block with official questions
    log("WARNING: Could not detect Q-A structure. Returning raw OCR blocks.")
    full_text = " ".join(answer_lines)
    qa_pairs = [{"question": q, "answer": ""} for q in official_questions] if official_questions \
               else [{"question": "Full Document", "answer": full_text}]

    log(f"Done — {len(qa_pairs)} Q-A pairs")
    return ocr_json, qa_pairs
