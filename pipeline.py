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
            "raw_text": page.get("markdown", "")
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
# REFERENCE BOOK OCR — page by page, base64 inline
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

    api_key = get_api_key("MISTRAL_API_KEY")

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
# DETECT DOCUMENT TYPE
#
# Type A — question paper separate from answer sheets:
#   Pages 1-N  : printed question paper
#   Pages N+1+ : handwritten answer sheets with TOPIC/DATE
#
# Type B — integrated format (IGNOU, university assignments):
#   All answers in one continuous document
#   Each answer starts with Q.N / Q-N / Ans / number label
#   No TOPIC/DATE markers
# =========================================================

def detect_document_type(pages: list) -> str:
    """Returns 'split' or 'integrated'"""
    for page in pages:
        if re.search(r'\bTOPIC\b', page["raw_text"]) and \
           re.search(r'\bDATE\b', page["raw_text"]):
            return "split"
    return "integrated"


# =========================================================
# SPLIT TYPE — question paper separate from answer sheets
# =========================================================

def find_answer_start_page(pages: list) -> int:
    for i, page in enumerate(pages):
        text = page["raw_text"]
        if re.search(r'\bTOPIC\b', text) and re.search(r'\bDATE\b', text):
            return i
    return len(pages) // 2


def extract_questions_from_pages(pages: list, end_idx: int) -> list:
    LABEL_PATTERNS = [
        r'^\(?[ivxIVX]+[\.\)]\s+\S',
        r'^\d+[\.\)]\s+\S',
        r'^Q\.?\s*\d+[\.\):\s]',
    ]
    questions = []
    for page in pages[:end_idx]:
        for line in page["raw_text"].split("\n"):
            stripped = line.strip()
            if len(stripped) < 20:
                continue
            for pat in LABEL_PATTERNS:
                if re.match(pat, stripped):
                    questions.append(stripped)
                    break

    seen = set()
    unique = []
    for q in questions:
        key = re.sub(r'\s+', ' ', q.lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


# =========================================================
# INTEGRATED TYPE — Q&A in same document
#
# Strategy:
#   Scan all lines for lines that START with a question label
#   The label line itself becomes the question boundary
#   Everything between two boundaries = answer of the first
#
# Question label patterns found in Hindi IGNOU assignments:
#   Q.1, Q.2, Q. 3, Q-4, Q. 9-      -> Q then dot/dash/space then digit
#   9-7., 9-8, 86., 95.              -> digit(s)-digit(s) or digit(s).
#   Ans-, A B:-, AB-3, A.15-         -> Answer markers (these follow questions)
# =========================================================

# Patterns that mark the START of a new question+answer block
Q_BOUNDARY_PATTERNS = [
    r'^Q[\.\-\s]+\d',              # Q.1  Q-4  Q. 3  Q. 9-
    r'^\d{1,2}[\.\-]\d{1,2}[\.\-\s]',  # 9-7.  9-8  86.
    r'^\d{2,3}\.\s*Q[\s\.]',       # 95. Q  86. Q
    r'^\d{1,2}\.\s+(?!\d)',        # 86. संघ  (number. then non-digit content)
]

# Patterns that mark an ANSWER block start (used to split Q from A within a block)
ANS_MARKER_PATTERNS = [
    r'^(?:Ans|AB|A\.?\d*|A\s*B)\s*[\.\-\:\→]',   # Ans- AB:- A.15- A B:-
    r'^(?:उत्तर|जवाब)\s*[\.\-\:]',                # Hindi answer markers
]

def is_question_boundary(line: str) -> bool:
    line = line.strip()
    line = re.sub(r'^\s*[-*#]+\s*', '', line)  # strip markdown
    for pat in Q_BOUNDARY_PATTERNS:
        if re.match(pat, line, re.IGNORECASE):
            return True
    return False


def is_ans_marker(line: str) -> bool:
    line = line.strip()
    for pat in ANS_MARKER_PATTERNS:
        if re.match(pat, line, re.IGNORECASE):
            return True
    return False


def extract_question_label(line: str) -> str:
    """Extract just the Q number label from a boundary line."""
    line = line.strip()
    # Q.1, Q-4, Q. 3 etc
    m = re.match(r'^(Q[\.\-\s]*\d+[\.\-]?\d*)', line, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 9-7, 9-8, 86 etc
    m = re.match(r'^(\d+[\.\-]\d+|\d{2,3})', line)
    if m:
        return m.group(1).strip()
    return line[:30]


def process_integrated(pages: list, question_pages: list, status_callback=None) -> list:
    """
    For integrated format where questions and answers are in the same document.
    Skips the question-paper pages (they contain only the printed questions).
    Scans answer pages for Q boundary lines, slices text between them.
    Returns list of {question, answer}
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    # Skip question paper pages — only process answer pages
    answer_page_nums = set(range(len(pages))) - set(question_pages)
    answer_pages = [pages[i] for i in sorted(answer_page_nums)]

    # Flatten all answer lines with page tracking
    all_lines = []
    for page in answer_pages:
        for line in page["raw_text"].split("\n"):
            all_lines.append(line)

    log(f"Scanning {len(all_lines)} lines for question boundaries...")

    # Find boundary positions
    boundaries = []
    for i, line in enumerate(all_lines):
        if is_question_boundary(line.strip()):
            label = extract_question_label(line.strip())
            boundaries.append({"line_index": i, "label": label, "raw_line": line.strip()})

    log(f"Found {len(boundaries)} question boundaries: {[b['label'] for b in boundaries]}")

    if not boundaries:
        raise Exception(
            "No question boundaries found in answer pages.\n"
            f"First 10 answer lines:\n" + "\n".join(all_lines[:10])
        )

    # Slice text between boundaries
    qa_pairs = []

    for i, b in enumerate(boundaries):
        start = b["line_index"]
        end   = boundaries[i + 1]["line_index"] if i + 1 < len(boundaries) else len(all_lines)

        block_lines = all_lines[start:end]

        # Within the block, split into question part and answer part
        # Question = lines before the first Ans marker
        # Answer   = lines from Ans marker onwards
        q_lines   = []
        ans_lines = []
        ans_started = False

        for line in block_lines:
            stripped = line.strip()
            if not ans_started and is_ans_marker(stripped):
                ans_started = True
            if ans_started:
                ans_lines.append(stripped)
            else:
                q_lines.append(stripped)

        # If no explicit Ans marker found, use the boundary line as question
        # and everything after it as answer
        if not ans_started:
            q_lines   = [block_lines[0].strip()] if block_lines else []
            ans_lines = [l.strip() for l in block_lines[1:]]

        question_text = " ".join(q for q in q_lines if q).strip()
        answer_text   = " ".join(a for a in ans_lines if a).strip()

        qa_pairs.append({
            "question": question_text,
            "answer":   answer_text
        })

    return qa_pairs


# =========================================================
# SPLIT TYPE HELPERS (unchanged from before)
# =========================================================

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


def is_noise_line(line: str) -> bool:
    patterns = [
        r'^\s*TOPIC\s*$', r'^\s*DATE\s*$',
        r'TOPIC\s*_+', r'DATE\s*_+',
        r'^\s*\d+\s*$', r'^\s*[-_]+\s*$',
    ]
    for p in patterns:
        if re.search(p, line, re.IGNORECASE):
            return True
    return False


def is_quote_question(question: str) -> bool:
    return bool(re.match(r'^\(?[ivxIVX]+[.)]\s', question.strip()))


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
                combined_words = normalize(combined_clean).split()
                q_words = normalize(strip_leading_label(q)).split()
                body_c = " ".join(combined_words[1:]) if len(combined_words) > 1 else ""
                body_q = " ".join(q_words[1:]) if len(q_words) > 1 else ""
                s3 = similarity(body_c, body_q) if body_c and body_q else 0
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


def find_answer_start_offset(answer_lines, boundary_idx, question):
    if not is_quote_question(question):
        return 1
    q_words = set(normalize(question).split()) - STOPWORDS
    offset = 1
    max_skip = min(8, len(answer_lines) - boundary_idx - 1)
    for k in range(1, max_skip + 1):
        idx = boundary_idx + k
        if idx >= len(answer_lines):
            break
        line = answer_lines[idx].strip()
        if not line or is_noise_line(line):
            offset = k + 1
            continue
        line_words = set(normalize(line).split()) - STOPWORDS
        if not line_words:
            offset = k + 1
            continue
        overlap = len(line_words & q_words) / max(len(line_words), 1)
        if overlap >= 0.55:
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
        raw_lines = [answer_lines[j] for j in range(a_start, a_end)
                     if not is_noise_line(answer_lines[j])]
        qa_pairs.append({
            "question": b["question"],
            "answer": " ".join(raw_lines).strip()
        })
    return qa_pairs


# =========================================================
# DETECT QUESTION PAPER PAGES
# Page 1 is usually cover page, page 2 is question paper
# Answer pages start where Q.N labels begin
# =========================================================

def find_question_paper_pages(pages: list) -> list:
    """
    Returns list of page indices (0-based) that are question paper pages.
    These pages contain only printed questions, no student answers.
    Heuristic: question paper pages have question lists but no Ans/AB markers.
    """
    q_paper_pages = []
    for i, page in enumerate(pages):
        text = page["raw_text"]
        has_printed_questions = bool(
            re.search(r'^\s*\d+[\.\)]\s+.{20,}', text, re.MULTILINE)
        )
        has_student_answers = bool(
            re.search(r'(?:^|\n)\s*(?:Ans|AB|A\.?\d*)\s*[\.\-\:\→]', text, re.IGNORECASE)
        ) or bool(
            re.search(r'(?:^|\n)\s*Q[\.\-\s]*\d+', text, re.IGNORECASE)
        )
        # If page has printed question list but student hasn't written on it
        if has_printed_questions and not has_student_answers:
            q_paper_pages.append(i)

    # Always include page index 0 (cover) and 1 (question paper) as non-answer
    for idx in [0, 1]:
        if idx not in q_paper_pages and idx < len(pages):
            q_paper_pages.append(idx)

    return sorted(set(q_paper_pages))


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

    log("Preprocessing PDF...")
    processed = preprocess_pdf(file_bytes)

    pages = run_ocr(processed, file_name, status_callback)

    log("Building OCR JSON...")
    ocr_json = build_ocr_json(pages)
    log(f"Total pages: {ocr_json['total_pages']}")

    # Detect document type
    doc_type = detect_document_type(pages)
    log(f"Document type detected: {doc_type}")

    if doc_type == "split":
        # Original flow — question paper separate from answer sheets
        log("Split format: finding answer start page...")
        answer_start = find_answer_start_page(pages)
        log(f"Question pages: 1-{answer_start} | Answer pages: {answer_start+1}-{len(pages)}")

        questions = extract_questions_from_pages(pages, answer_start)
        log(f"Found {len(questions)} questions")

        if not questions:
            raise Exception("No questions found in question pages.\n"
                            f"Preview:\n{pages[0]['raw_text'][:400]}")

        answer_lines = []
        for page in pages[answer_start:]:
            for line in page["raw_text"].split("\n"):
                answer_lines.append(line)

        boundaries = find_question_boundaries_in_answers(answer_lines, questions)
        log(f"Matched {len(boundaries)} boundaries")

        for q in questions:
            if q not in {b["question"] for b in boundaries}:
                log(f"WARNING: No boundary found for: {q[:80]}")

        if not boundaries:
            raise Exception("Could not match any questions in answer pages.\n"
                            f"Questions: {questions}")

        qa_pairs = slice_raw_answers(answer_lines, boundaries)

    else:
        # Integrated format — Q&A in same document (IGNOU style)
        log("Integrated format: scanning for Q.N boundaries...")
        question_paper_pages = find_question_paper_pages(pages)
        log(f"Skipping pages (question paper/cover): {[p+1 for p in question_paper_pages]}")

        qa_pairs = process_integrated(pages, question_paper_pages, status_callback)

    log(f"Done — {len(qa_pairs)} Q-A pairs")
    return ocr_json, qa_pairs