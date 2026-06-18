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
    Splits questions containing sub-parts like a) b) c) into separate
    question entries, since each sub-part typically has its own answer.
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

    # Split questions containing lettered sub-parts: a) ... b) ... c) ...
    final_questions = []
    SUBPART_RE = re.compile(r'(?:^|\s)([a-z])\)\s')

    for q in questions:
        matches = list(SUBPART_RE.finditer(q))

        if len(matches) >= 2:
            # Has 2+ sub-parts — split into separate questions
            # Preamble = text before the first sub-part marker
            preamble = q[:matches[0].start()].strip()

            for idx, m in enumerate(matches):
                start = m.start(1)  # position of the letter itself
                end   = matches[idx + 1].start() if idx + 1 < len(matches) else len(q)
                part_text = q[start:end].strip()
                # part_text looks like "a) Renaissance" — keep preamble for context
                full_q = f"{preamble} {part_text}".strip() if preamble else part_text
                final_questions.append(full_q)
        else:
            final_questions.append(q)

    return final_questions


NOISE_RE = re.compile(
    r'(?:Teacher\'?s?\s*Signature'
    r'|Tancher\'?s?\s*Signature'
    r'|Facebook\'?s?\s*Signature'
    r'|PAGE\s*NO'
    r'|^\s*DATE\b'
    r'|Neel?\s*Kamal'
    r'|Neal?\s*Kamal'
    r'|Need?\s*Komal'
    r'|Nod\s*Komal'
    r'|TAKMA\s*SINAN'
    r'|^\s*\d{1,3}\s*$)',
    re.IGNORECASE
)


def is_noise(line: str) -> bool:
    return bool(NOISE_RE.search(line))


# =========================================================
# FIND QUESTION BOUNDARIES IN ANSWER PAGES — similarity based
# Works for Hindi, English, any language
# Matches student-written question restatements against the
# official question paper text using word-overlap similarity
# =========================================================

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def similarity(a: str, b: str) -> float:
    wa = set(normalize(a).split())
    wb = set(normalize(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def strip_leading_label(text: str) -> str:
    """Strip leading numbering like '1.', 'Q1.', 'प्र.2', '20.', 'a)' etc."""
    text = text.strip()
    text = re.sub(r'^(?:Ans(?:wer)?[.\s]+)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^(?:उत्तर)\s*[\-\:\s]*', '', text)
    text = re.sub(r'^(?:प्र|प्रो|प्रश्न)[\.\s]*\d*[\.\s]*', '', text)
    text = re.sub(r'^[१-९०][०-९]*[\.\-\s]*', '', text)
    text = re.sub(r'^(?:Q\.?\s*)?\d+[.)]\s*', '', text)
    text = re.sub(r'^[a-z]\)\s*', '', text, flags=re.IGNORECASE)
    return text.strip()


def find_question_boundaries_by_similarity(
    answer_lines: list,
    questions: list,
    similarity_threshold: float = 0.30,
    window: int = 4
) -> list:
    """
    Scans answer lines for restated questions matching official questions.
    Uses sliding window to join multi-line question restatements.

    Correctness guarantees:
    1. Tracks how many lines (`span`) the matched question text occupies,
       so the answer slice can start AFTER the full question text.
    2. Enforces that boundaries appear in the SAME ORDER as the official
       questions list. If a question's best-scoring candidate would break
       order (e.g. a false-positive shares vocabulary with an earlier
       question), the NEXT best-scoring candidate for that same question
       is tried, and so on, rather than dropping the question entirely.
    """
    candidates_by_question = {}   # question -> list of candidates, sorted by score desc

    for i in range(len(answer_lines)):
        line_i = answer_lines[i].strip()
        if len(line_i) < 8:
            continue

        for w in range(1, window + 1):
            if i + w > len(answer_lines):
                break

            combined = " ".join(
                answer_lines[i + k].strip()
                for k in range(w) if answer_lines[i + k].strip()
            )
            if len(combined) < 10:
                continue

            combined_clean = strip_leading_label(combined)

            for q in questions:
                q_clean = strip_leading_label(q)
                s1 = similarity(combined, q)
                s2 = similarity(combined_clean, q_clean)
                score = max(s1, s2)

                if score >= similarity_threshold:
                    candidates_by_question.setdefault(q, []).append({
                        "question":   q,
                        "line_index": i,
                        "span":       w,
                        "score":      score
                    })

    # Sort each question's candidates by score, descending
    for q in candidates_by_question:
        candidates_by_question[q].sort(key=lambda c: -c["score"])

    # Walk questions in official order. For each, try candidates from
    # highest score downward, accepting the first one that comes after
    # the previously accepted boundary's line_index.
    final = []
    last_line_index = -1

    for q in questions:
        cands = candidates_by_question.get(q, [])
        chosen = None
        for c in cands:
            if c["line_index"] > last_line_index:
                chosen = c
                break
        if chosen is not None:
            final.append(chosen)
            last_line_index = chosen["line_index"]

    return final


def slice_raw_answers_by_boundaries(answer_lines: list, boundaries: list) -> list:
    """
    For each boundary, answer = raw lines starting AFTER the full matched
    question span (boundary["span"] lines), up to the next boundary.
    Pure text slicing, zero LLM.
    """
    qa_pairs = []
    for i, b in enumerate(boundaries):
        span    = b.get("span", 1)
        a_start = b["line_index"] + span
        a_end   = boundaries[i + 1]["line_index"] if i + 1 < len(boundaries) else len(answer_lines)

        raw = [
            answer_lines[j] for j in range(a_start, a_end)
            if answer_lines[j].strip() and not is_noise(answer_lines[j])
        ]

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
        log("No question paper page found")

    if not official_questions:
        raise Exception(
            "Could not extract official questions from the question paper page.\n"
            f"Preview of detected page:\n{pages[max(qp_idx,0)]['raw_text'][:500]}"
        )

    # Step 5: Get answer pages (everything after question paper)
    answer_start = qp_idx + 1 if qp_idx >= 0 else 0
    answer_pages = pages[answer_start:]

    # Step 6: Flatten answer page lines
    answer_lines = []
    for page in answer_pages:
        for line in page["raw_text"].split("\n"):
            if not is_noise(line):
                answer_lines.append(line)

    log(f"Flattened {len(answer_lines)} answer lines")

    # Step 7: Similarity-based matching — works for any language/format
    log("Matching questions via similarity (works for Hindi/English/any format)...")
    boundaries = find_question_boundaries_by_similarity(answer_lines, official_questions)
    log(f"Matched {len(boundaries)} of {len(official_questions)} questions")

    matched_qs = {b["question"] for b in boundaries}
    for q in official_questions:
        if q not in matched_qs:
            log(f"WARNING: No match found for: {q[:60]}")

    if not boundaries:
        raise Exception(
            "Could not match any questions in answer pages.\n"
            f"Official questions: {official_questions}\n"
            f"First 10 answer lines: {answer_lines[:10]}"
        )

    # Step 8: Slice raw answers — zero LLM, pure text slicing
    log("Slicing raw answers...")
    qa_pairs = slice_raw_answers_by_boundaries(answer_lines, boundaries)

    log(f"Done — {len(qa_pairs)} Q-A pairs")
    return ocr_json, qa_pairs
