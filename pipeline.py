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
# REFERENCE BOOK OCR
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
# UNIVERSAL Q&A EXTRACTOR
# No LLM. No structure assumptions.
#
# Strategy:
#   1. Flatten all pages into one line list
#   2. Score every line: does it look like a question/section START?
#   3. Boundaries = lines whose score exceeds threshold
#   4. If zero boundaries found, try progressively looser passes
#   5. If still nothing, return the whole document as one block
#
# Boundary patterns (in order of confidence):
#   HIGH   Q.1 / Q-1 / Q. 1 / Question 1 / Ques.1
#          1. text  /  1) text  (digit-dot/paren then text)
#          (a) text / (i) text / (iv) text   — lettered/roman sub-questions
#          a) text  / i) text                — without parens
#   MEDIUM ## Heading  /  **Bold label:**    — markdown structure
#          WORD:  (all-caps word followed by colon, ≥4 chars)
#   LOW    Any line that is SHORT (≤ 120 chars) and ends with ?
#
# Answer text is everything between two boundaries — raw, unmodified.
# =========================================================

# ── boundary detection patterns ───────────────────────────────────────────────

# Each entry: (pattern, score)
# Line is a boundary if total score ≥ THRESHOLD
BOUNDARY_PATTERNS = [
    # Q.1 / Q-1 / Q 1 / Q.1. / Ques.1 / Question 1
    (re.compile(r'^\s*(?:Q(?:ues(?:tion)?)?[\.\-\s]*\d+[\.\):\-]?\s*)', re.I), 10),

    # 1. text or 1) text — digit then dot/paren then space then non-digit content
    (re.compile(r'^\s*\d{1,2}[\.\)]\s+[^\d\s]'), 8),

    # (a) / (b) / (i) / (ii) / (iv) etc
    (re.compile(r'^\s*\([a-zA-Z]{1,3}\)\s+\S'), 7),

    # a) / b) / i) / ii) without parens
    (re.compile(r'^\s*[a-zA-Z]{1,3}\)\s+[A-Z\u0900-\u097F]'), 6),

    # Ans / Ans. / Ans- / Answer: — answer markers (lower confidence boundary)
    (re.compile(r'^\s*Ans(?:wer)?[\.\-\:\s]', re.I), 5),

    # ## Heading or ### Heading (markdown)
    (re.compile(r'^\s*#{1,4}\s+\S'), 5),

    # **Bold text** at start of line
    (re.compile(r'^\s*\*{1,2}[^*]{3,}\*{1,2}\s*[\:\-]?'), 4),

    # ALL-CAPS WORD: (like "TOPIC:", "NOTE:", "SECTION:")
    (re.compile(r'^\s*[A-Z]{4,}[\s\:]+'), 4),

    # Short line ending with ?
    (re.compile(r'^.{10,120}\?\s*$'), 3),
]

# Noise lines — never a boundary
NOISE_RE = re.compile(
    r'^\s*$'                           # blank
    r'|!\[.*?\]\(.*?\)'                # markdown images
    r'|https?://\S+'                   # URLs
    r'|^\s*[-_=]{3,}\s*$'             # horizontal rules
    r'|^\s*\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\s*$'  # bare dates
    r'|^\s*Page\s+\d+\s*$'            # page numbers
    , re.I
)

# Lines that are clearly junk-content pages (skip whole page if majority match)
JUNK_PAGE_SIGNALS = re.compile(
    r'Acknowledgment|acknowledgement|www\.|\.asp|\.html|'
    r'KNOW YOUR ADMISSION|REGISTRATION DETAILS|Indira Gandhi National Open University'
    r'|enrollment\s*no|enrolment\s*no|programme\s*code|study\s*centre',
    re.I
)


def _score_line(line: str) -> int:
    """Return boundary score for a line. 0 = not a boundary."""
    if NOISE_RE.search(line):
        return 0
    stripped = line.strip()
    if len(stripped) < 3:
        return 0
    total = 0
    for pat, score in BOUNDARY_PATTERNS:
        if pat.search(stripped):
            total += score
    return total


def _is_junk_page(page_text: str) -> bool:
    """True if page is clearly admin/cover/web content with no answers."""
    lines = [l for l in page_text.split('\n') if l.strip()]
    if not lines:
        return True
    junk_hits = sum(1 for l in lines if JUNK_PAGE_SIGNALS.search(l))
    return junk_hits / max(len(lines), 1) > 0.3


def flatten_pages(pages: list, log=None) -> list:
    """
    Returns list of {"line": str, "page_number": int}.
    Skips junk pages and noise lines.
    """
    result = []
    skipped_pages = []
    for p in pages:
        if _is_junk_page(p["raw_text"]):
            skipped_pages.append(p["page_number"])
            continue
        for line in p["raw_text"].split("\n"):
            if not NOISE_RE.search(line) and line.strip():
                result.append({"line": line, "page_number": p["page_number"]})

    if log and skipped_pages:
        log(f"Skipped junk/admin pages: {skipped_pages}")
    return result


def find_boundaries(flat_lines: list, threshold: int) -> list:
    """Return indices into flat_lines where a new Q/section starts."""
    boundaries = []
    for i, row in enumerate(flat_lines):
        if _score_line(row["line"]) >= threshold:
            boundaries.append(i)
    return boundaries


def slice_qa(flat_lines: list, boundaries: list) -> list:
    """
    Slice raw lines between boundaries into Q&A pairs.
    The line AT the boundary = question label / first line of question.
    Everything after until next boundary = answer body.
    Raw text — zero modification.
    """
    pairs = []
    n = len(flat_lines)
    for i, b_idx in enumerate(boundaries):
        next_idx = boundaries[i + 1] if i + 1 < len(boundaries) else n

        block = [flat_lines[j]["line"] for j in range(b_idx, next_idx)]
        if not block:
            continue

        # First line = question header
        # Rest = answer (may include sub-labels, all kept verbatim)
        question = block[0].strip()
        answer   = "\n".join(l for l in block[1:] if l.strip()).strip()

        pairs.append({"question": question, "answer": answer})
    return pairs


# =========================================================
# MAIN EXTRACTION — structure-agnostic, no LLM
# =========================================================

def extract_qa(pages: list, status_callback=None) -> list:
    """
    Multi-pass boundary detection.
    Pass 1: high confidence (score ≥ 8)  — numbered questions
    Pass 2: medium confidence (score ≥ 5) — lettered sub-qs, headings
    Pass 3: low confidence (score ≥ 3)   — anything structural
    Pass 4: fallback — split on blank-line paragraphs
    Pass 5: whole document as one block
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    flat_lines = flatten_pages(pages, log)
    log(f"Usable lines after filtering: {len(flat_lines)}")

    if not flat_lines:
        log("No usable text found in document.")
        return []

    # ── Pass 1: strict ───────────────────────────────────────────────────────
    for threshold, label in [(8, "strict"), (5, "medium"), (3, "loose")]:
        boundaries = find_boundaries(flat_lines, threshold)
        log(f"Pass ({label}, threshold={threshold}): {len(boundaries)} boundaries")
        if len(boundaries) >= 1:
            pairs = slice_qa(flat_lines, boundaries)
            log(f"Extracted {len(pairs)} Q-A pair(s) [{label}]")
            return pairs

    # ── Pass 4: paragraph split (blank-line separated blocks) ────────────────
    log("Pass (paragraph): splitting on blank lines...")
    pairs = []
    current_block = []
    for row in flat_lines:
        if row["line"].strip():
            current_block.append(row["line"])
        else:
            if current_block:
                pairs.append({
                    "question": current_block[0].strip(),
                    "answer":   "\n".join(current_block[1:]).strip()
                })
                current_block = []
    if current_block:
        pairs.append({
            "question": current_block[0].strip(),
            "answer":   "\n".join(current_block[1:]).strip()
        })

    if pairs:
        log(f"Extracted {len(pairs)} paragraph block(s)")
        return pairs

    # ── Pass 5: entire document as one block ─────────────────────────────────
    log("Pass (fallback): returning entire document as single block")
    full_text = "\n".join(row["line"] for row in flat_lines)
    return [{"question": "Full document", "answer": full_text}]


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

    log("Extracting Q&A (no LLM, structure-agnostic)...")
    qa_pairs = extract_qa(pages, status_callback)

    log(f"Pipeline complete — {len(qa_pairs)} Q-A pair(s)")
    return ocr_json, qa_pairs
