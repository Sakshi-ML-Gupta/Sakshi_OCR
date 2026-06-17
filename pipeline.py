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
# UNIVERSAL Q&A EXTRACTOR  v3
# No LLM. No fixed structure assumptions.
#
# KEY INSIGHT (from real samples across two very different
# IGNOU course formats):
#
#   The thing that reliably marks a NEW question is the
#   QUESTION LABEL itself — Q.1, Q-2, 1., (a), (i), etc.
#   The "Ans/AB/A.15-" marker is NOT reliable: OCR reads the
#   same handwritten word as "Ans-", "A B:-", "AB-3", "A.15-",
#   "AIS-7", "A13-3" etc. depending on handwriting noise.
#
#   So: a new Q&A block starts at a question-label line.
#   Everything until the next question-label line — INCLUDING
#   any "Ans-" noise word — is the answer (we just don't try
#   to parse the Ans-marker separately; we keep it as-is,
#   since stripping it raw-text-style without modification
#   is safest).
#
#   We ALSO strip page furniture that repeats on every page
#   and pollutes the text: "Experiment Name :", "Page No. N",
#   "Teacher's Signature & Date :", lone page numbers injected
#   mid-paragraph by the OCR (e.g. "8 9", "14 15").
# =========================================================

# ---- noise that should be removed before boundary detection ----
NOISE_LINE_RE = re.compile(
    r'^\s*$'                                  # blank
    r'|!\[.*?\]\(.*?\)'                       # markdown images
    r'|https?://\S+'                          # URLs
    r'|^\s*[-_=]{3,}\s*$'                     # horizontal rules
    r'|^\s*Experiment\s*Name\s*:?\s*$'        # repeating page furniture
    r'|^\s*Page\s*No\.?\s*\d*\s*$'
    r"|^\s*Teacher'?s\s*Signature\s*&?\s*Date\s*:?.*$"
    r'|^\s*\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}[,\s]*\d{0,2}:?\d{0,2}\s*$'  # bare dates/timestamps
    , re.I
)

# Lines that are entirely admin/cover/web junk — used for whole-page skip
JUNK_PAGE_SIGNALS = re.compile(
    r'Acknowledgment|acknowledgement|www\.|\.asp|\.html|'
    r'KNOW YOUR ADMISSION|REGISTRATION DETAILS|Indira Gandhi National Open University|'
    r'enrollment\s*no|enrolment\s*no|programme\s*code|study\s*centre|'
    r'Student Identity Card|IGNOU - Student|RC Code|Father\'?s Name'
    , re.I
)

# Mid-text injected page numbers like "8 9" or "14 15" standing alone
# (printed page numbers OCR'd between handwritten lines)
INLINE_PAGENUM_RE = re.compile(r'^\s*\d{1,3}(?:\s+\d{1,3}){0,2}\s*$')


# ---- question-label boundary patterns, highest confidence first ----
# Each: (regex, score). A line is a candidate boundary if total score
# across all matching patterns >= threshold used in that pass.
QUESTION_LABEL_PATTERNS = [
    # Q.1 / Q-1 / Q 1 / Q.1. / Ques.1 / Question 1 / Q.1- (with optional trailing dash)
    (re.compile(r'^\s*Q(?:ues(?:tion)?)?[\.\-\s]*\d+\s*[\.\)\-:]?', re.I), 10),

    # 1. text / 1) text  — digit + dot/paren + space + non-digit start
    (re.compile(r'^\s*\d{1,3}[\.\)]\s+[^\d\s]'), 8),

    # (a) (b) (i) (ii) (iv) — lettered/roman sub-question, parens
    (re.compile(r'^\s*\([a-zA-Z]{1,4}\)\s*\S'), 8),

    # a) b) i) ii) — without parens, followed by capital or Devanagari
    (re.compile(r'^\s*[a-zA-Z]{1,3}\)\s*[A-Z\u0900-\u097F]'), 6),

    # 9-7. / 9-8 / 86. / 95. — IGNOU-style cross-numbered question refs
    (re.compile(r'^\s*\d{1,2}[\.\-]\d{1,2}\b'), 6),
]

# Words that strongly indicate an answer-label noise token sitting at
# the START of a line right after a real question boundary — these are
# NOT boundaries themselves (too unreliable across OCR variants), but
# recognising them helps us avoid double-counting them as new questions.
ANSWER_NOISE_RE = re.compile(
    r'^\s*(?:Ans|A\s*B|A\d{0,2}|AIS|AR)[\.\-\:\s\d]{0,6}[\-\:→]?\s*$', re.I
)


def _score_line(line: str) -> int:
    stripped = line.strip()
    if len(stripped) < 2:
        return 0
    total = 0
    for pat, score in QUESTION_LABEL_PATTERNS:
        if pat.match(stripped):
            total += score
    return total


def _is_junk_page(page_text: str) -> bool:
    lines = [l for l in page_text.split('\n') if l.strip()]
    if not lines:
        return True
    junk_hits = sum(1 for l in lines if JUNK_PAGE_SIGNALS.search(l))
    return junk_hits / max(len(lines), 1) > 0.3


# A printed question paper looks like: several lines that ALL score as
# question-label boundaries, with very little non-label prose between
# them (each "question" is 1-3 lines, no real answer content).
# A genuine answer page has long stretches of prose between boundaries.
QUESTION_PAPER_HINTS = re.compile(
    r'Answer the following|Max\.?\s*Marks|Course Code|Assignment\b.*Session|'
    r'सभी प्रश्न अनिवार्य|कुल अंक|पाठ्यक्रम कोड|सत्रीय कार्य कोड',
    re.I
)


def _is_question_paper_page(page_text: str) -> bool:
    """
    True if this page is the printed question paper (not a student's
    answer). Strong signal: explicit question-paper phrasing. Weaker
    heuristic: MULTIPLE question labels on the page where NONE of them
    is followed by a real prose answer (every gap between labels is
    short — just the prompt text + mark allocation, no actual answer
    content). A single short answer page must NOT trigger this.
    """
    if QUESTION_PAPER_HINTS.search(page_text):
        return True

    lines = [l.strip() for l in page_text.split('\n') if l.strip()]
    if len(lines) < 4:
        return False

    label_idxs = [i for i, l in enumerate(lines) if _score_line(l) >= 6]
    if len(label_idxs) < 3:
        # Too few labels to confidently call this a question list;
        # could just be a normal answer page with 1-2 sub-questions.
        return False

    # Check the gap after each label (until the next label or end).
    # If EVERY gap is short (<= 2 lines / <=25 words), this page has
    # no real answers — it's a question list.
    max_gap_words = 0
    for k, idx in enumerate(label_idxs):
        next_idx = label_idxs[k + 1] if k + 1 < len(label_idxs) else len(lines)
        gap_lines = lines[idx:next_idx]
        gap_words = sum(len(l.split()) for l in gap_lines)
        max_gap_words = max(max_gap_words, gap_words)

    # If the longest gap is still short, no prose answer exists anywhere
    # on this page -> it's a printed question list.
    return max_gap_words < 30


def flatten_pages(pages: list, log=None) -> list:
    """
    Returns list of {"line": str, "page_number": int}.
    Skips junk pages, printed question-paper pages, and noise/furniture
    lines. Bare inline page numbers are dropped too (OCR artefacts).
    """
    result = []
    skipped_junk = []
    skipped_qp   = []
    for p in pages:
        if _is_junk_page(p["raw_text"]):
            skipped_junk.append(p["page_number"])
            continue
        if _is_question_paper_page(p["raw_text"]):
            skipped_qp.append(p["page_number"])
            continue
        for line in p["raw_text"].split("\n"):
            if NOISE_LINE_RE.search(line):
                continue
            if INLINE_PAGENUM_RE.match(line.strip()):
                continue
            if line.strip():
                result.append({"line": line, "page_number": p["page_number"]})

    if log:
        if skipped_junk:
            log(f"Skipped junk/admin pages: {skipped_junk}")
        if skipped_qp:
            log(f"Skipped printed question-paper pages: {skipped_qp}")
    return result


def find_boundaries(flat_lines: list, threshold: int) -> list:
    """Indices into flat_lines that look like the start of a new question."""
    boundaries = []
    for i, row in enumerate(flat_lines):
        if _score_line(row["line"]) >= threshold:
            boundaries.append(i)
    return boundaries


def slice_qa(flat_lines: list, boundaries: list) -> list:
    """
    Slice raw lines between boundaries.
    First line of the block = question.
    Everything else (including any Ans-/AB- style noise token, kept
    verbatim) = answer. Zero text modification — pure slicing.
    """
    pairs = []
    n = len(flat_lines)
    for i, b_idx in enumerate(boundaries):
        next_idx = boundaries[i + 1] if i + 1 < len(boundaries) else n
        block = [flat_lines[j]["line"] for j in range(b_idx, next_idx)]
        if not block:
            continue
        question = block[0].strip()
        answer   = "\n".join(l for l in block[1:] if l.strip()).strip()
        pairs.append({"question": question, "answer": answer})
    return pairs


def extract_qa(pages: list, status_callback=None) -> list:
    """
    Multi-pass, structure-agnostic boundary detection.
    Pass thresholds loosen progressively; first pass producing
    >=1 boundary wins. Falls back to paragraph-split, then whole-doc.
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

    for threshold, label in [(6, "structural")]:
        boundaries = find_boundaries(flat_lines, threshold)
        log(f"Pass ({label}, threshold={threshold}): {len(boundaries)} boundaries")
        if len(boundaries) >= 1:
            pairs = slice_qa(flat_lines, boundaries)
            log(f"Extracted {len(pairs)} Q-A pair(s) [{label}]")
            return pairs

    # ── Fallback: paragraph split (blank-line separated blocks) ──────────
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

    # ── Last resort: entire document as one block ────────────────────────
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
