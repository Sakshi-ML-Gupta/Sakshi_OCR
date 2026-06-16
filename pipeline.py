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
#
# KEY INSIGHT from real IGNOU documents:
#
#   The ANSWER always begins with an arrow marker:
#     →   Ans→   AB→   A→   AB-   Ans-   Ans:   A.B→
#     उत्तर→  उत्तर-  उत्तर:
#
#   This is universal across English + Hindi, printed + handwritten,
#   numbered + lettered, integrated + split formats.
#
# STRATEGY — Arrow-anchor extraction:
#   1. Flatten all non-junk pages to a line list
#   2. Find every line that STARTS WITH an arrow marker  → "answer anchors"
#   3. For each anchor, backtrack up to find its question
#      (question = labelled lines above the anchor, within a window)
#   4. Answer = everything from the anchor until the next anchor
#
# FALLBACK CHAIN (if no arrow markers found):
#   5. Question-label boundaries (Q.1, 1., (a), etc.)
#   6. Paragraph blocks
#   7. Whole document as one block
# =========================================================

# ── junk page filter ─────────────────────────────────────────────────────────

JUNK_PAGE_RE = re.compile(
    r'Acknowledgment|acknowledgement|www\.|\.asp|\.html|'
    r'KNOW YOUR ADMISSION|REGISTRATION DETAILS|'
    r'enrollment\s*no|enrolment\s*no|programme\s*code|'
    r'study\s*centre|Indira Gandhi National Open University|'
    r'Student Identity Card|Enrolment Number',
    re.I
)

NOISE_LINE_RE = re.compile(
    r'^\s*$'
    r'|!\[.*?\]\(.*?\)'          # markdown images
    r'|https?://\S+'             # URLs
    r'|^\s*[-_=]{3,}\s*$'       # horizontal rules
    r'|^\s*Page\s+\d+\s*$'      # bare page numbers
    r'|Teacher\'s\s+Signature'   # notebook footer
    r'|^\s*\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\s*$',  # bare dates
    re.I
)

def _is_junk_page(text: str) -> bool:
    lines = [l for l in text.split('\n') if l.strip()]
    if not lines:
        return True
    hits = sum(1 for l in lines if JUNK_PAGE_RE.search(l))
    return hits / max(len(lines), 1) > 0.25

def _is_noise(line: str) -> bool:
    return bool(NOISE_LINE_RE.search(line))


# ── arrow marker detection ────────────────────────────────────────────────────
#
# Matches lines where the answer marker appears at the START.
# Handles: →  Ans→  AB→  A→  AB-  Ans-  Ans:  A.B→  उत्तर→  उत्तर-
# Also handles inline: "Q.1 Ans→ ..." on same line

ARROW_START_RE = re.compile(
    r'^\s*'
    r'(?:'
    r'(?:Ans(?:wer)?|AB?|A\.?B?|उत्तर|जवाब)'  # optional prefix word
    r'\s*'
    r')?'
    r'(?:→|->|—>|⇒|-{1,2}|:)'                 # arrow/dash/colon
    r'\s*\S',                                    # something after
    re.I | re.UNICODE
)

# Inline: question label + answer marker on same line
# e.g.  "Q.1 Ans→ भाषा..."   or   "9-7. Ab- वाणिज्य..."
INLINE_ANS_RE = re.compile(
    r'(?:Ans(?:wer)?|AB?|उत्तर|जवाब)\s*(?:→|->|—>|-{1,2}|:)\s*\S',
    re.I | re.UNICODE
)

def _is_arrow_anchor(line: str) -> bool:
    """True if this line starts an answer block."""
    s = line.strip()
    if ARROW_START_RE.match(s):
        return True
    # standalone → arrow at start (common in handwritten OCR)
    if re.match(r'^→\s*\S', s):
        return True
    # ⇒ or => at start
    if re.match(r'^(?:⇒|=>)\s*\S', s):
        return True
    return False

def _split_inline(line: str):
    """
    If line has both Q-label AND answer marker inline,
    split into (question_part, answer_part).
    Returns (None, None) if not inline.
    """
    m = INLINE_ANS_RE.search(line)
    if not m:
        return None, None
    q_part  = line[:m.start()].strip()
    ans_part = line[m.start():].strip()
    return q_part, ans_part


# ── question label detection ──────────────────────────────────────────────────

Q_LABEL_RE = re.compile(
    r'^\s*'
    r'(?:'
    r'Q\.?\s*[-\.]?\s*\d[\d\.]*'        # Q.1  Q-1  Q. 3  Q.9-
    r'|(?:Ques(?:tion)?\.?\s*)?\d{1,2}[\.\)]\s+[^\d\s]'  # 1. text  2) text
    r'|\(\s*[a-zA-Z\u0900-\u097F]{1,3}\s*\)\s+\S'        # (a) (क) sub-Qs
    r'|[a-zA-Z]{1,3}\)\s+[A-Z\u0900-\u097F]'             # a) b) c)
    r'|\d{1,2}[-\.]\d{1,2}\.?\s'                          # 9-7.  9-8
    r'|\d{2,3}\.\s*Q'                                      # 95. Q
    r')',
    re.I | re.UNICODE
)

def _is_q_label(line: str) -> bool:
    return bool(Q_LABEL_RE.match(line.strip()))


# ── flatten pages ─────────────────────────────────────────────────────────────

def flatten_pages(pages: list, log=None) -> list:
    """Returns list of {"line": str, "page_number": int}, junk filtered."""
    result = []
    skipped = []
    for p in pages:
        if _is_junk_page(p["raw_text"]):
            skipped.append(p["page_number"])
            continue
        for line in p["raw_text"].split("\n"):
            if not _is_noise(line) and line.strip():
                result.append({"line": line, "page_number": p["page_number"]})
    if log and skipped:
        log(f"Skipped junk/admin pages: {skipped}")
    return result


# ── PASS 1: Arrow-anchor extraction ──────────────────────────────────────────

def extract_by_arrows(flat_lines: list, log=None) -> list:
    """
    Core extraction strategy.
    Finds arrow markers → backtracks for question → collects answer body.
    """
    n = len(flat_lines)
    anchors = []   # list of {"q_lines": [...], "ans_start": int}

    i = 0
    while i < n:
        line = flat_lines[i]["line"]

        # Check for inline Q+Ans on same line
        q_part, ans_part = _split_inline(line)
        if q_part and ans_part:
            anchors.append({
                "q_lines":  [q_part],
                "ans_start": i,
                "ans_inline": ans_part
            })
            i += 1
            continue

        if _is_arrow_anchor(line):
            # Backtrack up to 8 lines to gather question lines
            q_lines = []
            look_back = min(8, i)
            # Collect lines above that look like question content
            # Stop if we hit a previous anchor or a non-question block
            for j in range(i - 1, max(i - look_back - 1, -1), -1):
                prev = flat_lines[j]["line"].strip()
                if _is_arrow_anchor(prev):
                    break  # don't steal from previous answer
                # Stop at clearly unrelated content (long prose lines
                # that don't look like a question label or question text)
                if (len(prev) > 200
                        and not _is_q_label(prev)
                        and not prev.endswith('?')):
                    break
                q_lines.insert(0, prev)

            anchors.append({
                "q_lines":  q_lines,
                "ans_start": i,
                "ans_inline": None
            })
        i += 1

    if not anchors:
        return []

    pairs = []
    for k, anchor in enumerate(anchors):
        # Answer body = from ans_start to next anchor's ans_start
        ans_end = (anchors[k + 1]["ans_start"]
                   if k + 1 < len(anchors)
                   else n)

        if anchor["ans_inline"]:
            # Inline: answer starts mid-line
            body_lines = [anchor["ans_inline"]]
            body_lines += [flat_lines[j]["line"].strip()
                           for j in range(anchor["ans_start"] + 1, ans_end)
                           if flat_lines[j]["line"].strip()]
        else:
            body_lines = [flat_lines[j]["line"].strip()
                          for j in range(anchor["ans_start"], ans_end)
                          if flat_lines[j]["line"].strip()]

        question = " ".join(anchor["q_lines"]).strip()
        answer   = " ".join(body_lines).strip()

        if question or answer:
            pairs.append({"question": question, "answer": answer})

    return pairs


# ── PASS 2: Question-label boundaries ────────────────────────────────────────

def extract_by_labels(flat_lines: list) -> list:
    boundaries = [i for i, row in enumerate(flat_lines)
                  if _is_q_label(row["line"])]
    if not boundaries:
        return []

    pairs = []
    n = len(flat_lines)
    for k, b in enumerate(boundaries):
        end = boundaries[k + 1] if k + 1 < len(boundaries) else n
        block = [flat_lines[j]["line"].strip()
                 for j in range(b, end)
                 if flat_lines[j]["line"].strip()]
        if block:
            pairs.append({
                "question": block[0],
                "answer":   " ".join(block[1:]).strip()
            })
    return pairs


# ── PASS 3: Paragraph blocks ─────────────────────────────────────────────────

def extract_by_paragraphs(flat_lines: list) -> list:
    pairs  = []
    block  = []
    raw    = [row["line"] for row in flat_lines]

    for line in raw:
        if line.strip():
            block.append(line.strip())
        else:
            if block:
                pairs.append({
                    "question": block[0],
                    "answer":   " ".join(block[1:]).strip()
                })
                block = []
    if block:
        pairs.append({
            "question": block[0],
            "answer":   " ".join(block[1:]).strip()
        })
    return pairs


# ── MAIN EXTRACTION ───────────────────────────────────────────────────────────

def extract_qa(pages: list, status_callback=None) -> list:
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    flat_lines = flatten_pages(pages, log)
    log(f"Usable lines after filtering: {len(flat_lines)}")

    if not flat_lines:
        log("No usable text found.")
        return []

    # Pass 1 — arrow anchors (primary strategy)
    pairs = extract_by_arrows(flat_lines, log)
    log(f"Pass 1 (arrow anchors): {len(pairs)} pair(s)")
    if pairs:
        return pairs

    # Pass 2 — question label boundaries
    pairs = extract_by_labels(flat_lines)
    log(f"Pass 2 (Q-label boundaries): {len(pairs)} pair(s)")
    if pairs:
        return pairs

    # Pass 3 — paragraph blocks
    pairs = extract_by_paragraphs(flat_lines)
    log(f"Pass 3 (paragraph blocks): {len(pairs)} pair(s)")
    if pairs:
        return pairs

    # Pass 4 — whole document
    log("Pass 4 (fallback): returning full document as one block")
    full = " ".join(row["line"] for row in flat_lines)
    return [{"question": "Full document", "answer": full}]


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

    log("Extracting Q&A (arrow-anchor strategy, no LLM)...")
    qa_pairs = extract_qa(pages, status_callback)

    log(f"Pipeline complete — {len(qa_pairs)} Q-A pair(s)")
    return ocr_json, qa_pairs
