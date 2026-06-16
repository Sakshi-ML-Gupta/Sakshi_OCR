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
# API KEYS
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
# PREPROCESS PDF  (rasterise → clean OCR)
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
# OCR  (Mistral)
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
# FLATTEN PAGES → indexed lines
#
# Build a single list of all lines across all pages,
# keeping track of which page each line came from.
# All Q&A slicing is done on this flat list so the
# raw OCR text is NEVER modified.
# =========================================================

def flatten_pages(pages: list) -> list:
    """
    Returns list of {"line": str, "page_number": int}
    Skips lines that are pure noise (URLs, image tags, blank).
    """
    NOISE = re.compile(
        r'^\s*$'                          # blank
        r'|!\[.*?\]\(.*?\)'               # markdown images
        r'|https?://\S+'                  # bare URLs
        r'|^\s*[-_=]{3,}\s*$'            # horizontal rules
    )
    result = []
    for p in pages:
        for line in p["raw_text"].split("\n"):
            if not NOISE.search(line):
                result.append({"line": line, "page_number": p["page_number"]})
    return result


# =========================================================
# LLM BOUNDARY DETECTION
#
# Claude sees the text but ONLY returns line numbers.
# Raw OCR text is sliced from the flat list — never rewritten.
# =========================================================

BOUNDARY_SYSTEM_PROMPT = """You are a document structure analyser.

You receive OCR text from a student assignment. Each line is prefixed with its
line number like:
  [42] Q.1 What is photosynthesis?
  [43] Ans- Photosynthesis is the process...

Your job: find where each question+answer block STARTS.
Return ONLY a JSON array of objects, no explanation, no markdown fences.

Each object:
{
  "question_number": "Q.1",          // label as it appears, or "?" if unlabelled
  "question_start_line": 42,         // line number where the question text begins
  "answer_start_line": 43            // line number where the student answer begins
                                     // (null if no answer found)
}

Rules:
- Ignore cover pages, URLs, headers, footers, admin info, enrollment numbers,
  dates, watermarks, admit cards, web screenshots — skip those entirely.
- A question label can be Q.1 / Q-2 / Q. 3 / 1. / (i) / (a) / no label at all.
- An answer starts at: Ans / Ans- / AB- / A. / उत्तर / or just the line after
  the question if there is no explicit marker.
- If question and answer are on the same line, set answer_start_line = question_start_line.
- If a block has no answer written by the student, set answer_start_line to null.
- Do NOT rewrite, correct, or paraphrase any text. Only report line numbers.
- Return [] if no questions found.
"""

def _call_claude_boundaries(numbered_text: str, anthropic_api_key: str) -> list:
    """
    Send numbered OCR text to Claude; get back boundary line numbers only.
    Returns list of {question_number, question_start_line, answer_start_line}.
    """
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 2048,
            "system": BOUNDARY_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": numbered_text}]
        },
        timeout=120,
    )

    if resp.status_code != 200:
        raise Exception(f"Claude API error {resp.status_code}: {resp.text[:300]}")

    raw = resp.json()["content"][0]["text"].strip()
    # Strip any accidental markdown fences
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```\s*$', '', raw, flags=re.MULTILINE).strip()

    try:
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return []


def _boundaries_from_chunk(
    flat_lines: list,
    start_idx: int,
    end_idx: int,
    anthropic_api_key: str,
    offset: int
) -> list:
    """
    Build numbered text for a slice of flat_lines, call Claude,
    return boundaries with line indices adjusted to global flat_lines space.
    offset = start_idx (so Claude's local line numbers → global indices).
    """
    numbered = "\n".join(
        f"[{offset + i}] {flat_lines[start_idx + i]['line']}"
        for i in range(end_idx - start_idx)
    )
    raw_boundaries = _call_claude_boundaries(numbered, anthropic_api_key)

    # Validate & clamp line numbers
    result = []
    for b in raw_boundaries:
        qsl = b.get("question_start_line")
        asl = b.get("answer_start_line")
        if qsl is None:
            continue
        # Already in global space because we used offset+i above
        result.append({
            "question_number": b.get("question_number", "?"),
            "question_start_line": int(qsl),
            "answer_start_line": int(asl) if asl is not None else None,
        })
    return result


# =========================================================
# SLICE RAW TEXT  (no LLM involvement)
# =========================================================

def _slice_qa(flat_lines: list, boundaries: list) -> list:
    """
    Given boundary line indices into flat_lines, slice raw text directly.
    The LLM never touches answer content — only found boundaries.
    """
    qa_pairs = []
    n = len(flat_lines)

    for i, b in enumerate(boundaries):
        qsl = b["question_start_line"]
        asl = b.get("answer_start_line")

        # Next question starts where the next boundary begins
        next_start = (
            boundaries[i + 1]["question_start_line"]
            if i + 1 < len(boundaries)
            else n
        )

        # Clamp indices
        qsl = max(0, min(qsl, n - 1))
        next_start = max(qsl + 1, min(next_start, n))

        if asl is not None:
            asl = max(qsl, min(asl, next_start))
            q_lines = flat_lines[qsl:asl]
            a_lines = flat_lines[asl:next_start]
        else:
            q_lines = flat_lines[qsl:qsl + 1]
            a_lines = flat_lines[qsl + 1:next_start]

        question_text = " ".join(row["line"] for row in q_lines).strip()
        answer_text   = " ".join(row["line"] for row in a_lines).strip()

        qa_pairs.append({
            "question": question_text,
            "answer":   answer_text,
        })

    return qa_pairs


# =========================================================
# MAIN EXTRACTION ENTRY POINT
# =========================================================

CHUNK_LINES = 300   # lines per Claude call (safe for token limits)

def extract_qa_with_llm(pages: list, status_callback=None) -> list:
    """
    Structure-agnostic Q&A extraction.

    1. Flatten all pages to a single indexed line list.
    2. Send chunks of ~300 lines to Claude → get BOUNDARY LINE NUMBERS only.
    3. Slice raw OCR text at those boundaries — text is never rewritten.
    4. Merge & de-duplicate across chunks.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    anthropic_api_key = get_api_key("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise Exception(
            "ANTHROPIC_API_KEY not set. Add it to your Streamlit secrets "
            "(.streamlit/secrets.toml) as:\nANTHROPIC_API_KEY = 'sk-ant-...'"
        )

    flat_lines = flatten_pages(pages)
    total      = len(flat_lines)
    log(f"Flattened to {total} lines across {len(pages)} pages")

    if total == 0:
        log("No usable text found.")
        return []

    # Process in chunks
    all_boundaries = []
    chunk_start    = 0

    while chunk_start < total:
        chunk_end = min(chunk_start + CHUNK_LINES, total)
        log(f"  Analysing lines {chunk_start}–{chunk_end} ({chunk_end-chunk_start} lines)...")

        try:
            boundaries = _boundaries_from_chunk(
                flat_lines, chunk_start, chunk_end,
                anthropic_api_key, offset=chunk_start
            )
            log(f"    → {len(boundaries)} boundary/ies found")
            all_boundaries.extend(boundaries)
        except Exception as e:
            log(f"    ⚠ Chunk failed: {e}")

        chunk_start = chunk_end

    if not all_boundaries:
        log("⚠ No Q&A boundaries detected in any chunk.")
        # Fallback: return entire text as one block
        full_text = " ".join(row["line"] for row in flat_lines).strip()
        return [{
            "question": "Document content (no structured Q&A detected)",
            "answer":   full_text
        }]

    # Sort by question_start_line, de-duplicate
    all_boundaries.sort(key=lambda b: b["question_start_line"])
    seen_lines = set()
    deduped = []
    for b in all_boundaries:
        key = b["question_start_line"]
        if key not in seen_lines:
            seen_lines.add(key)
            deduped.append(b)

    log(f"Total unique boundaries: {len(deduped)}")

    qa_pairs = _slice_qa(flat_lines, deduped)
    log(f"Done — {len(qa_pairs)} Q-A pair(s) extracted")
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

    log("Preprocessing PDF...")
    processed = preprocess_pdf(file_bytes)

    pages = run_ocr(processed, file_name, status_callback)

    log("Building OCR JSON...")
    ocr_json = build_ocr_json(pages)
    log(f"Total pages: {ocr_json['total_pages']}")

    log("Extracting Q&A pairs (boundary detection via LLM, slicing from raw OCR)...")
    qa_pairs = extract_qa_with_llm(pages, status_callback)

    log(f"Pipeline complete — {len(qa_pairs)} Q-A pair(s)")
    return ocr_json, qa_pairs
