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


def get_groq_client():
    from groq import Groq
    key = get_api_key("GROQ_API_KEY")
    if not key:
        raise Exception("GROQ_API_KEY not found")
    return Groq(api_key=key)


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
# OCR — mistral-ocr-latest
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
# BUILD FLAT LINE INDEX
# Numbered list of every line across all pages
# [L0] line text
# [L1] line text ...
# This is what Groq reads to find question positions
# =========================================================

def build_line_index(pages: list) -> list:
    """Returns list of {line_number, page_number, text}"""
    index = []
    for page in pages:
        for line in page["raw_text"].split("\n"):
            index.append({
                "line_number":  len(index),
                "page_number":  page["page_number"],
                "text":         line
            })
    return index


# =========================================================
# GROQ — find question start lines only
# Groq ONLY returns line numbers — never touches content
# Works for any language, any format, any structure
# =========================================================

def find_boundaries_with_groq(line_index: list, status_callback=None) -> list:
    """
    Sends numbered lines to Groq.
    Groq returns ONLY the line numbers where each question starts.
    No content extraction — just positions.
    Returns list of {question_id, start_line, answer_start_line}
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    groq = get_groq_client()

    # Send lines in chunks to stay under token limits
    # Each chunk: up to 400 lines (~6000 tokens)
    CHUNK_SIZE = 400

    all_boundaries = []

    total_chunks = (len(line_index) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_idx in range(total_chunks):
        start = chunk_idx * CHUNK_SIZE
        end   = min(start + CHUNK_SIZE, len(line_index))

        chunk_lines = line_index[start:end]

        numbered = "\n".join(
            f"[L{entry['line_number']}] {entry['text']}"
            for entry in chunk_lines
        )

        log(f"Groq scanning lines {start}-{end} (chunk {chunk_idx+1}/{total_chunks})...")

        prompt = f"""You are scanning a document to find where questions and answers begin.

The document may be in ANY language (English, Hindi, etc.) and ANY format.
Each line is prefixed with its line number like [L42].

Your job:
1. Find every QUESTION in this chunk
2. Find where the ANSWER to each question starts (first line of student's response)

A "question" is any prompt the student is expected to answer:
- Numbered: "1.", "Q.1", "Q-4", "9-7."
- Section headers followed by a question
- Any instruction like "Examine...", "Explain...", "Write...", "बताइए", "समझाइए" etc.

Return ONLY this JSON — no explanation, no markdown:
{{
  "boundaries": [
    {{
      "question_id": "<label like Q1, Q.2, 9-7, i, ii — from the document>",
      "question_start_line": <integer>,
      "answer_start_line": <integer or null if answer not in this chunk>
    }}
  ]
}}

If no questions found in this chunk, return: {{"boundaries": []}}

NUMBERED LINES:
{numbered}"""

        try:
            resp = groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You find line numbers only. Return valid JSON only. Never modify or quote document content."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )

            raw  = resp.choices[0].message.content.strip()
            data = json.loads(raw)
            chunk_boundaries = data.get("boundaries", [])
            all_boundaries.extend(chunk_boundaries)
            log(f"  Found {len(chunk_boundaries)} question(s) in this chunk")

        except Exception as e:
            log(f"  Groq error on chunk {chunk_idx+1}: {e}")
            continue

    # Sort by question start line
    all_boundaries.sort(key=lambda b: b.get("question_start_line", 0))

    # Deduplicate — same line may appear in overlapping chunks
    seen = set()
    unique = []
    for b in all_boundaries:
        key = b.get("question_start_line")
        if key not in seen:
            seen.add(key)
            unique.append(b)

    log(f"Total questions found: {len(unique)}")
    return unique


# =========================================================
# SLICE RAW TEXT BETWEEN BOUNDARIES
# Pure Python — zero LLM involvement
# Answer = raw lines from answer_start_line to next question_start_line
# =========================================================

def slice_qa_pairs(line_index: list, boundaries: list) -> list:
    """
    For each boundary:
      question = raw lines from question_start_line to answer_start_line - 1
      answer   = raw lines from answer_start_line to next question_start_line - 1

    Everything is raw OCR text — untouched, unmodified.
    """
    qa_pairs = []

    for i, b in enumerate(boundaries):
        q_start   = b.get("question_start_line", 0)
        ans_start = b.get("answer_start_line")

        # If Groq couldn't find where answer starts, use question_start + 1
        if ans_start is None or ans_start <= q_start:
            ans_start = q_start + 1

        # Next boundary starts where this answer ends
        if i + 1 < len(boundaries):
            next_q_start = boundaries[i + 1].get("question_start_line", len(line_index))
        else:
            next_q_start = len(line_index)

        # Slice question lines
        q_lines = [
            line_index[j]["text"]
            for j in range(q_start, min(ans_start, len(line_index)))
            if line_index[j]["text"].strip()
        ]

        # Slice answer lines
        ans_lines = [
            line_index[j]["text"]
            for j in range(ans_start, min(next_q_start, len(line_index)))
            if line_index[j]["text"].strip()
        ]

        question_text = " ".join(q_lines).strip()
        answer_text   = " ".join(ans_lines).strip()

        qa_pairs.append({
            "question": question_text,
            "answer":   answer_text
        })

    return qa_pairs


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

    # Step 4: Build flat line index
    log("Building line index...")
    line_index = build_line_index(pages)
    log(f"Total lines: {len(line_index)}")

    # Step 5: Groq finds question+answer boundary line numbers
    log("Finding question boundaries (Groq — line numbers only)...")
    boundaries = find_boundaries_with_groq(line_index, status_callback)

    if not boundaries:
        raise Exception(
            "No question boundaries found.\n"
            f"First 10 lines of OCR:\n" +
            "\n".join(e["text"] for e in line_index[:10])
        )

    # Step 6: Slice raw text — zero LLM
    log("Slicing raw Q-A pairs (no LLM)...")
    qa_pairs = slice_qa_pairs(line_index, boundaries)

    log(f"Done — {len(qa_pairs)} Q-A pairs")
    return ocr_json, qa_pairs
