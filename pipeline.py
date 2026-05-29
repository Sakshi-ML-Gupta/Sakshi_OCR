import os
import io
import re
import json
import fitz
import base64
from pathlib import Path
from mistralai import Mistral

# =========================================================
# CLIENT SETUP — works locally and on Streamlit Cloud
# =========================================================

def get_mistral_client():
    try:
        import streamlit as st
        api_key = st.secrets["MISTRAL_API_KEY"]
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise Exception("MISTRAL_API_KEY not found in secrets or environment")

    return Mistral(api_key=api_key)


# =========================================================
# PREPROCESS PDF — rasterize to image-based PDF
# =========================================================

def preprocess_pdf(file_input, dpi=250):
    """
    Accepts file path (str/Path) or raw bytes.
    Returns rasterized PDF bytes (image-based, OCR-friendly).
    """
    try:
        if isinstance(file_input, (str, Path)):
            file_bytes = Path(file_input).read_bytes()
        else:
            file_bytes = bytes(file_input)

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

    except Exception as e:
        print(f"Preprocessing failed: {e}")
        if isinstance(file_input, (str, Path)):
            return Path(file_input).read_bytes()
        return file_input


# =========================================================
# OCR — one API call per page, tracks real page numbers
# =========================================================

def run_ocr(file_content: bytes, file_name: str, status_callback=None):
    """
    Rasterizes each PDF page and sends to Pixtral for OCR.
    Returns a list of dicts: [{page_number, text}, ...]
    Page numbers are REAL (1-indexed from the PDF).
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    client = get_mistral_client()

    log("Opening PDF for OCR...")
    src_doc = fitz.open(stream=file_content, filetype="pdf")
    total_pages = len(src_doc)
    log(f"PDF has {total_pages} page(s)")

    pages_output = []

    for page_num in range(total_pages):
        page = src_doc[page_num]
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        log(f"OCR: page {page_num + 1} of {total_pages}...")

        response = client.chat.complete(
            model="pixtral-12b-2409",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": (
                                "Extract ALL text from this image exactly as it appears. "
                                "Preserve the original structure, headings, numbering, "
                                "and line breaks. Output ONLY the raw extracted text — "
                                "no commentary, no markdown formatting, no code blocks."
                            )
                        }
                    ]
                }
            ]
        )

        page_text = response.choices[0].message.content or ""
        pages_output.append({
            "page_number": page_num + 1,   # real 1-indexed page number
            "text": page_text.strip()
        })

    src_doc.close()
    log(f"OCR complete — {total_pages} pages extracted")
    return pages_output


# =========================================================
# BUILD OCR JSON — preserves real page structure
# =========================================================

def build_ocr_json(pages_output: list) -> dict:
    """
    Takes the raw OCR output (list of {page_number, text})
    and deduplicates lines within each page.
    Does NOT split pages further — page count matches the PDF.
    """
    pages_data = []

    for page in pages_output:
        raw_lines = page["text"].split("\n")
        seen = set()
        clean_lines = []

        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            if line not in seen:
                seen.add(line)
                clean_lines.append(line)

        pages_data.append({
            "page_number": page["page_number"],
            "text": clean_lines
        })

    return {
        "total_pages": len(pages_data),
        "pages": pages_data
    }


# =========================================================
# EXTRACT Q&A — Claude does the smart extraction
# =========================================================

def extract_qa_with_claude(ocr_json: dict, status_callback=None) -> dict:
    """
    Sends the full OCR text to Claude and asks it to:
    1. Identify all questions (whatever structure exists in the doc)
    2. Find the corresponding answers
    3. Return structured JSON

    No hardcoded section names — works for any document format.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    log("Sending OCR text to Claude for Q&A extraction...")

    # Build full text with page markers so Claude knows the layout
    full_text_parts = []
    for page in ocr_json["pages"]:
        page_text = "\n".join(page["text"])
        if page_text.strip():
            full_text_parts.append(
                f"[PAGE {page['page_number']}]\n{page_text}"
            )

    full_text = "\n\n".join(full_text_parts)

    if not full_text.strip():
        raise Exception("OCR output is empty — nothing to extract")

    prompt = f"""You are an expert at extracting structured question-answer pairs from scanned assignment documents.

Below is the full OCR text of a document (with page markers). Your job is to:

1. Identify ALL questions in the document — they may be numbered, lettered, use roman numerals, or organized into sections. Do not assume any specific format.
2. For each question, find its corresponding answer (the student's written response).
3. Return a JSON object in EXACTLY this format — no extra text, no markdown, no code blocks:

{{
  "total_qa_pairs": <number>,
  "qa_pairs": [
    {{
      "question_id": "<a short unique id, e.g. Q1, A1(i), B3 — infer from the document structure>",
      "question": "<full question text>",
      "answer": "<full answer text, or empty string if no answer found>"
    }}
  ]
}}

Rules:
- Do NOT invent questions or answers
- If a question has no answer, set answer to ""
- Clean up obvious OCR artifacts (garbled characters, stray symbols) in both questions and answers
- Preserve the meaning and wording faithfully
- question_id should reflect the document's own numbering/lettering scheme

OCR TEXT:
{full_text}"""

    # Use Anthropic API (Claude) for intelligent extraction
    import urllib.request

    headers = {
        "Content-Type": "application/json",
        "x-api-key": "",   # handled by proxy
        "anthropic-version": "2023-06-01"
    }

    body = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers=headers,
        method="POST"
    )

    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    raw_response = result["content"][0]["text"].strip()

    log("Claude responded — parsing JSON...")

    # Strip markdown code fences if present
    clean = re.sub(r'^```(?:json)?\s*', '', raw_response, flags=re.MULTILINE)
    clean = re.sub(r'\s*```$', '', clean, flags=re.MULTILINE)
    clean = clean.strip()

    try:
        qa_json = json.loads(clean)
    except json.JSONDecodeError as e:
        raise Exception(
            f"Claude returned invalid JSON: {e}\n"
            f"Raw response (first 500 chars):\n{raw_response[:500]}"
        )

    log(f"Extracted {qa_json.get('total_qa_pairs', 0)} Q&A pairs")
    return qa_json


# =========================================================
# COMPLETE PIPELINE
# =========================================================

def process_pdf(file_input, status_callback=None):
    """
    Accepts:
    - A Streamlit UploadedFile object
    - A file path string or Path object (local use)

    Returns: (ocr_json, qa_json)
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    # ── Read bytes ─────────────────────────────────────────
    if isinstance(file_input, (str, Path)):
        file_bytes = Path(file_input).read_bytes()
        file_name = Path(file_input).name
    else:
        file_bytes = file_input.read()
        file_name = getattr(file_input, "name", "document.pdf")

    # ── Step 1: Preprocess ─────────────────────────────────
    log("Preprocessing PDF (rasterizing pages)...")
    processed_bytes = preprocess_pdf(file_bytes)
    log("Preprocessing complete")

    # ── Step 2: OCR ────────────────────────────────────────
    pages_output = run_ocr(
        processed_bytes,
        file_name,
        status_callback=status_callback
    )

    # ── Step 3: Build OCR JSON ─────────────────────────────
    log("Building OCR JSON...")
    ocr_json = build_ocr_json(pages_output)
    log(f"OCR JSON ready — {ocr_json['total_pages']} pages")

    # ── Step 4: Extract Q&A with Claude ───────────────────
    qa_json = extract_qa_with_claude(
        ocr_json,
        status_callback=status_callback
    )

    log(f"Pipeline complete — {qa_json['total_qa_pairs']} Q&A pairs")
    return ocr_json, qa_json