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

def extract_qa_with_groq(ocr_json: dict, status_callback=None) -> dict:

    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    # ── Get Groq API key ───────────────────────────────────
    try:
        import streamlit as st
        groq_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        groq_key = os.getenv("GROQ_API_KEY")

    if not groq_key:
        raise Exception("GROQ_API_KEY not found in secrets or environment")

    from groq import Groq
    groq_client = Groq(api_key=groq_key)

    # ── Build full text with page markers ─────────────────
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

    log(f"Total OCR text length: {len(full_text)} characters")

    # ── Step 1: Extract questions only first ───────────────
    log("Step 1: Extracting questions structure...")

    questions_prompt = f"""You are analyzing a scanned assignment document.

Your ONLY task right now is to identify and list ALL the questions in this document.

Return a JSON object in EXACTLY this format — no markdown, no explanation:

{{
  "questions": [
    {{
      "question_id": "<id inferred from document, e.g. Q1, A1(i), B3>",
      "question": "<full question text>",
      "approximate_location": "<brief hint like 'page 2, after Section A heading'>"
    }}
  ]
}}

OCR TEXT:
{full_text}"""

    q_response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a precise document extraction assistant. Respond with valid JSON only."
            },
            {
                "role": "user",
                "content": questions_prompt
            }
        ],
        temperature=0,
        max_tokens=2048,
        response_format={"type": "json_object"}
    )

    questions_raw = q_response.choices[0].message.content.strip()

    try:
        questions_data = json.loads(questions_raw)
        questions_list = questions_data.get("questions", [])
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse questions JSON: {e}\n{questions_raw[:300]}")

    log(f"Found {len(questions_list)} questions — now extracting full answers...")

    if not questions_list:
        raise Exception("No questions found in the document")

    # ── Step 2: Extract full answer for each question ──────
    # Send all questions + full text in one focused call
    # so Groq has complete context for answer boundaries

    questions_formatted = "\n".join([
        f"- {q['question_id']}: {q['question']}"
        for q in questions_list
    ])

    answers_prompt = f"""You are extracting student answers from a scanned assignment document.

Here are ALL the questions found in the document:
{questions_formatted}

Below is the full OCR text of the document. For each question above, extract the student's COMPLETE answer.

CRITICAL RULES for answer extraction:
- Capture the ENTIRE answer — do not truncate, summarize, or paraphrase
- The answer starts immediately after the question text ends
- The answer ends where the NEXT question begins (or at end of document)
- Include every sentence, every word the student wrote
- If a question has no answer written, use ""
- Do NOT include the question text itself in the answer field
- Do NOT add any commentary or notes

Return a JSON object in EXACTLY this format — no markdown, no explanation:

{{
  "qa_pairs": [
    {{
      "question_id": "<same id as above>",
      "question": "<full question text>",
      "answer": "<student's COMPLETE answer — every word, nothing cut off>"
    }}
  ]
}}

FULL OCR TEXT:
{full_text}"""

    log("Extracting complete answers...")

    a_response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a precise document extraction assistant. Extract complete, untruncated answers. Respond with valid JSON only."
            },
            {
                "role": "user",
                "content": answers_prompt
            }
        ],
        temperature=0,
        max_tokens=8192,        # increased for full answers
        response_format={"type": "json_object"}
    )

    answers_raw = a_response.choices[0].message.content.strip()

    clean = re.sub(r'^```(?:json)?\s*', '', answers_raw, flags=re.MULTILINE)
    clean = re.sub(r'\s*```$', '', clean, flags=re.MULTILINE).strip()

    try:
        answers_data = json.loads(clean)
    except json.JSONDecodeError as e:
        raise Exception(
            f"Failed to parse answers JSON: {e}\n"
            f"Raw response (first 500 chars):\n{answers_raw[:500]}"
        )

    qa_pairs = answers_data.get("qa_pairs", [])

    # ── Step 3: If answers still seem short, retry per-question ──
    SHORT_ANSWER_THRESHOLD = 80   # characters — tune if needed

    short_qa = [
        q for q in qa_pairs
        if len(q.get("answer", "")) < SHORT_ANSWER_THRESHOLD
        and q.get("answer", "") != ""
    ]

    if short_qa:
        log(f"{len(short_qa)} answers seem short — retrying those individually...")

        for qa in short_qa:
            retry_prompt = f"""From the document below, extract the student's COMPLETE answer to this specific question.

QUESTION ({qa['question_id']}): {qa['question']}

Rules:
- Return ONLY the answer text, nothing else
- Include every single word the student wrote in response to this question
- Do not summarize or shorten
- Stop only when the next question begins or the document ends

FULL OCR TEXT:
{full_text}"""

            retry_response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract the complete answer text exactly as written. Return only the answer, no JSON, no labels."
                    },
                    {
                        "role": "user",
                        "content": retry_prompt
                    }
                ],
                temperature=0,
                max_tokens=2048
            )

            full_answer = retry_response.choices[0].message.content.strip()

            # Update the answer in qa_pairs
            for q in qa_pairs:
                if q["question_id"] == qa["question_id"]:
                    q["answer"] = full_answer
                    break

            log(f"Retried {qa['question_id']} — got {len(full_answer)} chars")

    final_json = {
        "total_qa_pairs": len(qa_pairs),
        "qa_pairs": qa_pairs
    }

    log(f"Extraction complete — {len(qa_pairs)} Q&A pairs with full answers")
    return final_json


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
    qa_json = extract_qa_with_groq(
        ocr_json,
        status_callback=status_callback
    )

    log(f"Pipeline complete — {qa_json['total_qa_pairs']} Q&A pairs")
    return ocr_json, qa_json