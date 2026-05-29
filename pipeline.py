import os
import io
import re
import json
import fitz
import base64
from pathlib import Path
from mistralai import Mistral

# =========================================================
# CLIENT SETUP
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
        raise Exception("MISTRAL_API_KEY not found")
    return Mistral(api_key=api_key)


def get_groq_client():
    try:
        import streamlit as st
        groq_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise Exception("GROQ_API_KEY not found")
    from groq import Groq
    return Groq(api_key=groq_key)


# =========================================================
# PREPROCESS PDF
# =========================================================

def preprocess_pdf(file_input, dpi=250):
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
# OCR — one call per page, returns real page boundaries
# =========================================================

def run_ocr(file_content: bytes, file_name: str, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    client = get_mistral_client()
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
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        },
                        {
                            "type": "text",
                            "text": (
                                "Transcribe every character from this image exactly as it appears. "
                                "Do not correct spelling, grammar, or punctuation. "
                                "Do not add, remove, or change any word. "
                                "Preserve all line breaks and paragraph structure. "
                                "Output only the raw transcribed text, nothing else."
                            )
                        }
                    ]
                }
            ]
        )

        page_text = response.choices[0].message.content or ""
        pages_output.append({
            "page_number": page_num + 1,
            "text": page_text.strip()
        })

    src_doc.close()
    log(f"OCR complete — {total_pages} pages")
    return pages_output


# =========================================================
# BUILD OCR JSON — real page structure, no fake splitting
# =========================================================

def build_ocr_json(pages_output: list) -> dict:
    pages_data = []

    for page in pages_output:
        # Keep lines as-is, just strip empty lines
        raw_lines = page["text"].split("\n")
        lines = [l for l in raw_lines if l.strip()]

        pages_data.append({
            "page_number": page["page_number"],
            "text": lines,
            "raw_text": page["text"]   # preserve full raw text too
        })

    return {
        "total_pages": len(pages_data),
        "pages": pages_data
    }


# =========================================================
# BUILD FLAT LINE INDEX
# Each entry: {line_index, page_number, text}
# This is what we slice for answers — pure OCR text
# =========================================================

def build_line_index(ocr_json: dict) -> list:
    line_index = []
    for page in ocr_json["pages"]:
        for line in page["text"]:
            if line.strip():
                line_index.append({
                    "page_number": page["page_number"],
                    "text": line          # raw, unmodified
                })
    return line_index


# =========================================================
# USE GROQ ONLY TO FIND QUESTION POSITIONS
# Returns list of {question_id, question_text, line_hint}
# Groq only identifies — it does NOT touch answer text
# =========================================================

def find_questions_with_groq(ocr_json: dict, status_callback=None) -> list:
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    groq_client = get_groq_client()

    # Build full text with line numbers so Groq can reference them
    line_index = build_line_index(ocr_json)
    numbered_lines = "\n".join([
        f"[L{i}] {entry['text']}"
        for i, entry in enumerate(line_index)
    ])

    # Chunk if too large — only need first ~15000 chars for question detection
    # Questions are usually not spread across the entire doc
    text_for_groq = numbered_lines[:15000]

    log("Asking Groq to locate questions (line numbers only)...")

    prompt = f"""You are reading a numbered line dump of a scanned assignment document.
Each line is prefixed with its line number like [L0], [L1], [L2], etc.

Your ONLY job: identify every question and return the line number where each question starts.

Return ONLY this JSON — no markdown, no explanation:
{{
  "questions": [
    {{
      "question_id": "<id from document numbering, e.g. Q1, A1(i), B3>",
      "question_text": "<exact text of the question line, copied verbatim>",
      "start_line": <integer line number where this question starts>
    }}
  ]
}}

NUMBERED LINES:
{text_for_groq}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You identify question locations by line number. Return valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=2048,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content.strip()

    clean = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    clean = re.sub(r'\s*```$', '', clean, flags=re.MULTILINE).strip()

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        raise Exception(f"Groq JSON parse error: {e}\n{raw[:400]}")

    questions = data.get("questions", [])
    log(f"Groq identified {len(questions)} question(s)")
    return questions, line_index


# =========================================================
# SLICE RAW OCR TEXT FOR ANSWERS
# Pure text slicing — zero LLM involvement
# =========================================================

def slice_answers_from_ocr(questions: list, line_index: list) -> list:
    """
    For each question, the answer = all lines between
    (question start line + 1) and (next question start line - 1).
    Raw text only. No LLM. No modification.
    """

    if not questions:
        return []

    # Sort questions by their start line
    questions_sorted = sorted(questions, key=lambda q: q.get("start_line", 0))

    qa_pairs = []

    for i, q in enumerate(questions_sorted):
        start = q.get("start_line", 0)

        # Answer starts one line after the question
        answer_start = start + 1

        # Answer ends one line before the next question
        if i + 1 < len(questions_sorted):
            answer_end = questions_sorted[i + 1].get("start_line", len(line_index))
        else:
            answer_end = len(line_index)

        # Slice raw lines directly — no processing
        answer_lines = [
            line_index[j]["text"]
            for j in range(answer_start, answer_end)
            if j < len(line_index)
        ]

        # Join with newline to preserve paragraph structure
        raw_answer = "\n".join(answer_lines).strip()

        qa_pairs.append({
            "question_id": q["question_id"],
            "question": q["question_text"],
            "answer": raw_answer      # 100% raw OCR text, untouched
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

    # ── Read bytes ─────────────────────────────────────────
    if isinstance(file_input, (str, Path)):
        file_bytes = Path(file_input).read_bytes()
        file_name = Path(file_input).name
    else:
        file_bytes = file_input.read()
        file_name = getattr(file_input, "name", "document.pdf")

    # ── Step 1: Preprocess ─────────────────────────────────
    log("Preprocessing PDF...")
    processed_bytes = preprocess_pdf(file_bytes)

    # ── Step 2: OCR ────────────────────────────────────────
    pages_output = run_ocr(
        processed_bytes,
        file_name,
        status_callback=status_callback
    )

    # ── Step 3: Build OCR JSON ─────────────────────────────
    log("Building OCR JSON...")
    ocr_json = build_ocr_json(pages_output)
    log(f"OCR JSON ready — {ocr_json['total_pages']} real pages")

    # ── Step 4: Groq locates questions (line numbers only) ─
    questions, line_index = find_questions_with_groq(
        ocr_json,
        status_callback=status_callback
    )

    if not questions:
        raise Exception("No questions found in the document")

    # ── Step 5: Slice raw OCR text for answers ─────────────
    log("Slicing raw OCR text for answers (no LLM)...")
    qa_pairs = slice_answers_from_ocr(questions, line_index)

    final_json = {
        "total_qa_pairs": len(qa_pairs),
        "qa_pairs": qa_pairs
    }

    log(f"Done — {len(qa_pairs)} Q&A pairs, answers are raw OCR text")
    return ocr_json, final_json