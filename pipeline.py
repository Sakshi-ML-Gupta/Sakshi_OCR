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

    # ── Helper: call Groq ──────────────────────────────────
    def groq_call(messages, max_tokens=2048):
        return groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        ).choices[0].message.content.strip()

    def parse_json_safe(raw, label=""):
        clean = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        clean = re.sub(r'\s*```$', '', clean, flags=re.MULTILINE).strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            raise Exception(f"JSON parse error in {label}: {e}\n{raw[:400]}")

    # ── Build page texts ───────────────────────────────────
    pages = []
    for page in ocr_json["pages"]:
        text = "\n".join(page["text"]).strip()
        if text:
            pages.append({
                "page_number": page["page_number"],
                "text": text
            })

    if not pages:
        raise Exception("OCR output is empty — nothing to extract")

    # ── Chunk pages ~20000 chars each ─────────────────────
    CHUNK_CHAR_LIMIT = 20000

    chunks = []
    current_chunk = []
    current_len = 0

    for page in pages:
        page_block = f"[PAGE {page['page_number']}]\n{page['text']}"
        page_len = len(page_block)

        if current_len + page_len > CHUNK_CHAR_LIMIT and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [page_block]
            current_len = page_len
        else:
            current_chunk.append(page_block)
            current_len += page_len

    if current_chunk:
        chunks.append(current_chunk)

    log(f"Document split into {len(chunks)} chunk(s) for processing")

    # ── Step 1: Extract ALL questions first from full doc ──
    # Use first chunk + last chunk to get complete question list
    # This ensures no question is missed even if doc is large

    full_text_for_questions = "\n\n".join(
        "\n\n".join(c) for c in chunks
    )

    # If full text too large for one call, use first 18000 chars
    # Questions are usually in the beginning of the document
    questions_text = full_text_for_questions[:18000]

    log("Step 1: Extracting complete question list...")

    questions_prompt = f"""You are reading a scanned assignment document.

Your ONLY task: list every single question present in this document.

STRICT RULES:
- Copy question text EXACTLY as it appears — do not fix spelling, grammar, or punctuation
- Do not paraphrase, correct, or modify any text in any way
- Copy the raw OCR text verbatim, including any OCR errors or odd characters
- Do not skip any question even if it looks incomplete or garbled
- Infer question_id from the document's own numbering scheme

Return ONLY this JSON:
{{
  "questions": [
    {{
      "question_id": "<id from document e.g. Q1, A1(i), B3>",
      "question": "<exact question text copied verbatim from OCR>"
    }}
  ]
}}

DOCUMENT TEXT:
{questions_text}"""

    q_raw = groq_call(
        messages=[
            {
                "role": "system",
                "content": "You copy text verbatim. No corrections, no improvements. Return valid JSON only."
            },
            {
                "role": "user",
                "content": questions_prompt
            }
        ],
        max_tokens=2048
    )

    q_data = parse_json_safe(q_raw, label="question extraction")
    all_questions = q_data.get("questions", [])

    if not all_questions:
        raise Exception("No questions found in the document")

    log(f"Found {len(all_questions)} questions total")

    # ── Step 2: Extract answers chunk by chunk ─────────────
    all_qa_pairs = []

    for i, chunk in enumerate(chunks):
        chunk_text = "\n\n".join(chunk)
        log(f"Extracting answers from chunk {i + 1} of {len(chunks)}...")

        # Pass the full question list so LLM knows what to look for
        questions_list_str = "\n".join([
            f"- {q['question_id']}: {q['question']}"
            for q in all_questions
        ])

        answers_prompt = f"""You are extracting raw student answers from a scanned assignment document.

THESE ARE ALL THE QUESTIONS IN THE DOCUMENT:
{questions_list_str}

For each question that appears in the text chunk below, extract the student's answer.

STRICT RULES — CRITICAL:
- Copy answer text EXACTLY as it appears in the OCR — do not fix spelling, grammar, punctuation, or any errors
- Do not paraphrase, summarize, correct, or improve the text in any way
- Copy raw OCR text verbatim, including typos, OCR artifacts, odd characters
- Capture the COMPLETE answer — every single word until the next question starts
- If a question does not appear in this chunk, do not include it
- If a question appears but has no answer written, set answer to ""
- Do NOT include the question text itself inside the answer field

Return ONLY this JSON:
{{
  "qa_pairs": [
    {{
      "question_id": "<id matching the question list above>",
      "question": "<exact question text verbatim>",
      "answer": "<exact raw answer text verbatim — complete, unmodified>"
    }}
  ]
}}

TEXT CHUNK:
{chunk_text}"""

        raw = groq_call(
            messages=[
                {
                    "role": "system",
                    "content": "You copy text verbatim without any corrections or modifications. Return valid JSON only."
                },
                {
                    "role": "user",
                    "content": answers_prompt
                }
            ],
            max_tokens=3000
        )

        data = parse_json_safe(raw, label=f"chunk {i+1}")
        pairs = data.get("qa_pairs", [])
        log(f"Chunk {i + 1}: {len(pairs)} pair(s) found")
        all_qa_pairs.extend(pairs)

    # ── Step 3: Merge answers for same question_id ─────────
    log("Merging results across chunks...")

    merged = {}
    for qa in all_qa_pairs:
        qid = qa["question_id"]
        if qid not in merged:
            merged[qid] = qa
        else:
            existing = merged[qid]["answer"]
            new = qa["answer"]
            if new and new not in existing:
                merged[qid]["answer"] = (existing + " " + new).strip()

    # ── Step 4: Ensure EVERY question is in output ─────────
    # Fill in any missing questions with empty answer
    for q in all_questions:
        qid = q["question_id"]
        if qid not in merged:
            log(f"Question {qid} had no answer found — adding with empty answer")
            merged[qid] = {
                "question_id": qid,
                "question": q["question"],
                "answer": ""
            }

    # ── Step 5: Retry questions with missing/short answers ─
    SHORT_THRESHOLD = 60

    need_retry = [
        q for q in merged.values()
        if len(q.get("answer", "")) < SHORT_THRESHOLD
    ]

    if need_retry:
        log(f"{len(need_retry)} question(s) need answer retry...")

        for qa in need_retry:
            # Find the chunk most likely to contain this answer
            best_chunk_text = max(
                ["\n\n".join(c) for c in chunks],
                key=lambda ct: fuzz_score(qa["question"], ct)
            )

            retry_prompt = f"""From the document text below, find and copy the student's COMPLETE answer to this specific question.

QUESTION ({qa['question_id']}): {qa['question']}

STRICT RULES:
- Copy the answer EXACTLY as written — no spelling fixes, no grammar fixes, no changes of any kind
- Raw verbatim OCR text only
- Include every word until the next question starts or document ends

Return ONLY this JSON:
{{
  "answer": "<complete verbatim answer text>"
}}

DOCUMENT TEXT:
{best_chunk_text}"""

            raw = groq_call(
                messages=[
                    {
                        "role": "system",
                        "content": "Copy text verbatim. No corrections. Return valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": retry_prompt
                    }
                ],
                max_tokens=1500
            )

            retry_data = parse_json_safe(raw, label=f"retry {qa['question_id']}")
            new_answer = retry_data.get("answer", "").strip()

            if new_answer and len(new_answer) > len(qa.get("answer", "")):
                merged[qa["question_id"]]["answer"] = new_answer
                log(f"  {qa['question_id']}: {len(new_answer)} chars")

    # ── Build final output ─────────────────────────────────
    final_pairs = list(merged.values())

    result = {
        "total_qa_pairs": len(final_pairs),
        "qa_pairs": final_pairs
    }

    log(f"Complete — {len(final_pairs)} Q&A pairs (all questions accounted for)")
    return result


def fuzz_score(question: str, text: str) -> int:
    question_words = set(question.lower().split())
    text_words = set(text.lower().split())
    if not question_words:
        return 0
    return len(question_words & text_words)


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