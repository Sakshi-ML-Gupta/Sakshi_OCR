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
# CLIENT SETUP
# =========================================================

def get_api_key(name):
    try:
        import streamlit as st
        return st.secrets[name]
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv(name)


def get_mistral_client():
    api_key = get_api_key("MISTRAL_API_KEY")
    if not api_key:
        raise Exception("MISTRAL_API_KEY not found")
    return Mistral(api_key=api_key)


def get_groq_client():
    from groq import Groq
    groq_key = get_api_key("GROQ_API_KEY")
    if not groq_key:
        raise Exception("GROQ_API_KEY not found")
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
# OCR — mistral-ocr-latest, pure transcription
# =========================================================

def run_ocr(file_content: bytes, file_name: str, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    api_key = get_api_key("MISTRAL_API_KEY")
    client = get_mistral_client()

    log("Uploading PDF to Mistral OCR...")

    uploaded = client.files.upload(
        file={"file_name": file_name, "content": file_content},
        purpose="ocr"
    )
    log(f"Uploaded: {uploaded.id}")

    signed = client.files.get_signed_url(file_id=uploaded.id, expiry=1)
    log("Running mistral-ocr-latest (pure transcription, no corrections)...")

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
        raise Exception(f"OCR API error {resp.status_code}: {resp.text}")

    ocr_data = resp.json()

    pages_output = []
    for page in ocr_data.get("pages", []):
        pages_output.append({
            "page_number": page.get("index", 0) + 1,
            "text": page.get("markdown", "").strip()
        })

    try:
        client.files.delete(file_id=uploaded.id)
    except Exception:
        pass

    log(f"OCR complete — {len(pages_output)} pages extracted")
    return pages_output


# =========================================================
# BUILD OCR JSON — real page structure
# =========================================================

def build_ocr_json(pages_output: list) -> dict:
    pages_data = []
    for page in pages_output:
        raw_lines = [l for l in page["text"].split("\n") if l.strip()]
        pages_data.append({
            "page_number": page["page_number"],
            "text": raw_lines,
            "raw_text": page["text"]
        })
    return {
        "total_pages": len(pages_data),
        "pages": pages_data
    }


# =========================================================
# STEP 1 — GROQ: Detect document structure
# Finds which pages have questions vs answers
# =========================================================

def detect_document_structure(ocr_json: dict, status_callback=None) -> dict:
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    groq_client = get_groq_client()

    page_summaries = []
    for page in ocr_json["pages"]:
        preview = page["raw_text"][:300].replace("\n", " ")
        page_summaries.append(f"[PAGE {page['page_number']}]: {preview}")

    summary_text = "\n\n".join(page_summaries)

    prompt = f"""You are analyzing a scanned assignment document.

Below is a preview of each page. Based on the content:
1. Identify which pages contain QUESTIONS
2. Identify which pages contain ANSWERS (student responses)
3. List all question IDs in order as they appear (e.g. Q1, i, ii, B3, 1, 2 etc.)

Return ONLY this JSON:
{{
  "question_pages": [<list of page numbers>],
  "answer_pages": [<list of page numbers>],
  "question_ids": ["<id1>", "<id2>", ...]
}}

PAGE PREVIEWS:
{summary_text}"""

    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You analyze document structure. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=1024,
        response_format={"type": "json_object"}
    )

    raw = resp.choices[0].message.content.strip()
    clean = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    clean = re.sub(r'\s*```$', '', clean, flags=re.MULTILINE).strip()
    structure = json.loads(clean)

    log(f"Question pages: {structure.get('question_pages')}")
    log(f"Answer pages:   {structure.get('answer_pages')}")
    log(f"Question IDs:   {structure.get('question_ids')}")
    return structure


# =========================================================
# STEP 2 — GROQ: Extract questions verbatim from question pages
# =========================================================

def extract_questions(ocr_json: dict, question_pages: list, status_callback=None) -> list:
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    groq_client = get_groq_client()

    q_text_parts = []
    for page in ocr_json["pages"]:
        if page["page_number"] in question_pages:
            q_text_parts.append(
                f"[PAGE {page['page_number']}]\n{page['raw_text']}"
            )

    q_text = "\n\n".join(q_text_parts)[:14000]

    prompt = f"""Extract every question from this document text.
Copy question text EXACTLY as it appears — do not fix spelling, grammar, or punctuation.

Return ONLY this JSON:
{{
  "questions": [
    {{
      "question_id": "<id from document e.g. Q1, i, ii, 1, 2, B3>",
      "question": "<exact verbatim question text>"
    }}
  ]
}}

TEXT:
{q_text}"""

    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Copy text verbatim. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=2048,
        response_format={"type": "json_object"}
    )

    raw = resp.choices[0].message.content.strip()
    clean = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    clean = re.sub(r'\s*```$', '', clean, flags=re.MULTILINE).strip()
    data = json.loads(clean)
    questions = data.get("questions", [])
    log(f"Extracted {len(questions)} questions")
    return questions


# =========================================================
# STEP 3 — RAW SLICING: Extract answers — zero LLM
# Scans answer pages for question ID markers,
# slices everything between them as raw OCR text
# =========================================================

def extract_answers_raw(ocr_json: dict, answer_pages: list, question_ids: list, status_callback=None) -> dict:
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    # Build flat line list from answer pages
    all_lines = []
    for page in ocr_json["pages"]:
        if page["page_number"] in answer_pages:
            for line in page["text"]:
                if line.strip():
                    all_lines.append({
                        "page": page["page_number"],
                        "text": line
                    })

    # Fallback: use all pages if answer pages gave nothing
    if not all_lines:
        log("Answer pages empty — scanning all pages")
        for page in ocr_json["pages"]:
            for line in page["text"]:
                if line.strip():
                    all_lines.append({
                        "page": page["page_number"],
                        "text": line
                    })

    log(f"Total lines in answer section: {len(all_lines)}")

    def line_is_qid_marker(line_text, qid):
        """True if this line is the answer-section label for this question."""
        stripped = line_text.strip().rstrip(".:)- ").strip()
        if stripped.lower() == qid.lower():
            return True
        pattern = r'^' + re.escape(qid) + r'[\s\.\)\:\-]'
        if re.match(pattern, line_text.strip(), re.IGNORECASE):
            return True
        return False

    # Find first occurrence of each question_id in answer lines
    positions = {}
    for qid in question_ids:
        for i, line in enumerate(all_lines):
            if line_is_qid_marker(line["text"], qid):
                positions[qid] = i
                break

    log(f"Answer markers found: {list(positions.keys())}")
    missing = [q for q in question_ids if q not in positions]
    if missing:
        log(f"No marker found for: {missing}")

    # Sort found questions by position
    sorted_qids = sorted(positions.keys(), key=lambda q: positions[q])

    # Slice raw lines between markers — pure Python, no LLM
    answers = {}
    for idx, qid in enumerate(sorted_qids):
        start = positions[qid] + 1  # line after the label

        if idx + 1 < len(sorted_qids):
            end = positions[sorted_qids[idx + 1]]
        else:
            end = len(all_lines)

        answer_lines = [
            all_lines[j]["text"]
            for j in range(start, end)
            if all_lines[j]["text"].strip()
        ]

        # Raw join — no modification
        answers[qid] = "\n".join(answer_lines)

    # Questions with no marker get empty string
    for qid in question_ids:
        if qid not in answers:
            answers[qid] = ""

    return answers


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
    pages_output = run_ocr(processed_bytes, file_name, status_callback)

    # ── Step 3: Build OCR JSON ─────────────────────────────
    log("Building OCR JSON...")
    ocr_json = build_ocr_json(pages_output)
    log(f"OCR complete — {ocr_json['total_pages']} real pages")

    # ── Step 4: Detect structure ───────────────────────────
    log("Detecting document structure...")
    structure = detect_document_structure(ocr_json, status_callback)

    question_pages = structure.get("question_pages", [])
    answer_pages   = structure.get("answer_pages", [])
    question_ids   = structure.get("question_ids", [])

    # Fallback if detection failed
    if not question_pages or not answer_pages:
        log("Structure detection failed — splitting doc in half as fallback")
        mid = max(1, ocr_json["total_pages"] // 2)
        question_pages = list(range(1, mid + 1))
        answer_pages   = list(range(mid + 1, ocr_json["total_pages"] + 1))

    # ── Step 5: Extract questions verbatim ─────────────────
    log("Extracting questions from question pages...")
    questions = extract_questions(ocr_json, question_pages, status_callback)

    if not questions:
        raise Exception("No questions found in question pages")

    if not question_ids:
        question_ids = [q["question_id"] for q in questions]

    # ── Step 6: Slice raw answers — zero LLM ───────────────
    log("Slicing raw answers from answer pages (no LLM)...")
    answers = extract_answers_raw(
        ocr_json, answer_pages, question_ids, status_callback
    )

    # ── Step 7: Build final JSON — same format as sample ───
    qa_pairs = []
    for q in questions:
        qid = q["question_id"]
        qa_pairs.append({
            "question": q["question"],
            "answer": answers.get(qid, "")
        })

    # plain list — matches your sample format exactly
    log(f"Done — {len(qa_pairs)} Q&A pairs")
    return ocr_json, qa_pairs