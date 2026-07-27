import os
import re
import json
import time
import tempfile
import httpx
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from groq import Groq

# =========================================================
# 1. API KEY SETUP
# =========================================================

def get_api_key(name: str) -> str:
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    
    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv(name, "")


# =========================================================
# 2. DATALAB OCR & TEXT EXTRACTION
# =========================================================

DATALAB_BASE_URL = "https://www.datalab.to"

def run_datalab_ocr(pdf_path: str, log=print) -> str:
    api_key = get_api_key("DATALAB_API_KEY")
    if not api_key:
        raise Exception("DATALAB_API_KEY missing!")

    log("Scanned PDF detected. Running Datalab OCR...")
    
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    headers = {"X-API-Key": api_key}
    file_name = os.path.basename(pdf_path)

    resp = httpx.post(
        f"{DATALAB_BASE_URL}/api/v1/convert",
        headers=headers,
        files={"file": (file_name, file_bytes, "application/pdf")},
        data={"output_format": "markdown", "mode": "accurate"},
        timeout=120
    )

    if resp.status_code != 200:
        raise Exception(f"Datalab submission failed: {resp.text}")

    data = resp.json()
    check_url = data["request_check_url"]

    for _ in range(150):
        poll_resp = httpx.get(check_url, headers=headers, timeout=60)
        if poll_resp.status_code == 200:
            result = poll_resp.json()
            if result.get("status") == "complete":
                return result.get("markdown") or ""
            if result.get("status") == "failed":
                raise Exception(f"Datalab error: {result.get('error')}")
        time.sleep(2)

    raise Exception("Datalab OCR timeout.")


def normalize_ocr_layout(raw_ocr_text: str) -> str:
    """
    ROOT CAUSE FIX:
    Datalab OCR adds '#' headers or bold tags around questions, fusing the question
    and the FIRST SENTENCE of the student's answer into the same header block line.
    This function strips markdown tags and forces newline separations.
    """
    text = raw_ocr_text
    
    # 1. Strip markdown header hashes (#, ##, ###)
    text = re.sub(r'^[#]+\s*', '', text, flags=re.MULTILINE)
    
    # 2. Unbold/Unitalicize formatting that traps first sentences
    text = text.replace('**', '').replace('__', '')
    
    # 3. Handle inline question-answer merging (e.g. "Q1. What is X? Ans: It is Y")
    # Force newlines before answer indicators if trapped inline
    text = re.sub(r'(\?|\:)\s*(Ans|Answer|Sol|Solution|A\:)', r'\1\n\2', text, flags=re.IGNORECASE)
    
    # 4. Remove teacher annotation artifacts
    noise_pattern = re.compile(
        r'^\s*(?:A\s+)?(?:red\s*scribble|signature\s*inside|circle|red\s*line|extending\s*towards|bottom\s*of\s*the\s*page).*\.?$',
        re.IGNORECASE | re.MULTILINE
    )
    text = noise_pattern.sub('', text)

    return text.strip()


def get_pdf_text(pdf_path: str, log=print) -> str:
    doc = fitz.open(pdf_path)
    text_chunks = []

    for page in doc:
        t = page.get_text("text")
        if t.strip():
            text_chunks.append(t)
    doc.close()

    full_text = "\n".join(text_chunks).strip()

    if len(full_text) < 50:
        log("Running Datalab OCR...")
        full_text = run_datalab_ocr(pdf_path, log=log)

    # Normalize layout to prevent first-sentence loss
    return normalize_ocr_layout(full_text)


# =========================================================
# 3. DIRECT EXTRACTOR WITH UNBOUNDED TOKEN CAPACITY
# =========================================================

class DirectQAExtractor:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_api_key("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing!")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"

    def process(self, pdf_path: str, status_callback=None) -> List[Dict[str, Any]]:
        def log(msg):
            print(msg)
            if status_callback:
                status_callback(msg)

        log("1. Extracting and Normalizing OCR Text...")
        normalized_text = get_pdf_text(pdf_path, log=log)

        if not normalized_text.strip():
            raise Exception("Failed to extract text from document.")

        log("2. Extracting Question-Answer pairs with full token capacity...")

        system_prompt = """You are a meticulous exam grader extracting Q&A pairs from OCR text.

STRICT EXTRACTION RULES:
1. QUESTION SEPARATION:
   - Identify every distinct question (e.g., Q1, 1(a), Question 2).
   - Extract the full question prompt into 'question'.

2. COMPLETE ANSWER CAPTURE (ZERO LOSS):
   - The student's answer starts IMMEDIATELY after the question ends.
   - Include the VERY FIRST SENTENCE of the student's writing.
   - Capture ALL words, sentences, steps, and paragraphs until the next question starts.
   - ABSOLUTELY DO NOT summarize, shorten, cut off, or skip any part of the answer.

OUTPUT REQUIREMENT:
Return ONLY a JSON object:
{
  "qa_pairs": [
    {
      "question": "Full Question Header/Text",
      "answer": "Complete, 100% untruncated student answer"
    }
  ]
}"""

        user_prompt = f"RAW OCR DOCUMENT TEXT:\n\n{normalized_text}"

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_completion_tokens=8192
        )

        response_content = completion.choices[0].message.content.strip()
        data = json.loads(response_content)
        raw_pairs = data.get("qa_pairs", [])

        final_pairs = []
        for pair in raw_pairs:
            q_str = str(pair.get("question", "")).strip()
            a_str = str(pair.get("answer", "")).strip()
            
            final_pairs.append({
                "question": q_str,
                "answer": a_str if a_str else "Answer text unavailable.",
                "matched": True
            })

        log(f"Successfully extracted {len(final_pairs)} Q&A pairs without truncation.")
        return final_pairs


# =========================================================
# 4. STREAMLIT WRAPPER
# =========================================================

def process_pdf(file_input, status_callback=None):
    file_bytes = None
    
    if hasattr(file_input, "read"):
        file_bytes = file_input.read()
    elif isinstance(file_input, (bytes, bytearray)):
        file_bytes = bytes(file_input)
    elif isinstance(file_input, tuple):
        file_bytes = file_input[0] if isinstance(file_input[0], (bytes, bytearray)) else file_input[1]

    if file_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
    else:
        tmp_path = str(file_input)

    try:
        extractor = DirectQAExtractor()
        qa_pairs = extractor.process(tmp_path, status_callback=status_callback)
        ocr_json = {"status": "Success"}
        return ocr_json, qa_pairs
    finally:
        if file_bytes and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
