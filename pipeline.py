import os
import re
import json
import time
import tempfile
import httpx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
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
# 2. SCHEMAS FOR GROQ
# =========================================================

class QAPair(BaseModel):
    question: str = Field(
        description="The full question label and prompt e.g., 'Q1(a) What is momentum?'"
    )
    answer: str = Field(
        description="The complete, uncut student answer extracted for this question."
    )

class DocumentQASchema(BaseModel):
    qa_pairs: List[QAPair] = Field(
        description="List of all question-answer pairs extracted from the document in order."
    )


# =========================================================
# 3. DATALAB OCR & TEXT EXTRACTION
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
        log("Native text not found. Fallback to OCR...")
        full_text = run_datalab_ocr(pdf_path, log=log)

    return full_text


def post_clean_noise(text: str) -> str:
    """Removes signature / scribble noise descriptions directly from extracted text."""
    lines = text.split("\n")
    cleaned = []
    noise_pattern = re.compile(
        r'(red\s*scribble|signature\s*inside|circle|red\s*line|extending\s*towards|bottom\s*of\s*the\s*page)',
        re.IGNORECASE
    )
    for line in lines:
        if not noise_pattern.search(line):
            cleaned.append(line)
    return "\n".join(cleaned).strip()


# =========================================================
# 4. DIRECT STRUCTURED EXTRACTOR
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

        log("1. Reading PDF text...")
        raw_text = get_pdf_text(pdf_path, log=log)
        
        if not raw_text.strip():
            raise Exception("Failed to extract text from document.")

        log("2. Mapping Questions and Answers...")

        system_prompt = """You are an expert exam paper evaluator.
Your task is to analyze the provided OCR document containing student answer sheets and extract ALL question-answer pairs.

INSTRUCTIONS:
1. Identify every question (e.g., Q1, Q1(a), Q2, Q3) in exact order.
2. Group the COMPLETE student response under its respective question. DO NOT trim or cut answers midway.
3. Ignore visual descriptions of annotations like 'red scribble' or 'signature' in the answer text.
4. Output valid JSON matching the requested schema."""

        user_prompt = f"DOCUMENT TEXT:\n\n{raw_text[:25000]}"

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={
                "type": "json_object",
                "schema": DocumentQASchema.model_json_schema()
            },
            temperature=0.0
        )

        parsed = DocumentQASchema.model_validate_json(completion.choices[0].message.content)

        final_pairs = []
        for pair in parsed.qa_pairs:
            cleaned_answer = post_clean_noise(pair.answer)
            final_pairs.append({
                "question": pair.question.strip(),
                "answer": cleaned_answer if cleaned_answer else "Answer text unavailable.",
                "matched": True
            })

        log(f"Successfully mapped {len(final_pairs)} Q&A pairs.")
        return final_pairs


# =========================================================
# 5. STREAMLIT APP ENTRYPOINT
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
