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
    """Strips layout formatting trap tags and annotation noise."""
    text = raw_ocr_text
    text = re.sub(r'^[#]+\s*', '', text, flags=re.MULTILINE)
    text = text.replace('**', '').replace('__', '')
    text = re.sub(r'(\?|\:)\s*(Ans|Answer|Sol|Solution|A\:)', r'\1\n\2', text, flags=re.IGNORECASE)
    
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

    return normalize_ocr_layout(full_text)


def chunk_text(text: str, max_chars: int = 12000) -> List[str]:
    """Splits huge text into Groq TPM-compliant chunks."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > max_chars:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_len = len(para)
        else:
            current_chunk.append(para)
            current_len += len(para)

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


# =========================================================
# 3. DIRECT EXTRACTOR WITH RATE-LIMIT & CHUNKING LOGIC
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

        log("1. Reading PDF and Normalizing OCR Text...")
        normalized_text = get_pdf_text(pdf_path, log=log)

        if not normalized_text.strip():
            raise Exception("Failed to extract text from document.")

        log("2. Chunking text to meet Groq 12k Token Limit...")
        # 12,000 characters is ~3,000 tokens per chunk (Safe below TPM limit)
        chunks = chunk_text(normalized_text, max_chars=12000)
        log(f"Document split into {len(chunks)} chunk(s). Processing...")

        system_prompt = """You are a precise exam evaluator extracting Question and Answer pairs from OCR text.

STRICT INSTRUCTIONS:
1. Identify all complete Questions and Student Answers present in the snippet.
2. Ensure the student's answer includes the VERY FIRST SENTENCE and is 100% UNTRUNCATED.
3. Preserve all equations, text steps, and student explanations exactly as written.
4. If a question is cut across boundaries, extract what is completely visible.

Return ONLY a JSON object:
{
  "qa_pairs": [
    {
      "question": "Full question header/text",
      "answer": "Complete untruncated student answer"
    }
  ]
}"""

        all_qa_pairs = []

        for idx, chunk in enumerate(chunks):
            log(f"Extracting Q&A from Chunk {idx + 1}/{len(chunks)}...")
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"OCR TEXT SNIPPET:\n\n{chunk}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_completion_tokens=4096
            )

            response_content = completion.choices[0].message.content.strip()
            try:
                data = json.loads(response_content)
                raw_pairs = data.get("qa_pairs", [])
                
                for pair in raw_pairs:
                    q_str = str(pair.get("question", "")).strip()
                    a_str = str(pair.get("answer", "")).strip()
                    if q_str and a_str:
                        all_qa_pairs.append({
                            "question": q_str,
                            "answer": a_str,
                            "matched": True
                        })
            except Exception as e:
                log(f"Warning: Chunk {idx + 1} response JSON parsing error: {e}")

            # Sleep briefly between chunk requests to avoid Groq Rate Limit bursts
            if idx < len(chunks) - 1:
                time.sleep(1.5)

        log(f"Successfully mapped {len(all_qa_pairs)} total Q&A pairs without rate limits.")
        return all_qa_pairs


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
