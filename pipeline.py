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
                # Safe markdown field parsing (Handles all payload structures)
                md = result.get("markdown") or result.get("text") or ""
                if not md and "pages" in result:
                    md = "\n\n".join([p.get("markdown", "") for p in result["pages"]])
                return md
            if result.get("status") == "failed":
                raise Exception(f"Datalab error: {result.get('error')}")
        time.sleep(2)

    raise Exception("Datalab OCR timeout.")


def normalize_ocr_layout(raw_ocr_text: str) -> str:
    """Strips markdown header hashes and joins inline answer tags."""
    text = raw_ocr_text
    # Hash header tags strip karna
    text = re.sub(r'^[#]+\s*', '', text, flags=re.MULTILINE)
    text = text.replace('**', '').replace('__', '')
    # Inline Ans labels ko next line par push karna
    text = re.sub(r'(\?|\:)\s*(Ans|Answer|Sol|Solution|A\:)', r'\1\n\2', text, flags=re.IGNORECASE)
    
    # Teacher noise scribble patterns remove karna
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
        log("Running Datalab OCR for image PDF...")
        full_text = run_datalab_ocr(pdf_path, log=log)

    return normalize_ocr_layout(full_text)


def smart_question_chunking(text: str, max_chars: int = 10000) -> List[str]:
    """
    Question-Aware Smart Chunking (Safely handles empty splits and question boundaries).
    """
    q_pattern = re.compile(r'(?=\n(?:Q|Q\.|Question|\d+[\.\)])\s*\d*)', re.IGNORECASE)
    raw_blocks = q_pattern.split(text)
    
    # Filter empty blocks
    blocks = [b for b in raw_blocks if b.strip()]
    
    chunks = []
    curr_chunk = ""

    for block in blocks:
        if len(curr_chunk) + len(block) > max_chars and curr_chunk.strip():
            chunks.append(curr_chunk.strip())
            curr_chunk = block
        else:
            curr_chunk += block

    if curr_chunk.strip():
        chunks.append(curr_chunk.strip())

    return chunks


# =========================================================
# 3. DIRECT EXTRACTOR WITH RETRY LOGIC & DEDUPLICATION
# =========================================================

class DirectQAExtractor:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_api_key("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing!")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"

    def _call_groq_with_retry(self, messages: list, retries: int = 3) -> str:
        """Rate limit error mitigation with exponential backoff retries."""
        for attempt in range(retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_completion_tokens=4096
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                err_str = str(e).lower()
                if "rate_limit" in err_str or "413" in err_str or "429" in err_str:
                    time.sleep(4 * (attempt + 1))
                else:
                    if attempt == retries - 1:
                        raise e
                    time.sleep(2)
        return "{}"

    def process(self, pdf_path: str, status_callback=None) -> List[Dict[str, Any]]:
        def log(msg):
            print(msg)
            if status_callback:
                status_callback(msg)

        log("1. PDF Text Extract & Normalizing...")
        normalized_text = get_pdf_text(pdf_path, log=log)

        if not normalized_text.strip():
            raise Exception("Failed to extract readable text from document.")

        log("2. Running Question-Aware Smart Chunking...")
        chunks = smart_question_chunking(normalized_text, max_chars=10000)
        log(f"Document chunked into {len(chunks)} safe block(s). Processing...")

        system_prompt = """You are a precise exam evaluator extracting Question and Answer pairs from OCR text.

STRICT RULES:
1. Extract ALL distinct questions and their complete student answers.
2. Ensure the answer captures the VERY FIRST SENTENCE and contains NO TRUNCATION.
3. Preserve math expressions, structural text, and all steps.

Return ONLY JSON:
{
  "qa_pairs": [
    {
      "question": "Full question text",
      "answer": "Complete student answer"
    }
  ]
}"""

        raw_all_pairs = []

        for idx, chunk in enumerate(chunks):
            log(f"Extracting Q&A from Chunk {idx + 1}/{len(chunks)}...")
            
            response_content = self._call_groq_with_retry([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"OCR TEXT SNIPPET:\n\n{chunk}"}
            ])

            try:
                data = json.loads(response_content)
                pairs = data.get("qa_pairs", [])
                for p in pairs:
                    q = str(p.get("question", "")).strip()
                    a = str(p.get("answer", "")).strip()
                    if q and a:
                        raw_all_pairs.append({"question": q, "answer": a, "matched": True})
            except Exception as e:
                log(f"Warning: JSON parse error in Chunk {idx + 1}: {e}")

            time.sleep(2.0)  # Throttling between chunks for safe RPM

        # Deduplication by Question text similarity
        final_qa_pairs = []
        seen_questions = set()

        for item in raw_all_pairs:
            q_clean = re.sub(r'[\W_]+', '', item["question"].lower())[:50]
            if q_clean not in seen_questions:
                seen_questions.add(q_clean)
                final_qa_pairs.append(item)

        log(f"Successfully extracted {len(final_qa_pairs)} clean Q&A pairs!")
        return final_qa_pairs


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
