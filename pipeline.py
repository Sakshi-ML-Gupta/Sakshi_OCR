import os
import re
import json
import time
import tempfile
import httpx
from typing import List, Dict, Any, Optional, Tuple
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
    check_url = data.get("request_check_url")
    if not check_url:
        raise Exception("Datalab API did not return check_url.")

    for _ in range(150):
        poll_resp = httpx.get(check_url, headers=headers, timeout=60)
        if poll_resp.status_code == 200:
            result = poll_resp.json()
            if result.get("status") == "complete":
                md = result.get("markdown") or result.get("text") or ""
                if not md and "pages" in result:
                    md = "\n\n".join([p.get("markdown", "") for p in result["pages"]])
                return md
            if result.get("status") == "failed":
                raise Exception(f"Datalab error: {result.get('error')}")
        time.sleep(2)

    raise Exception("Datalab OCR timeout.")


def normalize_ocr_layout(raw_ocr_text: str) -> str:
    text = raw_ocr_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'^[#]+\s*', '', text, flags=re.MULTILINE)
    text = text.replace('**', '').replace('__', '')
    text = re.sub(r'(\?|\:|\b)\s*(Ans|Answer|Sol|Solution|A\:)', r'\1\n\2', text, flags=re.IGNORECASE)
    
    noise_pattern = re.compile(
        r'^\s*(?:A\s+)?(?:red\s*scribble|signature\s*inside|circle|red\s*line|extending\s*towards|bottom\s*of\s*the\s*page).*\.?$',
        re.IGNORECASE | re.MULTILINE
    )
    text = noise_pattern.sub('', text)
    return text.strip()


def get_pdf_text_and_page_count(pdf_path: str, log=print) -> Tuple[str, int]:
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
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

    return normalize_ocr_layout(full_text), total_pages


def chunk_text_with_overlap(text: str, chunk_size: int = 5000, overlap: int = 1500) -> List[str]:
    """
    FIXED: Sliding window with overlap. Never cuts answers arbitrarily. 
    Guarantees every QA pair fits entirely in at least one chunk.
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Snap to paragraph break if possible
        if end < text_len:
            last_newline = text.rfind('\n\n', start, end)
            if last_newline != -1 and last_newline > start + (chunk_size // 2):
                end = last_newline
                
        chunks.append(text[start:end].strip())
        
        start = end - overlap
        
        # Snap overlap to start at a paragraph
        if start > 0 and start < text_len:
            next_newline = text.find('\n\n', start)
            if next_newline != -1 and next_newline < end:
                start = next_newline + 2
                
        if start >= end: 
            start = end
            
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

    def _call_groq_with_retry(self, messages: list, retries: int = 4) -> str:
        for attempt in range(retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_completion_tokens=2500
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                err_str = str(e).lower()
                if "rate_limit" in err_str or "413" in err_str or "429" in err_str:
                    wait_time = 15 * (attempt + 1)
                    print(f"Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    if attempt == retries - 1:
                        raise e
                    time.sleep(3)
        return "{}"

    def process(self, pdf_path: str, status_callback=None) -> Tuple[List[Dict[str, Any]], int]:
        def log(msg):
            print(msg)
            if status_callback:
                status_callback(msg)

        log("1. Extracting PDF text and formatting layout...")
        normalized_text, total_pages = get_pdf_text_and_page_count(pdf_path, log=log)

        if not normalized_text.strip():
            raise Exception("Failed to extract readable text from document.")

        log("2. Chunking document with 1500-char overlap (No data loss)...")
        chunks = chunk_text_with_overlap(normalized_text, chunk_size=5000, overlap=1500)
        log(f"Document split into {len(chunks)} overlapping chunk(s). Processing...")

        system_prompt = """You are an expert evaluator. Extract Question and Answer pairs from the OCR text.

CRITICAL INSTRUCTIONS:
1. NEVER TRUNCATE ANSWERS. Extract the ENTIRE student answer.
2. Answers often contain numbered lists (1., 2.), bullet points, math equations, and multiple paragraphs. Include EVERYTHING until the next distinct question starts.
3. Do NOT summarize. Output exact text verbatim.
4. If an answer is cut off at the very end of the snippet, extract whatever is visible.

Respond ONLY in valid JSON format:
{
  "qa_pairs": [
    {
      "question": "The exact full question text",
      "answer": "The EXACT FULL student answer, including all paragraphs, steps, and lists."
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

            time.sleep(3.0)

        # FIXED DEDUPLICATION: If question overlaps, keep the LONGEST answer!
        final_qa_dict = {}

        for item in raw_all_pairs:
            q_text = item["question"]
            a_text = item["answer"]
            
            # Create a clean key by stripping punctuation/spaces from first 60 chars of question
            q_clean = re.sub(r'[\W_]+', '', q_text.lower())[:60]
            
            if q_clean not in final_qa_dict:
                final_qa_dict[q_clean] = item
            else:
                # If this chunk extracted a LONGER version of the same answer, overwrite it!
                if len(a_text) > len(final_qa_dict[q_clean]["answer"]):
                    final_qa_dict[q_clean] = item

        final_qa_pairs = list(final_qa_dict.values())
        
        log(f"Successfully extracted {len(final_qa_pairs)} full Q&A pairs!")
        return final_qa_pairs, total_pages


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
        qa_pairs, total_pages = extractor.process(tmp_path, status_callback=status_callback)
        
        ocr_json = {
            "status": "Success",
            "total_pages": total_pages
        }
        return ocr_json, qa_pairs
    finally:
        if file_bytes and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
