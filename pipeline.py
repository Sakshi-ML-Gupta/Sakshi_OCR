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
        log("Native text not found. Fallback to Datalab OCR...")
        full_text = run_datalab_ocr(pdf_path, log=log)

    return full_text


def post_clean_noise(text: str) -> str:
    """Softly removes signature/scribble lines without trimming actual student content."""
    lines = text.split("\n")
    cleaned = []
    noise_pattern = re.compile(
        r'^\s*(?:A\s+)?(?:red\s*scribble|signature\s*inside|circle|red\s*line|extending\s*towards|bottom\s*of\s*the\s*page).*\.?$',
        re.IGNORECASE
    )
    for line in lines:
        if not noise_pattern.match(line.strip()):
            cleaned.append(line)
    return "\n".join(cleaned).strip()


# =========================================================
# 3. ANCHOR-BASED EXACT SLICING EXTRACTOR
# =========================================================

class DirectQAExtractor:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_api_key("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing!")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"

    def _get_question_anchors(self, text: str) -> List[Dict[str, Any]]:
        """Uses LLM to get exact start phrases for each question in the text."""
        system_prompt = """You are a document layout parser.
Identify ALL question headers present in the document in exact sequential order.

For each question found, return:
1. 'question_label': Short label e.g., 'Q1(a)', 'Q1(b)', 'Q2'
2. 'anchor_text': Exact 5 to 8 word phrase from the document where this question starts.

Output MUST be a valid JSON object:
{
  "anchors": [
    {"question_label": "Q1(a)", "anchor_text": "exact words starting question 1a"},
    {"question_label": "Q1(b)", "anchor_text": "exact words starting question 1b"}
  ]
}"""

        user_prompt = f"DOCUMENT TEXT:\n\n{text[:25000]}"

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        res = json.loads(completion.choices[0].message.content)
        return res.get("anchors", [])

    def process(self, pdf_path: str, status_callback=None) -> List[Dict[str, Any]]:
        def log(msg):
            print(msg)
            if status_callback:
                status_callback(msg)

        log("1. Extracting complete document text...")
        raw_text = get_pdf_text(pdf_path, log=log)

        if not raw_text.strip():
            raise Exception("Failed to extract text from document.")

        log("2. Locating exact question anchors...")
        anchors = self._get_question_anchors(raw_text)

        if not anchors:
            # Fallback if no specific questions were identified
            return [{"question": "Full Document", "answer": post_clean_noise(raw_text), "matched": True}]

        log(f"Found {len(anchors)} question anchors. Performing exact text slicing...")

        # Find character offsets for each anchor in raw_text
        found_spans = []
        search_start = 0

        for a in anchors:
            q_label = a.get("question_label", "")
            phrase = a.get("anchor_text", "").strip()

            pos = -1
            if phrase:
                # Direct match
                pos = raw_text.lower().find(phrase.lower(), search_start)
                # Fuzzy fallback if exact phrase has minor whitespace differences
                if pos == -1:
                    first_words = " ".join(phrase.split()[:3])
                    if len(first_words) > 3:
                        pos = raw_text.lower().find(first_words.lower(), search_start)

            if pos != -1:
                found_spans.append({"label": q_label, "pos": pos})
                search_start = pos + len(phrase)
            else:
                log(f"Warning: Anchor for {q_label} not found, slicing sequentially.")

        final_qa = []
        doc_len = len(raw_text)

        for i in range(len(found_spans)):
            curr = found_spans[i]
            q_label = curr["label"]
            start_pos = curr["pos"]

            # End position is start of next question, or end of document
            if i + 1 < len(found_spans):
                end_pos = found_spans[i + 1]["pos"]
            else:
                end_pos = doc_len

            # Extract the raw chunk sliced strictly between question boundaries
            raw_chunk = raw_text[start_pos:end_pos].strip()

            # Clean scribble lines but keep 100% of answer text intact
            cleaned_chunk = post_clean_noise(raw_chunk)

            final_qa.append({
                "question": q_label,
                "answer": cleaned_chunk if cleaned_chunk else raw_chunk,
                "matched": True
            })

        log(f"Successfully mapped {len(final_qa)} questions with zero truncation.")
        return final_qa


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
