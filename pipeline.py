import os
import re
import io
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
# 2. SCHEMAS
# =========================================================

class QuestionItemSchema(BaseModel):
    question_number: Optional[str] = Field(
        default="", 
        alias="question",
        description="Question number e.g., '1' or 'Q1'"
    )
    sub_question: Optional[str] = Field(
        default="", 
        description="Sub-question identifier e.g., 'a', 'i', or '(a)'"
    )
    text: str = Field(
        description="The actual question text string"
    )

    class Config:
        populate_by_name = True

class QuestionExtractionSchema(BaseModel):
    questions: List[QuestionItemSchema] = Field(
        description="List of all individual questions extracted in exact order."
    )

class TargetLineSchema(BaseModel):
    found: bool = Field(
        description="True if the student's answer start line for the specified question is found."
    )
    start_line_index: Optional[int] = Field(
        default=None,
        description="The EXACT line index where the student BEGINS answering this question."
    )


# =========================================================
# 3. OCR FALLBACK FOR SCANNED / HANDWRITTEN PDFS
# =========================================================

DATALAB_BASE_URL = "https://www.datalab.to"

def run_datalab_ocr(pdf_path: str, log=print) -> str:
    api_key = get_api_key("DATALAB_API_KEY")
    if not api_key:
        raise Exception("DATALAB_API_KEY missing! Required for scanned PDF text extraction.")

    log("Scanned PDF detected. Submitting to Datalab OCR...")
    
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
        raise Exception(f"Datalab submit error {resp.status_code}: {resp.text}")

    data = resp.json()
    check_url = data["request_check_url"]

    log("Polling OCR engine for results...")
    for _ in range(150):
        poll_resp = httpx.get(check_url, headers=headers, timeout=60)
        if poll_resp.status_code == 200:
            result = poll_resp.json()
            if result.get("status") == "complete":
                return result.get("markdown") or ""
            if result.get("status") == "failed":
                raise Exception(f"Datalab conversion failed: {result.get('error')}")
        time.sleep(2)

    raise Exception("Datalab OCR conversion timed out.")


def extract_numbered_lines_from_pdf(pdf_path: str, log=print) -> List[str]:
    doc = fitz.open(pdf_path)
    lines = []
    
    noise_re = re.compile(
        r'(?:Teacher\'?s?\s*Signature|PAGE\s*NO|DATE\b|Neel?\s*Kamal|TAKMA\s*SINAN|^\s*\d{1,3}\s*$)',
        re.IGNORECASE
    )

    for page in doc:
        text = page.get_text("text")
        for line in text.split("\n"):
            line_str = line.strip()
            if line_str and not noise_re.search(line_str):
                lines.append(line_str)
                
    doc.close()

    if not lines:
        log("No selectable text found in PDF. Triggering OCR engine...")
        ocr_text = run_datalab_ocr(pdf_path, log=log)
        for line in ocr_text.split("\n"):
            line_str = line.strip()
            if line_str and not noise_re.search(line_str):
                lines.append(line_str)

    return lines


# =========================================================
# 4. HYBRID STRICT SEARCH MAPPING ENGINE
# =========================================================

class DirectQAExtractor:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_api_key("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing!")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"

    def extract_all_questions(self, full_text: str) -> List[str]:
        system_prompt = """You are an exam structure analyzer.
Extract all distinct questions and sub-questions (e.g., Q1, 1(a), Q2) in exact printed order.

For each item, populate:
- 'question': Main question number/label (e.g., '1')
- 'sub_question': Sub-question label if present (e.g., 'a')
- 'text': The full question prompt/text

Return strictly valid JSON according to schema."""

        user_prompt = f"Extract all questions from this text:\n\n{full_text[:8000]}"

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={
                "type": "json_object",
                "schema": QuestionExtractionSchema.model_json_schema()
            },
            temperature=0.0
        )
        
        parsed = QuestionExtractionSchema.model_validate_json(completion.choices[0].message.content)
        
        formatted_questions = []
        for q in parsed.questions:
            q_num = (q.question_number or "").strip()
            sub_q = (q.sub_question or "").strip()
            q_text = (q.text or "").strip()
            
            prefix = ""
            if q_num and sub_q:
                prefix = f"Q{q_num}({sub_q})"
            elif q_num:
                prefix = f"Q{q_num}"
            elif sub_q:
                prefix = f"({sub_q})"
                
            full_str = f"{prefix} {q_text}".strip()
            if full_str:
                formatted_questions.append(full_str)
                
        return formatted_questions

    def _find_header_line(self, target_q: str, lines: List[str], start_idx: int) -> int:
        """Fast Deterministic Search: Answer sheets me Q1, Ans 1, 1(a) patterns dhoondta hai."""
        match = re.search(r'(?:Q(?:uestion)?\s*|\b)?(\d+(?:\([a-z0-9]+\)|[a-z])?)', target_q, re.IGNORECASE)
        if not match:
            return -1

        q_id = match.group(1).lower().replace("(", r"\(").replace(")", r"\)")
        
        patterns = [
            rf'^\s*(?:ans(?:wer)?|q(?:uestion)?)\s*[:.\-]?\s*{q_id}\b',
            rf'^\s*{q_id}\s*[:.\-]',
            rf'\banswer\s+to\s+q(?:uestion)?\s*[:.\-]?\s*{q_id}\b'
        ]

        for idx in range(start_idx, len(lines)):
            line_str = lines[idx].lower()
            for pat in patterns:
                if re.search(pat, line_str, re.IGNORECASE):
                    return idx
        return -1

    def _llm_search_start_line(self, target_q: str, lines: List[str], start_idx: int) -> int:
        """LLM Fallback: Heading marker na milne par contextual search karta hai."""
        chunk = lines[start_idx:start_idx + 120]
        if not chunk:
            return -1

        formatted_block = "\n".join([f"[{start_idx + i}] {text}" for i, text in enumerate(chunk)])
        
        system_prompt = """You are a line matching assistant.
Locate the line index where the student BEGINS answering the TARGET QUESTION.
Return `found: true` and the exact line number `start_line_index` if present."""

        user_prompt = f"TARGET QUESTION: {target_q}\n\nANSWER SHEET LINES:\n{formatted_block}"

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={
                    "type": "json_object",
                    "schema": TargetLineSchema.model_json_schema()
                },
                temperature=0.0
            )

            res = TargetLineSchema.model_validate_json(completion.choices[0].message.content)
            if res.found and res.start_line_index is not None and res.start_line_index >= start_idx:
                return res.start_line_index
        except Exception:
            pass

        return -1

    def process(self, pdf_path: str, status_callback=None) -> List[Dict[str, Any]]:
        def log(msg):
            print(msg)
            if status_callback:
                status_callback(msg)

        log("1. Extracting text lines from PDF...")
        lines = extract_numbered_lines_from_pdf(pdf_path, log=log)
        
        if not lines:
            raise Exception("No text lines could be extracted.")

        full_doc_text = "\n".join(lines)

        log("2. Extracting Target Questions...")
        questions = self.extract_all_questions(full_doc_text)
        log(f"Extracted {len(questions)} target questions.")

        log("3. Sequential Answer Boundaries Resolution...")
        starts = []
        cursor = 0

        for idx, q in enumerate(questions):
            log(f"Mapping Q{idx+1}: {q[:30]}...")
            
            # Step A: Deterministic Pattern Match
            found_idx = self._find_header_line(q, lines, cursor)
            
            # Step B: LLM Fallback Match
            if found_idx == -1:
                found_idx = self._llm_search_start_line(q, lines, cursor)

            if found_idx != -1:
                starts.append({"question": q, "start": found_idx})
                cursor = found_idx + 1
            else:
                starts.append({"question": q, "start": None})

        # Step C: Compute Math Boundaries & Prevent Overlaps
        final_qa_pairs = []
        num_q = len(starts)

        for i in range(num_q):
            q_text = starts[i]["question"]
            s_idx = starts[i]["start"]

            if s_idx is None:
                final_qa_pairs.append({
                    "question": q_text,
                    "answer": "Answer not found in answer sheet.",
                    "matched": False
                })
                continue

            # End boundary is the next valid answer start index minus 1
            e_idx = len(lines) - 1
            for j in range(i + 1, num_q):
                if starts[j]["start"] is not None and starts[j]["start"] > s_idx:
                    e_idx = starts[j]["start"] - 1
                    break

            if s_idx <= e_idx:
                raw_ans = " ".join(lines[s_idx:e_idx + 1]).strip()
                # Clean header prefix from answer body
                cleaned_ans = re.sub(
                    r'^\s*(?:ans(?:wer)?|q(?:uestion)?|\d+[\.\)]?)\s*[\d\(\)a-z]*[:.\-]?\s*', 
                    '', 
                    raw_ans, 
                    flags=re.IGNORECASE
                ).strip()

                final_qa_pairs.append({
                    "question": q_text,
                    "answer": cleaned_ans if cleaned_ans else raw_ans,
                    "matched": True
                })
            else:
                final_qa_pairs.append({
                    "question": q_text,
                    "answer": "Answer not found in answer sheet.",
                    "matched": False
                })

        return final_qa_pairs


# =========================================================
# 5. STREAMLIT APP WRAPPER
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
        ocr_json = {"total_pages": 1, "status": "Strict Boundaries Mapped Successfully"}
        return ocr_json, qa_pairs
    finally:
        if file_bytes and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
