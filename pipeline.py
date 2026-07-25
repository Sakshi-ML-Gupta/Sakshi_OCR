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
        description="List of all individual questions and sub-questions extracted in exact order from the paper."
    )

class MappingItem(BaseModel):
    question_text: str = Field(description="The exact question text supplied.")
    start_line_index: int = Field(description="0-based line index where the answer starts. Return -1 if not found.")
    end_line_index: int = Field(description="0-based line index where the answer ends. Return -1 if not found.")


class AnswerMappingResult(BaseModel):
    mappings: List[MappingItem] = Field(description="Mapped question and answer boundaries.")


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
# 4. DIRECT GLOBAL MAPPER
# =========================================================

class DirectQAExtractor:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_api_key("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing!")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"

    def extract_all_questions(self, full_text: str) -> List[str]:
        system_prompt = """You are an exam paper structure analyzer.
Extract EVERY question and sub-question (e.g., 1(a), 1(b), Q2, Q3.i) in exact order as printed.

For each item, populate:
- 'question': Main question number (e.g. '1')
- 'sub_question': Sub question label if present (e.g. 'a')
- 'text': The full wording of the question

Return strictly valid JSON according to the schema."""

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
        log(f"Extracted {len(questions)} distinct questions.")

        log("3. Mapping answers via Full-Context Model Alignment...")
        
        indexed_lines = "\n".join([f"[{idx}] {text}" for idx, text in enumerate(lines)])
        questions_block = "\n".join([f"- {q}" for q in questions])

        system_prompt = """You are an accurate Answer Sheet Mapper.
Given a list of QUESTIONS and a line-numbered STUDENT ANSWER SHEET:
For EVERY question, locate the exact `start_line_index` and `end_line_index` in the line-numbered text.

Rules:
1. Do NOT guess or hallucinate line numbers. Look at the content carefully.
2. An answer starts where student begins answering that question and ends before the next answer starts.
3. If an answer to a question is NOT written in the sheet, set `start_line_index: -1` and `end_line_index: -1`.
4. Return strictly JSON adhering to the schema."""

        user_prompt = f"QUESTIONS:\n{questions_block}\n\nSTUDENT ANSWER SHEET LINES:\n{indexed_lines}"

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={
                    "type": "json_object",
                    "schema": AnswerMappingResult.model_json_schema()
                },
                temperature=0.0
            )

            result = AnswerMappingResult.model_validate_json(completion.choices[0].message.content)
            
            final_qa_pairs = []
            for item in result.mappings:
                s_idx = item.start_line_index
                e_idx = item.end_line_index

                if s_idx != -1 and e_idx != -1 and 0 <= s_idx <= e_idx < len(lines):
                    ans_text = " ".join(lines[s_idx:e_idx + 1]).strip()
                    # Clean up question prefixes from answer
                    ans_text = re.sub(r'^\s*(?:Ans(?:wer)?\s*\d*\s*[.:\-]?|उत्तर\s*\d*|Q\.?\s*\d+)\s*', '', ans_text, flags=re.IGNORECASE).strip()
                    matched = True
                else:
                    ans_text = "Answer not found in answer sheet."
                    matched = False

                final_qa_pairs.append({
                    "question": item.question_text,
                    "answer": ans_text,
                    "matched": matched
                })

            return final_qa_pairs

        except Exception as e:
            log(f"Error mapping answers: {e}")
            raise e


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
        ocr_json = {"total_pages": 1, "status": "Direct Full-Context Mapped"}
        return ocr_json, qa_pairs
    finally:
        if file_bytes and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
