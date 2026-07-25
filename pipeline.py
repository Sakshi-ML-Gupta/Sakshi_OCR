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
# 2. SCHEMAS FOR SEQUENTIAL SEARCH
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

class TargetLineSchema(BaseModel):
    found: bool = Field(
        description="True if the student's answer start line for the specified question was found in the chunk."
    )
    start_line_index: Optional[int] = Field(
        default=None,
        description="The EXACT global integer line index where the student BEGINS answering this question."
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
# 4. SEQUENTIAL SEARCH EXTRACTOR (WITH REGEX MATCHING)
# =========================================================

class SequentialSearchExtractor:
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
                prefix = f"{q_num}({sub_q})"
            elif q_num:
                prefix = f"{q_num}"
            elif sub_q:
                prefix = f"({sub_q})"
                
            full_str = f"{prefix} {q_text}".strip()
            if full_str:
                formatted_questions.append(full_str)
                
        return formatted_questions

    def _regex_find_start(self, question_text: str, lines: List[str], search_start_idx: int) -> int:
        """Rule-based pattern anchor: Fast scans answer sheet lines for Q/Ans markers."""
        # Extract possible label, e.g., '1(a)', '1', '(a)'
        match = re.match(r'^(?:Q|Ans|Question|Answer)?\s*\(?(\d+[a-z]?|\d+|[a-z])\)?', question_text, re.I)
        if not match:
            return -1

        label = match.group(1).lower()
        patterns = [
            rf'^\s*(?:ans(?:wer)?|q(?:uestion)?)\s*\.?:?\s*\(?{re.escape(label)}\)?\b',
            rf'^\s*\(?{re.escape(label)}\)?\s*[.:\-]',
            rf'^\s*ans\s*{re.escape(label)}\b'
        ]

        for idx in range(search_start_idx, len(lines)):
            line_clean = lines[idx].lower().strip()
            for pat in patterns:
                if re.search(pat, line_clean, re.I):
                    return idx
        return -1

    def find_start_line_for_target(
        self, 
        target_question: str, 
        lines: List[str], 
        search_start_idx: int, 
        chunk_size: int = 150, 
        overlap: int = 20
    ) -> int:
        # Step A: Attempt fast Rule-Based Regex Matching
        regex_idx = self._regex_find_start(target_question, lines, search_start_idx)
        if regex_idx != -1:
            return regex_idx

        # Step B: LLM Smart Forward Search Fallback
        system_prompt = """You are a line-matching assistant.
Find the EXACT line index where the student BEGINS answering the TARGET QUESTION.
Look for answer headings like 'Ans 1', 'Q1.', topic headers, or opening sentences relevant to the target question.
Return `found: true` along with `start_line_index` if found in this block. Otherwise `found: false`."""

        total_lines = len(lines)
        curr = search_start_idx

        while curr < total_lines:
            end_chunk = min(curr + chunk_size, total_lines)
            chunk_lines = lines[curr:end_chunk]
            formatted_block = "\n".join([f"[{curr + i}] {text}" for i, text in enumerate(chunk_lines)])
            
            user_prompt = f"TARGET QUESTION:\n{target_question}\n\nANSWER LINES:\n{formatted_block}"

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

                result = TargetLineSchema.model_validate_json(completion.choices[0].message.content)

                if result.found and result.start_line_index is not None:
                    if search_start_idx <= result.start_line_index < total_lines:
                        return result.start_line_index

            except Exception as e:
                print(f"Error searching '{target_question[:20]}...': {e}")

            if end_chunk >= total_lines:
                break
                
            curr += (chunk_size - overlap)

        return -1

    def process(self, pdf_path: str, status_callback=None) -> List[Dict[str, Any]]:
        def log(msg):
            print(msg)
            if status_callback:
                status_callback(msg)

        log("1. Extracting text lines (PyMuPDF + OCR Fallback)...")
        lines = extract_numbered_lines_from_pdf(pdf_path, log=log)
        
        if not lines:
            raise Exception("No text lines could be extracted even after running OCR.")

        full_doc_text = "\n".join(lines)

        log("2. Extracting Target Questions sequence...")
        questions = self.extract_all_questions(full_doc_text)
        log(f"Found {len(questions)} target questions.")

        log("3. Executing Hybrid Regex + LLM Forward Search...")
        question_starts = []
        search_cursor = 0

        for idx, q in enumerate(questions):
            log(f"Searching Q{idx+1}: '{q[:35]}...' (Cursor: Line {search_cursor})")
            start_idx = self.find_start_line_for_target(q, lines, search_cursor)
            
            if start_idx != -1:
                question_starts.append({"question": q, "start_line": start_idx})
                search_cursor = start_idx + 1
            else:
                log(f"⚠️ Direct start line not matched for Q{idx+1}. Applying structural anchor.")
                question_starts.append({"question": q, "start_line": None})

        log("4. Resolving Fallbacks & Math Slicing (End = Next Start - 1)...")
        final_qa_pairs = []
        num_found = len(question_starts)

        # Fallback resolution: If a question start wasn't found, anchor it right after previous answer
        last_known_start = 0
        for i in range(num_found):
            if question_starts[i]["start_line"] is not None:
                last_known_start = question_starts[i]["start_line"]
            else:
                question_starts[i]["start_line"] = last_known_start

        for i in range(num_found):
            item = question_starts[i]
            q_text = item["question"]
            start = item["start_line"]

            # Compute boundary
            end = len(lines) - 1
            for j in range(i + 1, num_found):
                if question_starts[j]["start_line"] > start:
                    end = question_starts[j]["start_line"] - 1
                    break

            if start <= end:
                answer_text = " ".join(lines[start:end + 1]).strip()
                answer_text = re.sub(r'^\s*(?:Ans(?:wer)?\s*\d*\s*[.:\-]?|उत्तर\s*\d*|Q\.?\s*\d+)\s*', '', answer_text, flags=re.IGNORECASE).strip()
            else:
                answer_text = ""

            final_qa_pairs.append({
                "question": q_text,
                "answer": answer_text if answer_text else "Answer text not clearly identified in sheet.",
                "matched": True if answer_text else False
            })

        return final_qa_pairs


# =========================================================
# 5. STREAMLIT APP COMPATIBILITY WRAPPER
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
        extractor = SequentialSearchExtractor()
        qa_pairs = extractor.process(tmp_path, status_callback=status_callback)
        ocr_json = {"total_pages": 1, "status": "Processed with Hybrid Matcher & Sequential Search"}
        return ocr_json, qa_pairs
    finally:
        if file_bytes and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
