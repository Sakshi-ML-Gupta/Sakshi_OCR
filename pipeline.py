import os
import re
import json
import time
import tempfile
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import fitz  # PyMuPDF
from groq import Groq

# =========================================================
# 1. API KEY SETUP
# =========================================================

def get_api_key(name: str = "GROQ_API_KEY") -> str:
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
# 2. PYDANTIC SCHEMAS FOR STRICT SINGLE-TARGET SEARCH
# =========================================================

class QuestionExtractionSchema(BaseModel):
    questions: List[str] = Field(
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
# 3. OCR & LINE PREPARATION
# =========================================================

def extract_numbered_lines_from_pdf(pdf_path: str) -> List[str]:
    """
    Extracts text from PDF page by page, cleans up noise, and returns 
    a flat list of indexed lines representing the student's answer sheet.
    """
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
    return lines


# =========================================================
# 4. SEQUENTIAL SINGLE-TARGET SEARCH ENGINE
# =========================================================

class SequentialSearchExtractor:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_api_key("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing! Set it in Streamlit Secrets or .env file.")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"

    def extract_all_questions(self, full_text: str) -> List[str]:
        """
        Extracts all questions and sub-questions from the document in strict sequence.
        """
        system_prompt = """You are an exam document analyzer.
Extract EVERY question and sub-question (e.g., 1(a), 1(b), Q2, Q3.i) in strict order as printed.
Return strictly JSON adhering to the provided schema."""

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
        return [q.strip() for q in parsed.questions if q.strip()]

    def find_start_line_for_target(
        self, 
        target_question: str, 
        lines: List[str], 
        search_start_idx: int, 
        chunk_size: int = 150, 
        overlap: int = 20
    ) -> int:
        """
        Performs a forward-only single-target search for ONE question at a time.
        Returns the absolute line index where the answer starts.
        """
        system_prompt = """You are a line-matching assistant.
You are given a TARGET QUESTION and a line-numbered block of student answer text.
Find the EXACT line index where the student BEGINS answering this TARGET QUESTION (e.g., "Ans 1", "Q1.", or topic heading).

STRICT RULES:
- ONLY locate the START line for the requested TARGET QUESTION.
- Do NOT worry about where the answer ends.
- Return `found: false` if this line block does not contain the beginning of the target question."""

        total_lines = len(lines)
        curr = search_start_idx

        while curr < total_lines:
            end_chunk = min(curr + chunk_size, total_lines)
            chunk_lines = lines[curr:end_chunk]
            
            # Format chunk with absolute index numbers
            formatted_block = "\n".join([f"[{curr + i}] {text}" for i, text in enumerate(chunk_lines)])
            
            user_prompt = f"TARGET QUESTION TO FIND:\n{target_question}\n\nANSWER SHEET LINES:\n{formatted_block}"

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
                    # Sanity check: Ensure returned index is valid and monotonic
                    if search_start_idx <= result.start_line_index < total_lines:
                        return result.start_line_index

            except Exception as e:
                print(f"Error during line search for target '{target_question[:20]}...': {e}")

            if end_chunk >= total_lines:
                break
                
            curr += (chunk_size - overlap)

        return -1  # Not found

    def process(self, pdf_path: str, status_callback=None) -> List[Dict[str, Any]]:
        def log(msg):
            print(msg)
            if status_callback:
                status_callback(msg)

        log("1. Extracting line-by-line text from PDF...")
        lines = extract_numbered_lines_from_pdf(pdf_path)
        if not lines:
            raise Exception("No text lines extracted from PDF.")

        full_doc_text = "\n".join(lines)

        log("2. Extracting Target Questions sequence...")
        questions = self.extract_all_questions(full_doc_text)
        log(f"Found {len(questions)} target questions.")

        log("3. Executing Sequential Single-Target Forward Search...")
        question_starts = []
        search_cursor = 0

        for idx, q in enumerate(questions):
            log(f"Searching start line for Q{idx+1}: '{q[:40]}...' (Search Cursor: Line {search_cursor})")
            
            start_idx = self.find_start_line_for_target(q, lines, search_cursor)
            
            if start_idx != -1:
                question_starts.append({"question": q, "start_line": start_idx})
                search_cursor = start_idx + 1  # Forward search guarantee: Never look backwards!
            else:
                log(f"⚠️ Start line not found for Q{idx+1}. Skipping target.")
                question_starts.append({"question": q, "start_line": None})

        log("4. Computing Math Boundaries (End = Next Start - 1) & Verbatim Slicing...")
        final_qa_pairs = []
        num_found = len(question_starts)

        for i in range(num_found):
            item = question_starts[i]
            q_text = item["question"]
            start = item["start_line"]

            if start is None:
                final_qa_pairs.append({
                    "question": q_text,
                    "answer": "",
                    "matched": False
                })
                continue

            # MATH COMPUTATION FOR END LINE (100% Deterministic - No LLM Hallucination)
            # Find the start line of the NEXT valid question
            end = len(lines) - 1
            for j in range(i + 1, num_found):
                if question_starts[j]["start_line"] is not None:
                    end = question_starts[j]["start_line"] - 1
                    break

            if start <= end:
                answer_text = " ".join(lines[start:end + 1]).strip()
                # Clean up leading question labels like "Ans 1.", "उत्तर 1", etc.
                answer_text = re.sub(r'^\s*(?:Ans(?:wer)?\s*\d*\s*[.:\-]?|उत्तर\s*\d*|Q\.?\s*\d+)\s*', '', answer_text, flags=re.IGNORECASE).strip()
            else:
                answer_text = lines[start].strip()

            final_qa_pairs.append({
                "question": q_text,
                "answer": answer_text,
                "matched": True
            })

        return final_qa_pairs


# =========================================================
# 5. WRAPPER FOR STREAMLIT APP (`app.py`) INTEGRATION
# =========================================================

def process_pdf(file_input, status_callback=None):
    """
    Direct drop-in replacement function for Streamlit (`from pipeline import process_pdf`).
    Handles file paths, bytes, or Streamlit UploadedFile objects automatically.
    """
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
        
        # Structure compatibility return for app.py
        ocr_json = {"total_pages": 1, "status": "Sequential Search Complete"}
        return ocr_json, qa_pairs
    finally:
        if file_bytes and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
