import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import fitz  # PyMuPDF
from groq import Groq

# =========================================================
# 1. PYDANTIC SCHEMAS FOR STRUCTURED OUTPUT
# =========================================================

class QAPair(BaseModel):
    question_number: Optional[str] = Field(
        default=None, 
        description="Question or Sub-question number if present (e.g., 'Q1', '1(a)', '2.i')"
    )
    question: str = Field(
        description="The exact full question text extracted verbatim from the chunk."
    )
    answer: str = Field(
        description="The complete, exact answer text corresponding to the question. Do not summarize or truncate."
    )

class ExtractedQAList(BaseModel):
    items: List[QAPair] = Field(
        default_factory=list,
        description="List of all question-answer pairs found in the given text chunk."
    )


# =========================================================
# 2. TEXT CHUNKING WITH OVERLAP
# =========================================================

def chunk_text_with_overlap(
    text: str, 
    chunk_size: int = 3500, 
    overlap: int = 200
) -> List[str]:
    """
    Splits text into chunks of `chunk_size` characters with `overlap` characters.
    Tries to split at sentence/newline boundaries to avoid breaking sentences midway.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to break at a newline or period boundary within the last 200 chars of chunk
        boundary = text.rfind('\n', start + chunk_size - 200, end)
        if boundary == -1:
            boundary = text.rfind('. ', start + chunk_size - 200, end)

        if boundary != -1 and boundary > start:
            actual_end = boundary + 1
        else:
            actual_end = end

        chunk = text[start:actual_end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start forward, backing off by overlap amount
        start = actual_end - overlap
        if start < 0:
            start = 0

    return chunks


# =========================================================
# 3. STRICT SYSTEM PROMPT
# =========================================================

STRICT_QA_SYSTEM_PROMPT = """You are a high-precision document extraction system.
Your sole job is to extract Question and Answer pairs from the provided text chunk with 100% accuracy.

CRITICAL MANDATORY RULES:
1. VERBATIM EXTRACTION ONLY:
   - Extract questions and answers EXACTLY as they appear in the source text.
   - DO NOT rephrase, summarize, condense, clean up grammar, or alter a single word.
2. NO SKIPPING / NO TRUNCATION:
   - Extract the COMPLETE answer from start to end.
   - Do NOT drop introductory sentences, concluding lines, or sub-points.
3. NO MERGING:
   - Every question/sub-question (e.g., 1(a), 1(b)) MUST be extracted as an individual object in the list.
   - NEVER combine two separate answers into one object.
4. HANDLE OVERLAPS & PARTIAL TEXT:
   - If a sentence or answer started in a previous chunk or ends midway due to chunk boundary, extract whatever complete or partial text is available in THIS chunk without inventing text.
5. IF NO QA PAIRS EXIST:
   - Return an empty list `{"items": []}` if no questions or answers are found.
"""


# =========================================================
# 4. CHUNK-BY-LOOP EXTRACTION ENGINE
# =========================================================

class PDFQAExtractor:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable or argument is required.")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"

    def extract_from_pdf(
        self, 
        pdf_path: str, 
        chunk_size: int = 3500, 
        overlap: int = 200,
        status_callback=None
    ) -> List[Dict[str, Any]]:
        
        def log(msg):
            print(msg)
            if status_callback:
                status_callback(msg)

        log(f"Reading PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        full_text_pages = []
        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text("text")
            if page_text.strip():
                full_text_pages.append(page_text)
        doc.close()

        full_text = "\n\n".join(full_text_pages)
        if not full_text.strip():
            log("No text extracted from PDF. (Might be an image-only PDF needing OCR)")
            return []

        # Step 1: Chunking with Overlap
        log(f"Total Text Length: {len(full_text)} characters.")
        chunks = chunk_text_with_overlap(full_text, chunk_size=chunk_size, overlap=overlap)
        log(f"Split into {len(chunks)} chunks with size ~{chunk_size} and overlap {overlap}.")

        # Step 2: Loop Processing
        all_extracted_items: List[QAPair] = []

        for idx, chunk in enumerate(chunks):
            log(f"Processing Chunk {idx + 1}/{len(chunks)} ({len(chunk)} chars)...")
            
            user_prompt = f"Extract all Question and Answer pairs from this text chunk:\n\n---\n{chunk}\n---"

            try:
                # Utilizing Structured Output with Pydantic JSON Schema
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": STRICT_QA_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={
                        "type": "json_object",
                        "schema": ExtractedQAList.model_json_schema()
                    },
                    temperature=0.0
                )

                response_content = completion.choices[0].message.content
                parsed_response = ExtractedQAList.model_validate_json(response_content)
                
                log(f"Chunk {idx + 1}: Found {len(parsed_response.items)} QA pairs.")
                all_extracted_items.extend(parsed_response.items)

            except Exception as e:
                log(f"Error processing Chunk {idx + 1}: {e}")
                time.sleep(1)  # Brief pause on error

        # Step 3: Deduplication & Clean Up
        log("Deduplicating cross-chunk overlapping QA pairs...")
        final_qa_list = self._deduplicate_qa_pairs(all_extracted_items)
        log(f"Final Count after Deduplication: {len(final_qa_list)} QA pairs.")

        return [item.model_dump() for item in final_qa_list]

    def _deduplicate_qa_pairs(self, items: List[QAPair]) -> List[QAPair]:
        """
        Merges duplicate or partial QA pairs that were extracted twice across overlapping chunks.
        """
        if not items:
            return []

        unique_items: List[QAPair] = []

        for current in items:
            if not current.question.strip():
                continue

            matched = False
            for idx, existing in enumerate(unique_items):
                # Clean up string comparisons
                q1 = re.sub(r'\s+', ' ', current.question.strip().lower())
                q2 = re.sub(r'\s+', ' ', existing.question.strip().lower())

                # If question texts are substantially similar/identical
                if q1 in q2 or q2 in q1 or self._similarity(q1, q2) > 0.85:
                    matched = True
                    # Take the longer answer (covers overlap boundary truncation)
                    if len(current.answer.strip()) > len(existing.answer.strip()):
                        unique_items[idx] = current
                    break

            if not matched:
                unique_items.append(current)

        return unique_items

    @staticmethod
    def _similarity(s1: str, s2: str) -> float:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1, s2).ratio()


# =========================================================
# 5. EXECUTION & UTILITY
# =========================================================

if __name__ == "__main__":
    # Specify PDF Path
    PDF_FILE_PATH = "sample_test.pdf"

    if os.path.exists(PDF_FILE_PATH):
        extractor = PDFQAExtractor()
        results = extractor.extract_from_pdf(
            pdf_path=PDF_FILE_PATH,
            chunk_size=3500,
            overlap=200
        )

        # Save Output to JSON
        output_file = "extracted_qa_pairs.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nExtraction Completed! Results saved to '{output_file}'")
    else:
        print(f"File '{PDF_FILE_PATH}' not found. Please update path.")
