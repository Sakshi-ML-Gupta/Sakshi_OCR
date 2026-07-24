import os
import io
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import fitz
import httpx

# =========================================================
# API KEYS
# =========================================================

def get_api_key(name: str) -> Optional[str]:
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv(name)


# =========================================================
# INPUT NORMALIZATION
# =========================================================

def _normalize_file_input(file_input, default_name: str = "document.pdf"):
    if isinstance(file_input, tuple):
        if len(file_input) < 2:
            raise ValueError(f"Tuple must have at least (filename, bytes), got {len(file_input)}")
        name, data = file_input[0], file_input[1]
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"Expected bytes, got {type(data).__name__}")
        return bytes(data), _coerce_name(name, default_name)

    if isinstance(file_input, (bytes, bytearray)):
        return bytes(file_input), default_name

    if isinstance(file_input, (str, Path)):
        p = Path(file_input)
        return p.read_bytes(), p.name

    if hasattr(file_input, "read"):
        data = file_input.read()
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"file_input.read() returned {type(data).__name__}, expected bytes")
        name = getattr(file_input, "name", default_name)
        return bytes(data), _coerce_name(name, default_name)

    raise TypeError(f"Unsupported file_input type: {type(file_input).__name__}")


def _coerce_name(name, default_name: str = "document.pdf") -> str:
    if not name or isinstance(name, (tuple, list)):
        return default_name
    try:
        return Path(str(name)).name or default_name
    except Exception:
        return default_name


# =========================================================
# OCR - DATALAB
# =========================================================

def run_ocr(file_content: bytes, file_name: str, status_callback=None) -> List[Dict]:
    """Run OCR using Datalab."""
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    api_key = get_api_key("DATALAB_API_KEY")
    if not api_key:
        raise Exception("DATALAB_API_KEY not found")

    headers = {"X-API-Key": api_key}
    log(f"Submitting to Datalab OCR...")

    resp = httpx.post(
        "https://www.datalab.to/api/v1/convert",
        headers=headers,
        files={"file": (file_name, file_content, "application/pdf")},
        data={"output_format": "markdown", "mode": "accurate", "paginate": "true"},
        timeout=120
    )

    if resp.status_code != 200:
        raise Exception(f"Datalab error {resp.status_code}: {resp.text}")

    data = resp.json()
    if not data.get("success", True):
        raise Exception(f"Datalab failed: {data.get('error')}")

    check_url = data["request_check_url"]
    log("Polling for OCR result...")

    result = None
    for attempt in range(150):
        poll_resp = httpx.get(check_url, headers=headers, timeout=60)
        if poll_resp.status_code != 200:
            raise Exception(f"Poll error {poll_resp.status_code}")

        result = poll_resp.json()
        status = result.get("status")

        if status == "complete":
            log("OCR complete")
            break
        if status == "failed" or result.get("error"):
            raise Exception(f"OCR failed: {result.get('error')}")
        time.sleep(2)
    else:
        raise Exception("OCR timed out")

    markdown = result.get("markdown", "")
    if not markdown.strip():
        raise Exception("Empty OCR output")

    # Parse pages
    pages = []
    
    # Try page markers
    markers = [
        r'\n\s*\{(\d+)\}\s*-{3,}\s*\n',
        r'\n\s*-{2,}\{(\d+)\}\s*-{2,}\s*\n',
        r'\n\s*\[PAGE\s*(\d+)\]\s*\n',
        r'\n\s*Page\s*(\d+)\s*\n',
    ]
    
    for pattern in markers:
        matches = list(re.finditer(pattern, markdown, re.IGNORECASE))
        if matches:
            parts = []
            start = 0
            for m in matches:
                parts.append(markdown[start:m.start()].strip())
                start = m.end()
            parts.append(markdown[start:].strip())
            parts = [p for p in parts if p]
            if len(parts) > 1:
                for idx, text in enumerate(parts):
                    pages.append({"page_number": idx + 1, "raw_text": text})
                break
    
    # Fallback: single page
    if not pages:
        pages = [{"page_number": 1, "raw_text": markdown.strip()}]
        log("WARNING: Could not detect page boundaries, treating as single page")

    log(f"OCR done: {len(pages)} pages")
    return pages


# =========================================================
# SIMPLE QUESTION EXTRACTION - REGEX FIRST, LLM ONLY IF NEEDED
# =========================================================

def extract_questions(pages: List[Dict], status_callback=None) -> Tuple[List[int], List[str]]:
    """Extract questions using regex first, LLM fallback."""
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    if not pages:
        return [], []

    # STEP 1: Try to find question pages using simple heuristics
    question_pages = []
    all_text = ""
    
    for i, page in enumerate(pages):
        text = page["raw_text"]
        # Check if page has numbered questions
        if re.search(r'(?m)^\s*\d+[\.\)]\s+[A-Za-z]', text):
            question_pages.append(i)
            all_text += text + "\n\n"
    
    if not question_pages:
        # If no clear question pages, check all pages
        for i, page in enumerate(pages):
            if len(page["raw_text"]) < 2000 and re.search(r'\d+[\.\)]', page["raw_text"]):
                question_pages.append(i)
                all_text += page["raw_text"] + "\n\n"
    
    # STEP 2: Extract questions using regex
    questions = []
    if all_text:
        # Pattern to match numbered questions
        pattern = r'(?m)^\s*(\d+[\.\)]\s*[^\n]+(?:\n\s+[^\n\d]+)*)'
        matches = re.findall(pattern, all_text)
        if matches:
            questions = [m.strip() for m in matches]
            log(f"Regex extracted {len(questions)} questions")
    
    # STEP 3: If regex fails, use LLM
    if not questions and question_pages:
        try:
            from groq import Groq
            api_key = get_api_key("GROQ_API_KEY")
            if api_key:
                client = Groq(api_key=api_key)
                
                # Take first 3 question pages or first 5000 chars
                qp_text = all_text[:8000]
                
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": "Extract all numbered questions from the text. Return JSON: {\"questions\": [\"1. Question text\", \"2. Question text\"]}"},
                        {"role": "user", "content": f"Questions text:\n{qp_text}"}
                    ],
                    temperature=0.0,
                )
                
                content = response.choices[0].message.content
                # Try to parse JSON
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    questions = data.get("questions", [])
                    log(f"LLM extracted {len(questions)} questions")
        except Exception as e:
            log(f"LLM fallback failed: {e}")
    
    # Clean up questions
    cleaned = []
    seen = set()
    for q in questions:
        # Remove duplicate question numbers
        q_clean = re.sub(r'^\d+[\.\)]\s*', '', q).strip()
        if q_clean and q_clean not in seen:
            seen.add(q_clean)
            cleaned.append(q)
    
    log(f"Final: {len(question_pages)} question pages, {len(cleaned)} questions")
    return question_pages, cleaned


# =========================================================
# SIMPLE ANSWER MAPPING - WITHOUT LLM
# =========================================================

def map_answers_simple(answer_lines: List[str], questions: List[str]) -> List[Dict]:
    """Simple answer mapping using text matching."""
    
    results = []
    
    for q_idx, question in enumerate(questions):
        # Extract question number (e.g., "1.", "2.")
        q_num_match = re.match(r'^\s*(\d+)', question)
        q_num = q_num_match.group(1) if q_num_match else str(q_idx + 1)
        
        # Find where this answer starts
        start_idx = None
        for i, line in enumerate(answer_lines):
            # Look for answer marker like "Ans 1", "Answer 1", "1.", "Q1."
            if re.search(rf'(?:Ans|Answer|उत्तर|प्र)\s*{q_num}[\.\s:-]', line, re.IGNORECASE):
                start_idx = i
                break
            # Or if line contains the question number and looks like an answer
            if re.search(rf'\b{q_num}[\.\)]\s+[A-Za-z]', line):
                start_idx = i
                break
        
        if start_idx is not None:
            # Find where next answer starts
            next_q_num = str(q_idx + 2)
            end_idx = len(answer_lines)
            for i in range(start_idx + 1, len(answer_lines)):
                if re.search(rf'(?:Ans|Answer|उत्तर|प्र)\s*{next_q_num}[\.\s:-]', answer_lines[i], re.IGNORECASE):
                    end_idx = i
                    break
                if re.search(rf'\b{next_q_num}[\.\)]\s+[A-Za-z]', answer_lines[i]):
                    end_idx = i
                    break
            
            # Extract answer text
            answer_text = " ".join(answer_lines[start_idx:end_idx]).strip()
            
            results.append({
                "question": question,
                "answer": answer_text,
                "matched": True
            })
        else:
            results.append({
                "question": question,
                "answer": "",
                "matched": False
            })
    
    return results


# =========================================================
# HYBRID ANSWER MAPPING - LLM ONLY FOR TOUGH CASES
# =========================================================

def map_answers_hybrid(answer_lines: List[str], questions: List[str], 
                       status_callback=None) -> List[Dict]:
    """Map answers using regex first, LLM for tough cases."""
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    
    # First try simple matching
    results = map_answers_simple(answer_lines, questions)
    matched_count = sum(1 for r in results if r["matched"])
    
    # If less than 50% matched, use LLM for the rest
    if matched_count < len(questions) * 0.5:
        log(f"Simple matching found only {matched_count}/{len(questions)}, using LLM...")
        
        try:
            from groq import Groq
            api_key = get_api_key("GROQ_API_KEY")
            if api_key:
                client = Groq(api_key=api_key)
                
                # Get unmatched questions
                unmatched = [r for r in results if not r["matched"]]
                if unmatched:
                    # Build numbered lines for LLM
                    numbered_lines = [f"[{i}] {line}" for i, line in enumerate(answer_lines)]
                    text = "\n".join(numbered_lines[:2000])  # Limit context
                    
                    for r in unmatched:
                        q = r["question"]
                        q_num = re.match(r'^\s*(\d+)', q)
                        q_num = q_num.group(1) if q_num else "unknown"
                        
                        user_prompt = f"""Question: {q}

Answer text (line numbers in [brackets]):
{text}

Find which line number the answer to this question starts at.
Return only the line number (just the number, e.g., 42).
If not found, return -1.

Line number:"""

                        response = client.chat.completions.create(
                            model="openai/gpt-oss-120b",
                            messages=[
                                {"role": "system", "content": "Return only a number."},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.0,
                            max_tokens=10,
                        )
                        
                        content = response.choices[0].message.content.strip()
                        num_match = re.search(r'(\d+)', content)
                        if num_match:
                            start_idx = int(num_match.group(1))
                            if 0 <= start_idx < len(answer_lines):
                                # Find end
                                end_idx = len(answer_lines)
                                next_q = str(int(q_num) + 1)
                                for i in range(start_idx + 1, len(answer_lines)):
                                    if re.search(rf'\b{next_q}[\.\)]', answer_lines[i]):
                                        end_idx = i
                                        break
                                
                                r["answer"] = " ".join(answer_lines[start_idx:end_idx]).strip()
                                r["matched"] = True
                                log(f"LLM found answer for question {q_num}")
        except Exception as e:
            log(f"LLM fallback failed: {e}")
    
    return results


# =========================================================
# MAIN PIPELINE
# =========================================================

def process_pdf(file_input, status_callback=None) -> Tuple[Dict, List[Dict]]:
    """Complete PDF processing pipeline."""
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    # Step 1: OCR
    log("Running OCR...")
    file_bytes, file_name = _normalize_file_input(file_input, default_name="document.pdf")
    pages = run_ocr(file_bytes, file_name, status_callback)
    ocr_json = {
        "total_pages": len(pages),
        "pages": [{"page_number": p["page_number"], "text": p["raw_text"]} for p in pages]
    }
    log(f"OCR complete: {len(pages)} pages")

    # Step 2: Extract questions
    log("Extracting questions...")
    question_pages, questions = extract_questions(pages, status_callback)
    
    if not questions:
        raise Exception("No questions found in the document")
    
    log(f"Found {len(questions)} questions")

    # Step 3: Extract answer text (skip question pages)
    answer_lines = []
    answer_page_indices = [i for i in range(len(pages)) if i not in question_pages]
    
    for page_idx in answer_page_indices:
        page = pages[page_idx]
        for line in page["raw_text"].split("\n"):
            if line.strip():
                answer_lines.append(line)
    
    log(f"Extracted {len(answer_lines)} answer lines")

    # Step 4: Map answers
    log("Mapping answers...")
    qa_pairs = map_answers_hybrid(answer_lines, questions, status_callback)
    
    matched = sum(1 for p in qa_pairs if p["matched"])
    log(f"Matched {matched}/{len(questions)} questions")

    return ocr_json, qa_pairs


def save_outputs(ocr_json: Dict, qa_pairs: List[Dict], output_dir: str = ".",
                  base_name: str = "document") -> Tuple[str, str]:
    """Save outputs to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    ocr_path = os.path.join(output_dir, f"{base_name}_ocr.json")
    qa_path = os.path.join(output_dir, f"{base_name}_qa_pairs.json")

    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(ocr_json, f, ensure_ascii=False, indent=2)

    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    return ocr_path, qa_path
