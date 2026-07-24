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
# EXTRACT QUESTIONS - SIMPLE REGEX
# =========================================================

def extract_questions(pages: List[Dict]) -> Tuple[List[int], List[str]]:
    """Extract questions from pages using regex."""
    
    question_pages = []
    all_text = ""
    
    for i, page in enumerate(pages):
        text = page["raw_text"]
        # Check if page has numbered questions
        if re.search(r'(?m)^\s*\d+[\.\)]\s+', text):
            question_pages.append(i)
            all_text += text + "\n\n"
    
    if not question_pages:
        # If no clear question pages, check all pages
        for i, page in enumerate(pages):
            if len(page["raw_text"]) < 2000 and re.search(r'\d+[\.\)]', page["raw_text"]):
                question_pages.append(i)
                all_text += page["raw_text"] + "\n\n"
    
    # Extract questions
    questions = []
    if all_text:
        # Pattern to match numbered questions
        pattern = r'(?m)^\s*(\d+[\.\)]\s*[^\n]+(?:\n\s+[^\n\d]+)*)'
        matches = re.findall(pattern, all_text)
        if matches:
            questions = [m.strip() for m in matches]
    
    # If still no questions, try simpler pattern
    if not questions:
        pattern = r'(?m)^\s*(\d+)\s*[\.\)]\s*([^\n]+)'
        matches = re.findall(pattern, all_text)
        if matches:
            questions = [f"{m[0]}. {m[1]}" for m in matches]
    
    print(f"Found {len(questions)} questions on {len(question_pages)} pages")
    return question_pages, questions


# =========================================================
# ANSWER MAPPING - DIRECT TEXT MATCHING (NO LLM)
# =========================================================

def map_answers_direct(answer_lines: List[str], questions: List[str]) -> List[Dict]:
    """Directly map answers by matching question numbers."""
    
    results = []
    
    for q_idx, question in enumerate(questions):
        # Extract question number
        q_num_match = re.match(r'^\s*(\d+)', question)
        q_num = q_num_match.group(1) if q_num_match else str(q_idx + 1)
        
        print(f"Looking for answer to question {q_num}: {question[:50]}...")
        
        # Find where this answer starts
        start_idx = None
        
        # Pattern 1: "Ans 1.", "Answer 1:", "उत्तर 1"
        patterns = [
            rf'(?i)(?:Ans|Answer|उत्तर|प्र)\s*{q_num}[\.\s:-]',
            rf'(?i)question\s*{q_num}',
            rf'\b{q_num}[\.\)]\s+[A-Z]',
        ]
        
        for i, line in enumerate(answer_lines):
            for pattern in patterns:
                if re.search(pattern, line):
                    start_idx = i
                    print(f"  Found start at line {i}: {line[:50]}...")
                    break
            if start_idx is not None:
                break
        
        # If not found, try to find any line containing the question number
        if start_idx is None:
            for i, line in enumerate(answer_lines):
                if re.search(rf'\b{q_num}\b', line) and len(line) > 20:
                    start_idx = i
                    print(f"  Found by number at line {i}: {line[:50]}...")
                    break
        
        if start_idx is not None:
            # Find where next answer starts
            next_q_num = str(int(q_num) + 1)
            end_idx = len(answer_lines)
            
            for i in range(start_idx + 1, len(answer_lines)):
                if re.search(rf'(?i)(?:Ans|Answer|उत्तर|प्र)\s*{next_q_num}[\.\s:-]', answer_lines[i]):
                    end_idx = i
                    break
                if re.search(rf'\b{next_q_num}[\.\)]\s+[A-Z]', answer_lines[i]):
                    end_idx = i
                    break
                # If we see a new question number
                if re.search(rf'\b\d+[\.\)]\s+[A-Z]', answer_lines[i]):
                    # Check if it's a different question
                    num_match = re.search(r'\b(\d+)[\.\)]', answer_lines[i])
                    if num_match and num_match.group(1) != q_num:
                        end_idx = i
                        break
            
            # Extract answer text
            answer_text = "\n".join(answer_lines[start_idx:end_idx]).strip()
            
            # Remove the "Ans" prefix if present
            answer_text = re.sub(r'(?i)^(?:Ans|Answer|उत्तर|प्र)\s*\d+[\.\s:-]*', '', answer_text).strip()
            
            results.append({
                "question": question,
                "answer": answer_text,
                "matched": True
            })
            print(f"  Matched: {len(answer_text)} chars")
        else:
            print(f"  No match found for question {q_num}")
            results.append({
                "question": question,
                "answer": "",
                "matched": False
            })
    
    return results


# =========================================================
# FALLBACK - SPLIT BY PAGE IF NO ANSWERS FOUND
# =========================================================

def map_answers_fallback(pages: List[Dict], question_pages: List[int]) -> List[Dict]:
    """Fallback: treat each page after question pages as an answer."""
    
    results = []
    
    # Get all non-question pages as answers
    answer_page_indices = [i for i in range(len(pages)) if i not in question_pages]
    
    if not answer_page_indices:
        return []
    
    # Combine all answer text
    all_answer_text = ""
    for idx in answer_page_indices:
        all_answer_text += pages[idx]["raw_text"] + "\n\n"
    
    # Try to split by question numbers
    answer_lines = all_answer_text.split("\n")
    answer_lines = [l.strip() for l in answer_lines if l.strip()]
    
    # Try to find question numbers in the answer text
    # If we can't find answers, just return the whole text as one answer
    if answer_lines:
        # Group by question numbers found in text
        q_pattern = r'(?m)^\s*(\d+)[\.\)]\s+'
        matches = list(re.finditer(q_pattern, all_answer_text))
        
        if matches:
            # Split by each question number found
            for i, match in enumerate(matches):
                q_num = match.group(1)
                start = match.start()
                end = matches[i+1].start() if i+1 < len(matches) else len(all_answer_text)
                answer_text = all_answer_text[start:end].strip()
                
                # Find matching question
                for q in questions:
                    if q.startswith(q_num):
                        results.append({
                            "question": q,
                            "answer": answer_text,
                            "matched": True
                        })
                        break
        else:
            # If no question numbers found, treat entire text as one answer
            if questions:
                results.append({
                    "question": questions[0],
                    "answer": all_answer_text.strip(),
                    "matched": True
                })
    
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
    question_pages, questions = extract_questions(pages)
    
    if not questions:
        raise Exception("No questions found in the document")
    
    log(f"Found {len(questions)} questions")

    # Step 3: Extract answer text
    answer_lines = []
    answer_page_indices = [i for i in range(len(pages)) if i not in question_pages]
    
    if answer_page_indices:
        log(f"Found {len(answer_page_indices)} answer pages")
        for page_idx in answer_page_indices:
            page = pages[page_idx]
            for line in page["raw_text"].split("\n"):
                if line.strip() and len(line.strip()) > 10:  # Skip very short lines
                    answer_lines.append(line.strip())
    else:
        log("No separate answer pages found, using all non-question text")
        # Use all text except question pages
        for page in pages:
            if page not in question_pages:
                for line in page["raw_text"].split("\n"):
                    if line.strip() and len(line.strip()) > 10:
                        answer_lines.append(line.strip())
    
    log(f"Extracted {len(answer_lines)} answer lines")

    # Step 4: Map answers
    log("Mapping answers directly...")
    qa_pairs = map_answers_direct(answer_lines, questions)
    
    matched = sum(1 for p in qa_pairs if p["matched"])
    log(f"Matched {matched}/{len(questions)} questions directly")
    
    # If no answers matched, try fallback
    if matched == 0:
        log("No answers matched directly, trying fallback...")
        qa_pairs = map_answers_fallback(pages, question_pages)
        matched = sum(1 for p in qa_pairs if p["matched"])
        log(f"Fallback matched {matched}/{len(questions)} questions")

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
