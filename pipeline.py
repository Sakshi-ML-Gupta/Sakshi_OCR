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
    
    if not pages:
        pages = [{"page_number": 1, "raw_text": markdown.strip()}]
        log("WARNING: Could not detect page boundaries, treating as single page")

    log(f"OCR done: {len(pages)} pages")
    return pages


# =========================================================
# SMART QUESTION EXTRACTION
# =========================================================

def extract_questions_smart(pages: List[Dict]) -> Tuple[List[int], List[Dict]]:
    """Extract questions with their context."""
    
    question_pages = []
    questions_with_context = []
    
    for i, page in enumerate(pages):
        text = page["raw_text"]
        
        # Check for question patterns
        # Pattern 1: Numbered questions with context
        pattern = r'(?m)^\s*(\d+[\.\)])\s*([^\n]+(?:\n\s+[^\n\d]+)*)'
        matches = re.findall(pattern, text)
        
        if matches:
            question_pages.append(i)
            for num, q_text in matches:
                # Clean up the question text
                q_text = q_text.strip()
                if len(q_text) > 10:  # Avoid very short matches
                    questions_with_context.append({
                        "page": i,
                        "number": num.strip(),
                        "text": f"{num} {q_text}",
                        "raw": q_text
                    })
    
    # If no clear question pages, try more patterns
    if not question_pages:
        for i, page in enumerate(pages):
            text = page["raw_text"]
            # Look for "Question" or "Answer" patterns
            if re.search(r'(?i)(?:question|answer|write|explain|discuss)', text):
                # Extract numbered items
                matches = re.findall(r'(?m)^\s*(\d+[\.\)])\s*([^\n]{20,})', text)
                if matches:
                    question_pages.append(i)
                    for num, q_text in matches:
                        if len(q_text) > 20:
                            questions_with_context.append({
                                "page": i,
                                "number": num.strip(),
                                "text": f"{num} {q_text}",
                                "raw": q_text
                            })
    
    # Group questions by their number - handle multiple sections
    grouped = {}
    for q in questions_with_context:
        num = q["number"]
        if num not in grouped:
            grouped[num] = []
        grouped[num].append(q)
    
    # For each number, keep the most detailed question
    final_questions = []
    for num, q_list in grouped.items():
        # Sort by length (longer is usually more complete)
        q_list.sort(key=lambda x: len(x["text"]), reverse=True)
        final_questions.append(q_list[0])
    
    # Sort by page order
    final_questions.sort(key=lambda x: (x["page"], x["number"]))
    
    print(f"Found {len(question_pages)} question pages, {len(final_questions)} unique questions")
    return question_pages, final_questions


# =========================================================
# CONTEXT-AWARE ANSWER MAPPING
# =========================================================

def map_answers_contextual(answer_lines: List[str], questions: List[Dict], 
                           question_pages: List[int]) -> List[Dict]:
    """Map answers using context and section boundaries."""
    
    results = []
    
    # Build full answer text with line numbers
    full_text = "\n".join([f"[{i}] {line}" for i, line in enumerate(answer_lines)])
    
    # Find section boundaries in the answer text
    # Look for patterns like "1." "2." "3." that indicate new sections
    section_boundaries = []
    for i, line in enumerate(answer_lines):
        if re.match(r'^\s*\d+[\.\)]\s+[A-Z]', line):
            section_boundaries.append(i)
    
    # If we have section boundaries, use them to split answers
    if section_boundaries:
        print(f"Found {len(section_boundaries)} section boundaries in answer text")
        
        for q in questions:
            q_num = q["number"].replace('.', '').replace(')', '').strip()
            q_text = q["text"]
            
            # Find where this answer starts - look for section with this number
            start_idx = None
            for i, line in enumerate(answer_lines):
                # Check if line starts with this question number
                if re.match(rf'^\s*{q_num}[\.\)]\s+', line):
                    start_idx = i
                    break
                # Check for "Ans X" or "Answer X"
                if re.search(rf'(?i)(?:ans|answer|उत्तर)\s*{q_num}[\.\s:-]', line):
                    start_idx = i
                    break
            
            if start_idx is not None:
                # Find end - next section or next question number
                end_idx = len(answer_lines)
                next_num = str(int(q_num) + 1)
                for i in range(start_idx + 1, len(answer_lines)):
                    if re.match(rf'^\s*{next_num}[\.\)]\s+', answer_lines[i]):
                        end_idx = i
                        break
                    if re.search(rf'(?i)(?:ans|answer|उत्तर)\s*{next_num}[\.\s:-]', answer_lines[i]):
                        end_idx = i
                        break
                
                # Extract answer
                answer_text = "\n".join(answer_lines[start_idx:end_idx]).strip()
                
                # Remove the question number prefix from answer
                answer_text = re.sub(rf'^\s*{q_num}[\.\)]\s+', '', answer_text)
                answer_text = re.sub(rf'(?i)^(?:ans|answer|उत्तर)\s*{q_num}[\.\s:-]*', '', answer_text)
                answer_text = answer_text.strip()
                
                results.append({
                    "question": q_text,
                    "answer": answer_text,
                    "matched": True
                })
                print(f"Matched question {q_num}: {len(answer_text)} chars")
            else:
                print(f"No match for question {q_num}")
                results.append({
                    "question": q_text,
                    "answer": "",
                    "matched": False
                })
    
    # Fallback: If no section boundaries, try to match by keywords
    else:
        print("No clear section boundaries, using keyword matching...")
        
        for q in questions:
            q_num = q["number"].replace('.', '').replace(')', '').strip()
            q_keywords = re.findall(r'[A-Za-z]{4,}', q["text"])[:5]  # Extract keywords
            
            # Find best matching answer
            best_match = None
            best_score = 0
            
            for i, line in enumerate(answer_lines):
                # Check for number match first
                if re.search(rf'\b{q_num}\b', line):
                    # Check if this is likely an answer start
                    if re.search(r'(?i)(?:ans|answer|उत्तर|प्र)', line) or len(line) > 30:
                        score = 100 + len(line)
                        if score > best_score:
                            best_score = score
                            best_match = i
            
            # If no number match, try keyword matching
            if best_match is None and q_keywords:
                for i, line in enumerate(answer_lines):
                    if len(line) > 50:  # Likely an answer line
                        keyword_score = sum(1 for kw in q_keywords if kw.lower() in line.lower())
                        if keyword_score > 0:
                            score = keyword_score * 10 + len(line)
                            if score > best_score:
                                best_score = score
                                best_match = i
            
            if best_match is not None:
                # Find end - next question or end
                end_idx = len(answer_lines)
                next_num = str(int(q_num) + 1)
                for i in range(best_match + 1, len(answer_lines)):
                    if re.search(rf'\b{next_num}\b', answer_lines[i]) and len(answer_lines[i]) < 50:
                        end_idx = i
                        break
                    if i - best_match > 100:  # Limit answer length
                        end_idx = i
                        break
                
                answer_text = "\n".join(answer_lines[best_match:end_idx]).strip()
                results.append({
                    "question": q_text,
                    "answer": answer_text,
                    "matched": True
                })
            else:
                results.append({
                    "question": q_text,
                    "answer": "",
                    "matched": False
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

    # Step 2: Extract questions with context
    log("Extracting questions...")
    question_pages, questions = extract_questions_smart(pages)
    
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
                line = line.strip()
                if line and len(line) > 10:  # Skip very short lines
                    answer_lines.append(line)
    else:
        log("No separate answer pages found, using all non-question text")
        for page in pages:
            if page not in question_pages:
                for line in page["raw_text"].split("\n"):
                    line = line.strip()
                    if line and len(line) > 10:
                        answer_lines.append(line)
    
    log(f"Extracted {len(answer_lines)} answer lines")

    # Step 4: Map answers with context
    log("Mapping answers contextually...")
    qa_pairs = map_answers_contextual(answer_lines, questions, question_pages)
    
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
