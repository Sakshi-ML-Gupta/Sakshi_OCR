import os
import io
import re
import json
import time
import threading
import difflib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import fitz
import httpx

# =========================================================
# API KEYS
# =========================================================

def get_api_key(name: str) -> Optional[str]:
    """Get API key from streamlit secrets or environment."""
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
    """Normalize various input types to (bytes, filename)."""
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
    """Extract filename from various input types."""
    if not name or isinstance(name, (tuple, list)):
        return default_name
    try:
        return Path(str(name)).name or default_name
    except Exception:
        return default_name


# =========================================================
# PREPROCESS PDF
# =========================================================

def preprocess_pdf(file_bytes: bytes, dpi: int = 250) -> bytes:
    """Convert PDF to image-based PDF for better OCR."""
    src_doc = fitz.open(stream=file_bytes, filetype="pdf")
    out_doc = fitz.open()
    for page in src_doc:
        pix = page.get_pixmap(dpi=dpi)
        new_page = out_doc.new_page(width=pix.width, height=pix.height)
        new_page.insert_image(new_page.rect, pixmap=pix)
    buf = io.BytesIO()
    out_doc.save(buf)
    src_doc.close()
    out_doc.close()
    buf.seek(0)
    return buf.read()


# =========================================================
# OCR -- Datalab
# =========================================================

DATALAB_BASE_URL = "https://www.datalab.to"

def run_ocr(file_content: bytes, file_name: str, status_callback=None) -> List[Dict]:
    """Run OCR using Datalab's Chandra model."""
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    file_name = _coerce_name(file_name, default_name="document.pdf")
    
    if not isinstance(file_content, (bytes, bytearray)):
        raise TypeError(f"Expected bytes, got {type(file_content).__name__}")

    api_key = get_api_key("DATALAB_API_KEY")
    if not api_key:
        raise Exception("DATALAB_API_KEY not found")

    # Check file size
    size_mb = len(file_content) / (1024 * 1024)
    MAX_MB = 45
    if size_mb > MAX_MB:
        raise Exception(f"File is {size_mb:.1f}MB, exceeds {MAX_MB}MB limit")

    headers = {"X-API-Key": api_key}
    log(f"Submitting to Datalab OCR... ({size_mb:.1f}MB)")

    resp = httpx.post(
        f"{DATALAB_BASE_URL}/api/v1/convert",
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

        if attempt % 5 == 0:
            log(f"Still processing... ({attempt * 2}s elapsed)")
        time.sleep(2)
    else:
        raise Exception("OCR timed out after 5 minutes")

    markdown = result.get("markdown", "")
    if not markdown.strip():
        raise Exception("Empty OCR output")

    # Parse pages - handle multiple formats
    pages = []
    
    # Try page markers
    page_markers = [
        (r'\n\s*\{(\d+)\}\s*-{3,}\s*\n', r'\{(\d+)\}'),
        (r'\n\s*-{2,}\{(\d+)\}\s*-{2,}\s*\n', r'\{(\d+)\}'),
        (r'\n\s*\[PAGE\s*(\d+)\]\s*\n', r'\[PAGE\s*(\d+)\]'),
        (r'\n\s*<!--\s*page\s*(\d+)\s*-->\s*\n', r'<!--\s*page\s*(\d+)\s*-->'),
        (r'\n\s*Page\s*(\d+)\s*\n', r'Page\s*(\d+)'),
    ]
    
    for pattern, _ in page_markers:
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
    
    # Fallback: split by form feed
    if not pages and '\f' in markdown:
        parts = [p.strip() for p in markdown.split('\f') if p.strip()]
        if len(parts) > 1:
            for idx, text in enumerate(parts):
                pages.append({"page_number": idx + 1, "raw_text": text})
    
    # Final fallback: single page
    if not pages:
        pages = [{"page_number": 1, "raw_text": markdown.strip()}]
        log(f"WARNING: Could not detect page boundaries, treating as single page")

    log(f"OCR done -- {len(pages)} page(s)")
    return pages


# =========================================================
# BUILD OCR JSON
# =========================================================

def build_ocr_json(pages: List[Dict]) -> Dict:
    """Convert pages to JSON format."""
    return {
        "total_pages": len(pages),
        "pages": [{"page_number": p["page_number"], "text": p["raw_text"]} for p in pages]
    }


# =========================================================
# REFERENCE BOOK OCR
# =========================================================

def process_reference(file_input, status_callback=None) -> Dict:
    """Process a reference book/PDF."""
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    file_bytes, file_name = _normalize_file_input(file_input, default_name="reference.pdf")
    pages = run_ocr(file_bytes, file_name, status_callback)
    log(f"Reference OCR complete -- {len(pages)} page(s)")
    return build_ocr_json(pages)


# =========================================================
# OPTIMIZED PROMPTS (Fewer tokens, clearer instructions)
# =========================================================

QP_SYSTEM_PROMPT = """You are analyzing OCR text from an exam answer booklet. Pages contain one of:
- ADMIN pages: roll numbers, letterheads, blank covers (NO question or answer content)
- QUESTION PAPER pages: numbered questions with instructions
- ANSWER pages: student's handwritten responses

For the pages shown, return ONLY valid JSON:
{
  "qp_pages": [1, 3, 5],
  "admin_pages": [2, 4]
}

Rules:
- Question pages have CONCISE prompts directed at the student
- Answer pages are LONGER, contain explanatory text, may restate questions
- When in doubt, do NOT classify as question page
- Use EXACT page numbers shown
- If no question pages, return []

JSON ONLY. No markdown. No explanation."""

CANONICAL_QUESTIONS_SYSTEM = """Extract ALL exam questions from these question paper pages. Return ONLY JSON:
{
  "questions": ["1. First question", "2. Second question"]
}

Rules:
- Keep EXACT original text
- Split multi-part questions into separate entries
- Preserve numbering
- JSON ONLY. No markdown."""

ANSWER_MAP_SYSTEM = """Map each question to its answer. Given:
1. Questions with REF-X labels
2. Student answers with line numbers [0], [1], etc.

Return JSON:
{
  "matches": [
    {"ref": "REF-A", "start": 12, "end": 25}
  ]
}

Rules:
- Each answer starts where student restates/references the question
- Include ALL lines of the answer
- If answer not found, omit it
- Use ONLY line numbers shown
- JSON ONLY. No markdown."""


# =========================================================
# QUESTION DETECTION (OPTIMIZED)
# =========================================================

def identify_questions(pages: List[Dict], status_callback=None) -> Tuple[List[int], List[str], List[int]]:
    """Identify question paper pages and extract questions."""
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    if not pages:
        return [], [], []

    # Quick heuristic check for question pages first
    candidate_qp_pages = []
    for i, page in enumerate(pages):
        text = page["raw_text"]
        # Look for question markers
        has_numbers = bool(re.search(r'(?m)^\s*\d+[\.\)]\s', text))
        is_short = len(text) < 2000
        # Looks like questions if it has numbered items and is relatively concise
        if has_numbers and is_short:
            candidate_qp_pages.append(i)
    
    # If we found candidates, use them directly
    if candidate_qp_pages:
        log(f"Found {len(candidate_qp_pages)} candidate question pages via heuristics")
        qp_indices = candidate_qp_pages
    else:
        # Fallback to LLM
        from groq import Groq
        
        api_key = get_api_key("GROQ_API_KEY")
        if not api_key:
            raise Exception("GROQ_API_KEY not found")
        
        client = Groq(api_key=api_key)
        
        # Process in chunks
        all_qp = []
        all_admin = []
        
        for i in range(0, len(pages), 5):
            chunk = pages[i:i+5]
            chunk_text = "\n\n".join([f"PAGE {p['page_number']}:\n{p['raw_text'][:3000]}" for p in chunk])
            
            user_prompt = f"Pages shown:\n{chunk_text}\n\nWhich pages are question paper pages? Admin pages?"
            
            try:
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": QP_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                data = json.loads(response.choices[0].message.content)
                qp_pages = data.get("qp_pages", [])
                admin_pages = data.get("admin_pages", [])
                
                # Convert 1-based to 0-based
                for p in qp_pages:
                    if 1 <= p <= len(pages):
                        all_qp.append(p - 1)
                for p in admin_pages:
                    if 1 <= p <= len(pages):
                        all_admin.append(p - 1)
                        
            except Exception as e:
                log(f"LLM call failed for chunk {i//5 + 1}: {e}")
                continue
        
        qp_indices = sorted(set(all_qp))
        admin_indices = sorted(set(all_admin))
        
        # Remove admin from qp
        qp_indices = [i for i in qp_indices if i not in admin_indices]
    
    # Extract questions from QP pages
    questions = []
    if qp_indices:
        # Combine all QP text
        qp_text = "\n\n".join([pages[i]["raw_text"] for i in qp_indices])
        
        # Try to extract questions with regex first (fast and reliable)
        q_patterns = [
            r'(?m)^\s*(\d+[\.\)]\s*[^\n]*(?:\n\s+[^\n\d]+)*)',  # Numbered questions
            r'(?m)^\s*[Qq]uestion\s*(\d+)[\.:]\s*([^\n]+)',      # "Question X:"
            r'(?m)^\s*(\d+)\s*[\.\)]\s*([^\n]+)',                # "1. text"
        ]
        
        for pattern in q_patterns:
            matches = re.findall(pattern, qp_text)
            if matches:
                if isinstance(matches[0], tuple):
                    questions = [f"{m[0]}. {m[1]}" if len(m) > 1 else str(m[0]) for m in matches]
                else:
                    questions = [str(m).strip() for m in matches]
                break
        
        # If regex failed, use LLM
        if not questions:
            from groq import Groq
            api_key = get_api_key("GROQ_API_KEY")
            if api_key:
                client = Groq(api_key=api_key)
                try:
                    response = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {"role": "system", "content": CANONICAL_QUESTIONS_SYSTEM},
                            {"role": "user", "content": f"Question pages:\n{qp_text[:8000]}"}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.0,
                    )
                    data = json.loads(response.choices[0].message.content)
                    questions = data.get("questions", [])
                except Exception as e:
                    log(f"Question extraction failed: {e}")
    
    # Remove duplicate questions
    seen = set()
    unique_questions = []
    for q in questions:
        q_clean = re.sub(r'^\d+[\.\)]\s*', '', q).strip()[:50]
        if q_clean and q_clean not in seen:
            seen.add(q_clean)
            unique_questions.append(q)
    
    log(f"Identified {len(qp_indices)} QP pages, {len(unique_questions)} questions")
    return qp_indices, unique_questions, admin_indices


# =========================================================
# ANSWER MAPPING (OPTIMIZED - Sequential)
# =========================================================

def map_answers(answer_lines: List[str], questions: List[str], 
                status_callback=None) -> List[Dict]:
    """Map questions to answers using sequential search."""
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    if not answer_lines or not questions:
        return []

    from groq import Groq
    
    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found")
    
    client = Groq(api_key=api_key)
    
    # Build numbered lines
    numbered = [(i, line) for i, line in enumerate(answer_lines)]
    total_lines = len(numbered)
    
    # Search for each question's answer
    results = []
    pointer = 0
    
    for q_idx, question in enumerate(questions):
        ref = f"REF-{chr(65 + q_idx)}"
        log(f"Searching for {ref}: {question[:60]}...")
        
        # Search forward from pointer
        found_start = None
        chunk_size = 200  # lines per chunk
        max_chunks = 50
        
        for chunk_start in range(pointer, total_lines, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_lines)
            chunk = numbered[chunk_start:chunk_end]
            
            # Build prompt
            chunk_text = "\n".join([f"[{idx}] {text}" for idx, text in chunk])
            user_prompt = f"""Question ({ref}): {question}

Student answer lines (numbered):
{chunk_text}

Find where the answer to this question STARTS. Return JSON:
{{"found": true, "line": 12}} or {{"found": false}}"""
            
            try:
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": "Find answer start. Return JSON."},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=100,  # Very short response
                )
                data = json.loads(response.choices[0].message.content)
                
                if data.get("found") and "line" in data:
                    line = int(data["line"])
                    # Verify line is in this chunk
                    if chunk_start <= line < chunk_end:
                        found_start = line
                        log(f"  Found start at line {line}")
                        break
                
            except Exception as e:
                log(f"  Search error in chunk {chunk_start}: {e}")
                continue
        
        if found_start is not None:
            pointer = found_start
            results.append({
                "ref": ref,
                "question": question,
                "start": found_start,
                "matched": True
            })
        else:
            log(f"  No match found for {ref}")
            results.append({
                "ref": ref,
                "question": question,
                "start": None,
                "matched": False
            })
    
    # Calculate end lines
    for i, result in enumerate(results):
        if not result["matched"]:
            continue
        if i + 1 < len(results) and results[i + 1]["matched"]:
            result["end"] = results[i + 1]["start"] - 1
        else:
            result["end"] = len(answer_lines) - 1
    
    # Extract answer text
    for result in results:
        if not result["matched"]:
            result["answer"] = ""
            continue
        
        start, end = result["start"], result["end"]
        raw_lines = [answer_lines[i] for i in range(start, end + 1) if answer_lines[i].strip()]
        result["answer_raw"] = " ".join(raw_lines)
        result["answer"] = strip_question_restatement(result["answer_raw"])
    
    log(f"Mapped {sum(1 for r in results if r['matched'])} of {len(questions)} questions")
    return results


# =========================================================
# TEXT CLEANING UTILITIES
# =========================================================

QUESTION_PREFIX_RE = re.compile(
    r'^\s*(?:Ans(?:wer)?\s*\d*\s*[.:-]?\s*|उत्तर\s*\d*\s*[-:]\s*|Q\.?\s*\d+[.:-]*\s*|प्रश्न?\s*\d+[.:-]*\s*)',
    re.IGNORECASE
)

NOISE_RE = re.compile(
    r'(?:signature|PAGE\s*NO|^\s*DATE\b|^\s*\d{1,3}\s*$)',
    re.IGNORECASE
)

def is_noise(line: str) -> bool:
    """Check if line is administrative noise."""
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r'^\s*\d{1,3}\s*$', stripped):
        return True
    if len(stripped) > 40:
        return False
    return bool(NOISE_RE.search(stripped))

def strip_question_restatement(text: str) -> str:
    """Remove question restatement prefix from answer."""
    text = text.strip()
    for _ in range(2):
        new_text = QUESTION_PREFIX_RE.sub('', text, count=1).strip()
        if new_text == text:
            break
        text = new_text
    return text


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
    log("Step 1: Running OCR...")
    file_bytes, file_name = _normalize_file_input(file_input, default_name="document.pdf")
    pages = run_ocr(file_bytes, file_name, status_callback)
    ocr_json = build_ocr_json(pages)
    log(f"OCR complete: {len(pages)} pages")

    # Step 2: Identify questions
    log("Step 2: Identifying questions...")
    qp_indices, questions, admin_indices = identify_questions(pages, status_callback)
    
    if not questions:
        log("WARNING: No questions found, using heuristics...")
        # Emergency fallback: treat all pages as answer pages and try to find numbered items
        all_text = "\n".join([p["raw_text"] for p in pages])
        matches = re.findall(r'(?m)^\s*(\d+[\.\)]\s*[^\n]+)', all_text)
        questions = [m.strip() for m in matches[:20]]  # Limit to 20 questions
        qp_indices = list(range(len(pages)))
        admin_indices = []
        log(f"Fallback: found {len(questions)} potential questions")

    # Step 3: Extract answer pages
    log("Step 3: Extracting answers...")
    excluded = set(qp_indices) | set(admin_indices)
    answer_page_indices = [i for i in range(len(pages)) if i not in excluded]
    
    # Build answer lines
    answer_lines = []
    for page_idx in answer_page_indices:
        page = pages[page_idx]
        for line in page["raw_text"].split("\n"):
            if not is_noise(line):
                answer_lines.append(line)
    
    log(f"Extracted {len(answer_lines)} answer lines from {len(answer_page_indices)} pages")

    # Step 4: Map answers
    log("Step 4: Mapping answers...")
    qa_pairs = map_answers(answer_lines, questions, status_callback)

    # Build final output format
    final_output = []
    for q_idx, q in enumerate(questions):
        match = next((p for p in qa_pairs if p.get("ref") == f"REF-{chr(65 + q_idx)}"), None)
        if match and match.get("matched"):
            final_output.append({
                "question": q,
                "answer": match.get("answer", ""),
                "answer_raw": match.get("answer_raw", ""),
                "matched": True
            })
        else:
            final_output.append({
                "question": q,
                "answer": "",
                "answer_raw": "",
                "matched": False
            })

    matched_count = sum(1 for p in final_output if p["matched"])
    log(f"Done: {matched_count}/{len(questions)} questions matched")

    # Check for quality issues
    if matched_count < len(questions) * 0.5:
        log("WARNING: Less than 50% questions matched - might be parsing issues")
    
    return ocr_json, final_output


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
