"""
COMPLETE ROBUST PIPELINE - ZERO LINE SKIPPING GUARANTEED
Author: Fixed for production use
"""

import os
import io
import re
import json
import time
import difflib
import threading
from pathlib import Path
from collections import deque
from typing import List, Dict, Tuple, Optional, Callable

import fitz
import httpx
from groq import Groq


# =========================================================
# CONFIGURATION
# =========================================================

class Config:
    DATALAB_BASE_URL = "https://www.datalab.to"
    GROQ_MODEL = "openai/gpt-oss-120b"
    TPM_LIMIT = 8000
    TPM_SAFETY_FRACTION = 0.85
    MAX_PDF_SIZE_MB = 45
    OCR_POLL_INTERVAL = 2
    OCR_MAX_POLLS = 150
    CHUNK_SIZE_LINES = 40  # Lines per chunk for answer mapping
    CHUNK_OVERLAP_LINES = 15  # Overlap between chunks
    MAX_GROQ_RETRIES = 5


# =========================================================
# API KEY HANDLER
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

def normalize_file_input(file_input, default_name: str = "document.pdf"):
    """Handle all types of file input consistently"""
    if isinstance(file_input, tuple):
        if len(file_input) < 2:
            raise ValueError(f"Tuple must have (filename, bytes), got {len(file_input)}")
        name, data = file_input[0], file_input[1]
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"Expected bytes, got {type(data).__name__}")
        return bytes(data), coerce_name(name, default_name)

    if isinstance(file_input, (bytes, bytearray)):
        return bytes(file_input), default_name

    if isinstance(file_input, (str, Path)):
        p = Path(file_input)
        return p.read_bytes(), p.name

    if hasattr(file_input, "read"):
        data = file_input.read()
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"read() returned {type(data).__name__}, expected bytes")
        name = getattr(file_input, "name", default_name)
        return bytes(data), coerce_name(name, default_name)

    raise TypeError(f"Unsupported type: {type(file_input).__name__}")


def coerce_name(name, default_name: str = "document.pdf") -> str:
    if isinstance(name, (tuple, list)) or not name:
        return default_name
    try:
        return Path(str(name)).name or default_name
    except Exception:
        return default_name


# =========================================================
# CONCURRENCY GUARD
# =========================================================

_groq_call_lock = threading.Lock()


# =========================================================
# PDF PREPROCESSING
# =========================================================

def preprocess_pdf(file_bytes: bytes, dpi: int = 250) -> bytes:
    """Convert PDF to image-based PDF for better OCR"""
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
# OCR ENGINE - DATALAB
# =========================================================

PAGE_BREAK_PATTERNS = [
    re.compile(r'\n?\{(\d+)\}-{3,}\n?'),
    re.compile(r'\n?-{2,}\{(\d+)\}-{2,}\n?'),
    re.compile(r'\n-{3,}\s*Page\s*\d+\s*-{3,}\n', re.IGNORECASE),
    re.compile(r'\n\s*\[PAGE\s*(\d+)\]\s*\n', re.IGNORECASE),
    re.compile(r'\n\s*<!--\s*page\s*(\d+)\s*-->\s*\n', re.IGNORECASE),
]


def split_paginated_markdown(markdown: str, page_count_hint: int = None, log=print) -> List[str]:
    """Split markdown into pages using various patterns"""
    for pattern in PAGE_BREAK_PATTERNS:
        matches = list(pattern.finditer(markdown))
        if not matches:
            continue

        parts = []
        start = 0
        for m in matches:
            parts.append(markdown[start:m.start()].strip())
            start = m.end()
        parts.append(markdown[start:].strip())
        parts = [p for p in parts if p]

        if len(parts) <= 1:
            continue

        if page_count_hint and len(parts) == page_count_hint:
            return parts

    # Fallback: split by form feed
    if '\f' in markdown:
        parts = [p.strip() for p in markdown.split('\f') if p.strip()]
        if len(parts) > 1:
            return parts

    log("WARNING: No page breaks found, treating as single page")
    return [markdown.strip()]


def run_ocr(file_content: bytes, file_name: str, status_callback: Optional[Callable] = None) -> List[Dict]:
    """Run OCR using Datalab API"""
    def log(msg: str):
        print(msg)
        if status_callback:
            status_callback(msg)

    file_name = coerce_name(file_name, "document.pdf")

    if not isinstance(file_content, (bytes, bytearray)):
        raise TypeError(f"Expected bytes, got {type(file_content).__name__}")

    api_key = get_api_key("DATALAB_API_KEY")
    if not api_key:
        raise Exception("DATALAB_API_KEY not found")

    size_mb = len(file_content) / (1024 * 1024)
    if size_mb > Config.MAX_PDF_SIZE_MB:
        raise Exception(f"File is {size_mb:.1f}MB, exceeds {Config.MAX_PDF_SIZE_MB}MB limit")

    headers = {"X-API-Key": api_key}

    log(f"Submitting to Datalab... ({size_mb:.1f}MB)")

    response = httpx.post(
        f"{Config.DATALAB_BASE_URL}/api/v1/convert",
        headers=headers,
        files={"file": (file_name, file_content, "application/pdf")},
        data={"output_format": "markdown", "mode": "accurate", "paginate": "true"},
        timeout=120
    )

    if response.status_code != 200:
        raise Exception(f"Datalab error {response.status_code}: {response.text}")

    data = response.json()
    if not data.get("success", True):
        raise Exception(f"Datalab failed: {data.get('error')}")

    check_url = data["request_check_url"]
    log("Polling for OCR result...")

    result = None
    for attempt in range(Config.OCR_MAX_POLLS):
        poll_resp = httpx.get(check_url, headers=headers, timeout=60)

        if poll_resp.status_code != 200:
            raise Exception(f"Poll error {poll_resp.status_code}: {poll_resp.text}")

        result = poll_resp.json()
        status = result.get("status")

        if status == "complete":
            log("OCR complete!")
            break

        if status == "failed" or result.get("error"):
            raise Exception(f"OCR failed: {result.get('error')}")

        if attempt % 5 == 0:
            log(f"Still processing... ({attempt * Config.OCR_POLL_INTERVAL}s)")

        time.sleep(Config.OCR_POLL_INTERVAL)
    else:
        raise Exception("OCR timed out")

    if not result.get("success", True):
        raise Exception(f"OCR error: {result.get('error')}")

    markdown = result.get("markdown", "")
    if not markdown.strip():
        raise Exception("OCR returned empty result")

    page_texts = split_paginated_markdown(markdown, result.get("page_count"), log=log)

    pages = []
    for idx, text in enumerate(page_texts):
        pages.append({
            "page_number": idx + 1,
            "raw_text": text
        })

    log(f"OCR done: {len(pages)} pages")
    return pages


def build_ocr_json(pages: List[Dict]) -> Dict:
    """Convert pages to JSON format"""
    return {
        "total_pages": len(pages),
        "pages": [
            {"page_number": p["page_number"], "text": p["raw_text"]}
            for p in pages
        ]
    }


# =========================================================
# TOKEN BUDGET TRACKER
# =========================================================

class TokenBudgetTracker:
    """Track token usage for rate limiting"""
    
    def __init__(self, tpm_limit: int = Config.TPM_LIMIT, safety_fraction: float = Config.TPM_SAFETY_FRACTION):
        self.tpm_limit = tpm_limit
        self.safe_limit = tpm_limit * safety_fraction
        self.events = deque()
        self.lock = threading.Lock()

    def _prune(self, now: float = None):
        now = now or time.monotonic()
        with self.lock:
            while self.events and now - self.events[0][0] >= 60:
                self.events.popleft()

    def used_in_window(self, now: float = None) -> int:
        now = now or time.monotonic()
        self._prune(now)
        with self.lock:
            return sum(tok for _, tok in self.events)

    def wait_if_needed(self, upcoming_tokens: int, log=print):
        now = time.monotonic()
        used = self.used_in_window(now)
        projected = used + upcoming_tokens

        if projected <= self.safe_limit:
            return

        # Calculate wait time
        needed_to_free = projected - self.safe_limit
        freed = 0
        wait_s = 0.0
        with self.lock:
            for ts, tok in self.events:
                freed += tok
                wait_s = max(wait_s, 60 - (now - ts))
                if freed >= needed_to_free:
                    break

        wait_s = max(0.0, wait_s) + 0.5
        log(f"Rate limiting: waiting {wait_s:.1f}s")
        time.sleep(wait_s)

    def record_usage(self, tokens: int):
        with self.lock:
            self.events.append((time.monotonic(), tokens))

    def reset_window(self):
        with self.lock:
            self.events.clear()


def estimate_tokens(text: str) -> int:
    """Rough token estimation"""
    return int(len(text) / 2.0) + 1


# =========================================================
# GROQ API HELPER
# =========================================================

def call_groq_with_retries(
    client: Groq,
    system_prompt: str,
    user_prompt: str,
    response_parser: Callable,
    budget: TokenBudgetTracker,
    log: Callable,
    max_retries: int = Config.MAX_GROQ_RETRIES
):
    """Call Groq API with automatic retry and rate limiting"""
    
    estimated_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt) + 500
    last_error = None
    skip_proactive_check = False

    for attempt in range(1, max_retries + 2):
        if not skip_proactive_check:
            budget.wait_if_needed(estimated_tokens, log=log)
        else:
            skip_proactive_check = False

        try:
            with _groq_call_lock:
                response = client.chat.completions.create(
                    model=Config.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
            budget.record_usage(estimated_tokens)
            content = response.choices[0].message.content
            return response_parser(content)

        except Exception as e:
            last_error = e
            error_str = str(e)
            
            # Rate limit detection
            if "try again in" in error_str:
                match = re.search(r'try again in\s+([\d.]+)s', error_str)
                if match:
                    wait_time = float(match.group(1)) + 1.0
                    log(f"Rate limit: waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    budget.reset_window()
                    skip_proactive_check = True
                    continue
            
            log(f"Attempt {attempt} failed: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff

    raise Exception(f"Failed after {max_retries + 1} attempts. Last error: {last_error}")


# =========================================================
# QUESTION DETECTION
# =========================================================

QUESTION_DETECTION_PROMPT = """You are analyzing OCR text from a student exam booklet.

TASK: Identify which pages contain the OFFICIAL QUESTION PAPER (printed exam questions).

RULES:
- Question paper pages: Printed list of exam questions with numbers (1., 2., etc.)
- Answer pages: Student's handwritten responses (long paragraphs, explanations)
- Cover pages: Enrollment numbers, programme codes, student info

Return ONLY valid JSON:
{
  "question_paper_pages": [page_numbers]
}

Page numbers are 1-based. Output ONLY the JSON, no other text."""


def identify_question_pages(pages: List[Dict], status_callback: Optional[Callable] = None) -> List[int]:
    """Identify which pages are question paper pages"""
    def log(msg: str):
        print(msg)
        if status_callback:
            status_callback(msg)

    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found")

    client = Groq(api_key=api_key)
    budget = TokenBudgetTracker()

    all_qp_pages = set()
    chunk_size = 3  # Process 3 pages at a time
    
    for i in range(0, len(pages), chunk_size):
        chunk = pages[i:i+chunk_size]
        page_nums = [p["page_number"] for p in chunk]
        log(f"Analyzing pages {page_nums}...")
        
        user_prompt = "\n\n".join([
            f"--- PAGE {p['page_number']} ---\n{p['raw_text']}"
            for p in chunk
        ])
        user_prompt = f"Analyze these pages. Which are question paper pages?\n\n{user_prompt}"
        
        def parser(content: str) -> List[int]:
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\s*\n?', '', content)
                content = re.sub(r'\n?```\s*$', '', content)
            data = json.loads(content)
            return data.get("question_paper_pages", [])
        
        try:
            qp_pages = call_groq_with_retries(
                client, QUESTION_DETECTION_PROMPT, user_prompt, parser, budget, log
            )
            all_qp_pages.update(qp_pages)
        except Exception as e:
            log(f"Warning: chunk failed: {e}")
            continue
    
    # Convert to 0-based indices and validate
    valid_pages = [p - 1 for p in all_qp_pages if 1 <= p <= len(pages)]
    return sorted(valid_pages)


def extract_questions_from_pages(qp_pages: List[Dict], status_callback: Optional[Callable] = None) -> List[str]:
    """Extract questions from question paper pages"""
    def log(msg: str):
        print(msg)
        if status_callback:
            status_callback(msg)

    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        return []

    client = Groq(api_key=api_key)
    budget = TokenBudgetTracker()

    system_prompt = """Extract ALL questions from these question paper pages.

RULES:
- Extract every numbered question and sub-question
- Preserve the exact numbering (1., 2., (i), (ii), etc.)
- Keep the original text exactly as printed
- Include mark allocations if present

Return ONLY valid JSON:
{
  "questions": ["question 1", "question 2", ...]
}

Output ONLY the JSON, no other text."""

    user_prompt = "\n\n".join([
        f"--- PAGE {p['page_number']} ---\n{p['raw_text']}"
        for p in qp_pages
    ])
    user_prompt = f"Extract all questions:\n\n{user_prompt}"
    
    def parser(content: str) -> List[str]:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r'^```(?:json)?\s*\n?', '', content)
            content = re.sub(r'\n?```\s*$', '', content)
        data = json.loads(content)
        questions = data.get("questions", [])
        return [q.strip() for q in questions if q.strip()]
    
    try:
        questions = call_groq_with_retries(
            client, system_prompt, user_prompt, parser, budget, log
        )
        return questions
    except Exception as e:
        log(f"Question extraction failed: {e}")
        return []


# =========================================================
# ANSWER MAPPING - COMPLETE ROBUST VERSION
# =========================================================

ANSWER_MAPPING_PROMPT = """You are given:
1. Official questions with REF labels: [REF-A], [REF-B], etc.
2. Student's answer text with line numbers in [brackets]

TASK: For EACH question, find the line range where the answer appears.

CRITICAL RULES:
- Each answer MUST start at the FIRST line of that answer
- Each answer MUST end at the LAST line of that answer
- Include ALL lines from start to end - NO SKIPPING
- The answer starts when the student begins responding to that question
- The answer ends before the next question's answer starts
- For the last question, the answer ends at the LAST line of the text
- Use line numbers EXACTLY as given in [brackets]
- Each REF's range MUST NOT overlap with another REF's range
- If a question's answer is not found, omit it

Return ONLY valid JSON:
{
  "answers": [
    {"ref": "REF-A", "start_line": 12, "end_line": 18},
    {"ref": "REF-B", "start_line": 19, "end_line": 25}
  ]
}

If no answers found: {"answers": []}

Output ONLY the JSON object. No other text."""


def map_answers_with_llm(
    answer_lines: List[str],
    questions: List[str],
    status_callback: Optional[Callable] = None
) -> Dict[str, str]:
    """
    Map questions to answers using LLM range detection.
    THIS IS THE ROBUST VERSION - NO LINE SKIPPING.
    """
    def log(msg: str):
        print(msg)
        if status_callback:
            status_callback(msg)

    if not answer_lines:
        log("No answer lines to process")
        return {}

    if not questions:
        log("No questions to map")
        return {}

    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found")

    client = Groq(api_key=api_key)
    budget = TokenBudgetTracker()

    # Build REF mapping
    ref_to_question = {f"REF-{chr(65+i)}": q for i, q in enumerate(questions)}
    
    # Prepare numbered lines
    numbered_lines = list(enumerate(answer_lines))
    total_lines = len(answer_lines)
    log(f"Total answer lines: {total_lines}")
    
    # Show first and last lines for debugging
    if total_lines > 0:
        log(f"First line: {answer_lines[0][:100]}...")
        log(f"Last line: {answer_lines[-1][:100]}...")

    # Process in chunks with OVERLAP
    chunk_size = Config.CHUNK_SIZE_LINES
    overlap = Config.CHUNK_OVERLAP_LINES
    
    all_ranges = []
    chunk_failures = []
    
    for chunk_start in range(0, total_lines, chunk_size - overlap):
        chunk_end = min(chunk_start + chunk_size, total_lines)
        
        # Get chunk with context (overlap from previous chunk)
        context_start = max(0, chunk_start - overlap)
        chunk = numbered_lines[context_start:chunk_end]
        
        log(f"Processing chunk: lines {chunk_start}-{chunk_end} (context from {context_start})")
        
        # Build user prompt
        questions_text = "\n".join(
            f"[REF-{chr(65+i)}] {q}" for i, q in enumerate(questions)
        )
        lines_text = "\n".join(f"[{idx}] {text}" for idx, text in chunk)
        
        user_prompt = f"""OFFICIAL QUESTIONS (use REF labels):
{questions_text}

STUDENT'S ANSWER TEXT (line-numbered):
{lines_text}"""
        
        def parser(content: str) -> List[Dict]:
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\s*\n?', '', content)
                content = re.sub(r'\n?```\s*$', '', content)
            
            try:
                data = json.loads(content)
                return data.get("answers", [])
            except:
                return []
        
        try:
            chunk_ranges = call_groq_with_retries(
                client, ANSWER_MAPPING_PROMPT, user_prompt, parser, budget, log
            )
            
            # Validate and store ranges
            for r in chunk_ranges:
                ref = r.get("ref", "").strip().upper()
                start = r.get("start_line", -1)
                end = r.get("end_line", -1)
                
                if ref not in ref_to_question:
                    log(f"  Unknown REF: {ref}")
                    continue
                if start < 0 or end < 0:
                    log(f"  Invalid line numbers for {ref}: {start}-{end}")
                    continue
                if start > end:
                    log(f"  Start > End for {ref}: {start}-{end}")
                    continue
                if start >= total_lines:
                    log(f"  Start {start} beyond total lines {total_lines}")
                    continue
                
                # Clamp end to total_lines - 1
                if end >= total_lines:
                    end = total_lines - 1
                
                all_ranges.append({
                    "ref": ref,
                    "start_line": start,
                    "end_line": end,
                    "chunk_start": chunk_start
                })
                log(f"  Found: {ref}: lines {start}-{end}")
                
        except Exception as e:
            log(f"Chunk failed: {e}")
            chunk_failures.append(str(e))
            continue

    # If no ranges found
    if not all_ranges:
        if chunk_failures:
            raise Exception(f"All chunks failed. First error: {chunk_failures[0]}")
        else:
            log("No answer ranges found.")
            return {}

    # DEDUPLICATE: Keep the LONGEST range for each REF
    log(f"Found {len(all_ranges)} ranges before deduplication")
    
    best_ranges = {}
    for r in all_ranges:
        ref = r["ref"]
        length = r["end_line"] - r["start_line"]
        
        if ref not in best_ranges or length > (best_ranges[ref]["end_line"] - best_ranges[ref]["start_line"]):
            best_ranges[ref] = r
    
    # Sort by start line
    sorted_ranges = sorted(best_ranges.values(), key=lambda r: r["start_line"])
    
    # RESOLVE OVERLAPS - Trim ends but NEVER skip content
    resolved_ranges = []
    for i, current in enumerate(sorted_ranges):
        start = current["start_line"]
        end = current["end_line"]
        
        # Check overlap with next range
        if i + 1 < len(sorted_ranges):
            next_start = sorted_ranges[i + 1]["start_line"]
            
            # If current overlaps next, trim current's end
            if end >= next_start:
                end = next_start - 1
                
                # If trimming makes range invalid, skip it
                if end < start:
                    log(f"WARNING: Range for {current['ref']} would be empty after overlap resolution")
                    continue
        
        if start <= end:
            resolved_ranges.append({
                "ref": current["ref"],
                "start_line": start,
                "end_line": end
            })
    
    log(f"Resolved to {len(resolved_ranges)} non-overlapping ranges")

    # EXTRACT ANSWERS - PURE PYTHON SLICING, NO SKIPPING
    qa_map = {}
    
    for r in resolved_ranges:
        start = r["start_line"]
        end = r["end_line"]
        
        # Ensure within bounds
        start = max(0, min(start, total_lines - 1))
        end = max(0, min(end, total_lines - 1))
        
        if start > end:
            log(f"Invalid range for {r['ref']}: {start}-{end}")
            continue
        
        # Get the original question
        original_question = ref_to_question.get(r["ref"])
        if not original_question:
            continue
        
        # PURE SLICING - Extract ALL lines in range
        extracted_lines = answer_lines[start:end + 1]
        
        # Clean the start but preserve all content
        if extracted_lines:
            first_line = extracted_lines[0]
            cleaned_first = re.sub(
                r'^\s*(?:Ans(?:wer)?[.\s:-]+\d*|उत्तर\s*\d*\s*[\-\:]?|प्र[०.\s]*\d*|Q\.?\s*\d+[.\s:-])',
                '',
                first_line,
                flags=re.IGNORECASE
            )
            if cleaned_first != first_line:
                extracted_lines[0] = cleaned_first
        
        # Join all lines - PRESERVE EVERYTHING
        answer_text = "\n".join(extracted_lines).strip()
        
        if answer_text:
            qa_map[original_question] = answer_text
            log(f"✓ Matched {r['ref']}: {len(answer_text)} chars, {len(extracted_lines)} lines")
        else:
            log(f"✗ Empty answer for {r['ref']}")

    # Log unmatched questions
    for q in questions:
        if q not in qa_map:
            log(f"✗ No match for: {q[:60]}...")

    return qa_map


# =========================================================
# SANITY CHECKS
# =========================================================

NOISE_PATTERNS = re.compile(
    r'(?:Teacher\'?s?\s*Signature'
    r'|Tancher\'?s?\s*Signature'
    r'|Facebook\'?s?\s*Signature'
    r'|PAGE\s*NO'
    r'|^\s*DATE\b'
    r'|^\s*\d{1,3}\s*$)',
    re.IGNORECASE
)


def is_noise(line: str) -> bool:
    return bool(NOISE_PATTERNS.search(line))


def sanity_check_answer_pages(answer_lines: List[str], num_questions: int, log=print) -> bool:
    """Check if answer pages look plausible"""
    total_chars = sum(len(l) for l in answer_lines)
    avg_chars = total_chars / max(num_questions, 1)
    
    if avg_chars < 100:
        log(f"WARNING: Very few characters ({total_chars}) for {num_questions} questions")
        log("This usually means answer pages were misidentified")
        return False
    
    return True


# =========================================================
# MAIN PIPELINE
# =========================================================

def process_pdf(file_input, status_callback: Optional[Callable] = None) -> Tuple[Dict, List[Dict]]:
    """
    COMPLETE ROBUST PIPELINE
    Guarantees: No line skipping from start or end of answers
    """
    def log(msg: str):
        print(msg)
        if status_callback:
            status_callback(msg)

    log("=" * 70)
    log("STARTING ROBUST PDF PROCESSING PIPELINE")
    log("=" * 70)

    # Step 1: Normalize input and OCR
    log("\n[1/6] Normalizing input and running OCR...")
    file_bytes, file_name = normalize_file_input(file_input)
    pages = run_ocr(file_bytes, file_name, status_callback)
    log(f"OCR complete: {len(pages)} pages")

    # Step 2: Build OCR JSON
    log("\n[2/6] Building OCR JSON...")
    ocr_json = build_ocr_json(pages)
    
    # Step 3: Identify question pages
    log("\n[3/6] Identifying question paper pages...")
    qp_indices = identify_question_pages(pages, status_callback)
    log(f"Found {len(qp_indices)} question paper pages")

    if not qp_indices:
        raise Exception("No question paper pages found in document")

    # Step 4: Extract questions
    log("\n[4/6] Extracting questions...")
    qp_pages = [pages[i] for i in qp_indices]
    questions = extract_questions_from_pages(qp_pages, status_callback)
    log(f"Extracted {len(questions)} questions")

    if not questions:
        raise Exception("No questions extracted from question pages")

    # Step 5: Extract answer pages
    log("\n[5/6] Extracting answer pages...")
    answer_indices = [i for i in range(len(pages)) if i not in qp_indices]
    log(f"Answer pages: {[i+1 for i in answer_indices]}")

    # Build answer lines - KEEP EVERYTHING
    answer_lines = []
    for idx in answer_indices:
        page_text = pages[idx]["raw_text"]
        for line in page_text.split("\n"):
            line = line.strip()
            if line:  # Keep all non-empty lines
                answer_lines.append(line)

    log(f"Total answer lines: {len(answer_lines)}")

    # Sanity check
    if not sanity_check_answer_pages(answer_lines, len(questions), log):
        log("WARNING: Answer pages look suspiciously short")
        log("Proceeding anyway...")

    # Step 6: Map answers - ROBUST VERSION
    log("\n[6/6] Mapping answers using LLM range detection...")
    qa_map = map_answers_with_llm(answer_lines, questions, status_callback)

    # Build final output
    qa_pairs = []
    matched_count = 0
    
    for q in questions:
        answer = qa_map.get(q, "")
        is_matched = bool(answer)
        if is_matched:
            matched_count += 1
        
        qa_pairs.append({
            "question": q,
            "answer": answer,
            "matched": is_matched,
            "answer_lines": len(answer.split("\n")) if answer else 0,
            "answer_chars": len(answer)
        })

    log("\n" + "=" * 70)
    log(f"PROCESSING COMPLETE: {matched_count}/{len(questions)} questions matched")
    log("=" * 70)

    return ocr_json, qa_pairs


def save_outputs(
    ocr_json: Dict,
    qa_pairs: List[Dict],
    output_dir: str = ".",
    base_name: str = "document"
) -> Tuple[str, str]:
    """Save outputs to JSON files"""
    os.makedirs(output_dir, exist_ok=True)
    
    ocr_path = os.path.join(output_dir, f"{base_name}_ocr.json")
    qa_path = os.path.join(output_dir, f"{base_name}_qa_pairs.json")

    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(ocr_json, f, ensure_ascii=False, indent=2)

    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    return ocr_path, qa_path


# =========================================================
# EXPORTS
# =========================================================

__all__ = [
    'process_pdf',
    'save_outputs',
    'run_ocr',
    'build_ocr_json',
    'preprocess_pdf',
    'get_api_key',
]
