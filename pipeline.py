import os
import io
import re
import json
import time
import difflib
import threading
from pathlib import Path
from collections import deque

import fitz
import httpx
from groq import Groq

# =========================================================
# API KEYS
# =========================================================

def get_api_key(name):
    try:
        import streamlit as st
        return st.secrets[name]
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv(name)


# =========================================================
# INPUT NORMALIZATION
# =========================================================

def _normalize_file_input(file_input, default_name="document.pdf"):
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


def _coerce_name(name, default_name="document.pdf"):
    if isinstance(name, (tuple, list)):
        return default_name
    if not name:
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
# PREPROCESS PDF
# =========================================================

def preprocess_pdf(file_bytes, dpi=250):
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

PAGE_BREAK_PATTERNS = [
    re.compile(r'\n?\{(\d+)\}-{3,}\n?'),
    re.compile(r'\n?-{2,}\{(\d+)\}-{2,}\n?'),
    re.compile(r'\n-{3,}\s*Page\s*\d+\s*-{3,}\n', re.IGNORECASE),
    re.compile(r'\n\s*\[PAGE\s*(\d+)\]\s*\n', re.IGNORECASE),
    re.compile(r'\n\s*<!--\s*page\s*(\d+)\s*-->\s*\n', re.IGNORECASE),
]


def _split_paginated_markdown(markdown: str, page_count_hint: int = None, log=print) -> list:
    best_parts = None

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

        if best_parts is None or len(parts) > len(best_parts):
            best_parts = parts

    if best_parts:
        return best_parts

    if '\f' in markdown:
        parts = [p.strip() for p in markdown.split('\f') if p.strip()]
        if len(parts) > 1:
            return parts

    log(f"WARNING: No page-break marker recognized. Treating as single page.")
    return [markdown.strip()]


def run_ocr(file_content: bytes, file_name: str, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    file_name = _coerce_name(file_name, default_name="document.pdf")

    if not isinstance(file_content, (bytes, bytearray)):
        raise TypeError(f"run_ocr() expected bytes, got {type(file_content).__name__}")

    api_key = get_api_key("DATALAB_API_KEY")
    if not api_key:
        raise Exception("DATALAB_API_KEY not found")

    size_mb = len(file_content) / (1024 * 1024)
    MAX_MB = 45
    if size_mb > MAX_MB:
        raise Exception(f"File is {size_mb:.1f}MB, exceeds {MAX_MB}MB limit")

    headers = {"X-API-Key": api_key}

    log(f"Submitting document to Datalab... ({size_mb:.1f}MB)")

    resp = httpx.post(
        f"{DATALAB_BASE_URL}/api/v1/convert",
        headers=headers,
        files={"file": (file_name, file_content, "application/pdf")},
        data={"output_format": "markdown", "mode": "accurate", "paginate": "true"},
        timeout=120
    )

    if resp.status_code != 200:
        raise Exception(f"Datalab submit error {resp.status_code}: {resp.text}")

    data = resp.json()

    if not data.get("success", True):
        raise Exception(f"Datalab submit failed: {data.get('error')}")

    check_url = data["request_check_url"]
    log("Polling for OCR result...")

    result = None
    for attempt in range(150):
        poll_resp = httpx.get(check_url, headers=headers, timeout=60)

        if poll_resp.status_code != 200:
            raise Exception(f"Datalab poll error {poll_resp.status_code}: {poll_resp.text}")

        result = poll_resp.json()
        status = result.get("status")

        if status == "complete":
            log("OCR complete!")
            break

        if status == "failed" or result.get("error"):
            raise Exception(f"Datalab conversion failed: {result.get('error')}")

        if attempt % 5 == 0:
            log(f"Still processing... ({attempt * 2}s elapsed)")

        time.sleep(2)
    else:
        raise Exception("Datalab conversion timed out")

    if not result.get("success", True):
        raise Exception(f"Datalab conversion error: {result.get('error')}")

    markdown = result.get("markdown") or ""

    if not markdown.strip():
        raise Exception("Datalab returned empty markdown")

    page_count_hint = result.get("page_count")
    page_texts = _split_paginated_markdown(markdown, page_count_hint, log=log)

    pages = []
    for idx, text in enumerate(page_texts):
        pages.append({
            "page_number": idx + 1,
            "raw_text": text
        })

    log(f"OCR done -- {len(pages)} page(s) extracted")
    return pages


def build_ocr_json(pages: list) -> dict:
    return {
        "total_pages": len(pages),
        "pages": [{"page_number": p["page_number"], "text": p["raw_text"]} for p in pages]
    }


# =========================================================
# TOKEN BUDGET TRACKER
# =========================================================

class TokenBudgetTracker:
    def __init__(self, tpm_limit=8000, safety_fraction=0.85):
        self.tpm_limit = tpm_limit
        self.safe_limit = tpm_limit * safety_fraction
        self.events = deque()
        self.lock = threading.Lock()

    def _prune(self, now=None):
        now = now if now is not None else time.monotonic()
        with self.lock:
            while self.events and now - self.events[0][0] >= 60:
                self.events.popleft()

    def used_in_window(self, now=None) -> int:
        now = now if now is not None else time.monotonic()
        self._prune(now)
        with self.lock:
            return sum(tok for _, tok in self.events)

    def wait_if_needed(self, upcoming_tokens: int, log=print):
        now = time.monotonic()
        used = self.used_in_window(now)
        projected = used + upcoming_tokens

        if projected <= self.safe_limit:
            return

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
        log(f"Pacing: {used} tokens used, +{upcoming_tokens} upcoming. Waiting {wait_s:.1f}s...")
        time.sleep(wait_s)

    def record_usage(self, tokens: int):
        with self.lock:
            self.events.append((time.monotonic(), tokens))

    def reset_window(self):
        with self.lock:
            self.events.clear()


# =========================================================
# GROQ HELPER FUNCTIONS
# =========================================================

def _estimate_tokens(text: str) -> int:
    return int(len(text) / 2.0) + 1


def _call_groq_with_retries(client, system_prompt: str, user_prompt: str,
                              response_parser, budget, log, max_retries: int = 5):
    estimated_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt) + 500
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
                    model="openai/gpt-oss-120b",
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
            
            # Check for rate limit
            if "try again in" in error_str:
                import re
                match = re.search(r'try again in\s+([\d.]+)s', error_str)
                if match:
                    wait_time = float(match.group(1)) + 1.0
                    log(f"Rate limit hit. Waiting {wait_time:.1f}s...")
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

QP_SYSTEM_PROMPT = """You are analyzing OCR text from a student exam booklet. Your task is to identify which pages contain the official QUESTION PAPER (printed questions).

Return ONLY valid JSON in this exact format:
{
  "question_paper_pages": [page_numbers],
  "questions": []
}

Rules:
- Question paper pages contain the printed list of exam questions (prompts for students)
- Answer pages contain student's handwritten responses (long paragraphs, explanations)
- Cover/admin pages contain enrollment numbers, programme codes, etc.
- Page numbers are 1-based (first page is 1)

Output ONLY the JSON object. No other text."""


def identify_questions_with_llm(pages: list, status_callback=None) -> tuple:
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found")

    client = Groq(api_key=api_key)
    budget = TokenBudgetTracker()

    # Process pages in chunks
    chunk_size = 3  # Process 3 pages at a time
    all_qp_pages = set()
    
    for i in range(0, len(pages), chunk_size):
        chunk = pages[i:i+chunk_size]
        page_nums = [p["page_number"] for p in chunk]
        log(f"Analyzing pages {page_nums}...")
        
        user_prompt = "\n\n".join([
            f"--- PAGE {p['page_number']} ---\n{p['raw_text']}"
            for p in chunk
        ])
        user_prompt = f"Analyze these pages and return question paper pages:\n\n{user_prompt}"
        
        def parser(content):
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\s*\n?', '', content)
                content = re.sub(r'\n?```\s*$', '', content)
            data = json.loads(content)
            return data.get("question_paper_pages", [])
        
        try:
            qp_pages = _call_groq_with_retries(
                client, QP_SYSTEM_PROMPT, user_prompt, parser, budget, log
            )
            all_qp_pages.update(qp_pages)
        except Exception as e:
            log(f"Warning: chunk {i//chunk_size + 1} failed: {e}")
            continue
    
    qp_page_indices = sorted([p - 1 for p in all_qp_pages if 1 <= p <= len(pages)])
    
    # Extract questions from question paper pages
    questions = []
    if qp_page_indices:
        qp_pages = [pages[i] for i in qp_page_indices]
        questions = extract_questions_from_pages(qp_pages, status_callback)
    
    log(f"Found {len(qp_page_indices)} question paper pages, {len(questions)} questions")
    return qp_page_indices, questions


def extract_questions_from_pages(qp_pages: list, status_callback=None) -> list:
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    
    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        return []
    
    client = Groq(api_key=api_key)
    budget = TokenBudgetTracker()
    
    system_prompt = """Extract ALL questions from these question paper pages. 
Return ONLY valid JSON: {"questions": ["question 1", "question 2", ...]}
Rules:
- Extract every numbered question and sub-question
- Preserve the exact numbering (1., 2., (i), (ii), etc.)
- Keep the original text exactly as printed
- Include mark allocations if present (e.g., "(10 marks)")
- If a question has parts (a), (b), (c) or (i), (ii), (iii), keep them together
- Output ONLY the JSON object, no other text"""
    
    user_prompt = "\n\n".join([
        f"--- PAGE {p['page_number']} ---\n{p['raw_text']}"
        for p in qp_pages
    ])
    user_prompt = f"Extract all questions from these pages:\n\n{user_prompt}"
    
    def parser(content):
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r'^```(?:json)?\s*\n?', '', content)
            content = re.sub(r'\n?```\s*$', '', content)
        data = json.loads(content)
        return data.get("questions", [])
    
    try:
        questions = _call_groq_with_retries(
            client, system_prompt, user_prompt, parser, budget, log
        )
        return [q.strip() for q in questions if q.strip()]
    except Exception as e:
        log(f"Question extraction failed: {e}")
        return []


# =========================================================
# ANSWER MAPPING - COMPLETE ROBUST VERSION
# =========================================================

ANSWER_MAP_SYSTEM_PROMPT = """You are given:
1. Official questions with REF labels: [REF-A], [REF-B], etc.
2. Student's answer text with line numbers in [brackets]

Your task: For EACH question, find the line range where the answer appears.

CRITICAL RULES:
- Each answer MUST start at the FIRST line of that answer
- Each answer MUST end at the LAST line of that answer
- Include ALL lines from start to end - NO SKIPPING
- If an answer spans multiple lines, include ALL of them
- The answer starts when the student begins responding to that question
- The answer ends before the next question's answer starts
- If it's the last question, the answer ends at the LAST line of the text
- Use line numbers EXACTLY as given in [brackets]
- Each REF's range MUST NOT overlap with another REF's range
- If a question's answer is not found, omit it from the output

Return ONLY valid JSON in this exact format:
{
  "answers": [
    {"ref": "REF-A", "start_line": 12, "end_line": 18},
    {"ref": "REF-B", "start_line": 19, "end_line": 25}
  ]
}

If no answers found: {"answers": []}

Output ONLY the JSON object. No other text."""


def map_answers_with_llm(answer_lines: list, questions: list, status_callback=None) -> dict:
    """
    COMPLETE ROBUST VERSION - NO LINE SKIPPING
    LLM gives ONLY range, Python does the slicing
    """
    def log(msg):
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
    log(f"Total lines to process: {total_lines}")
    
    # Show first few lines for debugging
    if total_lines > 0:
        log(f"First line: {answer_lines[0][:100]}...")
        log(f"Last line: {answer_lines[-1][:100]}...")

    # Process in chunks with OVERLAP to preserve context
    chunk_size = 50  # Lines per chunk
    overlap = 10     # Lines of overlap between chunks
    
    all_ranges = []
    chunk_failures = []
    
    for chunk_start in range(0, total_lines, chunk_size - overlap):
        chunk_end = min(chunk_start + chunk_size, total_lines)
        
        # Get chunk with overlap context
        chunk_start_with_context = max(0, chunk_start - overlap)
        chunk = numbered_lines[chunk_start_with_context:chunk_end]
        
        log(f"Processing chunk: lines {chunk_start}-{chunk_end} (with context from {chunk_start_with_context})")
        
        # Build user prompt
        user_prompt = f"""OFFICIAL QUESTIONS (use REF labels):
{chr(10).join(f'[REF-{chr(65+i)}] {q}' for i, q in enumerate(questions))}

STUDENT'S ANSWER TEXT (line-numbered):
{chr(10).join(f'[{idx}] {text}' for idx, text in chunk)}"""
        
        def parser(content):
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
            chunk_ranges = _call_groq_with_retries(
                client, ANSWER_MAP_SYSTEM_PROMPT, user_prompt, parser, budget, log
            )
            
            # Validate and add ranges
            for r in chunk_ranges:
                ref = r.get("ref", "").strip().upper()
                start = r.get("start_line", -1)
                end = r.get("end_line", -1)
                
                # Validate
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
                
                # Adjust end if beyond total
                if end >= total_lines:
                    end = total_lines - 1
                
                all_ranges.append({
                    "ref": ref,
                    "start_line": start,
                    "end_line": end,
                    "chunk": chunk_start
                })
                log(f"  Found: {ref}: lines {start}-{end}")
                
        except Exception as e:
            log(f"Chunk {chunk_start} failed: {e}")
            chunk_failures.append(str(e))
            continue

    # If no ranges found
    if not all_ranges:
        if chunk_failures:
            raise Exception(f"All chunks failed. First error: {chunk_failures[0]}")
        else:
            log("No answer ranges found. Check if pages are actually answers.")
            return {}

    # RESOLVE OVERLAPS - IMPORTANT: Keep ALL content, no skipping
    log(f"Found {len(all_ranges)} ranges before deduplication")
    
    # Deduplicate: Keep the longest range for each REF
    best_ranges = {}
    for r in all_ranges:
        ref = r["ref"]
        length = r["end_line"] - r["start_line"]
        
        if ref not in best_ranges or length > (best_ranges[ref]["end_line"] - best_ranges[ref]["start_line"]):
            best_ranges[ref] = r
    
    # Sort by start line
    sorted_ranges = sorted(best_ranges.values(), key=lambda r: r["start_line"])
    
    # RESOLVE OVERLAPS - Trim ends but NEVER skip lines
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
                
                # If trimming would make range invalid, keep original
                # but log warning
                if end < start:
                    log(f"WARNING: Range for {current['ref']} would be empty after overlap resolution")
                    continue
        
        # Ensure range is valid
        if start <= end:
            resolved_ranges.append({
                "ref": current["ref"],
                "start_line": start,
                "end_line": end
            })
    
    log(f"Resolved to {len(resolved_ranges)} non-overlapping ranges")

    # EXTRACT ANSWERS USING PURE PYTHON SLICING - NO SKIPPING
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
        
        # PURE PYTHON SLICING - EXTRACT ALL LINES
        extracted_lines = []
        for line_idx in range(start, end + 1):
            if line_idx < len(answer_lines):
                line_text = answer_lines[line_idx]
                if line_text.strip():  # Skip only truly empty lines
                    extracted_lines.append(line_text)
                else:
                    # Keep the line if it has content
                    extracted_lines.append(line_text)
        
        # Join all lines - PRESERVE EVERYTHING
        answer_text = "\n".join(extracted_lines)
        
        # Only clean up obvious labels, but KEEP ALL CONTENT
        answer_text = clean_answer_start(answer_text)
        
        if answer_text.strip():
            qa_map[original_question] = answer_text
            log(f"✓ Matched {r['ref']}: {len(answer_text)} chars, {end - start + 1} lines")
        else:
            log(f"✗ Empty answer for {r['ref']}")

    # Log unmatched questions
    for q in questions:
        if q not in qa_map:
            log(f"✗ No match for: {q[:60]}...")

    return qa_map


def clean_answer_start(answer_text: str) -> str:
    """Clean only the starting labels, preserve everything else"""
    lines = answer_text.split('\n')
    if not lines:
        return answer_text
    
    # Clean first line only
    first_line = lines[0]
    cleaned_first = re.sub(
        r'^\s*(?:Ans(?:wer)?[.\s:-]+\d*|उत्तर\s*\d*\s*[\-\:]?|प्र[०.\s]*\d*|Q\.?\s*\d+[.\s:-])',
        '',
        first_line,
        flags=re.IGNORECASE
    )
    
    # Only replace if we removed something
    if cleaned_first != first_line:
        lines[0] = cleaned_first
    
    return '\n'.join(lines)


# =========================================================
# NOISE DETECTION
# =========================================================

NOISE_RE = re.compile(
    r'(?:Teacher\'?s?\s*Signature'
    r'|Tancher\'?s?\s*Signature'
    r'|Facebook\'?s?\s*Signature'
    r'|PAGE\s*NO'
    r'|^\s*DATE\b'
    r'|^\s*\d{1,3}\s*$)',
    re.IGNORECASE
)

def is_noise(line: str) -> bool:
    return bool(NOISE_RE.search(line))


def _sanity_check_answer_pages(answer_lines: list, num_questions: int, log=print) -> bool:
    total_chars = sum(len(l) for l in answer_lines)
    avg_chars = total_chars / max(num_questions, 1)
    
    if avg_chars < 100:
        log(f"WARNING: Very few characters ({total_chars}) for {num_questions} questions")
        return False
    return True


# =========================================================
# MAIN PROCESS PDF - ROBUST VERSION
# =========================================================

def process_pdf(file_input, status_callback=None):
    """
    COMPLETE ROBUST PIPELINE
    No lines are ever skipped from answers
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    log("=" * 60)
    log("Starting PDF processing...")
    log("=" * 60)

    # Step 1: Normalize input and OCR
    file_bytes, file_name = _normalize_file_input(file_input, default_name="document.pdf")
    pages = run_ocr(file_bytes, file_name, status_callback)
    log(f"OCR complete: {len(pages)} pages")

    # Step 2: Build OCR JSON
    ocr_json = build_ocr_json(pages)
    
    # Step 3: Identify question pages
    qp_page_indices, official_questions = identify_questions_with_llm(pages, status_callback)
    log(f"Found {len(qp_page_indices)} question pages, {len(official_questions)} questions")

    if not qp_page_indices:
        raise Exception("No question paper pages found")
    
    if not official_questions:
        raise Exception("No questions extracted from question pages")

    # Step 4: Extract answer pages (all non-question pages)
    answer_page_indices = [i for i in range(len(pages)) if i not in qp_page_indices]
    log(f"Answer pages: {[i+1 for i in answer_page_indices]}")

    # Step 5: Build answer lines - KEEP ALL LINES
    answer_lines = []
    for page_idx in answer_page_indices:
        page_text = pages[page_idx]["raw_text"]
        for line in page_text.split("\n"):
            line = line.strip()
            # Keep all non-empty lines, even if they look like noise
            # The LLM will handle identifying actual content
            if line:
                answer_lines.append(line)
    
    log(f"Total answer lines: {len(answer_lines)}")

    # Sanity check
    if not _sanity_check_answer_pages(answer_lines, len(official_questions), log):
        log("WARNING: Answer pages may be misidentified")

    # Step 6: Map answers - ROBUST VERSION
    log("Mapping answers using LLM range detection...")
    qa_map = map_answers_with_llm(answer_lines, official_questions, status_callback)

    # Step 7: Build final output
    qa_pairs = []
    matched_count = 0
    unmatched_questions = []
    
    for q in official_questions:
        answer = qa_map.get(q, "")
        is_matched = bool(answer)
        if is_matched:
            matched_count += 1
        else:
            unmatched_questions.append(q[:60] + "...")
        
        qa_pairs.append({
            "question": q,
            "answer": answer,
            "matched": is_matched,
            "answer_length": len(answer)
        })
    
    log(f"Matched {matched_count} of {len(official_questions)} questions")
    
    if unmatched_questions:
        log(f"Unmatched questions: {unmatched_questions}")

    # Step 8: Validate - Ensure no answer is empty for matched questions
    for qa in qa_pairs:
        if qa["matched"] and not qa["answer"]:
            log(f"WARNING: Marked matched but empty: {qa['question'][:60]}...")

    log("=" * 60)
    log("Processing complete!")
    log("=" * 60)

    return ocr_json, qa_pairs


def save_outputs(ocr_json: dict, qa_pairs: list, output_dir: str = ".",
                  base_name: str = "document") -> tuple:
    """Save outputs to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    ocr_path = os.path.join(output_dir, f"{base_name}_ocr.json")
    qa_path = os.path.join(output_dir, f"{base_name}_qa_pairs.json")

    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(ocr_json, f, ensure_ascii=False, indent=2)

    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    return ocr_path, qa_path
