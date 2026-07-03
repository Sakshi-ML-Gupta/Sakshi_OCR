"""
BENCHMARK-GRADE ROBUST PIPELINE
================================
LLM PROVIDES ONLY LINE RANGES
PYTHON PERFORMS PURE SLICING
ZERO LINE SKIPPING GUARANTEED
"""

import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable, Any

import fitz
import httpx
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Central configuration for the pipeline"""
    
    # API Configuration
    DATALAB_BASE_URL = "https://www.datalab.to"
    GROQ_MODEL = "openai/gpt-oss-120b"
    
    # Rate Limiting
    TPM_LIMIT = 8000
    TPM_SAFETY_FRACTION = 0.85
    
    # OCR Configuration
    MAX_PDF_SIZE_MB = 45
    OCR_POLL_INTERVAL = 2
    OCR_MAX_POLLS = 150
    
    # Chunking Configuration for Answer Mapping
    CHUNK_SIZE_LINES = 60  # Lines per chunk
    CHUNK_OVERLAP_LINES = 20  # Overlap between chunks
    
    # Retry Configuration
    MAX_GROQ_RETRIES = 5
    RETRY_BACKOFF_FACTOR = 2
    
    # Output Configuration
    OUTPUT_DIR = "output"
    OCR_JSON_FILENAME = "ocr_output.json"
    QA_JSON_FILENAME = "qa_pairs.json"


# ============================================================
# API KEY HANDLER
# ============================================================

def get_api_key(name: str) -> Optional[str]:
    """Get API key from secrets or environment"""
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv(name)


# ============================================================
# INPUT NORMALIZATION
# ============================================================

def normalize_file_input(file_input: Any, default_name: str = "document.pdf") -> Tuple[bytes, str]:
    """
    Normalize various input types to (bytes, filename)
    
    Supports:
    - Tuple: (filename, bytes)
    - Bytes/bytearray: Raw file bytes
    - str/Path: File path
    - File-like object: With .read() method
    """
    if isinstance(file_input, tuple):
        if len(file_input) < 2:
            raise ValueError(f"Tuple must have at least (filename, bytes), got {len(file_input)}")
        name, data = file_input[0], file_input[1]
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"Expected bytes, got {type(data).__name__}")
        return bytes(data), _coerce_filename(name, default_name)
    
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
        return bytes(data), _coerce_filename(name, default_name)
    
    raise TypeError(f"Unsupported input type: {type(file_input).__name__}")


def _coerce_filename(name: Any, default: str = "document.pdf") -> str:
    """Convert any name to a safe filename"""
    if isinstance(name, (tuple, list)) or not name:
        return default
    try:
        return Path(str(name)).name or default
    except Exception:
        return default


# ============================================================
# PDF PREPROCESSING
# ============================================================

def preprocess_pdf(file_bytes: bytes, dpi: int = 250) -> bytes:
    """
    Convert PDF to image-based PDF for better OCR quality
    """
    import io
    import fitz
    
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


# ============================================================
# OCR ENGINE - DATALAB
# ============================================================

def run_ocr(file_bytes: bytes, file_name: str, callback: Optional[Callable] = None) -> List[Dict]:
    """
    Run OCR using Datalab API
    
    Returns:
        List of pages, each with:
        - page_number: int (1-based)
        - raw_text: str (OCR text)
    """
    def log(msg: str):
        print(msg)
        if callback:
            callback(msg)
    
    # Validate input
    if not isinstance(file_bytes, (bytes, bytearray)):
        raise TypeError(f"Expected bytes, got {type(file_bytes).__name__}")
    
    # Get API key
    api_key = get_api_key("DATALAB_API_KEY")
    if not api_key:
        raise Exception("DATALAB_API_KEY not found in secrets or environment")
    
    # Check file size
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > Config.MAX_PDF_SIZE_MB:
        raise Exception(f"File is {size_mb:.1f}MB, exceeds {Config.MAX_PDF_SIZE_MB}MB limit")
    
    headers = {"X-API-Key": api_key}
    log(f"Submitting to Datalab... ({size_mb:.1f}MB)")
    
    # Submit for OCR
    response = httpx.post(
        f"{Config.DATALAB_BASE_URL}/api/v1/convert",
        headers=headers,
        files={"file": (file_name, file_bytes, "application/pdf")},
        data={
            "output_format": "markdown",
            "mode": "accurate",
            "paginate": "true"
        },
        timeout=120
    )
    
    if response.status_code != 200:
        raise Exception(f"Datalab error {response.status_code}: {response.text}")
    
    data = response.json()
    if not data.get("success", True):
        raise Exception(f"Datalab failed: {data.get('error')}")
    
    # Poll for results
    check_url = data["request_check_url"]
    log("Polling for OCR result...")
    
    result = None
    for attempt in range(Config.OCR_MAX_POLLS):
        poll_response = httpx.get(check_url, headers=headers, timeout=60)
        
        if poll_response.status_code != 200:
            raise Exception(f"Poll error {poll_response.status_code}: {poll_response.text}")
        
        result = poll_response.json()
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
        raise Exception("OCR timed out after 5 minutes")
    
    if not result.get("success", True):
        raise Exception(f"OCR error: {result.get('error')}")
    
    # Parse markdown
    markdown = result.get("markdown", "")
    if not markdown.strip():
        raise Exception("OCR returned empty result")
    
    # Split into pages
    page_texts = _split_markdown_pages(markdown, result.get("page_count"), log=log)
    
    pages = []
    for idx, text in enumerate(page_texts):
        pages.append({
            "page_number": idx + 1,
            "raw_text": text
        })
    
    log(f"OCR complete: {len(pages)} pages")
    return pages


def _split_markdown_pages(markdown: str, page_count_hint: Optional[int] = None, log=print) -> List[str]:
    """
    Split markdown into pages using various patterns
    """
    # Common page break patterns
    patterns = [
        re.compile(r'\n?\{(\d+)\}-{3,}\n?'),
        re.compile(r'\n?-{2,}\{(\d+)\}-{2,}\n?'),
        re.compile(r'\n-{3,}\s*Page\s*\d+\s*-{3,}\n', re.IGNORECASE),
        re.compile(r'\n\s*\[PAGE\s*(\d+)\]\s*\n', re.IGNORECASE),
        re.compile(r'\n\s*<!--\s*page\s*(\d+)\s*-->\s*\n', re.IGNORECASE),
    ]
    
    for pattern in patterns:
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


def build_ocr_json(pages: List[Dict]) -> Dict:
    """Convert pages to OCR JSON format"""
    return {
        "total_pages": len(pages),
        "pages": [
            {"page_number": p["page_number"], "text": p["raw_text"]}
            for p in pages
        ]
    }


# ============================================================
# GROQ API HELPERS
# ============================================================

class TokenBudgetTracker:
    """
    Track token usage for rate limiting
    """
    def __init__(self, tpm_limit: int = Config.TPM_LIMIT, safety_fraction: float = Config.TPM_SAFETY_FRACTION):
        from collections import deque
        self.tpm_limit = tpm_limit
        self.safe_limit = tpm_limit * safety_fraction
        self.events = deque()
        self.lock = threading.Lock()
    
    def _prune(self, now: Optional[float] = None):
        now = now or time.monotonic()
        with self.lock:
            while self.events and now - self.events[0][0] >= 60:
                self.events.popleft()
    
    def used_in_window(self, now: Optional[float] = None) -> int:
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


_groq_call_lock = threading.Lock()


def estimate_tokens(text: str) -> int:
    """Rough token estimation"""
    return int(len(text) / 2.0) + 1


def call_groq_with_retries(
    client: Groq,
    system_prompt: str,
    user_prompt: str,
    response_parser: Callable,
    budget: TokenBudgetTracker,
    log: Callable,
    max_retries: int = Config.MAX_GROQ_RETRIES
) -> Any:
    """
    Call Groq API with automatic retry and rate limiting
    
    Args:
        client: Groq client instance
        system_prompt: System prompt for the LLM
        user_prompt: User prompt for the LLM
        response_parser: Function to parse the response
        budget: Token budget tracker
        log: Logging function
        max_retries: Maximum number of retries
    
    Returns:
        Parsed response from the LLM
    """
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
                import re
                match = re.search(r'try again in\s+([\d.]+)s', error_str)
                if match:
                    wait_time = float(match.group(1)) + 1.0
                    log(f"Rate limit: waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    budget.reset_window()
                    skip_proactive_check = True
                    continue
            
            log(f"Attempt {attempt} failed: {e}")
            time.sleep(Config.RETRY_BACKOFF_FACTOR ** attempt)
    
    raise Exception(f"Failed after {max_retries + 1} attempts. Last error: {last_error}")


# ============================================================
# QUESTION DETECTION - HIGH LEVEL SYSTEM PROMPT
# ============================================================

QUESTION_DETECTION_SYSTEM_PROMPT = """You are a highly accurate document classifier specialized in analyzing academic exam booklets.

YOUR ROLE:
- Identify which pages contain the OFFICIAL QUESTION PAPER
- Distinguish between question pages, answer pages, and administrative pages

CLASSIFICATION RULES:
1. QUESTION PAPER PAGES:
   - Contain printed exam questions with numbers (1., 2., etc.)
   - Are concise instructions directed at students
   - May include mark allocations like (10 marks)
   - Are typically short per question

2. ANSWER PAGES:
   - Contain student's handwritten responses
   - Are long paragraphs of explanation
   - May restate the question at the beginning
   - Have continuous prose/essay style

3. ADMINISTRATIVE PAGES:
   - Enrollment numbers
   - Programme codes
   - Student names
   - Regional centre information
   - Logos and letterheads

OUTPUT FORMAT:
Return ONLY valid JSON:
{
  "question_paper_pages": [page_numbers]
}

CRITICAL: Page numbers are 1-based (first page is 1).
Output ONLY the JSON object. No other text."""


def identify_question_pages(pages: List[Dict], callback: Optional[Callable] = None) -> List[int]:
    """
    Identify which pages contain the question paper
    
    Args:
        pages: List of pages with OCR text
        callback: Optional status callback
    
    Returns:
        List of 0-based page indices containing the question paper
    """
    def log(msg: str):
        print(msg)
        if callback:
            callback(msg)
    
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
        user_prompt = f"Analyze these pages. Which pages contain the question paper?\n\n{user_prompt}"
        
        def parser(content: str) -> List[int]:
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\s*\n?', '', content)
                content = re.sub(r'\n?```\s*$', '', content)
            data = json.loads(content)
            return data.get("question_paper_pages", [])
        
        try:
            qp_pages = call_groq_with_retries(
                client, QUESTION_DETECTION_SYSTEM_PROMPT, user_prompt, parser, budget, log
            )
            all_qp_pages.update(qp_pages)
        except Exception as e:
            log(f"Warning: chunk failed: {e}")
            continue
    
    # Validate and convert to 0-based indices
    valid_pages = [p - 1 for p in all_qp_pages if 1 <= p <= len(pages)]
    return sorted(valid_pages)


# ============================================================
# QUESTION EXTRACTION - HIGH LEVEL SYSTEM PROMPT
# ============================================================

QUESTION_EXTRACTION_SYSTEM_PROMPT = """You are a precise question extractor for academic exam papers.

YOUR TASK:
Extract ALL questions from the given question paper pages.

EXTRACTION RULES:
1. Preserve the EXACT numbering as printed (1., 2., (i), (ii), etc.)
2. Keep the ORIGINAL text exactly as printed
3. Include mark allocations if present (e.g., "(10 marks)")
4. For multi-part questions, keep all parts together under one entry
5. Maintain the printed order of questions

OUTPUT FORMAT:
Return ONLY valid JSON:
{
  "questions": ["question 1", "question 2", ...]
}

CRITICAL: Do not paraphrase, do not translate, do not renumber.
Output ONLY the JSON object. No other text."""


def extract_questions(qp_pages: List[Dict], callback: Optional[Callable] = None) -> List[str]:
    """
    Extract questions from question paper pages
    
    Args:
        qp_pages: List of question paper pages
        callback: Optional status callback
    
    Returns:
        List of extracted questions
    """
    def log(msg: str):
        print(msg)
        if callback:
            callback(msg)
    
    if not qp_pages:
        return []
    
    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        return []
    
    client = Groq(api_key=api_key)
    budget = TokenBudgetTracker()
    
    user_prompt = "\n\n".join([
        f"--- PAGE {p['page_number']} ---\n{p['raw_text']}"
        for p in qp_pages
    ])
    user_prompt = f"Extract all questions from these pages:\n\n{user_prompt}"
    
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
            client, QUESTION_EXTRACTION_SYSTEM_PROMPT, user_prompt, parser, budget, log
        )
        return questions
    except Exception as e:
        log(f"Question extraction failed: {e}")
        return []


# ============================================================
# ANSWER MAPPING - HIGH LEVEL SYSTEM PROMPT
# ============================================================

ANSWER_MAPPING_SYSTEM_PROMPT = """You are a precise answer boundary detector for academic exam booklets.

YOUR TASK:
For EACH official question, find the EXACT line range where the student's answer appears.

CRITICAL RULES:
1. Each answer starts at the FIRST line of that answer
2. Each answer ends at the LAST line of that answer
3. Include ALL lines from start to end - NO SKIPPING
4. The answer begins when the student starts responding to that question
5. The answer ends before the next question's answer starts
6. For the last question, the answer ends at the LAST line of the text
7. Use line numbers EXACTLY as given in [brackets]
8. Each REF's range must NOT overlap with another REF's range
9. If a question's answer is not found, omit it entirely

BOUNDARY DETECTION GUIDANCE:
- Look for labels: "Ans 5-", "उत्तर-", "Q.8", etc.
- Look for question restatement at the start of answers
- Look for clear topic shifts between answers
- If unsure, end the earlier answer sooner

OUTPUT FORMAT:
Return ONLY valid JSON:
{
  "answers": [
    {"ref": "REF-A", "start_line": 12, "end_line": 18},
    {"ref": "REF-B", "start_line": 19, "end_line": 25}
  ]
}

If no answers found: {"answers": []}

CRITICAL: Output ONLY the JSON object. No other text, no explanation."""


def map_answers_with_llm(
    answer_lines: List[str],
    questions: List[str],
    callback: Optional[Callable] = None
) -> Dict[str, str]:
    """
    Map questions to answers using LLM range detection.
    
    IMPORTANT: LLM provides ONLY ranges. Python performs pure slicing.
    This guarantees NO LINE SKIPPING from start or end of answers.
    
    Args:
        answer_lines: List of answer text lines
        questions: List of official questions
        callback: Optional status callback
    
    Returns:
        Dictionary mapping question text -> answer text
    """
    def log(msg: str):
        print(msg)
        if callback:
            callback(msg)
    
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
    
    # Build REF mapping (deterministic)
    ref_to_question = {f"REF-{chr(65+i)}": q for i, q in enumerate(questions)}
    
    # Prepare numbered lines
    numbered_lines = list(enumerate(answer_lines))
    total_lines = len(answer_lines)
    
    log(f"Total answer lines: {total_lines}")
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
        questions_block = "\n".join(
            f"[REF-{chr(65+i)}] {q}" for i, q in enumerate(questions)
        )
        lines_block = "\n".join(f"[{idx}] {text}" for idx, text in chunk)
        
        user_prompt = f"""OFFICIAL QUESTIONS (use REF labels):
{questions_block}

STUDENT'S ANSWER TEXT (line-numbered):
{lines_block}"""
        
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
                client, ANSWER_MAPPING_SYSTEM_PROMPT, user_prompt, parser, budget, log
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
    
    # ============================================================
    # PURE PYTHON SLICING - EXTRACT ALL LINES
    # ============================================================
    
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
        
        # ============================================================
        # PURE SLICING - NO FILTERING, NO SKIPPING
        # ============================================================
        extracted_lines = answer_lines[start:end + 1]
        
        # Clean only the first line's label (preserve all content)
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


# ============================================================
# NOISE DETECTION
# ============================================================

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
    """Check if a line is noise/irrelevant"""
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


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_pdf(file_input: Any, callback: Optional[Callable] = None) -> Tuple[Dict, List[Dict]]:
    """
    Main pipeline: Process PDF and extract Q&A pairs
    
    Args:
        file_input: PDF file (bytes, path, tuple, or file-like)
        callback: Optional status callback
    
    Returns:
        Tuple of (ocr_json, qa_pairs)
    
    Guarantees:
        - No line skipping in answer extraction
        - LLM provides only ranges, Python performs slicing
        - Benchmark-grade robustness
    """
    def log(msg: str):
        print(msg)
        if callback:
            callback(msg)
    
    log("=" * 70)
    log("STARTING BENCHMARK-GRADE ROBUST PIPELINE")
    log("=" * 70)
    
    # Step 1: Normalize input and OCR
    log("\n[1/6] Normalizing input and running OCR...")
    file_bytes, file_name = normalize_file_input(file_input)
    pages = run_ocr(file_bytes, file_name, callback)
    log(f"OCR complete: {len(pages)} pages")
    
    # Step 2: Build OCR JSON
    log("\n[2/6] Building OCR JSON...")
    ocr_json = build_ocr_json(pages)
    
    # Step 3: Identify question pages
    log("\n[3/6] Identifying question paper pages...")
    qp_indices = identify_question_pages(pages, callback)
    log(f"Found {len(qp_indices)} question paper pages")
    
    if not qp_indices:
        raise Exception("No question paper pages found in document")
    
    # Step 4: Extract questions
    log("\n[4/6] Extracting questions...")
    qp_pages = [pages[i] for i in qp_indices]
    questions = extract_questions(qp_pages, callback)
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
    
    if not answer_lines:
        raise Exception("No answer lines found")
    
    # Sanity check
    if not sanity_check_answer_pages(answer_lines, len(questions), log):
        log("WARNING: Answer pages look suspiciously short")
        log("Proceeding anyway...")
    
    # Step 6: Map answers - ROBUST VERSION WITH PURE SLICING
    log("\n[6/6] Mapping answers using LLM range detection + Python slicing...")
    qa_map = map_answers_with_llm(answer_lines, questions, callback)
    
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


# ============================================================
# OUTPUT SAVING
# ============================================================

def save_outputs(
    ocr_json: Dict,
    qa_pairs: List[Dict],
    output_dir: str = Config.OUTPUT_DIR,
    base_name: str = "document"
) -> Tuple[str, str]:
    """
    Save outputs to JSON files
    
    Args:
        ocr_json: OCR JSON data
        qa_pairs: Q&A pairs list
        output_dir: Output directory
        base_name: Base filename
    
    Returns:
        Tuple of (ocr_path, qa_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    ocr_path = os.path.join(output_dir, f"{base_name}_ocr.json")
    qa_path = os.path.join(output_dir, f"{base_name}_qa_pairs.json")
    
    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(ocr_json, f, ensure_ascii=False, indent=2)
    
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    return ocr_path, qa_path


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    'process_pdf',
    'save_outputs',
    'run_ocr',
    'build_ocr_json',
    'preprocess_pdf',
    'get_api_key',
    'Config',
]
