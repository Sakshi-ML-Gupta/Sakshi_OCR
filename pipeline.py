"""
UNIVERSAL EXAM PAPER PROCESSOR
Works for ANY PDF with question papers and answer sheets
Handles: IGNOU, CBSE, University exams, any format
"""

import os
import io
import re
import json
import time
import difflib
import threading
import fitz
import httpx
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple

# =========================================================
# API KEYS
# =========================================================

def get_api_key(name: str) -> Optional[str]:
    """Get API key from Streamlit secrets or environment variables."""
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv(name)


# =========================================================
# CONSTANTS
# =========================================================

TPM_LIMIT = 8000
TPM_SAFETY_FRACTION = 0.85
CHARS_PER_TOKEN_ESTIMATE = 2.0
MAX_CHARS_PER_CHUNK = 6000
CHUNK_OVERLAP_PAGES = 1

# Answer mapping chunk sizes
ANSWER_MAP_MAX_CHARS_PER_CHUNK = 11000
ANSWER_MAP_ABSOLUTE_MAX_CHARS = 60000

# OCR settings
DATALAB_BASE_URL = "https://www.datalab.to"
MAX_FILE_SIZE_MB = 45
OCR_TIMEOUT_SECONDS = 300
OCR_POLL_INTERVAL = 2

# Groq settings
GROQ_MODEL = "openai/gpt-oss-120b"
MAX_GROQ_RETRIES = 4


# =========================================================
# INPUT NORMALIZATION
# =========================================================

def _coerce_name(name: Any, default_name: str = "document.pdf") -> str:
    """Safely coerce various input types to a filename string."""
    if isinstance(name, (tuple, list)):
        return default_name
    if not name:
        return default_name
    try:
        return Path(str(name)).name or default_name
    except Exception:
        return default_name


def normalize_file_input(file_input: Union[str, Path, bytes, tuple, Any], 
                         default_name: str = "document.pdf") -> Tuple[bytes, str]:
    """
    Universal file input normalizer - handles ANY input format.
    
    Accepts:
    - str/Path: file path
    - bytes/bytearray: raw file data
    - file-like object: with .read() method
    - tuple: (filename, bytes)
    - Any other type: tries to convert to bytes
    
    Returns: (file_bytes, filename)
    """
    # Handle tuple input (filename, data)
    if isinstance(file_input, tuple):
        if len(file_input) < 2:
            raise ValueError(f"Tuple must have at least (filename, bytes), got {len(file_input)} items")
        
        name, data = file_input[0], file_input[1]
        
        # If data is also a tuple (nested), flatten it
        if isinstance(data, tuple) and len(data) >= 2:
            data = data[1]  # Take the bytes part
        
        if isinstance(data, (bytes, bytearray)):
            return bytes(data), _coerce_name(name, default_name)
        elif hasattr(data, 'read'):
            return bytes(data.read()), _coerce_name(name, default_name)
        else:
            raise TypeError(f"Expected bytes as second tuple element, got {type(data).__name__}")

    # Handle bytes/bytearray
    if isinstance(file_input, (bytes, bytearray)):
        return bytes(file_input), default_name

    # Handle path (str or Path)
    if isinstance(file_input, (str, Path)):
        p = Path(file_input)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        return p.read_bytes(), p.name

    # Handle file-like object
    if hasattr(file_input, "read"):
        try:
            data = file_input.read()
            if isinstance(data, (bytes, bytearray)):
                name = getattr(file_input, "name", default_name)
                return bytes(data), _coerce_name(name, default_name)
        except Exception as e:
            raise TypeError(f"Failed to read from file-like object: {e}")

    # Last resort: try to convert to bytes
    try:
        return bytes(file_input), default_name
    except Exception:
        raise TypeError(
            f"Unsupported file_input type: {type(file_input).__name__}. "
            f"Expected str, Path, bytes, file-like object, or (filename, bytes) tuple."
        )


# =========================================================
# DIAGNOSTIC GUARD
# =========================================================

def _diagnose_tuple_errors(func):
    """Decorator to catch and diagnose tuple-related errors."""
    import functools
    
    @functools.wraps(func)
    def wrapper(file_input, *args, **kwargs):
        try:
            return func(file_input, *args, **kwargs)
        except TypeError as e:
            if "os.PathLike object, not tuple" in str(e) or "expected str, bytes" in str(e):
                raise TypeError(
                    f"[DIAGNOSTIC] Caught path-like error in {func.__name__}()\n"
                    f"file_input type: {type(file_input).__name__}\n"
                    f"file_input repr: {repr(file_input)[:200]}\n"
                    f"Original error: {e}"
                ) from e
            raise
    return wrapper


# =========================================================
# CONCURRENCY GUARD
# =========================================================

_groq_call_lock = threading.Lock()


# =========================================================
# PDF PREPROCESSING
# =========================================================

def preprocess_pdf(file_bytes: bytes, dpi: int = 250) -> bytes:
    """
    Preprocess PDF by rendering pages to images and re-encoding.
    Helps with OCR quality for scanned documents.
    """
    try:
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
    except Exception as e:
        raise Exception(f"PDF preprocessing failed: {e}")


# =========================================================
# OCR ENGINE - Datalab
# =========================================================

PAGE_BREAK_PATTERNS = [
    re.compile(r'\n?\{(\d+)\}-{3,}\n?'),
    re.compile(r'\n?-{2,}\{(\d+)\}-{2,}\n?'),
    re.compile(r'\n-{3,}\s*Page\s*\d+\s*-{3,}\n', re.IGNORECASE),
    re.compile(r'\n\s*\[PAGE\s*(\d+)\]\s*\n', re.IGNORECASE),
    re.compile(r'\n\s*<!--\s*page\s*(\d+)\s*-->\s*\n', re.IGNORECASE),
    re.compile(r'\f'),  # Form feed character
]


def split_paginated_markdown(markdown: str, page_count_hint: int = None, log=print) -> List[str]:
    """Split markdown text into pages using various page break markers."""
    best_parts = None
    
    for pattern in PAGE_BREAK_PATTERNS:
        if pattern == re.compile(r'\f'):
            # Handle form feed separately
            parts = [p.strip() for p in markdown.split('\f') if p.strip()]
            if len(parts) > 1:
                if page_count_hint and len(parts) == page_count_hint:
                    return parts
                if best_parts is None or len(parts) > len(best_parts):
                    best_parts = parts
            continue
        
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
    
    log(f"WARNING: No page-break marker recognized. Treating as single page.")
    return [markdown.strip()]


def run_ocr(file_content: bytes, file_name: str, status_callback=None) -> List[Dict[str, Any]]:
    """
    Run OCR using Datalab API.
    Returns list of pages with raw text.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    
    file_name = _coerce_name(file_name, default_name="document.pdf")
    
    if not isinstance(file_content, (bytes, bytearray)):
        raise TypeError(f"Expected bytes, got {type(file_content).__name__}")
    
    api_key = get_api_key("DATALAB_API_KEY")
    if not api_key:
        raise Exception("DATALAB_API_KEY not found. Please set it in secrets or .env")
    
    size_mb = len(file_content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise Exception(
            f"File is {size_mb:.1f}MB, exceeds {MAX_FILE_SIZE_MB}MB limit. "
            f"Compress or split the PDF before uploading."
        )
    
    headers = {"X-API-Key": api_key}
    log(f"Submitting document to Datalab OCR... ({size_mb:.1f}MB)")
    
    try:
        resp = httpx.post(
            f"{DATALAB_BASE_URL}/api/v1/convert",
            headers=headers,
            files={"file": (file_name, file_content, "application/pdf")},
            data={
                "output_format": "markdown",
                "mode": "accurate",
                "paginate": "true"
            },
            timeout=120
        )
    except httpx.TimeoutException:
        raise Exception("Upload timeout - please try again with a smaller file")
    
    if resp.status_code != 200:
        raise Exception(f"Datalab upload failed: {resp.status_code} - {resp.text}")
    
    data = resp.json()
    if not data.get("success", True):
        raise Exception(f"Datalab upload failed: {data.get('error', 'Unknown error')}")
    
    check_url = data["request_check_url"]
    log("Document submitted, polling for OCR result...")
    
    result = None
    start_time = time.time()
    
    for attempt in range(OCR_TIMEOUT_SECONDS // OCR_POLL_INTERVAL):
        try:
            poll_resp = httpx.get(check_url, headers=headers, timeout=60)
        except httpx.TimeoutException:
            log("Poll timeout, retrying...")
            time.sleep(OCR_POLL_INTERVAL)
            continue
        
        if poll_resp.status_code != 200:
            log(f"Poll error {poll_resp.status_code}, retrying...")
            time.sleep(OCR_POLL_INTERVAL)
            continue
        
        result = poll_resp.json()
        status = result.get("status")
        
        if status == "complete":
            log("OCR complete!")
            break
        
        if status == "failed" or result.get("error"):
            raise Exception(f"OCR failed: {result.get('error', 'Unknown error')}")
        
        if attempt % 5 == 0:
            elapsed = int(time.time() - start_time)
            log(f"Still processing... ({elapsed}s elapsed)")
        
        time.sleep(OCR_POLL_INTERVAL)
    else:
        raise Exception(f"OCR timed out after {OCR_TIMEOUT_SECONDS} seconds")
    
    if not result.get("success", True):
        raise Exception(f"OCR failed: {result.get('error', 'Unknown error')}")
    
    markdown = result.get("markdown") or ""
    if not markdown.strip():
        raise Exception("OCR returned empty text. The document may be blank or unreadable.")
    
    page_count_hint = result.get("page_count")
    page_texts = split_paginated_markdown(markdown, page_count_hint, log=log)
    
    pages = []
    for idx, text in enumerate(page_texts):
        pages.append({
            "page_number": idx + 1,
            "raw_text": text
        })
    
    log(f"OCR extracted {len(pages)} page(s)")
    return pages


# =========================================================
# OCR JSON Builder
# =========================================================

def build_ocr_json(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build standard OCR JSON format."""
    return {
        "total_pages": len(pages),
        "pages": [
            {"page_number": p["page_number"], "text": p["raw_text"]}
            for p in pages
        ]
    }


# =========================================================
# REFERENCE BOOK OCR
# =========================================================

@_diagnose_tuple_errors
def process_reference(file_input, status_callback=None) -> Dict[str, Any]:
    """Process a reference book PDF."""
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    
    file_bytes, file_name = normalize_file_input(file_input, default_name="reference.pdf")
    pages = run_ocr(file_bytes, file_name, status_callback)
    log(f"Reference OCR complete: {len(pages)} page(s)")
    return build_ocr_json(pages)


# =========================================================
# TOKEN BUDGET TRACKER
# =========================================================

class TokenBudgetTracker:
    """Sliding-window token budget tracker."""
    
    def __init__(self, tpm_limit: int = TPM_LIMIT, safety_fraction: float = TPM_SAFETY_FRACTION):
        import collections
        self.tpm_limit = tpm_limit
        self.safe_limit = tpm_limit * safety_fraction
        self.events = collections.deque()
    
    def _prune(self, now: float = None):
        now = now if now is not None else time.monotonic()
        while self.events and now - self.events[0][0] >= 60:
            self.events.popleft()
    
    def used_in_window(self, now: float = None) -> int:
        now = now if now is not None else time.monotonic()
        self._prune(now)
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
        
        for ts, tok in self.events:
            freed += tok
            wait_s = max(wait_s, 60 - (now - ts))
            if freed >= needed_to_free:
                break
        
        wait_s = max(0.0, wait_s) + 0.5
        log(f"Rate limit pacing: waiting {wait_s:.1f}s before next request...")
        time.sleep(wait_s)
    
    def record_usage(self, tokens: int):
        self.events.append((time.monotonic(), tokens))
    
    def record_actual_from_error(self, used: int, limit: int):
        now = time.monotonic()
        current = self.used_in_window(now)
        if used > current:
            self.events.append((now, used - current))
        if limit:
            self.tpm_limit = limit
            self.safe_limit = limit * TPM_SAFETY_FRACTION
    
    def reset_window(self):
        self.events.clear()


# =========================================================
# GROQ LLM HELPERS
# =========================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string."""
    return int(len(text) / CHARS_PER_TOKEN_ESTIMATE) + 1


RATE_LIMIT_DETAIL_RE = re.compile(
    r'on\s+tokens\s+per\s+(minute|day)\s*\((TPM|TPD)\).*?'
    r'Limit\s+(\d+),\s*Used\s+(\d+),\s*Requested\s+(\d+).*?'
    r'try again in\s+(?:(\d+)m)?([\d.]+)s',
    re.IGNORECASE | re.DOTALL
)


def parse_rate_limit_detail(message: str) -> Optional[Dict[str, Any]]:
    """Parse Groq rate limit error details."""
    m = RATE_LIMIT_DETAIL_RE.search(message)
    if not m:
        return None
    period, limit_type, limit, used, requested, minutes, seconds = m.groups()
    wait_seconds = (int(minutes) * 60 if minutes else 0) + float(seconds)
    return {
        "limit_type": limit_type.upper(),
        "period": period.lower(),
        "limit": int(limit),
        "used": int(used),
        "requested": int(requested),
        "wait_seconds": wait_seconds,
    }


def call_groq_with_retries(client, system_prompt: str, user_prompt: str,
                           response_parser, budget: TokenBudgetTracker,
                           log, max_retries: int = MAX_GROQ_RETRIES):
    """Generic Groq call with retries and rate limiting."""
    import groq
    
    estimated_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt) + 800
    last_error = None
    skip_proactive_check = False
    
    for attempt in range(1, max_retries + 2):
        if not skip_proactive_check:
            budget.wait_if_needed(estimated_tokens, log=log)
        
        try:
            with _groq_call_lock:
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
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
        
        except groq.AuthenticationError as e:
            raise Exception(
                f"Groq API key invalid. Please check:\n"
                f"1. GROQ_API_KEY is set in secrets/.env\n"
                f"2. Key has no extra whitespace\n"
                f"3. Key is active in Groq console\n"
                f"Original: {e}"
            ) from e
        
        except (groq.RateLimitError, groq.BadRequestError) as e:
            last_error = e
            detail = parse_rate_limit_detail(str(e))
            
            if detail and detail["limit_type"] == "TPD":
                raise Exception(
                    f"Daily token quota exhausted: {detail['used']}/{detail['limit']} used. "
                    f"Reset in ~{detail['wait_seconds']/60:.0f} minutes."
                ) from e
            
            if detail:
                budget.record_actual_from_error(detail["used"], detail["limit"])
                wait = detail["wait_seconds"] + 0.5
                log(f"Rate limit (attempt {attempt}): waiting {wait:.1f}s...")
                time.sleep(wait)
                budget.reset_window()
                skip_proactive_check = True
            else:
                wait = min(5.0 * attempt, 30.0)
                log(f"Request failed (attempt {attempt}): waiting {wait:.1f}s...")
                time.sleep(wait)
        
        except Exception as e:
            last_error = e
            log(f"Attempt {attempt} failed: {e}")
            time.sleep(1)
    
    raise Exception(f"Failed after {max_retries + 1} attempts. Last error: {last_error}")


# =========================================================
# QUESTION PAPER DETECTION
# =========================================================

def chunk_pages_by_char_budget(pages: List[Dict], max_chars: int = MAX_CHARS_PER_CHUNK,
                               overlap_pages: int = CHUNK_OVERLAP_PAGES) -> List[List[Dict]]:
    """Split pages into chunks by character budget."""
    if not pages:
        return []
    
    chunks = []
    current_chunk = []
    current_chars = 0
    
    for page in pages:
        page_chars = len(page["raw_text"])
        
        if current_chunk and current_chars + page_chars > max_chars:
            chunks.append(current_chunk)
            overlap = current_chunk[-overlap_pages:] if overlap_pages > 0 else []
            current_chunk = list(overlap)
            current_chars = sum(len(p["raw_text"]) for p in current_chunk)
        
        current_chunk.append(page)
        current_chars += page_chars
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def build_qp_user_prompt(pages: List[Dict]) -> str:
    """Build user prompt for question paper detection."""
    blocks = []
    for p in pages:
        blocks.append(f"--- PAGE {p['page_number']} ---\n{p['raw_text']}")
    return "Here are the OCR'd pages shown in this chunk:\n\n" + "\n\n".join(blocks)


# =========================================================
# QUESTION PAPER DETECTION PROMPT
# =========================================================

QP_SYSTEM_PROMPT = """You are analyzing OCR text from a scanned exam booklet. The booklet contains:
1. COVER/ADMIN pages - ignore these
2. QUESTION PAPER pages - the official printed questions
3. ANSWER pages - student's handwritten answers

You are shown a PORTION of pages. For each page, decide if it's a question paper page.

CRITICAL: Student answers OFTEN start by restating the question itself. These pages look like question paper pages but are actually ANSWERS. How to tell:
- A real question paper page has SHORT, CONCISE questions (1-3 lines each)
- An answer page has LONG paragraphs of text (the student's explanation)
- If a page has 3+ paragraphs of running text, it's an ANSWER page
- If a page has short, numbered items with no long paragraphs, it's a QUESTION PAPER page

Return ONLY JSON:
{
  "question_paper_pages": [14, 16, 18],
  "questions": []
}

If no question paper pages in this chunk, return {"question_paper_pages": [], "questions": []}

Output ONLY the JSON object. No explanations. No markdown."""


def parse_qp_llm_response(content: str) -> Tuple[List[int], List[str]]:
    """Parse LLM response for question paper detection."""
    content = content.strip()
    
    # Remove markdown code fences
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()
    
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}\nContent: {content[:500]}")
    
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")
    
    qp_pages = data.get("question_paper_pages", [])
    if not isinstance(qp_pages, list):
        raise ValueError("question_paper_pages must be a list")
    
    qp_pages = [int(x) for x in qp_pages if isinstance(x, (int, str)) and str(x).isdigit()]
    return qp_pages, []


def try_split_concatenated_page_number(n: int, valid_pages: set, max_page: int) -> List[int]:
    """Attempt to recover concatenated page numbers."""
    if n in valid_pages:
        return []
    
    s = str(n)
    from itertools import product
    
    def attempt_split(s, widths):
        result = []
        i = 0
        for w in widths:
            if i + w > len(s):
                return None
            chunk = s[i:i+w]
            if chunk.startswith('0') and len(chunk) > 1:
                return None
            num = int(chunk)
            if num not in valid_pages:
                return None
            result.append(num)
            i += w
        if i != len(s):
            return None
        if len(set(result)) != len(result):
            return None
        return result
    
    max_digits = len(str(max_page))
    candidates = []
    
    for num_parts in range(2, len(s) + 1):
        for widths in product(range(1, max_digits + 1), repeat=num_parts):
            if sum(widths) != len(s):
                continue
            result = attempt_split(s, widths)
            if result:
                candidates.append(result)
    
    if not candidates:
        return []
    
    candidates.sort(key=len)
    return candidates[0]


def call_groq_for_chunk(client, pages_chunk: List[Dict], budget: TokenBudgetTracker,
                        log, max_retries: int = MAX_GROQ_RETRIES) -> Tuple[List[int], List[str]]:
    """Call Groq for a chunk of pages."""
    user_prompt = build_qp_user_prompt(pages_chunk)
    return call_groq_with_retries(
        client, QP_SYSTEM_PROMPT, user_prompt, parse_qp_llm_response,
        budget, log, max_retries
    )


def normalize_question_key(q: str) -> str:
    """Normalize question text for comparison."""
    text = q.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^\d+[\.\)]\s*[-–]?\s*', '', text)
    return text


def words_nearly_match(w1: str, w2: str) -> bool:
    """Check if two words are nearly the same (OCR-tolerant)."""
    if w1 == w2:
        return True
    if abs(len(w1) - len(w2)) > 2:
        return False
    return difflib.SequenceMatcher(None, w1, w2).ratio() >= 0.8


def is_near_duplicate_question(q1: str, q2: str) -> bool:
    """Check if two questions are near-duplicates."""
    k1, k2 = normalize_question_key(q1), normalize_question_key(q2)
    if k1 == k2:
        return True
    
    ratio = difflib.SequenceMatcher(None, k1, k2).ratio()
    if ratio < 0.90:
        return False
    
    words1 = sorted(set(re.findall(r'[a-z]{3,}', k1)))
    words2 = sorted(set(re.findall(r'[a-z]{3,}', k2)))
    if not words1 or not words2:
        return ratio >= 0.92
    
    matched = sum(1 for w1 in words1 if any(words_nearly_match(w1, w2) for w2 in words2))
    overlap = matched / max(len(words1), len(words2))
    
    return ratio >= 0.90 and overlap >= 0.92


def dedup_questions(questions: List[str]) -> List[str]:
    """Deduplicate questions preserving order."""
    unique = []
    for q in questions:
        if not any(is_near_duplicate_question(q, existing) for existing in unique):
            unique.append(q)
    return unique


def merge_chunk_results(chunk_results: List[Tuple[List[int], List[str]]]) -> Tuple[List[int], List[str]]:
    """Merge results from multiple chunks."""
    all_pages = set()
    all_questions = []
    
    for pages, questions in chunk_results:
        all_pages.update(pages)
        all_questions.extend(questions)
    
    return sorted(all_pages), dedup_questions(all_questions)


# =========================================================
# CANONICAL QUESTION EXTRACTION
# =========================================================

QUESTION_PAPER_ONLY_SYSTEM_PROMPT = """You are reading the OFFICIAL question paper pages. Extract the COMPLETE list of every distinct question/sub-part, exactly as printed.

Rules:
- If a question has labeled sub-parts like (i), (ii), (iii), extract EACH as a separate entry
- Preserve the EXACT original text
- Keep the original order

Return ONLY JSON:
{
  "questions": ["Question 1 text", "Question 2 text", ...]
}"""


def build_canonical_questions_prompt(qp_pages: List[Dict]) -> str:
    """Build prompt for canonical question extraction."""
    blocks = []
    for p in qp_pages:
        blocks.append(f"--- PAGE {p['page_number']} ---\n{p['raw_text']}")
    return "Complete question paper text:\n\n" + "\n\n".join(blocks)


def parse_canonical_questions_response(content: str) -> List[str]:
    """Parse canonical questions response."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()
    
    data = json.loads(content)
    if not isinstance(data, dict) or "questions" not in data:
        raise ValueError("Missing 'questions' key")
    
    questions = data["questions"]
    if not isinstance(questions, list):
        raise ValueError("'questions' must be a list")
    
    return [str(q).strip() for q in questions if str(q).strip()]


def extract_canonical_questions(qp_pages: List[Dict], status_callback=None) -> List[str]:
    """Extract canonical questions from question paper pages."""
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    
    if not qp_pages:
        return []
    
    from groq import Groq
    
    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found")
    
    client = Groq(api_key=api_key)
    budget = TokenBudgetTracker()
    
    user_prompt = build_canonical_questions_prompt(qp_pages)
    log(f"Extracting canonical questions from {len(qp_pages)} page(s)...")
    
    try:
        questions = call_groq_with_retries(
            client, QUESTION_PAPER_ONLY_SYSTEM_PROMPT, user_prompt,
            parse_canonical_questions_response, budget, log
        )
    except Exception as e:
        log(f"WARNING: Canonical extraction failed: {e}")
        return []
    
    log(f"Extracted {len(questions)} canonical question(s)")
    return questions


# =========================================================
# AUTO-CORRECT MISCLASSIFIED PAGES
# =========================================================

def auto_correct_misclassified_pages(pages: List[Dict], qp_indices: List[int], log=print) -> List[int]:
    """
    Auto-correct pages misclassified as question paper pages.
    Student answer pages that restate the question are often misclassified.
    """
    if len(qp_indices) < 2:
        return qp_indices
    
    # Get lengths of question paper pages
    qp_lengths = [(i, len(pages[i]["raw_text"])) for i in qp_indices]
    lengths = [length for _, length in qp_lengths]
    median = sorted(lengths)[len(lengths) // 2]
    
    # Patterns that indicate answer pages
    answer_patterns = re.compile(
        r'उत्तर|Ans|answer|समाधान|solution|इस प्रकार|इसलिए|अतः',
        re.IGNORECASE
    )
    
    corrected = []
    
    for page_idx, length in qp_lengths:
        # Check if this page is much longer than typical question pages
        is_outlier = length > max(median * 2.5, 1200)
        
        if is_outlier:
            text = pages[page_idx]["raw_text"]
            answer_count = len(answer_patterns.findall(text))
            
            # Check for long paragraphs (essay-like)
            paragraphs = [p for p in text.split('\n') if len(p.strip()) > 50]
            has_long_paragraphs = any(len(p) > 300 for p in paragraphs)
            
            # If it has answer markers AND long paragraphs, it's an answer page
            if answer_count > 3 or has_long_paragraphs:
                log(f"AUTO-CORRECT: Page {page_idx+1} moved from QP to answers (length: {length}, median: {median})")
                continue
        
        corrected.append(page_idx)
    
    return sorted(corrected)


# =========================================================
# MAIN QUESTION IDENTIFICATION
# =========================================================

def identify_questions_with_llm(pages: List[Dict], status_callback=None) -> Tuple[List[int], List[str]]:
    """
    Identify question paper pages and extract canonical questions.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    
    from groq import Groq
    
    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found")
    
    client = Groq(api_key=api_key)
    budget = TokenBudgetTracker()
    
    # Split into chunks
    chunks = chunk_pages_by_char_budget(pages)
    log(f"Split {len(pages)} pages into {len(chunks)} chunks")
    
    valid_pages = {p["page_number"] for p in pages}
    max_page = max(valid_pages) if valid_pages else 0
    
    chunk_results = []
    failures = []
    
    for i, chunk in enumerate(chunks):
        page_nums = [p["page_number"] for p in chunk]
        log(f"Analyzing chunk {i+1}/{len(chunks)} (pages {page_nums})...")
        
        try:
            qp_pages, _ = call_groq_for_chunk(client, chunk, budget, log)
        except Exception as e:
            log(f"WARNING: Chunk {i+1} failed: {e}")
            failures.append(str(e))
            continue
        
        # Recover concatenated page numbers
        recovered = []
        invalid = []
        for pn in qp_pages:
            if pn in valid_pages:
                recovered.append(pn)
            else:
                split_result = try_split_concatenated_page_number(pn, valid_pages, max_page)
                if split_result:
                    log(f"Recovered concatenated: {pn} -> {split_result}")
                    recovered.extend(split_result)
                else:
                    invalid.append(pn)
        
        if invalid:
            log(f"Ignoring invalid page numbers: {invalid}")
        
        chunk_results.append((sorted(set(recovered)), []))
    
    if failures and not chunk_results:
        raise Exception(f"All chunks failed. First error: {failures[0]}")
    
    # Merge results
    qp_pages_merged, _ = merge_chunk_results(chunk_results)
    qp_indices = sorted(pn - 1 for pn in qp_pages_merged)
    
    # Auto-correct misclassified pages
    corrected_indices = auto_correct_misclassified_pages(pages, qp_indices, log)
    if corrected_indices != qp_indices:
        log(f"Auto-correct applied: {len(qp_indices) - len(corrected_indices)} page(s) moved")
        qp_indices = corrected_indices
    
    log(f"Question paper pages: {[i+1 for i in qp_indices]}")
    
    # Extract canonical questions
    qp_pages_full = [pages[i] for i in qp_indices]
    questions = extract_canonical_questions(qp_pages_full, status_callback)
    
    return qp_indices, questions


# =========================================================
# ANSWER MAPPING
# =========================================================

ANSWER_MAP_SYSTEM_PROMPT = """You are mapping student answers to official questions.

Given:
1. Official questions labeled [REF-A], [REF-B], etc.
2. Student answer text with line numbers [0], [1], etc.

For EACH question, find where the answer starts and ends.

Return ONLY JSON:
{
  "answers": [
    {"ref": "REF-A", "start_line": 0, "end_line": 15},
    {"ref": "REF-B", "start_line": 16, "end_line": 30}
  ]
}

If an answer is not found, omit it."""


def build_answer_map_prompt(numbered_lines: List[Tuple[int, str]], questions: List[str]) -> str:
    """Build prompt for answer mapping."""
    q_block = "\n".join(f"[REF-{chr(65+i)}] {q}" for i, q in enumerate(questions))
    l_block = "\n".join(f"[{idx}] {text}" for idx, text in numbered_lines)
    return f"QUESTIONS:\n{q_block}\n\nSTUDENT ANSWERS:\n{l_block}"


def parse_answer_map_response(content: str) -> List[Dict[str, Any]]:
    """Parse answer mapping response."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()
    
    data = json.loads(content)
    if not isinstance(data, dict) or "answers" not in data:
        return []
    
    answers = data["answers"]
    if not isinstance(answers, list):
        return []
    
    result = []
    for item in answers:
        if not isinstance(item, dict):
            continue
        if "ref" not in item or "start_line" not in item or "end_line" not in item:
            continue
        try:
            result.append({
                "ref": str(item["ref"]).strip().upper(),
                "start_line": int(item["start_line"]),
                "end_line": int(item["end_line"]),
            })
        except (ValueError, TypeError):
            continue
    
    return result


# =========================================================
# ANSWER START DETECTION
# =========================================================

ANSWER_START_RE = re.compile(
    r'^\s*(?:Ans(?:wer)?[.\s:-]+\d*|उत्तर\s*\d*\s*[\-\:]?|प्र[०.\s]*\d*|Q\.?\s*\d+[.\s:-])',
    re.IGNORECASE
)


def normalize_for_overlap(text: str) -> str:
    """Normalize text for overlap comparison."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    return text


QUESTION_STOPWORDS = {
    'how', 'are', 'the', 'views', 'state', 'with', 'theme', 'examine',
    'write', 'detailed', 'note', 'their', 'corresponding', 'why', 'does',
    'plot', 'plan', 'comment', 'discuss', 'explain', 'describe', 'and',
    'what', 'when', 'where', 'which', 'who', 'integrated', 'analyse',
    'analyze', 'critically', 'briefly', 'elaborate', 'illustrate', 'for',
    'from', 'this', 'that', 'these', 'those', 'into', 'about', 'role',
    'significance', 'importance', 'short', 'long', 'play', 'text',
    'the', 'a', 'an', 'of', 'to', 'in', 'on', 'at', 'by', 'with', 'without'
}


def distinctive_words(text: str, max_words: int = 20) -> List[str]:
    """Extract distinctive words from text."""
    words = re.findall(r'[a-z]{3,}', normalize_for_overlap(text))[:max_words]
    return sorted(set(w for w in words if w not in QUESTION_STOPWORDS))


def line_starts_new_answer(line: str, questions: List[str], min_fraction: float = 0.5):
    """Detect if a line starts a new answer."""
    # Check for explicit label
    label_match = ANSWER_START_RE.match(line)
    if label_match:
        num_match = re.search(r'\d+', label_match.group(0))
        if num_match:
            label_num = num_match.group(0)
            for i, q in enumerate(questions):
                q_num = re.match(r'\s*(\d+)', q)
                if q_num and q_num.group(1) == label_num:
                    return i
        return -1
    
    # Check for question restatement (no label)
    line_words = sorted(set(re.findall(r'[a-z]{3,}', normalize_for_overlap(line))[:25]))
    if not line_words:
        return None
    
    for i, q in enumerate(questions):
        q_words = distinctive_words(q)
        if not q_words:
            continue
        
        matched = sum(
            1 for w in q_words
            if any(words_nearly_match(w, lw) for lw in line_words)
        )
        
        required = max(1, round(len(q_words) * min_fraction * 0.7))
        if matched >= required and len(line_words) >= 4:
            return i
    
    # Check for question number pattern
    num_pattern = re.match(r'^\s*(\d+)[\.\)]\s', line)
    if num_pattern:
        num = num_pattern.group(1)
        for i, q in enumerate(questions):
            if re.match(r'^\s*' + re.escape(num) + r'[\.\)]', q):
                return i
    
    return None


# =========================================================
# CHUNK LINES BY CHARACTER BUDGET
# =========================================================

def chunk_lines_by_char_budget(numbered_lines: List[Tuple[int, str]], questions: List[str],
                               max_chars: int = ANSWER_MAP_MAX_CHARS_PER_CHUNK,
                               abs_max_chars: int = ANSWER_MAP_ABSOLUTE_MAX_CHARS) -> List[List[Tuple[int, str]]]:
    """
    Split lines into chunks, preserving answer boundaries.
    """
    if not numbered_lines:
        return []
    
    chunks = []
    current_chunk = []
    current_chars = 0
    past_target = False
    current_q_idx = None
    total_lines = len(numbered_lines)
    
    for idx, (line_idx, text) in enumerate(numbered_lines):
        line_chars = len(text)
        
        # Detect new answer start
        matched_q = line_starts_new_answer(text, questions)
        is_new_start = matched_q is not None and (
            matched_q == -1 or matched_q != current_q_idx
        )
        
        # Always start first chunk at line 0
        if idx == 0:
            current_chunk = [(line_idx, text)]
            current_chars = line_chars
            if is_new_start and matched_q != -1:
                current_q_idx = matched_q
            continue
        
        # Check if past target
        if current_chunk and current_chars + line_chars > max_chars:
            past_target = True
        
        should_break = False
        
        # Break at new answer start if past target
        if past_target and is_new_start:
            should_break = True
        
        # Don't break near end of document
        remaining = total_lines - idx
        if should_break and remaining <= 15:
            if is_new_start and remaining < 20:
                should_break = False
                past_target = False
        
        # Absolute max - only break if not near end
        if current_chunk and current_chars + line_chars > abs_max_chars:
            if remaining > 10:
                should_break = True
            else:
                should_break = False
        
        # Execute break
        if should_break:
            if len(current_chunk) < 10 and len(chunks) > 0:
                chunks[-1].extend(current_chunk)
            else:
                chunks.append(current_chunk)
            current_chunk = [(line_idx, text)]
            current_chars = line_chars
            past_target = False
            current_q_idx = matched_q if (is_new_start and matched_q != -1) else current_q_idx
            continue
        
        # Add to current chunk
        current_chunk.append((line_idx, text))
        current_chars += line_chars
        
        if is_new_start and matched_q != -1:
            current_q_idx = matched_q
    
    # Add last chunk
    if current_chunk:
        if len(current_chunk) < 10 and len(chunks) > 0:
            chunks[-1].extend(current_chunk)
        else:
            chunks.append(current_chunk)
    
    return chunks


# =========================================================
# RESOLVE OVERLAPPING ANSWER RANGES
# =========================================================

def resolve_overlapping_ranges(answer_ranges: List[Dict]) -> List[Dict]:
    """Resolve overlapping answer ranges."""
    sorted_ranges = sorted(answer_ranges, key=lambda r: r["start_line"])
    resolved = []
    
    for i, r in enumerate(sorted_ranges):
        r = dict(r)
        if i + 1 < len(sorted_ranges):
            next_start = sorted_ranges[i + 1]["start_line"]
            if r["end_line"] >= next_start:
                r["end_line"] = next_start - 1
        if r["end_line"] >= r["start_line"]:
            resolved.append(r)
    
    return resolved


# =========================================================
# STRIP RESTATEMENTS
# =========================================================

QUESTION_PREFIX_RE = re.compile(
    r'^\s*(?:'
    r'Ans(?:wer)?\s*\d+\s*[.:\-]?\s*'
    r'|Ans(?:wer)?\s*[.:\-]\s*'
    r'|उत्तर\s*\d*\s*[\-\:]\s*'
    r'|प्र[०.\s]+\d+[.\s:-]*'
    r'|प्रश्न[.\s]+\d+[.\s:-]*'
    r'|Q\.?\s*\d+[.\s:-]*'
    r')',
    re.IGNORECASE
)


def strip_question_prefix(answer_text: str) -> str:
    """Strip question prefix (Ans-, उत्तर-, etc.) from answer."""
    text = answer_text
    for _ in range(2):
        new_text = QUESTION_PREFIX_RE.sub('', text, count=1).strip()
        if new_text == text:
            break
        text = new_text
    return text


PARENT_INSTRUCTION_PREFIX = re.compile(
    r'^\s*\d+[\.\)]?\s*(?:\([ivx]+\)|\([a-z]\)|\([क-घ]\))?\s*'
    r'(?:identify and explain the following|write (?:short )?notes? on|'
    r'comment on|explain the following|discuss the following)\s*:?\s*',
    re.IGNORECASE
)


def strip_question_echo(answer_text: str, question_text: str) -> str:
    """Strip full question restatement from answer start."""
    # Remove parent instruction prefix from question
    q_core = PARENT_INSTRUCTION_PREFIX.sub('', question_text).strip()
    if not q_core:
        q_core = question_text
    
    q_norm = normalize_for_overlap(q_core)
    q_word_count = len(q_norm.split())
    if q_word_count == 0:
        return answer_text
    
    answer_words = answer_text.split()
    if not answer_words:
        return answer_text
    
    min_n = max(3, int(q_word_count * 0.7))
    max_n = min(len(answer_words), int(q_word_count * 1.3) + 2)
    
    best_strip = 0
    best_ratio = 0.0
    
    for n in range(min_n, max_n + 1):
        prefix = " ".join(answer_words[:n])
        prefix_norm = normalize_for_overlap(prefix)
        ratio = difflib.SequenceMatcher(None, prefix_norm, q_norm).ratio()
        if ratio >= 0.75 and ratio > best_ratio:
            best_ratio = ratio
            best_strip = n
    
    if best_strip > 0:
        remaining = " ".join(answer_words[best_strip:]).strip()
        remaining = re.sub(r'^(?:Answer\s*[-:]\s*)', '', remaining, flags=re.IGNORECASE)
        return remaining.strip()
    
    return answer_text


# =========================================================
# NOISE FILTERING
# =========================================================

NOISE_RE = re.compile(
    r'(?:Teacher\'?s?\s*Signature'
    r'|Tancher\'?s?\s*Signature'
    r'|Facebook\'?s?\s*Signature'
    r'|PAGE\s*NO'
    r'|^\s*DATE\b'
    r'|Neel?\s*Kamal'
    r'|Neal?\s*Kamal'
    r'|Need?\s*Komal'
    r'|Nod\s*Komal'
    r'|TAKMA\s*SINAN'
    r'|^\s*\d{1,3}\s*$)',
    re.IGNORECASE
)


def is_noise(line: str) -> bool:
    """Check if a line is noise that should be filtered."""
    return bool(NOISE_RE.search(line))


# =========================================================
# MAIN ANSWER MAPPING
# =========================================================

def map_answers_with_llm(answer_lines: List[str], questions: List[str], 
                         status_callback=None) -> Dict[str, str]:
    """
    Map questions to answers using LLM-based boundary detection.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    
    from groq import Groq
    
    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found")
    
    client = Groq(api_key=api_key)
    budget = TokenBudgetTracker()
    
    ref_to_q = {f"REF-{chr(65+i)}": q for i, q in enumerate(questions)}
    numbered_lines = list(enumerate(answer_lines))
    
    chunks = chunk_lines_by_char_budget(numbered_lines, questions)
    log(f"Split {len(answer_lines)} lines into {len(chunks)} chunks")
    
    all_ranges = []
    failures = []
    zero_matches = 0
    
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        
        line_range = f"{chunk[0][0]}-{chunk[-1][0]}"
        log(f"Mapping chunk {i+1}/{len(chunks)} (lines {line_range})...")
        
        user_prompt = build_answer_map_prompt(chunk, questions)
        
        try:
            chunk_ranges = call_groq_with_retries(
                client, ANSWER_MAP_SYSTEM_PROMPT, user_prompt,
                parse_answer_map_response, budget, log
            )
        except Exception as e:
            log(f"WARNING: Chunk {i+1} failed: {e}")
            failures.append(str(e))
            continue
        
        if not chunk_ranges:
            zero_matches += 1
        
        # Validate ranges
        valid_indices = {idx for idx, _ in chunk}
        min_idx, max_idx = min(valid_indices), max(valid_indices)
        
        for r in chunk_ranges:
            if r["ref"] not in ref_to_q:
                log(f"WARNING: Unknown ref {r['ref']}")
                continue
            if min_idx <= r["start_line"] <= max_idx and min_idx <= r["end_line"] <= max_idx:
                all_ranges.append(r)
            else:
                log(f"WARNING: Out-of-range for {r['ref']}: {r['start_line']}-{r['end_line']}")
    
    # Deduplicate
    best_by_ref = {}
    for r in all_ranges:
        existing = best_by_ref.get(r["ref"])
        if existing is None or (r["end_line"] - r["start_line"]) > (existing["end_line"] - existing["start_line"]):
            best_by_ref[r["ref"]] = r
    
    deduped = list(best_by_ref.values())
    resolved = resolve_overlapping_ranges(deduped)
    
    log(f"Matched {len(resolved)} of {len(questions)} questions")
    
    if not resolved:
        if failures and len(failures) == len(chunks):
            raise Exception(f"All chunks failed. First: {failures[0]}")
        elif zero_matches == len(chunks):
            sample = [l for l in answer_lines[:15] if l.strip()][:5]
            raise Exception(f"No answers found. Sample text: {sample}")
    
    # Extract answers
    qa_map = {}
    for r in resolved:
        start, end = r["start_line"], r["end_line"]
        lines = [
            answer_lines[j] for j in range(start, end + 1)
            if 0 <= j < len(answer_lines) and answer_lines[j].strip() and not is_noise(answer_lines[j])
        ]
        q_text = ref_to_q[r["ref"]]
        answer = " ".join(lines).strip()
        answer = strip_question_prefix(answer)
        answer = strip_question_echo(answer, q_text)
        qa_map[q_text] = answer
    
    return qa_map


# =========================================================
# SANITY CHECK
# =========================================================

def sanity_check_answer_pages(answer_lines: List[str], num_questions: int, log=print) -> bool:
    """Check if answer pages contain enough text."""
    total_chars = sum(len(l) for l in answer_lines)
    avg = total_chars / max(num_questions, 1)
    MIN_PLAUSIBLE = 200
    
    if avg < MIN_PLAUSIBLE:
        log(
            f"WARNING: Only {total_chars} chars for {num_questions} questions "
            f"({avg:.0f} chars/question) - may be misclassified"
        )
        return False
    return True


# =========================================================
# COMPLETE PIPELINE
# =========================================================

@_diagnose_tuple_errors
def process_pdf(file_input, status_callback=None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Complete PDF processing pipeline.
    
    Returns:
    - ocr_json: Complete OCR text
    - qa_pairs: Question-answer pairs
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    
    # Normalize input
    file_bytes, file_name = normalize_file_input(file_input, default_name="document.pdf")
    log(f"Processing: {file_name}")
    
    # OCR
    log("Running OCR...")
    pages = run_ocr(file_bytes, file_name, status_callback)
    ocr_json = build_ocr_json(pages)
    log(f"OCR complete: {len(pages)} pages")
    
    # Identify questions
    log("Identifying question paper pages...")
    qp_indices, questions = identify_questions_with_llm(pages, status_callback)
    log(f"Found {len(qp_indices)} question pages, {len(questions)} questions")
    
    if not qp_indices:
        raise Exception("No question paper pages found")
    if not questions:
        raise Exception("No questions extracted")
    
    # Extract answer pages
    answer_indices = [i for i in range(len(pages)) if i not in qp_indices]
    answer_pages = [pages[i] for i in answer_indices]
    log(f"Answer pages: {[i+1 for i in answer_indices]}")
    
    # Flatten answer lines
    answer_lines = []
    for page in answer_pages:
        for line in page["raw_text"].split("\n"):
            if line.strip() and not is_noise(line):
                answer_lines.append(line.strip())
    log(f"Flattened {len(answer_lines)} answer lines")
    
    # Sanity check
    if not sanity_check_answer_pages(answer_lines, len(questions), log):
        log("WARNING: Answer pages seem too short - check page classification")
    
    # Map answers
    log("Mapping answers to questions...")
    qa_map = map_answers_with_llm(answer_lines, questions, status_callback)
    
    # Build output
    qa_pairs = []
    for q in questions:
        qa_pairs.append({
            "question": q,
            "answer": qa_map.get(q, ""),
            "matched": q in qa_map
        })
    
    matched = sum(1 for p in qa_pairs if p["matched"])
    log(f"Matched {matched}/{len(questions)} questions")
    
    return ocr_json, qa_pairs


# =========================================================
# SAVE OUTPUTS
# =========================================================

def save_outputs(ocr_json: Dict, qa_pairs: List[Dict], output_dir: str = ".",
                 base_name: str = "document") -> Tuple[str, str]:
    """Save outputs to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    ocr_path = os.path.join(output_dir, f"{base_name}_ocr.json")
    qa_path = os.path.join(output_dir, f"{base_name}_qa_pairs.json")
    
    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(ocr_json, f, ensure_ascii=False, indent=2)
    
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    return ocr_path, qa_path


# =========================================================
# MAIN - For testing
# =========================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <pdf_file>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    print(f"Processing: {pdf_path}")
    
    try:
        ocr_json, qa_pairs = process_pdf(pdf_path)
        ocr_path, qa_path = save_outputs(ocr_json, qa_pairs)
        print(f"\n✅ Done!")
        print(f"OCR saved to: {ocr_path}")
        print(f"QA pairs saved to: {qa_path}")
        print(f"Total questions: {len(qa_pairs)}")
        print(f"Matched: {sum(1 for p in qa_pairs if p['matched'])}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
