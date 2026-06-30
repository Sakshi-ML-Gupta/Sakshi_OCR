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
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
import collections

# =========================================================
# API KEYS
# =========================================================

def get_api_key(name: str) -> Optional[str]:
    """Get API key from Streamlit secrets or environment variables."""
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

def _normalize_file_input(file_input: Any, default_name: str = "document.pdf") -> Tuple[bytes, str]:
    """Normalize various input types to (bytes, filename)."""
    if isinstance(file_input, tuple):
        if len(file_input) < 2:
            raise ValueError(f"Tuple must have at least (filename, bytes), got {len(file_input)} items")
        name, data = file_input[0], file_input[1]
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"Expected bytes as second tuple element, got {type(data).__name__}")
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


def _coerce_name(name: Any, default_name: str = "document.pdf") -> str:
    """Extract filename from various types."""
    if isinstance(name, (tuple, list)):
        return default_name
    if not name:
        return default_name
    try:
        return Path(str(name)).name or default_name
    except Exception:
        return default_name


# =========================================================
# DIAGNOSTIC GUARD
# =========================================================

def _diagnose_tuple_errors(func):
    import functools
    @functools.wraps(func)
    def wrapper(file_input, *args, **kwargs):
        try:
            return func(file_input, *args, **kwargs)
        except TypeError as e:
            if "os.PathLike object, not tuple" in str(e):
                raise TypeError(
                    f"[DIAGNOSTIC] Caught error in {func.__name__}(). "
                    f"file_input: type={type(file_input).__name__}, repr={file_input!r}. "
                    f"Original: {e}"
                ) from e
            raise
    return wrapper


# =========================================================
# CONCURRENCY GUARD
# =========================================================

_groq_call_lock = threading.Lock()


# =========================================================
# PREPROCESS PDF - IMPROVED
# =========================================================

def preprocess_pdf(file_bytes: bytes, dpi: int = 250) -> bytes:
    """Preprocess PDF with improved cleaning."""
    src_doc = fitz.open(stream=file_bytes, filetype="pdf")
    out_doc = fitz.open()
    
    for page in src_doc:
        # Clean page content first
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
# TEXT CLEANING - NEW & IMPROVED
# =========================================================

def clean_ocr_text(text: str) -> str:
    """
    Comprehensive text cleaning for OCR output.
    Removes: page numbers, headers, footers, garbage markers,
    question numbering artifacts, and common OCR noise.
    """
    # Remove common headers/footers
    patterns = [
        r'^\s*Page\s*\d+\s*of\s*\d+\s*$',  # Page X of Y
        r'^\s*Page\s*\d+\s*$',              # Page X
        r'^\s*-\s*\d+\s*-\s*$',             # - X -
        r'^\s*\[Page\s*\d+\]\s*$',          # [Page X]
        r'^\s*P\.?\s*No\.?\s*\d+\s*$',      # P.No. X
        r'^\s*प्र\.\s*नं\.?\s*\d+\s*$',     # प्र. नं. X
        r'^\s*प्रश्न\s*नं\.?\s*\d+\s*$',   # प्रश्न नं. X
        r'^\s*Q\.?\s*No\.?\s*\d+\s*$',      # Q.No. X
        r'^\s*Question\s*No\.?\s*\d+\s*$',  # Question No. X
        r'^\s*#\s*.*$',                      # Lines starting with #
        r'^\s*प्रश्नोत्तर\s*.*$',           # प्रश्नोत्तर
        r'^\s*prashan\s*.*$',               # prashan
        r'^\s*[Pp]rashan\s*.*$',            # Prashan
        r'^\s*Answer\s*No\.?\s*\d+\s*$',    # Answer No. X
        r'^\s*उत्तर\s*नं\.?\s*\d+\s*$',     # उत्तर नं. X
        r'^\s*Roll\s*No\.?\s*\d+\s*$',      # Roll No. X
        r'^\s*Enrolment\s*No\.?\s*\d+\s*$', # Enrolment No. X
        r'^\s*[A-Z]{2,}\s*\d+\s*$',          # IGNOU123 etc.
        r'^\s*[A-Z]+\s*-\s*\d+\s*$',         # IGNOU - 123
        r'^\s*[A-Z]+\s*/\s*\d+\s*$',         # IGNOU/123
        r'^\s*[A-Z]+\s*\d+\s*/\s*\d+\s*$',   # IGNOU123/456
        r'^\s*[A-Z]+\s*\(\d+\)\s*$',         # IGNOU(123)
    ]
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line matches any pattern
        is_noise = False
        for pattern in patterns:
            if re.search(pattern, line, re.IGNORECASE):
                is_noise = True
                break
                
        # Check for very short lines that are likely noise
        if len(line) < 3 and not any(c.isalpha() for c in line):
            is_noise = True
            
        # Check for lines with too many special characters
        special_count = sum(1 for c in line if not c.isalnum() and not c.isspace())
        if len(line) > 0 and special_count / len(line) > 0.5:
            is_noise = True
            
        if not is_noise:
            cleaned_lines.append(line)
    
    # Join and clean extra whitespace
    text = ' '.join(cleaned_lines)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()


# =========================================================
# OCR -- Datalab (Chandra model)
# =========================================================

DATALAB_BASE_URL = "https://www.datalab.to"

PAGE_BREAK_PATTERNS = [
    re.compile(r'\n?\{(\d+)\}-{3,}\n?'),
    re.compile(r'\n?-{2,}\{(\d+)\}-{2,}\n?'),
    re.compile(r'\n-{3,}\s*Page\s*\d+\s*-{3,}\n', re.IGNORECASE),
    re.compile(r'\n\s*\[PAGE\s*(\d+)\]\s*\n', re.IGNORECASE),
    re.compile(r'\n\s*<!--\s*page\s*(\d+)\s*-->\s*\n', re.IGNORECASE),
    re.compile(r'\n\s*Page\s*(\d+)\s*\n', re.IGNORECASE),
]


def _split_paginated_markdown(markdown: str, page_count_hint: int = None, log=print) -> list:
    """Split markdown into pages using various page break patterns."""
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


def run_ocr(file_content: bytes, file_name: str, status_callback=None) -> List[Dict]:
    """Run OCR using Datalab's Chandra model."""
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
        raise Exception(f"File is {size_mb:.1f}MB, exceeds {MAX_MB}MB limit.")

    headers = {"X-API-Key": api_key}
    log(f"Submitting to Datalab (Chandra OCR)... ({size_mb:.1f}MB)")

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

    max_polls = 150
    poll_interval = 2
    result = None

    for attempt in range(max_polls):
        poll_resp = httpx.get(check_url, headers=headers, timeout=60)
        if poll_resp.status_code != 200:
            raise Exception(f"Poll error {poll_resp.status_code}: {poll_resp.text}")

        result = poll_resp.json()
        status = result.get("status")

        if status == "complete":
            log("OCR complete")
            break
        elif status == "failed" or result.get("error"):
            raise Exception(f"Conversion failed: {result.get('error')}")

        if attempt % 5 == 0:
            log(f"Still processing... ({attempt * poll_interval}s elapsed)")
        time.sleep(poll_interval)
    else:
        raise Exception("OCR timed out after 5 minutes")

    if not result.get("success", True):
        raise Exception(f"Conversion error: {result.get('error')}")

    markdown = result.get("markdown") or ""
    if not markdown.strip():
        raise Exception("Empty markdown output")

    # Clean the markdown text
    markdown = clean_ocr_text(markdown)

    page_count_hint = result.get("page_count")
    page_texts = _split_paginated_markdown(markdown, page_count_hint, log=log)

    pages = []
    for idx, text in enumerate(page_texts):
        pages.append({
            "page_number": idx + 1,
            "raw_text": text
        })

    log(f"OCR done -- {len(pages)} page(s)")
    return pages


# =========================================================
# BUILD OCR JSON
# =========================================================

def build_ocr_json(pages: list) -> dict:
    return {
        "total_pages": len(pages),
        "pages": [{"page_number": p["page_number"], "text": p["raw_text"]} for p in pages]
    }


# =========================================================
# REFERENCE BOOK OCR
# =========================================================

@_diagnose_tuple_errors
def process_reference(file_input, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    file_bytes, file_name = _normalize_file_input(file_input, default_name="reference.pdf")
    pages = run_ocr(file_bytes, file_name, status_callback)
    log(f"Reference OCR complete -- {len(pages)} page(s)")
    return build_ocr_json(pages)


# =========================================================
# TOKEN BUDGET TRACKER - IMPROVED
# =========================================================

TPM_LIMIT = 8000
TPM_SAFETY_FRACTION = 0.85
CHARS_PER_TOKEN_ESTIMATE = 2.0
MAX_CHARS_PER_CHUNK = 8000  # Increased for long answers
CHUNK_OVERLAP_PAGES = 1

class _TokenBudgetTracker:
    def __init__(self, tpm_limit=TPM_LIMIT, safety_fraction=TPM_SAFETY_FRACTION):
        self.tpm_limit = tpm_limit
        self.safe_limit = tpm_limit * safety_fraction
        self.events = collections.deque()

    def _prune(self, now=None):
        now = now if now is not None else time.monotonic()
        while self.events and now - self.events[0][0] >= 60:
            self.events.popleft()

    def used_in_window(self, now=None) -> int:
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
        log(f"Pacing requests: {used:.0f} tokens used, +{upcoming_tokens} upcoming. "
            f"Waiting {wait_s:.1f}s...")
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


_RATE_LIMIT_DETAIL_RE = re.compile(
    r'on\s+tokens\s+per\s+(minute|day)\s*\((TPM|TPD)\).*?'
    r'Limit\s+(\d+),\s*Used\s+(\d+),\s*Requested\s+(\d+).*?'
    r'try again in\s+(?:(\d+)m)?([\d.]+)s',
    re.IGNORECASE | re.DOTALL
)


def _parse_rate_limit_detail(message: str):
    m = _RATE_LIMIT_DETAIL_RE.search(message)
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


# =========================================================
# LLM PROMPTS - IMPROVED
# =========================================================

QP_SYSTEM_PROMPT = """You are analyzing OCR text from a student exam assignment booklet.

Your task: Identify which pages are the official question paper pages and extract the complete question list.

Rules:
1. QUESTION PAPER pages contain official exam questions (instructions/prompts directed at students)
2. ANSWER pages contain student responses (longer text with explanations)
3. ADMIN pages contain enrolment numbers, programme codes, etc.

CRITICAL: A page that looks like a question but is very long (>3x median length) is likely an ANSWER page where the student restated the question.

Return ONLY valid JSON:
{
  "question_paper_pages": [14, 16, 18],
  "questions": ["1. Question text (10)", "2. Another question (10)"]
}

If no question pages found, return {"question_paper_pages": [], "questions": []}"""


QUESTION_PAPER_ONLY_SYSTEM_PROMPT = """You are extracting questions from the official question paper.

Rules:
1. Extract EVERY distinct question/sub-part
2. For multi-part questions (i), (ii), (iii), split each into separate entries
3. Preserve EXACT original text
4. Keep original numbering
5. Return in printed order

Return ONLY valid JSON:
{
  "questions": ["1.(i) First sub-part", "1.(ii) Second sub-part", ...]
}"""


ANSWER_MAP_SYSTEM_PROMPT = """You are mapping answers to questions.

Given:
1. Questions with REF labels (REF-A, REF-B, etc.)
2. Line-numbered answer text

For EACH question, find the line range [start_line, end_line] where its answer appears.

IMPORTANT: 
- Answers can be LONG (5-10+ pages). Don't truncate!
- Include ALL lines that belong to that answer
- A new answer starts when the student references the next question
- If an answer isn't present, omit it

Return ONLY valid JSON:
{
  "answers": [
    {"ref": "REF-A", "start_line": 12, "end_line": 48},
    {"ref": "REF-B", "start_line": 49, "end_line": 95}
  ]
}"""


# =========================================================
# LLM CALL WITH RETRIES - IMPROVED
# =========================================================

def _estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN_ESTIMATE) + 1


def _call_groq_with_retries(client, system_prompt: str, user_prompt: str,
                             response_parser, budget: _TokenBudgetTracker,
                             log, max_retries: int = 4):
    """Generic Groq call with retry and rate-limit handling."""
    estimated_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt) + 800
    last_error = None
    skip_next_proactive_check = False

    for attempt in range(1, max_retries + 2):
        if skip_next_proactive_check:
            skip_next_proactive_check = False
        else:
            budget.wait_if_needed(estimated_tokens, log=log)

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
                    max_tokens=4096,  # Increased for long answers
                )
            budget.record_usage(estimated_tokens)
            content = response.choices[0].message.content
            return response_parser(content)

        except Exception as e:
            last_error = e
            detail = _parse_rate_limit_detail(str(e))
            
            if detail and detail["limit_type"] == "TPD":
                raise Exception(f"Daily quota exhausted: {detail['used']}/{detail['limit']}. "
                              f"Resets in {detail['wait_seconds']/60:.0f} min.") from e
            
            if detail:
                budget.record_actual_from_error(detail["used"], detail["limit"])
                log(f"Rate limit (attempt {attempt}): waiting {detail['wait_seconds']+0.5:.1f}s")
                time.sleep(detail["wait_seconds"] + 0.5)
                budget.reset_window()
                skip_next_proactive_check = True
            else:
                wait_s = min(5.0 * attempt, 30.0)
                log(f"Error (attempt {attempt}): {e}. Waiting {wait_s:.1f}s")
                time.sleep(wait_s)

    raise Exception(f"Failed after {max_retries+1} attempts. Last error: {last_error}")


# =========================================================
# QUESTION IDENTIFICATION - IMPROVED
# =========================================================

def _parse_qp_llm_response(content: str) -> tuple:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}\nContent: {content[:500]}")

    if not isinstance(data, dict) or "question_paper_pages" not in data or "questions" not in data:
        raise ValueError(f"Missing required keys. Got: {list(data.keys())}")

    qp_pages = [int(x) for x in data["question_paper_pages"]]
    questions = [str(x).strip() for x in data["questions"] if str(x).strip()]
    
    return qp_pages, questions


def _try_split_concatenated_page_number(n: int, valid_page_numbers: set, max_page: int) -> list:
    """Recover concatenated page numbers."""
    if n in valid_page_numbers:
        return []

    s = str(n)
    max_digits = len(str(max_page))
    
    def split_attempt(s, widths):
        result = []
        i = 0
        for w in widths:
            if i + w > len(s):
                return None
            chunk = s[i:i+w]
            if chunk.startswith('0') and len(chunk) > 1:
                return None
            num = int(chunk)
            if num not in valid_page_numbers:
                return None
            result.append(num)
            i += w
        if i != len(s) or len(set(result)) != len(result):
            return None
        return result

    candidates = []
    for num_parts in range(2, min(len(s) + 1, 4)):
        for widths in __import__('itertools').product(range(1, max_digits + 1), repeat=num_parts):
            if sum(widths) != len(s):
                continue
            result = split_attempt(s, widths)
            if result:
                candidates.append(result)

    return candidates[0] if candidates else []


def identify_questions_with_llm(pages: list, status_callback=None) -> tuple:
    """Identify question paper pages and extract questions."""
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    from groq import Groq

    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found")

    client = Groq(api_key=api_key)
    budget = _TokenBudgetTracker()

    # Clean pages before processing
    for page in pages:
        page["raw_text"] = clean_ocr_text(page["raw_text"])

    chunks = _chunk_pages_by_char_budget(pages)
    log(f"Split {len(pages)} page(s) into {len(chunks)} chunk(s)")

    valid_page_numbers = {p["page_number"] for p in pages}
    max_page_number = max(valid_page_numbers) if valid_page_numbers else 0
    chunk_results = []

    for i, chunk in enumerate(chunks):
        page_nums = [p["page_number"] for p in chunk]
        log(f"Analyzing chunk {i+1}/{len(chunks)} (pages {page_nums})...")

        try:
            user_prompt = _build_qp_user_prompt(chunk)
            qp_pages, _ = _call_groq_with_retries(
                client, QP_SYSTEM_PROMPT, user_prompt,
                _parse_qp_llm_response, budget, log
            )
        except Exception as e:
            log(f"WARNING: chunk {i+1} failed: {e}")
            continue

        recovered_pages = []
        for pn in qp_pages:
            if pn in valid_page_numbers:
                recovered_pages.append(pn)
            else:
                split_result = _try_split_concatenated_page_number(
                    pn, valid_page_numbers, max_page_number
                )
                if split_result:
                    log(f"Recovered concatenated: {pn} -> {split_result}")
                    recovered_pages.extend(split_result)

        chunk_results.append((sorted(set(recovered_pages)), []))

    # Merge results
    all_qp_pages = set()
    for qp_pages, _ in chunk_results:
        all_qp_pages.update(qp_pages)

    qp_page_indices = sorted(p - 1 for p in all_qp_pages)
    log(f"Question paper pages: {[p+1 for p in qp_page_indices]}")

    # Check for unusually long question pages (likely misclassified answers)
    if len(qp_page_indices) >= 2:
        qp_lengths = [len(pages[i]["raw_text"]) for i in qp_page_indices]
        median = sorted(qp_lengths)[len(qp_lengths)//2]
        for i, length in zip(qp_page_indices, qp_lengths):
            if length > max(median * 3, 2000):
                log(f"WARNING: Page {i+1} is very long ({length} chars) - likely an answer page")

    # Extract canonical questions
    qp_pages = [pages[i] for i in qp_page_indices]
    questions = extract_canonical_questions(qp_pages, status_callback)

    log(f"Extracted {len(questions)} canonical question(s)")
    return qp_page_indices, questions


def _chunk_pages_by_char_budget(pages: list, max_chars: int = MAX_CHARS_PER_CHUNK,
                                 overlap_pages: int = CHUNK_OVERLAP_PAGES) -> list:
    """Chunk pages by character budget with overlap."""
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


def _build_qp_user_prompt(pages: list) -> str:
    blocks = [f"--- PAGE {p['page_number']} ---\n{p['raw_text']}" for p in pages]
    return "Here are the OCR'd pages:\n\n" + "\n\n".join(blocks)


def extract_canonical_questions(qp_pages: list, status_callback=None) -> list:
    """Extract canonical question list from question paper pages."""
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
    budget = _TokenBudgetTracker()

    user_prompt = _build_canonical_questions_prompt(qp_pages)
    log(f"Extracting canonical questions from {len(qp_pages)} page(s)...")

    try:
        questions = _call_groq_with_retries(
            client, QUESTION_PAPER_ONLY_SYSTEM_PROMPT, user_prompt,
            _parse_canonical_questions_response, budget, log
        )
    except Exception as e:
        log(f"WARNING: canonical extraction failed: {e}")
        return []

    log(f"Extracted {len(questions)} canonical question(s)")
    return questions


def _build_canonical_questions_prompt(qp_pages: list) -> str:
    blocks = [f"--- PAGE {p['page_number']} ---\n{p['raw_text']}" for p in qp_pages]
    return "Complete question paper text:\n\n" + "\n\n".join(blocks)


def _parse_canonical_questions_response(content: str) -> list:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if not isinstance(data, dict) or "questions" not in data:
        raise ValueError(f"Missing 'questions' key")

    questions = data["questions"]
    if not isinstance(questions, list):
        raise ValueError(f"'questions' must be a list")

    return [str(q).strip() for q in questions if str(q).strip()]


# =========================================================
# ANSWER MAPPING - COMPLETELY REWRITTEN FOR LONG ANSWERS
# =========================================================

def _line_starts_new_answer(line: str, questions: list) -> Optional[int]:
    """Check if a line starts a new answer for a specific question."""
    # Check for formal labels
    label_patterns = [
        r'^\s*Ans(?:wer)?\s*\d+\s*[.:\-]\s*',
        r'^\s*Ans(?:wer)?\s*[.:\-]\s*',
        r'^\s*उत्तर\s*\d*\s*[\-\:]\s*',
        r'^\s*प्र[०.\s]+\d+[.\s:-]*',
        r'^\s*प्रश्न[.\s]+\d+[.\s:-]*',
        r'^\s*Q\.?\s*\d+[.\s:-]*',
    ]
    
    for pattern in label_patterns:
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            num_match = re.search(r'\d+', match.group(0))
            if num_match:
                label_num = num_match.group(0)
                for i, q in enumerate(questions):
                    q_num_match = re.match(r'\s*(\d+)', q)
                    if q_num_match and q_num_match.group(1) == label_num:
                        return i
            return -1  # Ambiguous label - treat as new start

    return None


def _chunk_lines_by_answer_boundaries(numbered_lines: list, questions: list,
                                       max_chars: int = 10000) -> list:
    """
    Chunk answer lines by answer boundaries.
    FIX: Handles very long answers (9-10 pages) by keeping complete answers together.
    """
    if not numbered_lines:
        return []

    chunks = []
    current_chunk = []
    current_chars = 0
    current_question_idx = None

    for idx, text in numbered_lines:
        line_chars = len(text)
        matched_idx = _line_starts_new_answer(text, questions)
        
        # Check if this is a genuine new answer
        is_new_answer = (matched_idx is not None and 
                        (matched_idx == -1 or matched_idx != current_question_idx))
        
        # Start new chunk if:
        # 1. We're past max_chars AND this is a new answer start
        # 2. OR we're about to exceed absolute max (safety)
        if (current_chunk and current_chars + line_chars > max_chars and is_new_answer) or \
           (current_chunk and current_chars + line_chars > max_chars * 2):
            chunks.append(current_chunk)
            current_chunk = []
            current_chars = 0

        if is_new_answer and matched_idx != -1:
            current_question_idx = matched_idx

        current_chunk.append((idx, text))
        current_chars += line_chars

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def map_answers_with_llm(answer_lines: list, questions: list, status_callback=None) -> dict:
    """Map each question to its answer using LLM-based boundary detection."""
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    from groq import Groq

    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found")

    client = Groq(api_key=api_key)
    budget = _TokenBudgetTracker()

    # Clean answer lines
    answer_lines = [clean_ocr_text(line) for line in answer_lines if line.strip()]

    if not answer_lines:
        log("No answer lines found")
        return {}

    numbered_lines = list(enumerate(answer_lines))
    ref_to_question = {f"REF-{chr(65+i)}": q for i, q in enumerate(questions)}
    
    # Chunk by answer boundaries to keep long answers intact
    chunks = _chunk_lines_by_answer_boundaries(numbered_lines, questions)
    log(f"Split {len(answer_lines)} lines into {len(chunks)} chunk(s)")

    all_ranges = []
    
    for i, chunk in enumerate(chunks):
        line_range = f"{chunk[0][0]}-{chunk[-1][0]}"
        log(f"Mapping chunk {i+1}/{len(chunks)} (lines {line_range})...")

        user_prompt = _build_answer_map_user_prompt(chunk, questions)
        
        try:
            chunk_ranges = _call_groq_with_retries(
                client, ANSWER_MAP_SYSTEM_PROMPT, user_prompt,
                _parse_answer_map_llm_response, budget, log
            )
        except Exception as e:
            log(f"WARNING: chunk {i+1} failed: {e}")
            continue

        # Validate ranges
        valid_indices = {idx for idx, _ in chunk}
        min_idx, max_idx = min(valid_indices), max(valid_indices)
        
        for r in chunk_ranges:
            if r["ref"] in ref_to_question and min_idx <= r["start_line"] <= max_idx:
                # Extend end_line to include all content until next answer
                if r["end_line"] < max_idx:
                    # Check if next few lines might be part of this answer
                    next_lines = [l for idx, l in numbered_lines[r["end_line"]+1:r["end_line"]+4] 
                                 if idx <= max_idx]
                    if next_lines and not any(_line_starts_new_answer(l, questions) is not None 
                                             for l in next_lines):
                        r["end_line"] = min(r["end_line"] + 3, max_idx)
                
                all_ranges.append(r)

    # Deduplicate and resolve overlaps
    best_by_ref = {}
    for r in all_ranges:
        existing = best_by_ref.get(r["ref"])
        if existing is None or (r["end_line"] - r["start_line"]) > (existing["end_line"] - existing["start_line"]):
            best_by_ref[r["ref"]] = r

    resolved_ranges = _resolve_overlapping_answer_ranges(list(best_by_ref.values()))
    
    log(f"Final mapping: {len(resolved_ranges)} of {len(questions)} question(s) matched")

    # Extract answers
    qa_map = {}
    for r in resolved_ranges:
        start, end = r["start_line"], r["end_line"]
        # Include ALL lines, don't filter aggressively
        verbatim_lines = [answer_lines[j] for j in range(start, end + 1) 
                         if 0 <= j < len(answer_lines)]
        
        original_question = ref_to_question[r["ref"]]
        answer_text = " ".join(verbatim_lines).strip()
        
        # Clean but don't truncate
        answer_text = strip_question_restatement(answer_text)
        answer_text = strip_full_question_echo(answer_text, original_question)
        
        if answer_text:
            qa_map[original_question] = answer_text

    return qa_map


def _build_answer_map_user_prompt(numbered_lines: list, questions: list) -> str:
    """Build user prompt for answer mapping."""
    questions_block = "\n".join(
        f"[REF-{chr(65+i)}] {q}" for i, q in enumerate(questions)
    )
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in numbered_lines)
    return (
        f"OFFICIAL QUESTIONS (use REF labels):\n{questions_block}\n\n"
        f"STUDENT'S ANSWERS (line-numbered):\n{lines_block}"
    )


def _parse_answer_map_llm_response(content: str) -> list:
    """Parse LLM response for answer mapping."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if not isinstance(data, dict) or "answers" not in data:
        raise ValueError(f"Missing 'answers' key")

    answers = data["answers"]
    if not isinstance(answers, list):
        raise ValueError(f"'answers' must be a list")

    result = []
    for item in answers:
        if isinstance(item, dict) and "ref" in item and "start_line" in item and "end_line" in item:
            try:
                result.append({
                    "ref": str(item["ref"]).strip().upper(),
                    "start_line": int(item["start_line"]),
                    "end_line": int(item["end_line"]),
                })
            except (ValueError, TypeError):
                continue

    return result


def _resolve_overlapping_answer_ranges(answer_ranges: list) -> list:
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


def strip_question_restatement(answer_text: str) -> str:
    """Strip leading question restatement labels."""
    patterns = [
        r'^\s*Ans(?:wer)?\s*\d*\s*[.:\-]\s*',
        r'^\s*उत्तर\s*\d*\s*[\-\:]\s*',
        r'^\s*प्र[०.\s]+\d+[.\s:-]*',
        r'^\s*प्रश्न[.\s]+\d+[.\s:-]*',
        r'^\s*Q\.?\s*\d+[.\s:-]*',
    ]
    
    text = answer_text
    for _ in range(2):
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE, count=1)
        text = text.strip()
        if text == answer_text:
            break
    
    return text


def strip_full_question_echo(answer_text: str, question_text: str) -> str:
    """Strip full question echo from answer start."""
    # Normalize for comparison
    q_norm = re.sub(r'[^\w\s]', ' ', question_text.lower())
    q_norm = re.sub(r'\s+', ' ', q_norm).strip()
    q_words = q_norm.split()
    
    if len(q_words) < 3:
        return answer_text
    
    answer_words = answer_text.split()
    if len(answer_words) < len(q_words):
        return answer_text
    
    # Check if answer starts with question echo
    prefix = " ".join(answer_words[:len(q_words)])
    prefix_norm = re.sub(r'[^\w\s]', ' ', prefix.lower())
    prefix_norm = re.sub(r'\s+', ' ', prefix_norm).strip()
    
    similarity = difflib.SequenceMatcher(None, prefix_norm, q_norm).ratio()
    
    if similarity >= 0.6:
        # Strip the echo
        remaining = " ".join(answer_words[len(q_words):])
        return remaining.strip()
    
    return answer_text


# =========================================================
# COMPLETE PIPELINE - FINAL
# =========================================================

@_diagnose_tuple_errors
def process_pdf(file_input, status_callback=None):
    """Complete PDF processing pipeline."""
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    # Step 1: Normalize input
    file_bytes, file_name = _normalize_file_input(file_input, default_name="document.pdf")
    
    # Step 2: Preprocess PDF
    log("Preprocessing PDF...")
    file_bytes = preprocess_pdf(file_bytes, dpi=250)
    
    # Step 3: OCR
    pages = run_ocr(file_bytes, file_name, status_callback)
    
    # Step 4: Build OCR JSON
    log("Building OCR JSON...")
    ocr_json = build_ocr_json(pages)
    
    # Step 5: Identify questions
    qp_page_indices, official_questions = identify_questions_with_llm(pages, status_callback)
    log(f"Found {len(official_questions)} questions on {len(qp_page_indices)} page(s)")
    
    if not official_questions:
        raise Exception("No questions extracted from the document.")
    
    # Step 6: Extract answer pages
    answer_page_indices = [i for i in range(len(pages)) if i not in qp_page_indices]
    answer_pages = [pages[i] for i in answer_page_indices]
    log(f"Found {len(answer_pages)} answer page(s)")
    
    # Step 7: Flatten answer lines
    answer_lines = []
    for page in answer_pages:
        lines = page["raw_text"].split("\n")
        for line in lines:
            line = clean_ocr_text(line)
            if line and len(line) > 5:  # Keep meaningful content
                answer_lines.append(line)
    
    log(f"Flattened {len(answer_lines)} answer lines")
    
    # Step 8: Validate answer pages
    if not _sanity_check_answer_pages(answer_lines, len(official_questions), log):
        raise Exception(
            f"Answer pages seem too short ({len(answer_lines)} lines for "
            f"{len(official_questions)} questions). Check page classification."
        )
    
    # Step 9: Map answers
    log("Mapping answers to questions...")
    qa_map = map_answers_with_llm(answer_lines, official_questions, status_callback)
    
    # Step 10: Build QA pairs
    qa_pairs = []
    matched_count = 0
    for q in official_questions:
        answer = qa_map.get(q, "")
        if answer:
            matched_count += 1
        qa_pairs.append({
            "question": q,
            "answer": answer,
            "matched": bool(answer)
        })
    
    log(f"Matched {matched_count} of {len(official_questions)} questions")
    
    if matched_count == 0:
        raise Exception("No answers could be matched to questions.")
    
    return ocr_json, qa_pairs


def _sanity_check_answer_pages(answer_lines: list, num_questions: int, log=print) -> bool:
    """Check if answer pages contain enough content."""
    total_chars = sum(len(line) for line in answer_lines)
    avg_chars_per_question = total_chars / max(num_questions, 1)
    
    # Very conservative threshold (50 chars per question means almost nothing)
    MIN_PLAUSIBLE_CHARS_PER_QUESTION = 100
    
    if avg_chars_per_question < MIN_PLAUSIBLE_CHARS_PER_QUESTION:
        log(f"WARNING: Only {total_chars} chars for {num_questions} questions. "
            f"Looks like misclassified pages.")
        return False
    
    return True


def save_outputs(ocr_json: dict, qa_pairs: list, output_dir: str = ".",
                  base_name: str = "document") -> tuple:
    """Save outputs to JSON files."""
    ocr_path = os.path.join(output_dir, f"{base_name}_ocr.json")
    qa_path = os.path.join(output_dir, f"{base_name}_qa_pairs.json")
    
    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(ocr_json, f, ensure_ascii=False, indent=2)
    
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    return ocr_path, qa_path


# =========================================================
# NOISE DETECTION - SIMPLIFIED
# =========================================================

NOISE_RE = re.compile(
    r'(?:Teacher\'?s?\s*Signature'
    r'|Tancher\'?s?\s*Signature'
    r'|PAGE\s*NO'
    r'|^\s*DATE\b'
    r'|^\s*\d{1,3}\s*$)',
    re.IGNORECASE
)


def is_noise(line: str) -> bool:
    """Check if a line is noise."""
    return bool(NOISE_RE.search(line))


# =========================================================
# MAIN - FOR DIRECT USAGE
# =========================================================

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"Processing: {pdf_path}")
        
        ocr_json, qa_pairs = process_pdf(pdf_path)
        ocr_path, qa_path = save_outputs(ocr_json, qa_pairs, base_name="output")
        
        print(f"\n✅ Done! Outputs saved to:")
        print(f"  - OCR: {ocr_path}")
        print(f"  - QA Pairs: {qa_path}")
        print(f"\n📊 Summary:")
        print(f"  - Total pages: {ocr_json['total_pages']}")
        print(f"  - Questions: {len(qa_pairs)}")
        print(f"  - Matched: {sum(1 for p in qa_pairs if p['matched'])}")
    else:
        print("Usage: python script.py <path_to_pdf>")
