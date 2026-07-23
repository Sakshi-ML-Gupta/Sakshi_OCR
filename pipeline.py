import os
import io
import re
import json
import time
import difflib
import threading
from concurrent.futures import ThreadPoolExecutor
import fitz
import httpx
from pathlib import Path
# =========================================================
# SEMANTIC MATCHING FOR ANSWER CLEANING
# =========================================================
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")

# =========================================================
# SAFE LOGGING (never let a broken status_callback crash the pipeline)
# =========================================================
def _make_safe_logger(status_callback=None):
    """
    Returns a log(msg) function that always prints, and only best-effort calls
    status_callback. Streamlit's status_callback breaks with NoSessionContext when
    called from a background thread (e.g. our batch ThreadPoolExecutor workers) --
    previously that exception propagated straight out of a warning-log call and took
    down the entire pipeline. Now it's swallowed here so a UI-logging hiccup can never
    abort real work.
    """
    def log(msg):
        print(msg)
        if status_callback:
            try:
                status_callback(msg)
            except Exception:
                pass
    return log
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
            raise ValueError(
                f"Tuple file_input must have at least (filename, bytes), got {len(file_input)} items"
            )
        name, data = file_input[0], file_input[1]
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(
                f"Expected bytes as second tuple element, got {type(data).__name__}"
            )
        return bytes(data), _coerce_name(name, default_name)
    if isinstance(file_input, (bytes, bytearray)):
        return bytes(file_input), default_name
    if isinstance(file_input, (str, Path)):
        p = Path(file_input)
        return p.read_bytes(), p.name
    if hasattr(file_input, "read"):
        data = file_input.read()
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(
                f"file_input.read() returned {type(data).__name__}, expected bytes. "
                f"Open the file in binary mode ('rb')."
            )
        name = getattr(file_input, "name", default_name)
        return bytes(data), _coerce_name(name, default_name)
    raise TypeError(
        f"Unsupported file_input type: {type(file_input).__name__}. "
        f"Expected str, Path, bytes, a file-like object with .read(), "
        f"or a (filename, bytes) tuple."
    )
def _coerce_name(name, default_name="document.pdf"):
    if isinstance(name, (tuple, list)):
        return default_name
    if not name:
        return default_name
    try:
        return Path(str(name)).name or default_name
    except Exception:
        return default_name
def _diagnose_tuple_errors(func):
    import functools
    @functools.wraps(func)
    def wrapper(file_input, *args, **kwargs):
        try:
            return func(file_input, *args, **kwargs)
        except TypeError as e:
            if "os.PathLike object, not tuple" in str(e):
                raise TypeError(
                    f"[DIAGNOSTIC] Caught the 'os.PathLike, not tuple' error INSIDE "
                    f"{func.__name__}(), not before it -- this means the bug genuinely "
                    f"is somewhere in this module's call chain. "
                    f"file_input received: type={type(file_input).__name__}, "
                    f"repr={file_input!r}. Original error: {e}"
                ) from e
            raise
    return wrapper
# Was a single Lock() before -- that fully serialized every Groq call even across
# threads, so parallel dispatch bought nothing. A semaphore lets a few requests be
# in flight over the network at once (still respects the token-budget pacing below),
# which is what actually cuts wall-clock time when we batch multiple chunks.
GROQ_MAX_CONCURRENT_CALLS = int(os.getenv("GROQ_MAX_CONCURRENT_CALLS", "2"))
_groq_call_semaphore = threading.Semaphore(GROQ_MAX_CONCURRENT_CALLS)
_budget_lock = threading.Lock()
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
# OCR -- Datalab (Chandra model) via /convert endpoint
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
    log(
        f"WARNING: No page-break marker recognized in Datalab output "
        f"(length={len(markdown)} chars, page_count_hint={page_count_hint}). "
        f"Treating entire document as a single page. "
        f"First 200 chars: {markdown[:200]!r}"
    )
    return [markdown.strip()]
def run_ocr(file_content: bytes, file_name: str, status_callback=None):
    log = _make_safe_logger(status_callback)
    file_name = _coerce_name(file_name, default_name="document.pdf")
    if not isinstance(file_content, (bytes, bytearray)):
        raise TypeError(
            f"run_ocr() expected file_content as bytes, got {type(file_content).__name__}"
        )
    api_key = get_api_key("DATALAB_API_KEY")
    if not api_key:
        raise Exception("DATALAB_API_KEY not found in secrets or environment")
    size_mb = len(file_content) / (1024 * 1024)
    MAX_MB  = 45
    if size_mb > MAX_MB:
        raise Exception(
            f"File is {size_mb:.1f}MB, which exceeds the {MAX_MB}MB upload limit. "
            f"Try compressing the PDF or splitting it into smaller files before uploading."
        )
    headers = {"X-API-Key": api_key}
    log(f"Submitting document to Datalab (Chandra OCR)... ({size_mb:.1f}MB)")
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
    if resp.status_code != 200:
        raise Exception(f"Datalab submit error {resp.status_code}: {resp.text}")
    data = resp.json()
    if not data.get("success", True):
        raise Exception(f"Datalab submit failed: {data.get('error')}")
    check_url = data["request_check_url"]
    log("Document submitted -- polling for OCR result...")
    max_polls = 150
    poll_interval = 2
    result = None
    for attempt in range(max_polls):
        poll_resp = httpx.get(check_url, headers=headers, timeout=60)
        if poll_resp.status_code != 200:
            raise Exception(f"Datalab poll error {poll_resp.status_code}: {poll_resp.text}")
        result = poll_resp.json()
        status = result.get("status")
        if status == "complete":
            log("OCR complete -- parsing pages...")
            break
        if status == "failed" or result.get("error"):
            raise Exception(f"Datalab conversion failed: {result.get('error')}")
        if attempt % 5 == 0:
            log(f"Still processing... ({attempt * poll_interval}s elapsed)")
        time.sleep(poll_interval)
    else:
        raise Exception("Datalab conversion timed out after 5 minutes")
    if not result.get("success", True):
        raise Exception(f"Datalab conversion error: {result.get('error')}")
    markdown = result.get("markdown") or ""
    if not markdown.strip():
        raise Exception("Datalab returned empty markdown output")
    page_count_hint = result.get("page_count")
    page_texts = _split_paginated_markdown(markdown, page_count_hint, log=log)
    pages = []
    for idx, text in enumerate(page_texts):
        pages.append({
            "page_number": idx + 1,
            "raw_text":    text
        })
    log(f"OCR done -- {len(pages)} page(s) extracted")
    if len(pages) == 1 and size_mb > 1.0:
        log(
            f"WARNING: Only 1 page extracted from a {size_mb:.1f}MB file. "
            f"This usually means the page-break marker format was not recognized. "
            f"Markdown length: {len(markdown)} chars."
        )
    return pages
import hashlib
def run_ocr_cached(file_bytes, file_name, status_callback=None, cache_dir="./.ocr_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha256(file_bytes).hexdigest()
    cache_path = os.path.join(cache_dir, f"{key}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    pages = run_ocr(file_bytes, file_name, status_callback)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False)
    return pages
# =========================================================
# BUILD OCR JSON
# =========================================================
def build_ocr_json(pages: list) -> dict:
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
def process_reference(file_input, status_callback=None):
    log = _make_safe_logger(status_callback)
    file_bytes, file_name = _normalize_file_input(file_input, default_name="reference.pdf")
    pages = run_ocr_cached(file_bytes, file_name, status_callback)
    log(f"Reference OCR complete -- {len(pages)} page(s)")
    return build_ocr_json(pages)
# =========================================================
# LLM-BASED QUESTION PAPER / QUESTION DETECTION (Groq)
# =========================================================
GROQ_MODEL = "openai/gpt-oss-120b"
TPM_LIMIT = int(os.getenv("GROQ_TPM_LIMIT", "8000"))
TPM_SAFETY_FRACTION = 0.85
CHARS_PER_TOKEN_ESTIMATE = 2.0
MAX_CHARS_PER_CHUNK = 13000
CHUNK_OVERLAP_PAGES = 1
QP_SYSTEM_PROMPT = """OCR text from a scanned exam/assignment booklet (any institution/subject/language). Pages are one of:
1. ADMIN/COVER: roll no., course code, name, letterhead, blank sheets -- no question or answer content.
2. QUESTION PAPER: the official printed numbered questions -- prompts DIRECTED at the student ("discuss", "explain", "write notes on", a "?", etc., in whatever language). May show mark allocations ("10","20").
3. ANSWER pages: the student's own long OCR'd response, often restating/labeling a question briefly then writing an extended answer. Numbered sub-points INSIDE that answer are the student's own content, not separate questions.
You see only a chunk of pages (order not guaranteed; some may be carried-over context). Classify each page shown.
KEY TRAP: students often restate the question as their answer's opening sentence (e.g. "Discuss X. The concept of X is..."). That page is the START OF AN ANSWER, not the question paper, even though it uses prompt verbs. Tells: it runs much longer than a real printed question would; prose reads like a developing argument, not a terse instruction; or the same/similar question already appears on a page you're more confident is the real, concise question paper. When unsure, brevity = real question paper, length = answer restatement -- exclude the long one.
Other rules:
- Numbered items following an "answer" label (Ans/उत्तर/etc., any script) or long explanatory prose = answer page, not question paper, even with multiple numbered lines.
- When genuinely unsure a page is a question-paper page, leave it out of question_paper_pages (and don't extract its items as questions).
- Admin/cover pages go in admin_pages (excluded from both question and answer text).
- Preserve exact original text/numbering of real questions -- no paraphrasing, renumbering, translating.
Return ONLY this JSON (no fences, no commentary):
{"question_paper_pages": [14, 16, 18], "admin_pages": [1, 2], "questions": ["1. Example question. (10)", "2. Another. (10)"]}
Page-number arrays must be individual comma-separated ints (e.g. [14,16,18], never merged like [141618]); a page never appears in both lists; empty question_paper_pages is valid if this chunk has none."""
def _chunk_pages_by_char_budget(pages: list, max_chars: int = MAX_CHARS_PER_CHUNK,
                                  overlap_pages: int = CHUNK_OVERLAP_PAGES) -> list:
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
    blocks = []
    for p in pages:
        blocks.append(f"--- PAGE {p['page_number']} ---\n{p['raw_text']}")
    return "Here are the OCR'd pages shown in this chunk:\n\n" + "\n\n".join(blocks)
def _estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN_ESTIMATE) + 1
class _TokenBudgetTracker:
    def __init__(self, tpm_limit=TPM_LIMIT, safety_fraction=TPM_SAFETY_FRACTION):
        import collections
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
        already_warned_oversized = False
        while True:
            with _budget_lock:
                now = time.monotonic()
                used = self.used_in_window(now)
                projected = used + upcoming_tokens
                if projected <= self.safe_limit:
                    # Reserve immediately, inside the same lock, so a concurrent
                    # thread checking a microsecond later already sees this usage.
                    self.events.append((now, upcoming_tokens))
                    return
                # CRITICAL: if this single request's own size already exceeds the
                # entire safe budget, no amount of waiting will ever fix it -- there
                # is nothing to "free up" (used can be 0 and it still won't fit).
                # Waiting forever here was an infinite-loop bug. Proceed anyway and
                # let Groq's own 429 (with its actual retry-after) handle it if needed.
                if upcoming_tokens >= self.safe_limit:
                    if not already_warned_oversized:
                        log(
                            f"NOTE: this request alone (~{upcoming_tokens} tokens) is at or "
                            f"above the tracked safe budget ({self.safe_limit:.0f}) -- proactive "
                            f"pacing can't help here, sending it directly. If Groq rejects it "
                            f"with a 429, the retry logic will back off using Groq's own "
                            f"reported wait time."
                        )
                        already_warned_oversized = True
                    self.events.append((now, upcoming_tokens))
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
            log(
                f"Proactively pacing requests: {used:.0f} tokens used in the last 60s, "
                f"+{upcoming_tokens} upcoming would exceed safe budget "
                f"({self.safe_limit:.0f}). Waiting {wait_s:.1f}s before sending next chunk..."
            )
            time.sleep(wait_s)
    def record_usage(self, tokens: int):
        with _budget_lock:
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
    @property
    def window_start(self):
        return time.monotonic()
    @window_start.setter
    def window_start(self, value):
        pass
    @property
    def window_tokens(self):
        return self.used_in_window()
    @window_tokens.setter
    def window_tokens(self, value):
        if value == 0:
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
def _parse_qp_llm_response(content: str) -> tuple:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM did not return valid JSON: {e}\nRaw content (first 500 chars): {content[:500]!r}"
        )
    if not isinstance(data, dict):
        raise ValueError(f"LLM response must be a JSON object, got: {type(data).__name__}")
    if "question_paper_pages" not in data or "questions" not in data:
        raise ValueError(
            f"LLM response missing required keys. Got keys: {list(data.keys())}"
        )
    qp_pages = data["question_paper_pages"]
    questions = data["questions"]
    admin_pages = data.get("admin_pages", [])
    if not isinstance(qp_pages, list):
        raise ValueError(f"question_paper_pages must be a list, got: {type(qp_pages).__name__}")
    qp_pages = [int(x) for x in qp_pages]
    if not isinstance(admin_pages, list):
        raise ValueError(f"admin_pages must be a list, got: {type(admin_pages).__name__}")
    admin_pages = [int(x) for x in admin_pages]
    if not isinstance(questions, list):
        raise ValueError(f"questions must be a list, got: {type(questions).__name__}")
    questions = [str(x).strip() for x in questions if str(x).strip()]
    return qp_pages, questions, admin_pages
def _try_split_concatenated_page_number(n: int, valid_page_numbers: set, max_page: int) -> list:
    if n in valid_page_numbers:
        return []
    s = str(n)
    max_digits = len(str(max_page))
    from itertools import product
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
        if i != len(s):
            return None
        if len(set(result)) != len(result):
            return None
        return result
    candidates = []
    for num_parts in range(2, len(s) + 1):
        for widths in product(range(1, max_digits + 1), repeat=num_parts):
            if sum(widths) != len(s):
                continue
            result = split_attempt(s, widths)
            if result:
                candidates.append(result)
    if not candidates:
        return []
    candidates.sort(key=len)
    return candidates[0]
def _call_groq_with_retries(client, system_prompt: str, user_prompt: str,
                              response_parser, budget: "_TokenBudgetTracker",
                              log, max_retries: int = 4):
    import groq
    estimated_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt) + 800
    last_error = None
    skip_next_proactive_check = False
    for attempt in range(1, max_retries + 2):
        if skip_next_proactive_check:
            skip_next_proactive_check = False
            reserved_via_wait = False
        else:
            budget.wait_if_needed(estimated_tokens, log=log)
            reserved_via_wait = True
        try:
            with _groq_call_semaphore:
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
            if not reserved_via_wait:
                budget.record_usage(estimated_tokens)
            content = response.choices[0].message.content
            return response_parser(content)
        except groq.AuthenticationError as e:
            raise Exception(
                f"Groq API rejected the API key (401 Invalid API Key). "
                f"This will NOT be fixed by retrying. Things to check:\n"
                f"  1. Is GROQ_API_KEY actually set in your environment or "
                f"st.secrets? (A missing key often falls back to None or "
                f"an empty string, which Groq also rejects as invalid.)\n"
                f"  2. Does the key have any extra whitespace, quotes, or "
                f"a line break copied in by accident?\n"
                f"  3. Has the key been revoked or rotated in your Groq "
                f"console (https://console.groq.com/keys)?\n"
                f"  4. If using st.secrets, did you restart the Streamlit "
                f"app after adding/changing the secret? Streamlit does "
                f"not always hot-reload secrets.toml changes.\n"
                f"Original error: {e}"
            ) from e
        except (groq.RateLimitError, groq.BadRequestError) as e:
            last_error = e
            detail = _parse_rate_limit_detail(str(e))
            if detail and detail["limit_type"] == "TPD":
                raise Exception(
                    f"Groq daily token quota (TPD) exhausted: "
                    f"{detail['used']}/{detail['limit']} tokens used today, "
                    f"{detail['requested']} more requested. This will reset "
                    f"in approximately {detail['wait_seconds']/60:.0f} minute(s). "
                    f"Retrying within this run will not help -- either wait "
                    f"for the daily reset, or upgrade your Groq tier at "
                    f"https://console.groq.com/settings/billing. "
                    f"(If you're processing the same document more than once "
                    f"per click/run, check for duplicate calls -- that doubles "
                    f"daily token consumption and exhausts this quota twice "
                    f"as fast.)"
                ) from e
            if detail:
                budget.record_actual_from_error(detail["used"], detail["limit"])
                log(
                    f"Chunk LLM call hit a rate/size limit (attempt {attempt}): "
                    f"{detail['limit_type']} limit={detail['limit']}, "
                    f"used={detail['used']}, requested={detail['requested']}. "
                    f"Waiting {detail['wait_seconds'] + 0.5:.1f}s (Groq-reported) "
                    f"before retrying..."
                )
                time.sleep(detail["wait_seconds"] + 0.5)
                budget.reset_window()
                skip_next_proactive_check = True
            else:
                wait_s = 5.0 * attempt
                log(
                    f"Chunk LLM call hit a rate/size limit (attempt {attempt}): {e}. "
                    f"Waiting {wait_s:.1f}s before retrying..."
                )
                time.sleep(wait_s)
        except Exception as e:
            last_error = e
            log(f"Chunk LLM call/parse attempt {attempt} failed: {e}")
            time.sleep(1)
    raise Exception(
        f"Chunk LLM call failed after {max_retries + 1} attempts. Last error: {last_error}"
    )
def _call_groq_for_chunk(client, pages_chunk: list, budget: "_TokenBudgetTracker",
                          log, max_retries: int = 4) -> tuple:
    user_prompt = _build_qp_user_prompt(pages_chunk)
    return _call_groq_with_retries(
        client, QP_SYSTEM_PROMPT, user_prompt, _parse_qp_llm_response,
        budget, log, max_retries
    )
def _normalize_question_key(q: str) -> str:
    text = q.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^\d+[\.\)]\s*[-–]?\s*', '', text)
    return text
def _words_nearly_match(w1: str, w2: str) -> bool:
    if w1 == w2:
        return True
    if abs(len(w1) - len(w2)) > 2:
        return False
    return difflib.SequenceMatcher(None, w1, w2).ratio() >= 0.8
def _is_near_duplicate_question(q1: str, q2: str) -> bool:
    k1, k2 = _normalize_question_key(q1), _normalize_question_key(q2)
    if k1 == k2:
        return True
    ratio = difflib.SequenceMatcher(None, k1, k2).ratio()
    if ratio < 0.90:
        return False
    words1 = sorted(set(re.findall(r'[a-z]{3,}', k1)))
    words2 = sorted(set(re.findall(r'[a-z]{3,}', k2)))
    if not words1 or not words2:
        return ratio >= 0.92
    matched = sum(1 for w1 in words1 if any(_words_nearly_match(w1, w2) for w2 in words2))
    overlap = matched / max(len(words1), len(words2))
    return ratio >= 0.90 and overlap >= 0.92
def _dedup_questions(questions: list) -> list:
    unique = []
    for q in questions:
        if not any(_is_near_duplicate_question(q, existing) for existing in unique):
            unique.append(q)
    return unique
def _merge_chunk_results(chunk_results: list) -> tuple:
    all_qp_pages = set()
    all_admin_pages = set()
    all_questions = []
    for qp_pages, questions, admin_pages in chunk_results:
        all_qp_pages.update(qp_pages)
        all_admin_pages.update(admin_pages)
        all_questions.extend(questions)
    deduped_questions = _dedup_questions(all_questions)
    return sorted(all_qp_pages), deduped_questions, sorted(all_admin_pages)
QUESTION_PAPER_ONLY_SYSTEM_PROMPT = """Complete, exact text of the OFFICIAL question paper pages (not the student's answers), in order. Extract the full clean list of every distinct question/sub-part exactly as printed, in printed order.
Multi-part rule: if a numbered question has labeled sub-parts -- (i)/(ii)/(iii), (a)/(b)/(c), (क)/(ख), or 1./2./3. used as sub-parts -- split EACH into its own entry, not merged. Keep each entry self-contained: carry the parent instruction forward or at least keep the label (e.g. "1.(i)", "1.(ii)"). Decide this once, consistently, for the whole set. Preserve exact original text (no paraphrase/translation); output in the same printed order, sub-parts grouped and ordered under their parent.
Return ONLY this JSON (no fences, no commentary):
{"questions": ["<exact text 1>", "<exact text 2>", ...]}"""
def _build_canonical_questions_prompt(qp_pages: list) -> str:
    blocks = []
    for p in qp_pages:
        blocks.append(f"--- PAGE {p['page_number']} ---\n{p['raw_text']}")
    return (
        "Here is the COMPLETE text of all question paper pages, in order:\n\n"
        + "\n\n".join(blocks)
    )
def _parse_canonical_questions_response(content: str) -> list:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}\nRaw (first 500 chars): {content[:500]!r}")
    if not isinstance(data, dict) or "questions" not in data:
        raise ValueError(f"Response missing 'questions' key. Got: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}")
    questions = data["questions"]
    if not isinstance(questions, list):
        raise ValueError(f"'questions' must be a list, got: {type(questions).__name__}")
    return [str(q).strip() for q in questions if str(q).strip()]
def extract_canonical_questions(qp_pages: list, status_callback=None) -> list:
    log = _make_safe_logger(status_callback)
    if not qp_pages:
        return []
    from groq import Groq
    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found in secrets or environment")
    client = Groq(api_key=api_key)
    budget = _TokenBudgetTracker()
    user_prompt = _build_canonical_questions_prompt(qp_pages)
    log(f"Extracting canonical question list from {len(qp_pages)} question-paper page(s) in a single pass...")
    try:
        questions = _call_groq_with_retries(
            client, QUESTION_PAPER_ONLY_SYSTEM_PROMPT, user_prompt,
            _parse_canonical_questions_response, budget, log
        )
    except Exception as e:
        log(f"WARNING: canonical question extraction failed: {e}")
        return []
    log(f"Canonical question list: {len(questions)} question(s), single consistent pass")
    return questions
def identify_questions_with_llm(pages: list, status_callback=None) -> tuple:
    log = _make_safe_logger(status_callback)
    from groq import Groq
    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found in secrets or environment")
    client = Groq(api_key=api_key)
    budget = _TokenBudgetTracker()
    chunks = _chunk_pages_by_char_budget(pages)
    log(f"Split {len(pages)} page(s) into {len(chunks)} LLM chunk(s) to respect token limits")
    valid_page_numbers = {p["page_number"] for p in pages}
    max_page_number = max(valid_page_numbers) if valid_page_numbers else 0
    chunk_results = []
    chunk_failures = []
    for i, chunk in enumerate(chunks):
        page_nums_in_chunk = [p["page_number"] for p in chunk]
        log(f"Asking LLM to analyze chunk {i+1}/{len(chunks)} (pages {page_nums_in_chunk})...")
        try:
            qp_pages_1based, questions, admin_pages_1based = _call_groq_for_chunk(client, chunk, budget, log)
        except Exception as e:
            log(f"WARNING: chunk {i+1}/{len(chunks)} question-identification failed, skipping: {e}")
            chunk_failures.append(str(e))
            continue
        def _recover_pages(pages_1based, label):
            recovered = []
            truly_invalid = []
            for pn in pages_1based:
                if pn in valid_page_numbers:
                    recovered.append(pn)
                    continue
                split_result = _try_split_concatenated_page_number(
                    pn, valid_page_numbers, max_page_number
                )
                if split_result:
                    log(f"Recovered concatenated {label} page numbers: {pn} -> {split_result}")
                    recovered.extend(split_result)
                else:
                    truly_invalid.append(pn)
            if truly_invalid:
                log(f"WARNING: LLM returned out-of-range {label} page numbers, ignoring: {truly_invalid}")
            return sorted(set(recovered))
        qp_pages_1based = _recover_pages(qp_pages_1based, "question-paper")
        admin_pages_1based = _recover_pages(admin_pages_1based, "admin")
        admin_pages_1based = [p for p in admin_pages_1based if p not in qp_pages_1based]
        log(
            f"Chunk {i+1}/{len(chunks)}: identified {len(qp_pages_1based)} question paper "
            f"page(s), {len(admin_pages_1based)} admin/cover page(s) "
            f"(questions from this stage are discarded -- see stage 2 below)"
        )
        chunk_results.append((qp_pages_1based, [], admin_pages_1based))
    if chunk_failures and not chunk_results:
        raise Exception(
            f"All {len(chunks)} chunk(s) failed during question identification. "
            f"First failure: {chunk_failures[0]}"
        )
    elif chunk_failures:
        log(
            f"NOTE: {len(chunk_failures)} of {len(chunks)} chunk(s) failed and were "
            f"skipped -- question PAGE detection below is PARTIAL."
        )
    qp_pages_1based_merged, _, admin_pages_1based_merged = _merge_chunk_results(chunk_results)
    qp_page_indices_0based = sorted(pn - 1 for pn in qp_pages_1based_merged)
    admin_page_indices_0based = sorted(pn - 1 for pn in admin_pages_1based_merged)
    log(f"Question paper pages identified: {len(qp_page_indices_0based)} page(s)")
    log(f"Admin/cover pages identified: {len(admin_page_indices_0based)} page(s) "
        f"(these will be excluded from BOTH question and answer text)")
    if len(qp_page_indices_0based) >= 2:
        qp_page_lengths = [
            (i, len(pages[i]["raw_text"])) for i in qp_page_indices_0based
        ]
        lengths_only = [length for _, length in qp_page_lengths]
        median_length = sorted(lengths_only)[len(lengths_only) // 2]
        outliers = [
            page_idx for page_idx, length in qp_page_lengths
            if length > max(median_length * 3, 1500)
        ]
        if outliers and len(outliers) <= len(qp_page_indices_0based) // 2:
            for page_idx in outliers:
                length = dict(qp_page_lengths)[page_idx]
                log(
                    f"RECLASSIFYING page {page_idx + 1}: was detected as a question "
                    f"paper page but is {length} chars long -- much longer than the "
                    f"typical {median_length} chars for this document's other question "
                    f"paper pages. This is almost always the OPENING of a student's "
                    f"answer (restating the question before their real response). "
                    f"Moving it to the answer pages so its content is not lost."
                )
            qp_page_indices_0based = [
                i for i in qp_page_indices_0based if i not in outliers
            ]
        elif outliers:
            log(
                f"WARNING: {len(outliers)} of {len(qp_page_indices_0based)} detected "
                f"question-paper pages are unusually long (median {median_length} chars). "
                f"That's too large a fraction to auto-reclassify safely -- leaving them "
                f"as question-paper pages, but this may mean the question/answer page "
                f"split for this document is unreliable. Pages flagged: "
                f"{[p+1 for p in outliers]}"
            )
    qp_pages_full = [pages[i] for i in qp_page_indices_0based]
    page_clusters = []
    for p in qp_pages_full:
        if page_clusters and p["page_number"] == page_clusters[-1][-1]["page_number"] + 1:
            page_clusters[-1].append(p)
        else:
            page_clusters.append([p])
    reclassified_as_answer_pages = []
    if len(page_clusters) > 1:
        def _cluster_size_key(cluster):
            return (len(cluster), sum(len(p["raw_text"]) for p in cluster))
        chosen_cluster = max(page_clusters, key=_cluster_size_key)
        discarded_clusters = [c for c in page_clusters if c is not chosen_cluster]
        log(
            f"WARNING: question-paper pages form {len(page_clusters)} separate, "
            f"non-contiguous cluster(s): "
            f"{[[p['page_number'] for p in c] for c in page_clusters]}. This document is "
            f"treated as having a SINGLE question paper -- only the largest cluster "
            f"(pages {[p['page_number'] for p in chosen_cluster]}) is used as the real "
            f"question paper. The other {len(discarded_clusters)} cluster(s) are moved "
            f"to the answer-page pool without extraction, so they cannot contribute "
            f"duplicate/extra questions."
        )
        for cluster in discarded_clusters:
            reclassified_as_answer_pages.extend(p["page_number"] - 1 for p in cluster)
        page_clusters = [chosen_cluster]
    questions = []
    for cluster in page_clusters:
        cluster_page_nums = [p["page_number"] for p in cluster]
        cluster_questions = extract_canonical_questions(cluster, status_callback)
        if cluster_questions:
            questions.extend(cluster_questions)
            log(f"Cluster (pages {cluster_page_nums}): extracted {len(cluster_questions)} question(s)")
        else:
            log(
                f"WARNING: cluster (pages {cluster_page_nums}) produced ZERO questions -- "
                f"treating it as misclassified and moving these page(s) to the answer-page "
                f"pool instead of discarding them."
            )
            reclassified_as_answer_pages.extend(p["page_number"] - 1 for p in cluster)
    if reclassified_as_answer_pages:
        qp_page_indices_0based = [i for i in qp_page_indices_0based if i not in reclassified_as_answer_pages]
        log(
            f"Reclassified {len(reclassified_as_answer_pages)} page(s) as answer pages "
            f"(1-based): {[i + 1 for i in sorted(set(reclassified_as_answer_pages))]}"
        )
    before_dedup = len(questions)
    questions = _dedup_questions(questions)
    if len(questions) != before_dedup:
        log(
            f"Deduplicated final question list: {before_dedup} -> {len(questions)} "
            f"question(s) (removed {before_dedup - len(questions)} duplicate(s) that came "
            f"from overlapping/duplicate question-paper clusters)"
        )
    log(
        f"Final result: {len(qp_page_indices_0based)} question paper "
        f"page(s), {len(questions)} canonical question(s), "
        f"{len(admin_page_indices_0based)} admin/cover page(s)"
    )
    return qp_page_indices_0based, questions, admin_page_indices_0based
# =========================================================
# SEQUENTIAL SINGLE-TARGET ANSWER MAPPING (main, and only, mapping strategy)
# =========================================================
SEQUENTIAL_SEARCH_SYSTEM_PROMPT = """You are searching a line-numbered OCR transcript of a student's exam answer booklet for the line where the response to ONE SPECIFIC question begins.
You are given:
1. The exact text of the target question.
2. A window of the student's answer text, each line prefixed by [line number]. This may be a slice of a larger document -- the answer may simply not be in this window, which is a normal, expected outcome.

How to decide the start line:
- A response usually begins where the student restates/labels the question ("Ans 5-", "उत्तर 6-", "प्र. 8", a matching number) OR, with no label, where the content clearly shifts to this question's own topic.
- A bare label matching this question's own number/sub-part (e.g. "Q1", "Q.i", "1)") immediately followed by content is a strong, sufficient signal on its own -- no restated question text needed.
- Report the EARLIEST line of the answer, including any short intro/transition sentence before the topic is explicitly stated -- never skip to a later, more on-topic line.
- The same fact/definition CAN legitimately repeat in more than one answer. Similar wording appearing earlier in the document does NOT disqualify a later, genuine occurrence for this question.
- Ignore OCR artifact-description lines (e.g. "there is a logo", "signature", "watermark", "red pen scribble") -- they are never the start of a real answer.
- Do not confuse this with a DIFFERENT question's answer, even if it appears in the same window.
- If a plausible start point exists anywhere in the window for this question -- even a bare label, or a clear topical match -- report it. Only say found=false if there is genuinely no content in this window addressing this question's topic at all. Do not withhold a plausible match purely out of caution; a wrong guess can be corrected downstream, but a missed real answer cannot.

Return ONLY valid JSON (no markdown fences, no commentary) in exactly one of these two shapes:
{"found": true, "start_line": 42}
or
{"found": false}
start_line MUST be an exact line number shown in [brackets] in this window."""
def _build_sequential_search_prompt(window_lines: list, question_text: str, ref_label: str,
                                      extra_reminder: str = None, context_before: list = None) -> str:
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in window_lines)
    reminder_block = f"{extra_reminder}\n\n" if extra_reminder else ""
    context_block = ""
    if context_before:
        context_lines_block = "\n".join(f"[{idx}] {text}" for idx, text in context_before)
        context_block = (
            f"CONTEXT -- lines immediately BEFORE this window (reference only, tells you "
            f"whether the window's opening lines are a genuine NEW start or a continuation "
            f"of what came before; NEVER report one of these as start_line):\n"
            f"{context_lines_block}\n\n"
        )
    return (
        f"{reminder_block}"
        f"{context_block}"
        f"TARGET QUESTION ({ref_label}): {question_text}\n\n"
        f"TEXT WINDOW (line-numbered) -- ONLY lines in THIS section may be reported as start_line:\n{lines_block}"
    )
def _parse_sequential_search_response(content: str) -> tuple:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}\nRaw (first 300 chars): {content[:300]!r}")
    if not isinstance(data, dict) or "found" not in data:
        raise ValueError(f"Response missing 'found' key. Got: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}")
    found = bool(data["found"])
    if not found:
        return False, None
    if "start_line" not in data:
        raise ValueError("Response has found=true but is missing 'start_line'")
    try:
        start_line = int(data["start_line"])
    except (ValueError, TypeError):
        raise ValueError(f"'start_line' must be an integer, got {data['start_line']!r}")
    return True, start_line
SEQUENTIAL_SEARCH_WINDOW_CHARS = 16000
SEQUENTIAL_SEARCH_MAX_WINDOWS = 200
SEQUENTIAL_SEARCH_OVERLAP_LINES = 5
_BARE_LABEL_RE = re.compile(
    r'^\s*(?:Q\.?\s*|प्र\.?\s*|प्रश्न\.?\s*)?'
    r'\(?([ivxlcdm]+|\d+)\)?\s*[.:\-)]?\s*$',
    re.IGNORECASE
)
def _find_bare_label_candidates(window_lines: list, question_text: str) -> list:
    q_num = _extract_leading_number(question_text)
    q_label = _extract_sub_part_label(question_text)
    if not q_num and not q_label:
        return []
    candidates = []
    for idx, text in window_lines:
        stripped = text.strip()
        if len(stripped) > 15:
            continue
        m = _BARE_LABEL_RE.match(stripped)
        if not m:
            continue
        token = m.group(1).lower()
        if q_num and token == str(q_num):
            candidates.append(idx)
        elif q_label:
            inner = q_label.strip("()").lower()
            if token == inner:
                candidates.append(idx)
    return candidates
def _find_answer_start_sequential(client, numbered_lines: list, question_text: str, ref_label: str,
                                    search_from_idx: int, budget: "_TokenBudgetTracker", log,
                                    window_chars: int = SEQUENTIAL_SEARCH_WINDOW_CHARS,
                                    max_windows: int = SEQUENTIAL_SEARCH_MAX_WINDOWS,
                                    extra_reminder: str = None,
                                    end_idx: int = None,
                                    context_lookback: int = 6):
    total_lines = len(numbered_lines) if end_idx is None else min(len(numbered_lines), end_idx)
    pointer = search_from_idx
    windows_tried = 0
    while pointer < total_lines and windows_tried < max_windows:
        window = []
        chars = 0
        idx = pointer
        while idx < total_lines and (not window or chars + len(numbered_lines[idx][1]) <= window_chars):
            window.append(numbered_lines[idx])
            chars += len(numbered_lines[idx][1])
            idx += 1
        if not window:
            break
        context_before = numbered_lines[max(0, pointer - context_lookback):pointer] if pointer > 0 else None
        label_hint = None
        candidates = _find_bare_label_candidates(window, question_text)
        if candidates:
            label_hint = (
                f"HINT: a bare numeric/roman label matching this question's own number "
                f"was detected (by simple pattern-matching, not verified) at line(s) "
                f"{candidates} in this window. A bare label like this (e.g. 'Q1', 'Q.i') "
                f"is a strong, valid start signal even with NO restated question text -- "
                f"check these lines carefully and accept one if the content that follows "
                f"genuinely addresses this question."
            )
        combined_reminder = "\n\n".join(filter(None, [extra_reminder, label_hint])) or None
        user_prompt = _build_sequential_search_prompt(window, question_text, ref_label, combined_reminder, context_before)
        try:
            found, start_line = _call_groq_with_retries(
                client, SEQUENTIAL_SEARCH_SYSTEM_PROMPT, user_prompt,
                _parse_sequential_search_response, budget, log
            )
        except Exception as e:
            log(f"WARNING: search call failed for {ref_label} (lines {window[0][0]}-{window[-1][0]}): {e}")
            found, start_line = False, None
        if found and start_line is not None:
            valid_ids = {i for i, _ in window}
            if start_line in valid_ids:
                return start_line
            log(
                f"WARNING: {ref_label} reported start_line {start_line}, which is outside "
                f"this window's actual range {window[0][0]}-{window[-1][0]} -- ignoring and "
                f"treating this window as a non-match"
            )
        if idx >= total_lines:
            break
        pointer = max(pointer + 1, idx - SEQUENTIAL_SEARCH_OVERLAP_LINES)
        windows_tried += 1
    return None
_LEADING_NUMBER_RE = re.compile(r'^\s*(\d+)[\.\)]')
_SUB_PART_LABEL_RE = re.compile(r'\(([ivxlcdm]{1,5}|[a-zA-Z]|[\u0900-\u097F])\)', re.IGNORECASE)
def _extract_leading_number(text: str):
    m = _LEADING_NUMBER_RE.match(text)
    return m.group(1) if m else None
def _extract_sub_part_label(text: str):
    m = _SUB_PART_LABEL_RE.search(text)
    return m.group(0) if m else None
def _build_sub_part_hint(questions: list, i: int) -> str:
    current_label = _extract_sub_part_label(questions[i])
    if not current_label:
        return None
    current_num = _extract_leading_number(questions[i])
    siblings = []
    for j, q in enumerate(questions):
        if j == i:
            continue
        if current_num and _extract_leading_number(q) == current_num:
            lbl = _extract_sub_part_label(q)
            if lbl:
                siblings.append(lbl)
    if not siblings:
        return None
    return (
        f"NOTE: this target question is sub-part {current_label} of a larger multi-part "
        f"question (question {current_num}). The OTHER sub-parts of this SAME parent "
        f"question are: {', '.join(siblings)}. Find the response to sub-part {current_label} "
        f"SPECIFICALLY. Sibling sub-parts often share the same overall topic but are still "
        f"separate, distinct responses -- do not match content that actually belongs to a "
        f"DIFFERENT sibling sub-part just because it discusses a closely related aspect."
    )
WINDOWED_MULTI_TARGET_SYSTEM_PROMPT = """You are resolving internal boundaries WITHIN a small, already-confirmed block of text belonging to a multi-part question's sibling sub-parts.
You are given:
1. A short list of candidate sub-parts (2-5 typically), each tagged with a REF label. These are CONFIRMED sibling sub-parts of the SAME parent question -- the window shown contains some or all of their answers, back to back, in order.
2. The exact text window (line-numbered) this group's answers fall within.
For each sibling, find the line where its own portion begins (a new label like (ii)/(iii), or a clear shift to that specific aspect of the shared topic).
- The FIRST sibling normally starts at/near the window's beginning -- focus on finding where LATER siblings begin.
- Only report a start_line if you can identify a genuine transition point; omitting an uncertain one is safe (it just stays folded into the preceding sibling).
- Content can legitimately repeat across siblings -- don't reject a real transition just because of earlier similar wording.
Return ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:
{"starts": [{"ref": "REF-B", "start_line": 42}, {"ref": "REF-C", "start_line": 58}]}
Every start_line MUST be an exact line number shown in [brackets] in this window."""
def _build_windowed_multi_target_prompt(window_lines: list, open_questions: list,
                                          sibling_note: str = None) -> str:
    questions_block = "\n".join(f"[{ref}] {q}" for ref, q in open_questions)
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in window_lines)
    note_block = f"{sibling_note}\n\n" if sibling_note else ""
    return (
        f"{note_block}"
        f"CANDIDATE QUESTIONS:\n{questions_block}\n\n"
        f"TEXT WINDOW (line-numbered):\n{lines_block}"
    )
def _parse_windowed_multi_target_response(content: str) -> list:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}\nRaw (first 300 chars): {content[:300]!r}")
    if not isinstance(data, dict) or "starts" not in data:
        raise ValueError(f"Response missing 'starts' key. Got: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}")
    starts = data["starts"]
    if not isinstance(starts, list):
        raise ValueError(f"'starts' must be a list, got {type(starts).__name__}")
    result = []
    for item in starts:
        if not isinstance(item, dict) or "ref" not in item or "start_line" not in item:
            continue
        try:
            result.append({"ref": str(item["ref"]).strip().upper(), "start_line": int(item["start_line"])})
        except (ValueError, TypeError):
            continue
    return result
def _build_group_sibling_note(open_questions: list) -> str:
    groups = {}
    for ref, q in open_questions:
        num = _extract_leading_number(q)
        label = _extract_sub_part_label(q)
        if num and label:
            groups.setdefault(num, []).append((ref, label))
    notes = [
        f"- Question {num} has sibling sub-parts among the candidates: "
        f"{', '.join(f'{ref}={label}' for ref, label in members)}. Each needs its OWN "
        f"separate start_line if its content is present -- do not let one absorb the others."
        for num, members in groups.items() if len(members) >= 2
    ]
    if not notes:
        return None
    return "SIBLING SUB-PART GROUPS PRESENT IN THIS CANDIDATE LIST:\n" + "\n".join(notes)
def _detect_sibling_groups(questions: list) -> dict:
    groups = {}
    i = 0
    n = len(questions)
    while i < n:
        label_i = _extract_sub_part_label(questions[i])
        num_i = _extract_leading_number(questions[i])
        if label_i and num_i:
            j = i + 1
            while (j < n and _extract_leading_number(questions[j]) == num_i
                   and _extract_sub_part_label(questions[j])):
                j += 1
            if j - i >= 2:
                groups[i] = list(range(i, j))
            i = j
        else:
            i += 1
    return groups
def _resolve_sibling_group_batch(client, numbered_lines: list, group_questions: list,
                                   lower: int, upper: int, budget: "_TokenBudgetTracker",
                                   log) -> dict:
    window = [nl for nl in numbered_lines if lower <= nl[0] <= upper]
    if not window:
        return {}
    later_siblings = group_questions[1:]
    if not later_siblings:
        return {}
    user_prompt = _build_windowed_multi_target_prompt(window, later_siblings)
    try:
        starts = _call_groq_with_retries(
            client, WINDOWED_MULTI_TARGET_SYSTEM_PROMPT, user_prompt,
            _parse_windowed_multi_target_response, budget, log
        )
    except Exception as e:
        log(f"WARNING: sibling-group batch resolution failed for lines {lower}-{upper}: {e}")
        return {}
    valid_ids = {i for i, _ in window}
    valid_refs = {ref for ref, _ in later_siblings}
    found = {}
    for item in starts:
        ref, sl = item["ref"], item["start_line"]
        if ref not in valid_refs:
            continue
        if sl not in valid_ids:
            log(f"WARNING: discarding out-of-range sibling start {sl} for {ref}")
            continue
        if sl <= lower:
            continue
        found[ref] = min(found[ref], sl) if ref in found else sl
    return found
def _rescue_unmatched_questions(client, numbered_lines: list, questions: list, ranges: list,
                                  budget: "_TokenBudgetTracker", log) -> list:
    ref_to_idx = {f"REF-{chr(65+i)}": i for i in range(len(questions))}
    idx_to_ref = {i: f"REF-{chr(65+i)}" for i in range(len(questions))}
    changed = True
    passes = 0
    while changed and passes < 2:
        changed = False
        passes += 1
        matched_refs = {r["ref"] for r in ranges}
        for i, q in enumerate(questions):
            ref = idx_to_ref[i]
            if ref in matched_refs:
                continue
            candidate = None
            for r in ranges:
                r_idx = ref_to_idx.get(r["ref"])
                if r_idx is not None and r_idx < i:
                    if candidate is None or r_idx > ref_to_idx[candidate["ref"]]:
                        candidate = r
            if candidate is None:
                continue
            lower = candidate["start_line"] + 1
            upper = candidate["end_line"]
            if upper <= lower:
                continue
            log(f"  RESCUE: {ref} is unmatched -- searching for it inside {candidate['ref']}'s range (lines {lower}-{upper})...")
            split_line = _find_answer_start_sequential(
                client, numbered_lines, q, ref, lower, budget, log,
                end_idx=upper + 1,
                extra_reminder=(
                    "REMINDER: this question's answer was not found in earlier passes. "
                    "Search this window carefully -- a bare label or clear topical shift "
                    "is enough evidence; do not withhold a plausible match."
                )
            )
            if split_line is not None and lower < split_line <= upper:
                log(f"  RESCUE: recovered {ref} at line {split_line} (was absorbed into {candidate['ref']})")
                new_end = candidate["end_line"]
                candidate["end_line"] = split_line - 1
                ranges.append({"ref": ref, "start_line": split_line, "end_line": new_end})
                changed = True
    return ranges
def _ref_to_question_index(ref: str) -> int:
    return int(ord(ref.split("-")[-1]) - ord("A"))
def _force_fill_missing_refs(questions: list, ranges: list, total_lines: int, log=print) -> list:
    """
    GUARANTEE every question gets a range in the final output. Nothing should ever be
    silently dropped just because the LLM search passes couldn't confidently confirm it --
    that's what was causing e.g. 11/12 questions to show up instead of 12/12. If a
    question truly can't be found, it gets assigned the leftover gap between its
    neighbours (or half of the nearest matched neighbour's range) instead of vanishing.
    """
    have = {r["ref"] for r in ranges}
    missing = [i for i in range(len(questions)) if f"REF-{chr(65+i)}" not in have]
    if not missing:
        return ranges
    for i in missing:
        ref = f"REF-{chr(65+i)}"
        ordered = sorted(ranges, key=lambda r: r["start_line"])
        prev_r = max((r for r in ordered if _ref_to_question_index(r["ref"]) < i),
                     key=lambda r: _ref_to_question_index(r["ref"]), default=None)
        next_r = min((r for r in ordered if _ref_to_question_index(r["ref"]) > i),
                     key=lambda r: _ref_to_question_index(r["ref"]), default=None)
        if prev_r and next_r and next_r["start_line"] - prev_r["end_line"] > 1:
            start, end = prev_r["end_line"] + 1, next_r["start_line"] - 1
        elif prev_r:
            mid = (prev_r["start_line"] + prev_r["end_line"]) // 2
            start, end = mid + 1, prev_r["end_line"]
            prev_r["end_line"] = mid
        elif next_r:
            mid = (next_r["start_line"] + next_r["end_line"]) // 2
            start, end = next_r["start_line"], mid
            next_r["start_line"] = mid + 1
        else:
            start, end = 0, total_lines - 1
        if start > end:
            end = start
        log(
            f"FORCE-FILL: {ref} was never confidently matched by any search pass -- "
            f"assigning fallback range {start}-{end} so it is never silently dropped "
            f"from the output. Worth a manual spot-check."
        )
        ranges.append({"ref": ref, "start_line": start, "end_line": end})
    return ranges
_ANSWER_START_RE = re.compile(
    r'^\s*(?:Ans(?:wer)?[.\s]+\d*|उत्तर\s*\d*\s*[\-\:]?|प्र[०.\s]*\d*|Q\.?\s*\d+[.\s:-])',
    re.IGNORECASE
)
def _build_label_anchor_index(answer_lines: list, questions: list, exclude_indices: set, log=print) -> dict:
    """Free, zero-LLM-cost pre-pass: anchor questions via explicit 'Ans/उत्तर/Q.' style labels."""
    q_number_to_indices = {}
    for qi, q in enumerate(questions):
        if qi in exclude_indices:
            continue
        qn = _extract_leading_number(q)
        if qn is not None:
            q_number_to_indices.setdefault(qn, []).append(qi)
    if not q_number_to_indices:
        return {}
    raw_hits = {}
    for i, line in enumerate(answer_lines):
        m = _ANSWER_START_RE.match(line)
        if not m:
            continue
        num_match = re.search(r'\d+', m.group(0))
        if not num_match:
            continue
        qis = q_number_to_indices.get(num_match.group(0))
        if not qis:
            continue
        for qi in qis:
            raw_hits.setdefault(qi, []).append(i)
    anchors = {}
    last_line = -1
    for qi in sorted(raw_hits.keys()):
        candidates = sorted(l for l in raw_hits[qi] if l > last_line)
        if candidates:
            anchors[qi] = candidates[0]
            last_line = candidates[0]
    if anchors:
        log(
            f"Label pre-pass (zero-LLM-cost): anchored {len(anchors)}/{len(questions)} "
            f"question(s) via explicit labels (Ans/उत्तर/Q. etc.) -- skipping LLM search "
            f"calls entirely for these."
        )
    return anchors
# =========================================================
# BATCHED MULTI-QUESTION START FINDER (primary strategy -- replaces one-LLM-call-per-
# question-per-window. Instead of scanning the document up to N_questions x N_windows
# times, we scan it ONCE, chunk by chunk, asking for ALL still-unfound questions at
# once per chunk. Chunks are independent so they run concurrently too. This is both
# the main token-cost cut and the main time-complexity cut in this pipeline.)
# =========================================================
BATCH_START_FINDER_SYSTEM_PROMPT = """Scan this chunk of a line-numbered OCR transcript (a student exam answer booklet) for where EACH listed target question's answer begins. Most questions will NOT start in this chunk -- that's expected, only report the ones that genuinely do.
Rules:
- A bare label matching a question's own number (e.g. "Q3", "3)", "(ii)") followed by relevant content is enough on its own -- no restated question text needed.
- Report the EARLIEST line of each match, including a short intro sentence before the topic becomes explicit.
- Similar content can legitimately repeat across different questions on the same broad topic -- don't reject a real match just because similar wording appeared elsewhere.
- Ignore OCR artifact-description lines ("there is a logo", "signature", "watermark", "red pen scribble") -- never report one as a start.
- If a plausible start exists for a question in this chunk, report it rather than omitting it out of caution.
Return ONLY valid JSON (no markdown fences, no commentary):
{"starts": [{"ref": "REF-C", "start_line": 88}, {"ref": "REF-D", "start_line": 140}]}
Omit any question whose answer isn't in this chunk. If none match, return {"starts": []}."""
BATCH_CHUNK_CHARS = int(os.getenv("GROQ_BATCH_CHUNK_CHARS", "7000"))  # ceiling; actual size is adaptive, see below
BATCH_CHUNK_OVERLAP_LINES = 3
BATCH_MAX_CONCURRENCY = GROQ_MAX_CONCURRENT_CALLS
BATCH_CHUNK_TARGET_FRACTION = 0.5  # aim to use at most half the safe budget per single call
def _compute_safe_chunk_chars(budget: "_TokenBudgetTracker", overhead_chars: int,
                                ceiling_chars: int = BATCH_CHUNK_CHARS,
                                target_fraction: float = BATCH_CHUNK_TARGET_FRACTION,
                                min_chunk_chars: int = 1200) -> int:
    """
    The question list (+ sibling note) gets re-sent as overhead on EVERY chunk call,
    so with many questions that overhead alone can already eat most of a tight TPM
    budget -- a fixed chunk-char size doesn't account for that and can produce a
    single request bigger than the whole safe budget (the "8289 > 6800, forever"
    situation). Size the chunk dynamically instead: budget the tokens, subtract
    the overhead, spend the rest on transcript text.
    """
    target_tokens = budget.safe_limit * target_fraction
    overhead_tokens = _estimate_tokens(" " * overhead_chars) + 800
    remaining_tokens = max(300, target_tokens - overhead_tokens)
    chunk_chars = int(remaining_tokens * CHARS_PER_TOKEN_ESTIMATE)
    return max(min_chunk_chars, min(ceiling_chars, chunk_chars))
def _chunk_numbered_lines_by_chars(numbered_lines: list, chunk_chars: int = BATCH_CHUNK_CHARS,
                                     overlap_lines: int = BATCH_CHUNK_OVERLAP_LINES) -> list:
    chunks = []
    i = 0
    n = len(numbered_lines)
    while i < n:
        chars = 0
        j = i
        while j < n and (j == i or chars + len(numbered_lines[j][1]) <= chunk_chars):
            chars += len(numbered_lines[j][1])
            j += 1
        chunks.append(numbered_lines[i:j])
        if j >= n:
            break
        i = max(i + 1, j - overlap_lines)
    return chunks
def _batch_find_all_starts(client, numbered_lines: list, questions: list, already_found: set,
                             budget: "_TokenBudgetTracker", log,
                             max_workers: int = BATCH_MAX_CONCURRENCY) -> dict:
    open_questions = [
        (f"REF-{chr(65+i)}", q) for i, q in enumerate(questions)
        if f"REF-{chr(65+i)}" not in already_found
    ]
    if not open_questions:
        return {}
    sibling_note = _build_group_sibling_note(open_questions)
    questions_block_chars = sum(len(ref) + len(q) + 4 for ref, q in open_questions)
    overhead_chars = questions_block_chars + len(sibling_note or "") + len(BATCH_START_FINDER_SYSTEM_PROMPT)
    chunk_chars = _compute_safe_chunk_chars(budget, overhead_chars)
    chunks = _chunk_numbered_lines_by_chars(numbered_lines, chunk_chars=chunk_chars)
    log(
        f"Batch pass: scanning {len(chunks)} chunk(s) (~{chunk_chars} chars each, sized to fit "
        f"this account's token budget) for all {len(open_questions)} remaining question(s) "
        f"(up to {max_workers} chunk(s) in flight at once) -- replaces what used to be up to "
        f"{len(open_questions) * 20}+ separate window calls."
    )
    def _run_chunk(chunk):
        user_prompt = _build_windowed_multi_target_prompt(chunk, open_questions, sibling_note)
        try:
            return _call_groq_with_retries(
                client, BATCH_START_FINDER_SYSTEM_PROMPT, user_prompt,
                _parse_windowed_multi_target_response, budget, log
            )
        except Exception as e:
            log(f"WARNING: batch chunk (lines {chunk[0][0]}-{chunk[-1][0]}) failed, skipping: {e}")
            return []
    results_per_chunk = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_run_chunk, c) for c in chunks]
        for c, fut in zip(chunks, futures):
            results_per_chunk.append((c, fut.result()))
    valid_refs = {ref for ref, _ in open_questions}
    hits = {}
    for chunk, starts in results_per_chunk:
        valid_ids = {idx for idx, _ in chunk}
        for item in starts:
            ref, sl = item["ref"], item["start_line"]
            if ref not in valid_refs or sl not in valid_ids:
                continue
            hits.setdefault(ref, []).append(sl)
    found = {ref: min(lines) for ref, lines in hits.items()}
    log(f"Batch pass matched {len(found)}/{len(open_questions)} remaining question(s) in {len(chunks)} chunk call(s).")
    return found
# =========================================================
# MAIN ANSWER MAPPING FUNCTION
# (Gemini verification and the extra Groq re-analysis/remap passes have been removed --
#  they added latency/cost without reliably improving accuracy, and in practice the strict
#  "only report found=true if certain" wording in the old boundary-check prompts was itself
#  a source of questions getting silently dropped. The guarantee now comes from
#  _force_fill_missing_refs below: every question ends up with SOME range in the output,
#  never just vanishes.)
# =========================================================
def map_answers_sequential(answer_lines: list, questions: list, status_callback=None,
                             answer_line_pages: list = None) -> list:
    log = _make_safe_logger(status_callback)
    from groq import Groq
    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found in secrets or environment")
    client = Groq(api_key=api_key)
    budget = _TokenBudgetTracker()
    numbered_lines = list(enumerate(answer_lines))
    total_lines = len(numbered_lines)
    n = len(questions)
    sibling_groups = _detect_sibling_groups(questions)
    group_member_of = {idx: first_idx for first_idx, members in sibling_groups.items() for idx in members}

    # STAGE 1 -- free, zero-LLM-cost label pre-pass (regex based).
    label_anchors = _build_label_anchor_index(answer_lines, questions, set(group_member_of.keys()), log)
    found_starts = {f"REF-{chr(65 + qi)}": line_idx for qi, line_idx in label_anchors.items()}

    # STAGE 2 -- ONE batched pass over the whole document for every question the
    # label pre-pass missed, instead of a separate windowed search per question.
    # This is the main token/time saver: O(chunks) calls instead of O(questions x windows).
    batch_starts = _batch_find_all_starts(client, numbered_lines, questions, set(found_starts.keys()), budget, log)
    found_starts.update(batch_starts)

    # STAGE 3 -- sibling sub-part groups: if a group's members collapsed onto the same
    # (or missing) start from the batch pass, resolve their internal split with one
    # bounded call per group (cheap -- only runs for actual multi-part questions).
    for first_idx, group_indices in sibling_groups.items():
        group_refs = [f"REF-{chr(65 + j)}" for j in group_indices]
        if all(r in found_starts for r in group_refs):
            continue
        group_questions = [(f"REF-{chr(65 + j)}", questions[j]) for j in group_indices]
        first_ref, first_q = group_questions[0]
        anchor_before = max(
            (v for k, v in found_starts.items() if _ref_to_question_index(k) < first_idx),
            default=-1,
        )
        anchor_after = min(
            (v for k, v in found_starts.items() if _ref_to_question_index(k) > group_indices[-1]),
            default=total_lines,
        )
        group_start = found_starts.get(first_ref)
        if group_start is None:
            group_start = _find_answer_start_sequential(
                client, numbered_lines, first_q, first_ref, anchor_before + 1, budget, log,
                end_idx=anchor_after
            )
        if group_start is None:
            log(f"NOTE: sibling group {group_refs} not found in this pass -- will be recovered by rescue/force-fill.")
            continue
        found_starts[first_ref] = group_start
        upper = anchor_after - 1 if anchor_after < total_lines else total_lines - 1
        if len(group_questions) > 1:
            sibling_starts = _resolve_sibling_group_batch(
                client, numbered_lines, group_questions, group_start, upper, budget, log
            )
            for ref, sl in sibling_starts.items():
                found_starts[ref] = sl
                log(f"  found {ref} (sibling) at line {sl}")

    ordered = sorted(found_starts.items(), key=lambda kv: kv[1])
    ranges = []
    for idx, (ref, start) in enumerate(ordered):
        end = ordered[idx + 1][1] - 1 if idx + 1 < len(ordered) else total_lines - 1
        ranges.append({"ref": ref, "start_line": start, "end_line": end})
    log(f"Label + batch pass found {len(ranges)} of {n} question(s) before rescue/force-fill")

    # STAGE 4 -- targeted rescue only for the (usually very few) stragglers.
    ranges = _rescue_unmatched_questions(client, numbered_lines, questions, ranges, budget, log)
    # STAGE 5 -- guarantee: whatever's still missing gets a fallback range, never dropped.
    ranges = _force_fill_missing_refs(questions, ranges, total_lines, log)
    log(f"Final: {len(ranges)} of {n} question(s) present in output (force-fill guarantees all {n} appear)")
    ranges_by_ref = {r["ref"]: r for r in ranges}
    results = []
    for i, q in enumerate(questions):
        ref = f"REF-{chr(65 + i)}"
        r = ranges_by_ref.get(ref)
        if r is None:
            results.append({
                "ref": ref, "question": q, "matched": False,
                "start_line": None, "end_line": None,
                "start_page": None, "end_page": None,
                "answer": "", "answer_raw": "",
            })
            continue
        s, e = r["start_line"], r["end_line"]
        if s >= len(answer_lines):
            s = len(answer_lines) - 1
        if e >= len(answer_lines):
            e = len(answer_lines) - 1
        if s > e:
            s, e = e, s
        verbatim_lines = [
            answer_lines[j] for j in range(s, e + 1)
            if 0 <= j < len(answer_lines) and answer_lines[j].strip() and not is_noise(answer_lines[j])
        ]
        answer_raw = " ".join(verbatim_lines).strip()
        answer_clean = strip_question_restatement(answer_raw)
        answer_clean = strip_full_question_echo(answer_clean, q)
        next_q_text = questions[i + 1] if i + 1 < len(questions) else None
        answer_clean = strip_trailing_leaked_next_question(answer_clean, next_q_text)
        answer_clean = strip_trailing_next_question_leadin(answer_clean)
        start_page = answer_line_pages[s] if answer_line_pages and 0 <= s < len(answer_line_pages) else None
        end_page = answer_line_pages[e] if answer_line_pages and 0 <= e < len(answer_line_pages) else None
        results.append({
            "ref": ref, "question": q, "matched": True,
            "start_line": s, "end_line": e,
            "start_page": start_page, "end_page": end_page,
            "answer": answer_clean, "answer_raw": answer_raw,
        })
    return results
_QUESTION_STOPWORDS = {
    'how', 'are', 'the', 'views', 'state', 'with', 'theme', 'examine',
    'write', 'detailed', 'note', 'their', 'corresponding', 'why', 'does',
    'plot', 'plan', 'comment', 'discuss', 'explain', 'describe', 'and',
    'what', 'when', 'where', 'which', 'who', 'integrated', 'analyse',
    'analyze', 'critically', 'briefly', 'elaborate', 'illustrate', 'for',
    'from', 'this', 'that', 'these', 'those', 'into', 'about', 'role',
    'significance', 'importance', 'short', 'long', 'play', 'text',
}
def _normalize_for_overlap_match(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    return text
def _distinctive_words(text: str, max_words: int = 20) -> list:
    words = re.findall(r'[a-z]{3,}', _normalize_for_overlap_match(text))[:max_words]
    return sorted(set(w for w in words if w not in _QUESTION_STOPWORDS))
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
def strip_question_restatement(answer_text: str) -> str:
    text = answer_text
    for _ in range(2):
        new_text = QUESTION_PREFIX_RE.sub('', text, count=1).strip()
        if new_text == text:
            break
        text = new_text
    return text
_TRAILING_LEADIN_RE = re.compile(
    r'(?:'
    r'\s*#{1,3}\s*\d*\s*$'
    r'|\s*(?:Section|Sec\.?)\s*[-:]?\s*[A-Za-z0-9]+\s*$'
    r'|\s*भाग\s*[-:]?\s*[०-९0-9]*\s*$'
    r'|\s*(?:Q\.?|Ans(?:wer)?\.?|प्र\.?|प्रश्न\.?|उत्तर)\s*[-:.]?\s*\d+\s*[-:.)]?\s*$'
    r'|\s*\(?[ivxlcdm]{1,5}\)?\s*[-:.)]?\s*$'
    r')',
    re.IGNORECASE
)
def strip_trailing_leaked_next_question(answer_text: str, next_question_text: str) -> str:
    if not next_question_text:
        return answer_text
    next_core = re.sub(r'^\s*(?:Q\.?\s*\d+[.)]\s*|\d+[.)]\s*)', '', next_question_text).strip()
    if len(next_core) < 8:
        return answer_text
    text = answer_text.rstrip()
    anchor = next_core[:40].lower()
    idx = text.lower().rfind(anchor[:20])
    if idx != -1 and idx > len(text) * 0.5:
        candidate = text[:idx]
        candidate = re.sub(
            r'(?:\s*\d{1,3}(?:[\+\/]\d{1,4}){0,2}\s*)?'
            r'(?:Q\.?\s*\d+\s*[.)]|Q\.?\s*[ivxlcdm]+\s*[.)])\s*[>\.\-:]?\s*$',
            '', candidate, flags=re.IGNORECASE
        ).rstrip()
        return candidate
    return text
def strip_trailing_next_question_leadin(answer_text: str, max_passes: int = 3) -> str:
    text = answer_text.rstrip()
    for _ in range(max_passes):
        new_text = _TRAILING_LEADIN_RE.sub('', text).rstrip()
        if new_text == text:
            break
        text = new_text
    return text.strip()
def _normalize_for_echo_compare(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    return text
_PARENT_INSTRUCTION_PREFIX_RE = re.compile(
    r'^\s*\d+[\.\)]?\s*(?:\([ivx]+\)|\([a-z]\)|\([क-घ]\))?\s*'
    r'(?:identify and explain the following|write (?:short )?notes? on|'
    r'comment on|explain the following|discuss the following)\s*:?\s*',
    re.IGNORECASE
)
def strip_full_question_echo(answer_text: str, question_text: str) -> str:
    question_core = _PARENT_INSTRUCTION_PREFIX_RE.sub('', question_text).strip()
    if not question_core:
        question_core = question_text
    q_norm = _normalize_for_echo_compare(question_core)
    q_word_count = len(q_norm.split())
    if q_word_count == 0:
        return answer_text
    answer_words = answer_text.split()
    if not answer_words:
        return answer_text
    min_n = max(3, int(q_word_count * 0.7))
    max_n = min(len(answer_words), int(q_word_count * 1.3) + 2)
    best_strip_count = 0
    best_ratio = 0.0
    for n in range(min_n, max_n + 1):
        prefix = " ".join(answer_words[:n])
        prefix_norm = _normalize_for_echo_compare(prefix)
        ratio = difflib.SequenceMatcher(None, prefix_norm, q_norm).ratio()
        if ratio >= 0.75 and ratio > best_ratio:
            best_ratio = ratio
            best_strip_count = n
    if best_strip_count > 0:
        remaining = " ".join(answer_words[best_strip_count:]).strip()
        remaining = re.sub(r'^(?:Answer\s*[-:]\s*)', '', remaining, flags=re.IGNORECASE)
        return remaining.strip()
    return answer_text
NOISE_RE = re.compile(
    r'(?:signature'
    r'|PAGE\s*NO'
    r'|^\s*DATE\b'
    r'|^\s*\d{1,3}\s*$)',
    re.IGNORECASE
)
NOISE_LINE_MAX_CHARS = 40
_IMAGE_META_RE = re.compile(
    r'^\s*[\[\(]?\s*(?:'
    r'there (?:is|are) (?:a |an |some )?(?:logo|stamp|seal|scribble|line|mark|drawing|doodle|sketch|figure|image|photo|watermark)'
    r'|(?:a |an )?(?:logo|stamp|seal|watermark)\s*(?:here|present|visible|seen)?'
    r'|scribbl(?:e|ed|ing)\s*(?:with|in)?\s*(?:a\s+)?(?:red|blue|black|green)\s*(?:pen|ink|marker)'
    r'|(?:red|blue|black|green)\s*(?:pen|ink)\s*(?:mark|line|scribble|underline)s?'
    r'|handwritten\s+(?:note|scribble|mark)s?\s*(?:in|on)?\s*(?:the\s+)?margin'
    r'|image\s*[:\-]\s*.{0,50}'
    r'|(?:logo|stamp|seal|signature|watermark|figure|diagram)\s*(?:image|icon)?'
    r')\s*[\]\)]?\s*$',
    re.IGNORECASE
)
def _is_image_description_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    return bool(_IMAGE_META_RE.match(stripped))
def is_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r'^\s*\d{1,3}\s*$', stripped):
        return True
    if _is_image_description_line(stripped):
        return True
    if len(stripped) > NOISE_LINE_MAX_CHARS:
        return False
    return bool(NOISE_RE.search(stripped))
def _sanity_check_answer_pages(answer_lines: list, num_questions: int, log=print) -> bool:
    total_chars = sum(len(l) for l in answer_lines)
    avg_chars_per_question = total_chars / max(num_questions, 1)
    MIN_PLAUSIBLE_CHARS_PER_QUESTION = 200
    if avg_chars_per_question < MIN_PLAUSIBLE_CHARS_PER_QUESTION:
        log(
            f"WARNING: 'answer pages' contain only {total_chars} total characters "
            f"for {num_questions} question(s) (~{avg_chars_per_question:.0f} chars/question). "
            f"This is far too little for real essay-style answers and strongly "
            f"suggests the question-paper/answer-page split misclassified pages -- "
            f"e.g. real answer pages may have been wrongly identified as question "
            f"paper pages. Check the 'Question paper pages detected' log line above "
            f"against the actual document structure."
        )
        return False
    return True
def _flag_suspiciously_short_answers(qa_pairs: list, log=print) -> None:
    matched_lengths = [len(p["answer"]) for p in qa_pairs if p.get("matched") and p["answer"].strip()]
    if len(matched_lengths) < 2:
        return
    matched_lengths.sort()
    median_len = matched_lengths[len(matched_lengths) // 2]
    if median_len < 50:
        return
    for p in qa_pairs:
        if not p.get("matched"):
            continue
        length = len(p["answer"])
        if length < median_len * 0.25 and length < 300:
            log(
                f"WARNING: possible truncated answer for '{p['question'][:60]}...' "
                f"-- only {length} chars vs this document's median matched answer "
                f"length of {median_len} chars. Worth spot-checking against the OCR."
            )
def verify_no_llm_text_rewriting(qa_pairs: list, answer_lines: list, log=print) -> bool:
    all_ok = True
    for p in qa_pairs:
        if not p.get("matched"):
            continue
        s, e = p["start_line"], p["end_line"]
        if s < 0:
            s = 0
        if e >= len(answer_lines):
            e = len(answer_lines) - 1
        if s > e:
            s, e = e, s
        expected_raw = " ".join(
            answer_lines[j] for j in range(s, e + 1)
            if 0 <= j < len(answer_lines) and answer_lines[j].strip() and not is_noise(answer_lines[j])
        ).strip()
        if expected_raw != p["answer_raw"]:
            all_ok = False
            log(
                f"CRITICAL: verbatim-integrity check FAILED for '{p['question'][:60]}...' -- "
                f"the reported answer_raw does not match a fresh re-slice of answer_lines "
                f"[{s}:{e}]. This indicates a real bug in the extraction code, not an LLM "
                f"hallucination -- please report this."
            )
    if all_ok:
        log(
            "Verbatim-safety check passed: every matched answer's raw text is a byte-for-byte "
            "slice of the original OCR lines -- the LLM was only ever asked for line numbers, "
            "never for text content, so it could not have silently corrected grammar or spelling."
        )
    return all_ok
# =========================================================
# COMPLETE PIPELINE WITH SEMANTIC CLEANING
# =========================================================
@_diagnose_tuple_errors
def process_pdf(file_input, status_callback=None):
    log = _make_safe_logger(status_callback)
    file_bytes, file_name = _normalize_file_input(file_input, default_name="document.pdf")
    pages = run_ocr(file_bytes, file_name, status_callback)
    log("Building OCR JSON...")
    ocr_json = build_ocr_json(pages)
    log(f"Total pages: {ocr_json['total_pages']}")
    qp_page_indices, official_questions, admin_page_indices = identify_questions_with_llm(pages, status_callback)
    log(f"Question paper pages detected: {[p+1 for p in qp_page_indices] if qp_page_indices else 'none'}")
    log(f"Admin/cover pages detected: {[p+1 for p in admin_page_indices] if admin_page_indices else 'none'}")
    log(f"Official questions extracted: {len(official_questions)}")
    if not qp_page_indices:
        raise Exception(
            "The LLM could not identify any question paper pages in this document.\n"
            f"Page 1 preview:\n{pages[0]['raw_text'][:500]}"
        )
    if not official_questions:
        raise Exception(
            "Question paper pages were identified, but no questions were extracted from any "
            "of them.\n"
            f"Detected pages: {[p+1 for p in qp_page_indices]}\n\n"
            "This means EVERY question-paper cluster individually failed extraction (see the "
            "'WARNING: cluster (pages ...) produced ZERO questions' log lines above for which "
            "specific pages and why). If the detected pages look scattered/non-contiguous, the "
            "page classification itself is likely unreliable for this document -- check the "
            "OCR text of those specific pages directly."
        )
    excluded_indices = set(qp_page_indices) | set(admin_page_indices)
    answer_page_indices = [i for i in range(len(pages)) if i not in excluded_indices]
    answer_pages = [pages[i] for i in answer_page_indices]
    log(f"Answer pages: {[i+1 for i in answer_page_indices]}")
    answer_lines = []
    answer_line_pages = []
    for page in answer_pages:
        for line in page["raw_text"].split("\n"):
            if not is_noise(line):
                answer_lines.append(line)
                answer_line_pages.append(page["page_number"])
    all_page_numbers = {p["page_number"] for p in pages}
    answer_page_numbers = {pages[i]["page_number"] for i in answer_page_indices}
    qp_page_numbers = {pages[i]["page_number"] for i in qp_page_indices}
    admin_page_numbers = {pages[i]["page_number"] for i in admin_page_indices}
    last_page_num = max(all_page_numbers)
    if last_page_num not in answer_page_numbers:
        log(
            f"WARNING: the LAST page of the document (page {last_page_num}) was NOT "
            f"classified as an answer page -- it went to "
            f"{'question-paper' if last_page_num in qp_page_numbers else 'admin'} pages "
            f"instead. If this page actually contains the tail of the last answer, "
            f"that content has been silently dropped. Please spot-check page {last_page_num}."
       )
    log(f"Flattened {len(answer_lines)} answer lines")
    pages_look_plausible = _sanity_check_answer_pages(answer_lines, len(official_questions), log)
    if not pages_look_plausible:
        raise Exception(
            "The 'answer pages' identified in this document do not contain enough "
            "text to plausibly hold real essay-style answers for the "
            f"{len(official_questions)} question(s) found. This usually means the "
            "question-paper/answer-page page split misclassified pages -- check the "
            "'Question paper pages detected' log line above against the actual "
            "document structure. No answer-mapping LLM calls were made, since they "
            "would be guaranteed to fail."
        )
    log("Mapping each question to its answer (sequential single-target search)...")
    qa_pairs = map_answers_sequential(
        answer_lines, official_questions, status_callback,
        answer_line_pages=answer_line_pages
    )
    matched_count = sum(1 for p in qa_pairs if p["matched"])
    log(f"Matched {matched_count} of {len(official_questions)} questions")
    for p in qa_pairs:
        if not p["matched"]:
            log(f"WARNING: No match found for: {p['question'][:60]}")
    if matched_count == 0:
        raise Exception(
            "Could not match any questions to answers.\n"
            f"Official questions: {official_questions}\n"
            f"First 10 answer lines: {answer_lines[:10]}"
        )
    verify_no_llm_text_rewriting(qa_pairs, answer_lines, log)
    _flag_suspiciously_short_answers(qa_pairs, log)
    log("Applying semantic cleaning to remove question restatements from answers...")
    matcher = get_semantic_matcher()
    qa_pairs = matcher.clean_qa_pairs(qa_pairs, status_callback)
    log(f"Done -- {len(qa_pairs)} Q-A pairs ({matched_count} matched)")
    return ocr_json, qa_pairs
def save_outputs(ocr_json: dict, qa_pairs: list, output_dir: str = ".",
                  base_name: str = "document") -> tuple:
    ocr_path = os.path.join(output_dir, f"{base_name}_ocr.json")
    qa_path = os.path.join(output_dir, f"{base_name}_qa_pairs.json")
    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(ocr_json, f, ensure_ascii=False, indent=2)
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    return ocr_path, qa_path
