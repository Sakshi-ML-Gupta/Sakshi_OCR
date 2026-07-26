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
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
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
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    file_bytes, file_name = _normalize_file_input(file_input, default_name="reference.pdf")
    pages = run_ocr_cached(file_bytes, file_name, status_callback)
    log(f"Reference OCR complete -- {len(pages)} page(s)")
    return build_ocr_json(pages)
# =========================================================
# LLM-BASED QUESTION PAPER / QUESTION DETECTION (Groq)
# =========================================================
GROQ_MODEL = "openai/gpt-oss-120b"
TPM_LIMIT = 30000
TPM_SAFETY_FRACTION = 0.85
CHARS_PER_TOKEN_ESTIMATE = 2.0
MAX_CHARS_PER_CHUNK = 9000  # increased from 6000 -- fewer chunks means fewer calls, each paying the ~500-token system-prompt overhead only once per (larger) chunk instead of once per (smaller) chunk
CHUNK_OVERLAP_PAGES = 1
QP_SYSTEM_PROMPT = """You are analyzing OCR text from a scanned student exam booklet -- ANY institution, subject, or language. Pages come in no guaranteed order:
1. ADMIN/COVER pages: roll number, course code, student name, letterhead, blank sheets -- no question or answer content.
2. QUESTION PAPER pages: the official numbered question list -- prompts DIRECTED AT the student ("Discuss X", "Explain Y", etc., in whatever language). Marks like "10"/"20" may appear.
3. ANSWER pages: the student's own OCR'd answers -- long, may restate a question then respond, and may contain their OWN numbered sub-points (these are NOT separate questions).
You see only a chunk of pages at a time; carried-over context from a prior chunk should still be classified on its own content.
Return ONLY valid JSON (no markdown fences, no commentary) in EXACTLY this shape:
{
  "question_paper_pages": [14, 16, 18],
  "admin_pages": [1, 2],
  "questions": ["1. Example question text. (10)", "2. Another example question. (10)"]
}
Page-number arrays must list EACH page as its own element (e.g. [14, 16, 18], never merged like [141618]). A page never appears in both lists.
Distinguishing rules:
- A genuine question is a PROMPT ("explain", "discuss", "write notes on", a question mark) asking the student to DO something. A numbered point inside a long answer is a STATEMENT/FACT, not an instruction.
- Numbered items following an "answer" label (in any language/script), or after a long explanatory paragraph, mean an ANSWER page -- exclude from question_paper_pages even with multiple numbered lines.
- A real question paper is concise per question. A long flowing essay with numbered sub-points is not.
- TRAP: students often RESTATE the question as their answer's first sentence before their own explanation. This can look like a question page (prompt verbs like "Examine") but is really the FIRST page of an ANSWER. Signals it's a restatement, not the real question paper: noticeably more text than a concise instruction needs; developing-argument prose rather than terse instruction; the same/similar text already appears on a more concise page you're confident is the real question paper. When unsure, brevity is the deciding signal -- genuine question papers are short per question.
- When uncertain, do NOT classify a page as a question page.
- Cover/admin pages go in "admin_pages" so they're excluded from both question and answer text.
- Empty question_paper_pages/admin_pages lists are valid and expected for answer-only chunks.
- Preserve exact original text/numbering -- no paraphrasing, renumbering, or translation.
- Output ONLY the JSON object. No prose, no code fences."""
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
        log(
            f"Proactively pacing requests: {used:.0f} tokens used in the last 60s, "
            f"+{upcoming_tokens} upcoming would exceed safe budget "
            f"({self.safe_limit:.0f}). Waiting {wait_s:.1f}s before sending next chunk..."
        )
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
# =========================================================
# GroqQuotaExhaustedError -- raised when EVERY configured Groq API
# key's daily quota (TPD) is exhausted. Distinct from a generic
# Exception on purpose: callers throughout this module re-raise it
# specifically instead of swallowing it as a normal "window didn't
# match" or "chunk failed" result. Swallowing it silently was a
# confirmed bug -- once quota ran out mid-document, every remaining
# search failed the same way but was treated like a genuine "not
# found", so later questions' content silently piled up into an
# earlier answer's range (two answers mixing together). Propagating
# this instead stops the WHOLE pipeline with one clear message the
# moment quota is truly gone, instead of quietly returning corrupted
# output.
# =========================================================
class GroqQuotaExhaustedError(Exception):
    pass
# =========================================================
# MULTI-KEY GROQ ROTATION (fallback across several Groq accounts)
#
# Add as many keys as you have to secrets.toml / the environment:
#   GROQ_API_KEY   = "key-from-account-1"
#   GROQ_API_KEY_2 = "key-from-account-2"
#   GROQ_API_KEY_3 = "key-from-account-3"
#   GROQ_API_KEY_4 = "key-from-account-4"
#
# _collect_groq_api_keys() picks up every one of these that's set (up
# to _MAX_GROQ_KEYS), in order. _RotatingGroqClient is a drop-in stand-
# in for a real groq.Groq client -- every existing call site in this
# file that does `client.chat.completions.create(...)` keeps working
# completely unchanged. Internally, when the CURRENTLY ACTIVE key's
# daily quota (TPD) is exhausted, or the key is invalid/revoked, it
# transparently switches to the NEXT key in the list and retries the
# EXACT SAME request -- so a search/verify call continues from exactly
# the window it was already on, never restarting the document or
# losing place. Only once EVERY configured key has been exhausted does
# it finally raise GroqQuotaExhaustedError.
# =========================================================
_MAX_GROQ_KEYS = 10
def _collect_groq_api_keys(env_prefix: str = "GROQ_API_KEY") -> list:
    keys = []
    primary = get_api_key(env_prefix)
    if primary:
        keys.append(primary)
    for n in range(2, _MAX_GROQ_KEYS + 1):
        k = get_api_key(f"{env_prefix}_{n}")
        if k:
            keys.append(k)
    return keys
class _RotatingCompletions:
    def __init__(self, pool: "_RotatingGroqClient"):
        self._pool = pool
    def create(self, **kwargs):
        import groq
        while True:
            try:
                return self._pool.client.chat.completions.create(**kwargs)
            except groq.AuthenticationError:
                if self._pool.rotate(reason="was rejected as invalid (401)"):
                    continue
                raise
            except (groq.RateLimitError, groq.BadRequestError) as e:
                detail = _parse_rate_limit_detail(str(e))
                if detail and detail["limit_type"] == "TPD":
                    if self._pool.rotate(reason="hit its daily token quota (TPD)"):
                        continue
                raise
class _RotatingChat:
    def __init__(self, pool: "_RotatingGroqClient"):
        self.completions = _RotatingCompletions(pool)
class _RotatingGroqClient:
    def __init__(self, api_keys: list, budget: "_TokenBudgetTracker" = None, log=print):
        if not api_keys:
            raise Exception("No Groq API key(s) configured")
        from groq import Groq
        self._Groq = Groq
        self._keys = api_keys
        self._index = 0
        self._budget = budget
        self._log = log
        self._client = self._Groq(api_key=self._keys[0])
        self.chat = _RotatingChat(self)
        if len(api_keys) > 1:
            log(f"Groq key rotation enabled: {len(api_keys)} key(s) configured -- will fall back key-to-key on quota exhaustion.")
    @property
    def client(self):
        return self._client
    @property
    def key_count(self) -> int:
        return len(self._keys)
    def rotate(self, reason: str = "") -> bool:
        if self._index + 1 >= len(self._keys):
            self._log(
                f"WARNING: Groq key #{self._index + 1} of {len(self._keys)} (the LAST configured "
                f"key) also {reason or 'hit its limit'} -- no more keys left to fall back to."
            )
            return False
        self._index += 1
        self._client = self._Groq(api_key=self._keys[self._index])
        if self._budget is not None:
            self._budget.reset_window()
        self._log(
            f"Groq key #{self._index} of {len(self._keys)} {reason or 'hit its limit'} -- "
            f"switching to key #{self._index + 1} and continuing the SAME request from exactly "
            f"where it stopped (no restart, no lost progress)."
        )
        return True
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
    admin_pages = data.get("admin_pages", [])  # optional, backward-compatible
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
                              log, max_retries: int = 4, max_tokens: int = None):
    import groq
    estimated_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt) + 800
    last_error = None
    skip_next_proactive_check = False
    for attempt in range(1, max_retries + 2):
        if skip_next_proactive_check:
            skip_next_proactive_check = False
        else:
            budget.wait_if_needed(estimated_tokens, log=log)
        try:
            create_kwargs = dict(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            if max_tokens is not None:
                create_kwargs["max_tokens"] = max_tokens
            with _groq_call_lock:
                response = client.chat.completions.create(**create_kwargs)
            budget.record_usage(estimated_tokens)
            content = response.choices[0].message.content
            if content is None:
                raise ValueError(
                    "Groq returned empty content (finish_reason="
                    f"{getattr(response.choices[0], 'finish_reason', '?')}). This usually "
                    "means the response was cut off before completing valid JSON -- often "
                    "because max_tokens was too low for a long output (e.g. a long question "
                    "list). Will retry."
                )
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
                raise GroqQuotaExhaustedError(
                    f"Groq daily token quota (TPD) exhausted on ALL configured key(s): "
                    f"{detail['used']}/{detail['limit']} tokens used today, "
                    f"{detail['requested']} more requested. This will reset "
                    f"in approximately {detail['wait_seconds']/60:.0f} minute(s). "
                    f"Retrying within this run will not help -- either wait "
                    f"for the daily reset, add more backup keys (GROQ_API_KEY_2, "
                    f"GROQ_API_KEY_3, ...), or upgrade your Groq tier at "
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
    """
    FIX: previously used fuzzy similarity (>=90% text-similarity ratio
    plus >=92% word overlap) to catch "near duplicate" questions. This
    was a confirmed, reproduced cause of a REAL question silently
    disappearing from the final list (e.g. "12 questions on the paper,
    only 11 extracted"): sibling sub-parts of a multi-part question
    deliberately carry the SAME parent instruction text forward into
    each sub-part for self-containment (per the extraction prompt's own
    rules) -- e.g. "1.(i) Identify and explain: theme of light." and
    "1.(ii) Identify and explain: theme of darkness." share almost all
    their wording and easily cross a 90% fuzzy-similarity threshold
    despite being two completely separate, genuine questions. Only an
    EXACT match (after normalizing case/whitespace/leading numbering)
    now counts as a duplicate -- fuzzy similarity is no longer used at
    all for this check, so two questions must be identical in wording,
    not merely similar, to be treated as the same question.
    """
    return _normalize_question_key(q1) == _normalize_question_key(q2)
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
QUESTION_PAPER_ONLY_SYSTEM_PROMPT = """You are reading the OFFICIAL question paper pages of a student exam assignment booklet (the printed list of questions, NOT the student's answers). You are given the complete, exact text of these pages, in order.
Your task: extract the COMPLETE, clean list of every distinct question/sub-part, exactly as printed, and return them in printed order.
Critical rules for multi-part questions:
- If a single numbered question contains multiple LABELED sub-parts -- e.g. "1. Identify and explain the following: (i) ... (ii) ... (iii) ... (iv) ..." -- output EACH labeled sub-part as its OWN SEPARATE entry, not merged into one block. Each sub-part entry should include enough of the parent question's context to be self-contained (e.g. carry forward the parent instruction like "Identify and explain the following:" into each sub-part's text, or at minimum keep the original numbering label, e.g. "1.(i)", "1.(ii)", "1.(iii)", "1.(iv)") so each entry is independently understandable without needing to look at a different entry for context.
- This applies to ANY labeled sub-structure: (i)/(ii)/(iii)/(iv), (a)/(b)/(c), (क)/(ख)/(ग), 1./2./3. used as sub-parts within a larger numbered question, etc. -- always split these into separate entries.
- Decide this ONCE, consistently, for the whole document -- you are seeing the COMPLETE question paper text in this single call, so there is no need to guess or produce different splits for different parts of the same question.
- Preserve the EXACT original text of each part -- do not paraphrase, do not translate. You MAY prepend the parent question's numbering/label to each split-out sub-part for self-contained context, as described above.
- Output entries in the SAME ORDER they appear on the question paper (monotonic, matching the printed sequence) -- sub-parts of the same parent question must stay together and in their own (i)/(ii)/(iii)/(iv) order; never reorder anything.
Return ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:
{
  "questions": ["<exact text of question/sub-part 1>", "<exact text of question/sub-part 2>", ...]
}"""
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
# =========================================================
# QUESTION-EXTRACTION SANITY CHECK
#
# FIX (this round): a confirmed bug -- extract_canonical_questions can
# occasionally miss ONE question/sub-part out of the full list (e.g.
# extracting 11 of 12), even though the question paper's own numbering
# clearly shows it should be there. Since map_answers_sequential only
# ever creates a REF for questions that actually made it into this
# list, a missed question has NO slot at all downstream -- its answer
# content silently gets absorbed into whichever neighboring question's
# range happens to span that text (visible as "one question's answer
# is merged into the answer above it", except the missing question
# never even shows up as "unmatched" -- it's simply not in the list).
#
# This deterministic (non-LLM) regex scan finds every top-level
# question number that appears at the start of a line in the RAW
# question-paper OCR text (e.g. "12." starting a line), and compares
# that against the top-level numbers actually present in the extracted
# question list. Any number found in the raw text but missing from the
# extraction triggers ONE targeted retry with an explicit reminder
# naming the exact missing number(s) -- far more effective than a
# generic "did you get everything?" retry, since it tells the model
# precisely what to look for.
# =========================================================
_TOP_LEVEL_QNUM_RE = re.compile(r'(?:^|\n)\s*(\d{1,2})[.)]\s')
def _detect_top_level_question_numbers(qp_pages: list) -> set:
    numbers = set()
    for p in qp_pages:
        for m in _TOP_LEVEL_QNUM_RE.finditer(p["raw_text"]):
            try:
                numbers.add(int(m.group(1)))
            except ValueError:
                pass
    return numbers
def _extracted_top_level_numbers(questions: list) -> set:
    nums = set()
    for q in questions:
        n = _extract_leading_number(q)
        if n is not None:
            try:
                nums.add(int(n))
            except ValueError:
                pass
    return nums
# =========================================================
# SUB-PART-SCOPED SANITY CHECK
#
# FIX (this round): the top-level check above only catches a WHOLE
# question number missing entirely (e.g. "12." never extracted). It
# does NOT catch the equally-real case where a top-level question WAS
# detected, but one of ITS OWN sub-parts got dropped -- e.g. question
# 7 has (i)/(ii)/(iii)/(iv) printed on the paper, but only (i)/(ii)/
# (iii) made it into the extracted list. That missing sub-part still
# has no REF downstream, so its answer silently merges into a
# neighboring one -- same symptom, different cause.
#
# For each detected top-level number, this scopes the RAW OCR text to
# just the segment between that number's own occurrence and the NEXT
# detected number's occurrence (or end of text), and collects every
# distinct sub-part label -- (i)/(ii)/(a)/(b)/etc. -- appearing in that
# scoped segment. Comparing that against the sub-part labels actually
# extracted for that same number catches a dropped sub-part precisely.
# =========================================================
def _detect_expected_subparts_per_number(qp_pages: list, detected_numbers: set) -> dict:
    full_text = "\n".join(p["raw_text"] for p in qp_pages)
    positions = {}
    for n in detected_numbers:
        m = re.search(rf'(?:^|\n)\s*{n}[.)]\s', full_text)
        if m:
            positions[n] = m.start()
    ordered = sorted(positions.items(), key=lambda kv: kv[1])
    expected = {}
    for idx, (n, pos) in enumerate(ordered):
        end = ordered[idx + 1][1] if idx + 1 < len(ordered) else len(full_text)
        segment = full_text[pos:end]
        labels = {m.group(0) for m in _SUB_PART_LABEL_RE.finditer(segment)}
        if labels:
            expected[n] = labels
    return expected
def _extracted_subparts_per_number(questions: list) -> dict:
    result = {}
    for q in questions:
        n = _extract_leading_number(q)
        lbl = _extract_sub_part_label(q)
        if n is not None and lbl is not None:
            try:
                result.setdefault(int(n), set()).add(lbl)
            except ValueError:
                pass
    return result
# =========================================================
# TARGETED SINGLE-QUESTION RECOVERY
#
# Extracting ONE specific, precisely-named question or sub-part is a
# far simpler and more reliable task for the model than re-producing
# an entire N-item list correctly in one shot -- there's exactly one
# thing to find, so there's no "got 11 of 12 right" partial-failure
# mode possible. Used as the last-resort recovery step when even a
# full-relist retry (with an explicit reminder) still leaves a gap.
# =========================================================
SINGLE_QUESTION_EXTRACT_SYSTEM_PROMPT = """You are extracting the EXACT text of ONE specific numbered question (or one specific labeled sub-part of a multi-part question) from a student exam question paper's OCR text.
You are given:
1. The target: an exact question number and, if applicable, a specific sub-part label to extract.
2. The complete OCR text of the question paper.
Find that EXACT question or sub-part and return its complete original text, unmodified (no paraphrasing, no translation). If it is a sub-part of a larger parent question, include enough of the parent instruction for it to be self-contained (or at minimum keep the original numbering/label, e.g. "12.(iii)").
Return ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:
{"question_text": "<exact text>"}
If you genuinely cannot find it anywhere in the text, return:
{"question_text": null}"""
def _build_single_question_prompt(qp_pages: list, number: int, subpart_label: str = None) -> str:
    blocks = [f"--- PAGE {p['page_number']} ---\n{p['raw_text']}" for p in qp_pages]
    full_text = "\n\n".join(blocks)
    target = f"question number {number}" + (f", sub-part {subpart_label}" if subpart_label else "")
    return f"TARGET: {target}\n\nFULL QUESTION PAPER TEXT:\n{full_text}"
def _parse_single_question_response(content: str):
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()
    data = json.loads(content)
    if not isinstance(data, dict) or "question_text" not in data:
        raise ValueError(f"Response missing 'question_text' key: {data!r}")
    return data["question_text"]
def _recover_single_missing_question(client, qp_pages: list, number: int, subpart_label,
                                       budget: "_TokenBudgetTracker", log):
    prompt = _build_single_question_prompt(qp_pages, number, subpart_label)
    try:
        text = _call_groq_with_retries(
            client, SINGLE_QUESTION_EXTRACT_SYSTEM_PROMPT, prompt,
            _parse_single_question_response, budget, log, max_retries=2, max_tokens=1024
        )
    except GroqQuotaExhaustedError:
        raise
    except Exception as e:
        label_desc = f" sub-part {subpart_label}" if subpart_label else ""
        log(f"  targeted recovery for question {number}{label_desc} failed: {e}")
        return None
    if text and str(text).strip():
        return str(text).strip()
    return None
_ROMAN_VALUES = {'i': 1, 'v': 5, 'x': 10, 'l': 50, 'c': 100, 'd': 500, 'm': 1000}
def _roman_to_int(s: str):
    s = s.lower()
    if not s or any(ch not in _ROMAN_VALUES for ch in s):
        return None
    total = 0
    prev = 0
    for ch in reversed(s):
        val = _ROMAN_VALUES[ch]
        total += val if val >= prev else -val
        prev = val
    return total
def _sub_part_sort_key(label: str):
    inner = label.strip("()")
    roman = _roman_to_int(inner)
    if roman is not None:
        return (0, roman)
    if len(inner) == 1 and inner.isalpha():
        return (1, ord(inner.lower()))
    return (2, inner)
def _reorder_questions_by_number(questions: list) -> list:
    """
    Rebuilds the question list in correct printed order after targeted
    recovery may have appended items out of order -- groups by leading
    top-level number (numeric order), and within each group orders
    sub-parts by their roman-numeral/alphabetic sequence, falling back
    to original relative order for anything without a detectable
    number/label.
    """
    numbered = []
    unnumbered = []
    for q in questions:
        n = _extract_leading_number(q)
        if n is None:
            unnumbered.append(q)
            continue
        lbl = _extract_sub_part_label(q)
        key = (int(n), _sub_part_sort_key(lbl) if lbl else (-1, -1))
        numbered.append((key, q))
    numbered.sort(key=lambda kv: kv[0])
    return [q for _, q in numbered] + unnumbered
def extract_canonical_questions(qp_pages: list, status_callback=None) -> list:
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    if not qp_pages:
        return []
    groq_keys = _collect_groq_api_keys()
    if not groq_keys:
        raise Exception("GROQ_API_KEY not found in secrets or environment")
    budget = _TokenBudgetTracker()
    client = _RotatingGroqClient(groq_keys, budget=budget, log=log)
    user_prompt = _build_canonical_questions_prompt(qp_pages)
    log(f"Extracting canonical question list from {len(qp_pages)} question-paper page(s) in a single pass...")
    try:
        questions = _call_groq_with_retries(
            client, QUESTION_PAPER_ONLY_SYSTEM_PROMPT, user_prompt,
            _parse_canonical_questions_response, budget, log, max_tokens=8192
        )
    except GroqQuotaExhaustedError:
        raise
    except Exception as e:
        log(f"WARNING: canonical question extraction failed: {e}")
        return []
    log(f"Canonical question list: {len(questions)} question(s), single consistent pass")
    def _check(questions_now):
        detected_nums = _detect_top_level_question_numbers(qp_pages)
        extracted_nums = _extracted_top_level_numbers(questions_now)
        missing_nums = sorted(detected_nums - extracted_nums)
        expected_subparts = _detect_expected_subparts_per_number(qp_pages, detected_nums)
        extracted_subparts = _extracted_subparts_per_number(questions_now)
        missing_subparts = {}
        for n, labels in expected_subparts.items():
            if n in missing_nums:
                continue  # already flagged as a fully-missing top-level question
            got = extracted_subparts.get(n, set())
            gap = labels - got
            if gap:
                missing_subparts[n] = sorted(gap)
        return missing_nums, missing_subparts
    missing_nums, missing_subparts = _check(questions)
    if missing_nums or missing_subparts:
        problem_desc = []
        if missing_nums:
            problem_desc.append(f"whole question number(s) {missing_nums} missing entirely")
        if missing_subparts:
            problem_desc.append(
                "sub-part(s) missing from otherwise-detected question(s): "
                + ", ".join(f"Q{n} missing {labels}" for n, labels in missing_subparts.items())
            )
        log(
            f"WARNING: question-paper sanity check found a problem -- {'; '.join(problem_desc)}. "
            f"Retrying once with an explicit reminder naming the exact gap(s)..."
        )
        reminder_parts = [
            "IMPORTANT: a sanity check of the raw page text found gaps in your extraction above. "
            "Look again very carefully at the FULL text and fix these specific gaps:"
        ]
        if missing_nums:
            reminder_parts.append(
                f"- Whole question number(s) {missing_nums} do not appear anywhere in your list -- "
                f"find and include them (with all their own labeled sub-parts, if any)."
            )
        if missing_subparts:
            for n, labels in missing_subparts.items():
                reminder_parts.append(
                    f"- Question {n} is in your list, but its sub-part(s) {labels} are missing -- "
                    f"find and include EACH of these as its own separate entry."
                )
        reminder_parts.append("Do not skip any numbered question or labeled sub-part, even ones easy to overlook among the others.")
        reminder = "\n\n" + "\n".join(reminder_parts)
        try:
            retried_questions = _call_groq_with_retries(
                client, QUESTION_PAPER_ONLY_SYSTEM_PROMPT, user_prompt + reminder,
                _parse_canonical_questions_response, budget, log, max_retries=2, max_tokens=8192
            )
            retried_missing_nums, retried_missing_subparts = _check(retried_questions)
            improved = (
                len(retried_missing_nums) < len(missing_nums)
                or sum(len(v) for v in retried_missing_subparts.values()) < sum(len(v) for v in missing_subparts.values())
                or len(retried_questions) > len(questions)
            )
            if improved:
                log(f"  retry improved the extraction -- using the retry's result ({len(retried_questions)} question(s) total)")
                questions = retried_questions
                missing_nums, missing_subparts = retried_missing_nums, retried_missing_subparts
            else:
                log(f"  retry did not improve on the original result -- keeping it")
        except GroqQuotaExhaustedError:
            raise
        except Exception as e:
            log(f"  retry attempt failed: {e}")
        # FINAL GUARANTEE: if the full-relist retry above still didn't
        # close every gap, fall back to a TARGETED single-item
        # extraction per missing piece instead of giving up. Extracting
        # ONE specific numbered question (or ONE specific sub-part) is
        # a much simpler, more reliable task for the model than
        # re-producing the ENTIRE list again -- it has one narrow thing
        # to find instead of needing to get 12 things right at once.
        if missing_nums or missing_subparts:
            log(
                f"  full-list retry still has gaps -- falling back to targeted single-question "
                f"recovery for each missing piece individually..."
            )
            for n in list(missing_nums):
                recovered_text = _recover_single_missing_question(client, qp_pages, n, None, budget, log)
                if recovered_text:
                    questions.append(recovered_text)
                    log(f"  RECOVERED question {n} via targeted extraction: {recovered_text[:60]!r}...")
                    missing_nums.remove(n)
                else:
                    log(f"  targeted recovery could not find question {n} either -- giving up on it")
            for n, labels in list(missing_subparts.items()):
                for lbl in list(labels):
                    recovered_text = _recover_single_missing_question(client, qp_pages, n, lbl, budget, log)
                    if recovered_text:
                        questions.append(recovered_text)
                        log(f"  RECOVERED question {n} sub-part {lbl} via targeted extraction: {recovered_text[:60]!r}...")
                        labels.remove(lbl)
                    else:
                        log(f"  targeted recovery could not find question {n} sub-part {lbl} either -- giving up on it")
                if not labels:
                    del missing_subparts[n]
            questions = _reorder_questions_by_number(questions)
        if missing_nums or missing_subparts:
            log(
                f"WARNING: after every recovery attempt, gaps still remain -- missing whole "
                f"question(s): {missing_nums or 'none'}; missing sub-part(s): "
                f"{missing_subparts or 'none'}. Downstream answer-mapping will have NO slot "
                f"for these -- their content may get silently absorbed into a neighboring "
                f"question's answer. Please double-check the question paper pages manually."
            )
        else:
            log("  all previously-missing question(s)/sub-part(s) were successfully recovered.")
    return questions
def identify_questions_with_llm(pages: list, status_callback=None) -> tuple:
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    groq_keys = _collect_groq_api_keys()
    if not groq_keys:
        raise Exception("GROQ_API_KEY not found in secrets or environment")
    budget = _TokenBudgetTracker()
    client = _RotatingGroqClient(groq_keys, budget=budget, log=log)
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
        except GroqQuotaExhaustedError:
            raise
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
        # A page can't legitimately be both -- if the model contradicted
        # itself, keep it as a question-paper page (the more consequential
        # classification to get right) and drop it from admin.
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
            reclassified_as_answer_pages.extend(p["page_number"] - 1 for p in cluster)  # back to 0-based
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
            reclassified_as_answer_pages.extend(p["page_number"] - 1 for p in cluster)  # back to 0-based
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
# LLM-BASED ANSWER MAPPING (Groq)
# =========================================================
ANSWER_MAP_SYSTEM_PROMPT = """You are analyzing a student's handwritten answers (OCR'd) from an exam assignment booklet. You are given:
1. A numbered list of the OFFICIAL exam questions, each tagged with a reference label like [REF-A], [REF-B], etc.
2. The student's answer text, with each line prefixed by its line number in [brackets].
Your task: for EACH official question, find WHERE in the answer text the student's response to that specific question starts and ends, and return the LINE NUMBER RANGE (inclusive) for each, identified by its REF label.
Important guidance for finding boundaries correctly:
- A new answer typically begins where the student restates or references a question (e.g. "Ans 5-", "उत्तर 6-", "प्र. 8", a question number, or a clear topic shift matching the next question's subject).
- An answer's content ends at the LAST line that is still part of that answer's reasoning/explanation, RIGHT BEFORE the next answer begins (whether or not the next answer is in your list of official questions).
- If a question's answer is genuinely not present anywhere in the text shown, do NOT invent a range -- omit that REF entirely from your output. It may appear in a different chunk of the document.
- Each REF's range must NOT overlap with another REF's range. If you are unsure exactly where one answer ends and the next begins, prefer ending the EARLIER answer sooner rather than letting it swallow content that belongs to a later answer -- a short correct answer is far more useful than a long answer that incorrectly absorbed unrelated content.
- IGNORE OCR ARTIFACT-DESCRIPTION LINES when deciding boundaries -- e.g. a line that just describes a non-text element ("there is a logo", "red pen scribble", "signature", "watermark", "stamp here"). These are never part of the student's actual written answer; never let one of these lines count as the start or end of a range, and never let it cause two real answers to be merged into one range.
- NEVER merge two DIFFERENT questions' content into a single REF's range, even if their answers are similar or adjacent in the text. Each REF's range must contain ONLY that one question's own response.
- Use the line numbers EXACTLY as given in [brackets] -- do not estimate, guess, or renumber.
- Use the EXACT REF label (e.g. "REF-A") to identify each question. Do NOT retype or paraphrase the question text itself -- the REF label is all that's needed.
- If a note at the top of this prompt tells you this chunk CONTINUES an answer from a previous chunk, treat that instruction as authoritative: the opening lines of this chunk likely belong to that same REF even though you cannot see the earlier part of the answer.
- CRITICAL: this chunk is deliberately kept SHORT and normally contains only a small number of distinct answers (often 1-3). You MUST scan the ENTIRE text shown, all the way to the last line, before responding -- do not stop after finding the first one or two answers. If you can identify 3 separate answer-start points in this text, your output must contain 3 entries, not fewer. Missing a clearly-present answer is a serious error.
Return ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:
{
  "answers": [
    {"ref": "REF-A", "start_line": 12, "end_line": 18},
    {"ref": "REF-B", "start_line": 19, "end_line": 25}
  ]
}
If NONE of the official questions' answers appear in the text shown, return {"answers": []} -- that is a valid and expected result for a chunk that doesn't contain any of these answers."""
# =========================================================
# SEQUENTIAL SINGLE-TARGET ANSWER MAPPING (recommended, default)
# =========================================================
SEQUENTIAL_SEARCH_SYSTEM_PROMPT = """You are searching for ONE thing: the line where the response to ONE SPECIFIC question begins, in a line-numbered OCR window from a student's exam booklet.
Given: the target question's text, and a window of line-numbered text (the answer may not be in this window at all -- that's normal and expected).
Rules:
- A response usually starts where the student restates/references the question (e.g. "Ans 5-", "उत्तर 6-", a matching number) OR, with no label, where content clearly starts addressing this question's specific topic.
- Bare short labels ("Q1", "Q.i", "(i)") matching this question's own number/sub-part are sufficient on their own -- no restated text needed.
- Report the EARLIEST line where the answer begins, including any short intro/transition sentence before the topic sentence -- never a later line just because it's more clearly on-topic. Skipping the true opening line is a serious error.
- The same fact/definition can legitimately repeat across multiple answers or as a recap -- don't reject a genuine match just because similar wording appeared earlier.
- Ignore OCR artifact-description lines (e.g. "there is a logo", "signature", "watermark", "red pen scribble") -- never treat one as start_line.
- Never let a DIFFERENT question's content count as a match -- if this window's tail belongs to another question, only the exact line where THIS question's content begins counts.
- If unsure, report found=false rather than guessing -- a wrong match silently corrupts a different answer, which is worse than a temporary miss (a later pass can still find it).
Return ONLY valid JSON (no markdown fences, no commentary) in exactly one of these two shapes:
{"found": true, "start_line": 42}
or
{"found": false}
start_line must be an exact line number shown in [brackets] -- never invent or estimate one."""
def _build_sequential_search_prompt(window_lines: list, question_text: str, ref_label: str,
                                      extra_reminder: str = None, context_before: list = None) -> str:
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in window_lines)
    reminder_block = f"{extra_reminder}\n\n" if extra_reminder else ""
    context_block = ""
    if context_before:
        context_lines_block = "\n".join(f"[{idx}] {text}" for idx, text in context_before)
        context_block = (
            f"CONTEXT -- the lines immediately BEFORE this window (for reference only, "
            f"so you can tell whether the window's opening lines are a genuine NEW start "
            f"or a continuation of what came before -- these lines are NOT part of the "
            f"searchable window and must NEVER be reported as start_line):\n"
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
SEQUENTIAL_SEARCH_WINDOW_CHARS = 16000  # increased from 11000 -- larger windows mean fewer calls to scan the same document, cutting repeated system-prompt token overhead (the biggest fixed cost per call)
SEQUENTIAL_SEARCH_MAX_WINDOWS = 200  # generous safety cap; a real document will exhaust far sooner
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
        except GroqQuotaExhaustedError:
            # Do NOT swallow this as "window didn't match" -- every
            # remaining window would fail identically, and treating it
            # as a genuine non-match is exactly what caused unrelated
            # questions' content to silently pile up into an earlier
            # answer's range (two answers mixing together). Propagate
            # so the whole pipeline stops with one clear error instead.
            raise
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
        f"SPECIFICALLY. Sibling sub-parts of one question are often similar in subject matter "
        f"(they share the same overall topic) but are still separate, distinct responses -- do "
        f"not match content that is actually the student's response to a DIFFERENT sibling "
        f"sub-part just because it discusses a closely related aspect of the same topic."
    )
BOUNDARY_CONFIRM_SYSTEM_PROMPT = """You are double-checking a proposed boundary line between two consecutive answers in a student's exam booklet.
You are given:
1. The text of the PREVIOUS question (whose answer should end right before the proposed boundary).
2. The text of the CURRENT/target question (whose answer should begin at the proposed boundary).
3. A short window of text (line-numbered) centered on the proposed boundary line.
Decide: does the proposed boundary line genuinely mark the FIRST line of the CURRENT question's answer -- i.e., is everything from the boundary line onward truly about the CURRENT question, and everything before it truly still about the PREVIOUS question (or noise/labels)?
Two kinds of mistakes are possible, and both are equally serious:
- TOO EARLY: the proposed line is still part of the PREVIOUS answer's content (e.g. a coincidental keyword overlap, a sub-point within the previous explanation) -- the CURRENT answer actually starts LATER. This mistake silently truncates the PREVIOUS answer and steals its final content into the CURRENT one.
- TOO LATE: the CURRENT answer actually starts EARLIER than proposed (e.g. an introductory line, or an entire page, was missed).
If the proposed line is correct, confirm it. If not, report the corrected line number (which MUST be one of the line numbers shown in the window). If you cannot confidently identify a better line within this window, keep the original proposed line -- do not guess.
Ignore OCR artifact-description lines (e.g. "there is a logo", "red pen scribble", "signature", "watermark") entirely -- never propose one of these as the boundary; treat them as if they were not there and look at the real text around them.
Return ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:
{"corrected_start_line": 42}"""
def _build_boundary_confirm_prompt(window_lines: list, prev_question: str, curr_question: str, proposed_line: int) -> str:
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in window_lines)
    return (
        f"PREVIOUS QUESTION: {prev_question}\n\n"
        f"CURRENT QUESTION (target): {curr_question}\n\n"
        f"PROPOSED BOUNDARY LINE: {proposed_line}\n\n"
        f"TEXT WINDOW (line-numbered):\n{lines_block}"
    )
def _parse_boundary_confirm_response(content: str) -> int:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()
    data = json.loads(content)
    return int(data["corrected_start_line"])
def _confirm_boundary(client, numbered_lines: list, prev_question: str, curr_question: str,
                        proposed_line: int, budget: "_TokenBudgetTracker", log,
                        radius: int = 15) -> int:
    lo = max(0, proposed_line - radius)
    hi = min(len(numbered_lines) - 1, proposed_line + radius)
    window = [nl for nl in numbered_lines if lo <= nl[0] <= hi]
    if not window:
        return proposed_line
    user_prompt = _build_boundary_confirm_prompt(window, prev_question, curr_question, proposed_line)
    try:
        corrected = _call_groq_with_retries(
            client, BOUNDARY_CONFIRM_SYSTEM_PROMPT, user_prompt,
            _parse_boundary_confirm_response, budget, log, max_retries=1
        )
    except Exception as e:
        log(f"  boundary confirm failed for line {proposed_line} (keeping original): {e}")
        return proposed_line
    valid_ids = {idx for idx, _ in window}
    if corrected in valid_ids and corrected != proposed_line:
        log(f"  boundary confirm adjusted start: {proposed_line} -> {corrected}")
        return corrected
    return proposed_line
# =========================================================
# TOKEN/TIME OPTIMIZATION 1: deterministic (zero-LLM-cost) label
# pre-pass. Scans the whole document ONCE with a regex for lines
# carrying an explicit answer label ("Ans 5-", "उत्तर 6-", "Q.7", etc.)
# whose number matches a STANDALONE question's own leading number, in
# strictly increasing document order. Any question resolved this way
# skips BOTH the LLM search AND the boundary-check call entirely --
# on a labeled document this can eliminate the large majority of Groq
# calls (and their tokens) the sequential search would otherwise spend,
# and turns what used to be several network round-trips per question
# into a single O(total_lines) regex scan (a huge wall-clock win too).
# Deliberately excludes sibling sub-part questions (i)/(ii)/(iii) --
# they share the same parent number, so this simple per-number match
# can't safely tell them apart; those keep going through the existing
# group-handling logic untouched.
# =========================================================
def _build_label_anchor_index(answer_lines: list, questions: list, exclude_indices: set, log=print) -> dict:
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
            f"and boundary-check calls entirely for these."
        )
    return anchors
COMBINED_BOUNDARY_CHECK_SYSTEM_PROMPT = """You are checking the boundary at the start of one answer in a student's exam booklet, in ONE pass that covers both possible mistakes at once.
You are given:
1. The text of the PREVIOUS question (whose answer should end right before the true boundary), or "(none -- this is the first question)" if there isn't one.
2. The text of the CURRENT/target question (whose answer should begin at the true boundary).
3. A window of line-numbered text spanning some lines BEFORE the proposed boundary through some lines AFTER it.
4. The PROPOSED boundary line (currently believed to be where the current question's answer starts).
Decide the TRUE boundary line -- the exact first line of the CURRENT question's answer. Two kinds of mistakes are equally possible:
- The proposed line is too LATE: a genuine opening line/sentence of the CURRENT answer was skipped -- the true boundary is EARLIER.
- The proposed line is too EARLY: content still belonging to the PREVIOUS answer was wrongly included -- the true boundary is LATER.
If the proposed line is already correct, confirm it as-is. Ignore OCR artifact-description lines (e.g. "there is a logo", "signature", "watermark", "red pen scribble") -- never treat one as the boundary.
Return ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:
{"corrected_start_line": 42}
corrected_start_line MUST be one of the exact line numbers shown in the window -- if unsure, return the original proposed line."""
def _build_combined_boundary_prompt(window_lines: list, prev_question, curr_question: str, proposed_line: int) -> str:
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in window_lines)
    prev_block = prev_question if prev_question else "(none -- this is the first question)"
    return (
        f"PREVIOUS QUESTION: {prev_block}\n\n"
        f"CURRENT QUESTION (target): {curr_question}\n\n"
        f"PROPOSED BOUNDARY LINE: {proposed_line}\n\n"
        f"TEXT WINDOW (line-numbered):\n{lines_block}"
    )
def _check_boundary_combined(client, numbered_lines: list, prev_question, curr_question: str,
                               proposed_line: int, search_from_idx: int,
                               budget: "_TokenBudgetTracker", log,
                               back_radius: int = 40, forward_radius: int = 15) -> int:
    lo = max(search_from_idx, proposed_line - back_radius)
    hi = min(len(numbered_lines) - 1, proposed_line + forward_radius) if numbered_lines else proposed_line
    window = [nl for nl in numbered_lines if lo <= nl[0] <= hi]
    if not window:
        return proposed_line
    user_prompt = _build_combined_boundary_prompt(window, prev_question, curr_question, proposed_line)
    try:
        corrected = _call_groq_with_retries(
            client, COMBINED_BOUNDARY_CHECK_SYSTEM_PROMPT, user_prompt,
            _parse_boundary_confirm_response, budget, log, max_retries=1
        )
    except Exception as e:
        log(f"  combined boundary check failed for line {proposed_line} (keeping original): {e}")
        return proposed_line
    valid_ids = {idx for idx, _ in window}
    if corrected in valid_ids and corrected != proposed_line:
        log(f"  combined boundary check adjusted start: {proposed_line} -> {corrected}")
        return corrected
    return proposed_line
WINDOWED_MULTI_TARGET_SYSTEM_PROMPT = """You are resolving the internal boundaries WITHIN a small, already-confirmed block of text that belongs to a multi-part question's sibling sub-parts.
You are given:
1. A short list of candidate sub-parts (2-5 typically), each tagged with a REF label (REF-B, REF-C, ...). These are CONFIRMED to be sibling sub-parts of the SAME parent question -- the text window shown definitely contains some or all of their answers, back to back, in order.
2. The exact text window (line-numbered) that this whole group's answers fall within.
Your task: for each sibling sub-part, find the line where ITS OWN portion begins (i.e. where the student moves on from the previous sibling's content to start addressing this one specifically).
Guidance:
- The siblings appear in order in this window. Look for the transition points: a new label (e.g. (ii), (iii)), a sub-part identifier, or a clear shift in exactly what specific aspect is now being addressed -- even though all siblings share the same overall topic (they're part of one parent question).
- The FIRST sibling in the list normally starts at or very near the beginning of this window (it may already be confirmed separately -- focus your effort on finding where the LATER siblings begin).
- PRECISION MATTERS MOST: only report a sibling's start_line if you can identify a genuine, specific transition point for it. If you cannot clearly tell where one sibling's content ends and the next begins, it is much better to leave that boundary unreported than to guess -- an incorrect split would mix one sibling's answer into another's. A sibling with no reported start will simply be treated as folded into the content of whichever sibling came before it, which is a safer default than a wrong guess.
- Content, definitions, or explanations CAN legitimately repeat across siblings -- do not reject a genuine transition just because similar wording appeared for an earlier sibling.
- Always report the EARLIEST line at which each identified sibling's own content begins.
Return ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:
{"starts": [{"ref": "REF-B", "start_line": 42}, {"ref": "REF-C", "start_line": 58}]}
Only include sibling(s) whose start you can confidently identify -- omitting an uncertain one is expected and safe. Every start_line MUST be an exact line number shown in [brackets] in this window."""
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
def _estimate_expected_refs_in_window(window_lines: list, open_questions: list) -> set:
    q_texts = [q for _, q in open_questions]
    found_indices = set()
    current_idx = None
    for _, text in window_lines:
        idx = _line_starts_new_answer_for_question(text, q_texts)
        if idx is not None and idx != -1 and idx != current_idx:
            found_indices.add(idx)
            current_idx = idx
    return {open_questions[i][0] for i in found_indices}
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
    except GroqQuotaExhaustedError:
        raise
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
            continue  # can't start before/at the group's own confirmed start
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
                end_idx=upper + 1
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
def _reanalyze_and_repair_boundaries(client, numbered_lines: list, questions: list, ranges: list,
                                       budget: "_TokenBudgetTracker", log, max_passes: int = 1) -> list:
    if len(ranges) < 2:
        return ranges
    for pass_num in range(1, max_passes + 1):
        changed_this_pass = False
        ordered = sorted(ranges, key=lambda r: r["start_line"])
        lengths = [r["end_line"] - r["start_line"] + 1 for r in ordered]
        median_len = sorted(lengths)[len(lengths) // 2] if lengths else 0
        for pos in range(1, len(ordered)):
            prev_r = ordered[pos - 1]
            curr_r = ordered[pos]
            prev_len = prev_r["end_line"] - prev_r["start_line"] + 1
            curr_len = curr_r["end_line"] - curr_r["start_line"] + 1
            if median_len >= 4 and prev_len >= median_len * 0.4 and curr_len >= median_len * 0.4:
                continue  # both neighbors look like plausible, normal-length answers -- skip
            prev_q_idx = _ref_to_question_index(prev_r["ref"])
            curr_q_idx = _ref_to_question_index(curr_r["ref"])
            proposed = curr_r["start_line"]
            corrected = _confirm_boundary(
                client, numbered_lines, questions[prev_q_idx], questions[curr_q_idx],
                proposed, budget, log, radius=20
            )
            if (corrected != proposed
                    and prev_r["start_line"] < corrected <= curr_r["end_line"]):
                log(
                    f"  REANALYZE (pass {pass_num}): boundary between {prev_r['ref']} and "
                    f"{curr_r['ref']} corrected -- start moved {proposed} -> {corrected}"
                )
                prev_r["end_line"] = corrected - 1
                curr_r["start_line"] = corrected
                changed_this_pass = True
        if not changed_this_pass:
            break
    return ranges
def _remap_incomplete_answers(client, numbered_lines: list, questions: list, ranges: list,
                                budget: "_TokenBudgetTracker", log, max_passes: int = 1) -> list:
    if len(ranges) < 2:
        return ranges
    max_line = max((nl[0] for nl in numbered_lines), default=0)
    for pass_num in range(1, max_passes + 1):
        ordered = sorted(ranges, key=lambda r: r["start_line"])
        lengths = [r["end_line"] - r["start_line"] + 1 for r in ordered]
        if len(lengths) < 2:
            break
        median_len = sorted(lengths)[len(lengths) // 2]
        if median_len < 4:
            break
        changed = False
        for pos, r in enumerate(ordered):
            length = r["end_line"] - r["start_line"] + 1
            if length >= max(median_len * 0.25, 3):
                continue
            q_idx = _ref_to_question_index(r["ref"])
            log(
                f"  REMAP (pass {pass_num}): {r['ref']} looks half/incomplete "
                f"({length} line(s) vs this document's median of {median_len}) "
                f"-- re-searching it specifically, without touching any other question..."
            )
            lower = ordered[pos - 1]["start_line"] + 1 if pos > 0 else 0
            upper = ordered[pos + 1]["end_line"] if pos + 1 < len(ordered) else max_line
            if upper <= lower:
                continue
            new_start = _find_answer_start_sequential(
                client, numbered_lines, questions[q_idx], r["ref"], lower, budget, log,
                end_idx=upper + 1,
                extra_reminder=(
                    "REMINDER: an earlier pass mapped this question's answer to a "
                    "suspiciously short span -- this usually means the wrong (too short) "
                    "occurrence was matched, or the answer's true start was missed. Search "
                    "carefully across this whole window for the genuine, complete answer to "
                    "this exact question."
                )
            )
            if new_start is None or not (lower <= new_start <= upper) or new_start == r["start_line"]:
                continue
            new_end = upper
            if pos + 1 < len(ordered):
                next_q_idx = _ref_to_question_index(ordered[pos + 1]["ref"])
                confirmed_next_start = _confirm_boundary(
                    client, numbered_lines, questions[q_idx], questions[next_q_idx],
                    ordered[pos + 1]["start_line"], budget, log, radius=20
                )
                if new_start < confirmed_next_start <= upper + 1:
                    new_end = confirmed_next_start - 1
                    ordered[pos + 1]["start_line"] = confirmed_next_start
            if pos > 0:
                ordered[pos - 1]["end_line"] = new_start - 1
            r["start_line"] = new_start
            r["end_line"] = new_end
            log(f"  REMAP (pass {pass_num}): {r['ref']} corrected to lines {new_start}-{new_end}")
            changed = True
        if not changed:
            break
    return ranges
# =========================================================
# GUARANTEED-MAPPING FINAL PASS
#
# Runs LAST, after label anchors, search, rescue, reanalyze, remap, and
# verify have all had their turn. Every prior pass can legitimately
# leave a question unmatched (that's the SAFE default when precision
# isn't there) -- but the requirement is that EVERY question in the
# question paper ends up with SOME mapped content, never a blank
# answer. For any question still unmatched at this point:
#   1. One more, more PERMISSIVE search over its entire available gap
#      (bounded strictly by its nearest already-confirmed neighbors,
#      so it can never invade a different question's content).
#   2. If that still finds nothing, FORCE-ASSIGN the remaining gap as
#      a low-confidence best-effort match, flagged so it's easy to
#      spot-check -- an approximate answer is far more useful for
#      grading than a silently blank one.
# This never touches or reduces any OTHER question's already-confirmed
# range -- it only ever fills genuinely unclaimed gaps.
# =========================================================
GUARANTEE_SEARCH_REMINDER = (
    "FINAL PASS: every question in this document must end up matched to something -- "
    "be MORE lenient than a normal pass. If there is any plausible content in this "
    "window that could reasonably be this question's answer, even a loose or partial "
    "match, report it as found. Only report found=false if this window is genuinely "
    "and completely unrelated to this question's topic."
)
def _guarantee_full_mapping(client, numbered_lines: list, questions: list, ranges: list,
                              budget: "_TokenBudgetTracker", log) -> list:
    total_lines = len(numbered_lines)
    idx_to_ref = {i: f"REF-{chr(65 + i)}" for i in range(len(questions))}
    ref_to_idx = {v: k for k, v in idx_to_ref.items()}
    still_missing = [i for i in range(len(questions)) if idx_to_ref[i] not in {r["ref"] for r in ranges}]
    if not still_missing:
        return ranges
    log(f"Guarantee pass: {len(still_missing)} question(s) still unmatched after all other passes -- ensuring every question gets SOME mapped content...")
    for i in still_missing:
        ref = idx_to_ref[i]
        q = questions[i]
        ordered = sorted(ranges, key=lambda r: r["start_line"])
        lower, upper = 0, total_lines - 1
        for r in ordered:
            ridx = ref_to_idx.get(r["ref"])
            if ridx is None:
                continue
            if ridx < i:
                lower = max(lower, r["end_line"] + 1)
            if ridx > i:
                upper = min(upper, r["start_line"] - 1)
        if lower > upper:
            log(f"  GUARANTEE: {ref} has no available gap left between its neighbors -- leaving unmatched.")
            continue
        found_start = _find_answer_start_sequential(
            client, numbered_lines, q, ref, lower, budget, log,
            end_idx=upper + 1, extra_reminder=GUARANTEE_SEARCH_REMINDER
        )
        if found_start is not None and lower <= found_start <= upper:
            containing = next((r for r in ordered if r["start_line"] <= found_start <= r["end_line"]), None)
            if containing is not None:
                new_end = containing["end_line"]
                containing["end_line"] = found_start - 1
                ranges.append({"ref": ref, "start_line": found_start, "end_line": new_end})
            else:
                ranges.append({"ref": ref, "start_line": found_start, "end_line": upper})
            log(f"  GUARANTEE: recovered {ref} at line {found_start} via a more permissive final search")
        else:
            ranges.append({"ref": ref, "start_line": lower, "end_line": upper, "low_confidence": True})
            log(
                f"  GUARANTEE: could not confidently locate {ref} -- force-assigning the "
                f"remaining gap (lines {lower}-{upper}) as a LOW-CONFIDENCE best-effort match "
                f"so this question is never left blank. Please spot-check this one."
            )
    return ranges
def map_answers_sequential(answer_lines: list, questions: list, status_callback=None,
                             answer_line_pages: list = None) -> list:
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    groq_keys = _collect_groq_api_keys()
    if not groq_keys:
        raise Exception("GROQ_API_KEY not found in secrets or environment")
    budget = _TokenBudgetTracker()
    client = _RotatingGroqClient(groq_keys, budget=budget, log=log)
    numbered_lines = list(enumerate(answer_lines))
    total_lines = len(numbered_lines)
    ref_to_question = {f"REF-{chr(65+i)}": q for i, q in enumerate(questions)}
    sibling_groups = _detect_sibling_groups(questions)
    group_member_of = {idx: first_idx for first_idx, members in sibling_groups.items() for idx in members}
    label_anchors = _build_label_anchor_index(answer_lines, questions, set(group_member_of.keys()), log)
    found_starts = {}
    for qi, line_idx in label_anchors.items():
        found_starts[f"REF-{chr(65 + qi)}"] = line_idx
    pointer = 0
    i = 0
    n = len(questions)
    quota_exhausted = False
    while i < n and not quota_exhausted:
      try:
        if i in sibling_groups:
            group_indices = sibling_groups[i]
            group_refs = [f"REF-{chr(65 + j)}" for j in group_indices]
            group_questions = [(f"REF-{chr(65 + j)}", questions[j]) for j in group_indices]
            log(f"Detected sibling sub-part group {group_refs} -- resolving as a bounded batch...")
            first_ref, first_q = group_questions[0]
            group_search_from = pointer
            group_start = _find_answer_start_sequential(client, numbered_lines, first_q, first_ref, pointer, budget, log)
            attempt = 1
            while group_start is None and attempt <= 1:
                reminder = (
                    "REMINDER: a previous search pass did not find this answer. The same "
                    "definition/explanation can legitimately repeat across the document -- "
                    "that does not disqualify a genuine match. Also check for a short "
                    "introductory line at the true start."
                )
                group_start = _find_answer_start_sequential(
                    client, numbered_lines, first_q, first_ref, pointer, budget, log,
                    extra_reminder=reminder
                )
                attempt += 1
            if group_start is None:
                log(f"WARNING: could not find the start of sibling group {group_refs} at all -- marking all as unmatched.")
                i = group_indices[-1] + 1
                continue
            group_start = _check_boundary_combined(
                client, numbered_lines, questions[i - 1] if i > 0 else None, first_q,
                group_start, group_search_from, budget, log
            )
            found_starts[first_ref] = group_start
            log(f"  found {first_ref} (group start) at line {group_start}")
            next_index = group_indices[-1] + 1
            group_end_bound = None
            if next_index < n:
                next_ref = f"REF-{chr(65 + next_index)}"
                next_q = questions[next_index]
                group_end_bound = _find_answer_start_sequential(
                    client, numbered_lines, next_q, next_ref, group_start + 1, budget, log
                )
                if group_end_bound is not None:
                    group_end_bound = _check_boundary_combined(
                        client, numbered_lines, first_q, next_q,
                        group_end_bound, group_start + 1, budget, log
                    )
            upper = (group_end_bound - 1) if group_end_bound is not None else (total_lines - 1)
            if len(group_questions) > 1:
                sibling_starts = _resolve_sibling_group_batch(
                    client, numbered_lines, group_questions, group_start, upper, budget, log
                )
                for ref, sl in sibling_starts.items():
                    found_starts[ref] = sl
                    log(f"  found {ref} (sibling) at line {sl}")
                # FIX: sub-question merge bug -- previously, if the batch
                # call above didn't confidently separate a LATER sibling,
                # its content silently stayed folded into whichever
                # sibling precedes it (the visible symptom: a sub-
                # question's answer appears merged into the answer
                # above it). Recover any still-unresolved sibling with a
                # dedicated, well-tested single-target search (the same
                # search used for every standalone question) instead of
                # accepting the batch call's silence as final. Bounded
                # strictly within [previous confirmed sibling's start,
                # upper] so it can never reach into a DIFFERENT
                # question's territory outside this group.
                search_floor = group_start
                for gi in group_indices[1:]:
                    gref = f"REF-{chr(65 + gi)}"
                    if gref in found_starts:
                        search_floor = found_starts[gref]
                        continue
                    if search_floor >= upper:
                        continue
                    gq = questions[gi]
                    log(
                        f"  sibling {gref} not separated by the batch call -- retrying with a "
                        f"dedicated targeted search (lines {search_floor + 1}-{upper})..."
                    )
                    recovered = _find_answer_start_sequential(
                        client, numbered_lines, gq, gref, search_floor + 1, budget, log,
                        end_idx=upper + 1,
                        extra_reminder=_build_sub_part_hint(questions, gi)
                    )
                    if recovered is not None and search_floor < recovered <= upper:
                        found_starts[gref] = recovered
                        search_floor = recovered
                        log(
                            f"  RECOVERED sibling {gref} at line {recovered} -- it would otherwise "
                            f"have been silently merged into the sibling above it"
                        )
            unresolved = [ref for ref in group_refs[1:] if ref not in found_starts]
            if unresolved:
                log(
                    f"NOTE: sibling(s) {unresolved} were not confidently separated within "
                    f"the group's bounded region (lines {group_start}-{upper}) even after a "
                    f"dedicated retry -- their content stays folded into the preceding "
                    f"sibling's answer rather than risking a wrong split."
                )
            if next_index < n and group_end_bound is not None:
                found_starts[f"REF-{chr(65 + next_index)}"] = group_end_bound
                log(f"  found REF-{chr(65 + next_index)} at line {group_end_bound}")
                pointer = group_end_bound + 1
                i = next_index + 1
            else:
                pointer = total_lines
                i = next_index
            continue
        # ---- Standalone question: strict single-target search ----
        ref = f"REF-{chr(65 + i)}"
        q = questions[i]
        if ref in found_starts:
            pointer = found_starts[ref] + 1
            i += 1
            continue
        log(f"Searching for the start of {ref} ({q[:60]}...) from line {pointer} onward...")
        sub_part_hint = _build_sub_part_hint(questions, i)
        search_from_idx = pointer
        future_anchor_lines = [v for k, v in found_starts.items() if _ref_to_question_index(k) > i]
        bound_end_idx = (min(future_anchor_lines) + 1) if future_anchor_lines else None
        start_line = _find_answer_start_sequential(
            client, numbered_lines, q, ref, pointer, budget, log,
            extra_reminder=sub_part_hint, end_idx=bound_end_idx
        )
        attempt = 1
        while start_line is None and attempt <= 1:
            log(f"  pass {attempt} found nothing for {ref} -- retrying with a stronger reminder...")
            reminder_parts = [
                "REMINDER: a previous pass did not find this answer. The same "
                "definition/explanation can legitimately repeat across the document -- "
                "that does not disqualify a genuine match. Also check for a short "
                "introductory line at the true start."
            ]
            if sub_part_hint:
                reminder_parts.append(sub_part_hint)
            start_line = _find_answer_start_sequential(
                client, numbered_lines, q, ref, pointer, budget, log,
                extra_reminder="\n\n".join(reminder_parts), end_idx=bound_end_idx
            )
            if start_line is not None:
                log(f"  retry (pass {attempt + 1}) recovered {ref} starting at line {start_line}")
            attempt += 1
        if start_line is not None:
            start_line = _check_boundary_combined(
                client, numbered_lines, questions[i - 1] if i > 0 else None, q,
                start_line, search_from_idx, budget, log
            )
        if start_line is not None:
            found_starts[ref] = start_line
            log(f"  found {ref} starting at line {start_line}")
            pointer = start_line + 1
        else:
            log(
                f"WARNING: could not find the start of {ref} anywhere from line {pointer} "
                f"to the end of the document ({total_lines} lines) -- marking as unmatched. "
                f"The search pointer is NOT advanced, so the next question is still searched "
                f"for over this same remaining text."
            )
        i += 1
      except GroqQuotaExhaustedError as e:
        # CRITICAL FIX: previously, this propagated all the way up and
        # aborted the ENTIRE function -- discarding every answer already
        # found so far, even ones resolved for free via label anchors.
        # On a very tight/small Groq quota, this meant a document that
        # used to get MOST questions mapped (with only occasional
        # skipped paragraphs) would instead get almost NOTHING mapped,
        # a severe regression. Now: stop searching for anything further
        # (further calls would fail identically anyway), but KEEP every
        # answer already found, and let already-unmatched questions stay
        # genuinely unmatched (never silently merged into a neighbor).
        log(
            f"WARNING: Groq quota exhausted while still searching (at question index {i}) -- "
            f"stopping further searches now, but KEEPING every answer already found so far "
            f"({len(found_starts)} of {n} question(s)) rather than discarding all progress. "
            f"{e}"
        )
        quota_exhausted = True
    ordered = sorted(found_starts.items(), key=lambda kv: kv[1])
    ranges = []
    for idx, (ref, start) in enumerate(ordered):
        end = ordered[idx + 1][1] - 1 if idx + 1 < len(ordered) else total_lines - 1
        ranges.append({"ref": ref, "start_line": start, "end_line": end})
    log(f"Sequential mapping found {len(ranges)} of {len(questions)} question(s)")
    # The post-processing passes below (rescue/reanalyze/remap/guarantee)
    # each also call the API -- if quota is already known to be
    # exhausted, skip them entirely rather than let them raise and lose
    # the ranges collected above. If quota gets exhausted PARTWAY
    # through one of them instead, catch it there too and keep whatever
    # that pass had already produced.
    if not quota_exhausted:
        try:
            ranges = _rescue_unmatched_questions(client, numbered_lines, questions, ranges, budget, log)
        except GroqQuotaExhaustedError as e:
            log(f"WARNING: Groq quota exhausted during the rescue pass -- keeping results as they stood before this pass. {e}")
            quota_exhausted = True
    if not quota_exhausted:
        try:
            ranges = _reanalyze_and_repair_boundaries(client, numbered_lines, questions, ranges, budget, log)
        except GroqQuotaExhaustedError as e:
            log(f"WARNING: Groq quota exhausted during the reanalyze pass -- keeping results as they stood before this pass. {e}")
            quota_exhausted = True
    if not quota_exhausted:
        try:
            ranges = _remap_incomplete_answers(client, numbered_lines, questions, ranges, budget, log)
        except GroqQuotaExhaustedError as e:
            log(f"WARNING: Groq quota exhausted during the remap pass -- keeping results as they stood before this pass. {e}")
            quota_exhausted = True
    if not quota_exhausted:
        try:
            ranges = _guarantee_full_mapping(client, numbered_lines, questions, ranges, budget, log)
        except GroqQuotaExhaustedError as e:
            log(f"WARNING: Groq quota exhausted during the guarantee pass -- keeping results as they stood before this pass. {e}")
            quota_exhausted = True
    if quota_exhausted:
        log(
            "NOTE: this document was only PARTIALLY processed because the Groq quota ran out "
            "partway through -- add more backup keys (GROQ_API_KEY_2, GROQ_API_KEY_3, ...) or "
            "wait for the daily reset, then reprocess this document to fill in the rest."
        )
    ranges_by_ref = {r["ref"]: r for r in ranges}
    results = []
    for i, q in enumerate(questions):
        ref = f"REF-{chr(65 + i)}"
        r = ranges_by_ref.get(ref)
        if r is None:
            results.append({
                "ref": ref,
                "question": q,
                "matched": False,
                "start_line": None,
                "end_line": None,
                "start_page": None,
                "end_page": None,
                "answer": "",
                "answer_raw": "",
                "low_confidence": False,
            })
            continue
        s, e = r["start_line"], r["end_line"]
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
        answer_clean = strip_decorative_ocr_artifacts(answer_clean)
        start_page = answer_line_pages[s] if answer_line_pages and 0 <= s < len(answer_line_pages) else None
        end_page = answer_line_pages[e] if answer_line_pages and 0 <= e < len(answer_line_pages) else None
        results.append({
            "ref": ref,
            "question": q,
            "matched": True,
            "start_line": s,
            "end_line": e,
            "start_page": start_page,
            "end_page": end_page,
            "answer": answer_clean,
            "answer_raw": answer_raw,
            "low_confidence": bool(r.get("low_confidence", False)),
        })
    return results
def _build_answer_map_user_prompt(numbered_lines: list, questions: list,
                                    carry_over_ref: str = None) -> str:
    questions_block = "\n".join(
        f"[REF-{chr(65+i)}] {q}" for i, q in enumerate(questions)
    )
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in numbered_lines)
    carry_over_note = ""
    if carry_over_ref:
        carry_over_note = (
            f"IMPORTANT CONTEXT: This chunk is a CONTINUATION of a long answer that "
            f"started in a previous chunk, cut off only because of a length limit -- "
            f"NOT because the answer actually ended. The opening lines below are "
            f"very likely still part of the answer to {carry_over_ref}. If they read "
            f"as continuing that answer's reasoning (no new question is being "
            f"addressed), include them in {carry_over_ref}'s range using the line "
            f"numbers shown in THIS chunk. Only start counting a line as the "
            f"beginning of a genuinely NEW/DIFFERENT answer once the content clearly "
            f"shifts to a different topic or question.\n\n"
        )
    return (
        f"{carry_over_note}"
        f"OFFICIAL QUESTIONS (each tagged with its own [REF-X] label -- "
        f"use the REF label, not retyped question text, to identify which "
        f"question an answer belongs to):\n{questions_block}\n\n"
        f"STUDENT'S ANSWER TEXT (line-numbered):\n{lines_block}"
    )
def _parse_answer_map_llm_response(content: str) -> list:
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
    if not isinstance(data, dict) or "answers" not in data:
        raise ValueError(f"LLM response missing 'answers' key. Got: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}")
    answers = data["answers"]
    if not isinstance(answers, list):
        raise ValueError(f"'answers' must be a list, got: {type(answers).__name__}")
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
ANSWER_MAP_MAX_CHARS_PER_CHUNK = 11000
ANSWER_MAP_ABSOLUTE_MAX_CHARS = 60000
_ANSWER_START_RE = re.compile(
    r'^\s*(?:Ans(?:wer)?[.\s:-]+\d*|उत्तर\s*\d*\s*[\-\:]?|प्र[०.\s]*\d*|Q\.?\s*\d+[.\s:-])',
    re.IGNORECASE
)
def _normalize_for_overlap_match(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    return text
_QUESTION_STOPWORDS = {
    'how', 'are', 'the', 'views', 'state', 'with', 'theme', 'examine',
    'write', 'detailed', 'note', 'their', 'corresponding', 'why', 'does',
    'plot', 'plan', 'comment', 'discuss', 'explain', 'describe', 'and',
    'what', 'when', 'where', 'which', 'who', 'integrated', 'analyse',
    'analyze', 'critically', 'briefly', 'elaborate', 'illustrate', 'for',
    'from', 'this', 'that', 'these', 'those', 'into', 'about', 'role',
    'significance', 'importance', 'short', 'long', 'play', 'text',
}
def _distinctive_words(text: str, max_words: int = 20) -> list:
    words = re.findall(r'[a-z]{3,}', _normalize_for_overlap_match(text))[:max_words]
    return sorted(set(w for w in words if w not in _QUESTION_STOPWORDS))
def _line_starts_new_answer_for_question(line: str, questions: list, min_fraction: float = 0.5):
    label_match = _ANSWER_START_RE.match(line)
    if label_match:
        num_match = re.search(r'\d+', label_match.group(0))
        if num_match:
            label_num = num_match.group(0)
            for i, q in enumerate(questions):
                q_num_match = re.match(r'\s*(\d+)', q)
                if q_num_match and q_num_match.group(1) == label_num:
                    return i
        return -1
    line_words = sorted(set(re.findall(r'[a-z]{3,}', _normalize_for_overlap_match(line))[:25]))
    if not line_words:
        return None
    for i, q in enumerate(questions):
        q_distinctive = _distinctive_words(q)
        if not q_distinctive:
            continue
        matched = sum(
            1 for w in q_distinctive
            if any(_words_nearly_match(w, lw) for lw in line_words)
        )
        required = max(1, round(len(q_distinctive) * min_fraction))
        if matched >= required:
            return i
    return None
MAX_ANSWERS_PER_CHUNK = 1
def _chunk_lines_by_char_budget(numbered_lines: list, questions: list,
                                  max_chars: int = ANSWER_MAP_MAX_CHARS_PER_CHUNK,
                                  absolute_max_chars: int = ANSWER_MAP_ABSOLUTE_MAX_CHARS,
                                  max_answers_per_chunk: int = MAX_ANSWERS_PER_CHUNK) -> list:
    if not numbered_lines:
        return []
    chunks = []
    carry_overs = []
    expected_new_indices_per_chunk = []
    current_chunk = []
    current_chars = 0
    past_target = False
    current_question_idx = None
    carry_over_for_current_chunk = None
    answers_seen_in_current_chunk = 0
    expected_new_indices_current = []
    for idx, text in numbered_lines:
        line_chars = len(text)
        if current_chunk and current_chars + line_chars > max_chars:
            past_target = True
        matched_q_idx = _line_starts_new_answer_for_question(text, questions)
        is_genuine_new_start = matched_q_idx is not None and (
            matched_q_idx == -1 or matched_q_idx != current_question_idx
        )
        should_break_at_answer_start = (
            is_genuine_new_start and
            (past_target or answers_seen_in_current_chunk >= max_answers_per_chunk)
        )
        should_force_break_absolute = (
            current_chunk and current_chars + line_chars > absolute_max_chars
        )
        if should_break_at_answer_start or should_force_break_absolute:
            chunks.append(current_chunk)
            carry_overs.append(carry_over_for_current_chunk)
            expected_new_indices_per_chunk.append(expected_new_indices_current)
            current_chunk = []
            current_chars = 0
            past_target = False
            answers_seen_in_current_chunk = 0
            expected_new_indices_current = []
            if should_force_break_absolute and not should_break_at_answer_start:
                carry_over_for_current_chunk = current_question_idx
            else:
                carry_over_for_current_chunk = None
        if is_genuine_new_start and matched_q_idx != -1:
            current_question_idx = matched_q_idx
            answers_seen_in_current_chunk += 1
            expected_new_indices_current.append(matched_q_idx)
        current_chunk.append((idx, text))
        current_chars += line_chars
    if current_chunk:
        chunks.append(current_chunk)
        carry_overs.append(carry_over_for_current_chunk)
        expected_new_indices_per_chunk.append(expected_new_indices_current)
    return list(zip(chunks, carry_overs, expected_new_indices_per_chunk))
def _resolve_overlapping_answer_ranges(answer_ranges: list) -> list:
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
    r'\s*#{1,3}\s*\d*\s*$'                                   # trailing "#", "##3"
    r'|\s*(?:Section|Sec\.?)\s*[-:]?\s*[A-Za-z0-9]+\s*$'      # "Section B", "Sec-2"
    r'|\s*भाग\s*[-:]?\s*[०-९0-9]*\s*$'                        # "भाग-2"
    r'|\s*(?:Q\.?|Ans(?:wer)?\.?|प्र\.?|प्रश्न\.?|उत्तर)\s*[-:.]?\s*\d+\s*[-:.)]?\s*$'  # "Q5", "Ans 6-", "प्र.7"
    r'|\s*\(?[ivxlcdm]{1,5}\)?\s*[-:.)]?\s*$'                 # trailing bare roman-numeral label, e.g. "(iii)"
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
# =========================================================
# DECORATIVE OCR ARTIFACT CLEANUP
#
# FIX: Datalab/Chandra OCR sometimes renders section/question dividers
# as decorative markdown-style headings with star symbols, e.g.:
#   "## भाग - 1 ### ★ प्रश्नोत्तर नं: 3 ★"
# These are page-layout decorations, never real answer content -- but
# since all answer lines get flattened into ONE space-joined string
# (no newlines survive), such a heading can end up EMBEDDED anywhere
# in the final text, not just at a clean boundary: most often at the
# very end of one answer (where the OCR line for the next question's
# divider got swept into the previous range) or at the very start of
# the next one. This runs on the FULL answer text (not just the
# trailing edge) so it catches the artifact wherever it landed.
# =========================================================
_DECORATIVE_STAR_CHARS = '★☆✦✧❋❖✩✪✫✬✭✮✯'
_DECORATIVE_STAR_BLOCK_RE = re.compile(
    rf'[{_DECORATIVE_STAR_CHARS}]+\s*[^{_DECORATIVE_STAR_CHARS}]{{0,60}}?[{_DECORATIVE_STAR_CHARS}]+'
)
_STRAY_STAR_RE = re.compile(rf'[{_DECORATIVE_STAR_CHARS}]+')
_MARKDOWN_HEADING_HASH_RE = re.compile(r'#{1,6}\s*')
_BHAG_SECTION_RE = re.compile(r'भाग\s*[-–:]?\s*[०-९0-9]+')
_PRASHNOTTAR_HEADING_RE = re.compile(r'प्रश्नोत्तर\s*नं\.?\s*[:\-]?\s*[०-९0-9]*')
def strip_decorative_ocr_artifacts(text: str) -> str:
    if not text:
        return text
    cleaned = _DECORATIVE_STAR_BLOCK_RE.sub(' ', text)   # "★ प्रश्नोत्तर नं: 3 ★" as a whole block
    cleaned = _PRASHNOTTAR_HEADING_RE.sub(' ', cleaned)   # any leftover "प्रश्नोत्तर नं: 3" without stars
    cleaned = _BHAG_SECTION_RE.sub(' ', cleaned)          # "भाग - 1" section markers
    cleaned = _STRAY_STAR_RE.sub(' ', cleaned)            # any remaining lone star symbols
    cleaned = _MARKDOWN_HEADING_HASH_RE.sub(' ', cleaned) # markdown "##"/"###" hashes
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned
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
def map_answers_with_llm(answer_lines: list, questions: list, status_callback=None) -> dict:
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    groq_keys = _collect_groq_api_keys()
    if not groq_keys:
        raise Exception("GROQ_API_KEY not found in secrets or environment")
    budget = _TokenBudgetTracker()
    client = _RotatingGroqClient(groq_keys, budget=budget, log=log)
    ref_to_question = {f"REF-{chr(65+i)}": q for i, q in enumerate(questions)}
    numbered_lines = list(enumerate(answer_lines))
    chunks_with_carry = _chunk_lines_by_char_budget(numbered_lines, questions)
    log(f"Split {len(answer_lines)} answer line(s) into {len(chunks_with_carry)} LLM chunk(s) for answer mapping "
        f"(max {MAX_ANSWERS_PER_CHUNK} distinct answers per chunk)")
    all_ranges = []  # list of {ref, start_line, end_line}
    chunk_failures = []
    chunk_zero_matches = 0
    for i, (chunk, carry_over_idx, expected_new_indices) in enumerate(chunks_with_carry):
        line_range = f"{chunk[0][0]}-{chunk[-1][0]}" if chunk else "empty"
        carry_over_ref = f"REF-{chr(65 + carry_over_idx)}" if carry_over_idx is not None else None
        if carry_over_ref:
            log(
                f"Chunk {i+1}/{len(chunks_with_carry)} continues an answer split across "
                f"chunks -- flagging {carry_over_ref} as carried over"
            )
        log(f"Asking LLM to map answers in chunk {i+1}/{len(chunks_with_carry)} (lines {line_range})...")
        user_prompt = _build_answer_map_user_prompt(chunk, questions, carry_over_ref)
        try:
            chunk_ranges = _call_groq_with_retries(
                client, ANSWER_MAP_SYSTEM_PROMPT, user_prompt,
                _parse_answer_map_llm_response, budget, log
            )
        except GroqQuotaExhaustedError:
            raise
        except Exception as e:
            log(f"WARNING: chunk {i+1}/{len(chunks_with_carry)} answer-mapping failed, skipping: {e}")
            chunk_failures.append(str(e))
            continue
        expected_refs = {f"REF-{chr(65 + qi)}" for qi in expected_new_indices}
        if carry_over_ref:
            expected_refs.add(carry_over_ref)
        returned_refs = {r.get("ref") for r in chunk_ranges if isinstance(r, dict)}
        missing_refs = expected_refs - returned_refs
        if missing_refs:
            missing_previews = [
                f"{ref} ({ref_to_question.get(ref, '?')[:50]}...)" for ref in sorted(missing_refs)
            ]
            log(
                f"WARNING: chunk {i+1}/{len(chunks_with_carry)} looks like it stopped early -- "
                f"expected answers for {sorted(expected_refs)} but only got {sorted(returned_refs)}. "
                f"Missing: {missing_previews}. Retrying this chunk once with an explicit reminder..."
            )
            reminder = (
                f"\n\nREMINDER: your previous attempt on this exact text did NOT include a range for "
                f"{sorted(missing_refs)}. A genuine answer-start for {'each of these' if len(missing_refs) > 1 else 'this'} "
                f"was detected in the text below. Look again at the FULL text, all the way to its last line, "
                f"and make sure your JSON output includes an entry for {sorted(missing_refs)} if their content "
                f"is present -- do not stop before reaching the end of the text shown."
            )
            try:
                retry_ranges = _call_groq_with_retries(
                    client, ANSWER_MAP_SYSTEM_PROMPT, user_prompt + reminder,
                    _parse_answer_map_llm_response, budget, log, max_retries=2
                )
                recovered = [r for r in retry_ranges if isinstance(r, dict) and r.get("ref") in missing_refs]
                if recovered:
                    log(f"  retry recovered {len(recovered)} of {len(missing_refs)} missing answer(s)")
                    chunk_ranges = chunk_ranges + recovered
                else:
                    log(f"  retry did not recover the missing answer(s) -- they may genuinely be split across chunk boundaries")
            except Exception as e:
                log(f"  retry attempt failed: {e}")
        if not chunk_ranges:
            chunk_zero_matches += 1
        valid_indices = {idx for idx, _ in chunk}
        min_idx, max_idx = min(valid_indices), max(valid_indices)
        for r in chunk_ranges:
            if r["ref"] not in ref_to_question:
                log(f"WARNING: discarding answer mapping with unknown ref {r['ref']!r}")
                continue
            if not (min_idx <= r["start_line"] <= max_idx and min_idx <= r["end_line"] <= max_idx):
                log(
                    f"WARNING: discarding out-of-range answer mapping for "
                    f"{r['ref']}: lines {r['start_line']}-{r['end_line']} "
                    f"outside this chunk's range {min_idx}-{max_idx}"
                )
                continue
            if carry_over_ref and r["ref"] == carry_over_ref:
                existing = next((x for x in reversed(all_ranges) if x["ref"] == carry_over_ref), None)
                if existing is not None:
                    existing["end_line"] = max(existing["end_line"], r["end_line"])
                    log(f"  merged continuation into existing {carry_over_ref} range -> now ends at line {existing['end_line']}")
                    continue
            all_ranges.append(r)
        log(f"Chunk {i+1}/{len(chunks_with_carry)}: mapped {len(chunk_ranges)} answer(s)")
    best_by_ref = {}
    for r in all_ranges:
        existing = best_by_ref.get(r["ref"])
        if existing is None or (r["end_line"] - r["start_line"]) > (existing["end_line"] - existing["start_line"]):
            best_by_ref[r["ref"]] = r
    deduped_ranges = list(best_by_ref.values())
    resolved_ranges = _resolve_overlapping_answer_ranges(deduped_ranges)
    log(f"Final answer mapping: {len(resolved_ranges)} of {len(questions)} question(s) matched")
    if not resolved_ranges:
        if chunk_failures and len(chunk_failures) == len(chunks_with_carry):
            raise Exception(
                f"Answer mapping failed: ALL {len(chunks_with_carry)} chunk(s) raised an "
                f"error (none succeeded). First failure: {chunk_failures[0]}"
            )
        elif chunk_zero_matches == len(chunks_with_carry):
            sample_lines = [l for l in answer_lines[:15] if l.strip()][:8]
            raise Exception(
                f"Answer mapping found ZERO matches across all {len(chunks_with_carry)} chunk(s), "
                f"even though the LLM calls themselves succeeded. This usually means "
                f"the 'answer pages' passed in do NOT actually contain the student's "
                f"answers -- most likely the question-paper/answer-page page split "
                f"upstream misclassified pages (e.g. real answer pages were wrongly "
                f"identified as question-paper pages, leaving only cover/admin pages "
                f"as 'answers'). Sample of the answer text actually searched: "
                f"{sample_lines}"
            )
    qa_map = {}
    for r in resolved_ranges:
        start, end = r["start_line"], r["end_line"]
        verbatim_lines = [
            answer_lines[j] for j in range(start, end + 1)
            if 0 <= j < len(answer_lines) and answer_lines[j].strip() and not is_noise(answer_lines[j])
        ]
        original_question = ref_to_question[r["ref"]]
        ref_idx = int(ord(r["ref"].split("-")[-1]) - ord("A")) if r["ref"].startswith("REF-") else None
        next_question_text = questions[ref_idx + 1] if ref_idx is not None and ref_idx + 1 < len(questions) else None
        answer_text = " ".join(verbatim_lines).strip()
        answer_text = strip_question_restatement(answer_text)
        answer_text = strip_full_question_echo(answer_text, original_question)
        answer_text = strip_trailing_leaked_next_question(answer_text, next_question_text)
        answer_text = strip_trailing_next_question_leadin(answer_text)
        answer_text = strip_decorative_ocr_artifacts(answer_text)
        qa_map[original_question] = answer_text
    return qa_map
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
    if strip_decorative_ocr_artifacts(stripped) == '':
        return True
    if len(stripped) > NOISE_LINE_MAX_CHARS:
        return False
    return bool(NOISE_RE.search(stripped))
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
def similarity(a: str, b: str) -> float:
    wa = set(normalize(a).split())
    wb = set(normalize(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))
def strip_leading_label(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^(?:Ans(?:wer)?[.\s]+)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^(?:उत्तर)\s*[\-\:\s]*', '', text)
    text = re.sub(r'^(?:प्र|प्रो|प्रश्न)[\.\s]*\d*[\.\s]*', '', text)
    text = re.sub(r'^[१-९०][०-९]*[\.\-\s]*', '', text)
    text = re.sub(r'^(?:Q\.?\s*)?\d+[.)]\s*', '', text)
    text = re.sub(r'^\(?[a-z]\)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\(?[क-घ]\)\s*', '', text)
    return text.strip()
def find_question_boundaries_by_similarity(
    answer_lines: list,
    questions: list,
    similarity_threshold: float = 0.30,
    window: int = 4
) -> list:
    candidates_by_question = {}
    for i in range(len(answer_lines)):
        line_i = answer_lines[i].strip()
        if len(line_i) < 8:
            continue
        for w in range(1, window + 1):
            if i + w > len(answer_lines):
                break
            combined = " ".join(
                answer_lines[i + k].strip()
                for k in range(w) if answer_lines[i + k].strip()
            )
            if len(combined) < 10:
                continue
            combined_clean = strip_leading_label(combined)
            for q in questions:
                q_clean = strip_leading_label(q)
                s1 = similarity(combined, q)
                s2 = similarity(combined_clean, q_clean)
                score = max(s1, s2)
                if score >= similarity_threshold:
                    candidates_by_question.setdefault(q, []).append({
                        "question":   q,
                        "line_index": i,
                        "span":       w,
                        "score":      score
                    })
    for q in candidates_by_question:
        candidates_by_question[q].sort(key=lambda c: -c["score"])
    final = []
    last_line_index = -1
    for q in questions:
        cands = candidates_by_question.get(q, [])
        chosen = None
        for c in cands:
            if c["line_index"] > last_line_index:
                chosen = c
                break
        if chosen is not None:
            final.append(chosen)
            last_line_index = chosen["line_index"]
    return final
def slice_raw_answers_by_boundaries(answer_lines: list, boundaries: list) -> list:
    qa_pairs = []
    for i, b in enumerate(boundaries):
        span    = b.get("span", 1)
        a_start = b["line_index"] + span
        a_end   = boundaries[i + 1]["line_index"] if i + 1 < len(boundaries) else len(answer_lines)
        raw = [
            answer_lines[j] for j in range(a_start, a_end)
            if answer_lines[j].strip() and not is_noise(answer_lines[j])
        ]
        qa_pairs.append({
            "question": b["question"],
            "answer":   " ".join(raw).strip()
        })
    return qa_pairs
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
# COMPLETE PIPELINE
# =========================================================
@_diagnose_tuple_errors
def process_pdf(file_input, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    file_bytes, file_name = _normalize_file_input(file_input, default_name="document.pdf")
    pages = run_ocr(file_bytes, file_name, status_callback)
    log("Building OCR JSON...")
    ocr_json = build_ocr_json(pages)
    log(f"Total pages: {ocr_json['total_pages']}")
    try:
        qp_page_indices, official_questions, admin_page_indices = identify_questions_with_llm(pages, status_callback)
    except GroqQuotaExhaustedError as e:
        raise Exception(
            f"Stopped processing during question-paper identification: every configured "
            f"Groq API key's daily quota is exhausted. {e}\n\n"
            f"Add more backup keys (GROQ_API_KEY_2, GROQ_API_KEY_3, ...) in secrets, or "
            f"wait for the daily reset, then reprocess this document."
        ) from e
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
    try:
        qa_pairs = map_answers_sequential(
            answer_lines, official_questions, status_callback,
            answer_line_pages=answer_line_pages
        )
    except GroqQuotaExhaustedError as e:
        raise Exception(
            f"Stopped processing partway through answer-mapping: every configured Groq API "
            f"key's daily quota is exhausted. {e}\n\n"
            f"This document was NOT fully mapped, and no partial/corrupted results are "
            f"returned -- add more backup keys (GROQ_API_KEY_2, GROQ_API_KEY_3, ...) in "
            f"secrets, or wait for the daily reset, then reprocess this document."
        ) from e
    matched_count = sum(1 for p in qa_pairs if p["matched"])
    low_conf_count = sum(1 for p in qa_pairs if p.get("low_confidence"))
    log(f"Matched {matched_count} of {len(official_questions)} questions"
        + (f" ({low_conf_count} low-confidence best-effort match(es) -- worth spot-checking)" if low_conf_count else ""))
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
    log(f"Done -- {len(qa_pairs)} Q-A pairs ({matched_count} matched)")
    return ocr_json, qa_pairs
def to_simple_qa_json(qa_pairs: list) -> list:
    """
    Simplifies the internal (rich) qa_pairs structure -- which carries
    debugging fields like start_line/end_line/start_page/end_page/
    matched/low_confidence/answer_raw for internal use -- down to
    EXACTLY what was requested for external consumption: a plain list
    of {"Q": ..., "A": ...} objects, nothing else. Unmatched questions
    still get an entry (with an empty "A") so the output always has
    one entry per question in the paper, in order.
    """
    return [{"Q": p["question"], "A": p["answer"]} for p in qa_pairs]
def save_outputs(ocr_json: dict, qa_pairs: list, output_dir: str = ".",
                  base_name: str = "document") -> tuple:
    ocr_path = os.path.join(output_dir, f"{base_name}_ocr.json")
    qa_path = os.path.join(output_dir, f"{base_name}_qa_pairs.json")
    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(ocr_json, f, ensure_ascii=False, indent=2)
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(to_simple_qa_json(qa_pairs), f, ensure_ascii=False, indent=2)
    return ocr_path, qa_path
