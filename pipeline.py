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
            raise ValueError(f"Tuple file_input must have at least (filename, bytes), got {len(file_input)} items")
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
            raise TypeError(f"file_input.read() returned {type(data).__name__}, expected bytes. Open the file in binary mode ('rb').")
        name = getattr(file_input, "name", default_name)
        return bytes(data), _coerce_name(name, default_name)
    raise TypeError(
        f"Unsupported file_input type: {type(file_input).__name__}. "
        f"Expected str, Path, bytes, a file-like object with .read(), or a (filename, bytes) tuple."
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
                    f"[DIAGNOSTIC] Caught the 'os.PathLike, not tuple' error INSIDE {func.__name__}(), "
                    f"not before it. file_input received: type={type(file_input).__name__}, repr={file_input!r}. "
                    f"Original error: {e}"
                ) from e
            raise
    return wrapper
# =========================================================
# SAFE LOGGING -- never let a broken status_callback crash the pipeline.
# Streamlit's status_callback raises NoSessionContext when called from a
# background thread (our ThreadPoolExecutor batch workers) -- previously that
# exception escaped straight out of a warning-log call and took down the whole
# pipeline. Now it's always swallowed here.
# =========================================================
def _make_safe_logger(status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            try:
                status_callback(msg)
            except Exception:
                pass
    return log
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
        f"Treating entire document as a single page. First 200 chars: {markdown[:200]!r}"
    )
    return [markdown.strip()]
def run_ocr(file_content: bytes, file_name: str, status_callback=None):
    log = _make_safe_logger(status_callback)
    file_name = _coerce_name(file_name, default_name="document.pdf")
    if not isinstance(file_content, (bytes, bytearray)):
        raise TypeError(f"run_ocr() expected file_content as bytes, got {type(file_content).__name__}")
    api_key = get_api_key("DATALAB_API_KEY")
    if not api_key:
        raise Exception("DATALAB_API_KEY not found in secrets or environment")
    size_mb = len(file_content) / (1024 * 1024)
    MAX_MB = 45
    if size_mb > MAX_MB:
        raise Exception(f"File is {size_mb:.1f}MB, which exceeds the {MAX_MB}MB upload limit. Try compressing or splitting the PDF.")
    headers = {"X-API-Key": api_key}
    log(f"Submitting document to Datalab (Chandra OCR)... ({size_mb:.1f}MB)")
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
    pages = [{"page_number": idx + 1, "raw_text": text} for idx, text in enumerate(page_texts)]
    log(f"OCR done -- {len(pages)} page(s) extracted")
    if len(pages) == 1 and size_mb > 1.0:
        log(f"WARNING: Only 1 page extracted from a {size_mb:.1f}MB file -- page-break marker format may not have been recognized. Markdown length: {len(markdown)} chars.")
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
def build_ocr_json(pages: list) -> dict:
    return {"total_pages": len(pages), "pages": [{"page_number": p["page_number"], "text": p["raw_text"]} for p in pages]}
@_diagnose_tuple_errors
def process_reference(file_input, status_callback=None):
    log = _make_safe_logger(status_callback)
    file_bytes, file_name = _normalize_file_input(file_input, default_name="reference.pdf")
    pages = run_ocr_cached(file_bytes, file_name, status_callback)
    log(f"Reference OCR complete -- {len(pages)} page(s)")
    return build_ocr_json(pages)
# =========================================================
# LLM PROVIDER ABSTRACTION
# Default provider is Anthropic Claude (claude-sonnet-5, the current Sonnet
# model as of this writing -- Claude Sonnet 4.6 was superseded by Sonnet 5 on
# 2026-06-30). Groq remains available as a fallback/cheaper option via
# LLM_PROVIDER=groq. Every call-site in this file goes through ONE function,
# _call_groq_with_retries (name kept for backward compatibility with the rest
# of this module), so switching providers never requires touching any of the
# question-identification / answer-mapping logic below.
# =========================================================
def _detect_llm_provider() -> str:
    """
    Respects an explicit LLM_PROVIDER env var if set. Otherwise auto-detects
    based on whichever API key is actually configured -- prefers Anthropic if
    both are present, falls back to Groq if only GROQ_API_KEY exists. This
    avoids a hard crash like "No ANTHROPIC API key found" just because someone
    only ever set up GROQ_API_KEY and never explicitly chose a provider.
    """
    explicit = os.getenv("LLM_PROVIDER")
    if explicit:
        return explicit.strip().lower()
    if get_api_key("ANTHROPIC_API_KEY"):
        return "anthropic"
    if get_api_key("GROQ_API_KEY"):
        return "groq"
    return "anthropic"  # neither configured -- keep this as the named default so the resulting error is specific and actionable
LLM_PROVIDER = _detect_llm_provider()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
CHARS_PER_TOKEN_ESTIMATE = 2.0
# Conservative defaults; override with LLM_TPM_LIMIT if your account/tier is higher.
_DEFAULT_TPM = {"anthropic": 30000, "groq": 8000}
TPM_LIMIT = int(os.getenv("LLM_TPM_LIMIT", str(_DEFAULT_TPM.get(LLM_PROVIDER, 20000))))
TPM_SAFETY_FRACTION = 0.85
GROQ_MAX_CONCURRENT_CALLS = int(os.getenv("LLM_MAX_CONCURRENT_CALLS", "3"))
_groq_call_semaphore = threading.Semaphore(GROQ_MAX_CONCURRENT_CALLS)
_budget_lock = threading.Lock()
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
                    self.events.append((now, upcoming_tokens))
                    return
                # If a single request alone already exceeds the whole safe budget,
                # waiting can never help (nothing accumulated to "free up") --
                # send it directly instead of looping forever.
                if upcoming_tokens >= self.safe_limit:
                    if not already_warned_oversized:
                        log(
                            f"NOTE: this request alone (~{upcoming_tokens} tokens) is at/above "
                            f"the tracked safe budget ({self.safe_limit:.0f}) -- sending directly; "
                            f"the retry logic will back off on a real 429 if one occurs."
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
                f"+{upcoming_tokens} upcoming would exceed safe budget ({self.safe_limit:.0f}). "
                f"Waiting {wait_s:.1f}s before sending next chunk..."
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
        "limit_type": limit_type.upper(), "period": period.lower(), "limit": int(limit),
        "used": int(used), "requested": int(requested), "wait_seconds": wait_seconds,
    }
class LLMQuotaExhaustedError(Exception):
    """Raised when every configured API key's quota is exhausted (Groq daily
    quota, or Anthropic account out of credit). Callers propagate this instead
    of swallowing it as a normal 'window didn't match' result -- swallowing it
    silently was a confirmed bug: once quota ran out mid-document, every
    remaining search failed the same way but was treated as a genuine
    non-match, so later questions' content quietly piled up into an earlier
    answer's range."""
    pass
_MAX_API_KEYS = 10
def _collect_api_keys() -> list:
    prefix = "ANTHROPIC_API_KEY" if LLM_PROVIDER == "anthropic" else "GROQ_API_KEY"
    keys = []
    primary = get_api_key(prefix)
    if primary:
        keys.append(primary)
    for n in range(2, _MAX_API_KEYS + 1):
        k = get_api_key(f"{prefix}_{n}")
        if k:
            keys.append(k)
    return keys
# Kept for any external code that imported the old Groq-specific name.
_collect_groq_api_keys = _collect_api_keys
class _RotatingLLMClient:
    """Provider-agnostic client. `.create(system_prompt, user_prompt, max_tokens)`
    returns (text, actual_tokens_used_or_None). Transparently rotates to the
    next configured API key on auth failure or (Groq-only) daily-quota
    exhaustion, retrying the SAME request from where it left off."""
    def __init__(self, api_keys: list, budget: "_TokenBudgetTracker" = None, log=print):
        if not api_keys:
            raise Exception(
            f"No {LLM_PROVIDER.upper()} API key found. Add "
            f"{'ANTHROPIC_API_KEY' if LLM_PROVIDER == 'anthropic' else 'GROQ_API_KEY'} "
            f"to st.secrets or your environment (or set LLM_PROVIDER=groq / LLM_PROVIDER=anthropic "
            f"explicitly to pick which one this should use)."
        )
        self._keys = api_keys
        self._index = 0
        self._budget = budget
        self._log = log
        self._make_client()
        if len(api_keys) > 1:
            log(f"{LLM_PROVIDER} key rotation enabled: {len(api_keys)} key(s) configured -- will fall back key-to-key if needed.")
    def _make_client(self):
        if LLM_PROVIDER == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._keys[self._index])
        else:
            from groq import Groq
            self._client = Groq(api_key=self._keys[self._index])
    @property
    def key_count(self) -> int:
        return len(self._keys)
    def rotate(self, reason: str = "") -> bool:
        if self._index + 1 >= len(self._keys):
            self._log(f"WARNING: key #{self._index + 1} of {len(self._keys)} (the LAST configured key) also {reason or 'hit its limit'} -- no more keys left.")
            return False
        self._index += 1
        self._make_client()
        if self._budget is not None:
            self._budget.reset_window()
        self._log(f"Key #{self._index} of {len(self._keys)} {reason or 'hit its limit'} -- switching to key #{self._index + 1} and continuing the SAME request.")
        return True
    def create(self, system_prompt: str, user_prompt: str, max_tokens: int = 2048):
        if LLM_PROVIDER == "anthropic":
            resp = self._client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                system=system_prompt + "\n\nRespond with ONLY the JSON object described above -- no markdown fences, no commentary, no preamble.",
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
            )
            text = "".join(getattr(b, "text", "") for b in resp.content)
            usage = getattr(resp, "usage", None)
            total_tokens = (usage.input_tokens + usage.output_tokens) if usage else None
            return text, total_tokens
        else:
            resp = self._client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content
            usage = getattr(resp, "usage", None)
            total_tokens = getattr(usage, "total_tokens", None) if usage else None
            return text, total_tokens
def _get_provider_sdk():
    if LLM_PROVIDER == "anthropic":
        import anthropic
        return anthropic, anthropic.AuthenticationError, anthropic.RateLimitError
    else:
        import groq
        return groq, groq.AuthenticationError, groq.RateLimitError
def _call_groq_with_retries(client, system_prompt: str, user_prompt: str, response_parser,
                              budget: "_TokenBudgetTracker", log, max_retries: int = 4,
                              max_tokens: int = None):
    _sdk, AuthErr, RateErr = _get_provider_sdk()
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
                content, actual_tokens = client.create(system_prompt, user_prompt, max_tokens=max_tokens or 2048)
            if not reserved_via_wait:
                budget.record_usage(actual_tokens or estimated_tokens)
            if content is None or not str(content).strip():
                raise ValueError(
                    "LLM returned empty content -- usually means the response was cut off "
                    "before completing valid JSON (max_tokens too low for a long output). Will retry."
                )
            return response_parser(content)
        except AuthErr as e:
            raise Exception(
                f"{LLM_PROVIDER.upper()} API rejected the API key (401). This will NOT be fixed by "
                f"retrying -- check that {'ANTHROPIC_API_KEY' if LLM_PROVIDER == 'anthropic' else 'GROQ_API_KEY'} "
                f"is set correctly (no extra whitespace/quotes) and hasn't been revoked. Original error: {e}"
            ) from e
        except RateErr as e:
            last_error = e
            detail = _parse_rate_limit_detail(str(e)) if LLM_PROVIDER == "groq" else None
            if detail and detail["limit_type"] == "TPD":
                if hasattr(client, "rotate") and client.rotate(reason="hit its daily token quota (TPD)"):
                    skip_next_proactive_check = True
                    continue
                raise LLMQuotaExhaustedError(
                    f"Groq daily token quota (TPD) exhausted on ALL configured key(s): "
                    f"{detail['used']}/{detail['limit']} tokens used today. Wait for the daily reset, "
                    f"add more backup keys (GROQ_API_KEY_2, ...), or switch LLM_PROVIDER=anthropic."
                ) from e
            if detail:
                budget.record_actual_from_error(detail["used"], detail["limit"])
                wait_s = detail["wait_seconds"] + 0.5
            else:
                retry_after = None
                resp_obj = getattr(e, "response", None)
                if resp_obj is not None and hasattr(resp_obj, "headers"):
                    retry_after = resp_obj.headers.get("retry-after")
                wait_s = (float(retry_after) + 0.5) if retry_after else 5.0 * attempt
            log(f"Rate limit hit (attempt {attempt}): {e}. Waiting {wait_s:.1f}s before retrying...")
            time.sleep(wait_s)
            budget.reset_window()
            skip_next_proactive_check = True
        except Exception as e:
            last_error = e
            log(f"LLM call/parse attempt {attempt} failed: {e}")
            time.sleep(1)
    raise Exception(f"LLM call failed after {max_retries + 1} attempts. Last error: {last_error}")
# =========================================================
# STAGE 1: QUESTION-PAPER / ANSWER PAGE CLASSIFICATION
# =========================================================
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
def _chunk_pages_by_char_budget(pages: list, max_chars: int = MAX_CHARS_PER_CHUNK, overlap_pages: int = CHUNK_OVERLAP_PAGES) -> list:
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
    return "Here are the OCR'd pages shown in this chunk:\n\n" + "\n\n".join(blocks)
def _parse_qp_llm_response(content: str) -> tuple:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content).strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}\nRaw content (first 500 chars): {content[:500]!r}")
    if not isinstance(data, dict):
        raise ValueError(f"LLM response must be a JSON object, got: {type(data).__name__}")
    if "question_paper_pages" not in data or "questions" not in data:
        raise ValueError(f"LLM response missing required keys. Got keys: {list(data.keys())}")
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
def _call_groq_for_chunk(client, pages_chunk: list, budget: "_TokenBudgetTracker", log, max_retries: int = 4) -> tuple:
    user_prompt = _build_qp_user_prompt(pages_chunk)
    return _call_groq_with_retries(client, QP_SYSTEM_PROMPT, user_prompt, _parse_qp_llm_response, budget, log, max_retries, max_tokens=4096)
def _normalize_question_key(q: str) -> str:
    # 100% STRICT: only whitespace-trim + collapse repeated internal whitespace.
    # No lowercasing, no leading-number stripping, no punctuation removal --
    # two questions are only ever treated as duplicates if their text is a
    # literal, exact match. Anything less than that stays as two separate
    # questions, even if they look extremely similar.
    return re.sub(r'\s+', ' ', q.strip())
def _words_nearly_match(w1: str, w2: str) -> bool:
    if w1 == w2:
        return True
    if abs(len(w1) - len(w2)) > 2:
        return False
    return difflib.SequenceMatcher(None, w1, w2).ratio() >= 0.8
def _is_near_duplicate_question(q1: str, q2: str) -> bool:
    """
    100% EXACT match only (after trimming/collapsing whitespace) -- nothing
    fuzzy, nothing case-insensitive, no stripping of leading numbering. This
    is deliberately the strictest possible rule: a question is only ever
    dropped as a duplicate if it is a literal, word-for-word, character-for-
    character match to one already kept (this legitimately happens when
    overlapping chunk boundaries cause the exact same printed text to be
    extracted twice). Anything short of 100% identical text -- including
    sibling sub-parts that share most of their wording, or two questions
    that merely look similar -- is kept as its own separate question. Fuzzy
    similarity was tried before and is exactly why a real question could
    silently vanish from the list; this rule guarantees that never happens.
    """
    return _normalize_question_key(q1) == _normalize_question_key(q2)
def _dedup_questions(questions: list) -> list:
    unique = []
    for q in questions:
        if not any(_is_near_duplicate_question(q, existing) for existing in unique):
            unique.append(q)
    return unique
def _merge_chunk_results(chunk_results: list) -> tuple:
    all_qp_pages, all_admin_pages, all_questions = set(), set(), []
    for qp_pages, questions, admin_pages in chunk_results:
        all_qp_pages.update(qp_pages)
        all_admin_pages.update(admin_pages)
        all_questions.extend(questions)
    return sorted(all_qp_pages), _dedup_questions(all_questions), sorted(all_admin_pages)
QUESTION_PAPER_ONLY_SYSTEM_PROMPT = """Complete, exact text of the OFFICIAL question paper pages (not the student's answers), in order. Extract the full clean list of every distinct question/sub-part exactly as printed, in printed order.
Multi-part rule: if a numbered question has labeled sub-parts -- (i)/(ii)/(iii), (a)/(b)/(c), (क)/(ख), or 1./2./3. used as sub-parts -- split EACH into its own entry, not merged. Keep each entry self-contained: carry the parent instruction forward or at least keep the label (e.g. "1.(i)", "1.(ii)"). Decide this once, consistently, for the whole set. Preserve exact original text (no paraphrase/translation); output in the same printed order, sub-parts grouped and ordered under their parent.
Return ONLY this JSON (no fences, no commentary):
{"questions": ["<exact text 1>", "<exact text 2>", ...]}"""
def _build_canonical_questions_prompt(qp_pages: list) -> str:
    blocks = [f"--- PAGE {p['page_number']} ---\n{p['raw_text']}" for p in qp_pages]
    return "Here is the COMPLETE text of all question paper pages, in order:\n\n" + "\n\n".join(blocks)
def _parse_canonical_questions_response(content: str) -> list:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content).strip()
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
# QUESTION-EXTRACTION SANITY CHECK -- deterministic regex scan of the RAW
# question-paper text catches a missed whole question number AND a missed
# sub-part within an otherwise-detected question, then retries with an
# explicit reminder, then falls back to targeted single-item recovery.
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
SINGLE_QUESTION_EXTRACT_SYSTEM_PROMPT = """You are extracting the EXACT text of ONE specific numbered question (or one specific labeled sub-part) from a student exam question paper's OCR text.
Given: the target question number and, if applicable, sub-part label; and the complete OCR text of the question paper.
Find that EXACT question/sub-part and return its complete original text, unmodified. If it's a sub-part, include enough of the parent instruction for it to be self-contained (or at minimum keep the numbering, e.g. "12.(iii)").
Return ONLY this JSON: {"question_text": "<exact text>"}
If genuinely not found: {"question_text": null}"""
def _build_single_question_prompt(qp_pages: list, number: int, subpart_label: str = None) -> str:
    blocks = [f"--- PAGE {p['page_number']} ---\n{p['raw_text']}" for p in qp_pages]
    full_text = "\n\n".join(blocks)
    target = f"question number {number}" + (f", sub-part {subpart_label}" if subpart_label else "")
    return f"TARGET: {target}\n\nFULL QUESTION PAPER TEXT:\n{full_text}"
def _parse_single_question_response(content: str):
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content).strip()
    data = json.loads(content)
    if not isinstance(data, dict) or "question_text" not in data:
        raise ValueError(f"Response missing 'question_text' key: {data!r}")
    return data["question_text"]
def _recover_single_missing_question(client, qp_pages: list, number: int, subpart_label, budget: "_TokenBudgetTracker", log):
    prompt = _build_single_question_prompt(qp_pages, number, subpart_label)
    try:
        text = _call_groq_with_retries(client, SINGLE_QUESTION_EXTRACT_SYSTEM_PROMPT, prompt, _parse_single_question_response, budget, log, max_retries=2, max_tokens=1024)
    except LLMQuotaExhaustedError:
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
    total, prev = 0, 0
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
    numbered, unnumbered = [], []
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
    log = _make_safe_logger(status_callback)
    if not qp_pages:
        return []
    api_keys = _collect_api_keys()
    if not api_keys:
        raise Exception(
            f"No {LLM_PROVIDER.upper()} API key found. Add "
            f"{'ANTHROPIC_API_KEY' if LLM_PROVIDER == 'anthropic' else 'GROQ_API_KEY'} "
            f"to st.secrets or your environment (or set LLM_PROVIDER=groq / LLM_PROVIDER=anthropic "
            f"explicitly to pick which one this should use)."
        )
    budget = _TokenBudgetTracker()
    client = _RotatingLLMClient(api_keys, budget=budget, log=log)
    user_prompt = _build_canonical_questions_prompt(qp_pages)
    log(f"Extracting canonical question list from {len(qp_pages)} question-paper page(s) in a single pass...")
    try:
        questions = _call_groq_with_retries(client, QUESTION_PAPER_ONLY_SYSTEM_PROMPT, user_prompt, _parse_canonical_questions_response, budget, log, max_tokens=8192)
    except LLMQuotaExhaustedError:
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
                continue
            gap = labels - extracted_subparts.get(n, set())
            if gap:
                missing_subparts[n] = sorted(gap)
        return missing_nums, missing_subparts
    missing_nums, missing_subparts = _check(questions)
    if missing_nums or missing_subparts:
        problem_desc = []
        if missing_nums:
            problem_desc.append(f"whole question number(s) {missing_nums} missing entirely")
        if missing_subparts:
            problem_desc.append("sub-part(s) missing: " + ", ".join(f"Q{n} missing {labels}" for n, labels in missing_subparts.items()))
        log(f"WARNING: question-paper sanity check found a problem -- {'; '.join(problem_desc)}. Retrying once with an explicit reminder...")
        reminder_parts = ["IMPORTANT: a sanity check found gaps in your extraction above. Look again very carefully and fix these specific gaps:"]
        if missing_nums:
            reminder_parts.append(f"- Whole question number(s) {missing_nums} do not appear anywhere in your list -- find and include them (with all their own sub-parts, if any).")
        if missing_subparts:
            for n, labels in missing_subparts.items():
                reminder_parts.append(f"- Question {n} is in your list, but its sub-part(s) {labels} are missing -- include EACH as its own separate entry.")
        reminder_parts.append("Do not skip any numbered question or labeled sub-part, even ones easy to overlook.")
        reminder = "\n\n" + "\n".join(reminder_parts)
        try:
            retried = _call_groq_with_retries(client, QUESTION_PAPER_ONLY_SYSTEM_PROMPT, user_prompt + reminder, _parse_canonical_questions_response, budget, log, max_retries=2, max_tokens=8192)
            r_missing_nums, r_missing_subparts = _check(retried)
            improved = (len(r_missing_nums) < len(missing_nums)
                        or sum(len(v) for v in r_missing_subparts.values()) < sum(len(v) for v in missing_subparts.values())
                        or len(retried) > len(questions))
            if improved:
                log(f"  retry improved the extraction -- using it ({len(retried)} question(s) total)")
                questions = retried
                missing_nums, missing_subparts = r_missing_nums, r_missing_subparts
            else:
                log("  retry did not improve -- keeping original result")
        except LLMQuotaExhaustedError:
            raise
        except Exception as e:
            log(f"  retry attempt failed: {e}")
        if missing_nums or missing_subparts:
            log("  full-list retry still has gaps -- falling back to targeted single-question recovery for each missing piece...")
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
                        log(f"  RECOVERED question {n} sub-part {lbl}: {recovered_text[:60]!r}...")
                        labels.remove(lbl)
                    else:
                        log(f"  targeted recovery could not find question {n} sub-part {lbl} either")
                if not labels:
                    del missing_subparts[n]
            questions = _reorder_questions_by_number(questions)
        if missing_nums or missing_subparts:
            log(f"WARNING: after every recovery attempt, gaps remain -- missing whole question(s): {missing_nums or 'none'}; missing sub-part(s): {missing_subparts or 'none'}. Please double-check manually.")
        else:
            log("  all previously-missing question(s)/sub-part(s) were successfully recovered.")
    return questions
def identify_questions_with_llm(pages: list, status_callback=None) -> tuple:
    log = _make_safe_logger(status_callback)
    api_keys = _collect_api_keys()
    if not api_keys:
        raise Exception(
            f"No {LLM_PROVIDER.upper()} API key found. Add "
            f"{'ANTHROPIC_API_KEY' if LLM_PROVIDER == 'anthropic' else 'GROQ_API_KEY'} "
            f"to st.secrets or your environment (or set LLM_PROVIDER=groq / LLM_PROVIDER=anthropic "
            f"explicitly to pick which one this should use)."
        )
    budget = _TokenBudgetTracker()
    client = _RotatingLLMClient(api_keys, budget=budget, log=log)
    chunks = _chunk_pages_by_char_budget(pages)
    log(f"Split {len(pages)} page(s) into {len(chunks)} LLM chunk(s) to respect token limits")
    valid_page_numbers = {p["page_number"] for p in pages}
    max_page_number = max(valid_page_numbers) if valid_page_numbers else 0
    chunk_results, chunk_failures = [], []
    for i, chunk in enumerate(chunks):
        page_nums_in_chunk = [p["page_number"] for p in chunk]
        log(f"Asking LLM to analyze chunk {i+1}/{len(chunks)} (pages {page_nums_in_chunk})...")
        try:
            qp_pages_1based, questions, admin_pages_1based = _call_groq_for_chunk(client, chunk, budget, log)
        except LLMQuotaExhaustedError:
            raise
        except Exception as e:
            log(f"WARNING: chunk {i+1}/{len(chunks)} question-identification failed, skipping: {e}")
            chunk_failures.append(str(e))
            continue
        def _recover_pages(pages_1based, label):
            recovered, truly_invalid = [], []
            for pn in pages_1based:
                if pn in valid_page_numbers:
                    recovered.append(pn)
                    continue
                split_result = _try_split_concatenated_page_number(pn, valid_page_numbers, max_page_number)
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
        log(f"Chunk {i+1}/{len(chunks)}: identified {len(qp_pages_1based)} question paper page(s), {len(admin_pages_1based)} admin/cover page(s)")
        chunk_results.append((qp_pages_1based, [], admin_pages_1based))
    if chunk_failures and not chunk_results:
        raise Exception(f"All {len(chunks)} chunk(s) failed during question identification. First failure: {chunk_failures[0]}")
    elif chunk_failures:
        log(f"NOTE: {len(chunk_failures)} of {len(chunks)} chunk(s) failed and were skipped -- question PAGE detection below is PARTIAL.")
    qp_pages_1based_merged, _, admin_pages_1based_merged = _merge_chunk_results(chunk_results)
    qp_page_indices_0based = sorted(pn - 1 for pn in qp_pages_1based_merged)
    admin_page_indices_0based = sorted(pn - 1 for pn in admin_pages_1based_merged)
    log(f"Question paper pages identified: {len(qp_page_indices_0based)} page(s)")
    log(f"Admin/cover pages identified: {len(admin_page_indices_0based)} page(s)")
    if len(qp_page_indices_0based) >= 2:
        qp_page_lengths = [(i, len(pages[i]["raw_text"])) for i in qp_page_indices_0based]
        lengths_only = [length for _, length in qp_page_lengths]
        median_length = sorted(lengths_only)[len(lengths_only) // 2]
        outliers = [page_idx for page_idx, length in qp_page_lengths if length > max(median_length * 3, 1500)]
        if outliers and len(outliers) <= len(qp_page_indices_0based) // 2:
            for page_idx in outliers:
                length = dict(qp_page_lengths)[page_idx]
                log(f"RECLASSIFYING page {page_idx + 1}: {length} chars, much longer than median {median_length} -- likely a student answer restating the question. Moving to answer pages.")
            qp_page_indices_0based = [i for i in qp_page_indices_0based if i not in outliers]
        elif outliers:
            log(f"WARNING: {len(outliers)} of {len(qp_page_indices_0based)} detected question-paper pages are unusually long (median {median_length}). Leaving as question-paper pages, but the split may be unreliable. Pages: {[p+1 for p in outliers]}")
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
        log(f"WARNING: question-paper pages form {len(page_clusters)} non-contiguous cluster(s) -- using only the largest (pages {[p['page_number'] for p in chosen_cluster]}) as the real question paper.")
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
            log(f"WARNING: cluster (pages {cluster_page_nums}) produced ZERO questions -- moving these page(s) to the answer-page pool.")
            reclassified_as_answer_pages.extend(p["page_number"] - 1 for p in cluster)
    if reclassified_as_answer_pages:
        qp_page_indices_0based = [i for i in qp_page_indices_0based if i not in reclassified_as_answer_pages]
        log(f"Reclassified {len(reclassified_as_answer_pages)} page(s) as answer pages (1-based): {[i + 1 for i in sorted(set(reclassified_as_answer_pages))]}")
    before_dedup = len(questions)
    questions = _dedup_questions(questions)
    if len(questions) != before_dedup:
        log(f"Deduplicated final question list: {before_dedup} -> {len(questions)} question(s)")
    log(f"Final result: {len(qp_page_indices_0based)} question paper page(s), {len(questions)} canonical question(s), {len(admin_page_indices_0based)} admin/cover page(s)")
    return qp_page_indices_0based, questions, admin_page_indices_0based
# =========================================================
# STAGE 2: ANSWER MAPPING
# =========================================================
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
    siblings = [_extract_sub_part_label(q) for j, q in enumerate(questions)
                if j != i and current_num and _extract_leading_number(q) == current_num and _extract_sub_part_label(q)]
    if not siblings:
        return None
    return (
        f"NOTE: this target question is sub-part {current_label} of a larger multi-part question "
        f"(question {current_num}). OTHER sub-parts of this SAME parent question: {', '.join(siblings)}. "
        f"Find sub-part {current_label} SPECIFICALLY -- siblings often share the same overall topic but "
        f"are still separate, distinct responses; do not match content that actually belongs to a "
        f"DIFFERENT sibling just because it discusses a closely related aspect."
    )
SEQUENTIAL_SEARCH_SYSTEM_PROMPT = """You are searching for ONE thing: the line where the response to ONE SPECIFIC question begins, in a line-numbered OCR window from a student's exam booklet.
Given: the target question's text, and a window of line-numbered text (the answer may not be in this window at all -- that's normal).
Rules:
- A response usually starts where the student restates/references the question (e.g. "Ans 5-", "उत्तर 6-", a matching number) OR, with no label, where content clearly starts addressing this question's specific topic.
- Bare short labels ("Q1", "Q.i", "(i)") matching this question's own number/sub-part are sufficient on their own -- no restated text needed.
- Report the EARLIEST line where the answer begins, including any short intro/transition sentence before the topic sentence.
- The same fact/definition can legitimately repeat across multiple answers -- don't reject a genuine match just because similar wording appeared earlier.
- Ignore OCR artifact-description lines ("there is a logo", "signature", "watermark") -- never treat one as start_line.
- If a plausible start exists for this question anywhere in the window -- even a bare label or clear topical match -- report it. Only report found=false if the window is genuinely and completely unrelated to this question's topic.
Return ONLY valid JSON: {"found": true, "start_line": 42} or {"found": false}
start_line must be an exact line number shown in [brackets]."""
def _build_sequential_search_prompt(window_lines: list, question_text: str, ref_label: str, extra_reminder: str = None, context_before: list = None) -> str:
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in window_lines)
    reminder_block = f"{extra_reminder}\n\n" if extra_reminder else ""
    context_block = ""
    if context_before:
        context_lines_block = "\n".join(f"[{idx}] {text}" for idx, text in context_before)
        context_block = f"CONTEXT -- lines BEFORE this window (reference only; NEVER report one of these as start_line):\n{context_lines_block}\n\n"
    return f"{reminder_block}{context_block}TARGET QUESTION ({ref_label}): {question_text}\n\nTEXT WINDOW (line-numbered) -- ONLY lines here may be start_line:\n{lines_block}"
def _parse_sequential_search_response(content: str) -> tuple:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content).strip()
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
        return True, int(data["start_line"])
    except (ValueError, TypeError):
        raise ValueError(f"'start_line' must be an integer, got {data['start_line']!r}")
SEQUENTIAL_SEARCH_WINDOW_CHARS = 16000
SEQUENTIAL_SEARCH_MAX_WINDOWS = 200
SEQUENTIAL_SEARCH_OVERLAP_LINES = 5
_BARE_LABEL_RE = re.compile(r'^\s*(?:Q\.?\s*|प्र\.?\s*|प्रश्न\.?\s*)?\(?([ivxlcdm]+|\d+)\)?\s*[.:\-)]?\s*$', re.IGNORECASE)
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
        elif q_label and token == q_label.strip("()").lower():
            candidates.append(idx)
    return candidates
def _find_answer_start_sequential(client, numbered_lines: list, question_text: str, ref_label: str,
                                    search_from_idx: int, budget: "_TokenBudgetTracker", log,
                                    window_chars: int = SEQUENTIAL_SEARCH_WINDOW_CHARS,
                                    max_windows: int = SEQUENTIAL_SEARCH_MAX_WINDOWS,
                                    extra_reminder: str = None, end_idx: int = None, context_lookback: int = 6):
    total_lines = len(numbered_lines) if end_idx is None else min(len(numbered_lines), end_idx)
    pointer = search_from_idx
    windows_tried = 0
    while pointer < total_lines and windows_tried < max_windows:
        window, chars, idx = [], 0, pointer
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
            label_hint = f"HINT: a bare label matching this question's own number was detected at line(s) {candidates} -- a strong, valid start signal even with no restated question text."
        combined_reminder = "\n\n".join(filter(None, [extra_reminder, label_hint])) or None
        user_prompt = _build_sequential_search_prompt(window, question_text, ref_label, combined_reminder, context_before)
        try:
            found, start_line = _call_groq_with_retries(client, SEQUENTIAL_SEARCH_SYSTEM_PROMPT, user_prompt, _parse_sequential_search_response, budget, log)
        except LLMQuotaExhaustedError:
            raise
        except Exception as e:
            log(f"WARNING: search call failed for {ref_label} (lines {window[0][0]}-{window[-1][0]}): {e}")
            found, start_line = False, None
        if found and start_line is not None:
            valid_ids = {i for i, _ in window}
            if start_line in valid_ids:
                return start_line
            log(f"WARNING: {ref_label} reported start_line {start_line} outside window {window[0][0]}-{window[-1][0]} -- ignoring")
        if idx >= total_lines:
            break
        pointer = max(pointer + 1, idx - SEQUENTIAL_SEARCH_OVERLAP_LINES)
        windows_tried += 1
    return None
# ---- Zero-LLM-cost label pre-pass ----
_ANSWER_START_RE = re.compile(r'^\s*(?:Ans(?:wer)?[.\s:-]+\d*|उत्तर\s*\d*\s*[\-\:]?|प्र[०.\s]*\d*|Q\.?\s*\d+[.\s:-])', re.IGNORECASE)
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
    anchors, last_line = {}, -1
    for qi in sorted(raw_hits.keys()):
        candidates = sorted(l for l in raw_hits[qi] if l > last_line)
        if candidates:
            anchors[qi] = candidates[0]
            last_line = candidates[0]
    if anchors:
        log(f"Label pre-pass (zero-LLM-cost): anchored {len(anchors)}/{len(questions)} question(s) via explicit labels -- no LLM calls needed for these.")
    return anchors
# ---- Batched multi-question search (main strategy: O(chunks) not O(questions x windows)) ----
BATCH_START_FINDER_SYSTEM_PROMPT = """Scan this chunk of a line-numbered OCR transcript (a student exam answer booklet) for where EACH listed target question's answer begins. Most questions will NOT start in this chunk -- only report the ones that genuinely do.
Rules:
- A bare label matching a question's own number (e.g. "Q3", "3)", "(ii)") followed by relevant content is enough on its own.
- Report the EARLIEST line of each match, including a short intro sentence before the topic becomes explicit.
- Similar content can legitimately repeat across different questions on the same broad topic.
- Ignore OCR artifact-description lines -- never report one as a start.
- If a plausible start exists for a question in this chunk, report it rather than omitting it out of caution.
Return ONLY: {"starts": [{"ref": "REF-C", "start_line": 88}, {"ref": "REF-D", "start_line": 140}]}
Omit any question whose answer isn't in this chunk. If none match, return {"starts": []}."""
BATCH_CHUNK_CHARS = int(os.getenv("LLM_BATCH_CHUNK_CHARS", "7000"))
BATCH_CHUNK_OVERLAP_LINES = 3
BATCH_MAX_CONCURRENCY = GROQ_MAX_CONCURRENT_CALLS
BATCH_CHUNK_TARGET_FRACTION = 0.5
def _compute_safe_chunk_chars(budget: "_TokenBudgetTracker", overhead_chars: int, ceiling_chars: int = BATCH_CHUNK_CHARS,
                                target_fraction: float = BATCH_CHUNK_TARGET_FRACTION, min_chunk_chars: int = 1200) -> int:
    target_tokens = budget.safe_limit * target_fraction
    overhead_tokens = _estimate_tokens(" " * overhead_chars) + 800
    remaining_tokens = max(300, target_tokens - overhead_tokens)
    chunk_chars = int(remaining_tokens * CHARS_PER_TOKEN_ESTIMATE)
    return max(min_chunk_chars, min(ceiling_chars, chunk_chars))
def _chunk_numbered_lines_by_chars(numbered_lines: list, chunk_chars: int = BATCH_CHUNK_CHARS, overlap_lines: int = BATCH_CHUNK_OVERLAP_LINES) -> list:
    chunks, i, n = [], 0, len(numbered_lines)
    while i < n:
        chars, j = 0, i
        while j < n and (j == i or chars + len(numbered_lines[j][1]) <= chunk_chars):
            chars += len(numbered_lines[j][1])
            j += 1
        chunks.append(numbered_lines[i:j])
        if j >= n:
            break
        i = max(i + 1, j - overlap_lines)
    return chunks
def _build_group_sibling_note(open_questions: list) -> str:
    groups = {}
    for ref, q in open_questions:
        num = _extract_leading_number(q)
        label = _extract_sub_part_label(q)
        if num and label:
            groups.setdefault(num, []).append((ref, label))
    notes = [f"- Question {num} has sibling sub-parts: {', '.join(f'{ref}={label}' for ref, label in members)}. Each needs its OWN start_line."
              for num, members in groups.items() if len(members) >= 2]
    return "SIBLING SUB-PART GROUPS PRESENT:\n" + "\n".join(notes) if notes else None
def _build_windowed_multi_target_prompt(window_lines: list, open_questions: list, sibling_note: str = None) -> str:
    questions_block = "\n".join(f"[{ref}] {q}" for ref, q in open_questions)
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in window_lines)
    note_block = f"{sibling_note}\n\n" if sibling_note else ""
    return f"{note_block}CANDIDATE QUESTIONS:\n{questions_block}\n\nTEXT WINDOW (line-numbered):\n{lines_block}"
def _parse_windowed_multi_target_response(content: str) -> list:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content).strip()
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
def _detect_sibling_groups(questions: list) -> dict:
    groups, i, n = {}, 0, len(questions)
    while i < n:
        label_i = _extract_sub_part_label(questions[i])
        num_i = _extract_leading_number(questions[i])
        if label_i and num_i:
            j = i + 1
            while j < n and _extract_leading_number(questions[j]) == num_i and _extract_sub_part_label(questions[j]):
                j += 1
            if j - i >= 2:
                groups[i] = list(range(i, j))
            i = j
        else:
            i += 1
    return groups
def _resolve_sibling_group_batch(client, numbered_lines: list, group_questions: list, lower: int, upper: int, budget: "_TokenBudgetTracker", log) -> dict:
    window = [nl for nl in numbered_lines if lower <= nl[0] <= upper]
    if not window:
        return {}
    later_siblings = group_questions[1:]
    if not later_siblings:
        return {}
    user_prompt = _build_windowed_multi_target_prompt(window, later_siblings)
    try:
        starts = _call_groq_with_retries(client, BATCH_START_FINDER_SYSTEM_PROMPT, user_prompt, _parse_windowed_multi_target_response, budget, log)
    except LLMQuotaExhaustedError:
        raise
    except Exception as e:
        log(f"WARNING: sibling-group batch resolution failed for lines {lower}-{upper}: {e}")
        return {}
    valid_ids = {i for i, _ in window}
    valid_refs = {ref for ref, _ in later_siblings}
    found = {}
    for item in starts:
        ref, sl = item["ref"], item["start_line"]
        if ref not in valid_refs or sl not in valid_ids or sl <= lower:
            continue
        found[ref] = min(found[ref], sl) if ref in found else sl
    return found
def _ref_to_question_index(ref: str) -> int:
    return int(ord(ref.split("-")[-1]) - ord("A"))
def _batch_find_all_starts(client, numbered_lines: list, questions: list, already_found: set,
                             budget: "_TokenBudgetTracker", log, max_workers: int = BATCH_MAX_CONCURRENCY) -> dict:
    open_questions = [(f"REF-{chr(65+i)}", q) for i, q in enumerate(questions) if f"REF-{chr(65+i)}" not in already_found]
    if not open_questions:
        return {}
    sibling_note = _build_group_sibling_note(open_questions)
    questions_block_chars = sum(len(ref) + len(q) + 4 for ref, q in open_questions)
    overhead_chars = questions_block_chars + len(sibling_note or "") + len(BATCH_START_FINDER_SYSTEM_PROMPT)
    chunk_chars = _compute_safe_chunk_chars(budget, overhead_chars)
    chunks = _chunk_numbered_lines_by_chars(numbered_lines, chunk_chars=chunk_chars)
    log(f"Batch pass: scanning {len(chunks)} chunk(s) (~{chunk_chars} chars each) for all {len(open_questions)} remaining question(s) (up to {max_workers} in flight at once).")
    def _run_chunk(chunk):
        user_prompt = _build_windowed_multi_target_prompt(chunk, open_questions, sibling_note)
        try:
            return _call_groq_with_retries(client, BATCH_START_FINDER_SYSTEM_PROMPT, user_prompt, _parse_windowed_multi_target_response, budget, log)
        except LLMQuotaExhaustedError:
            raise
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
def _rescue_unmatched_questions(client, numbered_lines: list, questions: list, ranges: list, budget: "_TokenBudgetTracker", log) -> list:
    ref_to_idx = {f"REF-{chr(65+i)}": i for i in range(len(questions))}
    idx_to_ref = {i: f"REF-{chr(65+i)}" for i in range(len(questions))}
    changed, passes = True, 0
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
                if r_idx is not None and r_idx < i and (candidate is None or r_idx > ref_to_idx[candidate["ref"]]):
                    candidate = r
            if candidate is None:
                continue
            lower, upper = candidate["start_line"] + 1, candidate["end_line"]
            if upper <= lower:
                continue
            log(f"  RESCUE: {ref} is unmatched -- searching inside {candidate['ref']}'s range (lines {lower}-{upper})...")
            split_line = _find_answer_start_sequential(client, numbered_lines, q, ref, lower, budget, log, end_idx=upper + 1,
                extra_reminder="REMINDER: this question's answer was not found earlier. A bare label or clear topical shift is enough evidence; do not withhold a plausible match.")
            if split_line is not None and lower < split_line <= upper:
                log(f"  RESCUE: recovered {ref} at line {split_line} (was absorbed into {candidate['ref']})")
                new_end = candidate["end_line"]
                candidate["end_line"] = split_line - 1
                ranges.append({"ref": ref, "start_line": split_line, "end_line": new_end})
                changed = True
    return ranges
def _force_fill_missing_refs(questions: list, ranges: list, total_lines: int, log=print) -> list:
    """GUARANTEE every question ends up with a range in the final output -- nothing
    is ever silently dropped just because search passes couldn't confidently confirm it."""
    have = {r["ref"] for r in ranges}
    missing = [i for i in range(len(questions)) if f"REF-{chr(65+i)}" not in have]
    if not missing:
        return ranges
    for i in missing:
        ref = f"REF-{chr(65+i)}"
        ordered = sorted(ranges, key=lambda r: r["start_line"])
        prev_r = max((r for r in ordered if _ref_to_question_index(r["ref"]) < i), key=lambda r: _ref_to_question_index(r["ref"]), default=None)
        next_r = min((r for r in ordered if _ref_to_question_index(r["ref"]) > i), key=lambda r: _ref_to_question_index(r["ref"]), default=None)
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
        log(f"FORCE-FILL: {ref} was never confidently matched -- assigning fallback range {start}-{end} (worth a manual spot-check) so it is never silently dropped.")
        ranges.append({"ref": ref, "start_line": start, "end_line": end, "low_confidence": True})
    return ranges
# =========================================================
# TEXT CLEANUP (question-restatement stripping, leaked-next-question stripping,
# and decorative-banner/star-symbol stripping)
# =========================================================
QUESTION_PREFIX_RE = re.compile(
    r'^\s*(?:Ans(?:wer)?\s*\d+\s*[.:\-]?\s*|Ans(?:wer)?\s*[.:\-]\s*|उत्तर\s*\d*\s*[\-\:]\s*|'
    r'प्र[०.\s]+\d+[.\s:-]*|प्रश्न[.\s]+\d+[.\s:-]*|Q\.?\s*\d+[.\s:-]*)', re.IGNORECASE)
def strip_question_restatement(answer_text: str) -> str:
    text = answer_text
    for _ in range(2):
        new_text = QUESTION_PREFIX_RE.sub('', text, count=1).strip()
        if new_text == text:
            break
        text = new_text
    return text
_TRAILING_LEADIN_RE = re.compile(
    r'(?:\s*#{1,3}\s*\d*\s*$|\s*(?:Section|Sec\.?)\s*[-:]?\s*[A-Za-z0-9]+\s*$|\s*भाग\s*[-:]?\s*[०-९0-9]*\s*$|'
    r'\s*(?:Q\.?|Ans(?:wer)?\.?|प्र\.?|प्रश्न\.?|उत्तर)\s*[-:.]?\s*\d+\s*[-:.)]?\s*$|\s*\(?[ivxlcdm]{1,5}\)?\s*[-:.)]?\s*$)',
    re.IGNORECASE)
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
        candidate = re.sub(r'(?:\s*\d{1,3}(?:[\+\/]\d{1,4}){0,2}\s*)?(?:Q\.?\s*\d+\s*[.)]|Q\.?\s*[ivxlcdm]+\s*[.)])\s*[>\.\-:]?\s*$', '', candidate, flags=re.IGNORECASE).rstrip()
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
    return re.sub(r'\s+', ' ', text)
_PARENT_INSTRUCTION_PREFIX_RE = re.compile(
    r'^\s*\d+[\.\)]?\s*(?:\([ivx]+\)|\([a-z]\)|\([क-घ]\))?\s*'
    r'(?:identify and explain the following|write (?:short )?notes? on|comment on|explain the following|discuss the following)\s*:?\s*', re.IGNORECASE)
def strip_full_question_echo(answer_text: str, question_text: str) -> str:
    question_core = _PARENT_INSTRUCTION_PREFIX_RE.sub('', question_text).strip() or question_text
    q_norm = _normalize_for_echo_compare(question_core)
    q_word_count = len(q_norm.split())
    if q_word_count == 0:
        return answer_text
    answer_words = answer_text.split()
    if not answer_words:
        return answer_text
    min_n = max(3, int(q_word_count * 0.7))
    max_n = min(len(answer_words), int(q_word_count * 1.3) + 2)
    best_strip_count, best_ratio = 0, 0.0
    for n in range(min_n, max_n + 1):
        prefix_norm = _normalize_for_echo_compare(" ".join(answer_words[:n]))
        ratio = difflib.SequenceMatcher(None, prefix_norm, q_norm).ratio()
        if ratio >= 0.75 and ratio > best_ratio:
            best_ratio, best_strip_count = ratio, n
    if best_strip_count > 0:
        remaining = " ".join(answer_words[best_strip_count:]).strip()
        remaining = re.sub(r'^(?:Answer\s*[-:]\s*)', '', remaining, flags=re.IGNORECASE)
        return remaining.strip()
    return answer_text
# ---- Decorative banner / star-symbol removal ("भाग - 1", "### ★ प्रश्नोत्तर नं: 3 ★", etc.) ----
_DECORATIVE_SYMBOLS = "★☆✦✧✩✪❋❃❖◆●○▪▫➤➔→»«‣·•■□❀❁✵✶✷✸✹✺"
_DECORATIVE_CHARCLASS = r'\s#=＝\-_~' + re.escape(_DECORATIVE_SYMBOLS)
# NOTE: deliberately no \b word-boundary -- it's unreliable right after Devanagari
# text in Python's `re` module and silently fails to match banners like the one above.
_DECORATIVE_BANNER_RE = re.compile(
    r'^[' + _DECORATIVE_CHARCLASS + r']*'
    r'(?:भाग|part|section|sec\.?|प्रश्नोत्तर\s*(?:नं\.?|number|no\.?)?|'
    r'q\s*(?:&|and)?\s*a\s*(?:no\.?|number)?|question\s*(?:&|and)?\s*answer\s*(?:no\.?|number)?)'
    r'[' + _DECORATIVE_CHARCLASS + r':.\-–०-९0-9]*$', re.IGNORECASE | re.UNICODE)
_PURE_DECORATIVE_RE = re.compile(r'^[' + _DECORATIVE_CHARCLASS + r']+$')
def _is_decorative_banner_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return bool(_PURE_DECORATIVE_RE.match(s) or _DECORATIVE_BANNER_RE.match(s))
_DECORATIVE_INLINE_RE = re.compile(
    r'(?:^|\s)[' + _DECORATIVE_CHARCLASS + r']*'
    r'(?:भाग\s*[-–:]?\s*\d*|प्रश्नोत्तर\s*(?:नं\.?|number|no\.?)?\s*[:\-–]?\s*\d*|q\s*(?:&|and)?\s*a\s*(?:no\.?|number)?\s*[:\-–]?\s*\d*)'
    r'[' + _DECORATIVE_CHARCLASS + r']*(?=\s|$)', re.IGNORECASE | re.UNICODE)
_PURE_SYMBOL_RUN_RE = re.compile(r'[' + re.escape(_DECORATIVE_SYMBOLS) + r']+')
_HASH_BANNER_RE = re.compile(r'#{2,}[^\n]{0,40}#{2,}|#{3,}')
def strip_decorative_markers(text: str) -> str:
    """Remove decorative section banners / star symbols / stray markdown hashes
    from anywhere inside a string -- Chandra OCR occasionally emits these and they
    are never part of the student's actual answer content."""
    if not text:
        return text
    text = _DECORATIVE_INLINE_RE.sub(' ', text)
    text = _HASH_BANNER_RE.sub(' ', text)
    text = _PURE_SYMBOL_RUN_RE.sub(' ', text)
    return re.sub(r'\s+', ' ', text).strip()
# ---- Cross-question leak removal: catches ANY other question's text leaking
# into the trailing end of an answer, not just the immediately-adjacent one ----
def _normalize_words_for_leak(text: str) -> list:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    return text.split()
def strip_any_question_leak(answer_text: str, all_questions: list, own_index: int,
                             tail_window_words: int = 25, min_match_words: int = 4) -> str:
    if not answer_text or not all_questions:
        return answer_text
    words = answer_text.split()
    if len(words) < min_match_words:
        return answer_text
    best_cut = None
    for qi, q in enumerate(all_questions):
        if qi == own_index:
            continue
        q_core = re.sub(r'^\s*\d+[\.\)]\s*(?:\([ivxlcdma-z]+\)\s*)?', '', q, flags=re.IGNORECASE).strip()
        q_words = _normalize_words_for_leak(q_core)[:8]
        if len(q_words) < min_match_words:
            continue
        tail_start = max(0, len(words) - tail_window_words - len(q_words))
        joined_tail = " ".join(_normalize_words_for_leak(" ".join(words[tail_start:])))
        joined_q = " ".join(q_words)
        idx = joined_tail.find(joined_q)
        if idx == -1:
            idx = joined_tail.find(" ".join(q_words[:5]))
        if idx != -1:
            words_before_match = joined_tail[:idx].split()
            cut_word_pos = tail_start + len(words_before_match)
            if best_cut is None or cut_word_pos < best_cut:
                best_cut = cut_word_pos
    if best_cut is not None and best_cut < len(words):
        result = " ".join(words[:best_cut]).strip()
        result = re.sub(r'[\s,;:\-–]*\(?\d{1,2}\)?\.?\s*$', '', result).strip()
        return result
    return answer_text
def finalize_qa_pairs(qa_pairs: list, all_questions: list) -> list:
    """Master final-pass cleaner -- run ONCE at the very end of process_pdf, over
    the fully-built qa_pairs list, to strip decorative banners and any residual
    cross-question leaks that survived per-answer cleanup during mapping."""
    for i, pair in enumerate(qa_pairs):
        answer = pair.get("answer", "")
        if not answer:
            continue
        answer = strip_decorative_markers(answer)
        answer = strip_any_question_leak(answer, all_questions, i)
        answer = strip_decorative_markers(answer)
        pair["answer"] = answer
    return qa_pairs
# ---- Noise-line filtering (used when flattening OCR pages into answer_lines) ----
NOISE_RE = re.compile(r'(?:signature|PAGE\s*NO|^\s*DATE\b|^\s*\d{1,3}\s*$)', re.IGNORECASE)
NOISE_LINE_MAX_CHARS = 40
_IMAGE_META_RE = re.compile(
    r'^\s*[\[\(]?\s*(?:there (?:is|are) (?:a |an |some )?(?:logo|stamp|seal|scribble|line|mark|drawing|doodle|sketch|figure|image|photo|watermark)'
    r'|(?:a |an )?(?:logo|stamp|seal|watermark)\s*(?:here|present|visible|seen)?'
    r'|scribbl(?:e|ed|ing)\s*(?:with|in)?\s*(?:a\s+)?(?:red|blue|black|green)\s*(?:pen|ink|marker)'
    r'|(?:red|blue|black|green)\s*(?:pen|ink)\s*(?:mark|line|scribble|underline)s?'
    r'|handwritten\s+(?:note|scribble|mark)s?\s*(?:in|on)?\s*(?:the\s+)?margin'
    r'|image\s*[:\-]\s*.{0,50}|(?:logo|stamp|seal|signature|watermark|figure|diagram)\s*(?:image|icon)?)\s*[\]\)]?\s*$',
    re.IGNORECASE)
def _is_image_description_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    return bool(_IMAGE_META_RE.match(stripped))
def is_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _is_decorative_banner_line(stripped):
        return True
    if re.match(r'^\s*\d{1,3}\s*$', stripped):
        return True
    if _is_image_description_line(stripped):
        return True
    if len(stripped) > NOISE_LINE_MAX_CHARS:
        return False
    return bool(NOISE_RE.search(stripped))
def _sanity_check_answer_pages(answer_lines: list, num_questions: int, log=print) -> bool:
    total_chars = sum(len(l) for l in answer_lines)
    avg = total_chars / max(num_questions, 1)
    if avg < 200:
        log(f"WARNING: 'answer pages' contain only {total_chars} total characters for {num_questions} question(s) (~{avg:.0f} chars/question) -- far too little for real essay answers. The question-paper/answer-page split likely misclassified pages.")
        return False
    return True
def _flag_suspiciously_short_answers(qa_pairs: list, log=print) -> None:
    matched_lengths = sorted(len(p["answer"]) for p in qa_pairs if p.get("matched") and p["answer"].strip())
    if len(matched_lengths) < 2:
        return
    median_len = matched_lengths[len(matched_lengths) // 2]
    if median_len < 50:
        return
    for p in qa_pairs:
        if p.get("matched") and len(p["answer"]) < median_len * 0.25 and len(p["answer"]) < 300:
            log(f"WARNING: possible truncated answer for '{p['question'][:60]}...' -- only {len(p['answer'])} chars vs median {median_len}. Worth spot-checking.")
def verify_no_llm_text_rewriting(qa_pairs: list, answer_lines: list, log=print) -> bool:
    all_ok = True
    for p in qa_pairs:
        if not p.get("matched"):
            continue
        s, e = p["start_line"], p["end_line"]
        expected_raw = " ".join(answer_lines[j] for j in range(s, e + 1) if 0 <= j < len(answer_lines) and answer_lines[j].strip() and not is_noise(answer_lines[j])).strip()
        if expected_raw != p["answer_raw"]:
            all_ok = False
            log(f"CRITICAL: verbatim-integrity check FAILED for '{p['question'][:60]}...' -- answer_raw doesn't match a fresh re-slice of answer_lines[{s}:{e}]. Real bug, please report.")
    if all_ok:
        log("Verbatim-safety check passed: every matched answer's raw text is a byte-for-byte slice of the OCR lines -- the LLM was only ever asked for line numbers, never text content.")
    return all_ok
def map_answers_sequential(answer_lines: list, questions: list, status_callback=None, answer_line_pages: list = None) -> list:
    log = _make_safe_logger(status_callback)
    api_keys = _collect_api_keys()
    if not api_keys:
        raise Exception(
            f"No {LLM_PROVIDER.upper()} API key found. Add "
            f"{'ANTHROPIC_API_KEY' if LLM_PROVIDER == 'anthropic' else 'GROQ_API_KEY'} "
            f"to st.secrets or your environment (or set LLM_PROVIDER=groq / LLM_PROVIDER=anthropic "
            f"explicitly to pick which one this should use)."
        )
    budget = _TokenBudgetTracker()
    client = _RotatingLLMClient(api_keys, budget=budget, log=log)
    numbered_lines = list(enumerate(answer_lines))
    total_lines = len(numbered_lines)
    n = len(questions)
    sibling_groups = _detect_sibling_groups(questions)
    group_member_of = {idx: first_idx for first_idx, members in sibling_groups.items() for idx in members}

    # STAGE A -- free, zero-LLM-cost label pre-pass (regex based).
    label_anchors = _build_label_anchor_index(answer_lines, questions, set(group_member_of.keys()), log)
    found_starts = {f"REF-{chr(65 + qi)}": line_idx for qi, line_idx in label_anchors.items()}

    # STAGE B -- ONE batched pass over the document for every question the label
    # pre-pass missed (O(chunks) calls instead of O(questions x windows)).
    batch_starts = _batch_find_all_starts(client, numbered_lines, questions, set(found_starts.keys()), budget, log)
    found_starts.update(batch_starts)

    # STAGE C -- sibling sub-part groups: resolve internal splits with one bounded
    # call per group if the batch pass didn't already separate them.
    for first_idx, group_indices in sibling_groups.items():
        group_refs = [f"REF-{chr(65 + j)}" for j in group_indices]
        if all(r in found_starts for r in group_refs):
            continue
        group_questions = [(f"REF-{chr(65 + j)}", questions[j]) for j in group_indices]
        first_ref, first_q = group_questions[0]
        anchor_before = max((v for k, v in found_starts.items() if _ref_to_question_index(k) < first_idx), default=-1)
        anchor_after = min((v for k, v in found_starts.items() if _ref_to_question_index(k) > group_indices[-1]), default=total_lines)
        group_start = found_starts.get(first_ref)
        if group_start is None:
            group_start = _find_answer_start_sequential(client, numbered_lines, first_q, first_ref, anchor_before + 1, budget, log, end_idx=anchor_after)
        if group_start is None:
            log(f"NOTE: sibling group {group_refs} not found in this pass -- will be recovered by rescue/force-fill.")
            continue
        found_starts[first_ref] = group_start
        upper = anchor_after - 1 if anchor_after < total_lines else total_lines - 1
        if len(group_questions) > 1:
            sibling_starts = _resolve_sibling_group_batch(client, numbered_lines, group_questions, group_start, upper, budget, log)
            for ref, sl in sibling_starts.items():
                found_starts[ref] = sl
                log(f"  found {ref} (sibling) at line {sl}")
            # Recover any sibling the batch call didn't confidently separate --
            # otherwise its content silently stays folded into the one before it.
            search_floor = group_start
            for gi in group_indices[1:]:
                gref = f"REF-{chr(65 + gi)}"
                if gref in found_starts:
                    search_floor = found_starts[gref]
                    continue
                if search_floor >= upper:
                    continue
                recovered = _find_answer_start_sequential(client, numbered_lines, questions[gi], gref, search_floor + 1, budget, log,
                    end_idx=upper + 1, extra_reminder=_build_sub_part_hint(questions, gi))
                if recovered is not None and search_floor < recovered <= upper:
                    found_starts[gref] = recovered
                    search_floor = recovered
                    log(f"  RECOVERED sibling {gref} at line {recovered} via dedicated retry")

    ordered = sorted(found_starts.items(), key=lambda kv: kv[1])
    ranges = []
    for idx, (ref, start) in enumerate(ordered):
        end = ordered[idx + 1][1] - 1 if idx + 1 < len(ordered) else total_lines - 1
        ranges.append({"ref": ref, "start_line": start, "end_line": end})
    log(f"Label + batch pass found {len(ranges)} of {n} question(s) before rescue/force-fill")

    # STAGE D -- targeted rescue for the (usually very few) stragglers.
    ranges = _rescue_unmatched_questions(client, numbered_lines, questions, ranges, budget, log)
    # STAGE E -- guarantee: whatever's still missing gets a fallback range, never dropped.
    ranges = _force_fill_missing_refs(questions, ranges, total_lines, log)
    log(f"Final: {len(ranges)} of {n} question(s) present in output (force-fill guarantees all {n} appear)")

    ranges_by_ref = {r["ref"]: r for r in ranges}
    results = []
    for i, q in enumerate(questions):
        ref = f"REF-{chr(65 + i)}"
        r = ranges_by_ref.get(ref)
        if r is None:
            results.append({"ref": ref, "question": q, "matched": False, "start_line": None, "end_line": None,
                             "start_page": None, "end_page": None, "answer": "", "answer_raw": "", "low_confidence": False})
            continue
        s, e = r["start_line"], r["end_line"]
        if s >= len(answer_lines):
            s = len(answer_lines) - 1
        if e >= len(answer_lines):
            e = len(answer_lines) - 1
        if s > e:
            s, e = e, s
        verbatim_lines = [answer_lines[j] for j in range(s, e + 1) if 0 <= j < len(answer_lines) and answer_lines[j].strip() and not is_noise(answer_lines[j])]
        answer_raw = " ".join(verbatim_lines).strip()
        answer_clean = strip_question_restatement(answer_raw)
        answer_clean = strip_full_question_echo(answer_clean, q)
        next_q_text = questions[i + 1] if i + 1 < len(questions) else None
        answer_clean = strip_trailing_leaked_next_question(answer_clean, next_q_text)
        answer_clean = strip_trailing_next_question_leadin(answer_clean)
        answer_clean = strip_decorative_markers(answer_clean)
        start_page = answer_line_pages[s] if answer_line_pages and 0 <= s < len(answer_line_pages) else None
        end_page = answer_line_pages[e] if answer_line_pages and 0 <= e < len(answer_line_pages) else None
        results.append({"ref": ref, "question": q, "matched": True, "start_line": s, "end_line": e,
                         "start_page": start_page, "end_page": end_page, "answer": answer_clean, "answer_raw": answer_raw,
                         "low_confidence": bool(r.get("low_confidence", False))})
    return results
# =========================================================
# COMPLETE PIPELINE
# =========================================================
@_diagnose_tuple_errors
def process_pdf(file_input, status_callback=None):
    log = _make_safe_logger(status_callback)
    file_bytes, file_name = _normalize_file_input(file_input, default_name="document.pdf")
    pages = run_ocr(file_bytes, file_name, status_callback)
    log("Building OCR JSON...")
    ocr_json = build_ocr_json(pages)
    log(f"Total pages: {ocr_json['total_pages']}")
    try:
        qp_page_indices, official_questions, admin_page_indices = identify_questions_with_llm(pages, status_callback)
    except LLMQuotaExhaustedError as e:
        raise Exception(f"Stopped during question-paper identification: LLM quota exhausted. {e}\n\nAdd more backup keys, wait for reset, or switch LLM_PROVIDER, then reprocess.") from e
    log(f"Question paper pages detected: {[p+1 for p in qp_page_indices] if qp_page_indices else 'none'}")
    log(f"Admin/cover pages detected: {[p+1 for p in admin_page_indices] if admin_page_indices else 'none'}")
    log(f"Official questions extracted: {len(official_questions)}")
    if not qp_page_indices:
        raise Exception(f"The LLM could not identify any question paper pages in this document.\nPage 1 preview:\n{pages[0]['raw_text'][:500]}")
    if not official_questions:
        raise Exception(
            "Question paper pages were identified, but no questions were extracted from any of them.\n"
            f"Detected pages: {[p+1 for p in qp_page_indices]}\n\n"
            "Check the 'WARNING: cluster ... produced ZERO questions' log lines above for details."
        )
    excluded_indices = set(qp_page_indices) | set(admin_page_indices)
    answer_page_indices = [i for i in range(len(pages)) if i not in excluded_indices]
    answer_pages = [pages[i] for i in answer_page_indices]
    log(f"Answer pages: {[i+1 for i in answer_page_indices]}")
    answer_lines, answer_line_pages = [], []
    for page in answer_pages:
        for line in page["raw_text"].split("\n"):
            if not is_noise(line):
                answer_lines.append(line)
                answer_line_pages.append(page["page_number"])
    all_page_numbers = {p["page_number"] for p in pages}
    answer_page_numbers = {pages[i]["page_number"] for i in answer_page_indices}
    qp_page_numbers = {pages[i]["page_number"] for i in qp_page_indices}
    last_page_num = max(all_page_numbers)
    if last_page_num not in answer_page_numbers:
        log(f"WARNING: the LAST page (page {last_page_num}) was NOT classified as an answer page -- it went to "
            f"{'question-paper' if last_page_num in qp_page_numbers else 'admin'} pages instead. Spot-check page {last_page_num}.")
    log(f"Flattened {len(answer_lines)} answer lines")
    if not _sanity_check_answer_pages(answer_lines, len(official_questions), log):
        raise Exception(
            f"The 'answer pages' identified do not contain enough text to plausibly hold real essay-style "
            f"answers for the {len(official_questions)} question(s) found. The question/answer page split "
            f"likely misclassified pages -- no answer-mapping LLM calls were made, since they'd be guaranteed to fail."
        )
    log("Mapping each question to its answer (label pre-pass + batched search + sibling handling + guarantee)...")
    try:
        qa_pairs = map_answers_sequential(answer_lines, official_questions, status_callback, answer_line_pages=answer_line_pages)
    except LLMQuotaExhaustedError as e:
        raise Exception(f"Stopped partway through answer-mapping: LLM quota exhausted. {e}\n\nThis document was NOT fully mapped -- add more backup keys, wait for reset, or switch LLM_PROVIDER, then reprocess.") from e
    matched_count = sum(1 for p in qa_pairs if p["matched"])
    low_conf_count = sum(1 for p in qa_pairs if p.get("low_confidence"))
    log(f"Matched {matched_count} of {len(official_questions)} questions" + (f" ({low_conf_count} low-confidence best-effort match(es) -- worth spot-checking)" if low_conf_count else ""))
    for p in qa_pairs:
        if not p["matched"]:
            log(f"WARNING: No match found for: {p['question'][:60]}")
    if matched_count == 0:
        raise Exception(f"Could not match any questions to answers.\nOfficial questions: {official_questions}\nFirst 10 answer lines: {answer_lines[:10]}")
    verify_no_llm_text_rewriting(qa_pairs, answer_lines, log)
    _flag_suspiciously_short_answers(qa_pairs, log)
    log("Final cleanup: stripping decorative markers and any cross-question text leaks...")
    qa_pairs = finalize_qa_pairs(qa_pairs, official_questions)
    log(f"Done -- {len(qa_pairs)} Q-A pairs ({matched_count} matched)")
    return ocr_json, qa_pairs
def save_outputs(ocr_json: dict, qa_pairs: list, output_dir: str = ".", base_name: str = "document") -> tuple:
    ocr_path = os.path.join(output_dir, f"{base_name}_ocr.json")
    qa_path = os.path.join(output_dir, f"{base_name}_qa_pairs.json")
    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(ocr_json, f, ensure_ascii=False, indent=2)
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    return ocr_path, qa_path
# =========================================================
# MULTI-PDF WRAPPER
# process_pdf() is intentionally single-document (one PDF = one question paper +
# one student's answers = one independent, isolated unit of work -- keeps
# retries/errors scoped to just that file). To process several uploaded PDFs in
# one go, loop with this wrapper instead of modifying process_pdf itself -- a
# failure on one file is caught and recorded, and does NOT abort the batch.
# =========================================================
def process_multiple_pdfs(file_inputs: list, status_callback=None):
    results = []
    for idx, file_input in enumerate(file_inputs):
        try:
            _, file_name = _normalize_file_input(file_input)
        except Exception:
            file_name = f"file_{idx + 1}"
        def _cb(msg, _name=file_name):
            if status_callback:
                try:
                    status_callback(f"[{_name}] {msg}")
                except Exception:
                    pass
        try:
            ocr_json, qa_pairs = process_pdf(file_input, status_callback=_cb)
            results.append({"file_name": file_name, "ocr_json": ocr_json, "qa_pairs": qa_pairs, "error": None})
        except Exception as e:
            _cb(f"FAILED: {e}")
            results.append({"file_name": file_name, "ocr_json": None, "qa_pairs": None, "error": str(e)})
    return results
