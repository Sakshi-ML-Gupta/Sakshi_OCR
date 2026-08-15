import os
import io
import re
import json
import time
import difflib
import hashlib
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
# PROMPT REUSE / RESPONSE CACHE
# =========================================================

_response_cache_lock = threading.Lock()
_RESPONSE_CACHE: dict = {}
_RESPONSE_CACHE_MAX_ENTRIES = 4000  # simple size cap so long-running processes don't grow unbounded


def _prompt_cache_key(system_prompt: str, user_prompt: str) -> str:
    h = hashlib.sha256()
    h.update(system_prompt.encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(user_prompt.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _cache_get(key: str):
    with _response_cache_lock:
        return _RESPONSE_CACHE.get(key)


def _cache_put(key: str, value) -> None:
    with _response_cache_lock:
        if len(_RESPONSE_CACHE) >= _RESPONSE_CACHE_MAX_ENTRIES:
            for k in list(_RESPONSE_CACHE.keys())[: _RESPONSE_CACHE_MAX_ENTRIES // 10]:
                _RESPONSE_CACHE.pop(k, None)
        _RESPONSE_CACHE[key] = value


# =========================================================
# SEMANTIC CHUNKING HELPERS
# =========================================================

_SENTENCE_END_RE = re.compile(r'[.!?।॥][\'"”’]?\s*$')


def _is_semantic_break_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return bool(_SENTENCE_END_RE.search(stripped))


def _nudge_to_semantic_break(lines: list, ideal_idx: int, lookahead: int = 6, lookback: int = 3) -> int:
    n = len(lines)
    if n == 0:
        return ideal_idx

    ideal_idx = max(0, min(ideal_idx, n - 1))

    for offset in range(0, lookahead + 1):
        j = ideal_idx + offset
        if j >= n:
            break
        if _is_semantic_break_line(lines[j]):
            return j

    for offset in range(1, lookback + 1):
        j = ideal_idx - offset
        if j < 0:
            break
        if _is_semantic_break_line(lines[j]):
            return j

    return ideal_idx


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

    pages = run_ocr(file_bytes, file_name, status_callback)
    log(f"Reference OCR complete -- {len(pages)} page(s)")
    return build_ocr_json(pages)


# =========================================================
# LLM-BASED QUESTION PAPER / QUESTION DETECTION (Groq)
# =========================================================

GROQ_MODEL = "openai/gpt-oss-120b"

TPM_LIMIT = 8000
TPM_SAFETY_FRACTION = 0.85
CHARS_PER_TOKEN_ESTIMATE = 2.0

MAX_CHARS_PER_CHUNK = 6000
CHUNK_OVERLAP_PAGES = 1

QP_SYSTEM_PROMPT = """You analyze OCR text from a scanned student exam booklet (any institution/subject/language). Pages fall into 3 types, in no guaranteed order:
1. ADMIN/COVER: roll no., course code, name, letterhead, blank sheets -- no question or answer content.
2. QUESTION PAPER: the official printed numbered questions -- prompts DIRECTED AT the student ("Discuss X", "Explain Y", etc., in whatever language). May show mark allocations like "10".
3. ANSWER: the student's own OCR'd answers -- long, may restate a question briefly then respond, and may contain the student's OWN numbered/bulleted sub-points as part of ONE answer (not separate questions).

You see only a portion (chunk) of the document's pages; some may be carried-over context from a previous chunk -- classify each on its own content.

Return ONLY this JSON (no fences/commentary):
{"question_paper_pages": [14, 16], "admin_pages": [1, 2], "questions": ["1. Example question. (10)"]}
Page-number arrays: each page number is its OWN element, e.g. [14, 16, 18] -- never merge digits into one number. A page can't appear in both lists.

Rules to tell question-paper vs. answer pages:
- A real question is a PROMPT ("explain/discuss/describe/compare", a "?", etc.) asking the student to DO something. A numbered point inside an answer is a STATEMENT/FACT, not an instruction.
- Numbered items following an "answer" label (in any language, e.g. "Ans-", "उत्तर") or after a paragraph of explanatory prose = ANSWER page, even if numbered -- exclude from question_paper_pages.
- Real question papers are short/self-contained per question, not long flowing prose.
- TRAP: students often restate the question as the answer's first sentence before their real response (e.g. "Examine theme X... The theme is central to..."). This looks question-like (prompt verbs) but is really the FIRST page of a long ANSWER. Signals it's really an answer: much more text than a concise instruction needs; prose reads like a developing argument; the same/similar question already appears verbatim on a more concise page you're confident is the real question paper. When unsure, brevity is the deciding signal -- question papers are short, answers (incl. their opening restatement) run long.
- If unsure whether a page is a question-paper page, exclude it.
- Cover/admin pages go in admin_pages (excluded from both question and answer text).
- No question-paper pages in this chunk -> empty list is valid.
- Preserve exact original text/numbering of questions -- no paraphrase, renumber, or translation.
Output ONLY the JSON object, nothing else."""


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
        self._lock = threading.Lock()   # NEW — protects self.events across threads

    def _prune(self, now=None):
        with self._lock:
            now = now if now is not None else time.monotonic()
            while self.events and now - self.events[0][0] >= 60:
                self.events.popleft()

    def used_in_window(self, now=None) -> int:
        self._prune(now)
        with self._lock:
            return sum(tok for _, tok in self.events)

    def record_usage(self, tokens: int):
        with self._lock:
            self.events.append((time.monotonic(), tokens))

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

    def record_actual_limit(self, limit: int, log=None):
        """
        FIX: previously the ONLY way this tracker ever learned the real
        account TPM limit was from a 429 error's message -- but since
        wait_if_needed() paces proactively against the hardcoded
        TPM_LIMIT=8000 guess, a 429 may never actually happen even when
        the real limit is far higher, so the tracker would keep pacing
        against a wrong, overly conservative number FOREVER, causing
        needless ~60s waits before every single call.

        Groq's API returns the real per-minute token limit on every
        successful response via the 'x-ratelimit-limit-tokens' header.
        Reading that on every call (see _call_groq_with_retries) means
        the tracker self-corrects to the account's ACTUAL limit within
        one call, instead of only ever discovering it via a rate-limit
        error that proactive pacing was specifically designed to avoid.
        """
        if not limit or limit == self.tpm_limit:
            return
        if log:
            log(
                f"Learned real Groq TPM limit from response headers: {limit} "
                f"(was assuming {self.tpm_limit}) -- adjusting pacing budget accordingly"
            )
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


def _max_user_prompt_chars(system_prompt: str, budget: "_TokenBudgetTracker",
                             output_reserve_tokens: int = 400, floor_chars: int = 1500,
                             ceiling_chars: int = 20000) -> int:
    system_tokens = _estimate_tokens(system_prompt)
    available_tokens = budget.safe_limit - system_tokens - output_reserve_tokens
    available_chars = int(available_tokens * CHARS_PER_TOKEN_ESTIMATE)
    return max(floor_chars, min(ceiling_chars, available_chars))


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
                              log, max_retries: int = 4, output_reserve_tokens: int = 800,
                              use_cache: bool = True):
    import groq

    # =====================================================================
    # FIX: Groq's response_format={"type": "json_object"} REQUIRES the
    # literal word "json" to appear somewhere in the messages, or it
    # rejects the request with a 400 error -- every single time, on every
    # retry, since retrying doesn't change the prompt. Several of this
    # module's system prompts (QUESTION_PAPER_ONLY_SYSTEM_PROMPT,
    # ANSWER_MAP_SYSTEM_PROMPT, SEQUENTIAL_SEARCH_SYSTEM_PROMPT) describe
    # the required output shape with raw braces/examples but never
    # literally say the word "json", which silently broke every call
    # using them (visible as repeated 400s exhausting all retries).
    #
    # Guarding it HERE, once, in the shared call path -- rather than
    # patching each prompt string individually -- means this can never
    # regress again even if a new system prompt is added later without
    # remembering to include the word.
    # =====================================================================
    if "json" not in system_prompt.lower() and "json" not in user_prompt.lower():
        system_prompt = system_prompt.rstrip() + "\n\nRespond with a single valid JSON object only."

    cache_key = _prompt_cache_key(system_prompt, user_prompt) if use_cache else None
    if cache_key is not None:
        cached = _cache_get(cache_key)
        if cached is not None:
            log("Reusing cached LLM response for an identical prompt (no API call made)")
            return cached

    estimated_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt) + output_reserve_tokens
    last_error = None

    skip_next_proactive_check = False

    for attempt in range(1, max_retries + 2):
        if skip_next_proactive_check:
            skip_next_proactive_check = False
        else:
            budget.wait_if_needed(estimated_tokens, log=log)

        try:
            raw_response = client.chat.completions.with_raw_response.create(
                model=GROQ_MODEL,
                messages=[
                   {"role": "system", "content": system_prompt},
                   {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            response = raw_response.parse()
            real_limit_header = raw_response.headers.get("x-ratelimit-limit-tokens")
            if real_limit_header:
                try:
                     budget.record_actual_limit(int(real_limit_header), log=log)
                except (ValueError, TypeError):
                     pass

            budget.record_usage(estimated_tokens)
            content = response.choices[0].message.content
            parsed = response_parser(content)
            if cache_key is not None:
                _cache_put(cache_key, parsed)
            return parsed

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

from concurrent.futures import ThreadPoolExecutor, as_completed

def _process_chunk(i, chunk):
    page_nums_in_chunk = [p["page_number"] for p in chunk]
    log(f"Asking LLM to analyze chunk {i+1}/{len(chunks)} (pages {page_nums_in_chunk})...")
    try:
        qp_pages_1based, questions, admin_pages_1based = _call_groq_for_chunk(client, chunk, budget, log)
    except Exception as e:
        log(f"WARNING: chunk {i+1}/{len(chunks)} question-identification failed, skipping: {e}")
        return None, str(e)

    qp_pages_1based = _recover_pages(qp_pages_1based, "question-paper")
    admin_pages_1based = _recover_pages(admin_pages_1based, "admin")
    admin_pages_1based = [p for p in admin_pages_1based if p not in qp_pages_1based]
    log(f"Chunk {i+1}/{len(chunks)}: identified {len(qp_pages_1based)} question paper "
        f"page(s), {len(admin_pages_1based)} admin/cover page(s)")
    return (qp_pages_1based, [], admin_pages_1based), None

chunk_results = []
chunk_failures = []
with ThreadPoolExecutor(max_workers=min(6, len(chunks) or 1)) as pool:
    futures = {pool.submit(_process_chunk, i, chunk): i for i, chunk in enumerate(chunks)}
    for fut in as_completed(futures):
        result, err = fut.result()
        if err:
            chunk_failures.append(err)
        else:
            chunk_results.append(result)
            
def _call_groq_for_chunk(client, pages_chunk: list, budget: "_TokenBudgetTracker",
                          log, max_retries: int = 4) -> tuple:
    user_prompt = _build_qp_user_prompt(pages_chunk)
    return _call_groq_with_retries(
        client, QP_SYSTEM_PROMPT, user_prompt, _parse_qp_llm_response,
        budget, log, max_retries, output_reserve_tokens=700
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


_SUBPART_LABEL_RE = re.compile(
    r'\(([ivxlcdm]{1,5}|[a-hA-H]|[क-ह])\)',
    re.IGNORECASE
)


def _extract_subpart_labels(text: str) -> set:
    """
    Returns the set of parenthesized sub-part labels -- (i), (ii), (a),
    (b), (क), (ख), etc. -- found anywhere in `text`, lowercased.

    Used to guard against false-positive deduplication: Stage-2 question
    extraction is instructed to prepend the shared parent instruction to
    EVERY sub-part for self-containment (e.g. "Identify and explain the
    following: (i) ..." / "Identify and explain the following: (ii)
    ..."). That shared prefix can dominate a plain text-similarity
    comparison and make two genuinely DIFFERENT sub-questions look like
    near-duplicates. An explicit, differing sub-part label is a much
    stronger and more direct signal of "these are different questions"
    than overall text similarity, so it should override the similarity
    check rather than be outweighed by it.
    """
    return set(m.group(1).lower() for m in _SUBPART_LABEL_RE.finditer(text))


_TOKEN_SPLIT_RE = re.compile(
    r'[\s।॥,.;:!?()\[\]{}"\'\u2018\u2019\u201c\u201d\-–—]+'
)


def _extract_distinctive_tokens(text: str) -> set:
    """
    Script-agnostic word/token extraction for the near-duplicate
    question detector's overlap safety-check.

    FIX (round 2): the first attempt at this used `re.findall(r'\\w{3,}', ...)`
    -- but Python's `\\w` does NOT match Devanagari combining marks
    (matras like ा/ि/ी, or the virama ्, which are Unicode category
    Mn/Mc). That shattered every multi-syllable Hindi word into useless
    fragments at each matra/virama (e.g. "राष्ट्रभाषा" became
    ['र','ष','ट','रभ','ष']), which made the overlap check compare
    garbage fragments instead of real words and still misfire.

    Splitting on whitespace/punctuation instead -- rather than trying to
    classify individual characters as "word characters" -- keeps each
    Devanagari syllable cluster intact (base consonant + its matras/
    virama stay together as one token), which is what actually makes
    the overlap comparison meaningful for this script.
    """
    tokens = [t for t in _TOKEN_SPLIT_RE.split(text) if t]
    return set(t for t in tokens if len(t) >= 2 and not t.isdigit())


def _is_near_duplicate_question(q1: str, q2: str) -> bool:
    labels1 = _extract_subpart_labels(q1)
    labels2 = _extract_subpart_labels(q2)
    if labels1 and labels2 and labels1 != labels2:
        # Both carry explicit sub-part labels and the labels differ --
        # these are genuinely different sub-questions, no matter how
        # similar the surrounding boilerplate text is. Do not fall
        # through to the similarity check below.
        return False

    k1, k2 = _normalize_question_key(q1), _normalize_question_key(q2)
    if k1 == k2:
        return True

    ratio = difflib.SequenceMatcher(None, k1, k2).ratio()
    if ratio < 0.90:
        return False

    words1 = sorted(_extract_distinctive_tokens(k1))
    words2 = sorted(_extract_distinctive_tokens(k2))
    if not words1 or not words2:
        # No extractable tokens at all (e.g. pure punctuation/symbols) --
        # can't validate overlap, so require a much stricter character
        # match before calling it a duplicate, rather than the old 0.92.
        return ratio >= 0.97

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


QUESTION_PAPER_ONLY_SYSTEM_PROMPT = """You're given the COMPLETE, exact text of a student exam's official question-paper pages (not answers), in order.

Task: extract the COMPLETE, clean list of every distinct question/sub-part, exactly as printed, in printed order.

Rules for multi-part questions:
- A numbered question with LABELED sub-parts (e.g. "1. Identify and explain: (i)...(ii)...(iii)...(iv)...") -> output EACH sub-part as its OWN entry, not merged. Make each entry self-contained: carry forward the parent instruction (e.g. "Identify and explain the following:") or at least the numbering label ("1.(i)", "1.(ii)", ...).
- Applies to ANY labeled sub-structure: (i)/(ii)/(iii), (a)/(b)/(c), (क)/(ख)/(ग), or 1./2./3. used as sub-parts -- always split.
- You see the FULL question paper in one call -- split consistently once, don't guess differently per part.
- Preserve EXACT original text -- no paraphrase, no translation. You may prepend parent numbering for self-containment.
- Keep printed order; sub-parts of one parent stay together in their own order.

Return ONLY this JSON object: {"questions": ["<exact text 1>", "<exact text 2>", ...]}"""


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
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

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

    # =========================================================================
    # FIX: this single-pass extraction had NO deduplication at all -- unlike
    # Stage 1's chunk merge (_merge_chunk_results), which already calls
    # _dedup_questions() on its output. When the LLM occasionally emits the
    # same question twice (most often while applying the multi-part
    # sub-question splitting rules and getting confused about whether a
    # parent question was already emitted as its own entry), the duplicate
    # went straight through unfiltered.
    #
    # This is not just cosmetic: two near-identical question texts directly
    # breaks answer mapping downstream. map_answers_sequential searches for
    # each question's answer by TOPIC/content -- given two near-duplicate
    # questions, the second one's search can find a false "start" somewhere
    # in the MIDDLE of the first (real, single) answer, since the topic
    # looks like a match. That silently splits one genuine answer into two
    # "half" answers, one wrongly attributed to each duplicate REF. Reusing
    # the same near-duplicate detector already used in Stage 1 (character
    # similarity + word overlap, not just exact-string match) here closes
    # that hole at the source, before it can ever reach answer mapping.
    # =========================================================================
    # =========================================================================
    # FIX (v2): the previous fix used _dedup_questions() here -- but that
    # function's lenient threshold (0.90 similarity / 0.92 word overlap)
    # was tuned for Stage 1's cross-chunk merge, where the SAME question
    # can be re-OCR'd slightly differently across overlapping chunk
    # boundaries and needs a forgiving match to be recognized as one.
    #
    # Stage 2 is a different situation: a single clean pass over already-
    # confirmed question-paper text. Here, multi-part questions routinely
    # share a long instructional prefix across sub-parts (e.g. "(i)" and
    # "(ii)" of "Identify and explain the following: X" / "...: Y") --
    # with the lenient threshold, two genuinely DIFFERENT sub-questions
    # that only differ in their final word/phrase could cross the
    # similarity bar and get wrongly merged as "duplicates". That doesn't
    # just lose a question -- it reshuffles map_answers_sequential's
    # entire search sequence, turning previously-correct answers
    # elsewhere in the document into new "half" answers.
    #
    # This dedup instead strips the shared instructional prefix first (so
    # comparison focuses on each sub-part's distinctive content) and uses
    # a near-exact threshold (0.97) so only genuine LLM self-repeats get
    # merged, not similarly-worded-but-distinct sub-questions.
    # =========================================================================
    def _dedup_canonical_questions(qs: list, threshold: float = 0.97) -> list:
        unique = []
        unique_keys = []
        for q in qs:
            core = _PARENT_INSTRUCTION_PREFIX_RE.sub('', q).strip() or q
            key = _normalize_question_key(core)
            is_dup = False
            for existing_key in unique_keys:
                if key == existing_key or difflib.SequenceMatcher(None, key, existing_key).ratio() >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(q)
                unique_keys.append(key)
        return unique

    deduped_questions = _dedup_canonical_questions(questions)
    if len(deduped_questions) < len(questions):
        removed = len(questions) - len(deduped_questions)
        log(
            f"WARNING: Stage-2 extraction returned {len(questions)} question(s), but "
            f"{removed} of them were near-EXACT duplicates of an earlier entry -- "
            f"removed before answer mapping. (Distinct sub-questions that merely "
            f"share an instructional prefix are deliberately NOT merged here.)"
        )
    questions = deduped_questions

    log(f"Canonical question list: {len(questions)} question(s), single consistent pass")
    return questions


def identify_questions_with_llm(pages: list, status_callback=None) -> tuple:
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    from groq import Groq

    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found in secrets or environment")

    client = Groq(api_key=api_key)
    budget = _TokenBudgetTracker()

    dynamic_max_chars = _max_user_prompt_chars(QP_SYSTEM_PROMPT, budget, output_reserve_tokens=700)
    chunks = _chunk_pages_by_char_budget(pages, max_chars=dynamic_max_chars)
    log(f"Split {len(pages)} page(s) into {len(chunks)} LLM chunk(s) "
        f"(~{dynamic_max_chars} chars/chunk, sized to the current TPM budget)")

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
    questions = extract_canonical_questions(qp_pages_full, status_callback)

    log(
        f"Final result: {len(qp_page_indices_0based)} question paper "
        f"page(s), {len(questions)} canonical question(s), "
        f"{len(admin_page_indices_0based)} admin/cover page(s)"
    )

    return qp_page_indices_0based, questions, admin_page_indices_0based


# =========================================================
# LLM-BASED ANSWER MAPPING (Groq)
# =========================================================

ANSWER_MAP_SYSTEM_PROMPT = """You analyze a student's OCR'd answers from an exam booklet. Given:
1. Official questions, each tagged [REF-A], [REF-B], etc.
2. Answer text, each line prefixed by its line number in [brackets].

Task: for EACH question, find where the response starts/ends -- return the inclusive LINE NUMBER RANGE per REF.

Rules:
- An answer typically begins where the student restates/references a question ("Ans 5-", "उत्तर 6-", "प्र. 8", a matching number) or a clear topic shift.
- An answer ends at the last line of its own reasoning, right before the next answer begins.
- If a question's answer isn't present here, omit that REF entirely -- don't invent a range.
- Ranges must not overlap; if unsure of the exact boundary, end the earlier answer sooner rather than letting it swallow later content.
- Use line numbers EXACTLY as shown -- never estimate/renumber. Identify questions by REF label only, never retype question text.
- If told this chunk CONTINUES an answer from a previous chunk, the opening lines likely belong to that REF even without seeing the earlier part.
- CRITICAL: this chunk is short and usually has only 1-3 distinct answers. Scan ALL the way to the last line before responding -- do not stop after the first match. Missing a present answer is a serious error.

Return ONLY this JSON object: {"answers": [{"ref": "REF-A", "start_line": 12, "end_line": 18}]}
No matches here -> {"answers": []}, which is valid."""


SEQUENTIAL_SEARCH_SYSTEM_PROMPT = """Find exactly ONE thing in line-numbered OCR text from a student's exam answer booklet: the line where the response to ONE SPECIFIC question begins.

Given: (1) the target question's exact text, (2) a window of answer text, each line prefixed [line_number]. The window may be a small slice of a larger document -- the answer might not be here at all; that's normal.

Decide: does the response to THIS EXACT question begin in this window?
- It typically starts where the student restates/references the question ("Ans 5-", "उत्तर 6-", "प्र. 8", matching number) OR, with no label, where content clearly starts on this question's distinctive topic.
- CRITICAL: section/heading markers -- decorative symbols (★, #, ##, ऋ, ☆, lines of dashes/asterisks), a "भाग-N" (Part N) label, "प्रश्नोत्तर नं. N" (Q&A no. N), a bolded/standalone title line, or any visually-set-off heading -- are just as strong a boundary signal as "Ans-"/"उत्तर". If such a marker appears immediately before content that matches THIS question's topic, the answer starts right after (or at) that marker, even though it isn't a plain "Ans-" label. Do not let content following a heading marker get treated as a continuation of whatever came before the marker.
- Don't confuse with a DIFFERENT question's answer even if it appears earlier in the window.
- CRITICAL: report the EARLIEST line, including any short intro/transition sentence before the topic becomes explicit -- never a later line just because it's more clearly on-topic. Skipping the true opening line is a serious error.
- CRITICAL: the same fact/definition can legitimately appear in more than one answer, or be restated as a recap. Similar earlier wording does NOT disqualify a later genuine occurrence -- judge each occurrence on its own context.
- If the answer isn't in this window, say so plainly -- don't force a match.

Return ONLY this JSON object: {"found": true, "start_line": 42} or {"found": false}
start_line must be one of the exact [line_number]s shown -- never estimate, always earliest correct line."""


def _build_sequential_search_prompt(window_lines: list, question_text: str, ref_label: str,
                                      extra_reminder: str = None, sibling_questions: list = None) -> str:
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in window_lines)
    reminder_block = f"{extra_reminder}\n\n" if extra_reminder else ""
    siblings_block = ""
    if sibling_questions:
        sib_text = "\n".join(f"  - {s}" for s in sibling_questions)
        siblings_block = (
            f"\nWARNING -- this question shares wording with other, DIFFERENT questions in this "
            f"paper. Do NOT match on the shared boilerplate alone; the answer boundary must be "
            f"specific to THIS question's distinctive content (e.g. its own quoted lines), not "
            f"theirs. Other questions to NOT confuse this with:\n{sib_text}\n"
        )
    return (
        f"{reminder_block}{siblings_block}"
        f"TARGET QUESTION ({ref_label}): {question_text}\n\n"
        f"TEXT WINDOW (line-numbered):\n{lines_block}"
    )

def _strip_trailing_next_question_echo(answer_text: str, other_questions: list,
                                         min_words: int = 8, ratio_threshold: float = 0.75) -> str:
    """If the tail of this answer closely echoes the start of a DIFFERENT question
    (student restating the next prompt/reference before their answer to it begins,
    which the boundary detector missed), cut it off here."""
    words = answer_text.split()
    if len(words) < min_words:
        return answer_text

    best_cut = None
    # Check suffixes of increasing length against the head of every other question
    for n in range(min_words, min(len(words), 60) + 1):
        suffix = " ".join(words[-n:])
        suffix_norm = _normalize_for_echo_compare(suffix)
        for q in other_questions:
            q_core = _PARENT_INSTRUCTION_PREFIX_RE.sub('', q).strip() or q
            q_head = " ".join(q_core.split()[:n])
            q_norm = _normalize_for_echo_compare(q_head)
            if not q_norm:
                continue
            ratio = difflib.SequenceMatcher(None, suffix_norm, q_norm).ratio()
            if ratio >= ratio_threshold:
                best_cut = len(words) - n
                break
        if best_cut is not None:
            break

    if best_cut is not None and best_cut > 0:
        return " ".join(words[:best_cut]).strip()
    return answer_text
                                             
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


SEQUENTIAL_SEARCH_WINDOW_CHARS = 20000   # was 11000 — halves the call count on long booklets
SEQUENTIAL_SEARCH_MAX_WINDOWS = 200

def _find_similar_sibling_questions(question_text: str, all_questions: list, own_index: int,
                                      threshold: float = 0.55) -> list:
    """Other questions sharing enough boilerplate with this one that the search
    LLM could plausibly confuse their answer boundaries (e.g. two sub-parts of
    the same 'reference to context' prompt that differ only in the quoted lines)."""
    siblings = []
    for i, q in enumerate(all_questions):
        if i == own_index:
            continue
        ratio = difflib.SequenceMatcher(None, _normalize_question_key(question_text),
                                         _normalize_question_key(q)).ratio()
        if ratio >= threshold:
            siblings.append(q)
    return siblings

def _find_answer_start_sequential(client, numbered_lines: list, question_text: str, ref_label: str,
                                    search_from_idx: int, budget: "_TokenBudgetTracker", log,
                                    window_chars: int = SEQUENTIAL_SEARCH_WINDOW_CHARS,
                                    max_windows: int = SEQUENTIAL_SEARCH_MAX_WINDOWS,
                                    extra_reminder: str = None):
    total_lines = len(numbered_lines)
    pointer = search_from_idx
    windows_tried = 0

    effective_window_chars = _max_user_prompt_chars(
        SEQUENTIAL_SEARCH_SYSTEM_PROMPT, budget, output_reserve_tokens=60
    )
    effective_window_chars = min(window_chars, effective_window_chars) if window_chars else effective_window_chars

    while pointer < total_lines and windows_tried < max_windows:
        window = []
        chars = 0
        idx = pointer
        while idx < total_lines and (not window or chars + len(numbered_lines[idx][1]) <= effective_window_chars):
            window.append(numbered_lines[idx])
            chars += len(numbered_lines[idx][1])
            idx += 1

        if idx < total_lines and window:
            raw_lines_only = [numbered_lines[k][1] for k in range(pointer, min(idx + 6, total_lines))]
            local_break = _nudge_to_semantic_break(raw_lines_only, idx - pointer - 1, lookahead=6, lookback=2)
            new_idx = pointer + local_break + 1
            if new_idx > idx:
                for k in range(idx, min(new_idx, total_lines)):
                    window.append(numbered_lines[k])
                idx = min(new_idx, total_lines)

        if not window:
            break

        user_prompt = _build_sequential_search_prompt(window, question_text, ref_label, extra_reminder)
        try:
            found, start_line = _call_groq_with_retries(
                client, SEQUENTIAL_SEARCH_SYSTEM_PROMPT, user_prompt,
                _parse_sequential_search_response, budget, log, output_reserve_tokens=60
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

        pointer = idx
        windows_tried += 1

    return None


def _heuristic_find_answer_start(answer_lines: list, question_text: str, search_from_idx: int, log=print):
    """
    Zero-cost (no LLM call) last-resort fallback for when BOTH sequential
    LLM search passes (original + reminder retry) fail to find a
    question's answer start.

    FIX: without this, a question the LLM genuinely missed had no
    recovery path at all -- its answer text was silently swallowed into
    whichever neighboring question's range happened to span those lines
    (visible as "no answer match found" for the missing question, and an
    unrelated extra chunk of text tacked onto a DIFFERENT question's
    answer). This is exactly the failure mode of section/heading-marker
    boundaries (e.g. "#### \u2605 \u092d\u093e\u0917- 3 \u2605 ...") that don't look
    like a plain "Ans-"/"\u0909\u0924\u094d\u0924\u0930" label.

    Reuses the existing keyword-overlap matcher (_line_starts_new_answer_for_question)
    that was already defined in this module for the older chunk-based
    mapper but wasn't wired into the sequential path. Purely local
    string comparison -- no network call, no tokens spent.
    """
    for idx in range(search_from_idx, len(answer_lines)):
        line = answer_lines[idx]
        if not line.strip():
            continue
        matched_idx = _line_starts_new_answer_for_question(line, [question_text], min_fraction=0.5)
        if matched_idx == 0:
            log(f"  heuristic keyword-overlap fallback found a plausible start at line {idx}: {line[:80]!r}")
            return idx
    return None


def map_answers_sequential(answer_lines: list, questions: list, status_callback=None,
                             answer_line_pages: list = None) -> list:
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    from groq import Groq

    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found in secrets or environment")

    client = Groq(api_key=api_key)
    budget = _TokenBudgetTracker()

    numbered_lines = list(enumerate(answer_lines))
    total_lines = len(numbered_lines)

    ref_to_question = {f"REF-{chr(65+i)}": q for i, q in enumerate(questions)}
    found_starts = {}
    pointer = 0

    for i, q in enumerate(questions):
        ref = f"REF-{chr(65 + i)}"
        log(f"Searching for the start of {ref} ({q[:60]}...) from line {pointer} onward...")

        start_line = _find_answer_start_sequential(
            client, numbered_lines, q, ref, pointer, budget, log
        )

        if start_line is None:
            log(f"  first pass found nothing for {ref} -- retrying once with an explicit reminder...")
            retry_reminder = (
                "REMINDER: a previous search pass over this exact text did not find this "
                "question's answer. One common reason for a missed match: the same "
                "definition/explanation legitimately appears more than once in this document "
                "(e.g. two different questions both touch on the same underlying concept, or "
                "the student restates something they already explained elsewhere). Seeing "
                "similar-looking content earlier does NOT mean this occurrence isn't a genuine, "
                "separate answer to THIS target question -- look again with that in mind, and "
                "also double-check you are not missing a short introductory/transitional line "
                "right at the true start of the answer."
            )
            start_line = _find_answer_start_sequential(
                client, numbered_lines, q, ref, pointer, budget, log,
                extra_reminder=retry_reminder
            )
            if start_line is not None:
                log(f"  retry recovered {ref} starting at line {start_line}")

        if start_line is None:
            log(
                f"WARNING: could not find the start of {ref} anywhere from line {pointer} "
                f"to the end of the document ({total_lines} lines) via forward search. "
                f"Will retry with a full-document, order-independent recovery pass after "
                f"all questions have been processed (see below) -- this covers cases "
                f"where the student answered questions out of the question-paper order."
            )
        else:
            found_starts[ref] = start_line
            log(f"  found {ref} starting at line {start_line}")
            pointer = start_line + 1

    # =========================================================================
    # OUT-OF-ORDER RECOVERY PASS
    #
    # The main loop above shares ONE pointer that only ever moves forward,
    # advancing past whatever question was just found. That's correct and
    # efficient when the student answers questions in the same order as
    # the question paper -- but students often don't. If a LATER question
    # (by question-paper order) is physically written BEFORE an EARLIER
    # question in the answer booklet, the shared pointer -- having just
    # found that later question further ahead -- ends up positioned PAST
    # the earlier question's true (earlier) location. A forward-only
    # search can then never find it: it gets reported as unmatched, and
    # its text is silently absorbed into whichever neighboring confirmed
    # range happens to span those lines.
    #
    # Fix: for anything still unmatched after the main pass, search the
    # ENTIRE document again from line 0, completely ignoring the shared
    # pointer / question-paper order. Any start line this finds simply
    # gets added to found_starts -- the final ranges below are always
    # built by SORTING found_starts by physical line number (not by
    # question order), so a recovered out-of-order start automatically
    # carves its answer back out of whatever range had wrongly absorbed
    # it, with no special-casing needed.
    # =========================================================================
    unmatched_refs = [
        f"REF-{chr(65+i)}" for i in range(len(questions))
        if f"REF-{chr(65+i)}" not in found_starts
    ]
    if unmatched_refs:
    log(f"Main forward pass left {len(unmatched_refs)} question(s) unmatched: {unmatched_refs}. "
        f"Running full-document, order-independent recovery pass in parallel...")

    def _recover_one(ref):
        q_idx = ord(ref[-1]) - 65
        q = questions[q_idx]
        start = _find_answer_start_sequential(client, numbered_lines, q, ref, 0, budget, log)
        if start is None:
            start = _heuristic_find_answer_start(answer_lines, q, 0, log)
        return ref, start

    with ThreadPoolExecutor(max_workers=min(6, len(unmatched_refs))) as pool:
        for ref, recovered_start in pool.map(_recover_one, unmatched_refs):
            if recovered_start is None:
                log(f"  could not recover {ref} anywhere in the document -- leaving unmatched")
                continue
            if recovered_start in found_starts.values():
                log(f"  recovery found {ref} at line {recovered_start}, but another question "
                    f"already claims that line -- skipping to avoid a conflicting range")
                continue
            found_starts[ref] = recovered_start
        for ref in unmatched_refs:
            q_idx = ord(ref[-1]) - 65
            q = questions[q_idx]
            log(f"  recovery search for {ref} across the FULL document (from line 0)...")

            recovered_start = _find_answer_start_sequential(
                client, numbered_lines, q, ref, 0, budget, log
            )

            if recovered_start is None:
                log(f"  recovery LLM search also failed for {ref} -- trying zero-cost keyword-overlap fallback...")
                recovered_start = _heuristic_find_answer_start(answer_lines, q, 0, log)

            if recovered_start is None:
                log(f"  could not recover {ref} anywhere in the document -- leaving unmatched")
                continue

            if recovered_start in found_starts.values():
                log(
                    f"  recovery found {ref} at line {recovered_start}, but another "
                    f"question already claims that exact line -- skipping to avoid "
                    f"a conflicting range"
                )
                continue

            log(f"  recovery pass found {ref} starting at line {recovered_start} (out-of-order answer)")
            found_starts[ref] = recovered_start

    ordered = sorted(found_starts.items(), key=lambda kv: kv[1])
    ranges = []
    for idx, (ref, start) in enumerate(ordered):
        end = ordered[idx + 1][1] - 1 if idx + 1 < len(ordered) else total_lines - 1
        ranges.append({"ref": ref, "start_line": start, "end_line": end})

    log(f"Sequential mapping found {len(ranges)} of {len(questions)} question(s)")

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
        other_qs = [oq for j, oq in enumerate(questions) if j != i]
        answer_clean = _strip_trailing_next_question_echo(answer_clean, other_qs)   # NEW

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

    from groq import Groq

    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found in secrets or environment")

    client = Groq(api_key=api_key)
    budget = _TokenBudgetTracker()

    ref_to_question = {f"REF-{chr(65+i)}": q for i, q in enumerate(questions)}

    numbered_lines = list(enumerate(answer_lines))
    dynamic_max_chars = _max_user_prompt_chars(ANSWER_MAP_SYSTEM_PROMPT, budget, output_reserve_tokens=300)
    chunks_with_carry = _chunk_lines_by_char_budget(numbered_lines, questions, max_chars=dynamic_max_chars)
    log(f"Split {len(answer_lines)} answer line(s) into {len(chunks_with_carry)} LLM chunk(s) for answer mapping "
        f"(max {MAX_ANSWERS_PER_CHUNK} distinct answers per chunk, ~{dynamic_max_chars} chars/chunk)")

    all_ranges = []
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
                _parse_answer_map_llm_response, budget, log, output_reserve_tokens=300
            )
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
                    _parse_answer_map_llm_response, budget, log, max_retries=2,
                    output_reserve_tokens=300
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
        answer_text = " ".join(verbatim_lines).strip()
        answer_text = strip_question_restatement(answer_text)
        answer_text = strip_full_question_echo(answer_text, original_question)
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


def is_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r'^\s*\d{1,3}\s*$', stripped):
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
            "Question paper pages were identified, but no questions were extracted.\n"
            f"Detected pages: {[p+1 for p in qp_page_indices]}"
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

    for p in qa_pairs:
        if not p["matched"]:
            continue
        s, e = p["start_line"], p["end_line"]
        expected_raw = " ".join(
            answer_lines[j] for j in range(s, e + 1)
            if 0 <= j < len(answer_lines) and answer_lines[j].strip() and not is_noise(answer_lines[j])
        ).strip()
        if expected_raw != p["answer_raw"]:
            log(
                f"CRITICAL: verbatim-integrity check FAILED for '{p['question'][:60]}...' -- "
                f"the reported answer_raw does not match a fresh re-slice of answer_lines "
                f"[{s}:{e}]. This indicates a real bug in the extraction code, not an LLM "
                f"hallucination -- please report this."
            )

    _flag_suspiciously_short_answers(qa_pairs, log)

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
