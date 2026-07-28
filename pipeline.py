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
                    f"{func.__name__}(), not before it. file_input: {file_input!r}. Error: {e}"
                ) from e
            raise
    return wrapper

_groq_call_lock = threading.Lock()

# Regex for detecting explicit answer start labels
_ANSWER_START_RE = re.compile(
    r'^\s*(?:Ans(?:wer)?|उत्तर|प्र(?:श्न)?|Q(?:uestion)?)\.?\s*[-:]?\s*\(?(\d+)\)?',
    re.IGNORECASE
)

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
    log(
        f"WARNING: No page-break marker recognized in Datalab output "
        f"(length={len(markdown)} chars). Treating as single page."
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
        raise Exception(f"File is {size_mb:.1f}MB, which exceeds the {MAX_MB}MB upload limit.")
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
    return {
        "total_pages": len(pages),
        "pages": [{"page_number": p["page_number"], "text": p["raw_text"]} for p in pages]
    }

@_diagnose_tuple_errors
def process_reference(file_input, status_callback=None):
    file_bytes, file_name = _normalize_file_input(file_input, default_name="reference.pdf")
    pages = run_ocr_cached(file_bytes, file_name, status_callback)
    return build_ocr_json(pages)

# =========================================================
# LLM & TOKEN BUDGET CONFIG (FIXED FOR GROQ TPM LIMITS)
# =========================================================
GROQ_MODEL = "openai/gpt-oss-120b"
TPM_LIMIT = 30000
TPM_SAFETY_FRACTION = 0.85
CHARS_PER_TOKEN_ESTIMATE = 2.0

# FIX: Reduced from 9000 & 16000 to keep tokens comfortably below Groq's 8000 TPM limit
MAX_CHARS_PER_CHUNK = 6000
SEQUENTIAL_SEARCH_WINDOW_CHARS = 6000
CHUNK_OVERLAP_PAGES = 1
SEQUENTIAL_SEARCH_MAX_WINDOWS = 200
SEQUENTIAL_SEARCH_OVERLAP_LINES = 5

QP_SYSTEM_PROMPT = """You are analyzing OCR text from a scanned student exam booklet -- ANY institution, subject, or language. Pages come in no guaranteed order:
1. ADMIN/COVER pages: roll number, course code, student name, letterhead, blank sheets.
2. QUESTION PAPER pages: official numbered question list.
3. ANSWER pages: student's answers.
Return ONLY valid JSON:
{
  "question_paper_pages": [14, 16],
  "admin_pages": [1, 2],
  "questions": ["1. Example question text. (10)", "2. Another example question. (10)"]
}"""

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
    blocks = [f"--- PAGE {p['page_number']} ---\n{p['raw_text']}" for p in pages]
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
        log(f"Pacing requests: {used:.0f} tokens used in last 60s. Waiting {wait_s:.1f}s...")
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
    r'on\s+tokens\s+per\s+(minute|day)\s*\((TPM\vert{}TPD)\).*?'
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

class GroqQuotaExhaustedError(Exception):
    pass

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
                if self._pool.rotate(reason="invalid key (401)"):
                    continue
                raise
            except (groq.RateLimitError, groq.BadRequestError) as e:
                detail = _parse_rate_limit_detail(str(e))
                if detail and detail["limit_type"] == "TPD":
                    if self._pool.rotate(reason="hit TPD daily quota"):
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

    @property
    def client(self):
        return self._client

    def rotate(self, reason: str = "") -> bool:
        if self._index + 1 >= len(self._keys):
            return False
        self._index += 1
        self._client = self._Groq(api_key=self._keys[self._index])
        if self._budget is not None:
            self._budget.reset_window()
        self._log(f"Switching to Groq key #{self._index + 1} ({reason})")
        return True

def _parse_qp_llm_response(content: str) -> tuple:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()
    data = json.loads(content)
    qp_pages = [int(x) for x in data.get("question_paper_pages", [])]
    admin_pages = [int(x) for x in data.get("admin_pages", [])]
    questions = [str(x).strip() for x in data.get("questions", []) if str(x).strip()]
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
            if i + w > len(s): return None
            chunk = s[i:i+w]
            if chunk.startswith('0') and len(chunk) > 1: return None
            num = int(chunk)
            if num not in valid_page_numbers: return None
            result.append(num)
            i += w
        if i != len(s) or len(set(result)) != len(result): return None
        return result
    candidates = []
    for num_parts in range(2, len(s) + 1):
        for widths in product(range(1, max_digits + 1), repeat=num_parts):
            if sum(widths) != len(s): continue
            res = split_attempt(s, widths)
            if res: candidates.append(res)
    if not candidates: return []
    candidates.sort(key=len)
    return candidates[0]

def _call_groq_with_retries(client, system_prompt: str, user_prompt: str,
                              response_parser, budget: "_TokenBudgetTracker",
                              log, max_retries: int = 4, max_tokens: int = None):
    import groq
    estimated_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt) + 400
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
                raise ValueError("Groq returned empty content.")
            return response_parser(content)
        except (groq.RateLimitError, groq.BadRequestError) as e:
            last_error = e
            detail = _parse_rate_limit_detail(str(e))
            if detail and detail["limit_type"] == "TPD":
                raise GroqQuotaExhaustedError(f"Groq daily quota exhausted: {e}") from e
            if detail:
                budget.record_actual_from_error(detail["used"], detail["limit"])
                time.sleep(detail["wait_seconds"] + 0.5)
                budget.reset_window()
                skip_next_proactive_check = True
            else:
                time.sleep(5.0 * attempt)
        except Exception as e:
            last_error = e
            time.sleep(1)
    raise Exception(f"Groq call failed after retries: {last_error}")

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

def _is_near_duplicate_question(q1: str, q2: str) -> bool:
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

QUESTION_PAPER_ONLY_SYSTEM_PROMPT = """Extract the COMPLETE, clean list of every distinct question/sub-part from these official question paper pages.
Return ONLY valid JSON:
{
  "questions": ["1. Example question text.", "2. Next question text."]
}"""

def _build_canonical_questions_prompt(qp_pages: list) -> str:
    blocks = [f"--- PAGE {p['page_number']} ---\n{p['raw_text']}" for p in qp_pages]
    return "Here is the COMPLETE text of all question paper pages:\n\n" + "\n\n".join(blocks)

def _parse_canonical_questions_response(content: str) -> list:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()
    data = json.loads(content)
    return [str(q).strip() for q in data.get("questions", []) if str(q).strip()]

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
    try:
        questions = _call_groq_with_retries(
            client, QUESTION_PAPER_ONLY_SYSTEM_PROMPT, user_prompt,
            _parse_canonical_questions_response, budget, log, max_tokens=4096
        )
    except Exception as e:
        log(f"WARNING: canonical question extraction failed: {e}")
        return []
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
    chunk_results = []
    valid_page_numbers = {p["page_number"] for p in pages}
    max_page_number = max(valid_page_numbers) if valid_page_numbers else 0
    for i, chunk in enumerate(chunks):
        try:
            qp_pages, questions, admin_pages = _call_groq_for_chunk(client, chunk, budget, log)
            chunk_results.append((qp_pages, [], admin_pages))
        except Exception as e:
            log(f"Chunk {i+1} failed: {e}")
            continue

    qp_page_indices_0based, _, admin_page_indices_0based = _merge_chunk_results(chunk_results)
    qp_pages_full = [pages[i-1] for i in qp_page_indices_0based if i-1 < len(pages)]
    questions = extract_canonical_questions(qp_pages_full, status_callback)
    
    qp_indices = [i-1 for i in qp_page_indices_0based]
    admin_indices = [i-1 for i in admin_page_indices_0based]
    return qp_indices, questions, admin_indices

# =========================================================
# SEQUENTIAL SINGLE-TARGET ANSWER MAPPING
# =========================================================
SEQUENTIAL_SEARCH_SYSTEM_PROMPT = """You are searching for ONE thing: the line where the response to ONE SPECIFIC question begins in a line-numbered OCR window.
Rules:
- Report the EARLIEST line where the answer begins.
- start_line must be an exact line number shown in [brackets].
- If not present in this window, report {"found": false}.
Return ONLY valid JSON:
{"found": true, "start_line": 42} OR {"found": false}"""

def _build_sequential_search_prompt(window_lines: list, question_text: str, ref_label: str,
                                      extra_reminder: str = None, context_before: list = None) -> str:
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in window_lines)
    reminder_block = f"{extra_reminder}\n\n" if extra_reminder else ""
    return (
        f"{reminder_block}"
        f"TARGET QUESTION ({ref_label}): {question_text}\n\n"
        f"TEXT WINDOW (line-numbered):\n{lines_block}"
    )

def _parse_sequential_search_response(content: str) -> tuple:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()
    data = json.loads(content)
    found = bool(data.get("found", False))
    if not found or "start_line" not in data:
        return False, None
    return True, int(data["start_line"])

def _find_answer_start_sequential(client, numbered_lines: list, question_text: str, ref_label: str,
                                    search_from_idx: int, budget: "_TokenBudgetTracker", log,
                                    window_chars: int = SEQUENTIAL_SEARCH_WINDOW_CHARS,
                                    max_windows: int = SEQUENTIAL_SEARCH_MAX_WINDOWS,
                                    extra_reminder: str = None, end_idx: int = None,
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
        user_prompt = _build_sequential_search_prompt(window, question_text, ref_label, extra_reminder)
        try:
            found, start_line = _call_groq_with_retries(
                client, SEQUENTIAL_SEARCH_SYSTEM_PROMPT, user_prompt,
                _parse_sequential_search_response, budget, log
            )
            if found and start_line is not None:
                valid_ids = {i for i, _ in window}
                if start_line in valid_ids:
                    return start_line
        except Exception as e:
            log(f"Search call failed for {ref_label}: {e}")
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
    lbl = _extract_sub_part_label(questions[i])
    num = _extract_leading_number(questions[i])
    if lbl and num:
        return f"Target is sub-part {lbl} of question {num}."
    return None

COMBINED_BOUNDARY_CHECK_SYSTEM_PROMPT = """You are confirming the exact line where an answer starts.
Return ONLY valid JSON:
{"corrected_start_line": 42}"""

def _build_combined_boundary_prompt(window_lines: list, prev_question, curr_question: str, proposed_line: int) -> str:
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in window_lines)
    return f"CURRENT QUESTION: {curr_question}\nPROPOSED LINE: {proposed_line}\n\nWINDOW:\n{lines_block}"

def _parse_boundary_confirm_response(content: str) -> int:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()
    data = json.loads(content)
    return int(data["corrected_start_line"])

def _check_boundary_combined(client, numbered_lines: list, prev_question, curr_question: str,
                               proposed_line: int, search_from_idx: int,
                               budget: "_TokenBudgetTracker", log,
                               back_radius: int = 20, forward_radius: int = 15) -> int:
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
        valid_ids = {idx for idx, _ in window}
        if corrected in valid_ids:
            return corrected
    except Exception:
        pass
    return proposed_line

WINDOWED_MULTI_TARGET_SYSTEM_PROMPT = """Find start lines for sibling sub-parts in this text window.
Return ONLY valid JSON:
{"starts": [{"ref": "REF-B", "start_line": 42}]}"""

def _parse_windowed_multi_target_response(content: str) -> list:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()
    data = json.loads(content)
    return data.get("starts", [])

def _resolve_sibling_group_batch(client, numbered_lines: list, group_questions: list,
                                   lower: int, upper: int, budget: "_TokenBudgetTracker",
                                   log) -> dict:
    window = [nl for nl in numbered_lines if lower <= nl[0] <= upper]
    if not window or len(group_questions) <= 1:
        return {}
    later_siblings = group_questions[1:]
    questions_block = "\n".join(f"[{ref}] {q}" for ref, q in later_siblings)
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in window)
    user_prompt = f"CANDIDATES:\n{questions_block}\n\nWINDOW:\n{lines_block}"
    try:
        starts = _call_groq_with_retries(
            client, WINDOWED_MULTI_TARGET_SYSTEM_PROMPT, user_prompt,
            _parse_windowed_multi_target_response, budget, log
        )
        found = {}
        valid_ids = {i for i, _ in window}
        valid_refs = {ref for ref, _ in later_siblings}
        for item in starts:
            ref, sl = item.get("ref"), item.get("start_line")
            if ref in valid_refs and sl in valid_ids and sl > lower:
                found[ref] = sl
        return found
    except Exception as e:
        log(f"Sibling batch resolution failed: {e}")
        return {}

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

def _build_label_anchor_index(answer_lines: list, questions: list, exclude_indices: set, log=print) -> dict:
    q_number_to_indices = {}
    for qi, q in enumerate(questions):
        if qi in exclude_indices: continue
        qn = _extract_leading_number(q)
        if qn is not None:
            q_number_to_indices.setdefault(qn, []).append(qi)
    if not q_number_to_indices: return {}
    anchors = {}
    last_line = -1
    for i, line in enumerate(answer_lines):
        m = _ANSWER_START_RE.match(line)
        if not m: continue
        num_match = re.search(r'\d+', m.group(0))
        if not num_match: continue
        qis = q_number_to_indices.get(num_match.group(0))
        if qis:
            for qi in qis:
                if qi not in anchors and i > last_line:
                    anchors[qi] = i
                    last_line = i
    return anchors

def _ref_to_question_index(ref: str) -> int:
    return int(ord(ref.split("-")[-1]) - ord("A"))

def _rescue_unmatched_questions(client, numbered_lines: list, questions: list, ranges: list,
                                  budget: "_TokenBudgetTracker", log) -> list:
    matched_refs = {r["ref"] for r in ranges}
    idx_to_ref = {i: f"REF-{chr(65+i)}" for i in range(len(questions))}
    for i, q in enumerate(questions):
        ref = idx_to_ref[i]
        if ref in matched_refs: continue
        # Find neighboring range
        for r in ranges:
            r_idx = _ref_to_question_index(r["ref"])
            if r_idx < i and r["end_line"] > r["start_line"] + 2:
                lower, upper = r["start_line"] + 1, r["end_line"]
                split_line = _find_answer_start_sequential(
                    client, numbered_lines, q, ref, lower, budget, log, end_idx=upper + 1
                )
                if split_line and lower < split_line <= upper:
                    log(f"RESCUE: Recovered {ref} at line {split_line}")
                    old_end = r["end_line"]
                    r["end_line"] = split_line - 1
                    ranges.append({"ref": ref, "start_line": split_line, "end_line": old_end})
                    break
    return ranges

def _reanalyze_and_repair_boundaries(client, numbered_lines: list, questions: list, ranges: list,
                                       budget: "_TokenBudgetTracker", log) -> list:
    if len(ranges) < 2: return ranges
    ordered = sorted(ranges, key=lambda r: r["start_line"])
    for pos in range(1, len(ordered)):
        prev_r, curr_r = ordered[pos - 1], ordered[pos]
        prev_q, curr_q = questions[_ref_to_question_index(prev_r["ref"])], questions[_ref_to_question_index(curr_r["ref"])]
        corrected = _check_boundary_combined(
            client, numbered_lines, prev_q, curr_q, curr_r["start_line"], prev_r["start_line"], budget, log
        )
        if corrected != curr_r["start_line"] and prev_r["start_line"] < corrected <= curr_r["end_line"]:
            prev_r["end_line"] = corrected - 1
            curr_r["start_line"] = corrected
    return ranges

def _remap_incomplete_answers(client, numbered_lines: list, questions: list, ranges: list,
                                budget: "_TokenBudgetTracker", log) -> list:
    return ranges

def _guarantee_full_mapping(client, numbered_lines: list, questions: list, ranges: list,
                              budget: "_TokenBudgetTracker", log) -> list:
    total_lines = len(numbered_lines)
    idx_to_ref = {i: f"REF-{chr(65 + i)}" for i in range(len(questions))}
    still_missing = [i for i in range(len(questions)) if idx_to_ref[i] not in {r["ref"] for r in ranges}]
    for i in still_missing:
        ref = idx_to_ref[i]
        q = questions[i]
        ordered = sorted(ranges, key=lambda r: r["start_line"])
        lower, upper = 0, total_lines - 1
        for r in ordered:
            ridx = _ref_to_question_index(r["ref"])
            if ridx < i: lower = max(lower, r["end_line"] + 1)
            if ridx > i: upper = min(upper, r["start_line"] - 1)
        if lower <= upper:
            ranges.append({"ref": ref, "start_line": lower, "end_line": upper, "low_confidence": True})
            log(f"GUARANTEE: Assigning unmapped {ref} to lines {lower}-{upper}")
    return ranges

def _split_ranges_on_embedded_labels(answer_lines: list, questions: list, ranges: list, log) -> list:
    return ranges

# =========================================================
# MAIN ANSWER MAPPING FUNCTION (FIXED)
# =========================================================
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
    if total_lines == 0 or not questions:
        return []

    sibling_groups = _detect_sibling_groups(questions)
    group_member_of = {idx: first_idx for first_idx, members in sibling_groups.items() for idx in members}
    label_anchors = _build_label_anchor_index(answer_lines, questions, set(group_member_of.keys()), log)

    found_starts = {}
    pointer = 0
    i = 0
    n = len(questions)

    while i < n:
        ref = f"REF-{chr(65 + i)}"
        q = questions[i]

        # 1. Handle Sibling Sub-part Groups
        if i in sibling_groups:
            group_indices = sibling_groups[i]
            group_refs = [f"REF-{chr(65 + j)}" for j in group_indices]
            group_questions = [(f"REF-{chr(65 + j)}", questions[j]) for j in group_indices]
            log(f"Detected sibling sub-part group {group_refs} -- resolving as batch...")
            
            first_ref, first_q = group_questions[0]
            group_start = _find_answer_start_sequential(client, numbered_lines, first_q, first_ref, pointer, budget, log)
            
            if group_start is None:
                log(f"WARNING: Group start for {group_refs} not found. Continuing sequential scan.")
                i = group_indices[-1] + 1
                continue

            group_start = _check_boundary_combined(
                client, numbered_lines, questions[i - 1] if i > 0 else None, first_q,
                group_start, pointer, budget, log
            )
            found_starts[first_ref] = group_start

            next_index = group_indices[-1] + 1
            group_end_bound = None
            if next_index < n:
                next_ref = f"REF-{chr(65 + next_index)}"
                next_q = questions[next_index]
                group_end_bound = _find_answer_start_sequential(
                    client, numbered_lines, next_q, next_ref, group_start + 1, budget, log
                )

            upper = (group_end_bound - 1) if group_end_bound is not None else (total_lines - 1)

            if len(group_questions) > 1:
                sibling_starts = _resolve_sibling_group_batch(
                    client, numbered_lines, group_questions, group_start, upper, budget, log
                )
                for s_ref, sl in sibling_starts.items():
                    found_starts[s_ref] = sl

            # CRITICAL FIX: Never set pointer = total_lines here!
            if next_index < n and group_end_bound is not None:
                found_starts[f"REF-{chr(65 + next_index)}"] = group_end_bound
                pointer = group_end_bound + 1
                i = next_index + 1
            else:
                pointer = group_start + 1
                i = next_index
            continue

        # 2. Handle Standalone Questions
        if ref in found_starts:
            pointer = found_starts[ref] + 1
            i += 1
            continue

        # Try Label Anchor
        anchor_line = label_anchors.get(i)
        if anchor_line is not None and anchor_line >= pointer:
            verified_anchor = _check_boundary_combined(
                client, numbered_lines, questions[i - 1] if i > 0 else None, q,
                anchor_line, pointer, budget, log
            )
            found_starts[ref] = verified_anchor
            pointer = verified_anchor + 1
            i += 1
            continue

        # Sequential LLM Search
        log(f"Searching for start of {ref} from line {pointer} onward...")
        start_line = _find_answer_start_sequential(client, numbered_lines, q, ref, pointer, budget, log)

        if start_line is not None:
            verified_start = _check_boundary_combined(
                client, numbered_lines, questions[i - 1] if i > 0 else None, q,
                start_line, pointer, budget, log
            )
            found_starts[ref] = verified_start
            pointer = verified_start + 1
        else:
            log(f"  Start line for {ref} not found in this window.")

        i += 1

    # 3. Construct Answer Ranges
    sorted_found = sorted(found_starts.items(), key=lambda kv: kv[1])
    ranges = []
    for idx, (r_ref, r_start) in enumerate(sorted_found):
        if idx + 1 < len(sorted_found):
            r_end = sorted_found[idx + 1][1] - 1
        else:
            r_end = total_lines - 1
        if r_end >= r_start:
            ranges.append({"ref": r_ref, "start_line": r_start, "end_line": r_end})

    # 4. Post-processing & Cleanup Passes
    log("Post-processing: Running rescue, boundary repairs, and guarantee passes...")
    ranges = _rescue_unmatched_questions(client, numbered_lines, questions, ranges, budget, log)
    ranges = _reanalyze_and_repair_boundaries(client, numbered_lines, questions, ranges, budget, log)
    ranges = _guarantee_full_mapping(client, numbered_lines, questions, ranges, budget, log)

    ranges.sort(key=lambda r: _ref_to_question_index(r["ref"]))
    log(f"Mapping complete: {len(ranges)}/{len(questions)} question(s) mapped.")
    return ranges
