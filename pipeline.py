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
    MAX_MB = 45
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
            "raw_text": text
        })

    log(f"OCR done -- {len(pages)} page(s) extracted")

    if len(pages) == 1 and size_mb > 1.0:
        log(
            f"WARNING: Only 1 page extracted from a {size_mb:.1f}MB file. "
            f"This usually means the page-break marker format was not recognized. "
            f"Markdown length: {len(markdown)} chars."
        )

    return pages


def build_ocr_json(pages: list) -> dict:
    return {
        "total_pages": len(pages),
        "pages": [
            {"page_number": p["page_number"], "text": p["raw_text"]}
            for p in pages
        ]
    }


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
# QUESTION DETECTION - USING LLM
# =========================================================

GROQ_MODEL = "openai/gpt-oss-120b"

TPM_LIMIT = 8000
TPM_SAFETY_FRACTION = 0.85
CHARS_PER_TOKEN_ESTIMATE = 2.0

MAX_CHARS_PER_CHUNK = 6000
CHUNK_OVERLAP_PAGES = 1

QP_SYSTEM_PROMPT = """You are analyzing OCR text extracted from a scanned student exam assignment booklet.

Your task: read the pages shown and return ONLY valid JSON (no markdown fences, no commentary) in EXACTLY this shape:

{
  "question_paper_pages": [14, 16, 18],
  "questions": ["1. Example question text. (10)", "2. Another example question. (10)"]
}

CRITICAL RULES:
- Extract questions EXACTLY as they appear on the question paper pages.
- If a question has sub-parts like (i), (ii), (iii), extract each as a SEPARATE question.
- Preserve the EXACT original text -- do not paraphrase, do not fix OCR errors.
- Output ONLY the JSON object. No prose before or after."""


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

    if not isinstance(qp_pages, list):
        raise ValueError(f"question_paper_pages must be a list, got: {type(qp_pages).__name__}")
    qp_pages = [int(x) for x in qp_pages]

    if not isinstance(questions, list):
        raise ValueError(f"questions must be a list, got: {type(questions).__name__}")
    questions = [str(x).strip() for x in questions if str(x).strip()]

    return qp_pages, questions


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
        else:
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
    all_questions = []

    for qp_pages, questions in chunk_results:
        all_qp_pages.update(qp_pages)
        all_questions.extend(questions)

    deduped_questions = _dedup_questions(all_questions)
    return sorted(all_qp_pages), deduped_questions


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
            qp_pages_1based, questions = _call_groq_for_chunk(client, chunk, budget, log)
        except Exception as e:
            log(f"WARNING: chunk {i+1}/{len(chunks)} question-identification failed, skipping: {e}")
            chunk_failures.append(str(e))
            continue

        recovered_pages = []
        truly_invalid = []
        for pn in qp_pages_1based:
            if pn in valid_page_numbers:
                recovered_pages.append(pn)
                continue
            split_result = _try_split_concatenated_page_number(
                pn, valid_page_numbers, max_page_number
            )
            if split_result:
                log(f"Recovered concatenated page numbers: {pn} -> {split_result}")
                recovered_pages.extend(split_result)
            else:
                truly_invalid.append(pn)

        if truly_invalid:
            log(f"WARNING: LLM returned out-of-range page numbers, ignoring: {truly_invalid}")

        qp_pages_1based = sorted(set(recovered_pages))

        log(
            f"Chunk {i+1}/{len(chunks)}: identified {len(qp_pages_1based)} question paper "
            f"page(s), {len(questions)} question(s)"
        )
        chunk_results.append((qp_pages_1based, questions))

    if chunk_failures and not chunk_results:
        raise Exception(
            f"All {len(chunks)} chunk(s) failed during question identification. "
            f"First failure: {chunk_failures[0]}"
        )
    elif chunk_failures:
        log(
            f"NOTE: {len(chunk_failures)} of {len(chunks)} chunk(s) failed and were "
            f"skipped -- results below are PARTIAL (based on the {len(chunk_results)} "
            f"chunk(s) that succeeded before the failure)."
        )

    qp_pages_1based_merged, questions = _merge_chunk_results(chunk_results)
    qp_page_indices_0based = sorted(pn - 1 for pn in qp_pages_1based_merged)

    log(
        f"Merged result across all chunks: {len(qp_page_indices_0based)} question paper "
        f"page(s), {len(questions)} question(s)"
    )

    return qp_page_indices_0based, questions


# =========================================================
# NEW APPROACH: DIRECT OCR EXTRACTION - NO LLM FOR ANSWERS
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
    return bool(NOISE_RE.search(line))


def extract_answer_directly(full_ocr_text: str, question: str) -> str:
    """
    Extracts answer directly from OCR text by finding where the question appears
    and taking everything after it until the next question marker.
    NO LLM INVOLVED - PURE PYTHON SLICING.
    """
    # Clean the question for matching
    q_clean = question.strip()
    # Remove numbering
    q_clean = re.sub(r'^[\d\.\(\)ivx]+\s*', '', q_clean)
    q_clean = re.sub(r'\([\d]+\)$', '', q_clean).strip()
    
    # Split the text into lines
    lines = full_ocr_text.split('\n')
    
    # Find where the question appears
    question_line_index = -1
    q_first_30 = q_clean[:30].lower()
    
    for i, line in enumerate(lines):
        if q_first_30 in line.lower() and len(line.strip()) > 20:
            question_line_index = i
            break
    
    if question_line_index == -1:
        # Try with first 50 chars
        q_first_50 = q_clean[:50].lower()
        for i, line in enumerate(lines):
            if q_first_50 in line.lower() and len(line.strip()) > 20:
                question_line_index = i
                break
    
    if question_line_index == -1:
        return ""
    
    # Find where the answer ends (next question marker or section marker)
    answer_end = len(lines)
    for i in range(question_line_index + 1, len(lines)):
        line = lines[i].strip()
        # Check for next question marker
        if re.search(r'^(?:Q\.|Q\s|Question|प्रश्न|Section|--- PAGE \d+ ---|\d+\.\s*[A-Z])', line, re.IGNORECASE):
            answer_end = i
            break
        # Also stop at teacher signature
        if re.search(r'Teacher\'?s?\s*Signature', line, re.IGNORECASE):
            answer_end = i
            break
    
    # Extract answer lines (skip the question line itself)
    answer_lines = []
    for i in range(question_line_index + 1, answer_end):
        line = lines[i].strip()
        if not line:
            continue
        if is_noise(line):
            continue
        # Skip page markers
        if re.search(r'^--- PAGE \d+ ---', line):
            continue
        answer_lines.append(line)
    
    # Join with newlines
    answer_text = "\n".join(answer_lines)
    
    # Remove question restatement from start if present
    if answer_text:
        # Remove "Ans:", "उत्तर:", "Answer:" etc.
        answer_text = re.sub(r'^(?:Ans|उत्तर|Answer)[\s:]*', '', answer_text, flags=re.IGNORECASE)
        # Remove the question text if it appears at the start
        q_start = q_clean[:40].lower()
        if answer_text.lower().startswith(q_start[:30]):
            # Find where the question ends
            for i in range(30, min(len(q_start), len(answer_text))):
                if answer_text[i] in ['.', '?', '!', '\n']:
                    answer_text = answer_text[i+1:].strip()
                    break
    
    return answer_text


def map_answers_directly(pages: list, questions: list, qp_page_indices: list) -> dict:
    """
    Maps questions to answers using DIRECT OCR EXTRACTION.
    NO LLM INVOLVED.
    """
    qa_map = {}
    
    # Get answer pages (all pages except question paper pages)
    answer_page_indices = [i for i in range(len(pages)) if i not in qp_page_indices]
    answer_pages = [pages[i] for i in answer_page_indices]
    
    # Build complete OCR text from answer pages
    full_ocr_text = ""
    for page in answer_pages:
        full_ocr_text += f"\n--- PAGE {page['page_number']} ---\n"
        full_ocr_text += page["raw_text"]
        full_ocr_text += "\n"
    
    print(f"Extracted {len(answer_pages)} answer pages with {len(full_ocr_text)} characters")
    
    # For each question, extract the answer
    for i, q in enumerate(questions):
        answer = extract_answer_directly(full_ocr_text, q)
        
        if answer:
            print(f"✓ Found answer for Q{i+1}: {q[:40]}... ({len(answer)} chars)")
        else:
            print(f"✗ No answer found for Q{i+1}: {q[:40]}...")
        
        qa_map[q] = answer
    
    return qa_map


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

    # Step 1: Extract questions using LLM (ONLY for questions)
    qp_page_indices, official_questions = identify_questions_with_llm(pages, status_callback)

    log(f"Question paper pages detected: {[p+1 for p in qp_page_indices] if qp_page_indices else 'none'}")
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

    # Step 2: Extract answers DIRECTLY from OCR - NO LLM
    log("Extracting answers DIRECTLY from OCR pages (NO LLM involvement)...")
    qa_map = map_answers_directly(pages, official_questions, qp_page_indices)

    matched_count = sum(1 for q in official_questions if q in qa_map and qa_map[q])
    log(f"Matched {matched_count} of {len(official_questions)} questions")

    # Build Q&A pairs with RAW text
    qa_pairs = []
    for q in official_questions:
        answer = qa_map.get(q, "")
        qa_pairs.append({
            "question": q,
            "answer": answer,
            "matched": bool(answer),
        })
        if not answer:
            log(f"WARNING: No answer found for: {q[:60]}...")

    # Log sample of first answer for verification
    for i, pair in enumerate(qa_pairs):
        if pair["answer"]:
            sample = pair["answer"][:500]
            log(f"VERIFICATION - Q{i+1} answer (first 500 chars, total {len(pair['answer'])} chars):\n{sample}\n...")
            break

    log(f"Done -- {len(qa_pairs)} Q&A pair(s) extracted from {len(pages)} page(s).")

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
