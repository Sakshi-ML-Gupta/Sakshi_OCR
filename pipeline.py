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
# LLM-BASED QUESTION PAPER / QUESTION DETECTION (via OpenRouter)
# =========================================================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "z-ai/glm-4.7-flash"

TPM_LIMIT = 30000
TPM_SAFETY_FRACTION = 0.85
CHARS_PER_TOKEN_ESTIMATE = 2.0

MAX_CHARS_PER_CHUNK = 6000
CHUNK_OVERLAP_PAGES = 1

QP_SYSTEM_PROMPT = """You are analyzing OCR text extracted from a scanned student exam/assignment answer booklet. This could be from ANY institution, ANY subject, and ANY language (or a mix of languages/scripts) -- do not assume a specific country, university, or language. The booklet mixes pages of different kinds, in no guaranteed order:

1. ADMINISTRATIVE/COVER pages: roll number/enrolment number, programme/course code, student name, registration details, institution letterhead, blank cover sheets. These contain NO exam question text and NO student answer content -- just identifying/bureaucratic information.
2. QUESTION PAPER pages: the official printed list of numbered exam questions the student must answer. These read as instructions/prompts DIRECTED AT the student (e.g. "Discuss X", "Explain Y with examples", "Write notes on the following:", or the equivalent phrasing in whatever language this document uses). Mark allocations may appear (e.g. "10", "20").
3. ANSWER pages: the student's own (handwritten, OCR'd) answers. These are typically long, restate or reference a question briefly then write an extended response, and may themselves contain numbered or bulleted sub-points as part of the student's OWN explanation. These numbered sub-points inside a long answer are NOT separate exam questions, even though superficially they look similar (number, period, text) -- they are part of the answer to ONE question.

You are being shown only a PORTION of the document's pages at a time (a chunk), not the whole document. Some pages you see may be partial context carried over from a previous chunk -- still classify them normally based on their own content.

Your task: read the pages shown and return ONLY valid JSON (no markdown fences, no commentary, no explanation) in EXACTLY this shape:

{
  "question_paper_pages": [14, 16, 18],
  "admin_pages": [1, 2],
  "questions": ["1. Example question text. (10)", "2. Another example question. (10)"]
}

IMPORTANT formatting requirement: "question_paper_pages" and "admin_pages" must be JSON arrays where EACH page number is a SEPARATE element separated by commas, like [14, 16, 18] -- NEVER merge multiple page numbers into one number like [141618]. Each integer must be a single, individually valid page number from the pages shown. A page must never appear in both lists.

Critical rules for telling question-paper pages apart from answer pages that happen to contain numbered content:
- A genuine question paper question is a PROMPT directed at the student ("explain", "discuss", "describe", "write notes on", "compare", a question mark, etc., in whatever language is used) -- it asks the student to DO something.
- A numbered point inside a long answer is typically a STATEMENT or FACT that is part of an explanation the student is giving -- it does not ask the reader to do anything; it's content, not an instruction.
- If a page's numbered items closely follow a label meaning "answer" (in whatever language/script the document uses -- e.g. "Ans", "Ans-", or its equivalent), or come after a long paragraph of explanatory prose in the same block, that page is almost certainly an ANSWER page, not a question paper page -- exclude it from question_paper_pages even if it has multiple numbered lines.
- A real question paper is usually self-contained and concise per question (a question, maybe a mark allocation) -- not a long flowing essay with numbered sub-points woven into running prose.
- CRITICAL TRAP TO AVOID: students very commonly RESTATE the question itself as the FIRST SENTENCE of their answer, before writing their actual response (e.g. an answer's opening page reads "Examine the theme of concealment in X. Discuss with reference to Y. The theme of concealment is central to..." where everything after the first sentence is the student's OWN original explanation, not more instructions). Such a page can superficially look like a question-paper page because it contains prompt-style verbs ("Examine", "Discuss") -- but it is the FIRST page of a long, multi-page ANSWER, not a question paper page. Signals that this is really an answer's opening page, not a real question paper page: (a) the page has noticeably MORE text than a typical printed question would need, especially if it keeps going well past where a concise instruction would end; (b) the prose quality looks like a developing argument/explanation rather than a terse instruction; (c) the SAME or very similar question text already appears verbatim on a page you are more confident is the genuine, concise question paper (in which case this longer, messier page is almost certainly the student's restatement -- exclude it). When uncertain whether a page is the real question paper or a student's restatement-then-answer, treat brevity and conciseness as the deciding signal: genuine question papers are short per question; answer pages (including their opening restatement) run much longer.
- When genuinely uncertain whether a page is a question paper page, prefer NOT including it as one, and prefer NOT extracting its numbered items as separate questions.
- If a page is a cover/administrative page (roll number, letterhead, blank sheet with no question or answer content), put it in "admin_pages" so it is excluded from BOTH the question paper AND the student's answer text -- it should never be treated as answer content.
- If NONE of the pages shown in this chunk are question paper pages, return an empty list for that field -- that is a valid and expected result for chunks that only contain answer/admin pages.
- Preserve the EXACT original text and numbering of real questions -- do not paraphrase, do not renumber, do not translate.
- Output ONLY the JSON object described above. No prose before or after it. No markdown code fences."""


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
                              log, max_retries: int = 4):
    import openai

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
                    model=LLM_MODEL,
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

        except openai.AuthenticationError as e:
            raise Exception(
                f"OpenRouter API rejected the API key (401 Invalid API Key). "
                f"This will NOT be fixed by retrying. Things to check:\n"
                f"  1. Is OPENROUTER_API_KEY actually set in your environment or "
                f"st.secrets? (A missing key often falls back to None or "
                f"an empty string, which OpenRouter also rejects as invalid.)\n"
                f"  2. Does the key have any extra whitespace, quotes, or "
                f"a line break copied in by accident?\n"
                f"  3. Has the key been revoked or rotated in your OpenRouter "
                f"dashboard (https://openrouter.ai/keys)?\n"
                f"  4. If using st.secrets, did you restart the Streamlit "
                f"app after adding/changing the secret? Streamlit does "
                f"not always hot-reload secrets.toml changes.\n"
                f"Original error: {e}"
            ) from e

        except (openai.RateLimitError, openai.BadRequestError) as e:
            last_error = e
            detail = _parse_rate_limit_detail(str(e))

            if detail and detail["limit_type"] == "TPD":
                raise Exception(
                    f"Daily token quota (TPD) exhausted: "
                    f"{detail['used']}/{detail['limit']} tokens used today, "
                    f"{detail['requested']} more requested. This will reset "
                    f"in approximately {detail['wait_seconds']/60:.0f} minute(s). "
                    f"Retrying within this run will not help -- either wait "
                    f"for the daily reset, or add more credits / upgrade your "
                    f"OpenRouter plan at https://openrouter.ai/settings/credits. "
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
                    f"Waiting {detail['wait_seconds'] + 0.5:.1f}s (provider-reported) "
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

    page_nums = [p["page_number"] for p in pages_chunk]

    def _debug_parser(content):
        log(f"  [debug] raw QP-classification response for chunk pages {page_nums}: {content[:1000]!r}")
        return _parse_qp_llm_response(content)

    return _call_groq_with_retries(
        client, QP_SYSTEM_PROMPT, user_prompt, _debug_parser,
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


QUESTION_PAPER_ONLY_SYSTEM_PROMPT = """You are reading the OFFICIAL question paper pages of a student exam assignment booklet (the printed list of questions, NOT the student's answers). You are given the complete, exact text of these pages, in order.

Your task: extract the COMPLETE, clean list of every distinct question/sub-part, exactly as printed, and return them in printed order.

Critical rules for multi-part questions:
- If a single numbered question contains multiple LABELED sub-parts -- e.g. "1. Identify and explain the following: (i) ... (ii) ... (iii) ... (iv) ..." -- output EACH labeled sub-part as its OWN SEPARATE entry, not merged into one block. Each sub-part entry should include enough of the parent question's context to be self-contained (e.g. carry forward the parent instruction like "Identify and explain the following:" into each sub-part's text, or at minimum keep the original numbering label, e.g. "1.(i)", "1.(ii)", "1.(iii)", "1.(iv)") so each entry is independently understandable without needing to look at a different entry for context.
- This applies to ANY labeled sub-structure: (i)/(ii)/(iii)/(iv), (a)/(b)/(c), (क)/(ख)/(ग), 1./2./3. used as sub-parts within a larger numbered question, etc. -- always split these into separate entries.
- Decide this ONCE, consistently, for the whole document -- you are seeing the COMPLETE question paper text in this single call, so there is no need to guess or produce different splits for different parts of the same question.
- Preserve the EXACT original text of each part -- do not paraphrase, do not translate. You MAY prepend the parent question's numbering/label to each split-out sub-part for self-contained context, as described above.
- Output entries in the SAME ORDER they appear on the question paper (monotonic, matching the printed sequence) -- sub-parts of the same parent question must stay together and in their own (i)/(ii)/(iii)/(iv) order; never reorder anything.
- Do NOT output the same question or sub-part more than once, even if it appears to be printed twice (e.g. once in a table of contents/index and once in the body) -- include each distinct question exactly one time.

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


def extract_canonical_questions(qp_pages: list, status_callback=None) -> list:
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    if not qp_pages:
        return []

    from openai import OpenAI

    api_key = get_api_key("OPENROUTER_API_KEY")
    if not api_key:
        raise Exception("OPENROUTER_API_KEY not found in secrets or environment")

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
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

    # =====================================================================
    # FIX: safety-net dedup. Even though the prompt above now explicitly
    # instructs the model not to repeat a question, and this is a single
    # consistent pass, duplicate/near-duplicate entries can still slip
    # through in practice -- e.g. a question printed both in an index/TOC
    # and in the body, or a sub-part emitted twice. A duplicate canonical
    # question causes the SAME question to be searched for twice
    # downstream, which is a confirmed, reproducible cause of "questions
    # repeating" in the final Q&A output. This reuses the same
    # near-duplicate detection already used elsewhere in this module.
    # =====================================================================
    deduped = _dedup_questions(questions)
    if len(deduped) != len(questions):
        log(
            f"Removed {len(questions) - len(deduped)} duplicate/near-duplicate question(s) "
            f"from the canonical list ({len(questions)} -> {len(deduped)})"
        )
    questions = deduped

    log(f"Canonical question list: {len(questions)} question(s), single consistent pass")
    return questions


def identify_questions_with_llm(pages: list, status_callback=None) -> tuple:
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    from openai import OpenAI

    api_key = get_api_key("OPENROUTER_API_KEY")
    if not api_key:
        raise Exception("OPENROUTER_API_KEY not found in secrets or environment")

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
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
        for p in chunk:
            preview = p["raw_text"][:120].replace("\n", " ")
            log(f"  [debug] page {p['page_number']} preview: {preview!r}")

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

    # Length-outlier reclassification: only when a question-paper page is
    # BOTH an outlier length AND contains an explicit answer-style marker
    # near its start. This is deliberately narrow -- a broader
    # "looks like flowing prose" heuristic was tried and reverted because
    # it reclassified genuine long question-paper pages (e.g. lengthy
    # closing instructions) as answers, leaking exam-paper text into the
    # answer pool.
    if len(qp_page_indices_0based) >= 2:
        qp_page_lengths = [
            (i, len(pages[i]["raw_text"])) for i in qp_page_indices_0based
        ]
        lengths_only = [length for _, length in qp_page_lengths]
        median_length = sorted(lengths_only)[len(lengths_only) // 2]

        length_outliers = [
            page_idx for page_idx, length in qp_page_lengths
            if length > max(median_length * 3, 1500)
        ]

        def _looks_like_student_answer(page_idx: int) -> bool:
            head = pages[page_idx]["raw_text"][:400]
            return bool(_ANSWER_START_RE.search(head))

        confirmed_outliers = [
            page_idx for page_idx in length_outliers if _looks_like_student_answer(page_idx)
        ]
        rejected_outliers = [
            page_idx for page_idx in length_outliers if page_idx not in confirmed_outliers
        ]

        if rejected_outliers:
            for page_idx in rejected_outliers:
                length = dict(qp_page_lengths)[page_idx]
                log(
                    f"NOT reclassifying page {page_idx + 1}: length is an outlier "
                    f"({length} chars vs median {median_length}), but no explicit "
                    f"answer-marker was found -- treating this as genuine (if unusually "
                    f"long) question-paper content, to avoid leaking exam-paper text "
                    f"into a student's answer."
                )

        if confirmed_outliers and len(confirmed_outliers) <= len(qp_page_indices_0based) // 2:
            for page_idx in confirmed_outliers:
                length = dict(qp_page_lengths)[page_idx]
                log(
                    f"RECLASSIFYING page {page_idx + 1}: was detected as a question "
                    f"paper page, is {length} chars long (median for this document's "
                    f"other question paper pages is {median_length}), AND contains an "
                    f"explicit answer-style marker (e.g. 'Ans'/'उत्तर') near its start. "
                    f"Moving it to the answer pages so its content is not lost."
                )
            qp_page_indices_0based = [
                i for i in qp_page_indices_0based if i not in confirmed_outliers
            ]
        elif confirmed_outliers:
            log(
                f"WARNING: {len(confirmed_outliers)} of {len(qp_page_indices_0based)} "
                f"detected question-paper pages are unusually long AND contain an "
                f"explicit answer-style marker. That's too large a fraction to "
                f"auto-reclassify safely -- leaving them as question-paper pages, but "
                f"this may mean the question/answer page split for this document is "
                f"unreliable. Pages flagged: {[p+1 for p in confirmed_outliers]}"
            )

    # Stage 2: single consistent pass over the CONFIRMED question-paper
    # pages' full text, producing one canonical, non-fragmented,
    # deduplicated question list.
    qp_pages_full = [pages[i] for i in qp_page_indices_0based]
    questions = extract_canonical_questions(qp_pages_full, status_callback)

    log(
        f"Final result: {len(qp_page_indices_0based)} question paper "
        f"page(s), {len(questions)} canonical question(s), "
        f"{len(admin_page_indices_0based)} admin/cover page(s)"
    )

    return qp_page_indices_0based, questions, admin_page_indices_0based


# =========================================================
# ANSWER MAPPING (Groq)
#
# DESIGN NOTE -- why a two-stage (start + transition) search is used,
# not a single isolated "does X start here?" question repeated N times:
#
# An isolated yes/no search for one target question, asked in a vacuum
# with no visibility into any OTHER question, has no reference point for
# what "not this answer" looks like. In practice this caused two
# confirmed, reproducible failures:
#   1. Genuine opening lines (a label, a restatement, a short transition
#      sentence) got skipped, because the model couldn't confidently
#      confirm them as "the start" without more forward context.
#   2. The next question's true start got reported too late (or missed),
#      because the model had no signal for what the NEXT topic looks
#      like -- so the CURRENT answer kept absorbing content that had
#      already moved on to a different question ("mixed answers").
#
# The fix is architectural, not just prompt wording: for every boundary
# AFTER the first, the LLM is shown BOTH the question whose answer is
# already open AND the question that comes next, and asked directly
# where the transition between them occurs. This gives it a genuine
# contrast to reason about, instead of an isolated guess. Only the very
# first question's start (which has no "previous answer" to contrast
# against) still uses an isolated single-target search.
#
# FURTHER NOTE -- backward verification (_verify_earliest_start):
# Even the contrastive transition search can still land a few lines (or
# an entire skipped OCR page) LATE, because the forward-sliding window
# search only ever moves forward and never re-checks what came right
# before a reported start. Once a "not found" verdict is issued for a
# window, those lines are effectively gone for that question -- this is
# a confirmed, reproducible cause of answers missing their opening
# paragraph/page. _verify_earliest_start() is a mandatory second pass,
# run every time a start_line is confirmed, that re-examines the OCR
# page the candidate falls on (plus the previous page, so a fully
# skipped page boundary is also covered) and asks explicitly whether an
# earlier line should actually be the true start.
# =========================================================

SEQUENTIAL_SEARCH_SYSTEM_PROMPT = """You are searching for exactly ONE thing in a block of line-numbered OCR text from a student's exam answer booklet: the EARLIEST line where the response to ONE SPECIFIC question begins.

You are given:
1. The exact text of the target question.
2. A window of the student's answer text, with each line prefixed by its line number in [brackets]. This window may be a small slice of a much larger document -- the answer you're looking for might not be in this window at all, and that is a normal, expected outcome.

Decide: does the student's response to THIS EXACT question begin somewhere in the window shown? If yes, find the VERY FIRST line of it.

===========================================================
RULE 1 -- ALWAYS THE EARLIEST LINE, NEVER THE "CLEAREST" ONE
===========================================================
Many student answers do NOT launch straight into an obviously on-topic sentence. Before the part that clearly and unmistakably discusses the topic, an answer very often opens with one or more of the following -- and if present, these opening lines ARE part of the answer and MUST be included as the start:
  - A short label ("Ans 5-", "उत्तर 6", "Q.5", "5)", "Answer:", "(a)", "(b)", "(i)")
  - A one-line restatement or paraphrase of the question itself
  - A brief introductory/transitional sentence that does not yet name the specific topic
  - An incomplete or fragment sentence carried over from a line/page break

If there is ANY doubt between two candidate lines, always choose the EARLIER one.

===========================================================
RULE 2 -- IGNORE OCR ARTIFACT/ANNOTATION DESCRIPTIONS
===========================================================
Some lines are the OCR engine's own description of a visual element on the page (e.g. "[Logo]", "There is a red pen mark here", "Scribbled line", "Stamp", "Signature") rather than actual student writing. These are NEVER the start of an answer. If genuine answer content begins on the line right after such a description, report THAT real content line, not the artifact-description line.

===========================================================
RULE 3 -- REPEATED CONTENT IS NORMAL
===========================================================
The SAME fact or definition can legitimately appear more than once across the document. Seeing similar wording earlier does NOT disqualify a later occurrence from being a genuine, separate answer start for the target question.

===========================================================
RULE 4 -- ERR TOWARD REPORTING A MATCH, NOT "NOT FOUND"
===========================================================
A false "not found" is a WORSE error than a slightly-early guess: if you say "not found" but the answer genuinely starts somewhere in this window, those opening lines get permanently lost from this answer and wrongly attributed to the wrong question -- they cannot be recovered later. Whereas if you report a start line that turns out to be a few lines earlier than ideal, that's a minor, low-cost error. So: if you have even MODERATE confidence (not just high confidence) that the answer begins somewhere in this window, report found=true with your best estimate of the earliest line -- do not withhold a match just because you're not 100% certain.

If the target question's answer does not begin anywhere in this window, say so plainly. It is common and expected for a window to not contain it -- do not force a match.

Return ONLY valid JSON (no markdown fences, no commentary) in exactly one of these two shapes:

{"found": true, "start_line": 42}

or

{"found": false}

If found, start_line MUST be one of the exact line numbers shown in [brackets] in this window -- never estimate or invent a number, and always prefer the earliest correct line over a later one."""


def _build_sequential_search_prompt(window_lines: list, question_text: str, ref_label: str,
                                      extra_reminder: str = None) -> str:
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


SEQUENTIAL_SEARCH_WINDOW_CHARS = 11000  # same safe-per-call char budget used elsewhere in this module
SEQUENTIAL_SEARCH_MAX_WINDOWS = 200  # generous safety cap; a real document will exhaust far sooner
SEARCH_WINDOW_OVERLAP_CHARS = 3000  # re-examine the tail of a "not found" window in the next pass,
                                     # so a genuine boundary that fell right at the edge of a window
                                     # (and was missed once) gets a second, fresh-eyes chance instead
                                     # of being permanently skipped.


def _retreat_pointer(numbered_lines: list, end_idx: int, start_idx: int,
                       overlap_chars: int = SEARCH_WINDOW_OVERLAP_CHARS) -> int:
    """
    Given that a window [start_idx, end_idx) just came back "not found",
    compute where the NEXT window should begin so it overlaps with the
    tail of this one, instead of jumping straight to end_idx. This makes
    a wrongly-missed boundary near the end of a window recoverable on the
    next pass. Always advances by at least one line past start_idx so the
    search cannot get stuck in an infinite loop.
    """
    chars = 0
    idx = end_idx - 1
    while idx > start_idx and chars < overlap_chars:
        chars += len(numbered_lines[idx][1])
        idx -= 1
    return max(idx + 1, start_idx + 1)


def _find_answer_start_sequential(client, numbered_lines: list, question_text: str, ref_label: str,
                                    search_from_idx: int, budget: "_TokenBudgetTracker", log,
                                    window_chars: int = SEQUENTIAL_SEARCH_WINDOW_CHARS,
                                    max_windows: int = SEQUENTIAL_SEARCH_MAX_WINDOWS,
                                    extra_reminder: str = None):
    """
    Isolated single-target search: slides forward through numbered_lines
    in (overlapping, on a "not found" verdict) windows, asking "does this
    ONE question's answer start somewhere in this window?" Used ONLY for
    the very first question (which has no previous answer to contrast
    against) and as a fallback if a transition search (see
    _find_transition_sequential) can't find a boundary.
    """
    total_lines = len(numbered_lines)
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

        pointer = _retreat_pointer(numbered_lines, idx, pointer)
        windows_tried += 1

    return None


TRANSITION_SEARCH_SYSTEM_PROMPT = """You are analyzing a student's exam answer booklet (OCR'd, line-numbered) to find the exact TRANSITION POINT between two consecutive answers.

You are given:
1. CURRENT QUESTION: the question whose answer has ALREADY STARTED before or at the beginning of the window shown.
2. NEXT QUESTION: the question that comes right after it in the question paper. Its answer has NOT started as of the beginning of the window, but MAY start somewhere within the window.
3. A window of the student's answer text, line-numbered.

Your task: find the FIRST line at which the student STOPS writing about the CURRENT question and STARTS writing about the NEXT question. Report that line number as next_start_line -- it is the first line belonging to the NEXT question's answer.

Because you can see BOTH questions, use that contrast directly:

===========================================================
RULE 1 -- USE THE CONTRAST, DON'T GUESS IN ISOLATION
===========================================================
Compare each candidate line against BOTH question texts. A line belongs to the NEXT question if its subject matter matches the NEXT question specifically and is a genuine departure from what the CURRENT question is asking about -- not merely because it contains a number or a new paragraph. A single long answer often contains its own internal numbered or bulleted sub-points as part of ONE continuous explanation for the CURRENT question -- these are NOT the transition; do not report them.

===========================================================
RULE 2 -- REPORT THE EARLIEST PLAUSIBLE TRANSITION LINE
===========================================================
A transition is often marked by an explicit label ("Ans 6-", "उत्तर 7", "Q.7)") -- if present, that exact line is the transition. If there is no such label, the NEXT answer may still open with a short introductory/transitional sentence before it becomes obviously specific to its own topic (a restatement, a generic opening line, a brief lead-in). If such a line reads as the beginning of addressing the NEXT question -- even vaguely -- treat THAT line as next_start_line, not a later line that states the topic more explicitly. Always prefer the earliest plausible line over a later, "clearer" one.

===========================================================
RULE 3 -- REPEATED CONTENT IS NORMAL
===========================================================
The SAME concept or definition can legitimately appear in both the CURRENT and NEXT question's answers (e.g. both questions touch a related idea, or the student recaps something). Do not assume a line belongs to the NEXT question just because it repeats earlier wording -- judge it by whether IT is genuinely answering the NEXT question at that point in the document.

===========================================================
RULE 4 -- IGNORE OCR ARTIFACT/ANNOTATION DESCRIPTIONS
===========================================================
Some lines are the OCR engine's own description of a visual element on the page (e.g. "[Logo]", "There is a red pen mark here", "Scribbled line", "Stamp", "Signature", doodles, underlines) rather than actual student writing. These are never the transition line. If real content resumes right after such a description, evaluate that real content line instead.

===========================================================
RULE 5 -- ERR TOWARD REPORTING A TRANSITION, NOT "NOT FOUND"
===========================================================
A false "not found" is a WORSE error than a slightly-early guess: if you say "not found" but the transition genuinely occurs somewhere in this window, the NEXT question's opening lines get permanently absorbed into the CURRENT question's answer instead -- they cannot be recovered later. Whereas if you report a transition line that turns out to be a little earlier than ideal, that's a minor, low-cost error. So: if you have even MODERATE confidence (not just high confidence) that the transition occurs somewhere in this window, report found=true with your best estimate of the earliest plausible line -- do not withhold a match just because you're not 100% certain.

If the transition does NOT occur anywhere within this window (i.e. the entire window shown is still part of the CURRENT question's answer), say so plainly -- this is common and expected for long answers.

Return ONLY valid JSON (no markdown fences, no commentary) in exactly one of these two shapes:

{"found": true, "next_start_line": 57}

or

{"found": false}

If found, next_start_line MUST be one of the exact line numbers shown in [brackets] in this window."""


def _build_transition_search_prompt(window_lines: list, current_q_text: str, next_q_text: str,
                                      extra_reminder: str = None) -> str:
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in window_lines)
    reminder_block = f"{extra_reminder}\n\n" if extra_reminder else ""
    return (
        f"{reminder_block}"
        f"CURRENT QUESTION: {current_q_text}\n\n"
        f"NEXT QUESTION: {next_q_text}\n\n"
        f"TEXT WINDOW (line-numbered):\n{lines_block}"
    )


def _parse_transition_search_response(content: str) -> tuple:
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

    if "next_start_line" not in data:
        raise ValueError("Response has found=true but is missing 'next_start_line'")
    try:
        next_start_line = int(data["next_start_line"])
    except (ValueError, TypeError):
        raise ValueError(f"'next_start_line' must be an integer, got {data['next_start_line']!r}")

    return True, next_start_line


def _find_transition_sequential(client, numbered_lines: list, current_q_text: str, next_q_text: str,
                                  search_from_idx: int, budget: "_TokenBudgetTracker", log,
                                  window_chars: int = SEQUENTIAL_SEARCH_WINDOW_CHARS,
                                  max_windows: int = SEQUENTIAL_SEARCH_MAX_WINDOWS,
                                  extra_reminder: str = None):
    """
    Comparative search: slides forward through numbered_lines, showing
    the LLM both the CURRENT (already-open) question and the NEXT
    question together, and asking where the transition between them
    occurs. This gives the model a genuine contrast to reason about,
    instead of an isolated "does X start here?" guess -- see the module
    docstring above SEQUENTIAL_SEARCH_SYSTEM_PROMPT for the full
    rationale. On a "not found" verdict, the next window overlaps with
    the tail of this one (see _retreat_pointer) so a boundary that fell
    right at the edge isn't permanently lost.
    """
    total_lines = len(numbered_lines)
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

        user_prompt = _build_transition_search_prompt(window, current_q_text, next_q_text, extra_reminder)
        try:
            found, next_start = _call_groq_with_retries(
                client, TRANSITION_SEARCH_SYSTEM_PROMPT, user_prompt,
                _parse_transition_search_response, budget, log
            )
        except Exception as e:
            log(f"WARNING: transition search call failed (lines {window[0][0]}-{window[-1][0]}): {e}")
            found, next_start = False, None

        if found and next_start is not None:
            valid_ids = {i for i, _ in window}
            if next_start in valid_ids:
                return next_start
            log(
                f"WARNING: transition search reported next_start_line {next_start}, outside "
                f"this window's actual range {window[0][0]}-{window[-1][0]} -- ignoring and "
                f"treating this window as a non-match"
            )

        pointer = _retreat_pointer(numbered_lines, idx, pointer)
        windows_tried += 1

    return None


# =========================================================
# BACKWARD VERIFICATION -- catches starts detected a few lines
# (or a whole skipped OCR page) too LATE. See module note above
# SEQUENTIAL_SEARCH_SYSTEM_PROMPT for why this is necessary even
# with the overlap fix in the forward-sliding search above.
# =========================================================

VERIFY_EARLIEST_START_SYSTEM_PROMPT = """You already found a CANDIDATE start line for a student's answer to a specific question. Your job now is ONLY to double-check: is there an EARLIER line, within the block shown, that should actually be the true start instead?

This check exists because answers commonly begin with a label, a one-line restatement of the question, or a short transitional sentence -- and these earlier lines are sometimes missed on a first pass, especially when they fall right at an OCR page boundary (the block shown may span the END of the previous page and the START of the current page).

You are given:
1. The target question's exact text.
2. The CANDIDATE start line number that was already found.
3. A block of line-numbered text that ends at or after the candidate line, and begins earlier (potentially a full previous OCR page back) so you can check for missed earlier content.

Look at every line BEFORE the candidate line in this block. Does the answer to THIS question genuinely begin earlier than the candidate? Only report an earlier line if it is clearly part of THIS answer (a label, restatement, or transition into this specific topic) -- not if it's still part of a different, previous answer, or noise/artifact text.

Return ONLY valid JSON (no markdown fences, no commentary):

{"earlier_start_found": true, "start_line": 118}

or

{"earlier_start_found": false}

If unsure, prefer {"earlier_start_found": false} -- this is a safety-net check, not a re-search from scratch."""


def _build_verify_earliest_prompt(block_lines: list, candidate_line: int, question_text: str, ref_label: str) -> str:
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in block_lines)
    return (
        f"TARGET QUESTION ({ref_label}): {question_text}\n\n"
        f"CANDIDATE START LINE: {candidate_line}\n\n"
        f"TEXT BLOCK (line-numbered):\n{lines_block}"
    )


def _parse_verify_earliest_response(content: str) -> tuple:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()
    data = json.loads(content)
    if not isinstance(data, dict) or "earlier_start_found" not in data:
        raise ValueError(f"Response missing 'earlier_start_found': {data!r}")
    if not data["earlier_start_found"]:
        return False, None
    if "start_line" not in data:
        raise ValueError("earlier_start_found=true but missing 'start_line'")
    return True, int(data["start_line"])


VERIFY_EARLIEST_BACK_PAGES = 2  # how many full OCR pages back to include for context

# Deterministic safety net: an explicit numbered label ("Ans 5-", "Q.5",
# "उत्तर 6", "5)") is an unambiguous, machine-checkable signal of where an
# answer truly begins -- it does not depend on an LLM's confidence, and
# is used to override/complement the LLM backward-check below whenever
# one is found earlier than the candidate start.
_EXPLICIT_ANSWER_LABEL_RE = re.compile(
    r'^\s*(?:'
    r'Ans(?:wer)?\s*\.?\s*\d+\s*[.\):\-]?'
    r'|उत्तर\s*\d+\s*[\-\:]?'
    r'|प्र[०.\s]*\d+[.\s:-]*'
    r'|Q\.?\s*\d+\s*[.\):\-]?'
    r'|\([a-z]\)'
    r'|\([ivxlcdm]+\)'
    r'|\([क-घ]\)'
    r')',
    re.IGNORECASE
)


def _find_deterministic_earlier_label(block: list, candidate_line: int) -> int:
    """
    Scans a backward-verification block (ascending order) for the
    EARLIEST line matching an explicit numbered answer label, strictly
    before candidate_line. Returns that line's index, or None.
    """
    for idx, text in block:
        if idx >= candidate_line:
            break
        if _EXPLICIT_ANSWER_LABEL_RE.match(text.strip()):
            return idx
    return None


def _verify_earliest_start(client, numbered_lines: list, answer_line_pages: list,
                             start_line: int, question_text: str, ref_label: str,
                             min_allowed_line: int, budget: "_TokenBudgetTracker", log) -> int:
    """
    Page-boundary-aware backward check: re-examines the OCR page the
    candidate start falls on, PLUS the previous VERIFY_EARLIEST_BACK_PAGES
    full page(s), to catch cases where the true start was a few lines (or
    a whole skipped page) earlier than what the forward search reported.
    min_allowed_line prevents this from ever moving the start earlier
    than the previous confirmed answer's boundary (so answers can never
    be made to overlap by this check).

    Combines TWO independent signals and takes whichever is earlier:
    1. A deterministic regex scan for an explicit numbered label (see
       _EXPLICIT_ANSWER_LABEL_RE) -- immune to LLM under-confidence.
    2. The LLM backward-check below, for restatement/transition openings
       that have no explicit label.
    """
    if start_line <= min_allowed_line or start_line >= len(numbered_lines):
        return start_line

    if answer_line_pages and start_line < len(answer_line_pages):
        seen_pages = []
        for p in reversed(answer_line_pages[:start_line + 1]):
            if p not in seen_pages:
                seen_pages.append(p)
            if len(seen_pages) > VERIFY_EARLIEST_BACK_PAGES:
                break
        target_pages = set(seen_pages)
        block_start = start_line
        for i in range(start_line, min_allowed_line, -1):
            if i < len(answer_line_pages) and answer_line_pages[i] in target_pages:
                block_start = i
            else:
                break
    else:
        block_start = max(min_allowed_line + 1, start_line - 40)

    block = [numbered_lines[i] for i in range(block_start, start_line + 1)]
    if len(block) <= 1:
        return start_line

    # Signal 1: deterministic explicit-label scan (cheap, no LLM call).
    deterministic_earlier = _find_deterministic_earlier_label(block, start_line)

    # Signal 2: LLM backward check, for restatement/transition openings
    # that carry no explicit numbered label.
    prompt = _build_verify_earliest_prompt(block, start_line, question_text, ref_label)
    llm_earlier = None
    try:
        found, earlier_line = _call_groq_with_retries(
            client, VERIFY_EARLIEST_START_SYSTEM_PROMPT, prompt,
            _parse_verify_earliest_response, budget, log, max_retries=2
        )
        if found and earlier_line is not None:
            valid_ids = {i for i, _ in block}
            if earlier_line in valid_ids and min_allowed_line < earlier_line <= start_line:
                llm_earlier = earlier_line
    except Exception as e:
        log(f"WARNING: earliest-start verification failed for {ref_label}: {e}")

    candidates = [c for c in (deterministic_earlier, llm_earlier)
                  if c is not None and min_allowed_line < c <= start_line]
    if candidates:
        final = min(candidates)
        if final != start_line:
            source = "explicit label" if final == deterministic_earlier else "LLM check"
            log(f"  earliest-start check: moved {ref_label} start from {start_line} back to {final} ({source})")
        return final

    return start_line


# Deterministic cap for the last answer's end: a markdown/plain-text
# section heading that signals the OVERALL ASSIGNMENT's own closing
# material (a document-level "Conclusion"/"Bibliography"/"References"
# section printed after ALL answers) can never legitimately be part of
# one specific answer -- regardless of what an LLM tail-check concludes.
# This only matches clear HEADING-style lines (a markdown '#' prefix, or
# a short standalone line that IS just the heading word), so a sentence
# that merely uses the word "conclusion" mid-paragraph as part of the
# student's own answer is never falsely matched and stays included.
_OVERALL_CLOSING_SECTION_RE = re.compile(
    r'^\s*#{1,6}\s*(?:conclusion|summary|bibliography|references?|acknowledge?ments?)\b'
    r'|^\s*(?:conclusion|bibliography|references?|acknowledge?ments?)\s*[:\-]?\s*$',
    re.IGNORECASE
)

# OCR frequently fuses ANY structural heading from the printed question
# paper/document (not just an overall closing section -- e.g. a section
# divider like "### SECTION-C") onto the tail of the preceding answer's
# last OCR line, with no newline in between. A heading is NEVER genuine
# answer content regardless of which section it marks, so it is always
# stripped out of the line wherever it's found. Requiring 2+ consecutive
# '#' characters keeps this safe from single stray '#' misreads.
_INLINE_HEADING_MARKER_RE = re.compile(r'#{2,6}\s*')

_CLOSING_HEADING_WORD_RE = re.compile(
    r'^(?:conclusion|summary|bibliography|references?|acknowledge?ments?)\b',
    re.IGNORECASE
)


def _truncate_before_overall_closing_heading(line: str):
    """
    Cuts a line at the first markdown heading marker ('##' through
    '######') found ANYWHERE within it, dropping the heading and
    everything after it (within this ONE raw line). Returns
    (kept_text, is_overall_closing) -- is_overall_closing is True ONLY
    when the heading text itself is a document-level closing section
    (conclusion/summary/bibliography/references/acknowledgements). In
    that case the caller should ALSO stop collecting any further lines
    at all, since such a section marks the true end of the document. Any
    OTHER heading (e.g. "### SECTION-C", a mid-document section divider)
    is still stripped from the line, but does NOT stop collection --
    there may be more answers after it.
    """
    m = _INLINE_HEADING_MARKER_RE.search(line)
    if not m:
        return line, False
    heading_text = line[m.end():].strip()
    is_closing = bool(_CLOSING_HEADING_WORD_RE.match(heading_text))
    return line[:m.start()].strip(), is_closing


def _find_overall_closing_heading(numbered_lines: list, start_idx: int):
    """
    Returns the index of the first OVERALL closing-section heading line
    at or after start_idx, or None if none exists. Used to hard-cap the
    chronologically last answer so a whole-assignment wrap-up section
    can never be absorbed into it, independent of the LLM tail-check.
    """
    for idx, text in numbered_lines[start_idx:]:
        stripped = text.strip()
        if stripped and _OVERALL_CLOSING_SECTION_RE.match(stripped):
            return idx
    return None


LAST_ANSWER_END_SYSTEM_PROMPT = """You are looking at the FINAL portion of a student's exam answer booklet (OCR'd, line-numbered), starting from where their LAST answer begins. This tail section may contain:
1. The remainder of the student's genuine answer content for the target question -- this should be INCLUDED.
2. AFTER the student's real answer content ends, there may be trailing material that is NOT part of the answer itself: e.g. an overall assignment/exam-level closing remark or conclusion (not specific to this one question), an institutional footer, "thank you" notes, or similar wrap-up text. This should be EXCLUDED.

Your task: find the LAST line number that is still genuinely part of the student's answer to the target question -- the line right before any such trailing, non-answer-specific material begins (if any exists). If the entire text shown is genuine answer content with no trailing wrap-up material, the last line of the text IS the answer's end.

Guidance:
- Only exclude trailing content if it is clearly NOT part of answering this specific question -- e.g. it talks about the assignment/paper/course as a whole rather than continuing this answer's explanation.
- Do NOT exclude a line just because it sounds like a summary or concluding sentence OF THIS ANSWER ITSELF -- a student's own concluding sentence for their answer is normal and should be INCLUDED.
- If you are not confident there is any trailing non-answer material, default to including everything (report the last line of the text shown).

Return ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:

{"end_line": 87}

end_line MUST be one of the exact line numbers shown in [brackets] in the text below."""


def _build_last_answer_end_prompt(tail_lines: list, question_text: str, ref_label: str) -> str:
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in tail_lines)
    return f"TARGET QUESTION ({ref_label}): {question_text}\n\nTEXT (line-numbered, this is the tail of the document):\n{lines_block}"


def _parse_last_answer_end_response(content: str) -> int:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()
    data = json.loads(content)
    if not isinstance(data, dict) or "end_line" not in data:
        raise ValueError(f"Response missing 'end_line' key: {data!r}")
    return int(data["end_line"])


LAST_ANSWER_END_TAIL_CHARS = 9000


def _check_last_answer_end(client, numbered_lines: list, question_text: str, ref_label: str,
                             start_line: int, budget: "_TokenBudgetTracker", log,
                             tail_chars: int = LAST_ANSWER_END_TAIL_CHARS) -> int:
    """
    Verifies where the chronologically LAST matched answer actually ends,
    instead of blindly assuming it runs to the very end of the document.
    Two layers:
    1. DETERMINISTIC cap: if an overall-assignment closing section
       heading (e.g. "## Conclusion") exists anywhere after start_line,
       the end can never be at or after it -- this is a hard ceiling,
       independent of LLM judgment.
    2. LLM tail-check within whatever remains after the deterministic
       cap, for un-headed trailing wrap-up material (footers, "thank
       you" notes, etc.) that has no clean heading marker.
    Falls back to the deterministic (or full-document) end on any LLM
    failure or invalid response.
    """
    total_lines = len(numbered_lines)
    fallback_end = total_lines - 1

    heading_idx = _find_overall_closing_heading(numbered_lines, start_line)
    if heading_idx is not None:
        capped_end = heading_idx - 1
        if capped_end < fallback_end:
            log(
                f"  last-answer-end check: found an overall assignment closing section "
                f"heading at line {heading_idx} (e.g. '## Conclusion') -- {ref_label}'s end "
                f"is hard-capped at line {capped_end}, regardless of the LLM tail check below"
            )
        fallback_end = min(fallback_end, capped_end)

    if start_line >= fallback_end:
        return max(start_line, fallback_end)

    chars = 0
    idx = fallback_end
    collected = []
    while idx >= start_line and chars < tail_chars:
        collected.append(numbered_lines[idx])
        chars += len(numbered_lines[idx][1])
        idx -= 1
    collected.reverse()

    if not collected:
        return fallback_end

    prompt = _build_last_answer_end_prompt(collected, question_text, ref_label)
    try:
        end_line = _call_groq_with_retries(
            client, LAST_ANSWER_END_SYSTEM_PROMPT, prompt,
            _parse_last_answer_end_response, budget, log, max_retries=2
        )
    except Exception as e:
        log(f"WARNING: last-answer-end check failed for {ref_label}: {e}")
        return fallback_end

    valid_ids = {i for i, _ in collected}
    if end_line in valid_ids and start_line <= end_line <= fallback_end:
        if end_line != fallback_end:
            log(
                f"  last-answer-end check: trimmed {ref_label}'s end from line {fallback_end} "
                f"to line {end_line} (excluded trailing non-answer content, e.g. an overall "
                f"assignment conclusion, that is not part of this specific answer)"
            )
        return end_line

    log(
        f"WARNING: last-answer-end check for {ref_label} returned invalid end_line "
        f"{end_line} -- falling back to line {fallback_end}"
    )
    return fallback_end


# =========================================================
# ANSWER MAPPING -- CLASSIFICATION STRATEGY (alternative to the
# sequential/transition strategy below)
#
# Instead of walking forward per QUESTION with isolated/transition
# sliding-window searches (map_answers_sequential), this shows the LLM
# char-budgeted, OVERLAPPING CHUNKS of the full line-numbered answer
# text alongside the COMPLETE list of questions, and asks a single
# classification question per chunk: "which of these questions' answers
# genuinely start somewhere in THIS chunk?" This is a simpler, more
# stable task for an LLM (pick matches from a fixed list) than inferring
# open-ended start/end boundaries one question at a time, and needs far
# fewer LLM calls overall -- roughly one call per chunk, instead of one
# or more calls per question.
# =========================================================

ANSWER_CLASSIFY_CHUNK_CHARS = 9000
ANSWER_CLASSIFY_OVERLAP_CHARS = 1500


def _ref_label(i: int) -> str:
    return f"REF-{chr(65 + i)}"


def _chunk_numbered_lines(numbered_lines: list, max_chars: int = ANSWER_CLASSIFY_CHUNK_CHARS,
                            overlap_chars: int = ANSWER_CLASSIFY_OVERLAP_CHARS) -> list:
    """
    Splits numbered_lines (contiguous global (idx, text) pairs) into
    char-budgeted chunks, with each chunk overlapping the TAIL of the
    previous chunk by roughly overlap_chars -- so an answer start that
    falls right at a chunk boundary is shown with full context in at
    least one chunk, instead of being split across two chunks and
    potentially missed by both.
    """
    if not numbered_lines:
        return []

    chunks = []
    n = len(numbered_lines)
    start = 0
    while start < n:
        chars = 0
        end = start
        while end < n and (end == start or chars + len(numbered_lines[end][1]) <= max_chars):
            chars += len(numbered_lines[end][1])
            end += 1
        chunks.append(numbered_lines[start:end])

        if end >= n:
            break

        back_chars = 0
        back_idx = end - 1
        while back_idx > start and back_chars < overlap_chars:
            back_chars += len(numbered_lines[back_idx][1])
            back_idx -= 1
        start = max(back_idx + 1, start + 1)

    return chunks


ANSWER_CLASSIFY_SYSTEM_PROMPT = """You are analyzing a WINDOW (a portion, not the whole document) of line-numbered OCR text from a student's exam answer booklet. You are also given the COMPLETE list of official exam questions, each with a short reference code (REF-A, REF-B, ...).

Your task: for EACH question in the list, decide whether the student's answer to THAT question genuinely BEGINS somewhere within this window. Most questions will NOT begin in any given window -- that is normal and expected. Only report a question if its answer's opening line is visible in this window.

===========================================================
RULE 1 -- REPORT A START, NOT A CONTINUATION
===========================================================
A window often shows the MIDDLE of an answer that already began earlier in the document, outside this window. If the text here reads as flowing prose clearly continuing an argument already in progress (no label, no restatement, no obvious topic-opening sentence), do NOT report it as a start -- it's a continuation.

===========================================================
RULE 2 -- WHAT A GENUINE START LOOKS LIKE
===========================================================
A genuine answer start is typically one of:
  - A short label ("Ans 5-", "उत्तर 6", "Q.5", "5)", "(a)", "(b)", "(i)", "Answer:")
  - A one-line restatement or paraphrase of the question, right before the real explanation begins
  - A brief introductory/transitional sentence that clearly opens discussion of that specific topic, understandable without needing earlier context
If there is genuine doubt between two nearby candidate lines for the SAME question, prefer the EARLIER one.

===========================================================
RULE 3 -- IGNORE OCR ARTIFACT/ANNOTATION DESCRIPTIONS
===========================================================
Some lines are the OCR engine's own description of a visual element (e.g. "[Logo]", "There is a red pen mark here", "Scribbled line", "Stamp", "Signature") rather than actual student writing. These are never a genuine answer start.

===========================================================
RULE 4 -- REPEATED CONTENT IS NORMAL, DON'T OVER-REPORT
===========================================================
The same fact/definition can legitimately appear more than once in the document. Be conservative: only report a question as starting here if THIS occurrence really reads like the beginning of a dedicated answer to it, not just a passing mention inside another answer.

===========================================================
RULE 5 -- A QUESTION CAN ONLY START ONCE
===========================================================
Report at most ONE start line per question for this window (the earliest, if multiple candidates exist here).

Return ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:

{"found": [{"ref": "REF-B", "start_line": 202}, {"ref": "REF-E", "start_line": 340}]}

If NO question's answer begins anywhere in this window, return {"found": []} -- this is a common and expected result. start_line MUST be one of the exact line numbers shown in [brackets] in this window."""


def _build_answer_classify_prompt(chunk_lines: list, questions: list) -> str:
    q_block = "\n".join(f"{_ref_label(i)}: {q}" for i, q in enumerate(questions))
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in chunk_lines)
    return (
        f"QUESTIONS (complete list, in order):\n{q_block}\n\n"
        f"TEXT WINDOW (line-numbered):\n{lines_block}"
    )


def _parse_answer_classify_response(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}\nRaw (first 400 chars): {content[:400]!r}")

    if not isinstance(data, dict) or "found" not in data:
        raise ValueError(f"Response missing 'found' key. Got: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}")

    found = data["found"]
    if not isinstance(found, list):
        raise ValueError(f"'found' must be a list, got: {type(found).__name__}")

    result = {}
    for item in found:
        if not isinstance(item, dict) or "ref" not in item or "start_line" not in item:
            continue
        ref = str(item["ref"]).strip()
        try:
            start_line = int(item["start_line"])
        except (ValueError, TypeError):
            continue
        result[ref] = start_line

    return result


def map_answers_classification(answer_lines: list, questions: list, status_callback=None,
                                 answer_line_pages: list = None) -> list:
    """
    Classification-based answer-mapping strategy (default). Chunks are
    processed in document order; for each ref, the FIRST chunk that
    reports it wins (a later chunk reporting the same ref again -- e.g.
    because the student briefly revisits the topic -- is ignored, since
    the first genuine detection in document order is almost always the
    true start).

    Safety nets, reused from the sequential strategy:
    - Monotonicity enforcement: a later question's start must come after
      the previous question's -- an inconsistent match is discarded and
      retried below, rather than silently producing overlapping/
      out-of-order ranges.
    - _verify_earliest_start backward check on every confirmed start.
    - A bounded isolated fallback search for any question no chunk ever
      flagged, scoped between its nearest confirmed neighbors.
    - The same Python-computed ends and last-answer-end verification
      used by map_answers_sequential.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    from openai import OpenAI

    api_key = get_api_key("OPENROUTER_API_KEY")
    if not api_key:
        raise Exception("OPENROUTER_API_KEY not found in secrets or environment")

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    budget = _TokenBudgetTracker()

    numbered_lines = list(enumerate(answer_lines))
    total_lines = len(numbered_lines)
    ref_to_question = {_ref_label(i): q for i, q in enumerate(questions)}

    chunks = _chunk_numbered_lines(numbered_lines)
    log(f"Split {total_lines} answer line(s) into {len(chunks)} classification chunk(s)")

    found_starts = {}  # ref -> global start_line

    for ci, chunk in enumerate(chunks):
        if len(found_starts) == len(questions):
            log(f"All {len(questions)} question(s) already matched -- skipping remaining chunks")
            break

        log(f"Classifying chunk {ci + 1}/{len(chunks)} (lines {chunk[0][0]}-{chunk[-1][0]})...")
        prompt = _build_answer_classify_prompt(chunk, questions)
        try:
            chunk_result = _call_groq_with_retries(
                client, ANSWER_CLASSIFY_SYSTEM_PROMPT, prompt,
                _parse_answer_classify_response, budget, log
            )
        except Exception as e:
            log(f"WARNING: classification failed for chunk {ci + 1}/{len(chunks)}, skipping: {e}")
            continue

        valid_ids = {idx for idx, _ in chunk}
        for ref, start_line in chunk_result.items():
            if ref not in ref_to_question:
                log(f"WARNING: chunk {ci + 1} reported unknown ref {ref!r} -- ignoring")
                continue
            if start_line not in valid_ids:
                log(
                    f"WARNING: chunk {ci + 1} reported {ref} at line {start_line}, outside "
                    f"this chunk's actual range {chunk[0][0]}-{chunk[-1][0]} -- ignoring"
                )
                continue
            if ref in found_starts:
                continue
            found_starts[ref] = start_line
            log(f"  {ref} starts at line {start_line} (chunk {ci + 1})")

    # Monotonicity enforcement (see docstring).
    prev_start = -1
    for i in range(len(questions)):
        ref = _ref_label(i)
        if ref not in found_starts:
            continue
        if found_starts[ref] <= prev_start:
            log(
                f"WARNING: {ref}'s classified start (line {found_starts[ref]}) is not after "
                f"the previous question's confirmed start (line {prev_start}) -- discarding "
                f"as inconsistent with document order; will retry with a bounded fallback search"
            )
            del found_starts[ref]
            continue
        prev_start = found_starts[ref]

    # Backward verification, in QUESTION order so min_allowed_line is the
    # previous QUESTION's confirmed start.
    ordered_refs = [r for r in (_ref_label(i) for i in range(len(questions))) if r in found_starts]
    prev_start = -1
    for ref in ordered_refs:
        q = ref_to_question[ref]
        verified = _verify_earliest_start(
            client, numbered_lines, answer_line_pages, found_starts[ref], q, ref,
            prev_start, budget, log
        )
        found_starts[ref] = verified
        prev_start = verified

    # Bounded fallback search for any question no chunk ever flagged.
    for i, q in enumerate(questions):
        ref = _ref_label(i)
        if ref in found_starts:
            continue

        later_start = None
        for j in range(i + 1, len(questions)):
            cand = _ref_label(j)
            if cand in found_starts:
                later_start = found_starts[cand]
                break

        earlier_start = 0
        for j in range(i - 1, -1, -1):
            cand = _ref_label(j)
            if cand in found_starts:
                earlier_start = found_starts[cand] + 1
                break

        hi = later_start if later_start is not None else total_lines
        log(
            f"Fallback: {ref} was not classified into any chunk -- retrying with a bounded "
            f"isolated search restricted to lines {earlier_start}-{hi - 1}..."
        )
        gap_slice = numbered_lines[earlier_start:hi]
        gap_start = _find_answer_start_sequential(client, gap_slice, q, ref, 0, budget, log)
        if gap_start is not None:
            gap_start = _verify_earliest_start(
                client, numbered_lines, answer_line_pages, gap_start, q, ref,
                earlier_start - 1, budget, log
            )
            found_starts[ref] = gap_start
            log(f"  fallback recovered {ref} starting at line {gap_start}")
        else:
            log(f"  fallback could not find {ref} either -- leaving unmatched")

    # Debug context (same as sequential strategy) for every confirmed start.
    for ref, start_line in sorted(found_starts.items(), key=lambda kv: kv[1]):
        ctx_lo = max(0, start_line - 4)
        ctx_hi = min(total_lines, start_line + 3)
        log(f"  [context] lines {ctx_lo}-{ctx_hi - 1} around {ref}'s confirmed start (>>> marks the chosen start line):")
        for cix in range(ctx_lo, ctx_hi):
            marker = ">>>" if cix == start_line else "   "
            preview = answer_lines[cix][:180].replace("\n", " ")
            log(f"  [context] {marker} [{cix}] {preview!r}")

    ordered = sorted(found_starts.items(), key=lambda kv: kv[1])
    ranges = []
    for idx, (ref, start) in enumerate(ordered):
        if idx + 1 < len(ordered):
            end = ordered[idx + 1][1] - 1
        else:
            q_last = ref_to_question[ref]
            end = _check_last_answer_end(client, numbered_lines, q_last, ref, start, budget, log)
        ranges.append({"ref": ref, "start_line": start, "end_line": end})

    log(f"Classification mapping found {len(ranges)} of {len(questions)} question(s)")

    ranges_by_ref = {r["ref"]: r for r in ranges}
    results = []
    for i, q in enumerate(questions):
        ref = _ref_label(i)
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
        verbatim_lines = [
            answer_lines[j] for j in range(s, e + 1)
            if 0 <= j < len(answer_lines) and answer_lines[j].strip() and not is_noise(answer_lines[j])
        ]
        answer_raw = " ".join(verbatim_lines).strip()
        answer_clean = strip_question_restatement(answer_raw)
        answer_clean = strip_full_question_echo(answer_clean, q)
        answer_clean = _strip_trailing_question_echo_sentences(answer_clean, questions)

        start_page = answer_line_pages[s] if answer_line_pages and 0 <= s < len(answer_line_pages) else None
        end_page = answer_line_pages[e] if answer_line_pages and 0 <= e < len(answer_line_pages) else None

        results.append({
            "ref": ref, "question": q, "matched": True,
            "start_line": s, "end_line": e,
            "start_page": start_page, "end_page": end_page,
            "answer": answer_clean, "answer_raw": answer_raw,
        })

    return results


def map_answers_sequential(answer_lines: list, questions: list, status_callback=None,
                             answer_line_pages: list = None) -> list:
    """
    Default answer-mapping strategy.

    Stage 1: find the start of the FIRST question via an isolated
    single-target search (no previous answer exists to contrast against).

    Stage 2: for every SUBSEQUENT question, find the TRANSITION from the
    previous (already-open) question to this one, showing the LLM BOTH
    question texts together for direct comparison. If a transition can't
    be found (or the previous question's own start was never found, so
    there's no anchor to search a transition from), fall back to an
    isolated single-target search for this specific question. This
    combination gives the accuracy benefit of contrastive search in the
    normal case, with the resilience of isolated search when a question
    was skipped or answered out of order.

    Stage 3: EVERY confirmed start_line (whichever stage found it) is run
    through _verify_earliest_start, a mandatory backward check that
    re-examines the OCR page it falls on (plus the previous page) for a
    missed earlier line. This is what catches the "answers missing their
    opening paragraph/page" failure mode -- see the module note above
    SEQUENTIAL_SEARCH_SYSTEM_PROMPT.

    Every answer's END (except the chronologically last one) is computed
    in Python as (next confirmed answer's start - 1) -- never asked of
    the LLM, so it can never be wrong in the way an LLM-guessed end line
    could be. The LAST matched answer's end is separately verified (see
    _check_last_answer_end) instead of blindly assumed to run to the end
    of the document, so a trailing whole-assignment conclusion/footer
    doesn't get absorbed into it.

    Returns a LIST of dicts, one per question, each carrying:
      - start_line / end_line: the exact 0-based indices into
        answer_lines this answer was sliced from
      - start_page / end_page: the OCR page number(s) the answer spans,
        if answer_line_pages was provided
      - answer_raw: the UNMODIFIED verbatim join of the sliced lines
      - answer: the same text after restatement-stripping cleanup
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    from openai import OpenAI

    api_key = get_api_key("OPENROUTER_API_KEY")
    if not api_key:
        raise Exception("OPENROUTER_API_KEY not found in secrets or environment")

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    budget = _TokenBudgetTracker()

    numbered_lines = list(enumerate(answer_lines))
    total_lines = len(numbered_lines)

    ref_to_question = {f"REF-{chr(65+i)}": q for i, q in enumerate(questions)}
    found_starts = {}  # ref -> start_line
    pointer = 0

    REPEAT_RETRY_REMINDER = (
        "REMINDER: a previous search pass over this exact text did not find this "
        "question's answer. One common reason for a missed match: the same "
        "definition/explanation legitimately appears more than once in this document. "
        "Seeing similar-looking content earlier does NOT mean this occurrence isn't a "
        "genuine, separate answer to THIS target question -- look again with that in "
        "mind, and also double-check you are not missing a short introductory/"
        "transitional line right at the true start of the answer."
    )

    for i, q in enumerate(questions):
        ref = f"REF-{chr(65 + i)}"

        if i == 0:
            # Stage 1: the very first question has no previous answer to
            # contrast against -- isolated single-target search is the
            # only option here.
            log(f"Searching for the start of {ref} ({q[:60]}...) from line {pointer} onward...")
            start_line = _find_answer_start_sequential(
                client, numbered_lines, q, ref, pointer, budget, log
            )
            if start_line is None:
                log(f"  first pass found nothing for {ref} -- retrying once with an explicit reminder...")
                start_line = _find_answer_start_sequential(
                    client, numbered_lines, q, ref, pointer, budget, log,
                    extra_reminder=REPEAT_RETRY_REMINDER
                )
                if start_line is not None:
                    log(f"  retry recovered {ref} starting at line {start_line}")
        else:
            prev_ref = f"REF-{chr(65 + i - 1)}"
            prev_q = questions[i - 1]

            if prev_ref not in found_starts:
                # No confirmed start for the previous question means
                # there's no anchor point to search a transition FROM --
                # fall back to an isolated search for this question
                # instead.
                log(
                    f"  {prev_ref} has no confirmed start -- falling back to an isolated "
                    f"search for {ref} ({q[:60]}...) from line {pointer} onward..."
                )
                start_line = _find_answer_start_sequential(
                    client, numbered_lines, q, ref, pointer, budget, log
                )
                if start_line is None:
                    start_line = _find_answer_start_sequential(
                        client, numbered_lines, q, ref, pointer, budget, log,
                        extra_reminder=REPEAT_RETRY_REMINDER
                    )
            else:
                # Stage 2: comparative transition search -- shows the LLM
                # BOTH questions together so it has a genuine contrast to
                # reason about, instead of an isolated guess.
                log(
                    f"Searching for the transition from {prev_ref} to {ref} "
                    f"({q[:60]}...) from line {pointer} onward..."
                )
                start_line = _find_transition_sequential(
                    client, numbered_lines, prev_q, q, pointer, budget, log
                )

                if start_line is None:
                    log(f"  transition search found nothing -- retrying once with an explicit reminder...")
                    transition_retry_reminder = (
                        "REMINDER: a previous search pass over this exact text did not find "
                        "the transition to the NEXT question. Look again -- the same concept "
                        "can legitimately appear in both answers, and the transition may begin "
                        "with a short introductory line rather than an obviously on-topic "
                        "sentence. Always prefer the earliest plausible transition line."
                    )
                    start_line = _find_transition_sequential(
                        client, numbered_lines, prev_q, q, pointer, budget, log,
                        extra_reminder=transition_retry_reminder
                    )

                if start_line is None:
                    # The NEXT question's answer might not immediately
                    # follow the previous one (e.g. the student skipped
                    # it, or answered out of order) -- fall back to an
                    # isolated single-target search for this specific
                    # question over the same remaining text, rather than
                    # giving up entirely.
                    log(f"  transition search failed -- falling back to an isolated search for {ref}")
                    start_line = _find_answer_start_sequential(
                        client, numbered_lines, q, ref, pointer, budget, log
                    )
                else:
                    log(f"  found transition -- {ref} starts at line {start_line}")

        if start_line is not None:
            # Stage 3: mandatory backward verification. Never moves the
            # start earlier than the previous confirmed answer's start,
            # so this can only recover genuinely-missed opening lines --
            # it can never cause answers to overlap.
            prev_start = found_starts.get(f"REF-{chr(65 + i - 1)}", -1) if i > 0 else -1
            start_line = _verify_earliest_start(
                client, numbered_lines, answer_line_pages, start_line, q, ref,
                prev_start, budget, log
            )
            found_starts[ref] = start_line
            log(f"  found {ref} starting at line {start_line}")

            # DEBUG CONTEXT: print the lines immediately before and after
            # the confirmed start, so a wrong/skipped start is directly
            # visible in the run logs without needing a separate
            # ocr.json download.
            ctx_lo = max(0, start_line - 4)
            ctx_hi = min(len(answer_lines), start_line + 3)
            log(f"  [context] lines {ctx_lo}-{ctx_hi - 1} around {ref}'s confirmed start (>>> marks the chosen start line):")
            for ci in range(ctx_lo, ctx_hi):
                marker = ">>>" if ci == start_line else "   "
                preview = answer_lines[ci][:180].replace("\n", " ")
                log(f"  [context] {marker} [{ci}] {preview!r}")

            pointer = start_line + 1
        else:
            log(
                f"WARNING: could not find the start of {ref} anywhere from line {pointer} "
                f"to the end of the document ({total_lines} lines) -- marking as unmatched. "
                f"The search pointer is NOT advanced, so the next question is still searched "
                f"for over this same remaining text."
            )

    # ---------------------------------------------------------------
    # GAP-FILL PASS: a question the main forward pass could not match
    # (isolated search AND transition search both failed) is not
    # necessarily unanswered in the document -- it commonly means the
    # search simply missed it while scanning past, and its content then
    # silently got absorbed into the PREVIOUS matched answer's range
    # (since that answer's end is computed as "next matched start - 1").
    # If this question has a LATER question that WAS matched, we now
    # know a hard upper bound for where its answer must be, and a hard
    # lower bound from the previous matched question (or 0). Re-searching
    # within that bounded gap is far more reliable than the original
    # unbounded forward search, and -- critically -- cannot ever report
    # a line inside a neighboring answer's territory, because the search
    # is never even shown lines outside the gap.
    # ---------------------------------------------------------------
    for i, q in enumerate(questions):
        ref = f"REF-{chr(65 + i)}"
        if ref in found_starts:
            continue

        later_start = None
        for j in range(i + 1, len(questions)):
            cand_ref = f"REF-{chr(65 + j)}"
            if cand_ref in found_starts:
                later_start = found_starts[cand_ref]
                break
        if later_start is None:
            continue  # no later anchor -- nothing to bound the gap with

        earlier_start = 0
        for j in range(i - 1, -1, -1):
            cand_ref = f"REF-{chr(65 + j)}"
            if cand_ref in found_starts:
                earlier_start = found_starts[cand_ref] + 1
                break

        log(
            f"Gap-fill: {ref} was not matched by the main pass -- retrying with a "
            f"bounded search restricted to lines {earlier_start}-{later_start - 1} "
            f"(bounded by the nearest matched questions on either side, so this "
            f"cannot bleed into a neighbor's answer)..."
        )
        gap_slice = numbered_lines[earlier_start:later_start]
        gap_start = _find_answer_start_sequential(
            client, gap_slice, q, ref, 0, budget, log
        )
        if gap_start is not None:
            gap_start = _verify_earliest_start(
                client, numbered_lines, answer_line_pages, gap_start, q, ref,
                earlier_start - 1, budget, log
            )
            found_starts[ref] = gap_start
            log(f"  gap-fill recovered {ref} starting at line {gap_start}")
        else:
            log(f"  gap-fill could not find {ref} either -- leaving unmatched")

    # End of each answer = the next (in document order) confirmed
    # answer's start, minus one -- computed purely in Python. The
    # exception is the chronologically LAST matched answer, whose end is
    # separately verified instead of being blindly assumed to run to the
    # end of the document (see _check_last_answer_end).
    ordered = sorted(found_starts.items(), key=lambda kv: kv[1])
    ranges = []
    for idx, (ref, start) in enumerate(ordered):
        if idx + 1 < len(ordered):
            end = ordered[idx + 1][1] - 1
        else:
            q_last = ref_to_question[ref]
            end = _check_last_answer_end(client, numbered_lines, q_last, ref, start, budget, log)
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
        answer_clean = _strip_trailing_question_echo_sentences(answer_clean, questions)

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


NOISE_RE = re.compile(
    r'(?:signature'
    r'|PAGE\s*NO'
    r'|^\s*DATE\b'
    r'|^\s*\d{1,3}\s*$'
    r'|\blogo\b'
    r'|\bwatermark\b'
    r'|\bstamp\b'
    r'|\bscribbl\w*\b'
    r'|\bdoodle\w*\b)',
    re.IGNORECASE
)

# A line is only treated as administrative noise if it's SHORT (a bare
# label like "Teacher's Signature" or "Date: __") -- not if "signature"/
# "date"/"page" merely appears as a word inside a much longer genuine
# sentence (e.g. a computer-science answer discussing "digital
# signature" is real content, not a label to strip). This length guard
# is what makes the generic keyword match safe to use across ANY
# document/subject instead of needing document-specific hardcoded names.
NOISE_LINE_MAX_CHARS = 40


# =========================================================
# OCR sometimes emits a full descriptive SENTENCE for a non-text visual
# element on the scanned page instead of transcribing actual student
# writing -- e.g. "There is a logo in the top right corner.",
# "Scribbled line in red pen.", "Stamp on the page.", markdown image
# placeholders, etc. These are OCR metadata/commentary about the PAGE,
# not student answer content. This pattern is checked regardless of line
# length, unlike NOISE_RE above, because these descriptive sentences can
# run longer than the short-label case NOISE_RE is designed for.
# =========================================================
_ANSWER_START_RE = re.compile(
    r'^\s*(?:Ans(?:wer)?[.\s:-]+\d*|उत्तर\s*\d*\s*[\-\:]?|प्र[०.\s]*\d*|Q\.?\s*\d+[.\s:-])',
    re.IGNORECASE
)

_OCR_ARTIFACT_DESCRIPTION_RE = re.compile(
    r'^\s*(?:'
    r'\[\s*(?:image|figure|logo|stamp|watermark|photo|picture|diagram)\s*\]'
    r'|<!--\s*image\s*-->'
    r'|!\[[^\]]*\]\([^)]*\)'
    r'|\(?\s*there\s+(?:is|are|seems?\s+to\s+be|appears?\s+to\s+be)\s+(?:a|an|some)?\s*'
    r'(?:logo|stamp|watermark|scribbl\w*|doodle\w*|drawing|symbol|mark(?:ing)?s?|'
    r'line|box|circle|underlin\w*|arrow|star|tick|cross|red\s*(?:pen|ink)\w*)'
    r'|\(?\s*(?:handwritten|scribbled|marked|underlined|circled)\s+(?:in\s+)?red\s*(?:pen|ink)?\s*\)?'
    r')',
    re.IGNORECASE
)

# Broader signature for OCR narrating an examiner's red-pen ANNOTATION
# (a circle around a question number, an arrow, a tick mark, etc.) as a
# full descriptive sentence, rather than transcribing real writing.
# Unlike _OCR_ARTIFACT_DESCRIPTION_RE above, these sentences frequently
# do NOT start with "there is" -- e.g. "A red circle containing the
# number 11. A long red arrow originates from the circle and points
# diagonally downwards..." -- so an anchored regex misses them. Instead,
# require at least 2 of 3 independent signal categories (mark-type word,
# color/ink word, description-action phrase) to co-occur in the same
# line, so genuine academic content that happens to mention "arrow" or
# "circle" once in passing is not falsely flagged.
_ANNOTATION_MARK_WORDS_RE = re.compile(
    r'\b(?:circl(?:e|ing|ed)|arrow|underlin\w*|scribbl\w*|doodle\w*|loop\w*|'
    r'tick\s*mark|cross\s*mark|strike[\s-]?through)\b',
    re.IGNORECASE
)
_ANNOTATION_COLOR_WORDS_RE = re.compile(r'\bred\s*(?:pen|ink|colou?r)?\b', re.IGNORECASE)
_ANNOTATION_ACTION_WORDS_RE = re.compile(
    r'\b(?:originates?\s+from|points?\s+(?:diagonally|towards?|downwards?|upwards?|'
    r'to\s+the\s+(?:left|right))|containing\s+the\s+number|blank\s+space\s+of\s+the\s+page|'
    r'corner\s+of\s+the\s+page|across\s+the\s+page|drawn\s+(?:in|on)|marked?\s+(?:in|with)|'
    r'extending\s+(?:from|to|towards)|(?:bottom|top)\s+(?:left|right)\s+(?:corner|of\s+the\s+page)?|'
    r'overlapping\s+loops?)\b',
    re.IGNORECASE
)
_ANNOTATION_DESCRIPTION_MAX_CHARS = 300


def _is_ocr_artifact_description(line: str) -> bool:
    """
    Detects lines where the OCR engine described a VISUAL artifact on the
    page (a logo, stamp, red-pen scribble/underline/circle/arrow, stray
    mark, doodle, etc.) in prose, instead of transcribing actual student
    writing. These are OCR commentary about the page, never student
    answer content, and must be excluded both from extracted answer text
    AND from candidate answer-start/transition lines.
    """
    stripped = line.strip()
    if not stripped:
        return False

    if _OCR_ARTIFACT_DESCRIPTION_RE.match(stripped):
        return True

    if len(stripped) <= _ANNOTATION_DESCRIPTION_MAX_CHARS:
        has_mark = bool(_ANNOTATION_MARK_WORDS_RE.search(stripped))
        has_color = bool(_ANNOTATION_COLOR_WORDS_RE.search(stripped))
        has_action = bool(_ANNOTATION_ACTION_WORDS_RE.search(stripped))
        if (has_mark + has_color + has_action) >= 2:
            return True

    return False


def _strip_inline_ocr_artifacts(line: str) -> str:
    """
    Some OCR lines mix genuine answer prose with an artifact-description
    sentence in the SAME line -- e.g. "...enduring relevance. A red
    scribble or signature mark. ## Conclusion..." -- which whole-line
    noise filtering (is_noise / NOISE_LINE_MAX_CHARS) cannot safely
    remove without also deleting real content, since the combined line
    is long and the artifact sentence isn't the whole line. This splits
    the line into sentences and drops ONLY the sentence(s) that look
    like an OCR artifact/annotation description, keeping everything else
    verbatim and in original order.
    """
    if not line.strip():
        return line
    sentences = re.split(r'(?<=[.!?])\s+', line)
    kept = [s for s in sentences if not _is_ocr_artifact_description(s.strip())]
    return " ".join(kept).strip()


def is_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r'^\s*\d{1,3}\s*$', stripped):
        return True  # bare page-number line -- always noise regardless of length
    if _is_ocr_artifact_description(stripped):
        return True  # OCR's own description of a visual artifact, not student content
    if len(stripped) > NOISE_LINE_MAX_CHARS:
        return False
    return bool(NOISE_RE.search(stripped))


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


def _split_into_sentences(text: str) -> list:
    parts = re.split(r'(?<=[.!?।])\s+', text.strip())
    return [p for p in parts if p.strip()]


def _strip_trailing_question_echo_sentences(answer_text: str, all_questions: list,
                                              ratio_threshold: float = 0.82,
                                              max_sentences_to_check: int = 3) -> str:
    """
    Deterministic (non-LLM), pure-string-similarity safety net for
    boundary leakage where a QUESTION's printed text (often the NEXT
    question, sometimes pre-printed at the bottom of an answer sheet)
    ends up appended at the TAIL of an extracted answer. Checks only the
    last few sentences (leakage of this kind is always at the tail, and
    is bounded so a genuinely long answer that happens to end with a
    sentence resembling a question is not over-trimmed) against EVERY
    question in the canonical question paper, using difflib string
    similarity only -- no LLM involved, so this can never rewrite or
    paraphrase real answer content, only remove a sentence that is a
    near-exact match to a question that was independently extracted
    from the actual printed question paper.
    """
    if not answer_text.strip() or not all_questions:
        return answer_text

    sentences = _split_into_sentences(answer_text)
    if len(sentences) <= 1:
        return answer_text

    normalized_questions = [_normalize_for_echo_compare(q) for q in all_questions if q.strip()]
    if not normalized_questions:
        return answer_text

    checks_done = 0
    while sentences and checks_done < max_sentences_to_check:
        checks_done += 1
        last_norm = _normalize_for_echo_compare(sentences[-1])
        if len(last_norm.split()) < 3:
            break  # too short to reliably judge against a question -- stop trimming

        matched = False
        for q_norm in normalized_questions:
            if q_norm and difflib.SequenceMatcher(None, last_norm, q_norm).ratio() >= ratio_threshold:
                matched = True
                break

        if not matched:
            break
        sentences.pop()

    return " ".join(sentences).strip()


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


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
    """
    Lightweight, always-on diagnostic (not a hard failure) that flags
    matched answers which are suspiciously short compared to the rest of
    the document's answers -- a strong signal of truncation (start or end
    clipped) even when a range WAS found.
    """
    matched_lengths = [len(p["answer"]) for p in qa_pairs if p.get("matched") and p["answer"].strip()]
    if len(matched_lengths) < 2:
        return
    matched_lengths.sort()
    median_len = matched_lengths[len(matched_lengths) // 2]
    if median_len < 50:
        return  # too short a document overall to say anything meaningful
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


def _flag_duplicate_matched_answers(qa_pairs: list, log=print) -> None:
    """
    Diagnostic safety net for "questions repeating" symptoms: flags any
    two MATCHED answers whose extracted text is near-identical, or whose
    line ranges overlap. This can happen if two distinct (but
    near-duplicate) canonical questions slipped past the dedup step, or
    if two refs were assigned overlapping ranges by the search stages.
    This never modifies the output -- it only surfaces the problem
    loudly in the log so it can be spotted immediately.
    """
    matched = [p for p in qa_pairs if p.get("matched") and p["answer"].strip()]
    for i in range(len(matched)):
        for j in range(i + 1, len(matched)):
            a, b = matched[i], matched[j]
            # Overlapping line ranges (should be impossible by construction,
            # but flag loudly if it ever happens).
            if a["start_line"] is not None and b["start_line"] is not None:
                if not (a["end_line"] < b["start_line"] or b["end_line"] < a["start_line"]):
                    log(
                        f"WARNING: overlapping answer ranges detected between "
                        f"'{a['question'][:50]}...' (lines {a['start_line']}-{a['end_line']}) "
                        f"and '{b['question'][:50]}...' (lines {b['start_line']}-{b['end_line']})"
                    )
            # Near-identical extracted answer text for two different
            # questions -- likely a duplicate-question slip-through.
            ratio = difflib.SequenceMatcher(None, a["answer"], b["answer"]).ratio()
            if ratio >= 0.9:
                log(
                    f"WARNING: near-identical answer text found for two different "
                    f"questions -- likely a duplicate/near-duplicate canonical question: "
                    f"'{a['question'][:50]}...' and '{b['question'][:50]}...'"
                )
                continue

            # SUBSTRING overlap check: catches a chunk of ONE answer's
            # content leaking into another's HEAD or TAIL, which the
            # whole-answer ratio check above misses whenever the leaked
            # portion is small relative to each answer's total length
            # (e.g. answer D is mostly genuine content of its own, but
            # ends with a paragraph that's actually answer B's content).
            # Uses the longest common contiguous substring between the
            # two answers as the signal, ignoring very short/generic
            # matches.
            matcher = difflib.SequenceMatcher(None, a["answer"], b["answer"])
            match = matcher.find_longest_match(0, len(a["answer"]), 0, len(b["answer"]))
            if match.size >= 150:
                shared = a["answer"][match.a:match.a + match.size]
                log(
                    f"WARNING: a {match.size}-character chunk of text is shared verbatim "
                    f"between '{a['question'][:50]}...' and '{b['question'][:50]}...' -- "
                    f"this usually means content from one answer's range leaked into the "
                    f"other's (a boundary/search bug), OR the source document genuinely "
                    f"repeats this passage twice. Shared text starts: {shared[:100]!r}"
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

    # Admin/cover pages (roll number, letterhead, etc.) are explicitly
    # excluded here too, not just question-paper pages. Otherwise ANY
    # page that wasn't classified as a question-paper page would fall
    # into "answer pages" by elimination -- including cover sheets -- so
    # their content (names, roll numbers, institution letterhead text)
    # could leak into an answer's range.
    excluded_indices = set(qp_page_indices) | set(admin_page_indices)
    answer_page_indices = [i for i in range(len(pages)) if i not in excluded_indices]
    answer_pages = [pages[i] for i in answer_page_indices]

    log(f"Answer pages: {[i+1 for i in answer_page_indices]}")

    # answer_line_pages[i] records which OCR page answer_lines[i] came
    # from -- kept in lockstep with answer_lines so every mapped answer
    # can report exactly which page(s) of the source PDF it was sliced
    # from, for direct manual verification against the scanned document.
    # Lines flagged as noise (bare page numbers, admin labels, OCR
    # artifact/annotation descriptions like a logo or a red-pen scribble)
    # are excluded here, at the source -- so they can never be picked as
    # a candidate answer-start/transition line, nor appear in any
    # extracted answer text.
    answer_lines = []
    answer_line_pages = []
    closing_section_reached = False
    for page in answer_pages:
        if closing_section_reached:
            log(
                f"Skipping page {page['page_number']} entirely -- the overall assignment "
                f"closing section (e.g. '## Conclusion') was already reached on an earlier "
                f"page, so nothing after it is treated as answer content."
            )
            continue
        page_kept_any = False
        for line in page["raw_text"].split("\n"):
            cleaned_line = _strip_inline_ocr_artifacts(line)
            kept_text, hit_closing = _truncate_before_overall_closing_heading(cleaned_line)
            if kept_text.strip() and not is_noise(kept_text):
                answer_lines.append(kept_text)
                answer_line_pages.append(page["page_number"])
                page_kept_any = True
            if hit_closing:
                log(
                    f"Reached the overall assignment closing section (e.g. '## Conclusion') "
                    f"on page {page['page_number']} -- stopping answer-line collection here. "
                    f"Everything from this point onward is document-level wrap-up, never part "
                    f"of any specific answer."
                )
                closing_section_reached = True
                break
        if not page_kept_any and page["raw_text"].strip() and not closing_section_reached:
            log(
                f"WARNING: page {page['page_number']} produced ZERO usable answer "
                f"lines after filtering, despite having {len(page['raw_text'])} chars "
                f"of raw OCR text. This usually means Chandra mis-transcribed the "
                f"WHOLE page as an annotation description (red-pen marks etc.) instead "
                f"of the actual handwriting -- real content may be LOST here, not just "
                f"filtered. Manually check page {page['page_number']} in the source PDF."
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

    log("Mapping each question to its answer (classification-based: chunk + full question list)...")
    qa_pairs = map_answers_classification(
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

    # =====================================================================
    # INTEGRITY CHECK: every answer_raw field above was built by a plain
    # Python slice of answer_lines (see map_answers_sequential) -- the LLM
    # is never given the opportunity to write or rephrase answer text, it
    # only ever returns a single line-number. This assertion makes that
    # guarantee mechanically verifiable rather than just a design claim.
    # =====================================================================
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
    _flag_duplicate_matched_answers(qa_pairs, log)

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
