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
# LLM-BASED QUESTION PAPER / QUESTION DETECTION (Groq)
# =========================================================

GROQ_MODEL = "openai/gpt-oss-120b"

TPM_LIMIT = 8000
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

    # =====================================================================
    # SAFETY NET: a page classified as "admin" that comes AFTER the last
    # question-paper page AND still has substantial text content is very
    # unlikely to genuinely be a blank cover sheet -- genuine admin pages
    # (roll number, letterhead) almost always sit at the START of a
    # booklet, before the question paper. A page near the END of the
    # document with real text is far more likely to be the tail end of
    # the student's LAST answer, which a chunk-boundary classification
    # call can mistake for "blank/admin" if that page happens to be short
    # (e.g. just a concluding paragraph). This was a confirmed cause of
    # "the last answer's final page silently disappeared" -- since an
    # admin page is excluded from BOTH question and answer text, its
    # content is gone entirely rather than merely misattributed.
    # =====================================================================
    TRAILING_ADMIN_MIN_CHARS = 150
    if admin_page_indices_0based and qp_page_indices_0based:
        last_qp_page = max(qp_page_indices_0based)
        reclassified = []
        for idx in list(admin_page_indices_0based):
            if idx > last_qp_page and len(pages[idx]["raw_text"].strip()) >= TRAILING_ADMIN_MIN_CHARS:
                reclassified.append(idx)

        if reclassified:
            for idx in reclassified:
                char_count = len(pages[idx]["raw_text"].strip())
                log(
                    f"RECLASSIFYING page {idx + 1}: was detected as an admin/cover page, but it "
                    f"comes AFTER the last question-paper page and contains {char_count} characters "
                    f"of real text -- almost certainly the tail end of the student's last answer, "
                    f"not a blank sheet. Moving it back to answer pages so its content isn't lost."
                )
            admin_page_indices_0based = [
                i for i in admin_page_indices_0based if i not in reclassified
            ]
            log(f"Admin/cover pages after trailing-page safety check: {len(admin_page_indices_0based)} page(s)")

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
# SEQUENTIAL SINGLE-TARGET ANSWER MAPPING (recommended, default)
# =========================================================

SEQUENTIAL_SEARCH_SYSTEM_PROMPT = """You are searching for exactly ONE thing in a block of line-numbered OCR text from a student's exam answer booklet: the line where the response to ONE SPECIFIC question begins.

You are given:
1. The exact text of the target question.
2. A window of the student's answer text, with each line prefixed by its line number in [brackets]. This window may be a small slice of a much larger document -- the answer you're looking for might not be in this window at all, and that is a normal, expected outcome.

Decide: does the student's response to THIS EXACT question begin somewhere in the window shown?

Guidance:
- A response typically begins where the student restates or references the question (e.g. "Ans 5-", "उत्तर 6-", "प्र. 8", a matching question number) OR, if there's no such label, where the content clearly starts addressing this specific question's topic for the first time (matching its distinctive subject matter, not just generic instructional words).
- Do not confuse this with a DIFFERENT question's answer, even if it appears earlier in the window -- you are looking for this one specific question only.
- If the target question's answer does not begin anywhere in this window, say so plainly. It is very common and expected for a window to not contain it -- do not force a match.

Return ONLY valid JSON (no markdown fences, no commentary) in exactly one of these two shapes:

{"found": true, "start_line": 42}

or

{"found": false}

If found, start_line MUST be one of the exact line numbers shown in [brackets] in this window -- never estimate or invent a number."""


def _build_sequential_search_prompt(window_lines: list, question_text: str, ref_label: str) -> str:
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in window_lines)
    return (
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
SEQUENTIAL_SEARCH_OVERLAP_LINES = 12  # windows overlap by this many lines so a restatement/marker
                                       # sitting right at a window boundary is never split in half
                                       # and rendered invisible to both windows


def _find_answer_start_sequential(client, numbered_lines: list, question_text: str, ref_label: str,
                                    search_from_idx: int, budget: "_TokenBudgetTracker", log,
                                    window_chars: int = SEQUENTIAL_SEARCH_WINDOW_CHARS,
                                    max_windows: int = SEQUENTIAL_SEARCH_MAX_WINDOWS,
                                    overlap_lines: int = SEQUENTIAL_SEARCH_OVERLAP_LINES):
    """
    Slides forward through numbered_lines in OVERLAPPING windows, starting
    at search_from_idx, asking a single yes/no+line-number question per
    window, until the target's start is found or the document is
    exhausted. Returns the found start_line, or None.

    Windows overlap by `overlap_lines` so that a restatement/answer-start
    marker sitting right at the boundary between two non-overlapping
    windows is never split in half and thus invisible to both windows --
    this was a confirmed cause of the search skipping straight past an
    answer's true beginning into a later, topically-similar paragraph
    (reported as "answer picked up mid-way, opening paragraph missing").
    """
    total_lines = len(numbered_lines)
    pointer = search_from_idx
    windows_tried = 0
    floor_idx = search_from_idx  # never search/return before this -- belongs to the previous answer

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

        user_prompt = _build_sequential_search_prompt(window, question_text, ref_label)
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
            if start_line in valid_ids and start_line >= floor_idx:
                return start_line
            log(
                f"WARNING: {ref_label} reported start_line {start_line}, which is outside "
                f"this window's valid range -- ignoring and treating this window as a non-match"
            )

        if idx >= total_lines:
            break
        # Advance with overlap: back up by overlap_lines from the end of
        # this window so the next window re-shows the tail of this one.
        # This guarantees no boundary marker is ever shown to NEITHER
        # window (only non-overlapping advancement could do that).
        pointer = max(pointer + 1, idx - overlap_lines)
        windows_tried += 1

    return None


BACKWARD_REFINE_SYSTEM_PROMPT = """You previously identified that a student's answer to a specific question begins at a certain line. Now double-check: does the answer ACTUALLY begin EARLIER than that, somewhere within the window of text shown here (which ends right at the previously-identified line)?

This matters a lot: the previous identification can sometimes be off by a WHOLE PAGE or more, not just a line or two -- e.g. it can accidentally latch onto a later paragraph that happens to restate the question's core definition more explicitly, while the answer's TRUE opening (an earlier, less obviously-labeled paragraph, possibly with no "Ans"/"उत्तर" label at all) sits further back and gets wrongly left attached to a DIFFERENT, earlier question's answer instead. Do not assume the previous identification is "probably close enough" -- actively scan the ENTIRE window shown, from its very first line, for the true earliest point where this specific question's subject matter is being addressed.

You are given:
1. The exact text of the target question.
2. A window of text ending exactly at the previously-identified start line (inclusive). This window may span more than one page's worth of the student's writing.

Decide: within THIS window, is there an EARLIER line where the answer to this exact question actually begins? Look for:
- An explicit restart label (e.g. "Ans 5-", "उत्तर 6-", "प्र. 8", a matching question number), OR
- A clear but UNLABELED shift in subject matter to this question's specific topic -- the student may simply start writing about it without any label at all, especially for sub-parts of a multi-part question.

Return ONLY valid JSON (no markdown fences, no commentary) in exactly one of these two shapes:

{"earlier_start_found": true, "start_line": 37}

or

{"earlier_start_found": false}

If earlier_start_found is true, start_line MUST be one of the exact line numbers shown in [brackets], and MUST be earlier than or equal to the last line shown. Report false only if you are genuinely confident this ENTIRE window is still part of an earlier, different topic and none of it belongs to the target question."""


def _build_backward_refine_prompt(window_lines: list, question_text: str, ref_label: str) -> str:
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in window_lines)
    return (
        f"TARGET QUESTION ({ref_label}): {question_text}\n\n"
        f"TEXT WINDOW ending at the previously-identified start line (line-numbered):\n{lines_block}"
    )


def _parse_backward_refine_response(content: str) -> tuple:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}\nRaw (first 300 chars): {content[:300]!r}")

    if not isinstance(data, dict) or "earlier_start_found" not in data:
        raise ValueError(f"Response missing 'earlier_start_found' key. Got: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}")

    found = bool(data["earlier_start_found"])
    if not found:
        return False, None

    if "start_line" not in data:
        raise ValueError("Response has earlier_start_found=true but is missing 'start_line'")
    try:
        start_line = int(data["start_line"])
    except (ValueError, TypeError):
        raise ValueError(f"'start_line' must be an integer, got {data['start_line']!r}")

    return True, start_line


BACKWARD_REFINE_WINDOW_CHARS = 3500  # deliberately smaller than the forward search's window --
                                       # roughly one page's worth of text per check, not two. A large
                                       # window gives the model too much to hold at once, making it
                                       # easy to miss a SUBTLE, unlabeled topic-shift buried partway
                                       # through -- exactly what caused a whole page to stay wrongly
                                       # attached to the wrong answer even with backward-refine active.
BACKWARD_REFINE_MAX_ITERATIONS = 30  # raised to compensate for the smaller window -- still cheap,
                                       # since most documents only need 1-2 iterations in practice


def _refine_start_backward(client, numbered_lines: list, question_text: str, ref_label: str,
                             candidate_start: int, floor_idx: int, budget: "_TokenBudgetTracker",
                             log, window_chars: int = BACKWARD_REFINE_WINDOW_CHARS,
                             max_iterations: int = BACKWARD_REFINE_MAX_ITERATIONS):
    """
    After the forward search finds a candidate start_line, double-check
    whether the TRUE start is actually earlier -- within this answer's own
    territory (never earlier than floor_idx, which is where the PREVIOUS
    answer's range ends, so this can never steal content from a
    different question).

    IMPORTANT: this walks BACKWARD across the ENTIRE gap [floor_idx,
    candidate_start), one window at a time, not just a small fixed
    lookback -- because the forward search can sometimes overshoot by a
    LOT (an entire page, or even an entire other question's worth of
    text), not just a few lines. A small fixed lookback would miss those
    bigger overshoots entirely (this was the confirmed cause of "an
    answer's whole starting page got skipped" and "the next question's
    full Q+A leaked into the end of this answer" -- the forward search
    found a real but much-too-late match, and the previous, narrow
    backward check couldn't see far back enough to catch it).

    Each time an earlier start is confirmed, we keep walking further
    back from THAT point (not stopping at the first improvement), since
    the true beginning could be even earlier still. We stop once a
    window reports no earlier start, or floor_idx / max_iterations is
    reached.
    """
    current_best = candidate_start
    scan_end = candidate_start
    iterations = 0

    while scan_end > floor_idx and iterations < max_iterations:
        window = []
        chars = 0
        idx = scan_end
        while idx >= floor_idx and (not window or chars + len(numbered_lines[idx][1]) <= window_chars):
            window.append(numbered_lines[idx])
            chars += len(numbered_lines[idx][1])
            idx -= 1
        window.reverse()

        if not window:
            break

        user_prompt = _build_backward_refine_prompt(window, question_text, ref_label)
        try:
            earlier_found, earlier_start = _call_groq_with_retries(
                client, BACKWARD_REFINE_SYSTEM_PROMPT, user_prompt,
                _parse_backward_refine_response, budget, log
            )
        except Exception as e:
            log(f"WARNING: backward-refine call failed for {ref_label}, keeping current best {current_best}: {e}")
            break

        iterations += 1

        if earlier_found and earlier_start is not None:
            valid_ids = {i for i, _ in window}
            if earlier_start in valid_ids and floor_idx <= earlier_start < current_best:
                log(f"  backward-refine moved {ref_label} start earlier: {current_best} -> {earlier_start} "
                    f"(iteration {iterations}, still checking further back...)")
                current_best = earlier_start
                scan_end = earlier_start - 1
                continue
            else:
                log(
                    f"WARNING: backward-refine for {ref_label} returned an out-of-range/non-improving "
                    f"line {earlier_start} -- stopping backward scan, keeping {current_best}"
                )
                break
        else:
            break  # no earlier start in this window -- current_best is the true beginning

    if current_best != candidate_start:
        log(f"  backward-refine final result for {ref_label}: {candidate_start} -> {current_best} "
            f"(scanned back {iterations} window(s), covering a gap of "
            f"{candidate_start - current_best} line(s))")

    return current_best


# =========================================================
# GROUPED RE-SPLIT for consecutive unmatched questions
# (fixes: "all sub-question answers collapsed under one sub-question",
#  and "one sub-question's answer bleeding into a sibling's")
# =========================================================

FORWARD_CHECK_SYSTEM_PROMPT = """You previously identified that a student's answer to a specific question begins at a certain line. Now double-check the OPPOSITE possibility: are these opening lines actually still finishing off a DIFFERENT, earlier topic/question, with the genuine start of THIS question's answer actually occurring a bit LATER?

This matters because a generic phrase or coincidental keyword overlap can sometimes trigger a false-positive match a line or two before the student has actually finished their previous point and moved on to this question -- which would incorrectly cut off the END of the PREVIOUS answer.

You are given:
1. The exact text of the target question.
2. A short window of text starting exactly at the previously-identified start line (inclusive).

Decide: within this window, does the answer to this exact question genuinely begin at the very FIRST line shown, or does it actually begin a bit LATER (because the first few lines here are still wrapping up a different point)?

Return ONLY valid JSON (no markdown fences, no commentary) in exactly one of these two shapes:

{"later_start_found": true, "start_line": 44}

or

{"later_start_found": false}

If later_start_found is true, start_line MUST be one of the exact line numbers shown in [brackets], and MUST be later than the first line shown. Only report a later line if you are confident the first few lines shown are genuinely NOT part of this answer -- if the previously-identified line already looks like the correct genuine start, return false."""


def _build_forward_check_prompt(window_lines: list, question_text: str, ref_label: str) -> str:
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in window_lines)
    return (
        f"TARGET QUESTION ({ref_label}): {question_text}\n\n"
        f"TEXT WINDOW starting at the previously-identified start line (line-numbered):\n{lines_block}"
    )


def _parse_forward_check_response(content: str) -> tuple:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}\nRaw (first 300 chars): {content[:300]!r}")

    if not isinstance(data, dict) or "later_start_found" not in data:
        raise ValueError(f"Response missing 'later_start_found' key. Got: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}")

    found = bool(data["later_start_found"])
    if not found:
        return False, None

    if "start_line" not in data:
        raise ValueError("Response has later_start_found=true but is missing 'start_line'")
    try:
        start_line = int(data["start_line"])
    except (ValueError, TypeError):
        raise ValueError(f"'start_line' must be an integer, got {data['start_line']!r}")

    return True, start_line


FORWARD_CHECK_WINDOW_LINES = 25  # kept deliberately small/conservative -- this check only exists to
                                   # recover a few stolen lines from the PREVIOUS answer, not to hunt
                                   # far ahead (that's the forward search's job, not this safety check)


def _refine_start_forward(client, numbered_lines: list, question_text: str, ref_label: str,
                            candidate_start: int, budget: "_TokenBudgetTracker", log,
                            window_lines: int = FORWARD_CHECK_WINDOW_LINES):
    """
    Symmetric counterpart to _refine_start_backward: checks whether the
    found start is actually a bit too EARLY (a false-positive match that
    is really still part of the PREVIOUS answer's closing lines), and if
    so, nudges it later.

    This directly targets the reported "an answer is missing its last
    2 lines" bug: those 2 lines weren't lost by THIS question's own
    processing -- they were stolen because the NEXT question's start was
    detected slightly too early. Since every question's end is computed
    as (next question's start - 1), correcting the next question's start
    here automatically restores the previous answer's true final lines.

    Deliberately conservative (small fixed window, single check, no
    iteration) -- the risk of drifting a start_line too far FORWARD is
    that it could eat into this question's OWN genuine content, which is
    a worse failure than leaving a small early-detection error
    uncorrected. This only recovers small (a handful of lines) errors,
    which matches what's been observed in practice.
    """
    window = [nl for nl in numbered_lines if candidate_start <= nl[0] < candidate_start + window_lines]
    if not window:
        return candidate_start

    user_prompt = _build_forward_check_prompt(window, question_text, ref_label)
    try:
        later_found, later_start = _call_groq_with_retries(
            client, FORWARD_CHECK_SYSTEM_PROMPT, user_prompt,
            _parse_forward_check_response, budget, log
        )
    except Exception as e:
        log(f"WARNING: forward-check call failed for {ref_label}, keeping start_line {candidate_start}: {e}")
        return candidate_start

    if later_found and later_start is not None:
        valid_ids = {i for i, _ in window}
        if later_start in valid_ids and later_start > candidate_start:
            log(f"  forward-check moved {ref_label} start later: {candidate_start} -> {later_start} "
                f"(recovers the previous answer's stolen final lines)")
            return later_start
        log(
            f"WARNING: forward-check for {ref_label} returned an out-of-range/non-improving "
            f"line {later_start} -- keeping {candidate_start}"
        )

    return candidate_start




REGROUP_SYSTEM_PROMPT = """You are given a SINGLE bounded block of a student's answer text (line-numbered) that is known to contain the answers to SEVERAL specific questions, back to back, possibly with NO clear individual restart labels between them (the student may have written them as one continuous flow, e.g. answering labeled sub-parts (i), (ii), (iii), (iv) of the same parent question without explicitly writing "Ans (ii)" etc. before each one).

You are given the exact text of EACH of these questions, each tagged with a REF label, in the SAME ORDER they should appear in the answer text.

Your task: find the line number where EACH question's individual answer begins within this block, in order. Since you are told these appear in this exact order and this block is scoped tightly around just their content, use topic/content shifts between consecutive questions' specific subject matter to locate each transition, even without explicit labels.

Rules:
- The FIRST question in the list starts at or very near the first line of this block (you may still be given a few lines of lead-in before it -- pick the actual content start).
- Every SUBSEQUENT question's answer starts strictly AFTER the previous one's start line.
- If you genuinely cannot distinguish a boundary for a particular question (the content is too blended to tell), it is better to OMIT that question from your output than to guess an arbitrary line -- an omitted question will simply be treated as part of the previous one's answer, which is a safer failure than a wrong split.
- Use the line numbers EXACTLY as shown in [brackets].

Return ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:

{
  "starts": [
    {"ref": "REF-1", "start_line": 100},
    {"ref": "REF-2", "start_line": 114}
  ]
}"""


def _build_regroup_prompt(window_lines: list, ref_question_pairs: list) -> str:
    questions_block = "\n".join(f"[{ref}] {q}" for ref, q in ref_question_pairs)
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in window_lines)
    return (
        f"QUESTIONS IN THIS BLOCK, IN ORDER:\n{questions_block}\n\n"
        f"ANSWER TEXT BLOCK (line-numbered):\n{lines_block}"
    )


def _parse_regroup_response(content: str) -> list:
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
        raise ValueError(f"'starts' must be a list, got: {type(starts).__name__}")

    result = []
    for item in starts:
        if not isinstance(item, dict) or "ref" not in item or "start_line" not in item:
            continue
        try:
            result.append({"ref": str(item["ref"]).strip().upper(), "start_line": int(item["start_line"])})
        except (ValueError, TypeError):
            continue
    return result


def _regroup_unmatched_run(client, numbered_lines: list, group_refs: list, group_questions: list,
                            block_start: int, block_end: int, budget: "_TokenBudgetTracker", log) -> dict:
    """
    Re-examines a bounded span of text [block_start, block_end] that is
    known to contain the answers to ALL of group_refs (in order), given
    to the LLM in ONE call with full sibling context. This is dramatically
    more reliable than independent blind sequential search for splitting
    un-labeled multi-part sub-questions, because the model sees the whole
    family at once instead of guessing one at a time with no visibility
    into where its siblings are.

    Returns {ref: start_line} for whichever refs it could confidently
    place (may be a subset of group_refs).
    """
    window = [nl for nl in numbered_lines if block_start <= nl[0] <= block_end]
    if not window:
        return {}

    ref_question_pairs = list(zip(group_refs, group_questions))
    user_prompt = _build_regroup_prompt(window, ref_question_pairs)

    log(f"  Re-examining block [lines {block_start}-{block_end}] to split {len(group_refs)} "
        f"question(s) that collapsed together: {group_refs}")

    try:
        starts = _call_groq_with_retries(
            client, REGROUP_SYSTEM_PROMPT, user_prompt, _parse_regroup_response, budget, log
        )
    except Exception as e:
        log(f"  WARNING: regroup call failed, leaving these questions merged: {e}")
        return {}

    valid_ids = {i for i, _ in window}
    result = {}
    for item in starts:
        ref, start_line = item["ref"], item["start_line"]
        if ref not in group_refs:
            continue
        if start_line not in valid_ids:
            log(f"  WARNING: regroup returned out-of-range start_line {start_line} for {ref}, discarding")
            continue
        result[ref] = start_line

    # Enforce monotonic order matching group_refs' known order -- if the
    # model returned something out of sequence, that's a sign it's
    # unreliable for this block, so drop anything that would create a
    # backwards or duplicate boundary rather than risk corrupting order.
    ordered_found = sorted(result.items(), key=lambda kv: group_refs.index(kv[0]))
    cleaned = {}
    last_start = block_start - 1
    for ref, start_line in ordered_found:
        if start_line > last_start:
            cleaned[ref] = start_line
            last_start = start_line
        else:
            log(f"  WARNING: regroup result for {ref} (line {start_line}) is not after the "
                f"previous sibling's start -- discarding to avoid corrupting order")

    log(f"  Regroup recovered {len(cleaned)} of {len(group_refs)} boundary(ies): {list(cleaned.keys())}")
    return cleaned


def map_answers_sequential(answer_lines: list, questions: list, status_callback=None,
                             answer_line_pages: list = None) -> list:
    """
    Default answer-mapping strategy. For each question in order:
      1. Search forward from wherever the previous question's answer was
         confirmed to start, for a line where THIS question's answer
         begins (overlapping-window search, see _find_answer_start_sequential).
      2. Run a backward-refinement pass on the found candidate to check
         whether the true start is actually a bit earlier (catches
         "answer picked up mid-paragraph, opening lines lost").
      3. Once found, the previous question's END is computed as
         (this start - 1) -- never asked of the LLM.

    After the main pass, a GROUPED RE-SPLIT step finds any run of
    consecutive questions that all collapsed under one earlier match
    (the classic un-labeled-multi-part-subquestion failure) and
    re-examines that whole span in ONE call with full sibling context to
    split them properly.
    """
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

    ref_to_question = {f"REF-{i+1}": q for i, q in enumerate(questions)}
    refs = list(ref_to_question.keys())
    found_starts = {}  # ref -> start_line
    pointer = 0

    for i, q in enumerate(questions):
        ref = f"REF-{i+1}"
        log(f"Searching for the start of {ref} ({q[:60]}...) from line {pointer} onward...")

        start_line = _find_answer_start_sequential(
            client, numbered_lines, q, ref, pointer, budget, log
        )

        if start_line is not None:
            refined_start = _refine_start_backward(
                client, numbered_lines, q, ref, start_line, pointer, budget, log
            )
            # Symmetric check: is this (possibly backward-refined) start
            # actually still a bit too EARLY -- i.e. a false-positive
            # match that's really the tail end of the PREVIOUS answer?
            # If so, nudge it later, which automatically recovers the
            # previous answer's true final lines (its end is computed as
            # this question's start - 1).
            final_start = _refine_start_forward(
                client, numbered_lines, q, ref, refined_start, budget, log
            )
            found_starts[ref] = final_start
            log(f"  found {ref} starting at line {final_start}")
            pointer = final_start + 1
        else:
            log(
                f"WARNING: could not find the start of {ref} anywhere from line {pointer} "
                f"to the end of the document ({total_lines} lines) -- marking as unmatched "
                f"for now (may be recovered by the grouped re-split pass below if it turns "
                f"out to be part of a multi-part question collapsed into a sibling's answer)."
            )

    # =====================================================================
    # GROUPED RE-SPLIT: find runs of consecutive UNMATCHED questions that
    # immediately follow a MATCHED one, and re-examine that whole span
    # (from the matched question's start to wherever the NEXT matched
    # question begins, or end of document) in a single call with full
    # sibling context. This is what fixes "all the sub-question answers
    # ended up under just one sub-question, the rest show as unmatched" --
    # independent blind search per sub-part is unreliable when the
    # student writes multiple labeled sub-parts as one continuous flow
    # with no individual restart markers, but a single call that SEES all
    # the sibling questions at once and is told they appear in this exact
    # order within this exact bounded span is far more reliable.
    # =====================================================================
    i = 0
    while i < len(refs):
        ref = refs[i]
        if ref not in found_starts:
            i += 1
            continue

        # collect the run of consecutive unmatched refs right after this one
        run = []
        j = i + 1
        while j < len(refs) and refs[j] not in found_starts:
            run.append(refs[j])
            j += 1

        if run:
            block_start = found_starts[ref]
            # block ends right before the NEXT matched question's start,
            # or at the end of the document if there isn't one
            next_matched_start = None
            for k in range(j, len(refs)):
                if refs[k] in found_starts:
                    next_matched_start = found_starts[refs[k]]
                    break
            block_end = (next_matched_start - 1) if next_matched_start is not None else (total_lines - 1)

            if block_end > block_start:
                group_refs = [ref] + run
                group_questions = [ref_to_question[r] for r in group_refs]
                recovered = _regroup_unmatched_run(
                    client, numbered_lines, group_refs, group_questions,
                    block_start, block_end, budget, log
                )
                # Only accept recovered starts for the RUN (not the
                # already-matched anchor ref, which stays as-is from the
                # original sequential search + backward refine)
                for r in run:
                    if r in recovered:
                        found_starts[r] = recovered[r]

        i = j

    # End of each answer = the next (in document order) confirmed
    # answer's start, minus one. Computed purely in Python -- never
    # asked of the LLM, so it can never be wrong in the way an
    # LLM-guessed end line could be.
    ordered = sorted(found_starts.items(), key=lambda kv: kv[1])
    ranges = []
    for idx, (ref, start) in enumerate(ordered):
        end = ordered[idx + 1][1] - 1 if idx + 1 < len(ordered) else total_lines - 1
        ranges.append({"ref": ref, "start_line": start, "end_line": end})

    log(f"Sequential mapping found {len(ranges)} of {len(questions)} question(s)")

    ranges_by_ref = {r["ref"]: r for r in ranges}
    results = []
    for i, q in enumerate(questions):
        ref = f"REF-{i+1}"
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


# Each pattern must match the ENTIRE stripped line (full-line anchors,
# not a bare .search() anywhere inside it) -- this is what makes it safe
# to use on ANY document/subject. A loose "contains this keyword"
# check would also catch genuine content lines that happen to start with
# a word like "Date" (e.g. "Date of the treaty was 1857...") or mention
# "signature" as a real vocabulary word, silently deleting real answer
# text -- a confirmed, reproducible cause of an answer missing its
# opening word/line when that word/line happened to be short.
NOISE_LINE_PATTERNS = [
    re.compile(r'^\s*(?:teacher\'?s?\s*|student\'?s?\s*)?signature\s*[:\-]?\s*(?:of\s+\w+)?\s*$', re.IGNORECASE),
    re.compile(r'^\s*page\s*no\.?\s*[:\-]?\s*\d*\s*$', re.IGNORECASE),
    re.compile(r'^\s*date\s*[:\-]?\s*[\d/\-\.\s]{0,15}$', re.IGNORECASE),
    re.compile(r'^\s*\d{1,3}\s*$'),  # bare page-number line
]

NOISE_LINE_MAX_CHARS = 40


def is_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r'^\s*\d{1,3}\s*$', stripped):
        return True  # bare page-number line -- always noise regardless of length
    if len(stripped) > NOISE_LINE_MAX_CHARS:
        return False
    return any(p.match(stripped) for p in NOISE_LINE_PATTERNS)


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
