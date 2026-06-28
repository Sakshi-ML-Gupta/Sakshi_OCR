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
                    f"[DIAGNOSTIC] Caught the 'os.PathLike, not tuple' error INSIDE "
                    f"{func.__name__}(), not before it -- this means the bug genuinely "
                    f"is somewhere in this module's call chain. "
                    f"file_input received: type={type(file_input).__name__}, "
                    f"repr={file_input!r}. Original error: {e}"
                ) from e
            raise

    return wrapper


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

    if resp.status_code == 403:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise Exception(
            f"Datalab OCR quota exhausted (403): {detail}\n"
            f"This is a Datalab BILLING limit, not a code bug -- retrying will not "
            f"help. Either add a payment method in your Datalab account to continue "
            f"on pay-as-you-go pricing, or wait for next month's free allowance to "
            f"reset."
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

QP_SYSTEM_PROMPT = """You are analyzing OCR text extracted from a scanned student exam assignment booklet (e.g. IGNOU-style, India). The booklet mixes pages of different kinds, in no guaranteed order:

1. ADMINISTRATIVE/COVER pages: enrolment number, programme code, learner name, registration details, regional centre info. NEVER question paper pages.
2. QUESTION PAPER pages: the official printed list of numbered exam questions the student must answer. These read as instructions/prompts DIRECTED AT the student (e.g. "Discuss X", "Explain Y with examples", "Write notes on the following:"). Mark allocations may appear (e.g. "10", "20").
3. ANSWER pages: the student's own (handwritten, OCR'd) answers. These are typically long, restate or reference a question briefly then write an extended response, and may themselves contain numbered or bulleted sub-points as part of the student's OWN explanation. These numbered sub-points inside a long answer are NOT separate exam questions, even though superficially they look similar (number, period, text) -- they are part of the answer to ONE question.

You are being shown only a PORTION of the document's pages at a time (a chunk), not the whole document. Some pages you see may be partial context carried over from a previous chunk -- still classify them normally based on their own content.

Your task: read the pages shown and return ONLY valid JSON (no markdown fences, no commentary, no explanation) in EXACTLY this shape:

{
  "question_paper_pages": [14, 16, 18],
  "questions": ["1. Example question text. (10)", "2. Another example question. (10)"]
}

IMPORTANT formatting requirement: "question_paper_pages" must be a JSON array where EACH page number is a SEPARATE element separated by commas, like [14, 16, 18] -- NEVER merge multiple page numbers into one number like [141618]. Each integer in that array must be a single, individually valid page number from the pages shown.

Critical rules for telling question-paper pages apart from answer pages that happen to contain numbered content:
- A genuine question paper question is a PROMPT directed at the student ("explain", "discuss", "describe", "write notes on", "compare", a question mark, etc.) -- it asks the student to DO something.
- A numbered point inside a long answer is typically a STATEMENT or FACT that is part of an explanation the student is giving -- it does not ask the reader to do anything; it's content, not an instruction.
- If a page's numbered items closely follow words like "उत्तर" (answer), "Ans", "Ans-", or come after a long paragraph of explanatory prose in the same block, that page is almost certainly an ANSWER page, not a question paper page -- exclude it from question_paper_pages even if it has multiple numbered lines.
- A real question paper question is typically a self-contained instruction (a prompt, maybe a mark allocation) -- this is a helpful general pattern, but NOT a hard rule on its own, since a real question can legitimately be long (multiple sub-parts, detailed multi-clause prompts).
- A student's answer commonly OPENS by restating the question itself before writing their real response (e.g. "Examine the theme of concealment in X. Discuss with reference to Y." followed by their own original explanation). Do NOT classify such a page as a question paper page just because its first sentence contains prompt-style verbs ("Examine", "Discuss") -- look at what comes AFTER that first sentence: if it continues into the student's own developing argument or analysis (not further instructions to the reader), it is an ANSWER page. This is a CONTENT signal, not a length signal -- do not use page length by itself to decide.
- If the SAME question text appears on two different pages, and one page is part of a concise, structured list of several distinct questions while the other page contains only that one question's wording followed by extended original prose, the latter is the student's answer-opening restatement -- exclude it.
- When genuinely uncertain whether a page is a question paper page, prefer NOT including it as one, and prefer NOT extracting its numbered items as separate questions.
- If NONE of the pages shown in this chunk are question paper pages, return empty lists for both fields -- that is a valid and expected result for chunks that only contain answer/admin pages.
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
    """Rough estimate using the current chars-per-token ratio."""
    return int(len(text) / CHARS_PER_TOKEN_ESTIMATE) + 1


# =========================================================
# Rolling token-budget tracker
# =========================================================

class _TokenBudgetTracker:
    """
    FIX: uses a sliding-window event log for accurate token tracking.
    """
    def __init__(self, tpm_limit=TPM_LIMIT, safety_fraction=TPM_SAFETY_FRACTION):
        import collections
        self.tpm_limit = tpm_limit
        self.safe_limit = tpm_limit * safety_fraction
        self.events = collections.deque()  # (timestamp, tokens)

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


# Parses Groq's rate-limit message
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
    """
    Generic Groq chat-completion caller with full retry/pacing/error
    handling.
    """
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
                f"st.secrets?\n"
                f"  2. Does the key have any extra whitespace, quotes, or "
                f"a line break copied in by accident?\n"
                f"  3. Has the key been revoked or rotated in your Groq "
                f"console (https://console.groq.com/keys)?\n"
                f"  4. If using st.secrets, did you restart the Streamlit "
                f"app after adding/changing the secret?\n"
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
                    f"https://console.groq.com/settings/billing."
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

    ratio_based_match = False
    if ratio >= 0.90:
        words1 = sorted(set(re.findall(r'[a-z]{3,}', k1)))
        words2 = sorted(set(re.findall(r'[a-z]{3,}', k2)))
        if not words1 or not words2:
            ratio_based_match = ratio >= 0.92
        else:
            matched = sum(1 for w1 in words1 if any(_words_nearly_match(w1, w2) for w2 in words2))
            overlap = matched / max(len(words1), len(words2))
            ratio_based_match = overlap >= 0.92

    if ratio_based_match:
        return True

    def _word_in_other(w, other_words):
        return any(_words_nearly_match(w, ow) for ow in other_words)

    dwords1 = _distinctive_words(q1, max_words=30)
    dwords2 = _distinctive_words(q2, max_words=30)
    if dwords1 and dwords2:
        missing_from_2 = [w for w in dwords1 if not _word_in_other(w, dwords2)]
        missing_from_1 = [w for w in dwords2 if not _word_in_other(w, dwords1)]
        shorter_len = min(len(dwords1), len(dwords2))

        if shorter_len < 6:
            if len(missing_from_2) == 0 or len(missing_from_1) == 0:
                return True
        else:
            contained_in_2 = (len(dwords1) - len(missing_from_2)) / len(dwords1)
            contained_in_1 = (len(dwords2) - len(missing_from_1)) / len(dwords2)
            if contained_in_2 >= 0.85 or contained_in_1 >= 0.85:
                return True

    return False


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
            f"page(s) (questions from this stage are discarded -- see stage 2 below)"
        )
        chunk_results.append((qp_pages_1based, []))

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

    qp_pages_1based_merged, _ = _merge_chunk_results(chunk_results)
    qp_page_indices_0based = sorted(pn - 1 for pn in qp_pages_1based_merged)

    log(f"Question paper pages identified: {len(qp_page_indices_0based)} page(s)")

    if len(qp_page_indices_0based) >= 2:
        qp_page_lengths = [
            (i, len(pages[i]["raw_text"])) for i in qp_page_indices_0based
        ]

        def _true_median(values):
            s = sorted(values)
            n = len(s)
            mid = n // 2
            return (s[mid - 1] + s[mid]) / 2 if n % 2 == 0 else s[mid]

        for page_idx, length in qp_page_lengths:
            other_lengths = [l for i, l in qp_page_lengths if i != page_idx]
            if not other_lengths:
                continue
            baseline = _true_median(other_lengths)
            if length > max(baseline * 3, 800):
                log(
                    f"WARNING: page {page_idx + 1} was classified as a question "
                    f"paper page but is {length} chars long -- much longer than "
                    f"the typical {baseline:.0f} chars for this document's other "
                    f"question paper pages. This commonly means the page is "
                    f"actually the OPENING of a student's answer (where they "
                    f"restated the question before writing their real response), "
                    f"which would cause that answer's first page to be silently "
                    f"excluded. Check page {page_idx + 1} in the OCR output if an "
                    f"answer appears to be missing its beginning."
                )

    qp_pages_full = [pages[i] for i in qp_page_indices_0based]
    questions = extract_canonical_questions(qp_pages_full, status_callback)

    log(
        f"Final result: {len(qp_page_indices_0based)} question paper "
        f"page(s), {len(questions)} canonical question(s)"
    )

    return qp_page_indices_0based, questions


# =========================================================
# LLM-BASED ANSWER MAPPING (Groq)
# =========================================================

ANSWER_MAP_SYSTEM_PROMPT = """You are analyzing a student's handwritten answers (OCR'd) from an exam assignment booklet. You are given:
1. A numbered list of the OFFICIAL exam questions, each tagged with a reference label like [REF-A], [REF-B], etc.
2. The student's answer text, with each line prefixed by its line number in [brackets].

Your task: for EACH official question, find WHERE in the answer text the student's response to that specific question starts and ends, and return the LINE NUMBER RANGE (inclusive) for each, identified by its REF label.

Important guidance for finding boundaries correctly:
- A new answer typically begins where the student restates or references a question (e.g. "Ans 5-", "उत्तर 6-", "प्र. 8", a question number, or a clear topic shift matching the next question's subject).
- CRITICAL -- introductory lines before the first numbered point: a student's answer frequently opens with 2-4 lines of general, introductory prose BEFORE reaching their first specific numbered point, sub-heading, or detailed argument (e.g. defining a general concept before listing specific examples). This introduction often does NOT explicitly restate the question's exact topic words. Do NOT mistake the first numbered point (e.g. "1.") or first detailed sub-heading for the TRUE start of the answer -- look BACKWARD from that point to check whether the immediately preceding lines are still part of the SAME train of thought (general scene-setting that leads into it), rather than belonging to a different, earlier question. If the preceding lines do not look like they belong to a different question (no restatement of a different topic, no different question's distinctive content), include them as part of THIS answer's start, even though they don't contain an obvious "start marker" themselves.
- An answer's content ends at the LAST line that is still part of that answer's reasoning/explanation, RIGHT BEFORE the next answer begins (whether or not the next answer is in your list of official questions).
- CRITICAL -- ambiguous boundaries between adjacent sub-parts (e.g. (क)/(ख)/(ग)/(घ) or (i)/(ii)/(iii)): when you cannot find a clear marker for where one labeled sub-part's answer ends and the next begins, do NOT default to including everything up to the next REF's start line as a fallback -- this routinely causes one sub-part's content to "bleed" into the next, mixing genuinely distinct paragraphs that belong to different sub-answers. Instead, look for a CONTENT-level shift: a new sentence that introduces a different specific concept, term, or sub-topic than what was just being discussed is a much more reliable boundary than line proximity alone. If you truly cannot distinguish where one sub-part ends and the next begins even by content, it is better to end the range slightly EARLIER (a shorter but cleanly correct answer) than to extend it through content that may belong to the next sub-part.
- If a question's answer is genuinely not present anywhere in the text shown, do NOT invent a range -- omit that REF entirely from your output. It may appear in a different chunk of the document.
- Each REF's range must NOT overlap with another REF's range. If you are unsure exactly where one answer ends and the next begins, prefer ending the EARLIER answer sooner rather than letting it swallow content that belongs to a later answer -- a short correct answer is far more useful than a long answer that incorrectly absorbed unrelated content.
- Use the line numbers EXACTLY as given in [brackets] -- do not estimate, guess, or renumber.
- Use the EXACT REF label (e.g. "REF-A") to identify each question. Do NOT retype or paraphrase the question text itself -- the REF label is all that's needed.

Return ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:

{
  "answers": [
    {"ref": "REF-A", "start_line": 12, "end_line": 18},
    {"ref": "REF-B", "start_line": 19, "end_line": 25}
  ]
}

If NONE of the official questions' answers appear in the text shown, return {"answers": []} -- that is a valid and expected result for a chunk that doesn't contain any of these answers."""


def _build_answer_map_user_prompt(numbered_lines: list, questions: list) -> str:
    """
    Enhanced to handle sub-part questions better by grouping them.
    """
    # Group questions by parent number for sub-parts
    grouped_questions = []
    current_group = []
    parent_prefix = None
    
    # Pattern for sub-part questions: 9.(क), 9.(ख), etc.
    sub_part_pattern = re.compile(r'^(\d+)\.\s*\(([क-घa-d])\)')
    
    for i, q in enumerate(questions):
        match = sub_part_pattern.match(q.strip())
        if match:
            question_num = match.group(1)
            if parent_prefix is None:
                parent_prefix = question_num
            if parent_prefix == question_num:
                current_group.append((i, q))
            else:
                if current_group:
                    grouped_questions.append(current_group)
                current_group = [(i, q)]
                parent_prefix = question_num
        else:
            if current_group:
                grouped_questions.append(current_group)
                current_group = []
                parent_prefix = None
            grouped_questions.append([(i, q)])
    
    if current_group:
        grouped_questions.append(current_group)
    
    # Build the prompt with grouped sub-parts
    questions_block_parts = []
    for group in grouped_questions:
        if len(group) > 1:
            # This is a multi-part question with sub-parts
            parent_idx, parent_q = group[0]
            questions_block_parts.append(f"[REF-{chr(65+parent_idx)}] {parent_q}")
            for idx, q in group[1:]:
                questions_block_parts.append(f"  [REF-{chr(65+idx)}] {q}")
        else:
            idx, q = group[0]
            questions_block_parts.append(f"[REF-{chr(65+idx)}] {q}")
    
    questions_block = "\n".join(questions_block_parts)
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in numbered_lines)
    
    return (
        f"OFFICIAL QUESTIONS (each tagged with its own [REF-X] label -- "
        f"use the REF label, not retyped question text, to identify which "
        f"question an answer belongs to):\n{questions_block}\n\n"
        f"STUDENT'S ANSWER TEXT (line-numbered):\n{lines_block}\n\n"
        f"IMPORTANT: For multi-part questions (like 9.(क), 9.(ख), 9.(ग), 9.(घ)), "
        f"each sub-part has its own REF label. Map each sub-part's answer separately "
        f"based on the specific sub-part content, not the parent question."
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
    r'^\s*(?:Ans(?:wer)?\s*\d+\s*[.:\-]?\s*'
    r'|Ans(?:wer)?\s*[.:\-]\s*'
    r'|उत्तर\s*\d*\s*[\-\:]\s*'
    r'|प्र[०.\s]+\d+[.\s:-]*'
    r'|प्रश्न[.\s]+\d+[.\s:-]*'
    r'|Q\.?\s*\d+\s*[.:\-]\s*)',
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
    'identify', 'following', 'with', 'reference', 'comment',
}


def _distinctive_words(text: str, max_words: int = 20) -> list:
    words = re.findall(r'[a-z]{3,}', _normalize_for_overlap_match(text))[:max_words]
    return sorted(set(w for w in words if w not in _QUESTION_STOPWORDS))


def _line_starts_new_answer_for_question(line: str, questions: list, min_fraction: float = 0.4) -> int:
    """
    Enhanced to better detect sub-part answer starts.
    """
    # First check for explicit label matches
    label_match = _ANSWER_START_RE.match(line)
    if label_match:
        num_match = re.search(r'\d+', label_match.group(0))
        if num_match:
            label_num = num_match.group(0)
            # Check for sub-part pattern in the line
            sub_match = re.search(r'[\(（]([क-घa-d])[\)）]', line, re.IGNORECASE)
            sub_label = sub_match.group(1) if sub_match else None
            
            for i, q in enumerate(questions):
                q_num_match = re.match(r'\s*(\d+)', q)
                if q_num_match and q_num_match.group(1) == label_num:
                    # Check if this is a sub-part question
                    q_sub_match = re.search(r'[\(（]([क-घa-d])[\)）]', q, re.IGNORECASE)
                    if q_sub_match and sub_label:
                        if q_sub_match.group(1).lower() == sub_label.lower():
                            return i
                    elif not q_sub_match and not sub_label:
                        return i
            return -1
        return -1

    # Content-based matching
    line_words = sorted(set(re.findall(r'[a-z]{3,}', _normalize_for_overlap_match(line))[:25]))
    if not line_words:
        return None

    # Check if the line contains sub-part indicators
    has_sub_part = bool(re.search(r'[\(（]([क-घa-d])[\)）]', line, re.IGNORECASE))
    
    for i, q in enumerate(questions):
        q_distinctive = _distinctive_words(q)
        if not q_distinctive:
            continue
            
        matched = sum(
            1 for w in q_distinctive
            if any(_words_nearly_match(w, lw) for lw in line_words)
        )
        
        # Check if this question is a sub-part
        q_has_sub_part = bool(re.search(r'[\(（]([क-घa-d])[\)）]', q, re.IGNORECASE))
        
        # If both have sub-parts, check if they match
        if has_sub_part and q_has_sub_part:
            q_sub = re.search(r'[\(（]([क-घa-d])[\)）]', q, re.IGNORECASE)
            line_sub = re.search(r'[\(（]([क-घa-d])[\)）]', line, re.IGNORECASE)
            if q_sub and line_sub and q_sub.group(1).lower() != line_sub.group(1).lower():
                continue
        
        def _required_matches(n_distinctive, fraction=min_fraction):
            if n_distinctive <= 2:
                return n_distinctive
            return max(2, round(n_distinctive * fraction))

        if matched >= _required_matches(len(q_distinctive)):
            return i

    return None


def _chunk_lines_by_char_budget(numbered_lines: list, questions: list,
                                  max_chars: int = ANSWER_MAP_MAX_CHARS_PER_CHUNK,
                                  absolute_max_chars: int = ANSWER_MAP_ABSOLUTE_MAX_CHARS) -> list:
    """
    Enhanced chunking that respects sub-part structure.
    """
    if not numbered_lines:
        return []

    # Build a map of which questions are sub-parts of the same parent
    question_groups = {}
    sub_part_pattern = re.compile(r'^(\d+)\.\s*\(([क-घa-d])\)')
    
    for i, q in enumerate(questions):
        match = sub_part_pattern.match(q.strip())
        if match:
            parent_num = match.group(1)
            question_groups.setdefault(parent_num, []).append(i)
    
    # Create a priority map: sub-parts of the same question should stay together
    chunk_priority = {}
    for parent, indices in question_groups.items():
        for idx in indices:
            chunk_priority[idx] = parent
    
    chunks = []
    current_chunk = []
    current_chars = 0
    past_target = False
    current_question_idx = None
    current_question_group = None
    
    pending_new_start_idx = None

    for idx, text in numbered_lines:
        line_chars = len(text)

        if current_chunk and current_chars + line_chars > max_chars:
            past_target = True

        matched_q_idx = _line_starts_new_answer_for_question(text, questions)
        
        has_formal_label = bool(_ANSWER_START_RE.match(text))
        is_new_index = matched_q_idx is not None and (
            matched_q_idx == -1 or matched_q_idx != current_question_idx
        )

        # Check if we're transitioning between sub-parts of the same parent
        is_same_group = False
        if matched_q_idx is not None and current_question_idx is not None:
            if matched_q_idx in chunk_priority and current_question_idx in chunk_priority:
                if chunk_priority[matched_q_idx] == chunk_priority[current_question_idx]:
                    is_same_group = True

        if has_formal_label and is_new_index:
            is_genuine_new_start = True
            pending_new_start_idx = None
        elif is_new_index and not is_same_group:
            if pending_new_start_idx == matched_q_idx:
                is_genuine_new_start = True
                pending_new_start_idx = None
            else:
                is_genuine_new_start = False
                pending_new_start_idx = matched_q_idx
        else:
            is_genuine_new_start = False
            pending_new_start_idx = None

        should_break_at_answer_start = past_target and is_genuine_new_start
        should_force_break_absolute = (
            current_chunk and current_chars + line_chars > absolute_max_chars
        )

        if should_break_at_answer_start or should_force_break_absolute:
            chunks.append(current_chunk)
            current_chunk = []
            current_chars = 0
            past_target = False

        if is_genuine_new_start and matched_q_idx != -1:
            current_question_idx = matched_q_idx
            current_question_group = chunk_priority.get(matched_q_idx)

        current_chunk.append((idx, text))
        current_chars += line_chars

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


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
    r'^\s*(?:Ans(?:wer)?\s*\d+\s*[.:\-]?\s*'
    r'|Ans(?:wer)?\s*[.:\-]\s*'
    r'|उत्तर\s*\d*\s*[\-\:]\s*'
    r'|प्र[०.\s]+\d+[.\s:-]*'
    r'|प्रश्न[.\s]+\d+[.\s:-]*'
    r'|Q\.?\s*\d+\s*[.:\-]\s*)',
    re.IGNORECASE
)


def strip_question_restatement(answer_text: str) -> str:
    """
    Enhanced to handle sub-part question restatements.
    """
    text = answer_text
    
    # Pattern for sub-part labels like "9.(क)", "9.(ख)", etc.
    sub_part_pattern = re.compile(
        r'^\s*(?:प्र\.?\s*)?\s*\d+\s*\.\s*[\(（][क-घa-d][\)）]\s*[:\-]?\s*',
        re.IGNORECASE
    )
    
    # Strip up to 3 times to handle nested labels
    for _ in range(3):
        # Try to strip sub-part label first
        new_text = sub_part_pattern.sub('', text, count=1).strip()
        if new_text != text:
            text = new_text
            continue
            
        # Try the original patterns
        new_text = QUESTION_PREFIX_RE.sub('', text, count=1).strip()
        if new_text == text:
            break
        text = new_text
    
    return text


_PARENT_INSTRUCTION_PREFIX_RE = re.compile(
    r'^\s*\d+[\.\)]?\s*(?:\([ivx]+\)|\([a-z]\)|\([क-घ]\))?\s*'
    r'(?:identify and explain the following|write (?:short )?notes? on|'
    r'comment on|explain the following|discuss the following)\s*:?\s*',
    re.IGNORECASE
)


def strip_full_question_echo(answer_text: str, question_text: str) -> str:
    """
    Enhanced to better handle sub-part question echoes.
    """
    # Extract the core of the sub-part question (remove parent prefix)
    question_core = question_text
    
    # For sub-parts, extract the actual sub-part text
    sub_part_match = re.match(
        r'^\d+\.\s*[\(（]([क-घa-d])[\)）]\s*(.*?)(?:\s*\d+\s*)?$',
        question_text.strip(),
        re.IGNORECASE | re.DOTALL
    )
    
    if sub_part_match:
        # The sub-part is the actual content after the label
        question_core = sub_part_match.group(2).strip()
        if not question_core:
            question_core = question_text
    
    # Remove parent instruction prefixes
    question_core = _PARENT_INSTRUCTION_PREFIX_RE.sub('', question_core).strip()
    
    if not question_core:
        question_core = question_text

    q_norm = _normalize_for_overlap_match(question_core)
    q_word_count = len(q_norm.split())
    if q_word_count == 0:
        return answer_text

    answer_words = answer_text.split()
    if not answer_words:
        return answer_text

    min_n = max(3, int(q_word_count * 0.6))
    max_n = min(len(answer_words), int(q_word_count * 1.4) + 3)

    best_strip_count = 0
    best_ratio = 0.0

    for n in range(min_n, max_n + 1):
        prefix = " ".join(answer_words[:n])
        prefix_norm = _normalize_for_overlap_match(prefix)
        ratio = difflib.SequenceMatcher(None, prefix_norm, q_norm).ratio()
        
        # Slightly lower threshold for sub-parts
        threshold = 0.70 if sub_part_match else 0.75
        
        if ratio >= threshold and ratio > best_ratio:
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
    chunks = _chunk_lines_by_char_budget(numbered_lines, questions)
    log(f"Split {len(answer_lines)} answer line(s) into {len(chunks)} LLM chunk(s) for answer mapping")

    all_ranges = []
    chunk_failures = []
    chunk_zero_matches = 0

    for i, chunk in enumerate(chunks):
        line_range = f"{chunk[0][0]}-{chunk[-1][0]}" if chunk else "empty"
        log(f"Asking LLM to map answers in chunk {i+1}/{len(chunks)} (lines {line_range})...")

        user_prompt = _build_answer_map_user_prompt(chunk, questions)
        try:
            chunk_ranges = _call_groq_with_retries(
                client, ANSWER_MAP_SYSTEM_PROMPT, user_prompt,
                _parse_answer_map_llm_response, budget, log
            )
        except Exception as e:
            log(f"WARNING: chunk {i+1}/{len(chunks)} answer-mapping failed, skipping: {e}")
            chunk_failures.append(str(e))
            continue

        if not chunk_ranges:
            chunk_zero_matches += 1

        valid_indices = {idx for idx, _ in chunk}
        min_idx, max_idx = min(valid_indices), max(valid_indices)
        for r in chunk_ranges:
            if r["ref"] not in ref_to_question:
                log(f"WARNING: discarding answer mapping with unknown ref {r['ref']!r}")
                continue
            if min_idx <= r["start_line"] <= max_idx and min_idx <= r["end_line"] <= max_idx:
                all_ranges.append(r)
            else:
                log(
                    f"WARNING: discarding out-of-range answer mapping for "
                    f"{r['ref']}: lines {r['start_line']}-{r['end_line']} "
                    f"outside this chunk's range {min_idx}-{max_idx}"
                )

        log(f"Chunk {i+1}/{len(chunks)}: mapped {len(chunk_ranges)} answer(s)")

    best_by_ref = {}
    for r in all_ranges:
        existing = best_by_ref.get(r["ref"])
        if existing is None or (r["end_line"] - r["start_line"]) > (existing["end_line"] - existing["start_line"]):
            best_by_ref[r["ref"]] = r

    deduped_ranges = list(best_by_ref.values())
    resolved_ranges = _resolve_overlapping_answer_ranges(deduped_ranges)

    log(f"Final answer mapping: {len(resolved_ranges)} of {len(questions)} question(s) matched")

    if not resolved_ranges:
        if chunk_failures and len(chunk_failures) == len(chunks):
            raise Exception(
                f"Answer mapping failed: ALL {len(chunks)} chunk(s) raised an "
                f"error (none succeeded). First failure: {chunk_failures[0]}"
            )
        elif chunk_zero_matches == len(chunks):
            sample_lines = [l for l in answer_lines[:15] if l.strip()][:8]
            raise Exception(
                f"Answer mapping found ZERO matches across all {len(chunks)} chunk(s), "
                f"even though the LLM calls themselves succeeded. This usually means "
                f"the 'answer pages' passed in do NOT actually contain the student's "
                f"answers -- most likely the question-paper/answer-page page split "
                f"upstream misclassified pages. Sample of the answer text actually searched: "
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


# =========================================================
# FIND QUESTION BOUNDARIES IN ANSWER PAGES -- similarity based
# (Kept for backward compatibility)
# =========================================================

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
    """
    Enhanced to handle sub-part questions properly.
    """
    total_chars = sum(len(l) for l in answer_lines)
    
    # Count actual answer content (exclude noise and extremely short lines)
    meaningful_lines = [l for l in answer_lines if len(l.strip()) > 20 and not is_noise(l)]
    meaningful_chars = sum(len(l) for l in meaningful_lines)
    
    # For sub-part questions, we need less text per "question"
    min_plausible_chars_per_question = 100
    
    avg_chars_per_question = meaningful_chars / max(num_questions, 1)
    
    if avg_chars_per_question < min_plausible_chars_per_question:
        log(
            f"WARNING: 'answer pages' contain only {meaningful_chars} meaningful characters "
            f"for {num_questions} question(s) (~{avg_chars_per_question:.0f} chars/question). "
            f"This may indicate misclassification or missing answer content."
        )
        return False
    return True


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

    answer_page_indices = [i for i in range(len(pages)) if i not in qp_page_indices]
    answer_pages = [pages[i] for i in answer_page_indices]

    log(f"Answer pages: {[i+1 for i in answer_page_indices]}")

    answer_lines = []
    for page in answer_pages:
        for line in page["raw_text"].split("\n"):
            if not is_noise(line):
                answer_lines.append(line)

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

    log("Mapping each question to its answer independently (LLM-based)...")
    qa_map = map_answers_with_llm(answer_lines, official_questions, status_callback)

    matched_count = sum(1 for q in official_questions if q in qa_map)
    log(f"Matched {matched_count} of {len(official_questions)} questions")

    for q in official_questions:
        if q not in qa_map:
            log(f"WARNING: No match found for: {q[:60]}")

    if not qa_map:
        raise Exception(
            "Could not match any questions to answers.\n"
            f"Official questions: {official_questions}\n"
            f"First 10 answer lines: {answer_lines[:10]}"
        )

    qa_pairs = []
    for q in official_questions:
        qa_pairs.append({
            "question": q,
            "answer": qa_map.get(q, ""),
            "matched": q in qa_map,
        })

    log(f"Done -- {len(qa_pairs)} Q-A pairs ({matched_count} matched)")

    return ocr_json, qa_pairs


def save_outputs(ocr_json: dict, qa_pairs: list, output_dir: str = ".",
                  base_name: str = "document") -> tuple:
    """
    Convenience helper: writes the two requested output files to disk
    and returns their paths.
    - {base_name}_ocr.json: the complete raw OCR of the whole PDF
    - {base_name}_qa_pairs.json: the mapped question -> answer pairs
    """
    ocr_path = os.path.join(output_dir, f"{base_name}_ocr.json")
    qa_path = os.path.join(output_dir, f"{base_name}_qa_pairs.json")

    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(ocr_json, f, ensure_ascii=False, indent=2)

    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    return ocr_path, qa_path
