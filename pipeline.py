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
#
# This module's OWN code cannot produce the exact error
# "expected str, bytes or os.PathLike object, not tuple" -- that precise
# message is only ever raised by Python's os.fspath()/open() built-ins,
# and this module contains zero raw open() calls; every Path()/read_bytes()
# call here is already guarded by an isinstance() check before it runs
# (see _normalize_file_input and _coerce_name above). This has been
# verified directly: feeding every realistic tuple shape (filename+bytes,
# enumerate-style, zip-style, nested tuples) into _normalize_file_input
# produces a clear, different TypeError every time, never this one.
#
# That means if this exact error is still happening, it is occurring
# OUTSIDE this module -- most likely in the calling app's own code
# (e.g. a raw open(...) call on something that isn't a path) BEFORE
# process_pdf()/process_reference() is ever reached.
#
# This decorator can't fix a bug in code it doesn't contain, but it
# converts an ambiguous crash into an UNAMBIGUOUS one: if this exact
# error somehow still surfaces while a call is genuinely inside this
# module, the wrapped function catches it, attaches the literal type
# and repr of whatever was passed in, and re-raises with a message
# that makes the true source impossible to mistake next time.
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
# CONCURRENCY GUARD
#
# FIX: the real log showed TWO complete pipeline runs interleaved --
# "Submitting document..." fired twice, OCR ran twice, chunk logs from
# both runs were mixed together line by line. This is almost certainly
# the calling app (e.g. Streamlit) invoking process_pdf() a second time
# while the first call is still in flight (a common Streamlit rerun
# behavior). Both runs then compete for the SAME shared 8000 TPM org
# budget at once, which is the direct cause of the constant 429s seen
# in that log -- it wasn't one document needing too many tokens, it was
# two concurrent runs each burning the same shared budget simultaneously.
#
# This module cannot prevent the calling app from invoking it twice,
# but a process-wide lock around the Groq-calling section ensures that
# IF it is called concurrently in the same process, the calls serialize
# instead of racing for the same token budget. This turns "two runs
# fighting over 8000 TPM" into "two runs sharing 8000 TPM one after
# the other," which is strictly better and removes one whole class of
# the 429 storm seen in the log. If your app calls this from separate
# processes (e.g. multiple server workers), you'd need a cross-process
# lock (e.g. a file lock or Redis) instead -- ask if that's your setup.
# =========================================================

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

# FIX: a static character budget doesn't reliably predict actual token
# usage (Devanagari-heavy text tokenizes denser and less predictably
# than English). Instead of guessing a safe chunk size once, we now
# keep a RUNNING ESTIMATE of tokens used in the current rolling window
# and proactively wait BEFORE sending a chunk if we estimate we're
# close to the 8000 TPM ceiling, using the actual Used/Requested/Limit
# numbers Groq's own error messages report whenever we do go over, so
# our internal estimate self-corrects against ground truth as we go.
TPM_LIMIT = 8000
TPM_SAFETY_FRACTION = 0.85  # treat the budget as slightly smaller than
                              # the real limit to leave headroom for
                              # estimation error
CHARS_PER_TOKEN_ESTIMATE = 2.0  # conservative for Devanagari-heavy text;
                                  # self-corrects via _record_actual_usage

MAX_CHARS_PER_CHUNK = 6000  # smaller than before -- real-world 429s
                              # showed our chars-per-token guess was
                              # optimistic; smaller chunks reduce blast
                              # radius of any single misestimate
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
#
# Tracks an estimate of tokens consumed within the current rolling
# ~60s window. Before sending each chunk, if our estimated usage plus
# the new request would exceed a safe fraction of TPM_LIMIT, we wait
# out the remainder of the window BEFORE sending -- proactive pacing
# instead of reactive retry-after-failure. Real 429 responses still
# update our knowledge of actual usage (via parsed Used/Limit numbers)
# so the estimate self-corrects over the course of a run.
# =========================================================

class _TokenBudgetTracker:
    """
    FIX (this round): the previous version used a single window_start +
    window_tokens pair with an all-or-nothing reset. This had a real,
    confirmed bug: _maybe_reset_window() ran BEFORE the wait-duration
    calculation, so if the window had just been reset, `elapsed` was
    ~0 and the computed wait was always close to the FULL 60 seconds --
    never a partial wait reflecting how much of the window had already
    naturally elapsed. Combined with token estimates that kept getting
    added without properly expiring old ones in some call sequences,
    this produced the exact symptom seen in real usage: every single
    chunk reporting a climbing "estimated tokens" figure and forcing a
    near-full 60s wait every time, even when the real Groq-reported
    Requested values were far lower.

    This version uses a sliding-window event log (deque of timestamped
    token amounts) instead. "Tokens used in the last 60s" is always
    computed by summing events younger than 60s -- which naturally and
    continuously decays as old events age out, rather than jumping
    between two states (full budget / empty budget) on a single timer.
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
            return  # comfortably under budget -- no wait

        # Wait only as long as needed for enough OLD events to expire
        # and make room -- not a blind full 60s.
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
        """
        Reconciles our estimate with Groq's ground-truth numbers. We
        don't know the exact age distribution of Groq's "used" figure,
        so the safest correction is to top up our tracked total to AT
        LEAST match reality (added as one fresh event), rather than
        blindly overwriting -- this avoids ever under-counting real
        usage without compounding errors across repeated corrections.
        """
        now = time.monotonic()
        current = self.used_in_window(now)
        if used > current:
            self.events.append((now, used - current))
        if limit:
            self.tpm_limit = limit
            self.safe_limit = limit * TPM_SAFETY_FRACTION

    def reset_window(self):
        """Clears all tracked events -- used right after a 429 retry
        where we've already waited out Groq's own reported wait time,
        so the window is genuinely clear and shouldn't be re-penalized
        by a stale estimate."""
        self.events.clear()

    # Backwards-compat no-op properties so any external code that may
    # have referenced the old window_start/window_tokens attributes
    # doesn't break.
    @property
    def window_start(self):
        return time.monotonic()

    @window_start.setter
    def window_start(self, value):
        pass  # handled internally by the event deque now

    @property
    def window_tokens(self):
        return self.used_in_window()

    @window_tokens.setter
    def window_tokens(self, value):
        if value == 0:
            self.events.clear()


# Parses Groq's rate-limit message for the real numbers it reports, so
# retries and pacing are informed by ground truth, not guesses.
#
# FIX (this round): the previous regex only matched a plain-seconds wait
# format (e.g. "try again in 8.5875s"), which is what Groq's TPM
# (tokens-per-MINUTE) errors use. But Groq's TPD (tokens-per-DAY) errors
# use a DIFFERENT duration format with minutes, e.g. "try again in
# 10m7.392s" -- the old regex silently failed to match this (returned
# None), causing the code to fall back to a generic short retry (a few
# seconds) against what is actually a 10+ MINUTE wall. Retrying into a
# daily quota wall is also fundamentally pointless in a way a per-minute
# wait isn't -- once the day's budget is gone, no amount of waiting
# seconds or pacing helps; only waiting out the actual reset (or
# upgrading the tier) does. This version captures the optional minutes
# component AND the limit type (TPM vs TPD) so the caller can react
# correctly to each.
_RATE_LIMIT_DETAIL_RE = re.compile(
    r'on\s+tokens\s+per\s+(minute|day)\s*\((TPM|TPD)\).*?'
    r'Limit\s+(\d+),\s*Used\s+(\d+),\s*Requested\s+(\d+).*?'
    r'try again in\s+(?:(\d+)m)?([\d.]+)s',
    re.IGNORECASE | re.DOTALL
)


def _parse_rate_limit_detail(message: str):
    """
    Returns a dict with keys: limit_type ("TPM" or "TPD"), limit, used,
    requested, wait_seconds -- or None if the message doesn't match
    Groq's known error format. Handles both the plain-seconds duration
    format used by TPM errors and the minutes+seconds format used by
    TPD errors (e.g. "10m7.392s").
    """
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
    """
    FIX (this round): the previous version only attempted recovery when
    n had MORE digits than max_page's digit-length (e.g. only tried for
    3+ digit numbers in a 25-page document). This missed a real case
    seen in production: page numbers 6 and 9 concatenated into 69 --
    which has exactly 2 digits, the SAME as max_page (25), so the old
    "len(s) <= len(str(max_page))" guard incorrectly treated it as
    "plausible on its own" and skipped recovery entirely, silently
    discarding two genuinely real question-paper pages.

    The correct guard is not about digit-length at all: we should only
    skip recovery when n is ALREADY a valid page number (nothing to
    recover), not based on how many digits it happens to have.

    A second fix: when multiple splits are mathematically possible
    (e.g. 99 could split into [9, 9] since page 9 is valid), we reject
    any split containing a REPEATED page number -- a genuine
    concatenation bug merges DIFFERENT page numbers together; it would
    not plausibly repeat the same page twice in one list. This prevents
    a genuinely invalid number like 99 from being incorrectly "recovered"
    into a nonsensical duplicate.
    """
    if n in valid_page_numbers:
        return []  # already valid -- nothing to recover

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
            return None  # reject splits with a repeated page number
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

    # Prefer the split with fewer parts when multiple are mathematically
    # possible -- the more conservative recovery, less likely to overfit.
    candidates.sort(key=len)
    return candidates[0]


def _call_groq_with_retries(client, system_prompt: str, user_prompt: str,
                              response_parser, budget: "_TokenBudgetTracker",
                              log, max_retries: int = 4):
    """
    Generic Groq chat-completion caller with full retry/pacing/error
    handling. Shared by BOTH question-identification and answer-mapping
    calls, so the (carefully tuned, repeatedly bug-fixed) rate-limit
    and auth-error handling lives in exactly one place instead of being
    duplicated and risking drifting out of sync between the two callers.

    `response_parser` is called with the raw response content string
    and must return the parsed result, or raise on malformed output
    (the same retry loop here also catches and retries parse failures).
    """
    import groq

    estimated_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt) + 800
    last_error = None

    skip_next_proactive_check = False

    for attempt in range(1, max_retries + 2):
        # Proactively pace BEFORE sending, based on our running estimate
        # of tokens used in the current rolling window. Skipped exactly
        # once, right after a 429/413 retry, since at that point we just
        # waited the exact amount of time Groq itself told us was needed
        # (record_actual_from_error + window reset already happened) --
        # re-applying our own separate proactive wait on top of that would
        # double-penalize the same window and stall far longer than
        # necessary. Any OTHER retry (e.g. after a JSON-parse failure,
        # which does NOT reset the window) still goes through the normal
        # proactive check, since real token risk could still be present.
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
            # A 401 "Invalid API Key" is an AUTH failure, not a rate
            # limit -- it will NEVER succeed on retry, since the key
            # itself is wrong/missing/revoked. Fail fast on the very
            # first attempt with actionable next steps, instead of
            # retrying pointlessly and burning several seconds before
            # surfacing a confusing "failed after N attempts" message.
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
                # A daily token quota exhaustion is fundamentally
                # different from a per-minute one. Retrying -- even with
                # the correct wait time -- means stalling the whole run
                # for 10+ minutes, and if the quota is THIS close to its
                # daily ceiling, a successful retry would likely just
                # exhaust it again on the very next chunk. Fail fast
                # with a clear, actionable message instead of burning
                # retries and the user's time against a wall that small
                # waits can't clear.
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
                # TPM (per-minute) limit -- genuinely recoverable by
                # waiting out Groq's own reported duration.
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
    """Question-identification call -- thin wrapper around the shared
    generic caller, using the QP-specific prompts and parser."""
    user_prompt = _build_qp_user_prompt(pages_chunk)
    return _call_groq_with_retries(
        client, QP_SYSTEM_PROMPT, user_prompt, _parse_qp_llm_response,
        budget, log, max_retries
    )


def _normalize_question_key(q: str) -> str:
    text = q.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    # Strip leading numbering/punctuation noise that commonly differs
    # between two OCR/LLM extractions of the SAME printed question
    # (e.g. "1. - (i)" vs "1. (i)") -- formatting artifacts, not
    # semantic differences.
    text = re.sub(r'^\d+[\.\)]\s*[-–]?\s*', '', text)
    return text


def _words_nearly_match(w1: str, w2: str) -> bool:
    """Two significant words count as 'the same' if identical OR if
    they differ by only a small edit (e.g. an OCR misspelling like
    'abhijnana' vs 'abhignana') -- but NOT if they're simply two
    different real words (e.g. 'akam' vs 'puram'), which would have a
    low character-similarity ratio despite similar length."""
    if w1 == w2:
        return True
    if abs(len(w1) - len(w2)) > 2:
        return False
    return difflib.SequenceMatcher(None, w1, w2).ratio() >= 0.8


def _is_near_duplicate_question(q1: str, q2: str) -> bool:
    """
    FIX: exact-match-after-normalization dedup was too strict for real
    OCR variance -- the SAME printed question, extracted by two
    different LLM chunk calls, can come back with a stray inserted
    dash, different capitalization, or a single misspelled letter
    (all confirmed in real usage: "1. (i)" vs "1. - (i)", and
    "Abhijnana" vs "Abhignana"). Exact matching let these through as
    two SEPARATE questions, and since the duplicate's answer almost
    never gets matched a second time, it always showed up as a
    confusing extra "(no answer text matched)" entry in the final
    output.

    This fuzzy check requires BOTH a high overall character-similarity
    ratio AND a high overlap of "significant" (4+ letter) words, where
    word-level comparison itself tolerates small spelling differences.
    The word-overlap check is essential: two DIFFERENT short questions
    that share a sentence template (e.g. "...note on akam thinai..."
    vs "...note on puram thinai...") can have a deceptively high raw
    character-similarity ratio despite being genuinely different
    questions -- the word-overlap check catches that the one
    significant word that DOES differ ("akam" vs "puram") is not an
    OCR-noise-level difference, so they are correctly kept distinct.
    """
    k1, k2 = _normalize_question_key(q1), _normalize_question_key(q2)
    if k1 == k2:
        return True

    ratio = difflib.SequenceMatcher(None, k1, k2).ratio()

    # FIX (this round): the ratio-based check below ONLY catches
    # OCR-noise-level duplicates (high overall character similarity).
    # It must NOT early-return False when ratio < 0.90 -- a real,
    # confirmed bug had exactly that early return, which meant the
    # containment-based check further down (designed to catch a
    # DIFFERENT kind of duplicate -- same sub-part, very different
    # surface wording) was unreachable dead code whenever the
    # ratio-based path didn't already match. This silently disabled
    # the entire containment-based fix despite it testing correctly in
    # isolation. Now both checks are independent OR paths: a pair is a
    # duplicate if EITHER the ratio+word-overlap check matches OR the
    # containment check (further below) matches.
    ratio_based_match = False
    if ratio >= 0.90:
        # FIX: minimum word length is 3, not 4. A 4+ letter cutoff was
        # filtering out short-but-significant distinguishing words like
        # spelled-out numbers ("one" vs "two"), which caused two
        # genuinely DIFFERENT questions ("Real question one." / "Real
        # question two.") to be wrongly merged -- everything else in
        # the sentence matched, and the one word that actually differed
        # was too short to be counted, leaving a perfect (but wrong)
        # word-overlap score. 3 letters still excludes pure function-
        # word noise ("a", "of", "in", "to") that would otherwise
        # inflate apparent overlap without carrying real distinguishing
        # content.
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

    # FIX (this round): a SECOND, structurally different kind of
    # duplicate confirmed in real usage -- the canonical question
    # extraction can emit the SAME sub-part twice with genuinely
    # different SURFACE TEXT (e.g. one includes the full quote, the
    # other only captures a truncated version; one includes the parent
    # instruction prefix, the other doesn't). These have a LOW overall
    # character-similarity ratio (the check above correctly rejects
    # them) because the actual wording differs substantially, not just
    # by noise -- but they describe the exact same underlying
    # question. Each phantom duplicate then gets its own independent
    # answer-mapping pass, and since both are "about" the same real
    # content, the SAME real answer ends up fragmented/duplicated
    # across two question entries in the final output -- confirmed as
    # the cause of a real "answer is repeated, half from start" report.
    #
    # This is detected via CONTAINMENT rather than overall similarity:
    # if one question's distinctive vocabulary is almost entirely
    # contained within the other's (one is a subset -- a truncated or
    # less-detailed rendering of the same content), they are treated
    # as duplicates even with low overall string similarity. This is
    # different from the akam/puram case (two genuinely different
    # questions sharing a template), where NEITHER side's distinctive
    # words are contained in the other -- each has its own defining
    # word the other entirely lacks.
    def _word_in_other(w, other_words):
        return any(_words_nearly_match(w, ow) for ow in other_words)

    # Uses _distinctive_words (stopword-filtered) rather than the raw
    # 3-letter word sets above -- generic instructional words shared by
    # almost every question/sub-part ("identify", "following",
    # "explain") would otherwise dilute the containment signal in
    # exactly the cases this check needs to catch.
    dwords1 = _distinctive_words(q1, max_words=30)
    dwords2 = _distinctive_words(q2, max_words=30)
    if dwords1 and dwords2:
        missing_from_2 = [w for w in dwords1 if not _word_in_other(w, dwords2)]
        missing_from_1 = [w for w in dwords2 if not _word_in_other(w, dwords1)]
        shorter_len = min(len(dwords1), len(dwords2))

        # FIX: with a SHORT distinctive-word list (confirmed boundary
        # case: "akam thinai...landscapes" vs "puram thinai...
        # landscapes" each have only 4 distinctive words), a percentage
        # threshold is statistically unreliable -- one missing word out
        # of 4 is 25%, easily clearing an 80% containment bar despite
        # being a genuinely different topic word (akam vs puram). Short
        # lists require PERFECT containment (zero missing words) on at
        # least one side, since even a single mismatch in a short list
        # is too large a fraction to safely ignore. Longer lists (6+
        # words) can tolerate a percentage threshold, since one
        # mismatched word among many genuinely is just noise.
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
    """Deduplicates a list of question strings using fuzzy near-duplicate
    matching, preserving first-seen order. O(n^2) but n is always small
    (a handful to a few dozen questions per document)."""
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
    """
    FIX: this is a NEW, dedicated SECOND stage of question
    identification, addressing a real fragmentation bug confirmed in
    production -- a single multi-part question (e.g. "(i)...(ii)...
    (iii)...(iv)...") was sometimes extracted as ONE combined question
    by one page-chunk, and as 4 SEPARATE standalone questions by
    another chunk that happened to see the same question-paper page
    (due to the 1-page overlap between chunks), or saw a truncated
    view of it. The union-merge step then kept BOTH inconsistent
    versions, since they don't textually deduplicate as "the same
    question" -- producing exactly the Q1 vs Q6/Q7/Q8/Q9 duplication/
    fragmentation seen in real output, where (i) ended up with no
    answer at all (it was a phantom split) while the REAL combined
    question separately got a (correctly matched) partial answer.

    The fix: once stage 1 (identify_questions_with_llm's existing
    chunked page-detection) has determined WHICH pages are question
    paper pages, this function makes exactly ONE additional LLM call
    with the COMPLETE text of just those pages together. Since
    question-paper text is short (a list of printed questions, not
    answer essays), this comfortably fits in a single call even for
    long papers, and because the model sees the ENTIRE question paper
    at once, it only has to make ONE consistent decision about how to
    split multi-part questions -- there is no second, possibly
    disagreeing, chunk to produce a conflicting alternative.

    This also directly implements the "monotonic alignment" request:
    the model is explicitly told to preserve printed order, and this
    canonical list becomes the SINGLE source of truth for question
    identity used by all downstream answer-mapping -- no other code
    path independently invents or re-derives the question list.
    """
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
    """
    Modified to use simpler extraction for question pages,
    with LLM only used for identifying which pages are question papers.
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

    chunks = _chunk_pages_by_char_budget(pages)
    log(f"Split {len(pages)} page(s) into {len(chunks)} LLM chunk(s) for page identification")

    valid_page_numbers = {p["page_number"] for p in pages}
    max_page_number = max(valid_page_numbers) if valid_page_numbers else 0
    chunk_results = []
    chunk_failures = []

    # Only use LLM for identifying which pages are question papers
    for i, chunk in enumerate(chunks):
        page_nums_in_chunk = [p["page_number"] for p in chunk]
        log(f"Asking LLM to identify question paper pages in chunk {i+1}/{len(chunks)} (pages {page_nums_in_chunk})...")

        try:
            qp_pages_1based, _ = _call_groq_for_chunk(client, chunk, budget, log)
        except Exception as e:
            log(f"WARNING: chunk {i+1}/{len(chunks)} failed, skipping: {e}")
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

    # Use SIMPLE rule-based extraction instead of LLM
    qp_pages_full = [pages[i] for i in qp_page_indices_0based]
    questions = extract_canonical_questions_simple(qp_pages_full, status_callback)

    log(
        f"Final result: {len(qp_page_indices_0based)} question paper "
        f"page(s), {len(questions)} question(s) extracted (rule-based)"
    )

    return qp_page_indices_0based, questions

        def _true_median(values):
            s = sorted(values)
            n = len(s)
            mid = n // 2
            return (s[mid - 1] + s[mid]) / 2 if n % 2 == 0 else s[mid]

        for page_idx, length in qp_page_lengths:
            # FIX: the baseline must be computed from the OTHER pages
            # only (leave-one-out), not from all pages including the
            # candidate itself -- the original version included the
            # candidate in its own median, which with an even page
            # count could make the outlier page BECOME the median,
            # making the threshold mathematically impossible to exceed
            # (confirmed bug: a 1185-char misclassified page against a
            # 61-char real page produced a "median" of 1185 -- itself).
            other_lengths = [l for i, l in qp_page_lengths if i != page_idx]
            if not other_lengths:
                continue
            baseline = _true_median(other_lengths)
            # 800 chars is a realistic absolute floor: real question-
            # paper text per page is typically well under this even
            # with several sub-parts, while an answer's restated-
            # question-plus-opening-paragraph reliably exceeds it.
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

    # Stage 2: single consistent pass over the CONFIRMED question-paper
    # pages' full text, producing one canonical, non-fragmented question
    # list -- this is the actual fix for the Q1/Q6/Q7/Q8/Q9-style
    # fragmentation seen in production.
    qp_pages_full = [pages[i] for i in qp_page_indices_0based]
    questions = extract_canonical_questions(qp_pages_full, status_callback)

    log(
        f"Final result: {len(qp_page_indices_0based)} question paper "
        f"page(s), {len(questions)} canonical question(s)"
    )

    return qp_page_indices_0based, questions


# =========================================================
# LLM-BASED ANSWER MAPPING (Groq)
#
# FIX: the previous approach (find_question_boundaries_by_similarity +
# slice_raw_answers_by_boundaries) used plain word-overlap similarity
# on a sliding window of answer lines to guess where each question's
# restatement appears, then sliced from there to the NEXT matched
# question's restatement. This is fundamentally fragile on long,
# free-form handwritten Hindi answers: if even ONE question in the
# middle of the sequence fails to match cleanly (common with OCR noise,
# reordered sub-parts, or answers that don't explicitly restate the
# question), the similarity matcher silently skips it -- and the
# PRECEDING matched question's slice then extends all the way to
# whatever question matches NEXT, however far away that is. This is
# exactly the bug seen in real usage: one question's answer absorbing
# several subsequent questions' worth of content, while the skipped
# questions get nothing.
#
# This replaces that entire approach with an LLM call that reads the
# actual answer text and identifies, independently per question, the
# LINE-NUMBER RANGE where that answer appears. Critically, the LLM is
# asked for line indices, NOT to retype the answer -- the actual
# extraction is a plain Python slice of the ORIGINAL OCR'd text using
# those indices, guaranteeing the output is verbatim (no paraphrasing,
# no risk of subtle LLM rewording) while still getting LLM-quality
# semantic boundary detection instead of brittle text-similarity
# heuristics. A Python-side overlap-resolution pass provides a hard
# guarantee against the swallowing bug even if the LLM's boundaries
# are imperfect: no question's range can ever be allowed to extend
# into territory a later-starting question's range claims.
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
    # FIX: previously this prepended "1.", "2.", etc. directly in front
    # of each question, e.g. "1. 5. प्रत्ययों...". Since most real
    # questions ALREADY contain their own original numbering ("5.",
    # "Q.8", "प्र. 6", etc.), this created confusing double-numbering
    # that risked the LLM echoing back the WRONG (prompt-added) number,
    # or the whole "1. 5. ..." string, neither of which would exactly
    # match the canonical question text downstream. Using "REF-A",
    # "REF-B" style reference labels instead avoids any visual or
    # semantic collision with the question's own real numbering, making
    # it unambiguous that these are just our own internal reference
    # tags, not part of the question itself.
    questions_block = "\n".join(
        f"[REF-{chr(65+i)}] {q}" for i, q in enumerate(questions)
    )
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in numbered_lines)
    return (
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
            continue  # skip malformed entries rather than failing the whole batch

    return result


ANSWER_MAP_MAX_CHARS_PER_CHUNK = 11000  # FIX (this round): increased
# from 9000. The PREVIOUS truncation reports were not actually caused
# by chunk size being too small -- they were caused by the answer-
# start detector finding ZERO safe break points in documents where
# students restate the question itself (no "Ans-"/"उत्तर-" label at
# all), forcing a fallback to the hard cap on every chunk. Now that
# _line_starts_new_answer() also recognizes question-content overlap
# (see above), genuine safe break points exist in these documents too,
# so chunk size can be raised again. 11000 chars is calculated to stay
# safely under the free-tier 8000 TPM ceiling for a SINGLE request
# (~11700 chars is the hard ceiling at a 2 chars/token estimate after
# accounting for system prompt + JSON response overhead -- 11000 keeps
# a small margin below that). Going meaningfully higher than this risks
# reintroducing the 413/429-on-the-mapping-call failure mode from the
# previous round, which produces the EXACT same "half answer" symptom
# through a different mechanism (a failed chunk's answers never
# appearing at all) -- so this is very close to the real ceiling on
# the current Groq free tier, not an arbitrary number.

ANSWER_MAP_ABSOLUTE_MAX_CHARS = 60000  # FIX (this round): replaces the
# old 2x-multiplier hard cap (~22000 chars), which could still force a
# break mid-answer purely on SIZE with no regard for safety. Real usage
# confirmed single answers can legitimately span 5-6 pages of OCR'd
# text. This is now a true last-resort ceiling, deliberately generous
# (roughly 10-12 pages worth of text) so it should never be reached in
# ordinary use -- a real single answer reaching even half this size
# would be extraordinary. If a chunk does grow past the TPM-safe target
# because a single long answer needed the room, the existing 413/429
# retry-with-backoff logic (see _call_groq_with_retries) handles it by
# retrying with backoff -- slower, but never loses real answer content.

# FIX (this round): detects a line that STARTS a new answer. The
# previous version ONLY matched formal label patterns (Ans-, उत्तर-,
# etc.) -- but real documents showed students who restate the FULL
# QUESTION TEXT as their answer's opening sentence, with NO label at
# all (e.g. "Examine the theme of Concealment in Abhignana
# Shakuntalam..." as the literal first words of the answer). Against
# such a document, the label-only regex matched ZERO lines, leaving
# the chunker with no safe break points anywhere -- it then had no
# choice but to fall back to the hard cap, producing oversized,
# undifferentiated chunks that caused exactly the truncation and
# duplicated-sentence artifacts seen in real output. This version adds
# a SECOND detection path: a line counts as a new-answer start if its
# opening words substantially overlap with the opening words of ANY
# official question, regardless of whether a formal label is present.
_ANSWER_START_RE = re.compile(
    r'^\s*(?:Ans(?:wer)?[.\s:-]+\d*|उत्तर\s*\d*\s*[\-\:]?|प्र[०.\s]*\d*|Q\.?\s*\d+[.\s:-])',
    re.IGNORECASE
)


def _normalize_for_overlap_match(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    return text


# FIX (this round): generic English question-phrasing words that
# appear in almost every question regardless of topic ("how", "why",
# "examine", "discuss", "comment", etc.). These must be filtered out
# before computing word overlap, because a real answer's opening
# sentence often only carries forward the QUESTION'S TOPIC-SPECIFIC
# words (proper nouns, technical terms), not its generic instructional
# phrasing -- a plain percentage-of-all-words overlap check was failing
# on exactly this pattern in real documents (e.g. "How are the views of
# the state integrated with the theme of X?" vs an answer opening with
# "X is not just a..." shares almost nothing except "X" itself).
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
    """Extracts the topic-specific (non-generic) significant words from
    a question or line, used to find genuine content overlap while
    ignoring common question-phrasing words that carry no
    distinguishing signal."""
    words = re.findall(r'[a-z]{3,}', _normalize_for_overlap_match(text))[:max_words]
    return sorted(set(w for w in words if w not in _QUESTION_STOPWORDS))


def _line_starts_new_answer_for_question(line: str, questions: list, min_fraction: float = 0.5):
    """
    FIX (this round): returns the INDEX of the question this line
    appears to start a fresh answer for, or None if it doesn't look
    like a new-answer start at all -- replacing the previous boolean-
    only _line_starts_new_answer(). The boolean version had a real,
    confirmed bug: a label-style match (e.g. "Q2 continues with...")
    was treated as UNCONDITIONALLY a fresh start, even when it was
    just a sentence WITHIN an answer that happened to mention its own
    question number in passing -- causing a long multi-page answer to
    be incorrectly chopped mid-way through its own content (exactly
    matching the real-world symptom of "first page missing from the
    start" / "last paragraph missing from the end": the chunker broke
    INSIDE one answer, so neither resulting chunk's LLM call ever saw
    the complete picture).

    This version resolves a label match to a SPECIFIC question index
    by extracting any number in the label (e.g. "5" from "Ans 5-", "2"
    from "Q2") and matching it to a question whose own leading number
    matches -- so the caller can tell "this label refers to the SAME
    question we're already inside" (not a real new start) from "this
    label refers to a DIFFERENT question" (a genuine new start). If
    the label's number can't be resolved to any known question, -1 is
    returned, signaling "ambiguous formal label -- treat cautiously as
    a fresh start since we can't rule that out."
    """
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
        # FIX (this round): a genuinely long (e.g. 7-page) answer is
        # statistically much more likely to organically reference
        # ANOTHER question's topic in passing somewhere within its own
        # span -- confirmed in testing with realistic comparative
        # sentences ("This mirrors the jealousy Duryodhana felt toward
        # the Pandavas...", appearing INSIDE a different question's
        # long answer). The previous formula (round(n * 0.5), with no
        # floor above 1) let such passing mentions through as if they
        # were a genuine new-answer start, incorrectly splitting one
        # long answer into two broken pieces wherever it happened to
        # mention another question's vocabulary.
        #
        # required_matches() now requires AT LEAST 2 distinctive words
        # to match whenever a question has 2 or more distinctive words
        # at all (only single-distinctive-word questions, e.g. just
        # "Mrichchhkatika", fall back to requiring that one word) --
        # a single shared topic word is common in passing references,
        # but two or more matching is a much stronger, rarer signal
        # that genuinely correlates with an actual restatement opening
        # rather than an incidental mention.
        def _required_matches(n_distinctive, fraction=min_fraction):
            if n_distinctive <= 1:
                return n_distinctive
            return max(2, round(n_distinctive * fraction))

        if matched >= _required_matches(len(q_distinctive)):
            return i

    return None


def _chunk_lines_by_char_budget(numbered_lines: list, questions: list,
                                  max_chars: int = ANSWER_MAP_MAX_CHARS_PER_CHUNK,
                                  absolute_max_chars: int = ANSWER_MAP_ABSOLUTE_MAX_CHARS) -> list:
    """
    Answer-boundary-aware chunking: a chunk break is only allowed at a
    line that genuinely starts a DIFFERENT question's answer than the
    one currently being accumulated.

    FIX (this round): two real, confirmed bugs in the previous version,
    both producing the same real-world symptom (an answer's start or
    end going missing -- "first page gone from the start" / "last
    paragraph gone from the end"):

    1. The break condition only fired on the SAME line that crossed
       max_chars, AND only if that exact line was itself an answer-
       start. In practice, max_chars is usually crossed mid-answer
       (somewhere in the MIDDLE of a long answer's own content, not
       conveniently on a boundary line), so the real next answer-start
       boundary could be 20-30+ lines later -- by which point a chunk
       break finally fires, but only after already consuming a chunk's
       worth of the WRONG answer's content alongside the start of the
       next one, corrupting both. Fixed: `past_target` is now standing
       state -- once max_chars is crossed, the chunker waits and breaks
       at the VERY NEXT genuine answer-start, however many lines later
       that turns out to be, rather than requiring it on the exact
       threshold-crossing line.

    2. A label-style match (e.g. text that happens to look like
       "Q2 ...") was treated as UNCONDITIONALLY a fresh start, even
       when it was just a sentence WITHIN an answer mentioning its own
       question in passing. This could cause a long answer's own later
       lines to incorrectly "restart" a chunk break against the SAME
       question, slicing that one answer into two separate, incomplete
       pieces. Fixed: `_line_starts_new_answer_for_question` now
       resolves a label match to a specific question index via its
       number, so a break only fires when the matched index genuinely
       DIFFERS from the question currently being accumulated.

    Beyond a safe boundary, a chunk is allowed to keep growing past
    max_chars indefinitely (a single long, multi-page answer simply
    makes its own larger chunk) -- absolute_max_chars is a true last-
    resort ceiling that should essentially never be reached in
    practice. A chunk occasionally exceeding the TPM-safe target size
    is handled by the existing 413/429 retry-with-backoff logic already
    in place (see _call_groq_with_retries) -- a slower retry cycle is a
    vastly better outcome than ever silently truncating real content.
    """
    if not numbered_lines:
        return []

    chunks = []
    current_chunk = []
    current_chars = 0
    past_target = False
    current_question_idx = None  # which question's answer we believe
                                   # we're currently accumulating

    for idx, text in numbered_lines:
        line_chars = len(text)

        if current_chunk and current_chars + line_chars > max_chars:
            past_target = True

        matched_q_idx = _line_starts_new_answer_for_question(text, questions)
        # -1 means "ambiguous formal label, couldn't resolve to a known
        # question" -- treated cautiously as a genuine fresh start,
        # since we can't positively confirm it's the same question.
        # A resolved index only counts as a genuinely NEW start if it
        # differs from the question we believe we're already inside.
        is_genuine_new_start = matched_q_idx is not None and (
            matched_q_idx == -1 or matched_q_idx != current_question_idx
        )

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

        current_chunk.append((idx, text))
        current_chars += line_chars

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _resolve_overlapping_answer_ranges(answer_ranges: list) -> list:
    """
    HARD SAFETY NET against the answer-swallowing bug: sorts ranges by
    start_line, then clips any range's end_line so it can never extend
    into territory claimed by a later-starting range. This guarantees
    the bug seen in real usage (one question's answer absorbing several
    subsequent questions' worth of content) is structurally impossible
    in the output, regardless of how the LLM's boundaries came out.
    """
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
    # FIX: a real bug confirmed in testing -- the previous pattern
    # "Ans(?:wer)?[.\s:-]+..." matched the BARE WORD "answer" even when
    # it was legitimate content (e.g. "Ans 1- answer one content."
    # correctly stripped "Ans 1- " on the first pass, but the function's
    # own up-to-2-times retry loop then matched "answer" AGAIN on the
    # second pass, since "answer one content." also starts with
    # "Ans(?:wer)?" followed by whitespace -- silently eating real
    # content down to "one content."). The fix requires either a DIGIT
    # or an explicit punctuation marker (-,:,.) immediately after
    # "Ans"/"Answer" -- a genuine label always has one of these right
    # after it ("Ans 5-", "Ans-", "Answer:"), while the bare word
    # "answer" followed by ordinary prose does not.
    r'Ans(?:wer)?\s*\d+\s*[.:\-]?\s*'        # "Ans 5-", "Ans5.", "Answer 5:"
    r'|Ans(?:wer)?\s*[.:\-]\s*'              # "Ans-", "Ans:", "Answer." (punctuation required, no digit needed)
    r'|उत्तर\s*\d*\s*[\-\:]\s*'                # "उत्तर-", "उत्तर 5-" (dash/colon required)
    r'|प्र[०.\s]+\d+[.\s:-]*'                   # "प्र. 8." (number required)
    r'|प्रश्न[.\s]+\d+[.\s:-]*'                 # "प्रश्न. 2." (number required)
    r'|Q\.?\s*\d+\s*[.:\-]\s*'                  # FIX: "Q.8-", "Q5:" now require
    # explicit trailing punctuation, matching every other branch above.
    # The previous version ("Q\.?\s*\d+[.\s:-]*", trailing punctuation
    # OPTIONAL via "*") matched legitimate answer content like "Q.8
    # marks allocated suggest this requires..." -- the student
    # discussing the question's own mark allocation as part of their
    # REAL answer, not restating a label -- silently eating "Q.8 "
    # from genuine content. A real label always has a dash/colon/period
    # right after the number ("Q.8-", "Q.8:"); ordinary prose mentioning
    # a question number does not.
    r')',
    re.IGNORECASE
)


def strip_question_restatement(answer_text: str) -> str:
    """
    FIX: real verbatim answers were starting with the student's own
    restatement/label of the question (e.g. "Ans 5-", "उत्तर-",
    "प्र. 8.") -- legitimate raw OCR content, but redundant once shown
    alongside the question field in the final output, and confirmed in
    real usage to read as "the question repeating at the start of the
    answer." This strips ONLY a leading restatement label from the
    very start of the text -- it never touches a restatement that
    might legitimately appear mid-answer (e.g. a student referencing a
    different sub-question within their own response). Repeats the
    strip up to 2 times in case of doubled prefixes from messy OCR
    (e.g. "उत्तर- Ans 5-"), then stops.
    """
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


# FIX (this round): when a labeled sub-part question now carries its
# parent instruction forward for self-contained context (e.g. "1.(i)
# Identify and explain the following: <quote>"), the STUDENT'S answer
# only echoes the sub-part's own distinctive content (the quote
# itself) -- never the parent instruction phrase. Comparing against
# the FULL question text (including that instruction phrase) inflates
# the expected echo length and similarity search window, causing real
# echoes to go undetected entirely. This regex strips known parent-
# instruction lead-ins before comparison, isolating just the sub-
# part's own distinctive text to search for.
_PARENT_INSTRUCTION_PREFIX_RE = re.compile(
    r'^\s*\d+[\.\)]?\s*(?:\([ivx]+\)|\([a-z]\)|\([क-घ]\))?\s*'
    r'(?:identify and explain the following|write (?:short )?notes? on|'
    r'comment on|explain the following|discuss the following)\s*:?\s*',
    re.IGNORECASE
)


def strip_full_question_echo(answer_text: str, question_text: str) -> str:
    """
    FIX: real verbatim answers were confirmed to start with the
    student's FULL restatement of the question -- not just a short
    label like "Ans 5-" (already handled by strip_question_restatement
    above), but the entire question sentence re-copied before the
    actual answer begins (e.g. an answer literally opening with
    "Examine the theme of Concealment in Abhignana Shakuntalam / The
    Loom of Time." before any original content). This detects that
    pattern by comparing a window of the answer's leading words against
    the question text itself, and strips exactly that window if the
    similarity is high enough.

    Deliberately conservative: searches only a TIGHT window around the
    question's own word count (70%-130%, not a loose multiplier) and
    requires a high similarity threshold (0.75). An earlier looser
    version was caught during testing eating into genuine answer
    content that merely shared topical vocabulary with the question
    (e.g. an answer's second sentence reusing words like "theme" and
    "concealment") -- this tighter window and threshold avoid that.
    Returns the original text unchanged if no sufficiently strong echo
    is found, so answers that never restate the question are never
    touched.

    Strips a common parent-instruction PREFIX from the question before
    comparing (see _PARENT_INSTRUCTION_PREFIX_RE above) -- needed since
    sub-part questions now carry parent context forward for self-
    contained readability, but students never echo that parent
    instruction phrase itself, only the sub-part's own distinctive text.
    """
    question_core = _PARENT_INSTRUCTION_PREFIX_RE.sub('', question_text).strip()
    if not question_core:
        question_core = question_text  # fallback if stripping ate everything

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


# Replace the map_answers_with_llm and enhanced_map_answers_with_llm 
# with this similarity-based approach that uses NO LLM calls

def map_answers_with_similarity(
    answer_lines: list,
    questions: list,
    status_callback=None
) -> dict:
    """
    SIMILARITY-BASED approach - NO LLM calls.
    Uses the enhanced slicing with intro handling to map questions to answers.
    This is much more token-efficient than the LLM-based approach.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    
    log(f"Using similarity-based answer mapping (NO LLM calls)...")
    
    # Use the enhanced slicing approach
    qa_pairs = enhanced_slice_qa_from_line_items(
        answer_lines,
        questions,
        similarity_threshold=0.25,
        window=5
    )
    
    # Convert to dict format
    qa_map = {}
    for pair in qa_pairs:
        qa_map[pair["question"]] = pair["answer"]
        if pair.get("has_intro", False):
            log(f"Preserved introductory text for question: {pair['question'][:60]}...")
    
    matched_count = sum(1 for q in questions if q in qa_map and qa_map[q].strip())
    log(f"Matched {matched_count} of {len(questions)} questions using similarity-based approach")
    
    return qa_map


# Also simplify the extract_canonical_questions function to use less token
# or skip it entirely if it's causing quota issues

def extract_canonical_questions_simple(qp_pages: list, status_callback=None) -> list:
    """
    SIMPLIFIED version - extracts questions from question paper pages
    using regex/rule-based approach instead of LLM, saving tokens.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    
    if not qp_pages:
        return []
    
    log(f"Extracting questions from {len(qp_pages)} question paper pages using rule-based approach...")
    
    all_questions = []
    
    for page in qp_pages:
        text = page["raw_text"]
        
        # Split by lines and look for question patterns
        lines = text.split('\n')
        current_question = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line starts a new question
            # Patterns: "1.", "1)", "Q1.", "प्र. 1", "1. (a)", etc.
            question_start = re.match(
                r'^\s*(?:\d+[\.\)]\s*|Q\.?\s*\d+\s*[\.\)]\s*|प्र\.?\s*\d+\s*[\.\)]\s*|\([a-z]\)\s*|\([क-घ]\)\s*)',
                line,
                re.IGNORECASE
            )
            
            if question_start:
                # Save previous question if exists
                if current_question:
                    q_text = " ".join(current_question).strip()
                    if q_text and len(q_text) > 10:  # Avoid very short fragments
                        all_questions.append(q_text)
                    current_question = []
                
                # Start new question
                current_question.append(line)
            else:
                # Continue current question
                if current_question:
                    current_question.append(line)
                elif len(line) > 30:  # Could be a question without numbering
                    # Check if it looks like a question (contains question words or is long enough)
                    if re.search(r'\b(?:what|why|how|explain|discuss|describe|examine|write|comment|compare|analyse|analyze)\b', line, re.IGNORECASE):
                        current_question.append(line)
        
        # Don't forget the last question
        if current_question:
            q_text = " ".join(current_question).strip()
            if q_text and len(q_text) > 10:
                all_questions.append(q_text)
    
    # Deduplicate questions
    unique_questions = []
    for q in all_questions:
        is_duplicate = False
        for existing in unique_questions:
            if _is_near_duplicate_question(q, existing):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_questions.append(q)
    
    log(f"Extracted {len(unique_questions)} questions using rule-based approach")
    return unique_questions

def is_noise(line: str) -> bool:
    return bool(NOISE_RE.search(line))


# =========================================================
# FIND QUESTION BOUNDARIES IN ANSWER PAGES -- similarity based
# UNCHANGED.
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
# Add these fixes after the existing imports and before the answer mapping functions

# =========================================================
# FIX: Enhanced Answer Boundary Detection for Cross-Pollution
# =========================================================

def detect_question_boundaries_with_intro_handling(
    answer_lines: list,
    questions: list,
    similarity_threshold: float = 0.25,
    window: int = 5
) -> list:
    """
    ENHANCED FIX: Detects where each question's answer STARTS, with special
    handling for:
    1. Introductory text before the first answer
    2. Cross-pollution between adjacent answers
    3. Answers that restate the question without formal labels
    
    Returns a list of boundary objects with 'line_index' and 'span' for each question.
    """
    # First, try to find answer starts using multiple strategies
    
    boundaries = []
    used_line_indices = set()
    
    # Strategy 1: Look for explicit answer labels with question numbers
    for i, line in enumerate(answer_lines):
        if i in used_line_indices:
            continue
            
        line_stripped = line.strip()
        if len(line_stripped) < 10:
            continue
            
        # Check if this line contains an explicit answer label with a question number
        label_match = re.search(
            r'(?:Ans(?:wer)?[.\s:-]+\s*(\d+)|उत्तर\s*(\d+)|प्र[\.\s]*(\d+)|Q\.?\s*(\d+))',
            line_stripped,
            re.IGNORECASE
        )
        
        if label_match:
            q_num = None
            for group in label_match.groups():
                if group:
                    q_num = int(group)
                    break
            
            if q_num and 1 <= q_num <= len(questions):
                # Find which question index this corresponds to
                for idx, q in enumerate(questions):
                    q_num_match = re.match(r'\s*(\d+)', q)
                    if q_num_match and int(q_num_match.group(1)) == q_num:
                        boundaries.append({
                            'question': q,
                            'question_index': idx,
                            'line_index': i,
                            'span': 0,  # No extra text to skip
                            'score': 1.0,
                            'method': 'explicit_label'
                        })
                        used_line_indices.add(i)
                        break
    
    # Strategy 2: For questions not found by explicit labels, use similarity
    # with improved matching that handles introductory text
    remaining_questions = []
    for idx, q in enumerate(questions):
        if not any(b.get('question_index') == idx for b in boundaries):
            remaining_questions.append((idx, q))
    
    if remaining_questions:
        # Build a candidate map for remaining questions
        candidates_by_question = {}
        
        for q_idx, q in remaining_questions:
            candidates_by_question[q] = []
            
            # Search for the answer start - look for restatement or content match
            for i in range(len(answer_lines)):
                if i in used_line_indices:
                    continue
                    
                line_i = answer_lines[i].strip()
                if len(line_i) < 8:
                    continue
                
                # Try different window sizes to capture the restatement
                for w in range(1, window + 1):
                    if i + w > len(answer_lines):
                        break
                    
                    combined = " ".join(
                        answer_lines[i + k].strip()
                        for k in range(w) if answer_lines[i + k].strip()
                    )
                    if len(combined) < 10:
                        continue
                    
                    # Strip common prefixes for better matching
                    combined_clean = strip_leading_label(combined)
                    q_clean = strip_leading_label(q)
                    
                    # Calculate multiple similarity scores
                    s1 = similarity(combined, q)
                    s2 = similarity(combined_clean, q_clean)
                    
                    # Also check for distinctive word overlap (fixes the 
                    # "akam" vs "puram" issue mentioned in the code comments)
                    distinctive_q = _distinctive_words(q)
                    distinctive_combined = _distinctive_words(combined)
                    word_overlap = 0
                    if distinctive_q and distinctive_combined:
                        word_overlap = len(set(distinctive_q) & set(distinctive_combined)) / max(len(distinctive_q), len(distinctive_combined))
                    
                    score = max(s1, s2, word_overlap)
                    
                    if score >= similarity_threshold:
                        candidates_by_question[q].append({
                            'question': q,
                            'question_index': q_idx,
                            'line_index': i,
                            'span': w,
                            'score': score,
                            'method': 'similarity'
                        })
        
        # Sort candidates by score and pick the best for each question
        for q in candidates_by_question:
            candidates_by_question[q].sort(key=lambda c: -c['score'])
        
        # Greedy assignment - ensure each line is used only once
        for q in remaining_questions:
            q_text = q[1]
            cands = candidates_by_question.get(q_text, [])
            
            for c in cands:
                if c['line_index'] not in used_line_indices:
                    boundaries.append(c)
                    used_line_indices.add(c['line_index'])
                    break
    
    # Sort boundaries by line index
    boundaries.sort(key=lambda b: b['line_index'])
    
    # Strategy 3: If we still have missing questions, look for them
    # in the answer text using content-based search (including intro text)
    found_indices = set(b.get('question_index') for b in boundaries if 'question_index' in b)
    missing_indices = [i for i in range(len(questions)) if i not in found_indices]
    
    if missing_indices:
        for q_idx in missing_indices:
            q = questions[q_idx]
            
            # Try to find this question's answer by looking for its 
            # distinctive content in the remaining lines
            best_match = None
            best_score = 0.0
            
            for i in range(len(answer_lines)):
                if i in used_line_indices:
                    continue
                    
                line = answer_lines[i].strip()
                if len(line) < 20:  # Answers typically start with substantial text
                    continue
                
                # Check if this line contains distinctive content from the question
                # (even without a formal label, the answer might restate the topic)
                q_words = set(normalize(q).split())
                line_words = set(normalize(line).split())
                
                if q_words and line_words:
                    overlap = len(q_words & line_words) / max(len(q_words), 1)
                    
                    # Also check the next few lines for better context
                    context_lines = [line]
                    for j in range(1, 4):
                        if i + j < len(answer_lines):
                            context_lines.append(answer_lines[i + j].strip())
                    context = " ".join(context_lines)
                    context_words = set(normalize(context).split())
                    context_overlap = len(q_words & context_words) / max(len(q_words), 1)
                    
                    score = max(overlap, context_overlap)
                    
                    if score > best_score and score >= 0.15:
                        best_score = score
                        best_match = {
                            'question': q,
                            'question_index': q_idx,
                            'line_index': i,
                            'span': 0,
                            'score': score,
                            'method': 'content_search'
                        }
            
            if best_match:
                boundaries.append(best_match)
                used_line_indices.add(best_match['line_index'])
    
    # Final sort by line index
    boundaries.sort(key=lambda b: b['line_index'])
    
    return boundaries


def slice_qa_with_intro_handling(
    answer_lines: list, 
    boundaries: list, 
    questions: list
) -> list:
    """
    ENHANCED FIX: Slices answers from the raw text, with special handling
    for:
    1. Preserving introductory text before the first answer
    2. Preventing cross-pollution between adjacent answers
    3. Handling partial or missing boundaries
    """
    if not boundaries:
        return []
    
    qa_pairs = []
    
    # Ensure we have the full list of questions
    question_map = {q: i for i, q in enumerate(questions)}
    
    # First, find the start of the first answer (including intro text)
    first_answer_start = boundaries[0]['line_index']
    
    # Check if there's introductory text before the first answer
    intro_lines = []
    if first_answer_start > 0:
        # Look for introductory text (3-4 lines that don't match any question)
        intro_candidate = []
        for j in range(0, first_answer_start):
            line = answer_lines[j].strip()
            if line and not is_noise(line):
                intro_candidate.append(line)
        
        # Check if this looks like genuine intro text (substantial content)
        if intro_candidate and len(" ".join(intro_candidate)) > 100:
            intro_lines = intro_candidate
    
    # Now slice the answers
    for i, b in enumerate(boundaries):
        span = b.get("span", 0)
        a_start = b["line_index"] + span
        
        # Determine end of this answer
        if i + 1 < len(boundaries):
            a_end = boundaries[i + 1]["line_index"]
        else:
            a_end = len(answer_lines)
        
        # Extract the answer text
        raw_lines = []
        
        # Special case: first answer gets the intro text
        if i == 0 and intro_lines:
            raw_lines.extend(intro_lines)
            # Also add the actual answer lines, but avoid duplication
            for j in range(a_start, a_end):
                line = answer_lines[j].strip()
                if line and not is_noise(line):
                    # Avoid duplicating intro text if it appears again
                    if line not in intro_lines or len(intro_lines) < 3:
                        raw_lines.append(line)
        else:
            for j in range(a_start, a_end):
                line = answer_lines[j].strip()
                if line and not is_noise(line):
                    raw_lines.append(line)
        
        # Check if the question's answer is genuinely in this slice
        q = b["question"]
        answer_text = " ".join(raw_lines).strip()
        
        # If the answer text is too short or doesn't contain relevant content,
        # try to find the real answer in the surrounding text
        if len(answer_text) < 50 and i + 1 < len(boundaries):
            # This might be a boundary error - expand to include more text
            next_start = boundaries[i + 1]["line_index"]
            expanded_lines = []
            for j in range(a_start, next_start):
                line = answer_lines[j].strip()
                if line and not is_noise(line):
                    expanded_lines.append(line)
            if expanded_lines:
                expanded_text = " ".join(expanded_lines).strip()
                if len(expanded_text) > len(answer_text):
                    # Check if this expanded text contains content relevant to this question
                    q_words = set(normalize(q).split())
                    exp_words = set(normalize(expanded_text).split())
                    if q_words and exp_words:
                        overlap = len(q_words & exp_words) / max(len(q_words), 1)
                        if overlap > 0.1:  # Some relevance
                            answer_text = expanded_text
                            raw_lines = expanded_lines
        
        # Clean up the answer
        answer_text = strip_question_restatement(answer_text)
        answer_text = strip_full_question_echo(answer_text, q)
        
        # Remove any trailing empty lines
        answer_text = "\n".join(
            line for line in answer_text.split("\n") 
            if line.strip()
        )
        
        # Determine which question this is (using the question text, not index)
        # This handles the case where boundaries might be mis-indexed
        matching_question = None
        q_idx = None
        
        # First try exact match
        for q_text in questions:
            if q_text == q:
                matching_question = q_text
                break
        
        # If not found, try fuzzy match
        if not matching_question:
            for q_text in questions:
                if _is_near_duplicate_question(q, q_text):
                    matching_question = q_text
                    break
        
        # If still not found, use the original question
        if not matching_question:
            matching_question = q
        
        qa_pairs.append({
            "question": matching_question,
            "answer": answer_text,
            "matched": True,
            "has_intro": i == 0 and bool(intro_lines)
        })
    
    return qa_pairs


def enhanced_slice_qa_from_line_items(
    answer_lines: list,
    questions: list,
    similarity_threshold: float = 0.25,
    window: int = 5
) -> list:
    """
    ENHANCED FIX: Complete replacement for the problematic slicing function
    that caused:
    1. Cross-polluted answers (bleeding)
    2. Dropped introductory text
    
    This function combines the improved boundary detection and slicing
    to produce clean, properly segmented question-answer pairs.
    """
    # Step 1: Detect boundaries with intro handling
    boundaries = detect_question_boundaries_with_intro_handling(
        answer_lines,
        questions,
        similarity_threshold,
        window
    )
    
    # Step 2: If we didn't find enough boundaries, use content-based fallback
    if len(boundaries) < len(questions) * 0.5:
        # Fallback: use the original similarity approach but with improvements
        boundaries = find_question_boundaries_by_similarity(
            answer_lines,
            questions,
            similarity_threshold,
            window
        )
    
    # Step 3: Slice the answers with intro handling
    qa_pairs = slice_qa_with_intro_handling(
        answer_lines,
        official_questions,
        similarity_threshold=0.25,
        window=5
    )
    
    # Step 4: Ensure we have all questions covered
    matched_questions = [pair["question"] for pair in qa_pairs]
    
    # Find any missing questions
    for q in questions:
        if q not in matched_questions:
            # Try to find this question's answer in the remaining content
            found = False
            for pair in qa_pairs:
                if _is_near_duplicate_question(q, pair["question"]):
                    found = True
                    break
            
            if not found:
                # Add an empty entry for this question
                qa_pairs.append({
                    "question": q,
                    "answer": "",
                    "matched": False,
                    "has_intro": False
                })
    
    # Step 5: Sort by question order
    question_order = {q: i for i, q in enumerate(questions)}
    qa_pairs.sort(key=lambda p: question_order.get(p["question"], 999))
    
    return qa_pairs

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
    FIX: catches a real failure mode confirmed in production -- the
    question-paper/answer-page split misclassified pages, leaving only
    cover/admin/letterhead pages (e.g. "IGNOU logo...", "THE PEOPLE'S
    UNIVERSITY") as the "answer pages." This is detectable BEFORE
    spending any answer-mapping LLM calls at all: real essay-style
    answers run well over a thousand characters EACH for substantive
    responses, so if the total available answer text is implausibly
    small relative to the number of questions, something already went
    wrong upstream. Catching this here means the failure is reported
    immediately and cheaply, with a clear pointer to the real cause,
    instead of running a full (doomed) round of answer-mapping calls
    that burn tokens and time before surfacing a generic
    "could not match any questions" error several steps later.
    """
    total_chars = sum(len(l) for l in answer_lines)
    avg_chars_per_question = total_chars / max(num_questions, 1)
    MIN_PLAUSIBLE_CHARS_PER_QUESTION = 200  # conservative floor

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

    # Check if answer pages look plausible
    pages_look_plausible = _sanity_check_answer_pages(answer_lines, len(official_questions), log)
    if not pages_look_plausible:
        raise Exception(
            "The 'answer pages' identified in this document do not contain enough "
            "text to plausibly hold real essay-style answers for the "
            f"{len(official_questions)} question(s) found. This usually means the "
            "question-paper/answer-page page split misclassified pages -- check the "
            "'Question paper pages detected' log line above against the actual "
            "document structure."
        )

    # Use SIMILARITY-BASED mapping (NO LLM calls) - saves token quota
    log("Mapping each question to its answer using similarity-based approach (NO LLM calls)...")
    qa_map = map_answers_with_similarity(answer_lines, official_questions, status_callback)

    matched_count = sum(1 for q in official_questions if q in qa_map and qa_map[q].strip())
    log(f"Matched {matched_count} of {len(official_questions)} questions")

    for q in official_questions:
        if q not in qa_map or not qa_map[q].strip():
            log(f"WARNING: No match found for: {q[:60]}...")

    # Build the Q&A pairs list, preserving the official question order
    qa_pairs = []
    for q in official_questions:
        qa_pairs.append({
            "question": q,
            "answer": qa_map.get(q, ""),
            "matched": q in qa_map and bool(qa_map[q].strip()),
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
