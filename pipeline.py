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
- A real question paper is usually self-contained and concise per question (a question, maybe a mark allocation) -- not a long flowing essay with numbered sub-points woven into running prose.
- CRITICAL TRAP TO AVOID: students very commonly RESTATE the question itself as the FIRST SENTENCE of their answer, before writing their actual response (e.g. an answer's opening page reads "Examine the theme of concealment in X. Discuss with reference to Y. The theme of concealment is central to..." where everything after the first sentence is the student's OWN original explanation, not more instructions). Such a page can superficially look like a question-paper page because it contains prompt-style verbs ("Examine", "Discuss") -- but it is the FIRST page of a long, multi-page ANSWER, not a question paper page. Signals that this is really an answer's opening page, not a real question paper page: (a) the page has noticeably MORE text than a typical printed question would need, especially if it keeps going well past where a concise instruction would end; (b) the prose quality looks like a developing argument/explanation rather than a terse instruction; (c) the SAME or very similar question text already appears verbatim on a page you are more confident is the genuine, concise question paper (in which case this longer, messier page is almost certainly the student's restatement -- exclude it). When uncertain whether a page is the real question paper or a student's restatement-then-answer, treat brevity and conciseness as the deciding signal: genuine question papers are short per question; answer pages (including their opening restatement) run much longer.
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
    if ratio < 0.90:
        return False

    # FIX: minimum word length is 3, not 4. A 4+ letter cutoff was
    # filtering out short-but-significant distinguishing words like
    # spelled-out numbers ("one" vs "two"), which caused two genuinely
    # DIFFERENT questions ("Real question one." / "Real question two.")
    # to be wrongly merged -- everything else in the sentence matched,
    # and the one word that actually differed was too short to be
    # counted, leaving a perfect (but wrong) word-overlap score. 3
    # letters still excludes pure function-word noise ("a", "of", "in",
    # "to") that would otherwise inflate apparent overlap without
    # carrying real distinguishing content.
    words1 = sorted(set(re.findall(r'[a-z]{3,}', k1)))
    words2 = sorted(set(re.findall(r'[a-z]{3,}', k2)))
    if not words1 or not words2:
        return ratio >= 0.92

    matched = sum(1 for w1 in words1 if any(_words_nearly_match(w1, w2) for w2 in words2))
    overlap = matched / max(len(words1), len(words2))

    return ratio >= 0.90 and overlap >= 0.92


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
        # NOTE: this stage's own per-chunk `questions` are intentionally
        # NOT collected anymore -- they are exactly the inconsistent,
        # possibly-conflicting splits described above. Only the PAGE
        # indices from this stage are kept; the actual question list
        # comes from extract_canonical_questions() in a single pass,
        # below, once we know which pages to look at.
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

    # FIX (this round): real-world failure confirmed -- a student's
    # ANSWER often opens by restating the question itself ("Examine
    # the theme of X. Discuss with reference to Y...") before writing
    # their actual original explanation. That opening page can
    # superficially look like a genuine question-paper page to the
    # LLM, since it legitimately contains prompt-style verbs. If this
    # happens, that page gets WRONGLY EXCLUDED from answer_lines
    # entirely (since only non-question-paper pages become answer
    # text), which silently deletes the FIRST page of that answer --
    # exactly matching the real symptom reported ("one page skipped
    # from the start" of an answer).
    #
    # A real question-paper page is reliably CONCISE (a question, maybe
    # a mark allocation) -- a misclassified answer-opening page is
    # reliably much LONGER (it's the start of a multi-page essay). This
    # checks for length outliers among the pages classified as
    # question-paper pages and logs a clear warning so the issue is
    # immediately visible rather than silently losing content -- it
    # does not auto-correct (since a genuinely long, dense question
    # paper page is possible, e.g. one with many sub-parts), but makes
    # the failure mode loud instead of silent.
    if len(qp_page_indices_0based) >= 2:
        qp_page_lengths = [
            (i, len(pages[i]["raw_text"])) for i in qp_page_indices_0based
        ]
        lengths_only = [length for _, length in qp_page_lengths]
        median_length = sorted(lengths_only)[len(lengths_only) // 2]
        for page_idx, length in qp_page_lengths:
            # An outlier page 3x+ longer than the median AND absolutely
            # long enough to plausibly be an essay opening (not just a
            # naturally longer but still-concise question) is flagged.
            if length > max(median_length * 3, 1500):
                log(
                    f"WARNING: page {page_idx + 1} was classified as a question "
                    f"paper page but is {length} chars long -- much longer than "
                    f"the typical {median_length} chars for this document's other "
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
}}


def _distinctive_words(text: str, max_words: int = 20) -> list:
    words = re.findall(r'[a-z]{3,}', _normalize_for_overlap_match(text))[:max_words]
    return sorted(set(w for w in words if w not in _QUESTION_STOPWORDS))


# =========================================================
# FIX 1: IMPROVED ANSWER START DETECTION
# =========================================================

def _line_starts_new_answer_for_question(line: str, questions: list, min_fraction: float = 0.4) -> int:
    """
    IMPROVED FIX: More generous detection of answer starts.
    Now catches more patterns including Hindi labels and restatements.
    """
    # Check for formal labels - these ALWAYS indicate a new answer
    label_match = _ANSWER_START_RE.match(line)
    if label_match:
        num_match = re.search(r'\d+', label_match.group(0))
        if num_match:
            label_num = num_match.group(0)
            # Try to match to a question number
            for i, q in enumerate(questions):
                q_num_match = re.match(r'\s*(\d+)', q)
                if q_num_match and q_num_match.group(1) == label_num:
                    return i
        # Even if number doesn't match specific question, treat as new answer
        return -1
    
    # Check for Hindi answer markers
    hindi_markers = ['उत्तर', 'प्रश्न', 'प्र.', 'Ans', 'Answer', 'Q.', 'Q']
    for marker in hindi_markers:
        if re.search(rf'^\s*{marker}\s*\d*', line, re.IGNORECASE):
            return -1
    
    # Check for question restatement - MORE GENEROUS
    line_words = sorted(set(re.findall(r'[a-z]{3,}', _normalize_for_overlap_match(line))[:25]))
    if not line_words:
        return None
    
    for i, q in enumerate(questions):
        q_distinctive = _distinctive_words(q)
        if not q_distinctive:
            continue
        
        # Lower threshold for matching
        matched = sum(
            1 for w in q_distinctive
            if any(_words_nearly_match(w, lw) for lw in line_words)
        )
        required = max(1, round(len(q_distinctive) * min_fraction))
        if matched >= required:
            return i
    
    return None


def _line_starts_new_answer_for_question(line: str, questions: list, min_fraction: float = 0.4) -> int:
    """
    IMPROVED FIX: More generous detection of answer starts.
    Now catches more patterns including Hindi labels and restatements.
    """
    # Check for formal labels - these ALWAYS indicate a new answer
    label_match = _ANSWER_START_RE.match(line)
    if label_match:
        num_match = re.search(r'\d+', label_match.group(0))
        if num_match:
            label_num = num_match.group(0)
            # Try to match to a question number
            for i, q in enumerate(questions):
                q_num_match = re.match(r'\s*(\d+)', q)
                if q_num_match and q_num_match.group(1) == label_num:
                    return i
        # Even if number doesn't match specific question, treat as new answer
        return -1
    
    # Check for Hindi answer markers
    hindi_markers = ['उत्तर', 'प्रश्न', 'प्र.', 'Ans', 'Answer', 'Q.', 'Q']
    for marker in hindi_markers:
        if re.search(rf'^\s*{marker}\s*\d*', line, re.IGNORECASE):
            return -1
    
    # Check for question restatement - MORE GENEROUS
    line_words = sorted(set(re.findall(r'[a-z]{3,}', _normalize_for_overlap_match(line))[:25]))
    if not line_words:
        return None
    
    for i, q in enumerate(questions):
        q_distinctive = _distinctive_words(q)
        if not q_distinctive:
            continue
        
        # Lower threshold for matching (40% instead of 50%)
        matched = sum(
            1 for w in q_distinctive
            if any(_words_nearly_match(w, lw) for lw in line_words)
        )
        required = max(1, round(len(q_distinctive) * min_fraction))
        if matched >= required:
            return i
    
    return None


def _chunk_lines_by_char_budget(numbered_lines: list, questions: list,
                                  max_chars: int = ANSWER_MAP_MAX_CHARS_PER_CHUNK,
                                  absolute_max_chars: int = ANSWER_MAP_ABSOLUTE_MAX_CHARS) -> list:
    """
    IMPROVED FIX: Much more generous chunking that never splits answers.
    Only splits at clear answer boundaries.
    """
    if not numbered_lines:
        return []

    chunks = []
    current_chunk = []
    current_chars = 0
    current_question_idx = None
    
    for idx, text in numbered_lines:
        line_chars = len(text)
        
        # Check if this line starts a new answer
        matched_q_idx = _line_starts_new_answer_for_question(text, questions)
        is_new_answer = matched_q_idx is not None
        
        # Force new chunk ONLY if:
        # 1. We have content, AND
        # 2. This is clearly a new answer (different question), AND
        # 3. Current chunk is already substantial
        if current_chunk and is_new_answer and current_question_idx != matched_q_idx:
            # Check if current chunk is at least half of max_chars
            if current_chars >= max_chars * 0.5:
                chunks.append(current_chunk)
                current_chunk = []
                current_chars = 0
        
        # If this is a new answer, update current question
        if matched_q_idx is not None and matched_q_idx != -1:
            current_question_idx = matched_q_idx
        
        current_chunk.append((idx, text))
        current_chars += line_chars
        
        # Absolute safety - if chunk gets too big, split at next boundary
        if current_chars > absolute_max_chars:
            chunks.append(current_chunk)
            current_chunk = []
            current_chars = 0
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def _resolve_overlapping_answer_ranges(answer_ranges: list) -> list:
    """
    IMPROVED FIX: Resolves overlaps but NEVER cuts an answer short.
    If ranges overlap, merges them into one continuous range.
    """
    if not answer_ranges:
        return []
    
    # Sort by start_line
    sorted_ranges = sorted(answer_ranges, key=lambda r: r["start_line"])
    
    # Merge overlapping ranges
    merged = []
    for r in sorted_ranges:
        if not merged:
            merged.append(dict(r))
            continue
        
        last = merged[-1]
        # If ranges overlap or touch, merge them
        if r["start_line"] <= last["end_line"] + 1:
            # Extend the range to include both
            last["end_line"] = max(last["end_line"], r["end_line"])
        else:
            merged.append(dict(r))
    
    return merged


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
    """
    IMPROVED FIX: Extracts COMPLETE answers including ALL pages.
    No content is ever cut or truncated.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    if not answer_lines or not questions:
        return {}

    from groq import Groq

    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found in secrets or environment")

    client = Groq(api_key=api_key)
    budget = _TokenBudgetTracker()

    # Deterministic REF label <-> question index mapping
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

        # Validate ranges
        valid_indices = {idx for idx, _ in chunk}
        min_idx, max_idx = min(valid_indices), max(valid_indices)
        for r in chunk_ranges:
            if r["ref"] not in ref_to_question:
                log(f"WARNING: discarding answer mapping with unknown ref {r['ref']!r}")
                continue
            # Be generous - allow ranges that are slightly out of bounds
            if min_idx - 5 <= r["start_line"] <= max_idx + 5:
                # Clamp to valid range
                start = max(0, min(r["start_line"], len(answer_lines) - 1))
                end = max(start, min(r["end_line"], len(answer_lines) - 1))
                all_ranges.append({
                    "ref": r["ref"],
                    "start_line": start,
                    "end_line": end
                })

        log(f"Chunk {i+1}/{len(chunks)}: mapped {len(chunk_ranges)} answer(s)")

    # Deduplicate: keep the longest range for each ref
    best_by_ref = {}
    for r in all_ranges:
        existing = best_by_ref.get(r["ref"])
        if existing is None or (r["end_line"] - r["start_line"]) > (existing["end_line"] - existing["start_line"]):
            best_by_ref[r["ref"]] = r

    deduped_ranges = list(best_by_ref.values())

    # Merge overlapping ranges instead of cutting them
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
                f"Answer mapping found ZERO matches across all {len(chunks)} chunk(s). "
                f"Sample lines: {sample_lines}"
            )

    # Extract COMPLETE answers - preserve EVERYTHING
    qa_map = {}
    for r in resolved_ranges:
        start, end = r["start_line"], r["end_line"]
        
        # Include ALL lines from start to end - no filtering!
        verbatim_lines = []
        for j in range(start, end + 1):
            if 0 <= j < len(answer_lines):
                line = answer_lines[j].strip()
                if line:  # Only skip completely empty lines
                    verbatim_lines.append(line)
        
        original_question = ref_to_question[r["ref"]]
        # Join with newlines to preserve structure
        answer_text = "\n".join(verbatim_lines).strip()
        
        # ONLY remove labels from the VERY start, nothing else
        answer_text = strip_question_restatement(answer_text)
        
        if answer_text:
            qa_map[original_question] = answer_text

    log(f"Extracted {len(qa_map)} complete answers with ALL content preserved")
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

    # FIX: catches a misclassified page split BEFORE spending any
    # answer-mapping LLM calls, rather than discovering it only after
    # a full (doomed) round of calls produces zero matches.
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

    # FIX: replaces the old similarity-based sliding-window matching
    # (which could let one question's answer swallow several others --
    # the exact bug seen in real usage) with LLM-based, per-question
    # INDEPENDENT answer extraction. Each question's answer boundary is
    # identified on its own merits by the LLM reading the actual text,
    # with a hard Python-side overlap-resolution safety net guaranteeing
    # no answer can ever absorb another's content.
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

    # Build the Q&A pairs list, preserving the official question order
    # and explicitly marking unmatched questions rather than silently
    # dropping them -- this makes it clear in the output which
    # questions were genuinely not found versus matched-but-empty.
    qa_pairs = []
    for q in official_questions:
        qa_pairs.append({
            "question": q,
            "answer": qa_map.get(q, ""),
            "matched": q in qa_map,
        })

    log(f"Done -- {len(qa_pairs)} Q-A pairs ({matched_count} matched)")

    # Returns BOTH requested outputs separately:
    # - ocr_json: the complete raw OCR of the whole PDF, every page
    # - qa_pairs: the clean, independently-mapped question -> answer
    #   pairs, in official question order
    # The caller is responsible for writing these to two separate
    # files (see save_outputs() below for a ready-made helper).
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
