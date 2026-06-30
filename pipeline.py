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
CHARS_PER_TOKEN_ESTIMATE = 2.0  # safe for Latin/English text

# FIX: real-world failure -- an all-Hindi (Devanagari) document had
# EVERY single chunk fail, ending in "Could not match any questions to
# answers" with zero questions ever mapped. Root cause: 2.0 chars/token
# is only a reasonable estimate for English. Devanagari text tokenizes
# FAR denser -- conjunct consonants, matras, and virama marks each
# routinely become separate BPE tokens, so a Devanagari-heavy chunk can
# easily run 3-5x more tokens than the same character count of English
# would. With the old single, English-tuned ratio, a chunk sized to
# "fit" under the 8000 TPM ceiling using chars/2.0 could actually need
# 3-5x that many tokens -- guaranteeing an oversized-request failure on
# literally every chunk of an all-Hindi document, every single time,
# with no possible recovery via retry/backoff (a too-large request
# stays too-large no matter how long you wait).
#
# This estimates the ratio PER TEXT based on its actual Devanagari
# character proportion, instead of one fixed global constant.
DEVANAGARI_CHARS_PER_TOKEN_ESTIMATE = 0.5  # conservative: real OCR'd
                                              # Devanagari text can run
                                              # even denser than typical
                                              # samples; keep solid
                                              # headroom under the TPM
                                              # ceiling rather than
                                              # landing right at the edge
DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')


def _detect_chars_per_token(text: str) -> float:
    """
    Returns a chars-per-token estimate appropriate for the SCRIPT MIX
    actually present in this text, instead of assuming English. Mixes
    the English and Devanagari ratios proportionally to how much of the
    text is actually Devanagari, so a mostly-English document with a
    little Hindi isn't unnecessarily over-chunked, while a mostly-Hindi
    document gets the much more conservative ratio it actually needs.
    """
    if not text:
        return CHARS_PER_TOKEN_ESTIMATE
    devanagari_chars = len(DEVANAGARI_RE.findall(text))
    ratio_devanagari = devanagari_chars / len(text)
    return (
        ratio_devanagari * DEVANAGARI_CHARS_PER_TOKEN_ESTIMATE
        + (1 - ratio_devanagari) * CHARS_PER_TOKEN_ESTIMATE
    )


MAX_CHARS_PER_CHUNK = 6000  # smaller than before -- real-world 429s
                              # showed our chars-per-token guess was
                              # optimistic; smaller chunks reduce blast
                              # radius of any single misestimate.
                              # NOTE: for Devanagari-heavy documents,
                              # _chunk_pages_by_char_budget below further
                              # reduces the effective per-chunk budget --
                              # see DEVANAGARI_CHAR_BUDGET_SCALE.

DEVANAGARI_CHAR_BUDGET_SCALE = 0.22  # tightened to match the more
# conservative DEVANAGARI_CHARS_PER_TOKEN_ESTIMATE above -- verified
# empirically (see test below) to keep chunk token estimates safely
# under the TPM ceiling including system-prompt overhead.

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
- When genuinely uncertain whether a page is a question paper page, prefer NOT including it as one, and prefer NOT extracting its numbered items as separate questions.
- If NONE of the pages shown in this chunk are question paper pages, return empty lists for both fields -- that is a valid and expected result for chunks that only contain answer/admin pages.
- Preserve the EXACT original text and numbering of real questions -- do not paraphrase, do not renumber, do not translate.
- Output ONLY the JSON object described above. No prose before or after it. No markdown code fences."""


def _scaled_char_budget(texts: list, base_max_chars: int) -> int:
    """
    Scales a character budget down based on how Devanagari-heavy the
    given texts actually are -- see DEVANAGARI_CHAR_BUDGET_SCALE for
    why a flat English-tuned character budget badly underprotects
    Hindi/Devanagari content from oversized-request failures.
    """
    sample = "".join(texts)[:4000]  # a representative sample is enough;
                                      # avoids re-scanning huge documents
    if not sample:
        return base_max_chars
    devanagari_ratio = len(DEVANAGARI_RE.findall(sample)) / len(sample)
    scale = 1.0 - devanagari_ratio * (1.0 - DEVANAGARI_CHAR_BUDGET_SCALE)
    return max(500, int(base_max_chars * scale))


def _chunk_pages_by_char_budget(pages: list, max_chars: int = MAX_CHARS_PER_CHUNK,
                                  overlap_pages: int = CHUNK_OVERLAP_PAGES) -> list:
    if not pages:
        return []

    effective_max_chars = _scaled_char_budget(
        [p["raw_text"] for p in pages], max_chars
    )

    chunks = []
    current_chunk = []
    current_chars = 0

    for page in pages:
        page_chars = len(page["raw_text"])

        if current_chunk and current_chars + page_chars > effective_max_chars:
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
    """Estimate using a chars-per-token ratio that adapts to the actual
    script mix of THIS text (see _detect_chars_per_token) -- a fixed
    English-tuned ratio badly underestimates Devanagari token usage."""
    ratio = _detect_chars_per_token(text)
    return int(len(text) / ratio) + 1


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

        # FIX: previously this call had no exception handling at all --
        # if chunk 3 of 6 hit a TPD (daily quota) wall, or any other
        # unrecoverable error, the exception propagated straight up and
        # ABORTED THE ENTIRE DOCUMENT, discarding whatever chunks 1-2
        # had already successfully found. Now a per-chunk failure is
        # caught and logged, and processing continues with the
        # remaining chunks -- so a document that runs out of daily
        # quota partway through still returns whatever was found before
        # the quota ran out, instead of losing everything. This mirrors
        # the same resilience already present in map_answers_with_llm.
        try:
            qp_pages_1based, questions = _call_groq_for_chunk(client, chunk, budget, log)
        except Exception as e:
            log(f"WARNING: chunk {i+1}/{len(chunks)} question-identification failed, skipping: {e}")
            chunk_failures.append(str(e))
            continue

        # FIX: recover page numbers the LLM accidentally concatenated
        # together (e.g. [14, 16, 18] emitted as [141618]) instead of
        # just discarding them as "out of range."
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
        # Every single chunk failed -- there is genuinely nothing to
        # return, so surface the failure clearly rather than silently
        # returning empty results that would produce a confusing
        # downstream "no questions found" error instead of the real
        # reason (e.g. quota exhaustion).
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
# QUESTION-PAPER RECONCILIATION (Groq)
#
# FIX: chunk-level dedup inside identify_questions_with_llm (see
# _is_near_duplicate_question) only catches NEAR-IDENTICAL text -- a
# fuzzy match on the whole string. It cannot catch a different bug
# confirmed in real usage: the SAME official question getting
# extracted TWICE at different GRANULARITIES -- once as a single
# compound multi-part question (a number with sub-parts (i), (ii),
# (iii), (iv) all under it), and again as several separate candidates,
# one per sub-part. A sub-part's text is a genuine SUBSET of the
# compound block's text, not a near-match of the whole thing, so the
# character-similarity ratio between them is low and dedup never
# merges them. The visible symptom: several extra questions in the
# final output whose "answer" is just "(no answer text matched)",
# because their real answer was already captured inside the compound
# question's answer, not on its own.
#
# This adds a final reconciliation pass after chunk-merging: the LLM
# is shown the ACTUAL, unmodified question-paper text directly (ground
# truth), alongside the current candidate list, and asked to pick
# exactly ONE winning candidate per TRUE distinct question -- in the
# paper's own printed order. This both removes the granularity-level
# duplicates the earlier fuzzy dedup can't catch, and guarantees the
# final question order always follows how the questions actually
# appear in the original paper (monotonic by official numbering),
# rather than whatever order chunk-by-chunk extraction happened to
# emit them in.
#
# Like the answer-mapping call, candidates are identified by an
# unambiguous CAND-A/CAND-B/... label assigned by US (not retyped by
# the model), so picking a "winner" is a deterministic Python list
# lookup with zero text-matching ambiguity.
# =========================================================

QP_RECONCILE_SYSTEM_PROMPT = """You are given:
1. The ACTUAL, COMPLETE, UNMODIFIED OCR text of the question paper pages from an exam assignment booklet.
2. A list of CANDIDATE question strings that were previously extracted from this same question paper by an earlier automated pass. The candidates may contain duplicates, or may break a single multi-part question (one official question number with sub-parts like (i), (ii), (iii), (iv)...) into several separate candidates instead of one combined item.

Each candidate is tagged with a label like [CAND-A], [CAND-B], etc.

Your task: read the ACTUAL question paper text and determine the TRUE, FINAL list of distinct exam questions, in the EXACT ORDER they appear in the original question paper -- following the paper's own printed sequence (by official question number), never reordered, never repeated.

For each true distinct question:
- Pick the SINGLE candidate label whose text is the MOST COMPLETE and FAITHFUL match to that question as it actually appears in the question paper. If the question paper presents a question as ONE numbered item with multiple sub-parts under it, prefer the candidate containing the FULL multi-part block over any candidate containing only ONE of its sub-parts in isolation.
- Each true question must be represented by EXACTLY ONE winning candidate label. Any other candidate that is really just a fragment or duplicate of an already-chosen question must be DROPPED entirely, not included again under a different position.
- If a true question genuinely has no matching candidate at all, omit it from the output (never invent text that isn't in the candidates).

Return ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:

{
  "final_order": ["CAND-C", "CAND-A", "CAND-E"]
}

Use ONLY the exact CAND labels given above. Do not retype, paraphrase, or merge candidate text yourself -- the label is all that's needed."""


def _build_qp_reconcile_user_prompt(qp_text: str, candidates: list) -> str:
    cand_block = "\n".join(
        f"[CAND-{chr(65+i)}] {q}" for i, q in enumerate(candidates)
    )
    return (
        f"ACTUAL QUESTION PAPER TEXT (verbatim OCR of the official question "
        f"paper pages):\n{qp_text}\n\n"
        f"CANDIDATE QUESTIONS (previously extracted -- may contain "
        f"duplicates or sub-part fragments of the same question):\n{cand_block}"
    )


def _parse_qp_reconcile_response(content: str) -> list:
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

    if not isinstance(data, dict) or "final_order" not in data:
        raise ValueError(
            f"LLM response missing 'final_order' key. Got: "
            f"{list(data.keys()) if isinstance(data, dict) else type(data).__name__}"
        )

    order = data["final_order"]
    if not isinstance(order, list):
        raise ValueError(f"'final_order' must be a list, got: {type(order).__name__}")

    return [str(x).strip().upper() for x in order]


def reconcile_questions_with_paper(qp_text: str, candidate_questions: list,
                                     status_callback=None) -> list:
    """
    Final ground-truth reconciliation pass -- see the module-level FIX
    comment above for the full rationale. Compares the candidate
    question list against the actual question-paper text and returns
    a deduplicated, canonically-ordered (monotonic, matching the
    paper's own printed sequence) final list.

    This is a refinement pass, not a required step: if it fails for
    any reason (network issue, malformed LLM output, missing API key),
    it logs a warning and falls back to the original candidate list
    rather than aborting an otherwise-successful run.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    if not candidate_questions:
        return []

    if len(candidate_questions) == 1:
        return candidate_questions

    from groq import Groq

    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        log("WARNING: GROQ_API_KEY not found, skipping question-paper reconciliation")
        return candidate_questions

    client = Groq(api_key=api_key)
    budget = _TokenBudgetTracker()

    cand_to_question = {f"CAND-{chr(65+i)}": q for i, q in enumerate(candidate_questions)}

    log(
        f"Reconciling {len(candidate_questions)} candidate question(s) against "
        f"the actual question paper text..."
    )

    user_prompt = _build_qp_reconcile_user_prompt(qp_text, candidate_questions)

    try:
        final_order = _call_groq_with_retries(
            client, QP_RECONCILE_SYSTEM_PROMPT, user_prompt,
            _parse_qp_reconcile_response, budget, log
        )
    except Exception as e:
        log(f"WARNING: question-paper reconciliation failed, keeping original candidate list: {e}")
        return candidate_questions

    reconciled = []
    seen_labels = set()
    for label in final_order:
        if label in seen_labels:
            continue  # defend against the LLM repeating a label
        seen_labels.add(label)
        q = cand_to_question.get(label)
        if q is None:
            log(f"WARNING: reconciliation returned unknown label {label!r}, ignoring")
            continue
        reconciled.append(q)

    if not reconciled:
        log("WARNING: reconciliation returned no valid questions, keeping original candidate list")
        return candidate_questions

    log(
        f"Reconciliation complete -- {len(reconciled)} final question(s), "
        f"reordered to match the question paper"
    )
    return reconciled


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


ANSWER_MAP_MAX_CHARS_PER_CHUNK = 11000  # FIX: bumped up from 8000.
# Real-world output showed essay answers (e.g. the "concealment" and
# "akam/puram thinai" answers) still getting cut off mid-sentence --
# the answer's true end simply fell outside the chunk that contained
# its start, even with overlap. A bigger per-chunk window means fewer
# answers straddle a boundary at all. NOTE: this is intentionally NOT
# pushed much higher than this -- estimated_tokens for a chunk this
# size (~11000 chars + prompt overhead) sits close to the entire
# 8000-token TPM ceiling (see TPM_LIMIT above) for a SINGLE request.
# Past that point, no amount of pacing/waiting helps, because Groq
# would reject one oversized request outright regardless of how much
# of the per-minute budget is free. The structural fix for answers
# that are STILL longer than one chunk is the cross-chunk range-merge
# below, not chunk size alone.

ANSWER_MAP_OVERLAP_CHARS = 5000  # FIX: bumped up alongside the chunk
# size above, for the same reason (a fixed LINE-COUNT overlap is
# unreliable because OCR'd answer lines vary wildly in length -- 5
# lines might cover a full paragraph or just a few words, depending on
# how the page wrapped). A long essay answer (confirmed in real usage
# to run 1500-3000+ characters) could easily exceed a small line-count
# overlap entirely, meaning NEITHER chunk that saw a piece of it ever
# saw the WHOLE thing -- which is exactly why answers were coming back
# cut off mid-sentence. Character-based overlap, sized generously
# above most realistic single answers, guarantees the complete answer
# usually appears intact in at least one chunk's view -- and for the
# rare answer still longer than that, the range-merge logic below
# stitches the pieces back together instead of just picking one.


def _chunk_lines_by_char_budget(numbered_lines: list,
                                  max_chars: int = ANSWER_MAP_MAX_CHARS_PER_CHUNK,
                                  overlap_chars: int = ANSWER_MAP_OVERLAP_CHARS) -> list:
    """
    Like _chunk_pages_by_char_budget, but operates on flattened answer
    LINES rather than pages, and uses CHARACTER-based overlap (not a
    fixed line count) between consecutive chunks -- see module-level
    comment above for why line-count overlap was insufficient for long
    answers in practice.
    """
    if not numbered_lines:
        return []

    # FIX: same Devanagari under-chunking problem as the question-paper
    # chunker -- see DEVANAGARI_CHAR_BUDGET_SCALE / _scaled_char_budget.
    # Without this, a Hindi-heavy answer booklet's chunks look
    # "safely sized" by character count alone but actually carry far
    # more real tokens than the same character count of English would,
    # causing every chunk's LLM call to fail as oversized.
    sample_texts = [text for _, text in numbered_lines[:50]]
    effective_max_chars = _scaled_char_budget(sample_texts, max_chars)
    effective_overlap_chars = min(overlap_chars, effective_max_chars // 2)

    chunks = []
    current_chunk = []
    current_chars = 0

    for idx, text in numbered_lines:
        line_chars = len(text)

        if current_chunk and current_chars + line_chars > effective_max_chars:
            chunks.append(current_chunk)

            # Build character-based overlap: walk backward from the
            # end of the current chunk, accumulating whole lines until
            # we've covered at least overlap_chars worth of content --
            # this is robust regardless of how long or short individual
            # OCR lines happen to be.
            overlap = []
            overlap_total = 0
            for item in reversed(current_chunk):
                overlap.insert(0, item)
                overlap_total += len(item[1])
                if overlap_total >= effective_overlap_chars:
                    break

            current_chunk = overlap
            current_chars = sum(len(t) for _, t in current_chunk)

        current_chunk.append((idx, text))
        current_chars += line_chars

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _close_gaps_between_consecutive_answers(ranges: list, log=print,
                                              gap_fill_max_lines: int = 80) -> list:
    """
    FIX: confirmed in real usage -- some answers were missing their
    OPENING lines (the line where the student writes "Ans 6-" / "उत्तर"
    or otherwise restates the question), even though the cross-chunk
    merge above already fixes truncated ENDINGS. The cause here is
    different: the chunk whose view actually contained that opening
    line simply failed to recognize/report it as the start of a NEW
    answer (a plain LLM recall miss for that specific ref, not a
    chunk-boundary issue) -- so the only range that DID get reported
    for that ref came from a LATER chunk, whose own view started well
    after the true opening, and which therefore reported its own
    (too-late) starting point as the start, since that's the earliest
    point IT personally could see.

    In a real sequential answer booklet, one answer's content runs
    straight into the next one's opening -- there is essentially never
    a genuine gap between them (administrative/noise lines in between
    are already filtered out of answer_lines before this point ever
    runs). So any leftover gap between the end of one matched answer
    and the start of the next is, in practice, almost always that
    missed opening -- not real unrelated content.

    This closes any such gap up to `gap_fill_max_lines` lines by
    pulling the NEXT range's start backward to right after the
    previous range's end, recovering the missing opening. Gaps LARGER
    than that are deliberately left untouched and logged instead of
    auto-closed -- a big gap more likely means a genuinely unmapped
    stretch (e.g. an answer to a question outside our official list),
    and blindly absorbing it would risk the exact swallowing bug the
    overlap-resolution safety net elsewhere in this module exists to
    prevent. The very first range is also left untouched (there's no
    earlier boundary to anchor it to) -- if ITS opening was also
    missed, that's logged separately so it's visible rather than
    silently guessed at.
    """
    if not ranges:
        return ranges

    sorted_ranges = sorted(ranges, key=lambda r: r["start_line"])
    closed = [dict(sorted_ranges[0])]

    if sorted_ranges[0]["start_line"] > 0:
        log(
            f"NOTE: the first matched answer (ref {sorted_ranges[0]['ref']}) starts "
            f"at line {sorted_ranges[0]['start_line']}, not line 0 -- if its opening "
            f"was also missed, there's no earlier matched answer to anchor a fix to, "
            f"so this one was left as-is."
        )

    for r in sorted_ranges[1:]:
        r = dict(r)
        prev_end = closed[-1]["end_line"]
        gap = r["start_line"] - prev_end - 1
        if 0 < gap <= gap_fill_max_lines:
            log(
                f"Closing a {gap}-line gap before ref {r['ref']}'s reported start "
                f"(line {r['start_line']}) -- pulling its start back to line "
                f"{prev_end + 1} to recover what was very likely a missed opening."
            )
            r["start_line"] = prev_end + 1
        elif gap > gap_fill_max_lines:
            log(
                f"WARNING: a {gap}-line gap exists before ref {r['ref']}'s reported "
                f"start (line {r['start_line']}) -- left UNCLOSED since it's larger "
                f"than the {gap_fill_max_lines}-line auto-fix threshold; this may be "
                f"a genuinely unmapped stretch of answer text rather than a missed "
                f"opening, worth checking manually."
            )
        closed.append(r)

    return closed


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


# =========================================================
# BOUNDARY REFINEMENT (Groq)
#
# FIX: the earlier gap-closing heuristic (_close_gaps_between_consecutive_answers)
# only ever pulls the NEXT answer's start backward to close a gap --
# it always assumes a gap belongs to the answer AFTER it. That's the
# right call when the next answer's OPENING was missed, but it's the
# WRONG call when the actual problem is that the PREVIOUS answer's
# ENDING was cut short -- in that case the heuristic still closes the
# gap, but credits the recovered lines to the wrong question (the one
# after), while the question whose ending was actually cut still ends
# up incomplete. Confirmed in real usage: truncation kept showing up
# at BOTH ends (some answers missing their start, others missing their
# end) because the heuristic can only ever fix one direction.
#
# There's also a second, distinct failure mode the heuristic can never
# catch at all: a chunk's LLM call can report a prev_end_line and a
# next_start_line that are already adjacent (no visible gap, so nothing
# LOOKS wrong) while BOTH are simply wrong -- e.g. the true boundary is
# 10 lines further along than reported, so the previous answer is
# missing its last few lines AND the next answer is missing its first
# few, with no gap ever appearing in between to signal that anything
# is off.
#
# This adds a precise, per-boundary LLM call that looks at the ACTUAL
# answer text around each adjacent pair of matched answers (not just
# line-number bookkeeping) and asks specifically: given THESE two
# official questions in THIS order, where does the earlier one's
# answer really end, and where does the later one's really begin?
# Because it's grounded in the real content on both sides, it can
# correctly resolve either a missed ending, a missed opening, or both
# at once -- which a one-directional heuristic structurally cannot.
# =========================================================

ANSWER_BOUNDARY_SYSTEM_PROMPT = """You are analyzing a student's handwritten exam answers (OCR'd, line-numbered). You are given two official exam questions, in the order the student answered them in this booklet -- [PREV] (answered first) and [NEXT] (answered immediately after it) -- plus a stretch of the student's answer text that covers the suspected boundary between the two answers. This stretch may include the tail end of PREV's answer, a stretch of text not yet assigned to either, and/or the opening of NEXT's answer.

Your task: read the actual content (not just line numbers) and determine:
- "prev_end_line": the LAST line number that is still genuinely part of PREV's answer -- its reasoning, explanation, examples, or concluding remarks for that specific question.
- "next_start_line": the FIRST line number that is genuinely part of NEXT's answer -- where the student starts addressing NEXT (e.g. restating it, writing "Ans", "उत्तर", a question/answer number, or a clear topic shift to NEXT's subject matter).

These do not have to be adjacent. If there is content in this stretch that doesn't actually belong to either PREV or NEXT, leave a gap between prev_end_line and next_start_line rather than forcing them together. Use the EXACT line numbers as shown in [brackets] -- do not estimate or renumber.

If you believe PREV's answer continues to or past the END of the stretch shown (its true end is not visible here), set "prev_end_line" to the LAST line number shown. If you believe NEXT's answer's true start is not visible in this stretch at all (it starts later, beyond what's shown), set "next_start_line" to a value GREATER than the last line number shown.

Return ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:

{
  "prev_end_line": 42,
  "next_start_line": 47
}"""


def _build_boundary_user_prompt(prev_question: str, next_question: str, numbered_lines: list) -> str:
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in numbered_lines)
    return (
        f"[PREV] {prev_question}\n\n"
        f"[NEXT] {next_question}\n\n"
        f"ANSWER TEXT STRETCH AROUND THE SUSPECTED BOUNDARY (line-numbered):\n{lines_block}"
    )


def _parse_boundary_llm_response(content: str) -> tuple:
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

    if not isinstance(data, dict) or "prev_end_line" not in data or "next_start_line" not in data:
        raise ValueError(
            f"LLM response missing required keys. Got: "
            f"{list(data.keys()) if isinstance(data, dict) else type(data).__name__}"
        )

    return int(data["prev_end_line"]), int(data["next_start_line"])


def refine_answer_boundaries(ranges: list, ref_to_question: dict, answer_lines: list,
                               client, budget: "_TokenBudgetTracker", log,
                               pad_lines: int = 80, max_window_lines: int = 400) -> list:
    """
    Runs a precise LLM boundary check between every pair of adjacent
    matched answers (sorted by document order), and corrects
    prev_end_line / next_start_line based on what the LLM actually
    finds in the surrounding text. See the module-level FIX comment
    above for why this -- rather than the simpler gap-closing
    heuristic alone -- is needed to fix BOTH missed-opening and
    missed-ending cases correctly.

    Falls back to leaving a boundary untouched (with a warning logged)
    if its refinement call fails for any reason -- this is a precision
    pass on top of already-reasonable boundaries, not a required step,
    so a transient failure on one boundary should never abort the run
    or affect any other boundary.
    """
    if len(ranges) < 2:
        return ranges

    refined = [dict(r) for r in sorted(ranges, key=lambda r: r["start_line"])]

    for i in range(len(refined) - 1):
        prev_r = refined[i]
        next_r = refined[i + 1]

        window_start = max(0, prev_r["end_line"] - pad_lines)
        window_end = min(len(answer_lines) - 1, next_r["start_line"] + pad_lines)

        if window_end - window_start + 1 > max_window_lines:
            # The gap (plus padding) is larger than our per-call window
            # cap -- center a capped window on the existing boundary
            # instead of sending the entire span. This still resolves
            # small-to-moderate misses correctly; a very large genuine
            # gap is logged rather than silently guessed at.
            log(
                f"NOTE: boundary window between {prev_r['ref']} and {next_r['ref']} "
                f"would span {window_end - window_start + 1} lines -- larger than the "
                f"{max_window_lines}-line refinement cap, centering a smaller window "
                f"on the existing boundary instead."
            )
            mid = (prev_r["end_line"] + next_r["start_line"]) // 2
            window_start = max(0, mid - max_window_lines // 2)
            window_end = min(len(answer_lines) - 1, mid + max_window_lines // 2)

        numbered_lines = [
            (j, answer_lines[j]) for j in range(window_start, window_end + 1)
            if answer_lines[j].strip() and not is_noise(answer_lines[j])
        ]
        if not numbered_lines:
            continue

        prev_question = ref_to_question.get(prev_r["ref"], "")
        next_question = ref_to_question.get(next_r["ref"], "")
        user_prompt = _build_boundary_user_prompt(prev_question, next_question, numbered_lines)

        try:
            prev_end_line, next_start_line = _call_groq_with_retries(
                client, ANSWER_BOUNDARY_SYSTEM_PROMPT, user_prompt,
                _parse_boundary_llm_response, budget, log
            )
        except Exception as e:
            log(
                f"WARNING: boundary refinement between {prev_r['ref']} and "
                f"{next_r['ref']} failed, keeping existing boundaries: {e}"
            )
            continue

        valid_min, valid_max = numbered_lines[0][0], numbered_lines[-1][0]

        if valid_min <= prev_end_line <= valid_max and prev_end_line >= prev_r["start_line"]:
            if prev_end_line != prev_r["end_line"]:
                log(
                    f"Boundary refinement: adjusting {prev_r['ref']}'s end from "
                    f"line {prev_r['end_line']} to {prev_end_line}"
                )
            prev_r["end_line"] = prev_end_line
        else:
            log(
                f"WARNING: boundary refinement returned out-of-range prev_end_line="
                f"{prev_end_line} for {prev_r['ref']} (window {valid_min}-{valid_max}), ignoring"
            )

        if valid_min <= next_start_line <= valid_max and next_start_line <= next_r["end_line"]:
            if next_start_line != next_r["start_line"]:
                log(
                    f"Boundary refinement: adjusting {next_r['ref']}'s start from "
                    f"line {next_r['start_line']} to {next_start_line}"
                )
            next_r["start_line"] = next_start_line
        else:
            log(
                f"WARNING: boundary refinement returned out-of-range next_start_line="
                f"{next_start_line} for {next_r['ref']} (window {valid_min}-{valid_max}), ignoring"
            )

        # Defensive re-clip: a refinement call could in principle move
        # the two boundaries past each other. If so, prefer NOT letting
        # one swallow the other -- same conservative rule used by the
        # overlap-resolution safety net elsewhere in this module.
        if prev_r["end_line"] >= next_r["start_line"]:
            midpoint = (prev_r["end_line"] + next_r["start_line"]) // 2
            prev_r["end_line"] = max(prev_r["start_line"], midpoint)
            next_r["start_line"] = min(next_r["end_line"], midpoint + 1)

    return refined


QUESTION_PREFIX_RE = re.compile(
    r'^\s*(?:'
    r'Ans(?:wer)?[.\s:-]+\d*[.\s:-]*'        # "Ans 5-", "Answer:", "Ans."
    r'|उत्तर\s*\d*\s*[\-\:]?\s*'              # "उत्तर-", "उत्तर 5-"
    r'|प्र[०.\s]*\d*[.\s:-]*'                 # "प्र. 8.", "प्र० 6."
    r'|प्रश्न[.\s:-]*\d*[.\s:-]*'              # "प्रश्न. 2."
    r'|Q\.?\s*\d+[.\s:-]*'                    # "Q.8", "Q5-"
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


def strip_leading_question_echo(answer_text: str, question_text: str,
                                  max_echo_chars: int = 500,
                                  similarity_threshold: float = 0.55,
                                  word_overlap_threshold: float = 0.6) -> str:
    """
    FIX: strip_question_restatement above only strips short LABELED
    prefixes ("Ans 5-", "उत्तर-", etc.). Confirmed in real usage, a
    different and very common pattern slips straight through that: the
    student simply RETYPES the question itself -- with no label at all,
    sometimes with minor OCR/spelling drift -- as the opening sentence
    of the answer, then starts the real explanation right after. E.g.:

        "Examine the theme of Concealment in Abhignana Shakuntalam /
        The Loom of Time. The theme of Concealment is central to ..."

    Everything up to the first period there is just the question
    restated, not the answer. Once shown next to the official question
    field, this redundant echo makes the question appear twice.

    This walks forward through sentence-ending boundaries near the
    start of the answer, and for each one checks whether the text UP
    TO that point is still substantially similar to the question
    (character-level ratio OR shared-word overlap -- the same two-part
    check the rest of this module already uses for question
    deduplication, since it likewise needs to tolerate small
    OCR/spelling drift without false-matching two genuinely different
    sentences). The boundary stops advancing as soon as a sentence no
    longer looks like part of the echo -- everything from there on is
    treated as the real answer and kept.

    Only operates on a bounded leading window (max_echo_chars), so a
    long answer that happens to share some vocabulary with the
    question deep into its body is never at risk of being cut.
    """
    q_norm = normalize(strip_leading_label(question_text))
    if not q_norm:
        return answer_text

    window = answer_text[:max_echo_chars]
    # FIX: a plain [.!?]\s pattern misses the extremely common case in
    # these quote-and-explain answers where the sentence-ending
    # punctuation sits INSIDE a closing quote mark, e.g. `streets." This
    # passage...` -- the period there is followed by a quote character,
    # not whitespace, so the old pattern never found a boundary there at
    # all and treated the entire quoted answer as a single "sentence",
    # which (being mostly identical to the quoted question) then either
    # matched as one giant echo or failed to match at all depending on
    # length. Allowing an optional closing quote/bracket character
    # between the punctuation and the whitespace fixes this.
    boundaries = [m.end() for m in re.finditer(r'[.!?][\"\'\u2019\u201d\)]?(?:\s|$)', window)]
    if not boundaries:
        return answer_text

    q_words = set(re.findall(r'\w{3,}', q_norm))

    best_cut = 0
    prev_end = 0
    for b in boundaries:
        # FIX: compare each NEW sentence on its own (the slice between
        # the previous boundary and this one), not the cumulative
        # growing prefix from the start. The cumulative version had a
        # real bug: cand_words only ever GROWS as more text gets
        # appended, so the overlap score (matched-question-words /
        # total-question-words) can never decrease once the genuine
        # echo sentence is included -- it stays at or near 1.0 forever,
        # even many unrelated sentences later, because all the
        # question's words are still present somewhere in the
        # ever-growing prefix. That silently kept extending best_cut
        # deep into genuine answer content instead of stopping right
        # after the echo. Scoring each sentence independently means an
        # unrelated sentence's own word set is what gets compared, so
        # the score correctly drops once the echo ends.
        sentence = answer_text[prev_end:b]
        sent_norm = normalize(strip_leading_label(sentence))
        ratio = difflib.SequenceMatcher(None, sent_norm, q_norm).ratio()

        sent_words = set(re.findall(r'\w{3,}', sent_norm))
        overlap = (len(sent_words & q_words) / len(q_words)) if q_words else 0.0

        if ratio >= similarity_threshold or overlap >= word_overlap_threshold:
            best_cut = b
            prev_end = b
        else:
            # Stop extending once a sentence boundary no longer looks
            # like part of the echoed question -- everything from here
            # on is the real answer beginning, not more echo.
            break

    if best_cut:
        remainder = answer_text[best_cut:].strip()
        return remainder if remainder else answer_text
    return answer_text


def map_answers_with_llm(answer_lines: list, questions: list, status_callback=None) -> dict:
    """
    Maps each official question to its verbatim answer text, extracted
    INDEPENDENTLY per question via LLM-identified line boundaries.

    FIX: this used to key the result dict on whatever question TEXT the
    LLM echoed back, then rely on that text matching the ORIGINAL
    question string later in process_pdf()'s qa_map.get(q, "") lookup --
    a plain, EXACT dict lookup. Any discrepancy between the echoed text
    and the original (different punctuation, the prompt's own added
    numbering accidentally retyped, subtle rewording despite
    instructions not to) meant the answer was built correctly but
    silently became UNREACHABLE under the original question's key,
    making it disappear from the final output. This was confirmed as
    a real, structural cause of badly incomplete Q&A mapping in
    production.

    This version has the LLM identify questions by an unambiguous
    REF-A/REF-B/... label (assigned by US, not retyped by the model)
    instead of by echoing question text at all. Resolving a REF label
    back to its question is a deterministic Python list index lookup
    with zero text-matching ambiguity -- the LLM's only job is finding
    line boundaries, never identifying *which* question by text.

    Returns {question_text: answer_text} for every question whose
    answer was found, using the EXACT original question strings from
    `questions` as keys -- guaranteed to match downstream lookups.
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

    # Deterministic REF label <-> question index mapping, built once
    # here and never touched by anything the LLM returns.
    ref_to_question = {f"REF-{chr(65+i)}": q for i, q in enumerate(questions)}

    numbered_lines = list(enumerate(answer_lines))
    chunks = _chunk_lines_by_char_budget(numbered_lines)
    log(f"Split {len(answer_lines)} answer line(s) into {len(chunks)} LLM chunk(s) for answer mapping")

    all_ranges = []  # list of {ref, start_line, end_line}
    chunk_failures = []

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

        # Validate BOTH that the ref label is one we actually issued
        # AND that line numbers are within THIS chunk's actual range
        # (defends against the LLM hallucinating either).
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

    if chunk_failures and not all_ranges:
        # FIX: every chunk failed and nothing was ever mapped. The old
        # behavior here was to silently fall through with an empty
        # qa_map, which process_pdf then reported as a generic "Could
        # not match any questions to answers" -- giving no hint of the
        # REAL cause (e.g. every request being rejected as oversized
        # due to under-estimated Devanagari token usage, or a quota/auth
        # failure). Surfacing the actual first failure reason here
        # turns a confusing dead end into an actionable error.
        raise Exception(
            f"All {len(chunks)} chunk(s) failed during answer mapping -- no answers "
            f"could be extracted. First failure: {chunk_failures[0]}"
        )
    elif chunk_failures:
        log(
            f"NOTE: {len(chunk_failures)} of {len(chunks)} answer-mapping chunk(s) "
            f"failed and were skipped -- results below are PARTIAL."
        )

    # FIX: a long essay answer can span MORE than two chunks (chunk
    # boundary -> overlap -> chunk boundary again), and the previous
    # "keep the longer single range" dedup only ever kept ONE chunk's
    # view of a ref -- even when that view was itself truncated by ITS
    # OWN chunk boundary. Confirmed in real usage: answers ending
    # mid-sentence right at a chunk boundary even though a generous
    # overlap existed, because the overlap makes the SAME ref visible
    # in two (or more) chunks, but each chunk's LLM call can only ever
    # report line numbers within ITS OWN chunk -- so no single chunk's
    # range alone covers the full answer if the answer itself outgrows
    # one chunk's view.
    #
    # Fix: instead of keeping a single winning range per ref, MERGE all
    # ranges found for the same ref across every chunk using standard
    # interval-merge logic. Two ranges for the same ref that overlap
    # (guaranteed whenever an answer spans a chunk boundary, since the
    # overlap region is shared between the two chunks) or sit within a
    # small gap of each other (defends against the overlap window not
    # perfectly bisecting wherever the LLM happened to place its
    # boundary) get combined into one continuous span covering their
    # union. If a ref ends up with multiple genuinely separate
    # (non-adjacent) merged spans -- unusual, but possible if the same
    # ref is mistakenly matched in two unrelated places -- the largest
    # span is kept, preserving the previous behavior's intent for that
    # edge case.
    MERGE_GAP_TOLERANCE_LINES = 5

    ranges_by_ref = {}
    for r in all_ranges:
        ranges_by_ref.setdefault(r["ref"], []).append(r)

    deduped_ranges = []
    for ref, ref_ranges in ranges_by_ref.items():
        ref_ranges = sorted(ref_ranges, key=lambda r: r["start_line"])
        merged = [dict(ref_ranges[0])]
        for r in ref_ranges[1:]:
            last = merged[-1]
            if r["start_line"] <= last["end_line"] + MERGE_GAP_TOLERANCE_LINES:
                last["start_line"] = min(last["start_line"], r["start_line"])
                last["end_line"] = max(last["end_line"], r["end_line"])
            else:
                merged.append(dict(r))

        best = max(merged, key=lambda r: r["end_line"] - r["start_line"])
        deduped_ranges.append(best)

    # FIX: recover answers whose OPENING lines were missed by whichever
    # chunk actually contained them -- see _close_gaps_between_consecutive_answers
    # above for the full rationale. This is a cheap, deterministic
    # pre-pass that catches the obvious/small cases for free before the
    # more precise (but LLM-call-per-boundary) refinement pass below.
    gap_closed_ranges = _close_gaps_between_consecutive_answers(deduped_ranges, log=log)

    # FIX: the heuristic above can only ever pull a gap's content INTO
    # the next answer -- it cannot fix a previous answer's ending being
    # cut short, and it cannot fix boundaries that are already adjacent
    # but simply WRONG on both sides (no visible gap, nothing to close,
    # yet still incomplete). This precise per-boundary LLM pass reads
    # the actual content around each adjacent pair and corrects
    # whichever side (or both) turns out to be off -- see the
    # module-level FIX comment on refine_answer_boundaries for the full
    # rationale.
    log("Refining answer boundaries against actual content (LLM-based)...")
    refined_ranges = refine_answer_boundaries(
        gap_closed_ranges, ref_to_question, answer_lines, client, budget, log
    )

    # HARD SAFETY NET: resolve any remaining overlaps so no answer can
    # ever swallow another's content, regardless of LLM output quality.
    resolved_ranges = _resolve_overlapping_answer_ranges(refined_ranges)

    log(f"Final answer mapping: {len(resolved_ranges)} of {len(questions)} question(s) matched")

    # Slice the ORIGINAL answer_lines verbatim using the resolved ranges
    # -- this is the only place the actual answer text is produced, and
    # it is a pure Python slice, guaranteeing no LLM paraphrasing risk.
    # The dict is keyed on the ORIGINAL canonical question text (looked
    # up deterministically via ref_to_question), guaranteeing it matches
    # whatever process_pdf() looks it up with later.
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
        answer_text = strip_leading_question_echo(answer_text, original_question)
        qa_map[original_question] = answer_text

    # FIX: last-resort recovery for any question that's STILL completely
    # unmatched after all of the above -- confirmed in real usage, this
    # happens when none of the per-chunk calls ever reported a range for
    # that ref at all (not a boundary precision issue, a full recall
    # miss). Since this is now down to a small number of leftover
    # questions, it's cheap to give each one its own dedicated,
    # focused LLM call across the FULL answer text (no chunking) asking
    # specifically and only for that one question -- a much easier task
    # than finding all questions in one chunked pass, so it often
    # recovers what the chunked pass missed.
    unmatched = [q for q in ref_to_question.values() if q not in qa_map]
    if unmatched:
        log(f"Attempting last-resort recovery for {len(unmatched)} still-unmatched question(s)...")
        all_numbered_lines = list(enumerate(answer_lines))
        for q in unmatched:
            single_ref = {"REF-A": q}
            user_prompt = _build_answer_map_user_prompt(all_numbered_lines, [q])
            try:
                result = _call_groq_with_retries(
                    client, ANSWER_MAP_SYSTEM_PROMPT, user_prompt,
                    _parse_answer_map_llm_response, budget, log
                )
            except Exception as e:
                log(f"WARNING: last-resort recovery failed for question {q[:60]!r}: {e}")
                continue

            match = next((r for r in result if r["ref"] == "REF-A"), None)
            if not match:
                log(f"WARNING: last-resort recovery found nothing for question {q[:60]!r}")
                continue

            start, end = match["start_line"], match["end_line"]
            if not (0 <= start <= end < len(answer_lines)):
                log(f"WARNING: last-resort recovery returned out-of-range lines for {q[:60]!r}, skipping")
                continue

            verbatim_lines = [
                answer_lines[j] for j in range(start, end + 1)
                if answer_lines[j].strip() and not is_noise(answer_lines[j])
            ]
            answer_text = " ".join(verbatim_lines).strip()
            answer_text = strip_question_restatement(answer_text)
            answer_text = strip_leading_question_echo(answer_text, q)
            if answer_text:
                qa_map[q] = answer_text
                log(f"Last-resort recovery succeeded for question {q[:60]!r}")

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


CLEANUP_TOKENS_RE = re.compile(
    r'#+|विभाग|प्रश्न',
    re.IGNORECASE
)


def clean_stray_tokens(text: str) -> str:
    """Strips stray '#', 'विभाग' (section), and 'प्रश्न' (question)
    tokens from final output text, then collapses any resulting double
    spaces left behind."""
    if not text:
        return text
    text = CLEANUP_TOKENS_RE.sub('', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n[ \t]+', '\n', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    return text.strip()


# =========================================================
# SUB-PART FORMATTING
#
# FIX: the question-paper reconciliation pass deliberately keeps a
# multi-part question (one official number with several roman-numeral
# sub-parts under it, e.g. "(i) ... (ii) ... (iii) ... (iv) ...") as
# ONE combined question, matching how it's actually presented in the
# question paper. That's the correct CONTENT decision, but showing all
# of its sub-parts run together in a single unbroken paragraph makes
# the question hard to read in the final output. This inserts a
# newline directly before each sub-part marker so every sub-part lands
# on its own line.
# =========================================================

SUBPART_MARKER_RE = re.compile(
    r'\(\s*(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\s*\)',
    re.IGNORECASE
)


def format_subparts_on_new_lines(text: str) -> str:
    """
    Inserts a newline immediately before every roman-numeral sub-part
    marker -- (i), (ii), (iii), (iv), etc, matched case-insensitively
    as the ENTIRE content of a parenthesised group (never a partial
    match), so a real word that happens to sit in parentheses (e.g. a
    book title) is never mistaken for a sub-part marker. The very
    first marker, if the text begins with one, is left exactly where
    it is rather than getting a stray leading blank line.
    """
    def repl(m):
        return m.group(0) if m.start() == 0 else "\n" + m.group(0)

    return SUBPART_MARKER_RE.sub(repl, text)


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

    # FIX: reconcile the extracted candidate questions against the
    # ACTUAL question-paper text itself. This both removes the
    # granularity-duplicates (e.g. one compound multi-part question
    # plus separate candidates for each of its sub-parts) that the
    # earlier fuzzy chunk-dedup can't catch, and guarantees the final
    # question order matches how the questions actually appear in the
    # original paper (monotonic by official numbering) rather than
    # whatever order chunk-by-chunk extraction happened to emit them in.
    qp_text = "\n\n".join(pages[i]["raw_text"] for i in qp_page_indices)
    official_questions = reconcile_questions_with_paper(qp_text, official_questions, status_callback)
    log(f"Questions after reconciliation with actual question paper: {len(official_questions)}")

    answer_page_indices = [i for i in range(len(pages)) if i not in qp_page_indices]
    answer_pages = [pages[i] for i in answer_page_indices]

    log(f"Answer pages: {[i+1 for i in answer_page_indices]}")

    answer_lines = []
    for page in answer_pages:
        for line in page["raw_text"].split("\n"):
            if not is_noise(line):
                answer_lines.append(line)

    log(f"Flattened {len(answer_lines)} answer lines")

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
    #
    # FIX: format_subparts_on_new_lines is applied here, at FINAL
    # display time only -- never to `q` before using it as a qa_map
    # lookup key. qa_map's keys are the exact original (unformatted)
    # strings from `official_questions` (see map_answers_with_llm's
    # docstring for why that guarantee matters); formatting before the
    # lookup would silently break it the same way the old text-echo
    # mismatch bug used to.
    qa_pairs = []
    for q in official_questions:
        raw_answer = qa_map.get(q, "")
        formatted_q = clean_stray_tokens(format_subparts_on_new_lines(q))
        formatted_a = clean_stray_tokens(format_subparts_on_new_lines(raw_answer)) if raw_answer else raw_answer
        qa_pairs.append({
            "question": formatted_q,
            "answer": formatted_a,
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
