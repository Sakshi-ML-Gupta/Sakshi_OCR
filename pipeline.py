import os
import io
import re
import json
import time
import base64
import difflib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz
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
# (unchanged from original)
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
# CHANGED FROM ORIGINAL: the original used a single global
# threading.Lock() around every Groq API call, which SERIALIZED every
# chunk call in the whole process -- including chunks *within the same
# document run*, which is exactly what made the pipeline slow (chunk 2
# could not even start until chunk 1's full round trip finished, even
# though they are logically independent calls).
#
# That lock's original purpose was to protect against a DIFFERENT
# problem: an accidental duplicate full-pipeline invocation (e.g.
# Streamlit rerunning process_pdf() a second time while the first is
# still in flight), which would otherwise let two full runs race for
# the same shared TPM budget.
#
# A bounded Semaphore solves BOTH problems at once:
#   - It caps the TOTAL number of concurrent Groq calls across the
#     whole process (including across accidental duplicate runs) to
#     MAX_CONCURRENT_GROQ_CALLS, so the rate-limit protection is not
#     lost.
#   - Unlike a Lock, it allows up to that many calls to run
#     CONCURRENTLY, which is what lets chunk-level parallelism inside
#     a single run actually reduce wall-clock latency.
# =========================================================

MAX_CONCURRENT_GROQ_CALLS = 4  # tuned conservatively: high enough to
# meaningfully parallelize chunk calls (typically 3-6x latency
# reduction for documents with several chunks), low enough that the
# existing TokenBudgetTracker + Groq's own 429 retry/backoff logic
# can still absorb the burstier concurrent request pattern without
# spiraling into constant rate-limit errors. Raise cautiously if your
# Groq tier has a comfortably higher TPM ceiling.

_groq_concurrency_semaphore = threading.Semaphore(MAX_CONCURRENT_GROQ_CALLS)


# =========================================================
# STREAMLIT THREAD-CONTEXT PROPAGATION
#
# FIX: parallelizing chunk calls via ThreadPoolExecutor (see above)
# means `status_callback` -- passed in from the caller, and in
# Streamlit apps typically touching `st.session_state` and/or a
# `st.empty()` placeholder to show live logs -- now gets INVOKED FROM
# WORKER THREADS instead of the main thread. Streamlit's session state
# is only attached to the thread that Streamlit itself runs the script
# on; any other thread touching `st.session_state` raises exactly the
# "st.session_state has no attribute ..." / missing ScriptRunContext
# error confirmed in real usage, because Streamlit has no way to know
# which session a plain background thread belongs to.
#
# This is a generic module with no hard dependency on Streamlit (it's
# also used outside Streamlit, e.g. via a CLI or plain script), so it
# can't unconditionally import streamlit. Instead: capture the current
# thread's Streamlit ScriptRunContext (if any -- this call itself is
# safe to make from a non-Streamlit context, it just returns None) on
# the MAIN thread right before dispatching work, then attach that same
# context to each worker thread before it runs. This is Streamlit's own
# documented pattern for using session state safely from background
# threads. Outside Streamlit (ctx is None), this is a complete no-op.
# =========================================================

def _get_streamlit_ctx():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx()
    except Exception:
        return None


def _with_streamlit_ctx(fn, ctx):
    """Wraps fn so that, when it runs on a NEW worker thread, that
    thread first gets the captured Streamlit ctx attached -- making it
    safe for fn (or anything it calls, like status_callback) to touch
    st.session_state / st widgets without raising a missing-context
    error. A complete no-op if ctx is None (i.e. not running under
    Streamlit, or context capture failed for any reason)."""
    if ctx is None:
        return fn

    def wrapper(*args, **kwargs):
        try:
            from streamlit.runtime.scriptrunner import add_script_run_ctx
            add_script_run_ctx(threading.current_thread(), ctx)
        except Exception:
            pass
        return fn(*args, **kwargs)

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
# OCR -- Mistral OCR (mistral-ocr-latest) via the official SDK
#
# CHANGED FROM ORIGINAL: switched from Datalab (Chandra model, a raw
# httpx submit-then-poll flow) to Mistral's OCR API.
#
# Two structural simplifications fall out of this switch:
#   1. Mistral's OCR endpoint is SYNCHRONOUS (results in seconds for
#      the standard endpoint) -- no submit/poll loop is needed at all,
#      removing an entire class of "polling timed out after 5 minutes"
#      failures that could happen with Datalab.
#   2. Mistral's response already returns page-separated markdown
#      (`response.pages[i].markdown`), so the fragile regex-based
#      `_split_paginated_markdown` / `PAGE_BREAK_PATTERNS` machinery
#      that tried to re-derive page boundaries from a single flat
#      markdown blob is no longer needed at all -- Mistral does that
#      splitting for us, correctly, every time.
#
# The rest of the pipeline (build_ocr_json, identify_questions_with_llm,
# map_answers_with_llm, etc.) is untouched: it only cares about the
# `pages` list shape ({"page_number": int, "raw_text": str}), which is
# preserved exactly.
# =========================================================

MISTRAL_OCR_MODEL = "mistral-ocr-latest"
MISTRAL_OCR_MAX_MB = 50  # Mistral's documented per-request file size limit


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

    api_key = get_api_key("MISTRAL_API_KEY")
    if not api_key:
        raise Exception("MISTRAL_API_KEY not found in secrets or environment")

    size_mb = len(file_content) / (1024 * 1024)
    if size_mb > MISTRAL_OCR_MAX_MB:
        raise Exception(
            f"File is {size_mb:.1f}MB, which exceeds the {MISTRAL_OCR_MAX_MB}MB "
            f"Mistral OCR upload limit. Try compressing the PDF or splitting it "
            f"into smaller files before uploading."
        )

    # CHANGED: mistralai SDK v2.0.0 (released March 2026) moved the
    # Mistral client class from the top-level `mistralai` package to
    # `mistralai.client` -- `from mistralai import Mistral` now raises
    # "cannot import name 'Mistral' from 'mistralai' (unknown location)"
    # on v2.x, even though it's still correct for v1.x. Trying both
    # import paths makes this work regardless of which SDK version got
    # installed in your environment (e.g. Streamlit Cloud pulling the
    # latest release without a version pin).
    try:
        from mistralai import Mistral
    except ImportError:
        from mistralai.client import Mistral

    client = Mistral(api_key=api_key)

    log(f"Submitting document to Mistral OCR ({MISTRAL_OCR_MODEL})... ({size_mb:.1f}MB)")

    # Embed the PDF directly as a base64 data URI -- this is the
    # documented approach for local files and avoids a separate
    # upload-then-get-signed-url round trip for typical exam-booklet
    # sized documents. If you routinely process very large files
    # (100+ pages / near the 50MB ceiling), switch to
    # client.files.upload(...) + client.files.get_signed_url(...) and
    # pass that signed URL as document_url instead -- ask if you need
    # that variant.
    encoded = base64.b64encode(file_content).decode("utf-8")
    data_uri = f"data:application/pdf;base64,{encoded}"

    try:
        ocr_response = client.ocr.process(
            model=MISTRAL_OCR_MODEL,
            document={
                "type": "document_url",
                "document_url": data_uri,
            },
            include_image_base64=False,  # we only need text -- skip
                                          # embedded image payloads to
                                          # keep the response small/fast
        )
    except Exception as e:
        raise Exception(f"Mistral OCR request failed: {e}") from e

    log("OCR complete -- parsing pages...")

    mistral_pages = getattr(ocr_response, "pages", None)
    if not mistral_pages:
        raise Exception("Mistral OCR returned no pages")

    pages = []
    for i, p in enumerate(mistral_pages):
        page_index = getattr(p, "index", None)
        markdown = getattr(p, "markdown", "") or ""
        # Mistral's page index is 0-based (matches the 0-based `pages`
        # request parameter documented for this endpoint); fall back
        # to enumeration order if a page is ever missing an index.
        page_number = (page_index + 1) if isinstance(page_index, int) else i + 1
        pages.append({
            "page_number": page_number,
            "raw_text": markdown,
        })

    if not any(p["raw_text"].strip() for p in pages):
        raise Exception("Mistral OCR returned empty markdown output for all pages")

    log(f"OCR done -- {len(pages)} page(s) extracted")

    usage = getattr(ocr_response, "usage_info", None)
    pages_processed = getattr(usage, "pages_processed", None) if usage else None
    if pages_processed and pages_processed != len(pages):
        log(
            f"NOTE: Mistral reports {pages_processed} page(s) processed, but "
            f"{len(pages)} page object(s) were parsed from the response -- "
            f"check the raw response if this looks wrong."
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
# CHANGED FROM ORIGINAL: now used from MULTIPLE THREADS concurrently
# (see MAX_CONCURRENT_GROQ_CALLS above), so every method that reads or
# mutates `self.events` now takes an internal lock. The original
# single-threaded version had no such protection, which would have
# caused a real race condition (lost updates / inconsistent budget
# reads) the moment more than one chunk call could be in flight at
# once.
# =========================================================

class _TokenBudgetTracker:
    def __init__(self, tpm_limit=TPM_LIMIT, safety_fraction=TPM_SAFETY_FRACTION):
        import collections
        self.tpm_limit = tpm_limit
        self.safe_limit = tpm_limit * safety_fraction
        self.events = collections.deque()  # (timestamp, tokens)
        self._lock = threading.Lock()

    def _prune(self, now=None):
        now = now if now is not None else time.monotonic()
        while self.events and now - self.events[0][0] >= 60:
            self.events.popleft()

    def used_in_window(self, now=None) -> int:
        with self._lock:
            now = now if now is not None else time.monotonic()
            self._prune(now)
            return sum(tok for _, tok in self.events)

    def wait_if_needed(self, upcoming_tokens: int, log=print):
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            used = sum(tok for _, tok in self.events)
            projected = used + upcoming_tokens

            if projected <= self.safe_limit:
                return  # comfortably under budget -- no wait

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
        # sleep OUTSIDE the lock so other threads can still read/update
        # the tracker (e.g. record actual usage from a 429) while this
        # one waits -- holding the lock during sleep would effectively
        # re-serialize everything and defeat the point of concurrency.
        time.sleep(wait_s)

    def record_usage(self, tokens: int):
        with self._lock:
            self.events.append((time.monotonic(), tokens))

    def record_actual_from_error(self, used: int, limit: int):
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            current = sum(tok for _, tok in self.events)
            if used > current:
                self.events.append((now, used - current))
            if limit:
                self.tpm_limit = limit
                self.safe_limit = limit * TPM_SAFETY_FRACTION

    def reset_window(self):
        with self._lock:
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
            self.reset_window()


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

    CHANGED FROM ORIGINAL: the API call itself now runs inside
    `_groq_concurrency_semaphore` (bounded parallelism) instead of
    `_groq_call_lock` (full serialization) -- see the CONCURRENCY GUARD
    section above for why. Everything else (rate-limit parsing, TPD
    fail-fast, auth fail-fast, pacing) is unchanged.
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
            with _groq_concurrency_semaphore:
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
    """
    CHANGED FROM ORIGINAL: chunk calls are now dispatched concurrently
    via ThreadPoolExecutor instead of one-by-one in a for loop. Each
    chunk's LLM call and its page-number-recovery post-processing are
    independent of every other chunk, so this is a pure latency win
    with no change to per-chunk logic or output.
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
    log(f"Split {len(pages)} page(s) into {len(chunks)} LLM chunk(s) to respect token limits")

    valid_page_numbers = {p["page_number"] for p in pages}
    max_page_number = max(valid_page_numbers) if valid_page_numbers else 0

    def process_one_chunk(i, chunk):
        page_nums_in_chunk = [p["page_number"] for p in chunk]
        log(f"Asking LLM to analyze chunk {i+1}/{len(chunks)} (pages {page_nums_in_chunk})...")

        try:
            qp_pages_1based, _questions = _call_groq_for_chunk(client, chunk, budget, log)
        except Exception as e:
            log(f"WARNING: chunk {i+1}/{len(chunks)} question-identification failed, skipping: {e}")
            return i, None, str(e)

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
        return i, qp_pages_1based, None

    ordered_results = [None] * len(chunks)
    chunk_failures = []

    ctx = _get_streamlit_ctx()  # captured on the calling (main) thread
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_GROQ_CALLS) as executor:
        futures = [
            executor.submit(_with_streamlit_ctx(process_one_chunk, ctx), i, chunk)
            for i, chunk in enumerate(chunks)
        ]
        for future in as_completed(futures):
            i, qp_pages_1based, err = future.result()
            if err is not None:
                chunk_failures.append(err)
            else:
                ordered_results[i] = qp_pages_1based

    chunk_results = [(pages_1based, []) for pages_1based in ordered_results if pages_1based is not None]

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
        lengths_only = [length for _, length in qp_page_lengths]
        median_length = sorted(lengths_only)[len(lengths_only) // 2]
        for page_idx, length in qp_page_lengths:
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
- An answer's content ends at the LAST line that is still part of that answer's reasoning/explanation, RIGHT BEFORE the next answer begins (whether or not the next answer is in your list of official questions).
- If a question's answer is genuinely not present anywhere in the text shown, do NOT invent a range -- omit that REF entirely from your output. It may appear in a different chunk of the document.
- Each REF's range must NOT overlap with another REF's range. If you are unsure exactly where one answer ends and the next begins, prefer ending the EARLIER answer sooner rather than letting it swallow content that belongs to a later answer -- a short correct answer is far more useful than a long answer that incorrectly absorbed unrelated content.
- Use the line numbers EXACTLY as given in [brackets] -- do not estimate, guess, or renumber.
- Use the EXACT REF label (e.g. "REF-A") to identify each question. Do NOT retype or paraphrase the question text itself -- the REF label is all that's needed.
- NOTE: student answers usually appear in roughly the SAME sequential order as the official questions are listed above. If a question's answer text doesn't obviously restate or resemble the question's own wording, use its POSITION in the question list -- relative to answers you can already identify with confidence for its neighboring questions -- as a secondary signal for where that answer likely falls in the text.
- CRITICAL for SIMILAR questions: if two or more official questions share a lot of general topic vocabulary (e.g. both are about the same author, text, historical period, or broad theme), do NOT rely on that shared vocabulary to tell their answers apart -- it will make BOTH answers look like a match for either question. Instead, look for whatever is UNIQUELY different between those specific questions (a particular sub-topic, a named example, a specific angle of comparison, a distinct instruction like "compare" vs "contrast" vs "list") and use THAT to decide which text belongs to which. If you genuinely cannot tell them apart from content alone, fall back to the sequential-position signal above rather than assigning most or all of the shared content to only one of them -- a wrong split that gives each question SOME of its rightful content is better than one question getting everything and the other getting nothing.

Return ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:

{
  "answers": [
    {"ref": "REF-A", "start_line": 12, "end_line": 18},
    {"ref": "REF-B", "start_line": 19, "end_line": 25}
  ]
}

If NONE of the official questions' answers appear in the text shown, return {"answers": []} -- that is a valid and expected result for a chunk that doesn't contain any of these answers."""


def _build_answer_map_user_prompt(numbered_lines: list, questions: list, common_words: frozenset = frozenset()) -> str:
    questions_block = "\n".join(
        f"[REF-{chr(65+i)}] {q}" for i, q in enumerate(questions)
    )
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in numbered_lines)

    shared_vocab_note = ""
    if common_words:
        shared_vocab_note = (
            f"\n\nNOTE: the following words appear in TWO OR MORE of the official "
            f"questions above, so they CANNOT be used to distinguish which question "
            f"an answer belongs to -- ignore them when matching, and instead look for "
            f"words/phrases that are UNIQUE to just one question: "
            f"{', '.join(sorted(common_words))}"
        )

    return (
        f"OFFICIAL QUESTIONS (each tagged with its own [REF-X] label -- "
        f"use the REF label, not retyped question text, to identify which "
        f"question an answer belongs to):\n{questions_block}{shared_vocab_note}\n\n"
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

# NEW: overlap between consecutive answer-mapping line-chunks, mirroring
# CHUNK_OVERLAP_PAGES used for page-chunking above. See the note on
# _chunk_lines_by_char_budget below for why this matters for the
# merge/skip/no-match bugs.
#
# INCREASED (this round -- addresses "last pages/paragraphs not
# mapping, answer becomes half"): 8 lines was too thin a safety margin
# specifically for FORCED breaks (should_force_break_absolute), which
# happen when no genuine answer-start is ever found (e.g. the true
# LAST answer in the document, which by definition has no "next
# question" to bound it) -- these are exactly the highest-risk splits,
# since they land at an arbitrary character-count boundary with zero
# semantic justification, right in the middle of real content.
CHUNK_OVERLAP_LINES = 15

# A forced break (hit ANSWER_MAP_ABSOLUTE_MAX_CHARS with no genuine
# boundary in sight) gets a bigger overlap than a normal answer-start
# break, since it's the case most likely to be splitting genuine
# content mid-answer rather than at an intentional boundary.
FORCED_BREAK_OVERLAP_LINES = 40

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


def _distinctive_words(text: str, max_words: int = 20, extra_stopwords: frozenset = frozenset()) -> list:
    words = re.findall(r'[a-z]{3,}', _normalize_for_overlap_match(text))[:max_words]
    return sorted(set(w for w in words if w not in _QUESTION_STOPWORDS and w not in extra_stopwords))


def _compute_cross_question_common_words(questions: list, max_words: int = 20) -> frozenset:
    """
    FIX (this round -- addresses "2 similar questions ka answer mix ho
    raha hai"): `_distinctive_words` only filtered GENERIC stopwords
    (how/why/explain/discuss/etc.), not words that are specific to this
    document's own question set but happen to be shared between two or
    more of ITS questions. When two official questions are worded
    similarly -- e.g. both about the same author, text, or broad topic
    -- those shared topic words pass right through the generic filter
    and get counted as "distinctive" for BOTH questions. The result: a
    line that's actually part of question B's answer scores a fuzzy
    match against question A too (since they share vocabulary), so the
    boundary detector either fails to recognize a genuine new start for
    B (the line looks like "more of A" instead), or -- just as bad --
    randomly attributes B's opening lines to A. Either way, one
    question's answer swallows the other's, and the other gets partial
    or no match at all -- exactly the reported symptom.

    Fix: compute which SIGNIFICANT words are shared across 2+ different
    questions IN THIS DOCUMENT'S OWN question list, and treat those as
    additional, document-specific stopwords on top of the generic list
    -- so only words that are actually UNIQUE to one particular
    question get used to recognize its answer boundary. This adapts
    automatically to whatever questions are actually present, rather
    than needing hardcoded topic words.
    """
    from collections import Counter
    doc_word_counts = Counter()
    for q in questions:
        words_in_q = set(re.findall(r'[a-z]{3,}', _normalize_for_overlap_match(q))[:max_words])
        for w in words_in_q:
            if w not in _QUESTION_STOPWORDS:
                doc_word_counts[w] += 1
    return frozenset(w for w, c in doc_word_counts.items() if c >= 2)


def _line_starts_new_answer_for_question(line: str, questions: list, min_fraction: float = 0.5,
                                            common_words: frozenset = frozenset()):
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
        q_distinctive = _distinctive_words(q, extra_stopwords=common_words)
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


def _chunk_lines_by_char_budget(numbered_lines: list, questions: list,
                                  max_chars: int = ANSWER_MAP_MAX_CHARS_PER_CHUNK,
                                  absolute_max_chars: int = ANSWER_MAP_ABSOLUTE_MAX_CHARS,
                                  overlap_lines: int = CHUNK_OVERLAP_LINES) -> list:
    """
    Answer-boundary-aware chunking.

    FIX (this round -- addresses "merge / skip / no-match" reports):
    the previous version started every new chunk EMPTY right after a
    break, with zero overlap into the previous chunk. Page-chunking
    (see `_chunk_pages_by_char_budget` above) already carries
    CHUNK_OVERLAP_PAGES=1 forward for exactly this reason, but the
    line-chunker had no equivalent -- a real, confirmed asymmetry.

    Consequence of that asymmetry: `_line_starts_new_answer_for_question`
    is a heuristic (formal label OR fuzzy word-overlap with the
    question text), not a guarantee. When it mis-detects a boundary --
    either missing a genuine answer-start (student paraphrased instead
    of restating) or firing on a false positive -- the chunk break can
    land INSIDE an answer. With zero overlap, each side of that split
    answer is only ever seen by ONE chunk's LLM call, so:
      - the chunk containing the FIRST half returns a truncated range
        (it never saw the rest, since the chunk simply ends)
      - the chunk containing the SECOND half returns a range that
        looks like an unrelated fresh start
      - the old "keep whichever range is longer" dedup then picks ONE
        of the two truncated halves and throws the other away --
        producing exactly the observed "answer skipped" or "answer cut
        short" symptom.

    FIX: carry the last `overlap_lines` lines of a chunk forward into
    the START of the next chunk. If the break point was wrong, the
    overlap gives the NEXT chunk's LLM call a chance to see the tail of
    the PREVIOUS answer too -- so at least one of the two chunk calls
    ends up seeing the complete answer. Combined with the improved
    merge logic in `map_answers_with_llm` below (which now MERGES
    overlapping/adjacent ranges for the same ref instead of just
    picking the longer one), this recovers the full answer instead of
    silently keeping a truncated half.
    """
    if not numbered_lines:
        return []

    # Computed ONCE for the whole chunking pass (not per-line) -- see
    # _compute_cross_question_common_words docstring for why this
    # matters when two official questions are worded similarly.
    common_words = _compute_cross_question_common_words(questions)

    chunks = []
    current_chunk = []
    current_chars = 0
    past_target = False
    current_question_idx = None

    for idx, text in numbered_lines:
        line_chars = len(text)

        if current_chunk and current_chars + line_chars > max_chars:
            past_target = True

        matched_q_idx = _line_starts_new_answer_for_question(text, questions, common_words=common_words)
        is_genuine_new_start = matched_q_idx is not None and (
            matched_q_idx == -1 or matched_q_idx != current_question_idx
        )

        should_break_at_answer_start = past_target and is_genuine_new_start
        should_force_break_absolute = (
            current_chunk and current_chars + line_chars > absolute_max_chars
        )

        if should_break_at_answer_start or should_force_break_absolute:
            chunks.append(current_chunk)
            # FIX: a forced break (no genuine boundary found, just hit
            # the hard char ceiling) gets a BIGGER overlap than a
            # normal boundary break -- it's the case most likely to be
            # slicing through the middle of real content, most notably
            # the document's final answer, which has no "next question"
            # to ever trigger a genuine boundary break at all.
            this_break_overlap = (
                FORCED_BREAK_OVERLAP_LINES if should_force_break_absolute and not should_break_at_answer_start
                else overlap_lines
            )
            overlap = current_chunk[-this_break_overlap:] if this_break_overlap > 0 else []
            current_chunk = list(overlap)
            current_chars = sum(len(t) for _, t in current_chunk)
            past_target = False

        if is_genuine_new_start and matched_q_idx != -1:
            current_question_idx = matched_q_idx

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


def _merge_or_pick_best_range(r1: dict, r2: dict, gap_tolerance: int = 12) -> dict:
    lo = min(r1["start_line"], r2["start_line"])
    hi = max(r1["end_line"], r2["end_line"])
    gap = max(r2["start_line"] - r1["end_line"], r1["start_line"] - r2["end_line"])

    # NEW: only auto-merge when the two ranges genuinely look like the
    # SAME continuous answer split across the CHUNK_OVERLAP_LINES zone
    # (i.e. real overlap or a tiny adjacent gap). If they're merely
    # "close enough" per the old flat tolerance, that's exactly the
    # case that lets a similar-question mismatch quietly merge two
    # different answers into one. Require actual overlap (gap <= 0) OR
    # a very small gap for the union; otherwise fall back to picking
    # the longer, more confident range.
    if gap <= 0 or gap <= min(gap_tolerance, 3):
        merged = dict(r1)
        merged["start_line"], merged["end_line"] = lo, hi
        return merged

    len1 = r1["end_line"] - r1["start_line"]
    len2 = r2["end_line"] - r2["start_line"]
    return r1 if len1 >= len2 else r2


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
    """
    FIX (structure-preservation round): the previous implementation did
    `answer_text.split()` (which splits on ANY whitespace, including
    newlines) and then rebuilt the retained portion with
    `" ".join(...)` -- so even though the CALLER now preserves the
    document's original line breaks (see map_answers_with_llm), this
    function was silently flattening them right back into one run-on
    line whenever it stripped an echoed question prefix. That's exactly
    what corrupted structured content (tables, numbered/lettered lists,
    parenthetical groupings) into garbled text.

    This version locates the same echoed-prefix word window as before,
    but instead of rebuilding from split words, it works out the
    CHARACTER offset where that prefix ends in the ORIGINAL string and
    slices from there -- so every line break, space, and piece of
    formatting in the retained portion is preserved byte-for-byte,
    exactly "as it is, jaise PDF mein hai".
    """
    question_core = _PARENT_INSTRUCTION_PREFIX_RE.sub('', question_text).strip()
    if not question_core:
        question_core = question_text

    q_norm = _normalize_for_echo_compare(question_core)
    q_word_count = len(q_norm.split())
    if q_word_count == 0:
        return answer_text

    # Word spans as (start_char, end_char) positions in the ORIGINAL
    # answer_text -- this lets us slice by character index later
    # instead of rebuilding the string from split tokens.
    word_spans = [m.span() for m in re.finditer(r'\S+', answer_text)]
    if not word_spans:
        return answer_text

    min_n = max(3, int(q_word_count * 0.7))
    max_n = min(len(word_spans), int(q_word_count * 1.3) + 2)

    best_strip_end_char = 0
    best_ratio = 0.0

    for n in range(min_n, max_n + 1):
        if n > len(word_spans):
            break
        prefix_end_char = word_spans[n - 1][1]
        prefix = answer_text[:prefix_end_char]
        prefix_norm = _normalize_for_echo_compare(prefix)
        ratio = difflib.SequenceMatcher(None, prefix_norm, q_norm).ratio()
        if ratio >= 0.75 and ratio > best_ratio:
            best_ratio = ratio
            best_strip_end_char = prefix_end_char

    if best_strip_end_char > 0:
        # Strip only the echoed prefix; keep everything after it -- and
        # ANY newlines/formatting within it -- exactly as originally
        # written. Only trim the handful of whitespace characters
        # immediately bridging the echo and the real content, not
        # collapse internal structure.
        remaining = answer_text[best_strip_end_char:]
        remaining = remaining.lstrip(' \t')
        remaining = re.sub(r'^\n+', '', remaining)
        remaining = re.sub(r'^(?:Answer\s*[-:]\s*)', '', remaining, flags=re.IGNORECASE)
        return remaining

    return answer_text


def map_answers_with_llm(answer_lines: list, questions: list, status_callback=None) -> dict:
    """
    Maps each official question to its verbatim answer text.

    CHANGED FROM ORIGINAL (latency): chunk calls now run concurrently
    via ThreadPoolExecutor instead of sequentially.

    CHANGED FROM ORIGINAL (merge/skip/no-match fixes):
      1. `_chunk_lines_by_char_budget` now overlaps consecutive chunks
         (CHUNK_OVERLAP_LINES) so a boundary-split answer is fully
         visible to at least one chunk's LLM call.
      2. When the same REF is found in more than one chunk, ranges are
         now MERGED (via `_merge_or_pick_best_range`) instead of just
         keeping the longer one -- recovering the full answer instead
         of silently discarding a truncated half.
      3. The system prompt now gives the model a positional fallback
         signal (answers usually follow question order) for cases
         where an answer doesn't textually resemble its question at
         all, which was a real cause of "no match found" for otherwise
         genuinely-present answers.
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

    ref_to_question = {f"REF-{chr(65+i)}": q for i, q in enumerate(questions)}

    numbered_lines = list(enumerate(answer_lines))
    chunks = _chunk_lines_by_char_budget(numbered_lines, questions)
    log(f"Split {len(answer_lines)} answer line(s) into {len(chunks)} LLM chunk(s) for answer mapping")

    common_words_for_prompt = _compute_cross_question_common_words(questions)

    def process_one_chunk(i, chunk):
        line_range = f"{chunk[0][0]}-{chunk[-1][0]}" if chunk else "empty"
        log(f"Asking LLM to map answers in chunk {i+1}/{len(chunks)} (lines {line_range})...")

        user_prompt = _build_answer_map_user_prompt(chunk, questions, common_words_for_prompt)
        try:
            chunk_ranges = _call_groq_with_retries(
                client, ANSWER_MAP_SYSTEM_PROMPT, user_prompt,
                _parse_answer_map_llm_response, budget, log
            )
        except Exception as e:
            log(f"WARNING: chunk {i+1}/{len(chunks)} answer-mapping failed, skipping: {e}")
            return i, [], str(e), False

        valid_indices = {idx for idx, _ in chunk}
        min_idx, max_idx = min(valid_indices), max(valid_indices)
        accepted = []
        for r in chunk_ranges:
            if r["ref"] not in ref_to_question:
                log(f"WARNING: discarding answer mapping with unknown ref {r['ref']!r}")
                continue
            if min_idx <= r["start_line"] <= max_idx and min_idx <= r["end_line"] <= max_idx:
                accepted.append(r)
            else:
                log(
                    f"WARNING: discarding out-of-range answer mapping for "
                    f"{r['ref']}: lines {r['start_line']}-{r['end_line']} "
                    f"outside this chunk's range {min_idx}-{max_idx}"
                )

        log(f"Chunk {i+1}/{len(chunks)}: mapped {len(accepted)} answer(s)")
        return i, accepted, None, (len(chunk_ranges) == 0)

    all_ranges = []
    chunk_failures = []
    chunk_zero_matches = 0

    ctx = _get_streamlit_ctx()  # captured on the calling (main) thread
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_GROQ_CALLS) as executor:
        futures = [
            executor.submit(_with_streamlit_ctx(process_one_chunk, ctx), i, chunk)
            for i, chunk in enumerate(chunks)
        ]
        for future in as_completed(futures):
            i, accepted, err, was_zero = future.result()
            if err is not None:
                chunk_failures.append(err)
            else:
                all_ranges.extend(accepted)
                if was_zero:
                    chunk_zero_matches += 1

    # Merge (not just "pick longer") ranges for the same ref across
    # chunks -- see _merge_or_pick_best_range docstring for why this
    # matters now that chunks overlap.
    best_by_ref = {}
    for r in all_ranges:
        existing = best_by_ref.get(r["ref"])
        if existing is None:
            best_by_ref[r["ref"]] = r
        else:
            best_by_ref[r["ref"]] = _merge_or_pick_best_range(existing, r)

    deduped_ranges = list(best_by_ref.values())

    resolved_ranges = _resolve_overlapping_answer_ranges(deduped_ranges)

    # FIX (this round -- addresses "last pages/paragraphs not mapping,
    # answer becomes half"): the range with the HIGHEST start_line is,
    # by construction, the LAST answer in the document -- there is no
    # question after it to give the chunker a genuine boundary to break
    # on, and no LATER range for the overlap-resolution step above to
    # clip against. That combination means this last answer is the one
    # most likely to get an artificially early end_line: a forced
    # (absolute-max) chunk split can cut it off mid-answer, or the LLM
    # itself may simply stop at a conservative point if it isn't sure
    # the answer keeps going in the (limited) text it was shown.
    #
    # Since NOTHING else can legitimately claim lines that come after
    # this last matched range and before the end of the document (there
    # is no other question left to answer), any such trailing lines are
    # safe to fold into this last range rather than silently dropping
    # them. This directly recovers the "answer got cut in half at the
    # very end" symptom.
    if resolved_ranges:
        last_range = max(resolved_ranges, key=lambda r: r["start_line"])
        true_end = len(answer_lines) - 1
        if last_range["end_line"] < true_end:
            # Only extend through genuinely meaningful trailing content --
            # stop early if the remaining tail is entirely blank/noise
            # (e.g. trailing signature lines, blank OCR artifacts), so we
            # don't glue unrelated boilerplate onto the real answer.
            trailing = [
                answer_lines[j] for j in range(last_range["end_line"] + 1, true_end + 1)
            ]
            if any(t.strip() and not is_noise(t) for t in trailing):
                log(
                    f"Extending last matched answer ({last_range['ref']}) from "
                    f"line {last_range['end_line']} to {true_end} -- no later "
                    f"question exists to claim these trailing lines, so they "
                    f"must belong to this answer."
                )
                last_range["end_line"] = true_end

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
        # FIX: previously joined with a single space (" ".join(...)),
        # which FLATTENS the original OCR'd line structure into one
        # run-on string -- destroying tables, numbered/lettered lists,
        # and parenthetical groupings exactly as printed in the PDF
        # (e.g. a Devanagari consonant-group listing like "(क) ख (ख)
        # ग (ग)..." across several lines came out garbled once every
        # line break was replaced by a plain space). Joining with "\n"
        # instead preserves the document's own line breaks verbatim --
        # "as it is, jaise PDF mein hai" -- with no reformatting.
        answer_text = "\n".join(verbatim_lines).strip()
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
    r'|^\s*\d{1,3}\s*$'
    # NEW: generic date formats commonly printed on assignment cover
    # sheets / margins (e.g. "12/03/2024", "12-03-24", "12.03.2024") --
    # a whole line that is JUST a date, nothing else.
    r'|^\s*\d{1,2}\s*[/\-.]\s*\d{1,2}\s*[/\-.]\s*\d{2,4}\s*$'
    # NEW: generic "Page X" / "Page X of Y" lines.
    r'|^\s*Page\s*\d+(?:\s*(?:of|\/)\s*\d+)?\s*$'
    # NEW (this round -- "answer main Section/bhag/prashna/Q/Ans hata
    # do jo unnecessary ho"): a line that is JUST a bare section/part/
    # question/answer LABEL with no other real content -- e.g. a
    # printed "Section - A" or "भाग-1" header repeated in the margin,
    # or a stray "Q.5" / "Ans-" left on its own line after slicing.
    # Deliberately whole-line-only ($  anchored) so genuine prose that
    # happens to START with these words (e.g. "Answer: the theme of...")
    # is never touched -- only lines that are NOTHING BUT the label.
    r'|^\s*Section\s*[-:]?\s*[A-Za-z0-9]{0,3}\s*$'
    r'|^\s*(?:भाग|अनुभाग)\s*[-:]?\s*[०-९0-9]{0,3}\s*$'
    r'|^\s*Ans(?:wer)?\s*(?:\d+\s*[.:\-]?\s*\d*|[.:\-]\s*\d*)\s*$'
    r'|^\s*उत्तर\s*(?:\d+\s*[\-\:]?\s*\d*|[\-\:]\s*\d*)\s*$'
    r'|^\s*(?:प्र|प्रश्न)[०.\s]*(?:\d+[.\s:-]*\d*|[.\s:-]+\d*)\s*$'
    r'|^\s*Q\s*(?:\d+[.\s:-]*\d*|\.\s*\d*)\s*$)',
    re.IGNORECASE
)


def is_noise(line: str) -> bool:
    return bool(NOISE_RE.search(line))


# =========================================================
# PAGE MARGIN / BOILERPLATE DETECTION
#
# NEW (this round -- "pages main margin bhi add karo, company
# name/date/page number answer main aa raha hai"): fixed regexes (like
# NOISE_RE above) can only catch noise patterns we already know to
# expect. A scanned assignment booklet's printed letterhead/margin
# (university name, programme code, a repeated footer, etc.) is
# DOCUMENT-SPECIFIC text that no generic regex can anticipate -- but it
# has one very reliable property: it is the letterhead, so it repeats,
# near-verbatim, in the same TOP/BOTTOM margin position on most pages.
#
# This detects that repetition directly instead of guessing the exact
# wording: normalize each candidate margin line (lowercase, numbers
# collapsed to a placeholder so "Page 4" and "Page 17" count as the
# SAME recurring pattern, punctuation stripped), then count how many
# distinct pages each normalized line appears on. Anything that shows
# up in the margin of a large fraction of pages is almost certainly
# printed boilerplate, not the student's own answer content, and gets
# excluded when the answer text is assembled.
# =========================================================

def _normalize_margin_line(line: str) -> str:
    text = line.strip().lower()
    text = re.sub(r'\d+', '#', text)  # "Page 4" and "Page 17" -> same key
    text = re.sub(r'[^\w\s#]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def detect_margin_boilerplate_lines(pages: list, margin_lines: int = 3,
                                      min_page_fraction: float = 0.3,
                                      min_pages_absolute: int = 3) -> set:
    """
    Returns a set of NORMALIZED line signatures (see
    _normalize_margin_line) that recur often enough, in a margin
    position, across `pages` to be treated as printed boilerplate
    rather than real content -- e.g. a university/company name, a
    repeated date stamp, or "Page X of Y" style footers.

    `margin_lines`: how many lines at the very TOP and very BOTTOM of
    each page's raw text are considered "margin position" candidates --
    printed letterheads/footers live there, not in the middle of a
    page's body text, so mid-page repeats (e.g. a student reusing a
    common phrase) are never flagged.

    `min_page_fraction` / `min_pages_absolute`: a normalized line must
    appear on at least this fraction of pages (and at least this many
    pages in absolute terms, to avoid false positives on very short
    documents) before it's treated as boilerplate -- a genuine one-off
    line, even if it happens to sit in the margin zone of a single
    page, is left alone.
    """
    import collections

    if not pages:
        return set()

    counts = collections.Counter()
    total_pages = len(pages)

    for page in pages:
        lines = [l for l in page["raw_text"].split("\n") if l.strip()]
        if not lines:
            continue
        margin_candidates = lines[:margin_lines] + lines[-margin_lines:]
        seen_on_this_page = set()
        for line in margin_candidates:
            norm = _normalize_margin_line(line)
            # require a little real content -- very short normalized
            # strings (e.g. just "#" from a lone page number, already
            # caught by NOISE_RE anyway) are too generic to safely
            # treat as a unique boilerplate signature.
            if len(norm) < 6:
                continue
            if norm in seen_on_this_page:
                continue  # count each page at most once per line
            seen_on_this_page.add(norm)
            counts[norm] += 1

    threshold = max(min_pages_absolute, int(total_pages * min_page_fraction))
    return {norm for norm, count in counts.items() if count >= threshold}


def _is_margin_boilerplate(line: str, boilerplate_signatures: set) -> bool:
    if not boilerplate_signatures:
        return False
    return _normalize_margin_line(line) in boilerplate_signatures


# =========================================================
# FIND QUESTION BOUNDARIES IN ANSWER PAGES -- similarity based
# UNCHANGED -- kept for backwards-compat / diagnostic use only, no
# longer on the primary answer-mapping path.
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
# FINAL ANSWER POSTPROCESSING
#
# NEW (this round -- "end main postprocessing bhi add karo jismain
# section, bhag, prashna, Q, Ans woh sab remove hojaye jo unnecessary
# ho answer main"): strip_question_restatement / strip_full_question_echo
# above already clean up a label or restated question at the very
# START of an answer. This is a broader, final safety-net pass that
# runs on the FINISHED answer text (after all slicing/merging/joining
# is done) and removes any STANDALONE section/part/question/answer
# label tokens that ended up ANYWHERE in the text -- e.g. a leftover
# "Section - B" or "भाग-2" heading that fell inside a merged/extended
# range, or a stray "Q.5" / "Ans-" that survived because it wasn't the
# very first token of the sliced text.
#
# Deliberately conservative: only removes a token when it is NOT
# attached to surrounding letters/digits (word-boundary-safe), so real
# prose that happens to contain these words as part of a sentence (a
# student legitimately writing "Answer to this question requires...")
# is left completely untouched -- only isolated label-shaped tokens
# (matching the same label patterns used elsewhere in this module) are
# removed.
# =========================================================

_STANDALONE_LABEL_TOKEN_RE = re.compile(
    r'(?<!\S)(?:'
    r'Section\s*[-:]?\s*[A-Za-z0-9]{1,3}(?!\S)'
    r'|(?:भाग|अनुभाग)\s*[-:]?\s*[०-९0-9]{1,3}(?!\S)'
    r'|Ans(?:wer)?\s*(?:\d+\s*[.:\-]?\s*\d*|[.:\-]\s*\d*)(?!\S)'
    r'|उत्तर\s*(?:\d+\s*[\-\:]?\s*\d*|[\-\:]\s*\d*)(?!\S)'
    r'|प्र[०.\s]*(?:\d+[.\s:-]*\d*|[.\s:-]+\d*)(?!\S)'
    r'|प्रश्न[.\s]*(?:\d+[.\s:-]*\d*|[.\s:-]+\d*)(?!\S)'
    r'|Q\s*(?:\d+[.\s:-]*\d*|\.\s*\d*)(?!\S)'
    r')',
    re.IGNORECASE
)


def final_answer_cleanup(text: str) -> str:
    if not text:
        return text
    cleaned = _STANDALONE_LABEL_TOKEN_RE.sub(' ', text)
    # FIX (structure-preservation round): `re.sub(r'\s+', ' ', ...)`
    # matches `\n` too, so this used to collapse every line break in
    # the answer into a single space -- flattening tables, numbered/
    # lettered lists, and parenthetical groupings into garbled run-on
    # text even after upstream code started preserving them. Only
    # collapse HORIZONTAL whitespace (spaces/tabs) here; leave newlines
    # (and blank lines from removed standalone-label lines) alone, only
    # trimming down any excessive run of 3+ blank lines to a max of one
    # blank line for readability.
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = '\n'.join(line.strip() for line in cleaned.split('\n'))
    return cleaned.strip()


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

    # NEW: detect repeated margin/letterhead content (university name,
    # date stamp, "Page X of Y" footers, etc.) across the WHOLE document
    # -- not just the answer pages -- since a printed letterhead
    # typically appears on every page (question paper pages included),
    # so using all pages gives the most reliable signal of what's
    # genuinely boilerplate versus what's unique per-page content.
    margin_boilerplate = detect_margin_boilerplate_lines(pages)
    if margin_boilerplate:
        log(
            f"Detected {len(margin_boilerplate)} recurring margin/letterhead "
            f"line pattern(s) across the document -- excluding these from "
            f"answer text (e.g. university name, date stamps, page footers)."
        )

    answer_lines = []
    for page in answer_pages:
        for line in page["raw_text"].split("\n"):
            if is_noise(line):
                continue
            if _is_margin_boilerplate(line, margin_boilerplate):
                continue
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
        raw_answer = qa_map.get(q, "")
        qa_pairs.append({
            "question": q,
            "answer": final_answer_cleanup(raw_answer),
            "matched": q in qa_map,
        })

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
