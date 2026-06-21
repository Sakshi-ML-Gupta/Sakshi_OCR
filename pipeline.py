import os
import io
import re
import json
import time
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
    def __init__(self, tpm_limit=TPM_LIMIT, safety_fraction=TPM_SAFETY_FRACTION):
        self.tpm_limit = tpm_limit
        self.safe_limit = tpm_limit * safety_fraction
        self.window_start = time.monotonic()
        self.window_tokens = 0

    def _maybe_reset_window(self):
        now = time.monotonic()
        if now - self.window_start >= 60:
            self.window_start = now
            self.window_tokens = 0

    def wait_if_needed(self, upcoming_tokens: int, log=print):
        self._maybe_reset_window()
        projected = self.window_tokens + upcoming_tokens
        if projected > self.safe_limit:
            elapsed = time.monotonic() - self.window_start
            wait_s = max(0.0, 60 - elapsed) + 0.5
            log(
                f"Proactively pacing requests: estimated {projected:.0f} tokens "
                f"would be used in this 60s window (safe budget {self.safe_limit:.0f}). "
                f"Waiting {wait_s:.1f}s before sending next chunk..."
            )
            time.sleep(wait_s)
            self.window_start = time.monotonic()
            self.window_tokens = 0

    def record_usage(self, tokens: int):
        self._maybe_reset_window()
        self.window_tokens += tokens

    def record_actual_from_error(self, used: int, limit: int):
        """Reconciles our estimate with ground truth from a 429 error."""
        self._maybe_reset_window()
        self.window_tokens = max(self.window_tokens, used)
        if limit:
            self.tpm_limit = limit
            self.safe_limit = limit * TPM_SAFETY_FRACTION


# Parses Groq's rate-limit message for the real numbers it reports,
# so retries and pacing are informed by ground truth, not guesses.
_RATE_LIMIT_DETAIL_RE = re.compile(
    r'Limit\s+(\d+),\s*Used\s+(\d+),\s*Requested\s+(\d+).*?try again in\s+([\d.]+)s',
    re.IGNORECASE | re.DOTALL
)


def _parse_rate_limit_detail(message: str):
    """Returns (limit, used, requested, wait_seconds) or None if the
    message doesn't match Groq's known error format."""
    m = _RATE_LIMIT_DETAIL_RE.search(message)
    if not m:
        return None
    limit, used, requested, wait_s = m.groups()
    return int(limit), int(used), int(requested), float(wait_s)


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
    FIX: the LLM occasionally emits multiple page numbers merged into
    one integer due to a formatting slip, e.g. [14, 16, 18] rendered
    as [141618]. This is syntactically valid JSON (just one big int),
    so it parses fine but is semantically wrong. Rather than silently
    discarding it as "out of range," we attempt to recover the original
    page numbers by greedily splitting its digit string into chunks
    that are all currently-valid page numbers for this document.
    Returns [] if no valid split is found (in which case the caller
    should fall back to discarding it, as before).
    """
    s = str(n)
    if len(s) <= len(str(max_page)):
        return []  # not implausibly large -- not a concatenation case

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
        return result

    for num_parts in range(2, len(s) + 1):
        for widths in product(range(1, max_digits + 1), repeat=num_parts):
            if sum(widths) != len(s):
                continue
            result = split_attempt(s, widths)
            if result:
                return result
    return []


def _call_groq_for_chunk(client, pages_chunk: list, budget: "_TokenBudgetTracker",
                          log, max_retries: int = 4) -> tuple:
    import groq

    user_prompt = _build_qp_user_prompt(pages_chunk)
    estimated_tokens = _estimate_tokens(QP_SYSTEM_PROMPT) + _estimate_tokens(user_prompt) + 800
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
                        {"role": "system", "content": QP_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
            budget.record_usage(estimated_tokens)
            content = response.choices[0].message.content
            return _parse_qp_llm_response(content)

        except (groq.RateLimitError, groq.BadRequestError) as e:
            last_error = e
            detail = _parse_rate_limit_detail(str(e))
            if detail:
                limit, used, requested, wait_s = detail
                budget.record_actual_from_error(used, limit)
                log(
                    f"Chunk LLM call hit a rate/size limit (attempt {attempt}): "
                    f"TPM limit={limit}, used={used}, requested={requested}. "
                    f"Waiting {wait_s + 0.5:.1f}s (Groq-reported) before retrying..."
                )
                time.sleep(wait_s + 0.5)
                # Groq's own reported wait already accounts for the real
                # usage in the window we just hit the limit in. Once we've
                # waited it out, that window is effectively over -- reset
                # our tracker so the upcoming retry isn't ALSO blocked by
                # our own proactive pacer re-checking the same stale figure.
                budget.window_start = time.monotonic()
                budget.window_tokens = 0
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


def _merge_chunk_results(chunk_results: list) -> tuple:
    all_qp_pages = set()
    all_questions = []
    seen_keys = set()

    for qp_pages, questions in chunk_results:
        all_qp_pages.update(qp_pages)
        for q in questions:
            key = re.sub(r'\s+', ' ', q.lower().strip())
            if key not in seen_keys:
                seen_keys.add(key)
                all_questions.append(q)

    return sorted(all_qp_pages), all_questions


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

    for i, chunk in enumerate(chunks):
        page_nums_in_chunk = [p["page_number"] for p in chunk]
        log(f"Asking LLM to analyze chunk {i+1}/{len(chunks)} (pages {page_nums_in_chunk})...")

        qp_pages_1based, questions = _call_groq_for_chunk(client, chunk, budget, log)

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

    qp_pages_1based_merged, questions = _merge_chunk_results(chunk_results)
    qp_page_indices_0based = sorted(pn - 1 for pn in qp_pages_1based_merged)

    log(
        f"Merged result across all chunks: {len(qp_page_indices_0based)} question paper "
        f"page(s), {len(questions)} question(s)"
    )

    return qp_page_indices_0based, questions


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


# =========================================================
# COMPLETE PIPELINE
# =========================================================

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

    log("Matching questions via similarity (works for Hindi/English/any format)...")
    boundaries = find_question_boundaries_by_similarity(answer_lines, official_questions)
    log(f"Matched {len(boundaries)} of {len(official_questions)} questions")

    matched_qs = {b["question"] for b in boundaries}
    for q in official_questions:
        if q not in matched_qs:
            log(f"WARNING: No match found for: {q[:60]}")

    if not boundaries:
        raise Exception(
            "Could not match any questions in answer pages.\n"
            f"Official questions: {official_questions}\n"
            f"First 10 answer lines: {answer_lines[:10]}"
        )

    log("Slicing raw answers...")
    qa_pairs = slice_raw_answers_by_boundaries(answer_lines, boundaries)

    log(f"Done -- {len(qa_pairs)} Q-A pairs")
    return ocr_json, qa_pairs
