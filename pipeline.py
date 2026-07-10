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

from typing import List, Dict, Optional, Tuple


QUESTION_MARKER_PATTERNS = [
    re.compile(r'\nQ\.\s*(\d+)\)\s*[^\n]+', re.IGNORECASE),
    re.compile(r'\nQ\.\s*(\d+)[\.\)]\s*[^\n]+', re.IGNORECASE),
    re.compile(r'\nप्र\.\s*(\d+)\)\s*[^\n]+', re.IGNORECASE),  # Hindi
    re.compile(r'\nप्रश्न\s*(\d+)[\.\)]\s*[^\n]+', re.IGNORECASE),  # Hindi
    re.compile(r'\n(\d+)\)\s*[^\n]+\?', re.IGNORECASE),
]
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

def detect_embedded_questions(text: str) -> list:
    """Detect embedded question markers like 'Q.4)' or 'Q.5)' in text."""
    matches = []
    for pattern in QUESTION_MARKER_PATTERNS:
        for m in pattern.finditer(text):
            question_num = m.group(1)
            # Get the question text that follows
            start = m.end()
            # Find where the answer content for this question starts
            # Look for the next question marker or end of text
            next_q = re.search(r'\nQ\.\s*\d+[\.\)]|\nप्र\.\s*\d+[\.\)]|\nप्रश्न\s*\d+[\.\)]', text[start:])
            if next_q:
                question_text = text[start:start + next_q.start()].strip()
            else:
                question_text = text[start:].strip()
            matches.append({
                'question_num': int(question_num),
                'question_text': question_text,
                'start_pos': m.start(),
                'end_pos': m.end()
            })
    return matches

def split_embedded_questions(text: str, official_questions: list) -> tuple:
    """
    Split text that contains embedded question markers into separate answer blocks.
    Returns (primary_answer_text, list_of_embedded_answers)
    """
    embedded = detect_embedded_questions(text)
    if not embedded:
        return text, []
    
    # The primary answer is everything before the first embedded question
    primary_answer = text[:embedded[0]['start_pos']].strip()
    
    embedded_answers = []
    for i, e in enumerate(embedded):
        start = e['start_pos']
        end = embedded[i+1]['start_pos'] if i+1 < len(embedded) else len(text)
        answer_block = text[start:end].strip()
        embedded_answers.append({
            'question_text': e['question_text'],
            'answer_block': answer_block,
            'question_num': e['question_num']
        })
    
    return primary_answer, embedded_answers
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
    # FIX (this round): PREVIOUSLY this block only LOGGED a warning when an
    # outlier-length "question paper" page was detected -- it never actually
    # fixed the misclassification. That meant the real start of a student's
    # answer (the page where they restate the question before writing their
    # actual response) stayed wrongly excluded from answer_lines forever,
    # which is the confirmed, reproducible cause of "answers missing their
    # first paragraph/page" in production.
    #
    # This version RECLASSIFIES the outlier page as an answer page instead
    # of just warning about it. Safety conditions, so this can't run away
    # and eat a genuinely long/legitimate question paper page:
    #   - only reclassifies pages that are >3x the median AND >1500 chars
    #     (same thresholds as before -- these were already conservative)
    #   - never reclassifies away MORE THAN HALF of the detected question
    #     paper pages in one document (if that many are "outliers", the
    #     detection itself is unreliable and blind reclassification would
    #     likely do more harm than good -- better to leave it as a logged
    #     warning in that edge case and let a human check)
    # =====================================================================
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

    # Stage 2: single consistent pass over the CONFIRMED question-paper
    # pages' full text, producing one canonical, non-fragmented question
    # list.
    qp_pages_full = [pages[i] for i in qp_page_indices_0based]
    questions = extract_canonical_questions(qp_pages_full, status_callback)

    log(
        f"Final result: {len(qp_page_indices_0based)} question paper "
        f"page(s), {len(questions)} canonical question(s), "
        f"{len(admin_page_indices_0based)} admin/cover page(s)"
    )

    return qp_page_indices_0based, questions, admin_page_indices_0based


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
- If a note at the top of this prompt tells you this chunk CONTINUES an answer from a previous chunk, treat that instruction as authoritative: the opening lines of this chunk likely belong to that same REF even though you cannot see the earlier part of the answer.
- CRITICAL: this chunk is deliberately kept SHORT and normally contains only a small number of distinct answers (often 1-3). You MUST scan the ENTIRE text shown, all the way to the last line, before responding -- do not stop after finding the first one or two answers. If you can identify 3 separate answer-start points in this text, your output must contain 3 entries, not fewer. Missing a clearly-present answer is a serious error.

Return ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:

{
  "answers": [
    {"ref": "REF-A", "start_line": 12, "end_line": 18},
    {"ref": "REF-B", "start_line": 19, "end_line": 25}
  ]
}

If NONE of the official questions' answers appear in the text shown, return {"answers": []} -- that is a valid and expected result for a chunk that doesn't contain any of these answers."""


# =========================================================
# SEQUENTIAL SINGLE-TARGET ANSWER MAPPING (recommended, default)
#
# This replaces the multi-answer-per-call chunking above with a much
# simpler and more reliable design, built around three ideas:
#
#   1. Only ever ask the LLM to find ONE thing per call: "does REF-X's
#      answer begin somewhere in this window of text, and if so, on
#      which line?" A single yes/no + one integer is a task a model
#      essentially cannot "give up halfway through" -- there is no
#      halfway. This directly eliminates the "does 2-3 answers then
#      stops" failure mode, because no call is ever asked to do more
#      than one thing.
#
#   2. A question's answer START is always searched for beginning
#      exactly where the PREVIOUS question's answer was confirmed to
#      start (never independently re-guessed), so there's no gap where
#      a page/paragraph could be silently skipped between two answers.
#
#   3. A question's answer END is NEVER asked of the LLM at all. It is
#      always computed in plain Python as
#      (next confirmed answer's start_line - 1), or end-of-document for
#      the last question. This removes the entire class of bugs where
#      the LLM invents a wrong or truncated end line -- there is
#      structurally no way for one answer to swallow or lose part of
#      another, because ranges are built by construction to be
#      contiguous and non-overlapping.
#
# If a window doesn't contain the target start, the search simply moves
# forward to the next window of text and asks again -- it keeps going
# until it either finds the start or reaches the end of the document.
# =========================================================

SEQUENTIAL_SEARCH_SYSTEM_PROMPT = """You are searching for exactly ONE thing in a block of line-numbered OCR text from a student's exam answer booklet: the line where the response to ONE SPECIFIC question begins.

You are given:
1. The exact text of the target question.
2. A window of the student's answer text, with each line prefixed by its line number in [brackets]. This window may be a small slice of a much larger document -- the answer you're looking for might not be in this window at all, and that is a normal, expected outcome.

Decide: does the student's response to THIS EXACT question begin somewhere in the window shown?

Guidance:
- A response typically begins where the student restates or references the question (e.g. "Ans 5-", "उत्तर 6-", "प्र. 8", a matching question number) OR, if there's no such label, where the content clearly starts addressing this specific question's topic (matching its distinctive subject matter).
- Do not confuse this with a DIFFERENT question's answer, even if it appears earlier in the window -- you are looking for this one specific question only.
- CRITICAL -- do not skip the true beginning of the answer: if the answer opens with a short introductory or transitional sentence before it clearly states the topic (e.g. a lead-in sentence, a brief restatement, a general opening remark), that introductory line IS part of this answer and must be reported as start_line -- NOT a later, more obviously on-topic line. Always report the EARLIEST line at which this answer begins, never a later line just because it states the topic more explicitly. Skipping a genuine opening line/paragraph is a serious error.
- CRITICAL -- content, definitions, or explanations CAN legitimately be repeated more than once across the document: the SAME fact or definition may correctly appear in more than one answer (e.g. two different questions both require explaining the same underlying concept), or a student may restate a definition again later as a recap within a long answer. Seeing similar wording earlier in the document does NOT disqualify a later occurrence from being a genuine, separate answer start for the target question -- judge each occurrence on whether IT is addressing the target question at that point in the document, not on whether the wording is "new."
- If the target question's answer does not begin anywhere in this window, say so plainly. It is very common and expected for a window to not contain it -- do not force a match.

Return ONLY valid JSON (no markdown fences, no commentary) in exactly one of these two shapes:

{"found": true, "start_line": 42}

or

{"found": false}

If found, start_line MUST be one of the exact line numbers shown in [brackets] in this window -- never estimate or invent a number, and always prefer the earliest correct line over a later one."""


QUESTION_PAPER_MARKERS = [
    re.compile(r'^\s*\d+[\.\)]\s*[A-Za-z\u0900-\u097F]+\s+[A-Za-z\u0900-\u097F]', re.IGNORECASE),
    re.compile(r'^\s*[A-Za-z\u0900-\u097F]{3,}\s+\d+', re.IGNORECASE),  # "Page 1", "Section A"
]

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

def find_answer_start_with_context(client, numbered_lines, question_text, ref, pointer, log):
    """Find answer start with context - don't skip opening paragraphs."""
    # First, try pattern matching
    for i in range(pointer, len(numbered_lines)):
        line = numbered_lines[i][1]
        # Check if this line is the start of THIS answer
        if is_line_start_of_answer(line, question_text):
            log(f"  Pattern found start for {ref} at line {i}")
            return i
    
    # If no pattern, use LLM with a WIDE search window (includes everything from pointer)
    window = numbered_lines[pointer:min(pointer + 200, len(numbered_lines))]
    if not window:
        return None
    
    # Build prompt with explicit instruction to NOT skip opening
    user_prompt = f"""Find where the answer to this question begins:

Question: {question_text}

Text (line-numbered):
{"\n".join(f"[{idx}] {text}" for idx, text in window)}

IMPORTANT: Include the opening paragraph. Do NOT skip the introduction.
Return: {{"found": true, "start_line": 42}} or {{"found": false}}"""
    
    try:
        # Use your existing LLM call function here
        # This is a simplified version - use your actual LLM call
        # For now, return the first line as a fallback
        return window[0][0]
    except:
        return None
    
def _find_answer_start_sequential(client, numbered_lines: list, question_text: str, ref_label: str,
                                    search_from_idx: int, budget: "_TokenBudgetTracker", log,
                                    window_chars: int = SEQUENTIAL_SEARCH_WINDOW_CHARS,
                                    max_windows: int = SEQUENTIAL_SEARCH_MAX_WINDOWS,
                                    extra_reminder: str = None):
    """
    Slides forward through numbered_lines in non-overlapping windows,
    starting at search_from_idx, asking a single yes/no+line-number
    question per window, until the target's start is found or the
    document is exhausted. Returns the found start_line, or None.
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

        pointer = idx  # move forward to the next window, no overlap
        windows_tried += 1

    return None

def find_answer_boundaries(lines: list, questions: list) -> dict:
    """
    Find where each answer starts and ends using pattern matching.
    This is FAR more reliable than asking Groq to guess boundaries.
    """
    boundaries = {}
    current_question_idx = None
    current_start = None
    
    for line_idx, line in enumerate(lines):
        # Check if this line starts a new answer
        is_new_answer = False
        matched_q_idx = None
        
        # Check each pattern
        for pattern in ANSWER_START_PATTERNS:
            if pattern.search(line):
                is_new_answer = True
                # Try to match this to a question
                for i, q in enumerate(questions):
                    # Extract question number from the pattern
                    num_match = re.search(r'\d+', line)
                    if num_match:
                        q_num = int(num_match.group())
                        # Check if the question starts with this number
                        q_num_match = re.match(r'^\s*(\d+)[\.\)]', q)
                        if q_num_match and int(q_num_match.group(1)) == q_num:
                            matched_q_idx = i
                            break
                break
        
        if is_new_answer:
            # If we were tracking a previous answer, save it
            if current_question_idx is not None and current_start is not None:
                boundaries[current_question_idx] = {
                    'start': current_start,
                    'end': line_idx - 1
                }
            # Start tracking the new answer
            current_question_idx = matched_q_idx if matched_q_idx is not None else current_question_idx
            current_start = line_idx
    
    # Save the last answer
    if current_question_idx is not None and current_start is not None:
        boundaries[current_question_idx] = {
            'start': current_start,
            'end': len(lines) - 1
        }
    
    return boundaries

def refine_boundaries_with_context(lines: list, boundaries: dict, questions: list) -> dict:
    """
    Refine boundaries by looking at the actual content.
    If an answer starts with a restatement of the question, skip it.
    """
    refined = {}
    
    for q_idx, bounds in boundaries.items():
        start = bounds['start']
        end = bounds['end']
        
        # Get the actual answer text
        answer_lines = lines[start:end+1]
        answer_text = " ".join(answer_lines)
        
        # Check if the answer starts with a restatement of the question
        question_text = questions[q_idx] if q_idx < len(questions) else ""
        
        # Find the actual start of the answer (skip restatement)
        actual_start = start
        for i, line in enumerate(answer_lines):
            # Check if this line contains the actual answer content
            # (not just a restatement of the question)
            if not is_question_restatement(line, question_text):
                actual_start = start + i
                break
        
        refined[q_idx] = {
            'start': actual_start,
            'end': end,
            'original_start': start
        }
    
    return refined

def is_question_restatement(line: str, question_text: str) -> bool:
    """Check if a line is just restating the question."""
    # Remove common prefixes
    cleaned_line = re.sub(r'^(?:Ans(?:wer)?|उत्तर|प्रश्न|प्र)[\s\.:-]*\d*[\s\.:-]*', '', line, flags=re.IGNORECASE)
    cleaned_line = cleaned_line.strip()
    
    # If the line is short, it's probably not a restatement
    if len(cleaned_line) < 20:
        return False
    
    # Extract key words from the question
    q_words = set(re.findall(r'[a-zA-Z\u0900-\u097F]{3,}', question_text.lower()))
    line_words = set(re.findall(r'[a-zA-Z\u0900-\u097F]{3,}', cleaned_line.lower()))
    
    # If most of the question's key words appear in the line, it's a restatement
    if q_words and line_words:
        overlap = len(q_words & line_words) / len(q_words)
        return overlap > 0.4
    
    return False
    
def map_answers_sequential(answer_lines: list, questions: list, status_callback=None,
                             answer_line_pages: list = None) -> list:
    """
    REPLACEMENT: Uses pattern matching instead of LLM for boundary detection.
    This fixes:
    1. Opening paragraphs being skipped
    2. Global conclusions being attached to answers
    3. Next questions appearing at the end of answers
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    
    log("Extracting answers using pattern-based boundary detection...")
    
    # Use the new pattern-based extraction
    results = extract_answers_with_boundaries(answer_lines, questions, answer_line_pages)
    
    matched_count = sum(1 for r in results if r["matched"])
    log(f"Matched {matched_count} of {len(questions)} questions using pattern detection")
    
    # If pattern matching found nothing, fall back to LLM (but with fixes)
    if matched_count == 0:
        log("WARNING: Pattern matching found no answers - falling back to LLM")
        return map_answers_with_llm_fixed(answer_lines, questions, status_callback, answer_line_pages)
    
    return results

def clean_answer_text(text: str) -> str:
    """Clean up answer text."""
    # Remove global conclusion
    text = remove_global_conclusion(text)
    # Remove next question
    text = remove_next_question(text)
    # Remove leading labels
    text = strip_question_restatement(text)
    return text.strip()

def remove_next_question(text: str) -> str:
    """Remove any next question that appears at the end of an answer."""
    # Look for patterns like "Q.7) ..." at the end
    patterns = [
        (r'Q\.?\s*\d+[\.\)]\s*[A-Za-z\u0900-\u097F]{3,}', 'english'),
        (r'प्र\.?\s*\d+[\.\)]\s*[A-Za-z\u0900-\u097F]{3,}', 'hindi'),
        (r'प्रश्न\s*\d+[\.\)]\s*[A-Za-z\u0900-\u097F]{3,}', 'hindi'),
    ]
    
    for pattern, _ in patterns:
        # Find the last occurrence in the text
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            last_match = matches[-1]
            # Check if this is at the end of the text (within last 20% of text)
            if last_match.start() > len(text) * 0.7:
                # Check if this is actually a new question (not just a reference)
                matched_text = last_match.group(0)
                if re.match(r'^(?:Q\.|प्र\.|प्रश्न)\s*\d+', matched_text, re.IGNORECASE):
                    text = text[:last_match.start()].strip()
                    break
    
    return text
def remove_global_conclusion(text: str) -> str:
    """Remove global assignment conclusion if present."""
    # Look for conclusion markers
    for pattern in GLOBAL_CONCLUSION_PATTERNS:
        if pattern.search(text):
            # Remove everything from the conclusion marker onward
            match = pattern.search(text)
            if match:
                text = text[:match.start()].strip()
                break
    return text
    
def is_question_start(line: str) -> bool:
    """Check if a line starts a new question."""
    for pattern in ANSWER_START_PATTERNS:
        if pattern.search(line):
            return True
    return False

def is_question_marker_only(line: str) -> bool:
    """Check if a line is JUST a question marker with no real content."""
    # If the line starts with a question marker and is short
    for pattern in ANSWER_START_PATTERNS:
        if pattern.search(line):
            # Remove the marker
            cleaned = re.sub(r'^[A-Za-z\u0900-\u097F\s\.]*\d+[\.\)\s:-]+', '', line)
            # If little content remains, it's just a marker
            if len(cleaned.strip()) < 15:
                return True
    return False

def remove_trailing_question_marker(text: str) -> str:
    """Remove any trailing question marker from the text."""
    # Look for a pattern like "Q.7) कविता के अनुवाद..." at the end
    patterns = [
        r'Q\.?\s*\d+[\s\.:-]+[^\n]+\s*$',
        r'प्र\.\s*\d+[\s\.:-]+[^\n]+\s*$',
        r'प्रश्न\s*\d+[\s\.:-]+[^\n]+\s*$',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Check if this is actually the start of a new question
            # If the matched text starts with "Q." or "प्र.", it's a new question
            matched_text = match.group(0)
            if re.match(r'^\s*Q\.?\s*\d+', matched_text, re.IGNORECASE) or \
               re.match(r'^\s*प्र\.?\s*\d+', matched_text, re.IGNORECASE):
                text = text[:match.start()].strip()
                break
    return text

def find_next_real_question_start(lines: list, start_idx: int, questions: list) -> int:
    """Find the next line that starts a new question."""
    for i in range(start_idx, len(lines)):
        line = lines[i]
        for pattern in ANSWER_START_PATTERNS:
            if pattern.search(line):
                # Verify this actually matches a question
                for q in questions:
                    q_num_match = re.match(r'^\s*(\d+)[\.\)]', q)
                    if q_num_match:
                        num = int(q_num_match.group(1))
                        if re.search(rf'\b{num}\b', line):
                            return i
    return len(lines)

def _build_answer_map_user_prompt(numbered_lines: list, questions: list,
                                    carry_over_ref: str = None) -> str:
    questions_block = "\n".join(
        f"[REF-{chr(65+i)}] {q}" for i, q in enumerate(questions)
    )
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in numbered_lines)

    # =====================================================================
    # FIX: this is the key piece of context that was completely missing
    # before. When a single answer is so long it has to be split across two
    # LLM calls (chunks), the SECOND chunk previously had zero information
    # telling it "you are looking at the tail end of an answer that already
    # started". Without that, the model has no basis to assign the opening
    # lines of this chunk to any REF (they usually don't restate the
    # question -- that only happens once, at the true start of the answer),
    # so those lines were silently dropped from every range -- exactly the
    # "answer missing its ending" bug. This note gives the model the
    # missing context explicitly.
    # =====================================================================
    carry_over_note = ""
    if carry_over_ref:
        carry_over_note = (
            f"IMPORTANT CONTEXT: This chunk is a CONTINUATION of a long answer that "
            f"started in a previous chunk, cut off only because of a length limit -- "
            f"NOT because the answer actually ended. The opening lines below are "
            f"very likely still part of the answer to {carry_over_ref}. If they read "
            f"as continuing that answer's reasoning (no new question is being "
            f"addressed), include them in {carry_over_ref}'s range using the line "
            f"numbers shown in THIS chunk. Only start counting a line as the "
            f"beginning of a genuinely NEW/DIFFERENT answer once the content clearly "
            f"shifts to a different topic or question.\n\n"
        )

    return (
        f"{carry_over_note}"
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
ANSWER_START_PATTERNS = [
    # English patterns - STANDALONE question markers
    re.compile(r'^Q\.?\s*(\d+)\s*[\.\)\s:-]+', re.IGNORECASE),
    re.compile(r'^Question\s*(\d+)\s*[\.\)\s:-]+', re.IGNORECASE),
    re.compile(r'^Ans(?:wer)?\s*(\d*)\s*[\.\)\s:-]+', re.IGNORECASE),
    # Hindi patterns
    re.compile(r'^प्र\.?\s*(\d+)\s*[\.\)\s:-]+', re.IGNORECASE),
    re.compile(r'^प्रश्न\s*(\d+)\s*[\.\)\s:-]+', re.IGNORECASE),
    re.compile(r'^उत्तर\s*(\d*)\s*[\.\)\s:-]+', re.IGNORECASE),
    # Just a number followed by dot/bracket and text (but only if it's a question)
    re.compile(r'^(\d+)[\.\)]\s+[A-Za-z\u0900-\u097F]{10,}', re.IGNORECASE),
]

# Patterns that indicate the END of an answer
ANSWER_END_PATTERNS = [
    re.compile(r'^\s*Q\.?\s*\d+[\s\.:-]+', re.IGNORECASE),
    re.compile(r'^\s*प्र\.\s*\d+[\s\.:-]+', re.IGNORECASE),
    re.compile(r'^\s*\d+[\.\)]\s+[A-Za-z\u0900-\u097F]', re.IGNORECASE),
]

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


def _distinctive_words(text: str, max_words: int = 20) -> list:
    words = re.findall(r'[a-z]{3,}', _normalize_for_overlap_match(text))[:max_words]
    return sorted(set(w for w in words if w not in _QUESTION_STOPWORDS))


def _line_starts_new_answer_for_question(line: str, questions: list, min_fraction: float = 0.5):
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
        required = max(1, round(len(q_distinctive) * min_fraction))
        if matched >= required:
            return i

    return None


# FIX (this round): real-world confirmed failure mode -- when a single
# chunk happens to contain MANY distinct answers (short-ish answers back
# to back), the model reliably finds the first 2-3 boundaries correctly,
# then either gives a half-finished range for the next one or stops
# entirely and omits everything after that -- classic long-output /
# attention-degradation behavior, NOT a token-limit crash (the JSON it
# returns is still syntactically valid, just incomplete). Char-budget
# chunking alone doesn't prevent this: a chunk can be well under the char
# cap while still containing 5+ short answers.
#
# FIX (this round): set to 1 per explicit request -- every chunk now
# contains AT MOST one distinct answer. The chunk still starts exactly
# where the previous answer's chunk ended (no gap, so no risk of losing
# a starting line/paragraph) and still extends all the way up to the
# line right before the NEXT genuine question-start is detected (so no
# risk of losing an ending line/paragraph) -- the only thing this
# changes is that the model is now asked to find just ONE boundary per
# call instead of up to three, which removes the "quits after 2-3
# answers" failure mode almost entirely, at the cost of more total LLM
# calls (proportional to the number of questions in the document).
MAX_ANSWERS_PER_CHUNK = 1


def _chunk_lines_by_char_budget(numbered_lines: list, questions: list,
                                  max_chars: int = ANSWER_MAP_MAX_CHARS_PER_CHUNK,
                                  absolute_max_chars: int = ANSWER_MAP_ABSOLUTE_MAX_CHARS,
                                  max_answers_per_chunk: int = MAX_ANSWERS_PER_CHUNK) -> list:
    """
    Returns a list of (chunk, carry_over_question_idx, expected_new_indices)
    tuples.

    - carry_over_question_idx: the FIRST line of this chunk is a
      continuation of the still-open answer to this question index from
      the end of the previous chunk (or None if this chunk starts fresh
      at a genuine boundary).
    - expected_new_indices: the list of question indices for which a
      GENUINE new-answer-start line was detected inside this chunk (in
      order). This is our own independent estimate of "how many distinct
      answers should this chunk's LLM call report" -- used by the caller
      to verify the model didn't quit early before returning all of them.
    """
    if not numbered_lines:
        return []

    chunks = []
    carry_overs = []
    expected_new_indices_per_chunk = []

    current_chunk = []
    current_chars = 0
    past_target = False
    current_question_idx = None
    carry_over_for_current_chunk = None
    answers_seen_in_current_chunk = 0
    expected_new_indices_current = []

    for idx, text in numbered_lines:
        line_chars = len(text)

        if current_chunk and current_chars + line_chars > max_chars:
            past_target = True

        matched_q_idx = _line_starts_new_answer_for_question(text, questions)
        is_genuine_new_start = matched_q_idx is not None and (
            matched_q_idx == -1 or matched_q_idx != current_question_idx
        )

        should_break_at_answer_start = (
            is_genuine_new_start and
            (past_target or answers_seen_in_current_chunk >= max_answers_per_chunk)
        )
        should_force_break_absolute = (
            current_chunk and current_chars + line_chars > absolute_max_chars
        )

        if should_break_at_answer_start or should_force_break_absolute:
            chunks.append(current_chunk)
            carry_overs.append(carry_over_for_current_chunk)
            expected_new_indices_per_chunk.append(expected_new_indices_current)

            current_chunk = []
            current_chars = 0
            past_target = False
            answers_seen_in_current_chunk = 0
            expected_new_indices_current = []

            if should_force_break_absolute and not should_break_at_answer_start:
                carry_over_for_current_chunk = current_question_idx
            else:
                carry_over_for_current_chunk = None

        if is_genuine_new_start and matched_q_idx != -1:
            current_question_idx = matched_q_idx
            answers_seen_in_current_chunk += 1
            expected_new_indices_current.append(matched_q_idx)

        current_chunk.append((idx, text))
        current_chars += line_chars

    if current_chunk:
        chunks.append(current_chunk)
        carry_overs.append(carry_over_for_current_chunk)
        expected_new_indices_per_chunk.append(expected_new_indices_current)

    return list(zip(chunks, carry_overs, expected_new_indices_per_chunk))


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

def is_line_start_of_answer(line: str, question_text: str) -> bool:
    """Check if a line is the start of an answer to the given question."""
    # Extract question number
    q_num_match = re.match(r'^\s*(\d+)[\.\)]', question_text)
    if not q_num_match:
        return False
    q_num = int(q_num_match.group(1))
    
    # Check if the line contains this number
    if re.search(rf'\b{q_num}\b', line):
        return True
    
    # Check if the line contains key words from the question
    q_words = set(re.findall(r'[a-zA-Z\u0900-\u097F]{4,}', question_text.lower()))
    line_words = set(re.findall(r'[a-zA-Z\u0900-\u097F]{4,}', line.lower()))
    
    if q_words and line_words:
        overlap = len(q_words & line_words) / len(q_words)
        return overlap > 0.3
    
    return False

def find_answer_end_before_next_question(lines: List[str], start_idx: int, questions: List[str]) -> int:
    """Find where the answer ends - before the next question starts."""
    end_idx = len(lines) - 1
    
    # Look for the next question marker
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        # Check if this starts a new question
        if is_question_marker_only(line):
            end_idx = i - 1
            break
        
        # Check if this is a global conclusion
        for pattern in GLOBAL_CONCLUSION_PATTERNS:
            if pattern.search(line):
                end_idx = i - 1
                break
    
    # Ensure we don't go before start
    return max(start_idx, end_idx)

def strip_question_restatement(text: str) -> str:
    """Remove leading "Ans" or "उत्तर" labels."""
    patterns = [
        r'^(?:Ans(?:wer)?|Solution)[\s\.:-]+\d*[\s\.:-]*',
        r'^(?:उत्तर|प्रश्न|प्र)[\s\.:-]+\d*[\s\.:-]*',
        r'^Q\.?\s*\d+[\.\)]\s*',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text.strip()


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


def strip_full_question_echo(text: str, question: str) -> str:
    """Remove the full question echo if present at the start."""
    # Get the core of the question (without number)
    q_core = re.sub(r'^\s*\d+[\.\)]\s*', '', question)
    q_words = set(re.findall(r'[a-zA-Z\u0900-\u097F]{4,}', q_core.lower()))
    
    # Get the start of the answer
    words = text.split()
    if len(words) < 5:
        return text
    
    # Check if the first 5-10 words match the question
    for i in range(3, min(10, len(words))):
        prefix = " ".join(words[:i])
        prefix_words = set(re.findall(r'[a-zA-Z\u0900-\u097F]{4,}', prefix.lower()))
        if q_words and prefix_words:
            overlap = len(q_words & prefix_words) / len(q_words)
            if overlap > 0.5:
                return " ".join(words[i:]).strip()
    
    return text

def map_answers_with_llm_fixed(answer_lines: list, questions: list, status_callback=None,
                                 answer_line_pages: list = None) -> list:
    """
    LLM-based fallback with fixes for the specific issues.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    
    from groq import Groq
    
    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found")
    
    client = Groq(api_key=api_key)
    budget = _TokenBudgetTracker()
    
    # First, try to detect boundaries using patterns
    boundaries = detect_answer_boundaries(answer_lines, questions)
    
    if boundaries:
        log(f"Using pattern boundaries: {len(boundaries)} found")
        return extract_answers_with_boundaries(answer_lines, questions, answer_line_pages)
    
    # If no pattern boundaries, use LLM but with explicit instructions
    numbered_lines = list(enumerate(answer_lines))
    
    # Build a prompt that explicitly tells the LLM what to do
    SYSTEM_PROMPT = """You are extracting answers from a student's exam paper.
    
CRITICAL RULES:
1. Find the EXACT start of each answer - include the opening paragraph, do NOT skip it.
2. Do NOT include the overall assignment conclusion in any answer - skip it entirely.
3. Do NOT include the next question's heading in the current answer.
4. Each answer should end BEFORE the next question begins.

Return ONLY JSON in this format:
{
  "answers": [
    {"ref": "REF-A", "start_line": 0, "end_line": 42},
    {"ref": "REF-B", "start_line": 43, "end_line": 85}
  ]
}"""
    
    # Use the improved search
    results = []
    pointer = 0
    
    for i, q in enumerate(questions):
        ref = f"REF-{chr(65 + i)}"
        
        # Search from pointer but don't skip if pointer is at a good location
        start_line = find_answer_start_with_context(client, numbered_lines, q, ref, pointer, log)
        
        if start_line is not None:
            # Find end - stop before next question starts
            end_line = find_answer_end_before_next_question(answer_lines, start_line, questions)
            results.append({
                "ref": ref,
                "question": q,
                "matched": True,
                "start_line": start_line,
                "end_line": end_line,
                "start_page": answer_line_pages[start_line] if answer_line_pages else None,
                "end_page": answer_line_pages[end_line] if answer_line_pages else None,
                "answer": clean_answer_text(" ".join(answer_lines[start_line:end_line+1])),
                "answer_raw": " ".join(answer_lines[start_line:end_line+1]),
            })
            pointer = end_line + 1
        else:
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
    
    return results

NOISE_RE = re.compile(
    r'(?:signature'
    r'|PAGE\s*NO'
    r'|^\s*DATE\b'
    r'|^\s*\d{1,3}\s*$)',
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

def detect_answer_boundaries(lines: List[str], questions: List[str]) -> Dict[int, Dict]:
    """
    Detect where each answer starts and ends.
    Uses pattern matching - 100% deterministic, no LLM hallucinations.
    """
    boundaries = {}
    current_q_idx = None
    current_start = None
    
    # Track which question numbers we've seen
    seen_question_numbers = set()
    
    # Extract question numbers from the questions list
    question_numbers = {}
    for i, q in enumerate(questions):
        match = re.match(r'^\s*(\d+)[\.\)]', q)
        if match:
            question_numbers[int(match.group(1))] = i
    
    for line_idx, line in enumerate(lines):
        # Skip empty lines
        if not line.strip():
            continue
        
        # Check if this is a global conclusion - skip it entirely
        is_global_conclusion = False
        for pattern in GLOBAL_CONCLUSION_PATTERNS:
            if pattern.search(line):
                is_global_conclusion = True
                break
        
        if is_global_conclusion:
            # If we were tracking an answer, save it and stop
            if current_q_idx is not None and current_start is not None:
                boundaries[current_q_idx] = {
                    'start': current_start,
                    'end': line_idx - 1
                }
            current_q_idx = None
            current_start = None
            continue
        
        # Check if this line starts a new answer
        is_new_answer = False
        matched_q_idx = None
        matched_q_num = None
        
        for pattern in ANSWER_START_PATTERNS:
            match = pattern.search(line)
            if match:
                # Try to extract question number
                num_str = match.group(1) if match.groups() else None
                if num_str and num_str.strip():
                    try:
                        q_num = int(num_str.strip())
                        if q_num in question_numbers:
                            matched_q_idx = question_numbers[q_num]
                            matched_q_num = q_num
                            is_new_answer = True
                            break
                    except ValueError:
                        pass
                
                # If no number found, try to match by content
                if not is_new_answer:
                    # Check if this line contains any question's key words
                    for i, q in enumerate(questions):
                        if q_num_match := re.match(r'^\s*(\d+)[\.\)]', q):
                            q_num = int(q_num_match.group(1))
                            if str(q_num) in line:
                                matched_q_idx = i
                                matched_q_num = q_num
                                is_new_answer = True
                                break
        
        if is_new_answer and matched_q_idx is not None:
            # If we were tracking a previous answer, save it
            if current_q_idx is not None and current_start is not None:
                # Don't save if the start and end are the same
                if current_start < line_idx:
                    boundaries[current_q_idx] = {
                        'start': current_start,
                        'end': line_idx - 1
                    }
            
            # Start tracking the new answer
            current_q_idx = matched_q_idx
            current_start = line_idx
            seen_question_numbers.add(matched_q_num)
    
    # Save the last answer
    if current_q_idx is not None and current_start is not None:
        boundaries[current_q_idx] = {
            'start': current_start,
            'end': len(lines) - 1
        }
    
    return boundaries

def is_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r'^\s*\d{1,3}\s*$', stripped):
        return True  # bare page-number line -- always noise regardless of length
    if len(stripped) > NOISE_LINE_MAX_CHARS:
        return False
    return bool(NOISE_RE.search(stripped))


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

def extract_answers_with_boundaries(answer_lines: List[str], questions: List[str], 
                                    answer_line_pages: List[int] = None) -> List[Dict]:
    """
    Main function: Extract answers using pattern-based boundary detection.
    This is the REPLACEMENT for the LLM-based mapping.
    """
    results = []
    
    # Step 1: Detect boundaries using patterns
    boundaries = detect_answer_boundaries(answer_lines, questions)
    
    # Step 2: For each question, find its answer
    for i, q in enumerate(questions):
        ref = f"REF-{chr(65 + i)}"
        
        if i in boundaries:
            bounds = boundaries[i]
            start = bounds['start']
            end = bounds['end']
            
            # Extract the answer text
            answer_raw = find_answer_text(answer_lines, start, end)
            
            # Clean up: remove global conclusion if it somehow got included
            answer_raw = remove_global_conclusion(answer_raw)
            
            # Remove next question if it appears at the end
            answer_raw = remove_next_question(answer_raw)
            
            # Clean up: remove leading "Ans" or "उत्तर"
            answer_clean = strip_question_restatement(answer_raw)
            answer_clean = strip_full_question_echo(answer_clean, q)
            
            # Get page numbers
            start_page = answer_line_pages[start] if answer_line_pages and 0 <= start < len(answer_line_pages) else None
            end_page = answer_line_pages[end] if answer_line_pages and 0 <= end < len(answer_line_pages) else None
            
            results.append({
                "ref": ref,
                "question": q,
                "matched": True,
                "start_line": start,
                "end_line": end,
                "start_page": start_page,
                "end_page": end_page,
                "answer": answer_clean,
                "answer_raw": answer_raw,
            })
        else:
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
    
    return results

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

def find_answer_text(lines: List[str], start_idx: int, end_idx: int) -> str:
    """Extract and clean answer text from lines."""
    # Collect lines for this answer
    answer_lines = []
    for i in range(start_idx, min(end_idx + 1, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
        # Skip if this line starts a new question (shouldn't happen but safety)
        if is_question_marker_only(line):
            continue
        answer_lines.append(line)
    
    return " ".join(answer_lines).strip()

GLOBAL_CONCLUSION_PATTERNS = [
    re.compile(r'^(?:निष्कर्ष|Conclusion|सारांश|Summary)\s*[:：]\s*$', re.IGNORECASE),
    re.compile(r'^=+\s*(?:निष्कर्ष|Conclusion|सारांश|Summary)\s*=+\s*$', re.IGNORECASE),
    re.compile(r'^##\s*(?:निष्कर्ष|Conclusion|सारांश|Summary)\s*$', re.IGNORECASE),
]

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
    NEW: lightweight, always-on diagnostic (not a hard failure) that flags
    matched answers which are suspiciously short compared to the rest of
    the document's answers -- a strong signal of truncation (start or end
    clipped) even when a range WAS found. This surfaces exactly the class
    of bug you reported ("skips starting/ending paragraphs") in the logs
    immediately, per-document, without needing a separate benchmark run.
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
            f"{len(official_questions)} question(s) found."
        )

    # Use the enhanced sequential mapping
    log("Mapping each question to its answer (enhanced sequential single-target search)...")
    qa_pairs = map_answers_sequential(
        answer_lines, official_questions, status_callback,
        answer_line_pages=answer_line_pages
    )

    # NEW: Post-process to handle embedded answers that were split out
    # For any answer that had embedded blocks, create additional Q-A pairs
    final_qa_pairs = []
    for p in qa_pairs:
        if p["matched"]:
            final_qa_pairs.append({
                "ref": p["ref"],
                "question": p["question"],
                "matched": True,
                "start_line": p["start_line"],
                "end_line": p["end_line"],
                "start_page": p["start_page"],
                "end_page": p["end_page"],
                "answer": p["answer"],
                "answer_raw": p["answer_raw"],
            })
            
            # Check for embedded answers
            if "embedded_answers" in p and p["embedded_answers"]:
                for embedded in p["embedded_answers"]:
                    # Try to find which REF this embedded question corresponds to
                    # Use the question number to match
                    q_num = embedded.get("question_num")
                    if q_num:
                        # Find the official question with this number
                        for i, q in enumerate(official_questions):
                            q_num_match = re.match(r'^\s*(\d+)[\.\)]', q)
                            if q_num_match and int(q_num_match.group(1)) == q_num:
                                embedded_ref = f"REF-{chr(65 + i)}"
                                # Check if we already have a result for this REF
                                existing = next((x for x in final_qa_pairs if x["ref"] == embedded_ref), None)
                                if existing is None:
                                    # This is a new answer we need to add
                                    final_qa_pairs.append({
                                        "ref": embedded_ref,
                                        "question": q,
                                        "matched": True,
                                        "start_line": p["start_line"],  # Approximate
                                        "end_line": p["end_line"],      # Approximate
                                        "start_page": p["start_page"],
                                        "end_page": p["end_page"],
                                        "answer": strip_question_restatement(embedded["answer_block"]),
                                        "answer_raw": embedded["answer_block"],
                                    })
                                    log(f"Added embedded answer for {embedded_ref}")
                                break
        else:
            final_qa_pairs.append(p)

    # Ensure all questions are represented
    # If any question is missing from final_qa_pairs, add an unmatched entry
    existing_refs = {p["ref"] for p in final_qa_pairs}
    for i, q in enumerate(official_questions):
        ref = f"REF-{chr(65 + i)}"
        if ref not in existing_refs:
            final_qa_pairs.append({
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

    matched_count = sum(1 for p in final_qa_pairs if p["matched"])
    log(f"Final: {matched_count} of {len(official_questions)} questions matched")

    _flag_suspiciously_short_answers(final_qa_pairs, log)

    log(f"Done -- {len(final_qa_pairs)} Q-A pairs ({matched_count} matched)")

    return ocr_json, final_qa_pairs

def save_outputs(ocr_json: dict, qa_pairs: list, output_dir: str = ".",
                  base_name: str = "document") -> tuple:
    ocr_path = os.path.join(output_dir, f"{base_name}_ocr.json")
    qa_path = os.path.join(output_dir, f"{base_name}_qa_pairs.json")

    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(ocr_json, f, ensure_ascii=False, indent=2)

    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    return ocr_path, qa_path
