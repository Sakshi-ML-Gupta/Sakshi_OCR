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
# STAGE 1: OCR -- Datalab (Chandra model) via /convert endpoint
#
# NOTE ON LAYOUT/BBOX: as of this writing, Chandra/Datalab's convert
# endpoint does NOT expose pixel-level bounding boxes for text lines
# (this is a confirmed, currently-open feature request on the
# datalab-to/chandra GitHub repo, issue #102 -- only Markdown/HTML/JSON
# text output is available, with no per-line coordinates). Because of
# that, "layout" signal in this pipeline is approximated from what IS
# available in the Markdown text: line order (position in the document)
# and leading whitespace/indentation -- NOT true (x, y) bounding boxes.
# If Datalab later exposes bbox data, build_line_records() below is the
# only place that needs to change to consume it.
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
# GROQ CALL INFRASTRUCTURE (shared plumbing)
# =========================================================

GROQ_MODEL = "openai/gpt-oss-120b"

TPM_LIMIT = 8000
TPM_SAFETY_FRACTION = 0.85
CHARS_PER_TOKEN_ESTIMATE = 2.0


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
                f"This will NOT be fixed by retrying. Check GROQ_API_KEY in your "
                f"environment/secrets. Original error: {e}"
            ) from e

        except (groq.RateLimitError, groq.BadRequestError) as e:
            last_error = e
            detail = _parse_rate_limit_detail(str(e))

            if detail and detail["limit_type"] == "TPD":
                raise Exception(
                    f"Groq daily token quota (TPD) exhausted: "
                    f"{detail['used']}/{detail['limit']} tokens used today. "
                    f"Resets in ~{detail['wait_seconds']/60:.0f} minute(s)."
                ) from e

            if detail:
                budget.record_actual_from_error(detail["used"], detail["limit"])
                log(
                    f"Rate/size limit hit (attempt {attempt}): {detail['limit_type']} "
                    f"limit={detail['limit']}, used={detail['used']}, "
                    f"requested={detail['requested']}. Waiting "
                    f"{detail['wait_seconds'] + 0.5:.1f}s before retrying..."
                )
                time.sleep(detail["wait_seconds"] + 0.5)
                budget.reset_window()
                skip_next_proactive_check = True
            else:
                wait_s = 5.0 * attempt
                log(f"Rate/size limit hit (attempt {attempt}): {e}. Waiting {wait_s:.1f}s...")
                time.sleep(wait_s)

        except Exception as e:
            last_error = e
            log(f"LLM call/parse attempt {attempt} failed: {e}")
            time.sleep(1)

    raise Exception(f"LLM call failed after {max_retries + 1} attempts. Last error: {last_error}")


def _strip_json_fences(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()
    return content


# =========================================================
# STAGE 1b: QUESTION PAPER PAGE / QUESTION DETECTION (Groq)
# (unchanged in spirit from earlier versions -- this part identifies
# WHICH pages are the printed question paper vs admin/cover vs answer
# pages, and extracts the canonical list of official questions. It is
# not the part of the pipeline that was misbehaving, so it is kept as
# a single validated LLM pass, with a dedup safety net.)
# =========================================================

MAX_CHARS_PER_CHUNK = 6000
CHUNK_OVERLAP_PAGES = 1

QP_SYSTEM_PROMPT = """You are analyzing OCR text extracted from a scanned student exam/assignment answer booklet. This could be from ANY institution, ANY subject, and ANY language (or a mix of languages/scripts). The booklet mixes pages of different kinds, in no guaranteed order:

1. ADMINISTRATIVE/COVER pages: roll number, programme/course code, student name, registration details, institution letterhead, blank cover sheets. NO exam question text, NO answer content.
2. QUESTION PAPER pages: the official printed list of numbered exam questions. These read as instructions/prompts DIRECTED AT the student (e.g. "Discuss X", "Explain Y", "Write notes on the following:"). Mark allocations may appear.
3. ANSWER pages: the student's own (handwritten, OCR'd) answers -- long, may restate a question briefly then write an extended response, and may contain the student's OWN numbered sub-points as part of their explanation (not separate exam questions).

You are shown only a PORTION of the document's pages at a time. Return ONLY valid JSON (no markdown fences, no commentary) in EXACTLY this shape:

{
  "question_paper_pages": [14, 16, 18],
  "admin_pages": [1, 2],
  "questions": ["1. Example question text. (10)", "2. Another example question. (10)"]
}

Rules:
- question_paper_pages/admin_pages: JSON arrays of SEPARATE page-number integers, e.g. [14, 16, 18] -- never merge into one number.
- A genuine question paper question is a PROMPT directed at the student. A numbered point inside a long answer is typically a STATEMENT/FACT, not an instruction.
- If a page's numbered items follow an "answer" label (in any language, e.g. "Ans", "Ans-"), or come after explanatory prose, that page is an ANSWER page -- exclude from question_paper_pages.
- CRITICAL TRAP: students often RESTATE the question as the first sentence of their answer before writing their own explanation. Such a page can look like a question paper page but is really the opening of a long ANSWER. Signals it's really an answer: noticeably more text than a concise instruction needs; developing-argument prose quality; the same/similar question already appears verbatim on a more concise, confident question-paper page (in which case exclude this longer one).
- When uncertain, prefer NOT marking a page as a question-paper page.
- Cover/admin pages go in admin_pages so they're excluded from BOTH question paper and answer text.
- Preserve EXACT original text/numbering of real questions.
- Output ONLY the JSON object."""


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
    blocks = [f"--- PAGE {p['page_number']} ---\n{p['raw_text']}" for p in pages]
    return "Here are the OCR'd pages shown in this chunk:\n\n" + "\n\n".join(blocks)


def _parse_qp_llm_response(content: str) -> tuple:
    content = _strip_json_fences(content)
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}\nRaw (first 500 chars): {content[:500]!r}")

    if not isinstance(data, dict):
        raise ValueError(f"LLM response must be a JSON object, got: {type(data).__name__}")
    if "question_paper_pages" not in data or "questions" not in data:
        raise ValueError(f"LLM response missing required keys. Got: {list(data.keys())}")

    qp_pages = [int(x) for x in data["question_paper_pages"]]
    admin_pages = [int(x) for x in data.get("admin_pages", [])]
    questions = [str(x).strip() for x in data["questions"] if str(x).strip()]

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
        if i != len(s) or len(set(result)) != len(result):
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


def _call_groq_for_chunk(client, pages_chunk: list, budget: "_TokenBudgetTracker", log, max_retries: int = 4) -> tuple:
    user_prompt = _build_qp_user_prompt(pages_chunk)
    return _call_groq_with_retries(client, QP_SYSTEM_PROMPT, user_prompt, _parse_qp_llm_response, budget, log, max_retries)


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
    return sorted(all_qp_pages), _dedup_questions(all_questions), sorted(all_admin_pages)


QUESTION_PAPER_ONLY_SYSTEM_PROMPT = """You are reading the OFFICIAL question paper pages of a student exam booklet (the printed list of questions, NOT the student's answers). You are given the complete, exact text of these pages, in order.

Your task: extract the COMPLETE, clean list of every distinct question/sub-part, exactly as printed, in printed order.

Rules for multi-part questions:
- If a numbered question has multiple LABELED sub-parts (e.g. "1. Identify and explain: (i)... (ii)... (iii)..."), output EACH sub-part as its OWN entry, carrying forward enough parent context (e.g. "1.(i)") to be self-contained.
- Applies to any labeled sub-structure: (i)/(ii)/(iii), (a)/(b)/(c), (क)/(ख)/(ग), etc.
- Decide this ONCE, consistently, for the whole document.
- Preserve EXACT original text -- no paraphrasing, no translation.
- Output in the SAME printed order -- never reorder.
- Do NOT output the same question or sub-part more than once, even if it appears to be printed twice (e.g. once in an index and once in the body).

Return ONLY valid JSON (no markdown fences, no commentary):

{"questions": ["<exact text 1>", "<exact text 2>", ...]}"""


def _build_canonical_questions_prompt(qp_pages: list) -> str:
    blocks = [f"--- PAGE {p['page_number']} ---\n{p['raw_text']}" for p in qp_pages]
    return "Here is the COMPLETE text of all question paper pages, in order:\n\n" + "\n\n".join(blocks)


def _parse_canonical_questions_response(content: str) -> list:
    content = _strip_json_fences(content)
    data = json.loads(content)
    if not isinstance(data, dict) or "questions" not in data:
        raise ValueError(f"Response missing 'questions' key: {data!r}")
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
    log(f"Extracting canonical question list from {len(qp_pages)} question-paper page(s)...")

    try:
        questions = _call_groq_with_retries(
            client, QUESTION_PAPER_ONLY_SYSTEM_PROMPT, user_prompt,
            _parse_canonical_questions_response, budget, log
        )
    except Exception as e:
        log(f"WARNING: canonical question extraction failed: {e}")
        return []

    deduped = _dedup_questions(questions)
    if len(deduped) != len(questions):
        log(f"Removed {len(questions) - len(deduped)} duplicate question(s) ({len(questions)} -> {len(deduped)})")
    questions = deduped

    log(f"Canonical question list: {len(questions)} question(s)")
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
    log(f"Split {len(pages)} page(s) into {len(chunks)} LLM chunk(s)")

    valid_page_numbers = {p["page_number"] for p in pages}
    max_page_number = max(valid_page_numbers) if valid_page_numbers else 0
    chunk_results = []
    chunk_failures = []

    for i, chunk in enumerate(chunks):
        page_nums_in_chunk = [p["page_number"] for p in chunk]
        log(f"Analyzing chunk {i+1}/{len(chunks)} (pages {page_nums_in_chunk})...")

        try:
            qp_pages_1based, questions, admin_pages_1based = _call_groq_for_chunk(client, chunk, budget, log)
        except Exception as e:
            log(f"WARNING: chunk {i+1}/{len(chunks)} failed, skipping: {e}")
            chunk_failures.append(str(e))
            continue

        def _recover_pages(pages_1based, label):
            recovered, truly_invalid = [], []
            for pn in pages_1based:
                if pn in valid_page_numbers:
                    recovered.append(pn)
                    continue
                split_result = _try_split_concatenated_page_number(pn, valid_page_numbers, max_page_number)
                if split_result:
                    log(f"Recovered concatenated {label} page numbers: {pn} -> {split_result}")
                    recovered.extend(split_result)
                else:
                    truly_invalid.append(pn)
            if truly_invalid:
                log(f"WARNING: out-of-range {label} page numbers ignored: {truly_invalid}")
            return sorted(set(recovered))

        qp_pages_1based = _recover_pages(qp_pages_1based, "question-paper")
        admin_pages_1based = _recover_pages(admin_pages_1based, "admin")
        admin_pages_1based = [p for p in admin_pages_1based if p not in qp_pages_1based]

        log(f"Chunk {i+1}/{len(chunks)}: {len(qp_pages_1based)} QP page(s), {len(admin_pages_1based)} admin page(s)")
        chunk_results.append((qp_pages_1based, [], admin_pages_1based))

    if chunk_failures and not chunk_results:
        raise Exception(f"All {len(chunks)} chunk(s) failed. First failure: {chunk_failures[0]}")
    elif chunk_failures:
        log(f"NOTE: {len(chunk_failures)} of {len(chunks)} chunk(s) failed -- QP page detection is PARTIAL")

    qp_pages_1based_merged, _, admin_pages_1based_merged = _merge_chunk_results(chunk_results)
    qp_page_indices_0based = sorted(pn - 1 for pn in qp_pages_1based_merged)
    admin_page_indices_0based = sorted(pn - 1 for pn in admin_pages_1based_merged)

    log(f"Question paper pages identified: {len(qp_page_indices_0based)} page(s)")
    log(f"Admin/cover pages identified: {len(admin_page_indices_0based)} page(s)")

    if len(qp_page_indices_0based) >= 2:
        qp_page_lengths = [(i, len(pages[i]["raw_text"])) for i in qp_page_indices_0based]
        lengths_only = [length for _, length in qp_page_lengths]
        median_length = sorted(lengths_only)[len(lengths_only) // 2]
        length_outliers = [pi for pi, length in qp_page_lengths if length > max(median_length * 3, 1500)]

        def _looks_like_student_answer(page_idx: int) -> bool:
            head = pages[page_idx]["raw_text"][:400]
            return bool(_ANSWER_START_RE.search(head))

        confirmed_outliers = [pi for pi in length_outliers if _looks_like_student_answer(pi)]
        rejected_outliers = [pi for pi in length_outliers if pi not in confirmed_outliers]

        for pi in rejected_outliers:
            length = dict(qp_page_lengths)[pi]
            log(
                f"NOT reclassifying page {pi + 1}: length is an outlier ({length} vs median "
                f"{median_length}) but no explicit answer-marker found -- treating as genuine QP content."
            )

        if confirmed_outliers and len(confirmed_outliers) <= len(qp_page_indices_0based) // 2:
            for pi in confirmed_outliers:
                length = dict(qp_page_lengths)[pi]
                log(f"RECLASSIFYING page {pi + 1}: outlier length ({length} chars) + explicit answer-marker found.")
            qp_page_indices_0based = [i for i in qp_page_indices_0based if i not in confirmed_outliers]
        elif confirmed_outliers:
            log(
                f"WARNING: {len(confirmed_outliers)} of {len(qp_page_indices_0based)} QP pages look like "
                f"answer openings -- too large a fraction to auto-reclassify safely, leaving as-is."
            )

    qp_pages_full = [pages[i] for i in qp_page_indices_0based]
    questions = extract_canonical_questions(qp_pages_full, status_callback)

    log(
        f"Final: {len(qp_page_indices_0based)} QP page(s), {len(questions)} canonical question(s), "
        f"{len(admin_page_indices_0based)} admin page(s)"
    )

    return qp_page_indices_0based, questions, admin_page_indices_0based


# =========================================================
# NOISE / OCR-ARTIFACT FILTERING
# (applied when building line records, so artifact lines can never
# become candidates, be validated as anchors, or appear in answer text)
# =========================================================

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
NOISE_LINE_MAX_CHARS = 40

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

# =========================================================
# Broader graphical-annotation-description detector: catches sentences
# describing examiner/grading marks (a circled number, an arrow drawn on
# the page, a tick, a cross) in ANY phrasing, not just lines starting
# with "there is a...". Confirmed real OCR output that slipped past the
# narrower check above: "A red circle containing the number 11. A long
# red arrow originates from the circle and points diagonally downwards
# and to the left across the blank space of the page." A line is
# flagged only if it contains BOTH an annotation-object noun (circle,
# arrow, tick, cross, underline, doodle, stamp...) AND annotation-context
# language (red pen/ink, diagonally, points to/from, blank space,
# margin, corner of the page, containing the number...) -- requiring
# both avoids false positives on genuine content that merely mentions
# one of these words in isolation (e.g. a geometry answer discussing
# "the circle" without any of the annotation-context language).
# =========================================================
_ANNOTATION_NOUNS_RE = re.compile(
    r'\b(?:circle[sd]?|arrow[s]?|tick\s*mark[s]?|cross(?:es)?|underlin\w*|scribbl\w*|'
    r'doodle\w*|stamp\w*|watermark\w*|logo\w*|checkmark[s]?|squiggle\w*|highlight\w*)\b',
    re.IGNORECASE
)
_ANNOTATION_CONTEXT_RE = re.compile(
    r'\b(?:red\s*(?:pen|ink)|diagonally|originates?|points?\s+(?:to|towards|from|at|down|up|left|right)|'
    r'pointing|blank\s+space|margin|corner\s+of\s+the\s+page|top\s+(?:right|left)\s+corner|'
    r'bottom\s+(?:right|left)\s+corner|containing\s+the\s+number|encircled|hand[- ]?drawn|'
    r'annotation|marked\s+in\s+red|written\s+in\s+red)\b',
    re.IGNORECASE
)


def _is_graphical_annotation_description(text: str) -> bool:
    if not text:
        return False
    return bool(_ANNOTATION_NOUNS_RE.search(text)) and bool(_ANNOTATION_CONTEXT_RE.search(text))


def _is_ocr_artifact_description(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _OCR_ARTIFACT_DESCRIPTION_RE.match(stripped):
        return True
    return _is_graphical_annotation_description(stripped)


def is_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r'^\s*\d{1,3}\s*$', stripped):
        return True
    if _is_ocr_artifact_description(stripped):
        return True
    if len(stripped) > NOISE_LINE_MAX_CHARS:
        return False
    return bool(NOISE_RE.search(stripped))


_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def _strip_graphical_annotation_sentences(text: str) -> str:
    """
    Removes any individual sentence within a longer block of already-
    extracted answer text that is itself an OCR-hallucinated description
    of a graphical annotation, even when embedded inside an otherwise
    genuine paragraph rather than standing as its own line. Belt-and-
    suspenders on top of the line-level is_noise() filtering above.
    """
    if not text:
        return text
    sentences = _SENTENCE_SPLIT_RE.split(text)
    kept = [s for s in sentences if not _is_graphical_annotation_description(s)]
    if len(kept) == len(sentences):
        return text
    return " ".join(s for s in kept if s.strip()).strip()


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


# =========================================================
# STAGE 2: LINE-WISE STRUCTURE
#
# Builds (page, line_idx, indent, text) records from the OCR'd answer
# pages, filtering out noise/artifact lines at the source so they can
# never become anchor candidates or leak into extracted answers. Indent
# (leading-whitespace count) is used later as a weak layout signal since
# true pixel bbox is not available from Chandra -- see the note above
# run_ocr().
# =========================================================

def build_line_records(answer_pages: list) -> list:
    """
    answer_pages: list of {"page_number": int, "raw_text": str} for
    pages already confirmed to be answer pages (QP/admin pages excluded
    upstream).

    Returns a list of dicts: {gidx, page, text, indent, blank_before}
    -- one entry per non-noise line, in document order. gidx is the
    stable global 0-based index used throughout the rest of the
    pipeline (equivalent to the old "answer_lines" index).
    """
    records = []
    for page in answer_pages:
        prev_blank = True
        for raw_line in page["raw_text"].split("\n"):
            if is_noise(raw_line):
                prev_blank = not raw_line.strip()
                continue
            stripped = raw_line.strip()
            if not stripped:
                prev_blank = True
                continue
            indent = len(raw_line) - len(raw_line.lstrip())
            records.append({
                "gidx": len(records),
                "page": page["page_number"],
                "text": stripped,
                "indent": indent,
                "blank_before": prev_blank,
            })
            prev_blank = False
    return records


# =========================================================
# STAGE 3: REGEX + LAYOUT CANDIDATE DETECTION
# =========================================================

_ANSWER_LABEL_NUM_RE = re.compile(
    r'^\s*(?:Ans(?:wer)?[.\s:-]*|उत्तर\s*|प्र[०.\s]*|प्रश्न[.\s]*|Q\.?\s*)(\d+)',
    re.IGNORECASE
)

_LEADING_QNUM_RE = re.compile(r'^\s*(\d+)\s*[\.\)]')


def _extract_label_number(text: str):
    m = _ANSWER_LABEL_NUM_RE.match(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _extract_question_number(question_text: str):
    m = _LEADING_QNUM_RE.match(question_text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def find_regex_layout_candidates(line_records: list, questions: list) -> dict:
    """
    STAGE 3a (regex): scans line_records for explicit answer-number
    labels (e.g. "Ans 5-", "उत्तर 6", "Q.7)") and matches the extracted
    number against each question's own leading number. A candidate whose
    line also sits at a natural layout boundary (indent<=2, i.e. left-
    aligned, and/or preceded by a blank line) is marked layout_ok=True --
    a light-weight proxy for "genuine paragraph start" given that true
    bbox/layout data isn't available (see note above run_ocr()).

    Returns: dict question_idx -> list of candidate dicts
             {gidx, source, confidence, layout_ok}
    """
    question_numbers = [_extract_question_number(q) for q in questions]
    candidates_by_qidx = {i: [] for i in range(len(questions))}

    for rec in line_records:
        label_num = _extract_label_number(rec["text"])
        if label_num is None:
            continue
        layout_ok = rec["blank_before"] or rec["indent"] <= 2
        for qi, qn in enumerate(question_numbers):
            if qn is not None and qn == label_num:
                candidates_by_qidx[qi].append({
                    "gidx": rec["gidx"],
                    "source": "regex",
                    "confidence": "strong",
                    "layout_ok": layout_ok,
                })

    return candidates_by_qidx


def find_question_boundaries_by_similarity(
    line_records: list,
    questions: list,
    similarity_threshold: float = 0.30,
    window: int = 4
) -> list:
    """
    STAGE 3b (layout/content fallback): for questions with NO explicit
    numeric label anywhere in the text (common when students don't write
    "Ans N-" at all), this scores short spans of consecutive lines
    against each question's own wording via word-overlap similarity, and
    proposes the best-scoring, order-respecting candidate per question.
    This is the "layout" signal in cases where regex labels are absent
    -- weaker/lower-confidence than a regex match, but far better than
    no candidate at all, and it still respects document order (a
    candidate for question N is never chosen before question N-1's
    chosen candidate).
    """
    texts = [r["text"] for r in line_records]
    candidates_by_question = {}

    for i in range(len(texts)):
        line_i = texts[i]
        if len(line_i) < 8:
            continue
        for w in range(1, window + 1):
            if i + w > len(texts):
                break
            combined = " ".join(texts[i + k] for k in range(w) if texts[i + k])
            if len(combined) < 10:
                continue
            combined_clean = strip_leading_label(combined)
            for q in questions:
                q_clean = strip_leading_label(q)
                score = max(similarity(combined, q), similarity(combined_clean, q_clean))
                if score >= similarity_threshold:
                    candidates_by_question.setdefault(q, []).append({
                        "gidx": line_records[i]["gidx"],
                        "span": w,
                        "score": score,
                    })

    for q in candidates_by_question:
        candidates_by_question[q].sort(key=lambda c: -c["score"])

    final = []
    last_gidx = -1
    for q in questions:
        cands = candidates_by_question.get(q, [])
        chosen = None
        for c in cands:
            if c["gidx"] > last_gidx:
                chosen = c
                break
        if chosen is not None:
            final.append({"question": q, **chosen})
            last_gidx = chosen["gidx"]

    return final


def add_similarity_candidates(line_records: list, questions: list, candidates_by_qidx: dict) -> dict:
    """
    Supplements candidates_by_qidx with weak, similarity-based candidates
    for any question that got ZERO regex-label candidates.
    """
    missing_qidxs = [i for i, c in candidates_by_qidx.items() if not c]
    if not missing_qidxs:
        return candidates_by_qidx

    missing_questions = [questions[i] for i in missing_qidxs]
    boundaries = find_question_boundaries_by_similarity(line_records, missing_questions)

    text_to_qidxs = {}
    for i in missing_qidxs:
        text_to_qidxs.setdefault(questions[i], []).append(i)

    for b in boundaries:
        idxs = text_to_qidxs.get(b["question"])
        if not idxs:
            continue
        qi = idxs.pop(0)
        start_gidx = b["gidx"] + b.get("span", 1)
        candidates_by_qidx[qi].append({
            "gidx": min(start_gidx, len(line_records) - 1) if line_records else b["gidx"],
            "source": "similarity",
            "confidence": "weak",
            "score": b["score"],
            "layout_ok": True,
        })

    return candidates_by_qidx


# =========================================================
# STAGE 4: GROQ -- VALIDATE / CORRECT ONLY THE ANCHOR CANDIDATES
#
# This is deliberately a SMALL, BOUNDED task per question: "here are 1-3
# candidate lines with a bit of surrounding context -- which one (if
# any) is the true start, or correct me if the true start is visible
# nearby." This is fundamentally different from the earlier "search a
# whole unbounded window for a start line with no candidates at all" --
# it plays to the LLM's strength (judging/verifying a small, well-
# specified set of options) instead of asking it to do an open-ended
# search, which is what caused missed openings and merged answers
# before.
# =========================================================

ANCHOR_VALIDATION_SYSTEM_PROMPT = """You are verifying the exact starting line of a student's answer to ONE specific exam question, inside an OCR'd, line-numbered exam booklet.

You are given:
1. The exact text of the target question.
2. One or more CANDIDATE lines that a regex/layout heuristic flagged as possibly marking where this answer begins, each shown with a few lines of surrounding context (all line-numbered).

Your task: decide which candidate (if any) is the TRUE start of this question's answer. If none of the shown candidates are correct but the true start IS visible somewhere in the context shown, report that line instead (status "corrected"). If the true start is not visible anywhere in what's shown, say so (status "not_found") -- do not guess.

RULE -- ALWAYS THE EARLIEST LINE: an answer's true start includes any opening label, restatement, or short introductory sentence that precedes the "obviously on-topic" part. Never pick a later, clearer line if an earlier one already begins the same answer.

RULE -- IGNORE OCR ARTIFACT/ANNOTATION DESCRIPTIONS: lines describing a visual mark on the page (a red circle, an arrow, a stamp, a logo, a scribble, a tick mark) are never real student content and can never be the true start. If real content resumes right after such a description, that real content line is the one to report.

RULE -- REPEATED CONTENT IS NORMAL: the same fact/definition can legitimately appear in more than one answer. Do not reject a candidate just because similar wording appears elsewhere.

Return ONLY valid JSON (no markdown fences, no commentary) in exactly one of these three shapes:

{"status": "confirmed", "line": 42}
{"status": "corrected", "line": 40}
{"status": "not_found"}"""

ANCHOR_CONTEXT_BEFORE = 3
ANCHOR_CONTEXT_AFTER = 8
ANCHOR_MAX_CANDIDATES_SHOWN = 3


def _build_anchor_validation_prompt(line_records: list, question_text: str, candidates: list) -> str:
    total = len(line_records)
    blocks = []
    for c in candidates:
        gidx = c["gidx"]
        lo = max(0, gidx - ANCHOR_CONTEXT_BEFORE)
        hi = min(total - 1, gidx + ANCHOR_CONTEXT_AFTER)
        lines_block = "\n".join(f"[{line_records[i]['gidx']}] {line_records[i]['text']}" for i in range(lo, hi + 1))
        blocks.append(f"CANDIDATE at line {gidx} (source={c['source']}):\n{lines_block}")
    candidates_block = "\n\n".join(blocks)
    return f"TARGET QUESTION: {question_text}\n\n{candidates_block}"


def _parse_anchor_validation_response(content: str) -> tuple:
    content = _strip_json_fences(content)
    data = json.loads(content)
    if not isinstance(data, dict) or "status" not in data:
        raise ValueError(f"Response missing 'status' key: {data!r}")
    status = data["status"]
    if status == "not_found":
        return "not_found", None
    if status in ("confirmed", "corrected"):
        if "line" not in data:
            raise ValueError(f"status={status!r} but 'line' missing: {data!r}")
        return status, int(data["line"])
    raise ValueError(f"Unrecognized status: {status!r}")


def validate_anchor_for_question(client, line_records: list, question_text: str, candidates: list,
                                   budget: "_TokenBudgetTracker", log) -> int:
    """
    Sends up to ANCHOR_MAX_CANDIDATES_SHOWN candidates (best-first) to
    Groq for confirmation/correction. Returns the confirmed/corrected
    global line index, or None if not found / call failed / result out
    of bounds.
    """
    if not candidates:
        return None

    candidates_sorted = sorted(
        candidates,
        key=lambda c: (0 if c["confidence"] == "strong" else 1, -c.get("score", 0))
    )[:ANCHOR_MAX_CANDIDATES_SHOWN]

    prompt = _build_anchor_validation_prompt(line_records, question_text, candidates_sorted)
    try:
        status, line = _call_groq_with_retries(
            client, ANCHOR_VALIDATION_SYSTEM_PROMPT, prompt,
            _parse_anchor_validation_response, budget, log, max_retries=2
        )
    except Exception as e:
        log(f"WARNING: anchor validation call failed: {e}")
        return None

    if status == "not_found" or line is None:
        return None
    if 0 <= line < len(line_records):
        return line
    log(f"WARNING: anchor validation returned out-of-range line {line} -- ignoring")
    return None


# =========================================================
# STAGE 5 (Python): compute ranges = next_confirmed_anchor - current
# STAGE 6: VALIDATOR (no missing lines, no overlaps, no gaps)
# STAGE 7: retry only failed regions (bounded, isolated search)
# =========================================================

def _enforce_monotonic_anchors(confirmed: dict, num_questions: int, log) -> tuple:
    """
    Walks questions in order and keeps only anchors that strictly
    increase relative to the previous KEPT anchor. Any anchor that is
    <= the previous kept anchor cannot be trusted (it would create an
    overlap/reordering) and is discarded, marking that question for
    retry instead.

    Returns (ordered_valid, invalid_qidxs) where ordered_valid is a list
    of (qidx, gidx) tuples in increasing gidx order.
    """
    ordered_valid = []
    invalid_qidxs = []
    last_gidx = -1

    for qi in range(num_questions):
        gidx = confirmed.get(qi)
        if gidx is None:
            invalid_qidxs.append(qi)
            continue
        if gidx <= last_gidx:
            log(
                f"WARNING: confirmed anchor for question {qi} (line {gidx}) is not after "
                f"the previous confirmed anchor (line {last_gidx}) -- discarding as untrustworthy, "
                f"will retry"
            )
            invalid_qidxs.append(qi)
            continue
        ordered_valid.append((qi, gidx))
        last_gidx = gidx

    return ordered_valid, invalid_qidxs


RETRY_SEARCH_SYSTEM_PROMPT = """You are searching for exactly ONE thing in a block of line-numbered OCR text from a student's exam answer booklet: the EARLIEST line where the response to ONE SPECIFIC question begins.

You are given the exact text of the target question and a bounded window of text -- this window is known to sit between two OTHER already-confirmed answers, so the target answer (if present at all) is somewhere in this exact stretch, or is entirely absent (the student may have skipped this question).

RULE -- ALWAYS THE EARLIEST LINE: an answer's true start includes any opening label, restatement, or short introductory sentence -- never pick a later, "clearer" line if an earlier one already begins the same answer.

RULE -- IGNORE OCR ARTIFACT/ANNOTATION DESCRIPTIONS: lines describing a visual mark on the page (a red circle, an arrow, a stamp, a scribble) are never the start of an answer.

If the target question's answer is genuinely not present anywhere in this window, say so plainly -- that is a valid, expected outcome (the student may have skipped the question).

Return ONLY valid JSON (no markdown fences, no commentary):

{"found": true, "start_line": 42}
or
{"found": false}

If found, start_line MUST be one of the exact line numbers shown in [brackets]."""


def _parse_retry_search_response(content: str) -> tuple:
    content = _strip_json_fences(content)
    data = json.loads(content)
    if not isinstance(data, dict) or "found" not in data:
        raise ValueError(f"Response missing 'found' key: {data!r}")
    if not data["found"]:
        return False, None
    if "start_line" not in data:
        raise ValueError("found=true but 'start_line' missing")
    return True, int(data["start_line"])


RETRY_WINDOW_CHARS = 11000


def retry_failed_region(client, line_records: list, question_text: str,
                          region_start: int, region_end: int,
                          budget: "_TokenBudgetTracker", log) -> int:
    """
    STAGE 7: targeted retry, bounded to [region_start, region_end]
    inclusive -- i.e. only the gap between two already-confirmed
    neighboring anchors (or the document edges), never the whole
    document. This is what makes the retry cheap and safe: even if the
    LLM's isolated judgment is imperfect here, it can't reach into
    already-confirmed territory and corrupt it.
    """
    if region_start > region_end or region_start >= len(line_records):
        return None

    window = [r for r in line_records if region_start <= r["gidx"] <= region_end]
    if not window:
        return None

    # If the region is large, slide through it in sub-windows; otherwise
    # a single call covers it.
    pos = 0
    while pos < len(window):
        sub = []
        chars = 0
        i = pos
        while i < len(window) and (not sub or chars + len(window[i]["text"]) <= RETRY_WINDOW_CHARS):
            sub.append(window[i])
            chars += len(window[i]["text"])
            i += 1

        lines_block = "\n".join(f"[{r['gidx']}] {r['text']}" for r in sub)
        prompt = f"TARGET QUESTION: {question_text}\n\nTEXT WINDOW (line-numbered):\n{lines_block}"

        try:
            found, start_line = _call_groq_with_retries(
                client, RETRY_SEARCH_SYSTEM_PROMPT, prompt,
                _parse_retry_search_response, budget, log, max_retries=2
            )
        except Exception as e:
            log(f"WARNING: retry search call failed for region [{region_start},{region_end}]: {e}")
            found, start_line = False, None

        if found and start_line is not None:
            valid_ids = {r["gidx"] for r in sub}
            if start_line in valid_ids:
                return start_line
            log(f"WARNING: retry search returned out-of-window line {start_line} -- ignoring")

        pos = i

    return None


LAST_ANSWER_END_SYSTEM_PROMPT = """You are looking at the FINAL portion of a student's exam answer booklet (OCR'd, line-numbered), starting from where their LAST answer begins. This tail section may contain:
1. The remainder of the student's genuine answer content for the target question -- INCLUDE this.
2. AFTER the real answer content ends, there may be trailing material that is NOT part of the answer: an overall assignment/exam-level closing remark (not specific to this question), an institutional footer, "thank you" notes, etc. -- EXCLUDE this.

Find the LAST line number that is still genuinely part of the student's answer to the target question. If the entire text shown is genuine answer content, the last line of the text IS the answer's end.

Do NOT exclude a line just because it sounds like a summary/concluding sentence OF THIS ANSWER ITSELF -- a student's own concluding sentence is normal and should be INCLUDED. Only exclude content that is clearly about the assignment/paper/course as a whole.

If you are not confident there is any trailing non-answer material, default to including everything.

Return ONLY valid JSON (no markdown fences, no commentary):

{"end_line": 87}"""


def _parse_last_answer_end_response(content: str) -> int:
    content = _strip_json_fences(content)
    data = json.loads(content)
    if not isinstance(data, dict) or "end_line" not in data:
        raise ValueError(f"Response missing 'end_line' key: {data!r}")
    return int(data["end_line"])


LAST_ANSWER_END_TAIL_CHARS = 9000


def check_last_answer_end(client, line_records: list, question_text: str,
                            start_gidx: int, budget: "_TokenBudgetTracker", log) -> int:
    """
    Verifies where the chronologically LAST matched answer actually ends
    instead of blindly assuming it runs to the very end of the document,
    so a trailing whole-assignment conclusion doesn't get absorbed into
    it. Bounded to the tail of the document, so this stays cheap.
    """
    total = len(line_records)
    fallback_end = line_records[-1]["gidx"] if line_records else start_gidx
    tail = [r for r in line_records if r["gidx"] >= start_gidx]
    if not tail:
        return fallback_end

    chars = 0
    collected = []
    for r in reversed(tail):
        if collected and chars + len(r["text"]) > LAST_ANSWER_END_TAIL_CHARS:
            break
        collected.append(r)
        chars += len(r["text"])
    collected.reverse()

    lines_block = "\n".join(f"[{r['gidx']}] {r['text']}" for r in collected)
    prompt = f"TARGET QUESTION: {question_text}\n\nTEXT (line-numbered, tail of the document):\n{lines_block}"

    try:
        end_line = _call_groq_with_retries(
            client, LAST_ANSWER_END_SYSTEM_PROMPT, prompt,
            _parse_last_answer_end_response, budget, log, max_retries=2
        )
    except Exception as e:
        log(f"WARNING: last-answer-end check failed: {e}")
        return fallback_end

    valid_ids = {r["gidx"] for r in collected}
    if end_line in valid_ids and end_line >= start_gidx:
        if end_line != fallback_end:
            log(f"  last-answer-end check: trimmed end from line {fallback_end} to line {end_line}")
        return end_line

    log(f"WARNING: last-answer-end check returned invalid end_line {end_line} -- keeping full tail")
    return fallback_end


def validate_ranges(ranges: list, total_lines: int, num_questions: int, log) -> dict:
    """
    STAGE 6 -- Validator. Checks the final set of answer ranges for:
      - no overlaps (ranges must be strictly increasing and non-overlapping)
      - no unexplained internal gaps larger than a small tolerance
        (a few lines of blank/noise between answers is normal and
        already filtered out upstream; a LARGE gap suggests a missed
        answer that never got a confirmed anchor at all)
      - completeness (every question has a matched range)
    Returns a report dict; never raises -- issues are logged and
    returned for the caller to act on (e.g. decide whether to warn the
    user), consistent with this pipeline's existing "log loudly, don't
    silently drop data" philosophy.
    """
    report = {"overlaps": [], "large_gaps": [], "unmatched": [], "ok": True}

    matched = sorted([r for r in ranges if r.get("matched")], key=lambda r: r["start_line"])
    for i in range(len(matched) - 1):
        a, b = matched[i], matched[i + 1]
        if a["end_line"] >= b["start_line"]:
            report["overlaps"].append((a["ref"], b["ref"]))
            report["ok"] = False
        else:
            gap = b["start_line"] - a["end_line"] - 1
            if gap > 30:  # a handful of blank/noise lines is normal; dozens is suspicious
                report["large_gaps"].append((a["ref"], b["ref"], gap))

    unmatched = [r["ref"] for r in ranges if not r.get("matched")]
    report["unmatched"] = unmatched
    if unmatched:
        report["ok"] = False

    if report["overlaps"]:
        log(f"VALIDATOR: found {len(report['overlaps'])} overlapping range(s): {report['overlaps']}")
    if report["large_gaps"]:
        log(
            f"VALIDATOR: found {len(report['large_gaps'])} unusually large gap(s) between "
            f"consecutive answers (may indicate a skipped/missed answer in between): "
            f"{report['large_gaps']}"
        )
    if unmatched:
        log(f"VALIDATOR: {len(unmatched)} question(s) still unmatched after retry: {unmatched}")
    if report["ok"]:
        log("VALIDATOR: all ranges are contiguous, non-overlapping, and complete.")

    return report


def map_answers_anchor_based(answer_pages: list, questions: list, status_callback=None) -> list:
    """
    Full STAGE 2-7 pipeline:
      2. build_line_records       -- line-wise structure (page, indent, text)
      3. find_regex_layout_candidates + add_similarity_candidates
                                   -- regex/layout candidate anchors per question
      4. validate_anchor_for_question (Groq)
                                   -- validate/correct ONLY the candidates, per question
      5. Python range computation -- end = next confirmed anchor - 1
      6. validate_ranges          -- completeness/overlap/gap check
      7. retry_failed_region (Groq, bounded)
                                   -- only for questions still unmatched after stage 4,
                                      searched ONLY within the gap between its confirmed
                                      neighbors, not the whole document

    Returns a list of dicts, one per question (same shape as before):
      ref, question, matched, start_line, end_line, start_page, end_page,
      answer, answer_raw
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

    # STAGE 2
    line_records = build_line_records(answer_pages)
    total_lines = len(line_records)
    log(f"Stage 2: built {total_lines} line record(s) from {len(answer_pages)} answer page(s)")

    if total_lines == 0:
        return [
            {"ref": f"REF-{chr(65+i)}", "question": q, "matched": False,
             "start_line": None, "end_line": None, "start_page": None, "end_page": None,
             "answer": "", "answer_raw": ""}
            for i, q in enumerate(questions)
        ]

    # STAGE 3
    candidates_by_qidx = find_regex_layout_candidates(line_records, questions)
    n_regex = sum(1 for c in candidates_by_qidx.values() if c and c[0]["source"] == "regex")
    candidates_by_qidx = add_similarity_candidates(line_records, questions, candidates_by_qidx)
    n_with_candidates = sum(1 for c in candidates_by_qidx.values() if c)
    log(
        f"Stage 3: found candidate anchor(s) for {n_with_candidates}/{len(questions)} question(s) "
        f"({n_regex} with an explicit regex label match)"
    )

    # STAGE 4
    confirmed = {}
    for qi, q in enumerate(questions):
        ref = f"REF-{chr(65 + qi)}"
        cands = candidates_by_qidx.get(qi, [])
        if not cands:
            log(f"  {ref}: no candidates from stage 3 -- deferring to stage 7 retry")
            continue
        log(f"  {ref}: validating {len(cands)} candidate(s) with Groq...")
        gidx = validate_anchor_for_question(client, line_records, q, cands, budget, log)
        if gidx is not None:
            confirmed[qi] = gidx
            log(f"    confirmed at line {gidx}")
        else:
            log(f"    not confirmed -- deferring to stage 7 retry")

    # STAGE 5 + monotonic-order enforcement
    ordered_valid, invalid_qidxs = _enforce_monotonic_anchors(confirmed, len(questions), log)
    log(f"Stage 5: {len(ordered_valid)} question(s) have a trustworthy confirmed anchor")

    # STAGE 7: retry only the failed regions, bounded by confirmed neighbors
    if invalid_qidxs:
        log(f"Stage 7: retrying {len(invalid_qidxs)} unconfirmed question(s), bounded to their gap regions...")
        anchor_map = dict(ordered_valid)  # qidx -> gidx, for confirmed ones so far

        for qi in invalid_qidxs:
            ref = f"REF-{chr(65 + qi)}"
            # bound the retry window using the nearest CONFIRMED neighbors
            # in question order (not just document order), so the retry
            # never overlaps already-confirmed territory.
            prev_gidx = -1
            for pj in range(qi - 1, -1, -1):
                if pj in anchor_map:
                    prev_gidx = anchor_map[pj]
                    break
            next_gidx = total_lines  # exclusive upper bound if no later anchor found
            for nj in range(qi + 1, len(questions)):
                if nj in anchor_map:
                    next_gidx = anchor_map[nj]
                    break

            region_start = prev_gidx + 1
            region_end = next_gidx - 1
            if region_start > region_end:
                log(f"  {ref}: no room between neighboring confirmed anchors -- leaving unmatched")
                continue

            log(f"  {ref}: retrying within lines [{region_start}, {region_end}]...")
            gidx = retry_failed_region(client, line_records, questions[qi], region_start, region_end, budget, log)
            if gidx is not None:
                anchor_map[qi] = gidx
                log(f"    recovered at line {gidx}")
            else:
                log(f"    still not found -- marking unmatched")

        ordered_valid = sorted(anchor_map.items(), key=lambda kv: kv[1])

    # Build final ranges: end = next confirmed anchor's start - 1, except
    # for the chronologically LAST one, which is separately verified.
    ranges_by_qidx = {}
    for idx, (qi, gidx) in enumerate(ordered_valid):
        if idx + 1 < len(ordered_valid):
            end_gidx = ordered_valid[idx + 1][1] - 1
        else:
            end_gidx = check_last_answer_end(client, line_records, questions[qi], gidx, budget, log)
        ranges_by_qidx[qi] = {"start_line": gidx, "end_line": end_gidx}

    # Assemble per-question results
    text_by_gidx = {r["gidx"]: r["text"] for r in line_records}
    page_by_gidx = {r["gidx"]: r["page"] for r in line_records}
    all_gidx_sorted = [r["gidx"] for r in line_records]

    results = []
    ranges_for_validator = []
    for qi, q in enumerate(questions):
        ref = f"REF-{chr(65 + qi)}"
        r = ranges_by_qidx.get(qi)
        if r is None:
            results.append({
                "ref": ref, "question": q, "matched": False,
                "start_line": None, "end_line": None,
                "start_page": None, "end_page": None,
                "answer": "", "answer_raw": "",
            })
            ranges_for_validator.append({"ref": ref, "matched": False})
            continue

        s, e = r["start_line"], r["end_line"]
        verbatim = [text_by_gidx[g] for g in all_gidx_sorted if s <= g <= e]
        answer_raw = " ".join(verbatim).strip()
        answer_clean = strip_question_restatement(answer_raw)
        answer_clean = strip_full_question_echo(answer_clean, q)
        answer_clean = _strip_graphical_annotation_sentences(answer_clean)

        results.append({
            "ref": ref, "question": q, "matched": True,
            "start_line": s, "end_line": e,
            "start_page": page_by_gidx.get(s), "end_page": page_by_gidx.get(e),
            "answer": answer_clean, "answer_raw": answer_raw,
        })
        ranges_for_validator.append({"ref": ref, "matched": True, "start_line": s, "end_line": e})

    # STAGE 6: Validator
    validate_ranges(ranges_for_validator, total_lines, len(questions), log)

    matched_count = sum(1 for r in results if r["matched"])
    log(f"Anchor-based mapping complete: {matched_count}/{len(questions)} question(s) matched")

    return results


def _sanity_check_answer_pages(line_records: list, num_questions: int, log=print) -> bool:
    total_chars = sum(len(r["text"]) for r in line_records)
    avg_chars_per_question = total_chars / max(num_questions, 1)
    MIN_PLAUSIBLE_CHARS_PER_QUESTION = 200

    if avg_chars_per_question < MIN_PLAUSIBLE_CHARS_PER_QUESTION:
        log(
            f"WARNING: 'answer pages' contain only {total_chars} total characters for "
            f"{num_questions} question(s) (~{avg_chars_per_question:.0f} chars/question). "
            f"This is far too little for real essay-style answers and strongly suggests "
            f"the question-paper/answer-page split misclassified pages."
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
                f"WARNING: possible truncated answer for '{p['question'][:60]}...' -- only "
                f"{length} chars vs median {median_len} chars. Worth spot-checking."
            )


def _flag_duplicate_matched_answers(qa_pairs: list, log=print) -> None:
    matched = [p for p in qa_pairs if p.get("matched") and p["answer"].strip()]
    for i in range(len(matched)):
        for j in range(i + 1, len(matched)):
            a, b = matched[i], matched[j]
            ratio = difflib.SequenceMatcher(None, a["answer"], b["answer"]).ratio()
            if ratio >= 0.9:
                log(
                    f"WARNING: near-identical answer text for two different questions -- "
                    f"likely a duplicate/near-duplicate canonical question: "
                    f"'{a['question'][:50]}...' and '{b['question'][:50]}...'"
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

    line_records = build_line_records(answer_pages)
    log(f"Flattened {len(line_records)} answer line(s)")

    pages_look_plausible = _sanity_check_answer_pages(line_records, len(official_questions), log)
    if not pages_look_plausible:
        raise Exception(
            "The 'answer pages' identified in this document do not contain enough text "
            f"to plausibly hold real essay-style answers for the {len(official_questions)} "
            "question(s) found. This usually means the question-paper/answer-page split "
            "misclassified pages -- check 'Question paper pages detected' above."
        )

    log("Mapping each question to its answer (regex+layout candidates -> Groq anchor validation -> Python ranges -> validator -> targeted retry)...")
    qa_pairs = map_answers_anchor_based(answer_pages, official_questions, status_callback)

    matched_count = sum(1 for p in qa_pairs if p["matched"])
    log(f"Matched {matched_count} of {len(official_questions)} questions")

    for p in qa_pairs:
        if not p["matched"]:
            log(f"WARNING: No match found for: {p['question'][:60]}")

    if matched_count == 0:
        raise Exception(
            "Could not match any questions to answers.\n"
            f"Official questions: {official_questions}"
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
