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
                f"file_input.read() returned {type(data).__name__}, expected bytes. Open binary mode ('rb')."
            )
        name = getattr(file_input, "name", default_name)
        return bytes(data), _coerce_name(name, default_name)

    raise TypeError(
        f"Unsupported file_input type: {type(file_input).__name__}."
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
                    f"[DIAGNOSTIC] 'os.PathLike, not tuple' error INSIDE {func.__name__}(). "
                    f"file_input type={type(file_input).__name__}, repr={file_input!r}. Error: {e}"
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
        f"WARNING: No page-break marker recognized in Datalab output. "
        f"Treating entire document as single page."
    )
    return [markdown.strip()]


def run_ocr(file_content: bytes, file_name: str, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    file_name = _coerce_name(file_name, default_name="document.pdf")

    if not isinstance(file_content, (bytes, bytearray)):
        raise TypeError(f"run_ocr() expected file_content as bytes, got {type(file_content).__name__}")

    api_key = get_api_key("DATALAB_API_KEY")
    if not api_key:
        raise Exception("DATALAB_API_KEY not found in secrets or environment")

    size_mb = len(file_content) / (1024 * 1024)
    if size_mb > 45:
        raise Exception(f"File is {size_mb:.1f}MB, which exceeds 45MB upload limit.")

    headers = {"X-API-Key": api_key}
    log(f"Submitting document to Datalab OCR... ({size_mb:.1f}MB)")

    resp = httpx.post(
        f"{DATALAB_BASE_URL}/api/v1/convert",
        headers=headers,
        files={"file": (file_name, file_content, "application/pdf")},
        data={"output_format": "markdown", "mode": "accurate", "paginate": "true"},
        timeout=120
    )

    if resp.status_code != 200:
        raise Exception(f"Datalab submit error {resp.status_code}: {resp.text}")

    data = resp.json()
    if not data.get("success", True):
        raise Exception(f"Datalab submit failed: {data.get('error')}")

    check_url = data["request_check_url"]
    log("Document submitted -- polling for OCR result...")

    for attempt in range(150):
        poll_resp = httpx.get(check_url, headers=headers, timeout=60)
        if poll_resp.status_code != 200:
            raise Exception(f"Datalab poll error {poll_resp.status_code}: {poll_resp.text}")

        result = poll_resp.json()
        if result.get("status") == "complete":
            log("OCR complete -- parsing pages...")
            break
        if result.get("status") == "failed" or result.get("error"):
            raise Exception(f"Datalab conversion failed: {result.get('error')}")

        time.sleep(2)
    else:
        raise Exception("Datalab conversion timed out after 5 minutes")

    markdown = result.get("markdown") or ""
    if not markdown.strip():
        raise Exception("Datalab returned empty markdown output")

    page_texts = _split_paginated_markdown(markdown, result.get("page_count"), log=log)
    pages = [{"page_number": idx + 1, "raw_text": text} for idx, text in enumerate(page_texts)]
    log(f"OCR done -- {len(pages)} page(s) extracted")
    return pages


def build_ocr_json(pages: list) -> dict:
    return {
        "total_pages": len(pages),
        "pages": [{"page_number": p["page_number"], "text": p["raw_text"]} for p in pages]
    }


@_diagnose_tuple_errors
def process_reference(file_input, status_callback=None):
    file_bytes, file_name = _normalize_file_input(file_input, default_name="reference.pdf")
    pages = run_ocr(file_bytes, file_name, status_callback)
    return build_ocr_json(pages)


# =========================================================
# LLM & TOKEN BUDGET
# =========================================================

GROQ_MODEL = "openai/gpt-oss-120b"
TPM_LIMIT = 8000
TPM_SAFETY_FRACTION = 0.85
CHARS_PER_TOKEN_ESTIMATE = 2.0
MAX_CHARS_PER_CHUNK = 6000


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
        log(f"Pacing requests: waiting {wait_s:.1f}s to avoid TPM throttle...")
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
    r'on\s+tokens\s+per\s+(minute|day)\s*\((TPM\vert{}TPD)\).*?'
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
        "limit": int(limit),
        "used": int(used),
        "requested": int(requested),
        "wait_seconds": wait_seconds,
    }


def _call_groq_with_retries(client, system_prompt: str, user_prompt: str,
                              response_parser, budget: "_TokenBudgetTracker",
                              log, max_retries: int = 4):
    import groq

    estimated_tokens = int((len(system_prompt) + len(user_prompt)) / CHARS_PER_TOKEN_ESTIMATE) + 800
    skip_next_proactive_check = False

    for attempt in range(1, max_retries + 2):
        if not skip_next_proactive_check:
            budget.wait_if_needed(estimated_tokens, log=log)
        else:
            skip_next_proactive_check = False

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
            raise Exception(f"Groq API Key Invalid (401): {e}") from e

        except (groq.RateLimitError, groq.BadRequestError) as e:
            detail = _parse_rate_limit_detail(str(e))
            if detail and detail["limit_type"] == "TPD":
                raise Exception(f"Groq Daily Quota Exhausted: {detail['used']}/{detail['limit']}") from e

            if detail:
                budget.record_actual_from_error(detail["used"], detail["limit"])
                log(f"Rate limit hit (attempt {attempt}). Waiting {detail['wait_seconds'] + 0.5:.1f}s...")
                time.sleep(detail["wait_seconds"] + 0.5)
                budget.reset_window()
                skip_next_proactive_check = True
            else:
                time.sleep(5.0 * attempt)
        except Exception as e:
            log(f"LLM call attempt {attempt} error: {e}")
            time.sleep(2)

    raise Exception(f"Groq API call failed after {max_retries + 1} attempts.")


# =========================================================
# QUESTION EXTRACTION (STRICT NO-HALLUCINATION)
# =========================================================

QP_SYSTEM_PROMPT = """You are an expert exam paper structure analyzer.
Examine the OCR text provided and identify:
1. "question_paper_pages": Array of page numbers that are purely official Question Paper pages.
2. "questions": Array of extracted questions exactly as printed.

CRITICAL INSTRUCTIONS TO PREVENT HALLUCINATION:
- DO NOT invent questions that do not exist in text.
- ONLY output pages that contain the official question list. Ignore student answers.
- Return ONLY valid JSON in this exact shape:
{
  "question_paper_pages": [1, 2],
  "questions": ["1. Question text here...", "2. Second question..."]
}"""


def _parse_qp_llm_response(content: str) -> tuple:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content).strip()

    data = json.loads(content)
    qp_pages = [int(x) for x in data.get("question_paper_pages", [])]
    questions = [str(x).strip() for x in data.get("questions", []) if str(x).strip()]
    return qp_pages, questions


QUESTION_PAPER_ONLY_SYSTEM_PROMPT = """You are reading official printed Question Paper pages.
Extract EVERY question/sub-part exactly as written.

Rules:
- Split sub-parts like (i), (ii), (a), (b) into distinct canonical entries if they require separate answers.
- Do NOT paraphrase, summarize, or alter the question text.
- Maintain original sequence.

Return valid JSON:
{
  "questions": ["1.(i) Question text...", "1.(ii) Next subpart..."]
}"""


def _parse_canonical_questions_response(content: str) -> list:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content).strip()
    data = json.loads(content)
    return [str(q).strip() for q in data.get("questions", []) if str(q).strip()]


def extract_canonical_questions(qp_pages: list, status_callback=None) -> list:
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    if not qp_pages:
        return []

    from groq import Groq
    api_key = get_api_key("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    budget = _TokenBudgetTracker()

    blocks = [f"--- PAGE {p['page_number']} ---\n{p['raw_text']}" for p in qp_pages]
    user_prompt = "Official Question Paper Text:\n\n" + "\n\n".join(blocks)

    try:
        questions = _call_groq_with_retries(
            client, QUESTION_PAPER_ONLY_SYSTEM_PROMPT, user_prompt,
            _parse_canonical_questions_response, budget, log
        )
    except Exception as e:
        log(f"Canonical extraction failed: {e}")
        return []

    return questions


def identify_questions_with_llm(pages: list, status_callback=None) -> tuple:
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    from groq import Groq
    api_key = get_api_key("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    budget = _TokenBudgetTracker()

    # Simple page chunks
    chunks = []
    curr, curr_len = [], 0
    for p in pages:
        p_len = len(p["raw_text"])
        if curr and curr_len + p_len > 5000:
            chunks.append(curr)
            curr, curr_len = [], 0
        curr.append(p)
        curr_len += p_len
    if curr:
        chunks.append(curr)

    valid_pages = {p["page_number"] for p in pages}
    qp_detected_pages = set()

    for i, chunk in enumerate(chunks):
        blocks = [f"--- PAGE {p['page_number']} ---\n{p['raw_text']}" for p in chunk]
        user_prompt = "Analyze these pages:\n\n" + "\n\n".join(blocks)
        try:
            detected, _ = _call_groq_with_retries(
                client, QP_SYSTEM_PROMPT, user_prompt, _parse_qp_llm_response, budget, log
            )
            qp_detected_pages.update([p for p in detected if p in valid_pages])
        except Exception as e:
            log(f"QP detection chunk {i+1} failed: {e}")

    qp_indices_0based = sorted([p - 1 for p in qp_detected_pages])
    qp_pages_full = [pages[i] for i in qp_indices_0based]

    questions = extract_canonical_questions(qp_pages_full, status_callback)
    return qp_indices_0based, questions


# =========================================================
# PERFECT ANSWER MAPPING & ZERO HALLUCINATION SLICING
# =========================================================

ANSWER_MAP_SYSTEM_PROMPT = """You are an accurate exam evaluator.
Given:
1. List of OFFICIAL QUESTIONS identified by tags like [REF-A], [REF-B], etc.
2. Line-numbered OCR text of student's answer sheet.

Task: Identify exact [line_number] ranges where student wrote the answer for each question.

STRICT HALLUCINATION & ACCURACY RULES:
- Only map line numbers that ACTUALLY EXIST in the provided line-numbered text.
- Start line MUST be where student begins answering that specific question (e.g. "Ans 1", "उत्तर 1", or main topic restatement).
- End line MUST be where that answer finishes before the next question or document ends.
- Answers appear in monotonic sequential order. Do NOT overlap or re-order line ranges.
- Omit [REF-X] if no answer for that question exists in the text.

Return strictly JSON:
{
  "answers": [
    {"ref": "REF-A", "start_line": 10, "end_line": 45},
    {"ref": "REF-B", "start_line": 46, "end_line": 90}
  ]
}"""


def _parse_answer_map_llm_response(content: str) -> list:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content).strip()

    data = json.loads(content)
    answers = data.get("answers", [])
    result = []
    for item in answers:
        if isinstance(item, dict) and "ref" in item and "start_line" in item and "end_line" in item:
            try:
                result.append({
                    "ref": str(item["ref"]).strip().upper(),
                    "start_line": int(item["start_line"]),
                    "end_line": int(item["end_line"])
                })
            except (ValueError, TypeError):
                continue
    return result


def _enforce_monotonic_boundaries(ranges: list) -> list:
    """Fixes hallucinated line numbers by enforcing strict non-overlapping sequential boundaries."""
    if not ranges:
        return []

    # Sort ranges by ref order (REF-A, REF-B...)
    sorted_ranges = sorted(ranges, key=lambda x: x["ref"])
    cleaned = []

    last_line = -1
    for r in sorted_ranges:
        start = max(r["start_line"], last_line + 1)
        end = max(r["end_line"], start)

        if start <= end:
            cleaned.append({
                "ref": r["ref"],
                "start_line": start,
                "end_line": end
            })
            last_line = end

    return cleaned


def strip_question_restatement(answer_text: str) -> str:
    pattern = re.compile(
        r'^\s*(?:Ans(?:wer)?\s*\d*\s*[.:\-]?|उत्तर\s*\d*\s*[\-\:]?|प्र[०.\s]*\d*|Q\.?\s*\d+)\s*',
        re.IGNORECASE
    )
    return pattern.sub('', answer_text).strip()


def map_answers_with_llm(answer_lines: list, questions: list, status_callback=None) -> dict:
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    from groq import Groq
    api_key = get_api_key("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    budget = _TokenBudgetTracker()

    ref_to_question = {f"REF-{chr(65+i)}": q for i, q in enumerate(questions)}
    numbered_lines = list(enumerate(answer_lines))

    # Chunk lines with generous overlap to preserve continuous context
    chunk_size = 150
    overlap = 20
    chunks = []
    i = 0
    while i < len(numbered_lines):
        chunks.append(numbered_lines[i: i + chunk_size])
        if i + chunk_size >= len(numbered_lines):
            break
        i += (chunk_size - overlap)

    raw_ranges = []
    for c_idx, chunk in enumerate(chunks):
        q_block = "\n".join([f"[{ref}] {q}" for ref, q in ref_to_question.items()])
        l_block = "\n".join([f"[{idx}] {text}" for idx, text in chunk])
        user_prompt = f"QUESTIONS:\n{q_block}\n\nSTUDENT ANSWER LINES:\n{l_block}"

        try:
            chunk_res = _call_groq_with_retries(
                client, ANSWER_MAP_SYSTEM_PROMPT, user_prompt,
                _parse_answer_map_llm_response, budget, log
            )
            raw_ranges.extend(chunk_res)
        except Exception as e:
            log(f"Answer map chunk {c_idx+1} error: {e}")

    # Deduplicate ranges across chunks (keep longest range for each REF)
    ref_best = {}
    for r in raw_ranges:
        ref = r["ref"]
        span = r["end_line"] - r["start_line"]
        if ref not in ref_best or span > (ref_best[ref]["end_line"] - ref_best[ref]["start_line"]):
            ref_best[ref] = r

    # Post-process: Enforce non-hallucinated sequential monotonicity
    resolved_ranges = _enforce_monotonic_boundaries(list(ref_best.values()))

    # Verbatim line slicing
    qa_map = {}
    for r in resolved_ranges:
        q_text = ref_to_question.get(r["ref"])
        if not q_text:
            continue
        
        start, end = r["start_line"], r["end_line"]
        lines = [answer_lines[idx] for idx in range(start, min(end + 1, len(answer_lines)))]
        full_ans = " ".join(lines).strip()
        cleaned_ans = strip_question_restatement(full_ans)
        qa_map[q_text] = cleaned_ans

    return qa_map


def is_noise(line: str) -> bool:
    noise_re = re.compile(
        r'(?:Teacher\'?s?\s*Signature|PAGE\s*NO|DATE\b|Neel?\s*Kamal|TAKMA\s*SINAN|^\s*\d{1,3}\s*$)',
        re.IGNORECASE
    )
    return bool(noise_re.search(line))


# =========================================================
# MAIN PIPELINE
# =========================================================

@_diagnose_tuple_errors
def process_pdf(file_input, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    file_bytes, file_name = _normalize_file_input(file_input, default_name="document.pdf")
    pages = run_ocr(file_bytes, file_name, status_callback)
    ocr_json = build_ocr_json(pages)

    qp_indices, official_questions = identify_questions_with_llm(pages, status_callback)

    if not official_questions:
        raise Exception("Could not detect any official questions from question paper pages.")

    answer_page_indices = [i for i in range(len(pages)) if i not in qp_indices]
    answer_pages = [pages[i] for i in answer_page_indices]

    answer_lines = []
    for page in answer_pages:
        for line in page["raw_text"].split("\n"):
            if line.strip() and not is_noise(line):
                answer_lines.append(line.strip())

    if not answer_lines:
        raise Exception("No readable student answer text found in the document.")

    qa_map = map_answers_with_llm(answer_lines, official_questions, status_callback)

    qa_pairs = []
    for q in official_questions:
        ans = qa_map.get(q, "")
        qa_pairs.append({
            "question": q,
            "answer": ans,
            "matched": bool(ans)
        })

    log(f"Pipeline Finished: {len(qa_pairs)} questions processed successfully.")
    return ocr_json, qa_pairs


def save_outputs(ocr_json: dict, qa_pairs: list, output_dir: str = ".", base_name: str = "document") -> tuple:
    ocr_path = os.path.join(output_dir, f"{base_name}_ocr.json")
    qa_path = os.path.join(output_dir, f"{base_name}_qa_pairs.json")

    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(ocr_json, f, ensure_ascii=False, indent=2)

    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    return ocr_path, qa_path
