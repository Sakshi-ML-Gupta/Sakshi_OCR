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
# PATTERNS - DEFINED EARLY SO THEY'RE AVAILABLE
# =========================================================

# Patterns that indicate a NEW answer is starting
ANSWER_START_PATTERNS = [
    re.compile(r'^Q\.?\s*(\d+)\s*[\.\)\s:-]+', re.IGNORECASE),
    re.compile(r'^Question\s*(\d+)\s*[\.\)\s:-]+', re.IGNORECASE),
    re.compile(r'^Ans(?:wer)?\s*(\d*)\s*[\.\)\s:-]+', re.IGNORECASE),
    re.compile(r'^प्र\.?\s*(\d+)\s*[\.\)\s:-]+', re.IGNORECASE),
    re.compile(r'^प्रश्न\s*(\d+)\s*[\.\)\s:-]+', re.IGNORECASE),
    re.compile(r'^उत्तर\s*(\d*)\s*[\.\)\s:-]+', re.IGNORECASE),
    re.compile(r'^(\d+)[\.\)]\s+[A-Za-z\u0900-\u097F]{10,}', re.IGNORECASE),
]

# Patterns that indicate global assignment conclusions
GLOBAL_CONCLUSION_PATTERNS = [
    re.compile(r'^(?:निष्कर्ष|Conclusion|सारांश|Summary)\s*[:：]?\s*$', re.IGNORECASE),
    re.compile(r'^=+\s*(?:निष्कर्ष|Conclusion|सारांश|Summary)\s*=+\s*$', re.IGNORECASE),
    re.compile(r'^##\s*(?:निष्कर्ष|Conclusion|सारांश|Summary)\s*$', re.IGNORECASE),
]

QUESTION_MARKER_PATTERNS = [
    re.compile(r'\nQ\.\s*(\d+)\)\s*[^\n]+', re.IGNORECASE),
    re.compile(r'\nQ\.\s*(\d+)[\.\)]\s*[^\n]+', re.IGNORECASE),
    re.compile(r'\nप्र\.\s*(\d+)\)\s*[^\n]+', re.IGNORECASE),
    re.compile(r'\nप्रश्न\s*(\d+)[\.\)]\s*[^\n]+', re.IGNORECASE),
    re.compile(r'\n(\d+)\)\s*[^\n]+\?', re.IGNORECASE),
]

# =========================================================
# INPUT NORMALIZATION
# =========================================================

def _normalize_file_input(file_input, default_name="document.pdf"):
    if isinstance(file_input, tuple):
        if len(file_input) < 2:
            raise ValueError(f"Tuple must have (filename, bytes), got {len(file_input)} items")
        name, data = file_input[0], file_input[1]
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"Expected bytes, got {type(data).__name__}")
        return bytes(data), _coerce_name(name, default_name)
    if isinstance(file_input, (bytes, bytearray)):
        return bytes(file_input), default_name
    if isinstance(file_input, (str, Path)):
        p = Path(file_input)
        return p.read_bytes(), p.name
    if hasattr(file_input, "read"):
        data = file_input.read()
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"file_input.read() returned {type(data).__name__}, expected bytes")
        name = getattr(file_input, "name", default_name)
        return bytes(data), _coerce_name(name, default_name)
    raise TypeError(f"Unsupported file_input type: {type(file_input).__name__}")

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
                raise TypeError(f"[DIAGNOSTIC] file_input: type={type(file_input).__name__}, repr={file_input!r}. Original: {e}") from e
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
# OCR -- Datalab
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
    log(f"WARNING: No page-break marker recognized. Treating as single page.")
    return [markdown.strip()]

def run_ocr(file_content: bytes, file_name: str, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    file_name = _coerce_name(file_name, default_name="document.pdf")
    if not isinstance(file_content, (bytes, bytearray)):
        raise TypeError(f"run_ocr() expected bytes, got {type(file_content).__name__}")
    api_key = get_api_key("DATALAB_API_KEY")
    if not api_key:
        raise Exception("DATALAB_API_KEY not found")
    size_mb = len(file_content) / (1024 * 1024)
    MAX_MB = 45
    if size_mb > MAX_MB:
        raise Exception(f"File is {size_mb:.1f}MB, exceeds {MAX_MB}MB limit")
    headers = {"X-API-Key": api_key}
    log(f"Submitting document to Datalab... ({size_mb:.1f}MB)")
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
    log("Polling for OCR result...")
    max_polls = 150
    poll_interval = 2
    result = None
    for attempt in range(max_polls):
        poll_resp = httpx.get(check_url, headers=headers, timeout=60)
        if poll_resp.status_code != 200:
            raise Exception(f"Datalab poll error {poll_resp.status_code}")
        result = poll_resp.json()
        status = result.get("status")
        if status == "complete":
            log("OCR complete")
            break
        if status == "failed" or result.get("error"):
            raise Exception(f"Datalab conversion failed: {result.get('error')}")
        if attempt % 5 == 0:
            log(f"Still processing... ({attempt * poll_interval}s elapsed)")
        time.sleep(poll_interval)
    else:
        raise Exception("Datalab conversion timed out")
    if not result.get("success", True):
        raise Exception(f"Datalab conversion error: {result.get('error')}")
    markdown = result.get("markdown") or ""
    if not markdown.strip():
        raise Exception("Datalab returned empty markdown")
    page_count_hint = result.get("page_count")
    page_texts = _split_paginated_markdown(markdown, page_count_hint, log=log)
    pages = []
    for idx, text in enumerate(page_texts):
        pages.append({"page_number": idx + 1, "raw_text": text})
    log(f"OCR done -- {len(pages)} page(s)")
    return pages

def build_ocr_json(pages: list) -> dict:
    return {"total_pages": len(pages), "pages": [{"page_number": p["page_number"], "text": p["raw_text"]} for p in pages]}

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
# LLM-BASED QUESTION PAPER DETECTION
# =========================================================

GROQ_MODEL = "openai/gpt-oss-120b"
TPM_LIMIT = 8000
TPM_SAFETY_FRACTION = 0.85
CHARS_PER_TOKEN_ESTIMATE = 2.0
MAX_CHARS_PER_CHUNK = 6000
CHUNK_OVERLAP_PAGES = 1

QP_SYSTEM_PROMPT = """You are analyzing OCR text extracted from a scanned student exam/assignment answer booklet.

Your task: read the pages shown and return ONLY valid JSON (no markdown fences, no commentary) in EXACTLY this shape:

{
  "question_paper_pages": [14, 16, 18],
  "admin_pages": [1, 2],
  "questions": ["1. Example question text. (10)", "2. Another example question. (10)"]
}

Critical rules:
- A genuine question paper page contains questions directed at the student ("explain", "discuss", "describe", etc.)
- Answer pages contain the student's responses with "Ans" or "उत्तर" markers
- Admin pages contain roll numbers, letterheads, etc.
- Preserve the EXACT original text of questions.

Output ONLY the JSON object described above."""

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
        log(f"Pacing requests: waiting {wait_s:.1f}s")
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
    return {"limit_type": limit_type.upper(), "period": period.lower(), "limit": int(limit), 
            "used": int(used), "requested": int(requested), "wait_seconds": wait_seconds}

def _parse_qp_llm_response(content: str) -> tuple:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}")
    if not isinstance(data, dict):
        raise ValueError(f"LLM response must be a dict, got {type(data).__name__}")
    if "question_paper_pages" not in data or "questions" not in data:
        raise ValueError(f"Missing required keys. Got: {list(data.keys())}")
    qp_pages = [int(x) for x in data["question_paper_pages"]] if isinstance(data["question_paper_pages"], list) else []
    admin_pages = [int(x) for x in data.get("admin_pages", [])] if isinstance(data.get("admin_pages"), list) else []
    questions = [str(x).strip() for x in data["questions"] if str(x).strip()] if isinstance(data["questions"], list) else []
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
            raise Exception(f"Groq API rejected the API key (401): {e}") from e
        except (groq.RateLimitError, groq.BadRequestError) as e:
            last_error = e
            detail = _parse_rate_limit_detail(str(e))
            if detail and detail["limit_type"] == "TPD":
                raise Exception(f"Daily token quota exhausted: {detail['used']}/{detail['limit']} tokens used") from e
            if detail:
                budget.record_actual_from_error(detail["used"], detail["limit"])
                log(f"Rate limit hit (attempt {attempt}): waiting {detail['wait_seconds'] + 0.5:.1f}s")
                time.sleep(detail["wait_seconds"] + 0.5)
                budget.reset_window()
                skip_next_proactive_check = True
            else:
                wait_s = 5.0 * attempt
                log(f"Rate limit hit (attempt {attempt}): waiting {wait_s:.1f}s")
                time.sleep(wait_s)
        except Exception as e:
            last_error = e
            log(f"Attempt {attempt} failed: {e}")
            time.sleep(1)
    raise Exception(f"Failed after {max_retries + 1} attempts. Last error: {last_error}")

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

# =========================================================
# QUESTION EXTRACTION - FALLBACK STRATEGIES
# =========================================================

def is_question_paper_page(text: str) -> bool:
    """Check if a page looks like a question paper page."""
    if not text or len(text) < 100:
        return False
    numbered_items = re.findall(r'\n\s*(\d+)[\.\)]\s*[A-Za-z\u0900-\u097F]', text)
    if len(numbered_items) >= 3:
        return True
    indicators = [
        r'(?:SECTION|PART)\s*[A-Z]',
        r'(?:Marks|अंक)\s*:',
        r'Answer\s+(?:any|all)\s+\d+',
        r'(?:Questions|प्रश्न)\s*:',
        r'Time\s*:',
        r'Max\.?\s*Marks',
        r'समय\s*:',
        r'पूर्णांक\s*:',
    ]
    for pattern in indicators:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def extract_questions_from_page(page_text: str) -> List[str]:
    """Extract questions from a single page."""
    questions = []
    seen = set()
    lines = page_text.split('\n')
    current_question = []
    current_num = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(r'^(\d+)[\.\)]\s*(.*)', line)
        if match:
            if current_question and current_num is not None:
                q_text = " ".join(current_question).strip()
                if len(q_text) > 20:
                    key = q_text[:50]
                    if key not in seen:
                        seen.add(key)
                        questions.append(f"{current_num}. {q_text}")
            current_num = match.group(1)
            rest = match.group(2).strip()
            current_question = [rest] if rest else []
        else:
            if current_num is not None:
                current_question.append(line)
    if current_question and current_num is not None:
        q_text = " ".join(current_question).strip()
        if len(q_text) > 20:
            key = q_text[:50]
            if key not in seen:
                seen.add(key)
                questions.append(f"{current_num}. {q_text}")
    return questions

def extract_questions_from_pages_fixed(pages: List[Dict], qp_page_indices: List[int]) -> List[str]:
    """Extract questions from pages using pattern matching."""
    questions = []
    seen = set()
    for page_idx in qp_page_indices:
        if page_idx >= len(pages):
            continue
        page_text = pages[page_idx].get("raw_text", "")
        if not page_text.strip():
            continue
        # Method 1: Find numbered questions
        lines = page_text.split('\n')
        current_question = []
        current_num = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'^(\d+)[\.\)]\s*(.*)', line)
            if match:
                if current_question and current_num is not None:
                    q_text = " ".join(current_question).strip()
                    if len(q_text) > 20:
                        key = q_text[:50]
                        if key not in seen:
                            seen.add(key)
                            questions.append(f"{current_num}. {q_text}")
                current_num = match.group(1)
                rest = match.group(2).strip()
                current_question = [rest] if rest else []
            else:
                if current_num is not None:
                    current_question.append(line)
        if current_question and current_num is not None:
            q_text = " ".join(current_question).strip()
            if len(q_text) > 20:
                key = q_text[:50]
                if key not in seen:
                    seen.add(key)
                    questions.append(f"{current_num}. {q_text}")
        # Method 2: Find questions with marks
        marks_pattern = r'(\d+)[\.\)]\s*([^\n]+?)\s*\((\d+)\)'
        matches = re.findall(marks_pattern, page_text)
        for num, q_text, marks in matches:
            q_text = q_text.strip()
            if len(q_text) > 20:
                key = q_text[:50]
                if key not in seen:
                    seen.add(key)
                    questions.append(f"{num}. {q_text} ({marks})")
    # Remove duplicates
    unique_questions = []
    seen = set()
    for q in questions:
        q_clean = q.strip()
        key = q_clean[:50]
        if q_clean and key not in seen:
            seen.add(key)
            unique_questions.append(q_clean)
    return unique_questions

def extract_question_from_text(text: str) -> str:
    """Extract a question-like sentence from text."""
    if not text:
        return ""
    sentences = re.split(r'[.!?]+\s+', text)
    for sent in sentences:
        sent = sent.strip()
        if '?' in sent and len(sent) > 20:
            return sent
        elif any(word in sent.lower() for word in ['discuss', 'explain', 'describe', 'compare', 'analyse', 'analyze']):
            if len(sent) > 30:
                return sent
        elif any(word in sent for word in ['कीजिए', 'समझाइए', 'विचार', 'बताइए', 'लिखिए']):
            if len(sent) > 30:
                return sent
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 30:
            return sent[:100]
    return text[:100]

# =========================================================
# FIXED: identify_questions_with_llm
# =========================================================

def identify_questions_with_llm(pages: list, status_callback=None) -> tuple:
    """Main function for identifying questions - with multiple fallbacks."""
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    
    from groq import Groq
    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        log("WARNING: GROQ_API_KEY not found. Using pattern-based extraction only.")
        return identify_questions_with_patterns(pages, log)
    
    try:
        client = Groq(api_key=api_key)
        budget = _TokenBudgetTracker()
        chunks = _chunk_pages_by_char_budget(pages)
        log(f"Split {len(pages)} page(s) into {len(chunks)} LLM chunk(s)")
        valid_page_numbers = {p["page_number"] for p in pages}
        max_page_number = max(valid_page_numbers) if valid_page_numbers else 0
        chunk_results = []
        chunk_failures = []
        for i, chunk in enumerate(chunks):
            page_nums = [p["page_number"] for p in chunk]
            log(f"Processing chunk {i+1}/{len(chunks)} (pages {page_nums})...")
            try:
                qp_pages_1based, questions, admin_pages_1based = _call_groq_for_chunk(client, chunk, budget, log)
            except Exception as e:
                log(f"WARNING: chunk {i+1} failed: {e}")
                chunk_failures.append(str(e))
                continue
            # Recover page numbers
            def _recover_pages(pages_list, label):
                recovered = []
                invalid = []
                for pn in pages_list:
                    if pn in valid_page_numbers:
                        recovered.append(pn)
                    else:
                        split_result = _try_split_concatenated_page_number(pn, valid_page_numbers, max_page_number)
                        if split_result:
                            log(f"Recovered concatenated {label}: {pn} -> {split_result}")
                            recovered.extend(split_result)
                        else:
                            invalid.append(pn)
                if invalid:
                    log(f"WARNING: Invalid {label} pages: {invalid}")
                return sorted(set(recovered))
            qp_pages_1based = _recover_pages(qp_pages_1based, "question-paper")
            admin_pages_1based = _recover_pages(admin_pages_1based, "admin")
            admin_pages_1based = [p for p in admin_pages_1based if p not in qp_pages_1based]
            log(f"Chunk {i+1}: {len(qp_pages_1based)} question pages, {len(admin_pages_1based)} admin pages")
            chunk_results.append((qp_pages_1based, questions, admin_pages_1based))
        if not chunk_results:
            log("All LLM chunks failed. Using pattern-based extraction.")
            return identify_questions_with_patterns(pages, log)
        qp_pages_merged, questions_merged, admin_pages_merged = _merge_chunk_results(chunk_results)
        qp_indices = sorted(pn - 1 for pn in qp_pages_merged)
        admin_indices = sorted(pn - 1 for pn in admin_pages_merged)
        log(f"LLM: {len(qp_indices)} question pages, {len(questions_merged)} questions")
        
        # If no questions extracted, use fallback
        if not questions_merged and qp_indices:
            log("No questions extracted by LLM. Using fallback.")
            questions_merged = extract_questions_from_pages_fixed(pages, qp_indices)
            log(f"Fallback found {len(questions_merged)} questions")
        
        return qp_indices, questions_merged, admin_indices
        
    except Exception as e:
        log(f"LLM approach failed: {e}. Using pattern-based extraction.")
        return identify_questions_with_patterns(pages, log)

def identify_questions_with_patterns(pages: list, log=print) -> tuple:
    """Pattern-based fallback for question identification."""
    qp_pages = []
    admin_pages = []
    all_questions = []
    for i, page in enumerate(pages):
        page_text = page.get("raw_text", "")
        if not page_text:
            continue
        if is_question_paper_page(page_text):
            qp_pages.append(i)
            questions = extract_questions_from_page(page_text)
            if questions:
                all_questions.extend(questions)
    # If we found question pages but no questions, try harder
    if qp_pages and not all_questions:
        log("Question pages found but no questions extracted. Attempting aggressive extraction...")
        for page_idx in qp_pages:
            page_text = pages[page_idx].get("raw_text", "")
            if page_text:
                lines = page_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if re.match(r'^\d+[\.\)]\s*', line) and len(line) > 20:
                        all_questions.append(line)
    # If we have no question pages but we have questions
    if not qp_pages and all_questions:
        log("Questions found but no explicit question pages. Marking first pages as question pages...")
        for i, page in enumerate(pages):
            page_text = page.get("raw_text", "")
            if page_text:
                for q in all_questions[:3]:
                    if q[:30] in page_text:
                        qp_pages.append(i)
                        break
        qp_pages = sorted(set(qp_pages))
    # Deduplicate
    unique_questions = []
    seen = set()
    for q in all_questions:
        q_clean = q.strip()
        key = q_clean[:50]
        if q_clean and key not in seen:
            seen.add(key)
            unique_questions.append(q_clean)
    log(f"Pattern extraction: {len(qp_pages)} question pages, {len(unique_questions)} questions")
    return qp_pages, unique_questions, admin_pages

# =========================================================
# ANSWER BOUNDARY DETECTION - PATTERN BASED
# =========================================================

def is_question_marker_only(line: str) -> bool:
    """Check if a line is JUST a question marker with no real content."""
    for pattern in ANSWER_START_PATTERNS:
        if pattern.search(line):
            cleaned = re.sub(r'^[A-Za-z\u0900-\u097F\s\.]*\d+[\.\)\s:-]+', '', line)
            if len(cleaned.strip()) < 15:
                return True
    return False

def is_line_start_of_answer(line: str, question_text: str) -> bool:
    """Check if a line is the start of an answer to the given question."""
    q_num_match = re.match(r'^\s*(\d+)[\.\)]', question_text)
    if not q_num_match:
        return False
    q_num = int(q_num_match.group(1))
    if re.search(rf'\b{q_num}\b', line):
        return True
    q_words = set(re.findall(r'[a-zA-Z\u0900-\u097F]{4,}', question_text.lower()))
    line_words = set(re.findall(r'[a-zA-Z\u0900-\u097F]{4,}', line.lower()))
    if q_words and line_words:
        overlap = len(q_words & line_words) / len(q_words)
        return overlap > 0.3
    return False

def detect_answer_boundaries(lines: List[str], questions: List[str]) -> Dict[int, Dict]:
    """Detect where each answer starts and ends using pattern matching."""
    boundaries = {}
    current_q_idx = None
    current_start = None
    question_numbers = {}
    for i, q in enumerate(questions):
        match = re.match(r'^\s*(\d+)[\.\)]', q)
        if match:
            question_numbers[int(match.group(1))] = i
    for line_idx, line in enumerate(lines):
        if not line.strip():
            continue
        # Check if this is a global conclusion
        is_global_conclusion = False
        for pattern in GLOBAL_CONCLUSION_PATTERNS:
            if pattern.search(line):
                is_global_conclusion = True
                break
        if is_global_conclusion:
            if current_q_idx is not None and current_start is not None:
                boundaries[current_q_idx] = {'start': current_start, 'end': line_idx - 1}
            current_q_idx = None
            current_start = None
            continue
        # Check if this line starts a new answer
        is_new_answer = False
        matched_q_idx = None
        for pattern in ANSWER_START_PATTERNS:
            match = pattern.search(line)
            if match:
                num_str = match.group(1) if match.groups() else None
                if num_str and num_str.strip():
                    try:
                        q_num = int(num_str.strip())
                        if q_num in question_numbers:
                            matched_q_idx = question_numbers[q_num]
                            is_new_answer = True
                            break
                    except ValueError:
                        pass
        if is_new_answer and matched_q_idx is not None:
            if current_q_idx is not None and current_start is not None:
                if current_start < line_idx:
                    boundaries[current_q_idx] = {'start': current_start, 'end': line_idx - 1}
            current_q_idx = matched_q_idx
            current_start = line_idx
    if current_q_idx is not None and current_start is not None:
        boundaries[current_q_idx] = {'start': current_start, 'end': len(lines) - 1}
    return boundaries

def find_answer_text(lines: List[str], start_idx: int, end_idx: int) -> str:
    """Extract and clean answer text from lines."""
    answer_lines = []
    for i in range(start_idx, min(end_idx + 1, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
        if is_question_marker_only(line):
            continue
        answer_lines.append(line)
    return " ".join(answer_lines).strip()

def remove_global_conclusion(text: str) -> str:
    """Remove global assignment conclusion if present."""
    for pattern in GLOBAL_CONCLUSION_PATTERNS:
        if pattern.search(text):
            match = pattern.search(text)
            if match:
                text = text[:match.start()].strip()
                break
    return text

def remove_next_question(text: str) -> str:
    """Remove any next question that appears at the end of an answer."""
    patterns = [
        (r'Q\.?\s*\d+[\.\)]\s*[A-Za-z\u0900-\u097F]{3,}', 'english'),
        (r'प्र\.?\s*\d+[\.\)]\s*[A-Za-z\u0900-\u097F]{3,}', 'hindi'),
        (r'प्रश्न\s*\d+[\.\)]\s*[A-Za-z\u0900-\u097F]{3,}', 'hindi'),
    ]
    for pattern, _ in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            last_match = matches[-1]
            if last_match.start() > len(text) * 0.7:
                matched_text = last_match.group(0)
                if re.match(r'^(?:Q\.|प्र\.|प्रश्न)\s*\d+', matched_text, re.IGNORECASE):
                    text = text[:last_match.start()].strip()
                    break
    return text

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

def strip_full_question_echo(text: str, question: str) -> str:
    """Remove the full question echo if present at the start."""
    q_core = re.sub(r'^\s*\d+[\.\)]\s*', '', question)
    q_words = set(re.findall(r'[a-zA-Z\u0900-\u097F]{4,}', q_core.lower()))
    words = text.split()
    if len(words) < 5:
        return text
    for i in range(3, min(10, len(words))):
        prefix = " ".join(words[:i])
        prefix_words = set(re.findall(r'[a-zA-Z\u0900-\u097F]{4,}', prefix.lower()))
        if q_words and prefix_words:
            overlap = len(q_words & prefix_words) / len(q_words)
            if overlap > 0.5:
                return " ".join(words[i:]).strip()
    return text

def extract_answers_with_boundaries(answer_lines: List[str], questions: List[str], 
                                    answer_line_pages: List[int] = None) -> List[Dict]:
    """Extract answers using pattern-based boundary detection."""
    results = []
    boundaries = detect_answer_boundaries(answer_lines, questions)
    for i, q in enumerate(questions):
        ref = f"REF-{chr(65 + i)}"
        if i in boundaries:
            bounds = boundaries[i]
            start = bounds['start']
            end = bounds['end']
            answer_raw = find_answer_text(answer_lines, start, end)
            answer_raw = remove_global_conclusion(answer_raw)
            answer_raw = remove_next_question(answer_raw)
            answer_clean = strip_question_restatement(answer_raw)
            answer_clean = strip_full_question_echo(answer_clean, q)
            start_page = answer_line_pages[start] if answer_line_pages and 0 <= start < len(answer_line_pages) else None
            end_page = answer_line_pages[end] if answer_line_pages and 0 <= end < len(answer_line_pages) else None
            results.append({
                "ref": ref, "question": q, "matched": True,
                "start_line": start, "end_line": end,
                "start_page": start_page, "end_page": end_page,
                "answer": answer_clean, "answer_raw": answer_raw,
            })
        else:
            results.append({
                "ref": ref, "question": q, "matched": False,
                "start_line": None, "end_line": None,
                "start_page": None, "end_page": None,
                "answer": "", "answer_raw": "",
            })
    return results

# =========================================================
# FALLBACK ANSWER MAPPING
# =========================================================

def aggressive_answer_mapping(answer_lines: List[str], questions: List[str], 
                              answer_line_pages: List[int] = None) -> List[Dict]:
    """Aggressive fallback: try to map answers using any available method."""
    results = []
    if not answer_lines:
        for i, q in enumerate(questions):
            ref = f"REF-{chr(65 + i)}"
            results.append({"ref": ref, "question": q, "matched": False,
                "start_line": None, "end_line": None, "start_page": None, "end_page": None,
                "answer": "", "answer_raw": ""})
        return results
    # If we have few questions, split the answer lines evenly
    if len(questions) <= 3 and len(answer_lines) > 10:
        lines_per_question = len(answer_lines) // len(questions)
        for i, q in enumerate(questions):
            ref = f"REF-{chr(65 + i)}"
            start = i * lines_per_question
            end = (i + 1) * lines_per_question - 1 if i < len(questions) - 1 else len(answer_lines) - 1
            if start < len(answer_lines):
                answer_raw = " ".join(answer_lines[start:end+1]).strip()
                answer_clean = strip_question_restatement(answer_raw)
                results.append({"ref": ref, "question": q, "matched": True,
                    "start_line": start, "end_line": end,
                    "start_page": answer_line_pages[start] if answer_line_pages else None,
                    "end_page": answer_line_pages[end] if answer_line_pages else None,
                    "answer": answer_clean, "answer_raw": answer_raw})
            else:
                results.append({"ref": ref, "question": q, "matched": False,
                    "start_line": None, "end_line": None, "start_page": None, "end_page": None,
                    "answer": "", "answer_raw": ""})
        return results
    # Look for answer indicators
    answer_indices = []
    for i, line in enumerate(answer_lines):
        if re.search(r'(?:Ans|Answer|उत्तर|प्र)', line, re.IGNORECASE):
            answer_indices.append(i)
    if answer_indices:
        for i, q in enumerate(questions):
            ref = f"REF-{chr(65 + i)}"
            if i < len(answer_indices):
                start = answer_indices[i]
                end = answer_indices[i+1] - 1 if i+1 < len(answer_indices) else len(answer_lines) - 1
                answer_raw = " ".join(answer_lines[start:end+1]).strip()
                answer_clean = strip_question_restatement(answer_raw)
                results.append({"ref": ref, "question": q, "matched": True,
                    "start_line": start, "end_line": end,
                    "start_page": answer_line_pages[start] if answer_line_pages else None,
                    "end_page": answer_line_pages[end] if answer_line_pages else None,
                    "answer": answer_clean, "answer_raw": answer_raw})
            else:
                results.append({"ref": ref, "question": q, "matched": False,
                    "start_line": None, "end_line": None, "start_page": None, "end_page": None,
                    "answer": "", "answer_raw": ""})
        return results
    # Last resort: assign all to first question
    all_text = " ".join(answer_lines).strip()
    for i, q in enumerate(questions):
        ref = f"REF-{chr(65 + i)}"
        if i == 0:
            results.append({"ref": ref, "question": q, "matched": True,
                "start_line": 0, "end_line": len(answer_lines) - 1,
                "start_page": answer_line_pages[0] if answer_line_pages else None,
                "end_page": answer_line_pages[-1] if answer_line_pages else None,
                "answer": all_text, "answer_raw": all_text})
        else:
            results.append({"ref": ref, "question": q, "matched": False,
                "start_line": None, "end_line": None, "start_page": None, "end_page": None,
                "answer": "", "answer_raw": ""})
    return results

def map_answers_robust(answer_lines: list, questions: list, status_callback=None,
                        answer_line_pages: list = None) -> list:
    """Robust answer mapping with multiple fallbacks."""
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    # Strategy 1: Pattern-based extraction
    log("Attempting pattern-based answer extraction...")
    results = extract_answers_with_boundaries(answer_lines, questions, answer_line_pages)
    matched_count = sum(1 for r in results if r["matched"])
    if matched_count > 0:
        log(f"Pattern-based extraction matched {matched_count} questions")
        return results
    # Strategy 2: Aggressive fallback
    log("Pattern-based extraction failed - using aggressive fallback...")
    return aggressive_answer_mapping(answer_lines, questions, answer_line_pages)

# =========================================================
# NOISE DETECTION
# =========================================================

NOISE_RE = re.compile(r'(?:signature|PAGE\s*NO|^\s*DATE\b|^\s*\d{1,3}\s*$)', re.IGNORECASE)
NOISE_LINE_MAX_CHARS = 40

def is_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r'^\s*\d{1,3}\s*$', stripped):
        return True
    if len(stripped) > NOISE_LINE_MAX_CHARS:
        return False
    return bool(NOISE_RE.search(stripped))

# =========================================================
# DIAGNOSTICS
# =========================================================

def _sanity_check_answer_pages(answer_lines: list, num_questions: int, log=print) -> bool:
    total_chars = sum(len(l) for l in answer_lines)
    avg_chars_per_question = total_chars / max(num_questions, 1)
    MIN_PLAUSIBLE_CHARS_PER_QUESTION = 200
    if avg_chars_per_question < MIN_PLAUSIBLE_CHARS_PER_QUESTION:
        log(f"WARNING: Answer pages contain only {total_chars} chars for {num_questions} questions (~{avg_chars_per_question:.0f} chars/question)")
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
            log(f"WARNING: Possible truncated answer for '{p['question'][:60]}...' -- only {length} chars vs median {median_len} chars")

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

    # STEP 1: Identify question paper pages and extract questions
    qp_page_indices, official_questions, admin_page_indices = identify_questions_with_llm(pages, status_callback)

    log(f"Question paper pages detected: {[p+1 for p in qp_page_indices] if qp_page_indices else 'none'}")
    log(f"Admin/cover pages detected: {[p+1 for p in admin_page_indices] if admin_page_indices else 'none'}")
    log(f"Official questions extracted: {len(official_questions)}")

    # STEP 2: If no question pages found, try to find them differently
    if not qp_page_indices:
        log("WARNING: No question paper pages detected. Attempting to find questions in all pages...")
        for i, page in enumerate(pages):
            page_text = page.get("raw_text", "")
            if not page_text:
                continue
            if is_question_paper_page(page_text):
                qp_page_indices.append(i)
                log(f"  Found potential question page: {i+1}")
        if not qp_page_indices:
            log("WARNING: No question pages found. Assuming first 3 pages contain questions...")
            qp_page_indices = [0, 1, 2] if len(pages) >= 3 else [0]
            qp_page_indices = [i for i in qp_page_indices if i < len(pages)]

    # STEP 3: If no questions extracted, use fallback strategies
    if not official_questions and qp_page_indices:
        log("No questions were extracted from question pages. Using fallback extraction...")
        official_questions = extract_questions_from_pages_fixed(pages, qp_page_indices)
        if official_questions:
            log(f"Fallback extraction found {len(official_questions)} questions")
        else:
            log("Creating placeholder questions from question pages...")
            for i, page_idx in enumerate(qp_page_indices):
                if page_idx >= len(pages):
                    continue
                page_text = pages[page_idx].get("raw_text", "")
                preview = page_text[:200].replace('\n', ' ').strip()
                if preview:
                    question_text = extract_question_from_text(preview)
                    if question_text:
                        official_questions.append(f"{i+1}. {question_text}")
                    else:
                        official_questions.append(f"{i+1}. {preview[:100]}")
                else:
                    official_questions.append(f"{i+1}. [Question from page {page_idx+1}]")
            log(f"Created {len(official_questions)} placeholder questions")

    # STEP 4: ENSURE we have at least one question
    if not official_questions:
        log("WARNING: No questions found anywhere. Creating default question...")
        official_questions = ["1. Please answer the questions from the exam paper."]
        for page in pages:
            text = page.get("raw_text", "")
            if text and len(text) > 100:
                sentences = re.split(r'[.!?]+\s+', text)
                for sent in sentences[:5]:
                    sent = sent.strip()
                    if len(sent) > 30 and any(word in sent.lower() for word in ['what', 'why', 'how', 'discuss', 'explain', 'describe', 'compare']):
                        official_questions = [f"1. {sent}"]
                        break
                if official_questions:
                    break

    log(f"Final question list: {len(official_questions)} question(s)")

    # STEP 5: Prepare answer pages
    excluded_indices = set(qp_page_indices) | set(admin_page_indices)
    answer_page_indices = [i for i in range(len(pages)) if i not in excluded_indices]
    answer_pages = [pages[i] for i in answer_page_indices]

    log(f"Answer pages: {[i+1 for i in answer_page_indices]}")

    # STEP 6: Flatten answer text into lines
    answer_lines = []
    answer_line_pages = []
    for page in answer_pages:
        for line in page["raw_text"].split("\n"):
            if not is_noise(line):
                answer_lines.append(line)
                answer_line_pages.append(page["page_number"])

    log(f"Flattened {len(answer_lines)} answer lines")

    # STEP 7: Map answers to questions
    log("Mapping each question to its answer...")
    qa_pairs = map_answers_robust(
        answer_lines, official_questions, status_callback,
        answer_line_pages=answer_line_pages
    )

    matched_count = sum(1 for p in qa_pairs if p["matched"])
    log(f"Matched {matched_count} of {len(official_questions)} questions")

    # STEP 8: Final diagnostics
    if matched_count == 0 and answer_lines:
        log("WARNING: No answers matched. Assigning all text to the first question as fallback.")
        if answer_lines and official_questions:
            all_text = " ".join(answer_lines).strip()
            if all_text:
                qa_pairs[0]["matched"] = True
                qa_pairs[0]["answer"] = all_text
                qa_pairs[0]["answer_raw"] = all_text
                qa_pairs[0]["start_line"] = 0
                qa_pairs[0]["end_line"] = len(answer_lines) - 1
                log("Assigned all text to the first question as fallback.")

    _flag_suspiciously_short_answers(qa_pairs, log)

    log(f"Done -- {len(qa_pairs)} Q-A pairs ({sum(1 for p in qa_pairs if p['matched'])} matched)")

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
