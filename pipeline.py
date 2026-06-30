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
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
import collections

# =========================================================
# API KEYS
# =========================================================

def get_api_key(name: str) -> Optional[str]:
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

def _normalize_file_input(file_input: Any, default_name: str = "document.pdf") -> Tuple[bytes, str]:
    if isinstance(file_input, tuple):
        if len(file_input) < 2:
            raise ValueError(f"Tuple must have at least (filename, bytes), got {len(file_input)} items")
        name, data = file_input[0], file_input[1]
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"Expected bytes as second tuple element, got {type(data).__name__}")
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


def _coerce_name(name: Any, default_name: str = "document.pdf") -> str:
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
                    f"[DIAGNOSTIC] Caught error in {func.__name__}(). "
                    f"file_input: type={type(file_input).__name__}, repr={file_input!r}. "
                    f"Original: {e}"
                ) from e
            raise
    return wrapper


# =========================================================
# CONCURRENCY GUARD
# =========================================================

_groq_call_lock = threading.Lock()


# =========================================================
# PREPROCESS PDF
# =========================================================

def preprocess_pdf(file_bytes: bytes, dpi: int = 250) -> bytes:
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
# TEXT CLEANING - AGGRESSIVE
# =========================================================

def clean_ocr_text(text: str) -> str:
    """Aggressive cleaning of OCR text."""
    # Remove common headers/footers
    patterns = [
        r'^\s*Page\s*\d+\s*of\s*\d+\s*$',
        r'^\s*Page\s*\d+\s*$',
        r'^\s*-\s*\d+\s*-\s*$',
        r'^\s*\[Page\s*\d+\]\s*$',
        r'^\s*P\.?\s*No\.?\s*\d+\s*$',
        r'^\s*प्र\.\s*नं\.?\s*\d+\s*$',
        r'^\s*प्रश्न\s*नं\.?\s*\d+\s*$',
        r'^\s*Q\.?\s*No\.?\s*\d+\s*$',
        r'^\s*Question\s*No\.?\s*\d+\s*$',
        r'^\s*#\s*.*$',
        r'^\s*प्रश्नोत्तर\s*.*$',
        r'^\s*prashan\s*.*$',
        r'^\s*[Pp]rashan\s*.*$',
        r'^\s*Answer\s*No\.?\s*\d+\s*$',
        r'^\s*उत्तर\s*नं\.?\s*\d+\s*$',
        r'^\s*Roll\s*No\.?\s*\d+\s*$',
        r'^\s*Enrolment\s*No\.?\s*\d+\s*$',
        r'^\s*[A-Z]{2,}\s*\d+\s*$',
        r'^\s*[A-Z]+\s*-\s*\d+\s*$',
        r'^\s*[A-Z]+\s*/\s*\d+\s*$',
    ]
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        is_noise = False
        for pattern in patterns:
            if re.search(pattern, line, re.IGNORECASE):
                is_noise = True
                break
                
        if len(line) < 3 and not any(c.isalpha() for c in line):
            is_noise = True
            
        if not is_noise:
            cleaned_lines.append(line)
    
    text = ' '.join(cleaned_lines)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()


# =========================================================
# OCR
# =========================================================

DATALAB_BASE_URL = "https://www.datalab.to"

PAGE_BREAK_PATTERNS = [
    re.compile(r'\n?\{(\d+)\}-{3,}\n?'),
    re.compile(r'\n?-{2,}\{(\d+)\}-{2,}\n?'),
    re.compile(r'\n-{3,}\s*Page\s*\d+\s*-{3,}\n', re.IGNORECASE),
    re.compile(r'\n\s*\[PAGE\s*(\d+)\]\s*\n', re.IGNORECASE),
    re.compile(r'\n\s*<!--\s*page\s*(\d+)\s*-->\s*\n', re.IGNORECASE),
    re.compile(r'\n\s*Page\s*(\d+)\s*\n', re.IGNORECASE),
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

    return [markdown.strip()]


def run_ocr(file_content: bytes, file_name: str, status_callback=None) -> List[Dict]:
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    file_name = _coerce_name(file_name, default_name="document.pdf")

    api_key = get_api_key("DATALAB_API_KEY")
    if not api_key:
        raise Exception("DATALAB_API_KEY not found")

    size_mb = len(file_content) / (1024 * 1024)
    MAX_MB = 45
    if size_mb > MAX_MB:
        raise Exception(f"File is {size_mb:.1f}MB, exceeds {MAX_MB}MB limit.")

    headers = {"X-API-Key": api_key}
    log(f"Submitting to Datalab (Chandra OCR)... ({size_mb:.1f}MB)")

    resp = httpx.post(
        f"{DATALAB_BASE_URL}/api/v1/convert",
        headers=headers,
        files={"file": (file_name, file_content, "application/pdf")},
        data={"output_format": "markdown", "mode": "accurate", "paginate": "true"},
        timeout=120
    )

    if resp.status_code != 200:
        raise Exception(f"Datalab error {resp.status_code}: {resp.text}")

    data = resp.json()
    if not data.get("success", True):
        raise Exception(f"Datalab failed: {data.get('error')}")

    check_url = data["request_check_url"]
    log("Polling for OCR result...")

    max_polls = 150
    poll_interval = 2
    result = None

    for attempt in range(max_polls):
        poll_resp = httpx.get(check_url, headers=headers, timeout=60)
        if poll_resp.status_code != 200:
            raise Exception(f"Poll error {poll_resp.status_code}: {poll_resp.text}")

        result = poll_resp.json()
        status = result.get("status")

        if status == "complete":
            log("OCR complete")
            break
        elif status == "failed" or result.get("error"):
            raise Exception(f"Conversion failed: {result.get('error')}")

        if attempt % 5 == 0:
            log(f"Still processing... ({attempt * poll_interval}s elapsed)")
        time.sleep(poll_interval)
    else:
        raise Exception("OCR timed out after 5 minutes")

    if not result.get("success", True):
        raise Exception(f"Conversion error: {result.get('error')}")

    markdown = result.get("markdown") or ""
    if not markdown.strip():
        raise Exception("Empty markdown output")

    markdown = clean_ocr_text(markdown)

    page_count_hint = result.get("page_count")
    page_texts = _split_paginated_markdown(markdown, page_count_hint, log=log)

    pages = []
    for idx, text in enumerate(page_texts):
        pages.append({
            "page_number": idx + 1,
            "raw_text": text
        })

    log(f"OCR done -- {len(pages)} page(s)")
    return pages


# =========================================================
# BUILD OCR JSON
# =========================================================

def build_ocr_json(pages: list) -> dict:
    return {
        "total_pages": len(pages),
        "pages": [{"page_number": p["page_number"], "text": p["raw_text"]} for p in pages]
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
# TOKEN BUDGET TRACKER
# =========================================================

TPM_LIMIT = 8000
TPM_SAFETY_FRACTION = 0.85
CHARS_PER_TOKEN_ESTIMATE = 2.0
MAX_CHARS_PER_CHUNK = 15000  # DRAMATICALLY INCREASED for long answers
CHUNK_OVERLAP_PAGES = 1

class _TokenBudgetTracker:
    def __init__(self, tpm_limit=TPM_LIMIT, safety_fraction=TPM_SAFETY_FRACTION):
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
        log(f"Pacing requests: waiting {wait_s:.1f}s...")
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


# =========================================================
# LLM PROMPTS
# =========================================================

QP_SYSTEM_PROMPT = """You are analyzing OCR text from a student exam assignment booklet.

Your task: Identify which pages are the official question paper pages.

RULES:
1. QUESTION PAPER pages contain official exam questions (instructions/prompts)
2. ANSWER pages contain student responses (longer text with explanations)
3. ADMIN pages contain enrolment numbers, programme codes, etc.

CRITICAL: If a page looks like it could be a question but is VERY LONG (>3x median length), 
it's likely an ANSWER page where the student restated the question.

Return ONLY valid JSON:
{
  "question_paper_pages": [14, 16, 18],
  "questions": ["1. Question text (10)", "2. Another question (10)"]
}

If no question pages found, return {"question_paper_pages": [], "questions": []}"""


QUESTION_PAPER_ONLY_SYSTEM_PROMPT = """You are extracting questions from the official question paper.

RULES:
1. Extract EVERY distinct question/sub-part
2. For multi-part questions (i), (ii), (iii), split each into separate entries
3. Preserve EXACT original text
4. Keep original numbering
5. Return in printed order

Return ONLY valid JSON:
{
  "questions": ["1.(i) First sub-part", "1.(ii) Second sub-part", ...]
}"""


# =========================================================
# CRITICAL FIX: ANSWER MAPPING - COMPLETE REWRITE
# =========================================================

ANSWER_MAP_SYSTEM_PROMPT = """You are matching answers to questions from a student's exam.

You are given:
1. Questions with REF labels (REF-A, REF-B, etc.)
2. The COMPLETE student answer text with line numbers

Your task: For EACH question, find the EXACT line range where its answer appears.

CRITICAL RULES:
1. Answers can be VERY LONG (10+ pages). DO NOT TRUNCATE!
2. Include ALL lines that belong to each answer
3. An answer starts when the student begins responding to that question
4. An answer ends when the student starts the NEXT question's answer
5. If you're unsure about a boundary, BE GENEROUS - include more lines rather than fewer
6. Each answer MUST be continuous - no gaps in the middle of an answer

Return ONLY valid JSON:
{
  "answers": [
    {"ref": "REF-A", "start_line": 0, "end_line": 45},
    {"ref": "REF-B", "start_line": 46, "end_line": 92}
  ]
}

If an answer isn't present, OMIT it from the output."""


def _line_starts_answer(line: str) -> Tuple[bool, Optional[str]]:
    """Detect if line starts a new answer and extract question number."""
    patterns = [
        (r'^\s*Ans(?:wer)?\s*[:.-]?\s*(\d+)', 'Ans'),
        (r'^\s*उत्तर\s*[:.-]?\s*(\d+)', 'उत्तर'),
        (r'^\s*प्रश्न\s*[.:]?\s*(\d+)', 'प्रश्न'),
        (r'^\s*प्र\.\s*(\d+)', 'प्र.'),
        (r'^\s*Q\.?\s*[:.-]?\s*(\d+)', 'Q'),
    ]
    
    for pattern, _ in patterns:
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            return True, match.group(1)
    
    return False, None


def map_answers_with_llm(answer_lines: list, questions: list, status_callback=None) -> dict:
    """
    Map questions to answers using the ENTIRE answer text at once.
    FIX: Process ALL answer lines in ONE chunk to avoid splitting answers.
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
        raise Exception("GROQ_API_KEY not found")

    client = Groq(api_key=api_key)
    budget = _TokenBudgetTracker()

    # Clean answer lines
    answer_lines = [clean_ocr_text(line) for line in answer_lines if line.strip()]
    
    if not answer_lines:
        log("No answer lines found")
        return {}

    log(f"Processing {len(answer_lines)} answer lines as a SINGLE chunk...")
    
    # Build REF mapping
    ref_to_question = {f"REF-{chr(65+i)}": q for i, q in enumerate(questions)}
    
    # Create numbered lines
    numbered_lines = list(enumerate(answer_lines))
    
    # Build prompt with ALL lines
    user_prompt = _build_answer_map_user_prompt(numbered_lines, questions)
    
    try:
        # Call LLM once with ALL answer text
        chunk_ranges = _call_groq_with_retries(
            client, ANSWER_MAP_SYSTEM_PROMPT, user_prompt,
            _parse_answer_map_llm_response, budget, log
        )
    except Exception as e:
        log(f"WARNING: Answer mapping failed: {e}")
        return {}

    # Validate and resolve ranges
    valid_ranges = []
    for r in chunk_ranges:
        if r["ref"] in ref_to_question:
            # Ensure ranges are within bounds
            start = max(0, min(r["start_line"], len(answer_lines) - 1))
            end = max(start, min(r["end_line"], len(answer_lines) - 1))
            valid_ranges.append({
                "ref": r["ref"],
                "start_line": start,
                "end_line": end
            })

    # Sort and resolve overlaps
    valid_ranges.sort(key=lambda x: x["start_line"])
    resolved = []
    
    for i, r in enumerate(valid_ranges):
        if i + 1 < len(valid_ranges):
            next_start = valid_ranges[i + 1]["start_line"]
            if r["end_line"] >= next_start:
                r["end_line"] = next_start - 1
        if r["end_line"] >= r["start_line"]:
            resolved.append(r)

    log(f"Resolved {len(resolved)} answer ranges")

    # Extract answers
    qa_map = {}
    for r in resolved:
        start, end = r["start_line"], r["end_line"]
        # Include ALL lines from start to end
        verbatim_lines = []
        for j in range(start, end + 1):
            if 0 <= j < len(answer_lines) and answer_lines[j].strip():
                verbatim_lines.append(answer_lines[j])
        
        original_question = ref_to_question[r["ref"]]
        answer_text = " ".join(verbatim_lines).strip()
        
        # Clean but preserve content
        answer_text = strip_question_restatement(answer_text)
        answer_text = strip_full_question_echo(answer_text, original_question)
        
        if answer_text:
            qa_map[original_question] = answer_text

    log(f"Extracted {len(qa_map)} complete answers")
    return qa_map


def _build_answer_map_user_prompt(numbered_lines: list, questions: list) -> str:
    """Build user prompt with ALL answer lines."""
    questions_block = "\n".join(
        f"[REF-{chr(65+i)}] {q}" for i, q in enumerate(questions)
    )
    
    # Include ALL lines
    lines_block = "\n".join(f"[{idx}] {text}" for idx, text in numbered_lines)
    
    return (
        f"OFFICIAL QUESTIONS (use REF labels):\n{questions_block}\n\n"
        f"STUDENT'S COMPLETE ANSWER TEXT (ALL lines numbered):\n{lines_block}\n\n"
        f"FIND THE LINE RANGE FOR EACH QUESTION'S ANSWER. Include ALL content for each answer."
    )


def _parse_answer_map_llm_response(content: str) -> list:
    """Parse LLM response for answer mapping."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if not isinstance(data, dict) or "answers" not in data:
        raise ValueError(f"Missing 'answers' key")

    answers = data["answers"]
    if not isinstance(answers, list):
        raise ValueError(f"'answers' must be a list")

    result = []
    for item in answers:
        if isinstance(item, dict) and "ref" in item and "start_line" in item and "end_line" in item:
            try:
                result.append({
                    "ref": str(item["ref"]).strip().upper(),
                    "start_line": int(item["start_line"]),
                    "end_line": int(item["end_line"]),
                })
            except (ValueError, TypeError):
                continue

    return result


def strip_question_restatement(answer_text: str) -> str:
    """Strip leading question restatement labels."""
    patterns = [
        r'^\s*Ans(?:wer)?\s*\d*\s*[.:\-]\s*',
        r'^\s*उत्तर\s*\d*\s*[\-\:]\s*',
        r'^\s*प्र[०.\s]+\d+[.\s:-]*',
        r'^\s*प्रश्न[.\s]+\d+[.\s:-]*',
        r'^\s*Q\.?\s*\d+[.\s:-]*',
    ]
    
    text = answer_text
    for _ in range(2):
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE, count=1)
        text = text.strip()
        if text == answer_text:
            break
    
    return text


def strip_full_question_echo(answer_text: str, question_text: str) -> str:
    """Strip full question echo from answer start."""
    q_norm = re.sub(r'[^\w\s]', ' ', question_text.lower())
    q_norm = re.sub(r'\s+', ' ', q_norm).strip()
    q_words = q_norm.split()
    
    if len(q_words) < 3:
        return answer_text
    
    answer_words = answer_text.split()
    if len(answer_words) < len(q_words):
        return answer_text
    
    prefix = " ".join(answer_words[:len(q_words)])
    prefix_norm = re.sub(r'[^\w\s]', ' ', prefix.lower())
    prefix_norm = re.sub(r'\s+', ' ', prefix_norm).strip()
    
    similarity = difflib.SequenceMatcher(None, prefix_norm, q_norm).ratio()
    
    if similarity >= 0.5:  # Lowered threshold to catch more echoes
        remaining = " ".join(answer_words[len(q_words):])
        return remaining.strip()
    
    return answer_text


# =========================================================
# LLM CALL WITH RETRIES
# =========================================================

def _estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN_ESTIMATE) + 1


def _call_groq_with_retries(client, system_prompt: str, user_prompt: str,
                             response_parser, budget: _TokenBudgetTracker,
                             log, max_retries: int = 4):
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
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=8192,  # MAXIMUM for long answers
                )
            budget.record_usage(estimated_tokens)
            content = response.choices[0].message.content
            return response_parser(content)

        except Exception as e:
            last_error = e
            detail = _parse_rate_limit_detail(str(e))
            
            if detail and detail["limit_type"] == "TPD":
                raise Exception(f"Daily quota exhausted. Resets in {detail['wait_seconds']/60:.0f} min.") from e
            
            if detail:
                budget.record_actual_from_error(detail["used"], detail["limit"])
                log(f"Rate limit: waiting {detail['wait_seconds']+0.5:.1f}s")
                time.sleep(detail["wait_seconds"] + 0.5)
                budget.reset_window()
                skip_next_proactive_check = True
            else:
                wait_s = min(5.0 * attempt, 30.0)
                log(f"Error (attempt {attempt}): {e}. Waiting {wait_s:.1f}s")
                time.sleep(wait_s)

    raise Exception(f"Failed after {max_retries+1} attempts. Last error: {last_error}")


# =========================================================
# QUESTION IDENTIFICATION
# =========================================================

def _parse_qp_llm_response(content: str) -> tuple:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if not isinstance(data, dict) or "question_paper_pages" not in data or "questions" not in data:
        raise ValueError(f"Missing required keys")

    qp_pages = [int(x) for x in data["question_paper_pages"]]
    questions = [str(x).strip() for x in data["questions"] if str(x).strip()]
    
    return qp_pages, questions


def identify_questions_with_llm(pages: list, status_callback=None) -> tuple:
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

    for page in pages:
        page["raw_text"] = clean_ocr_text(page["raw_text"])

    chunks = _chunk_pages_by_char_budget(pages)
    log(f"Split {len(pages)} page(s) into {len(chunks)} chunk(s)")

    valid_page_numbers = {p["page_number"] for p in pages}
    max_page_number = max(valid_page_numbers) if valid_page_numbers else 0
    chunk_results = []

    for i, chunk in enumerate(chunks):
        page_nums = [p["page_number"] for p in chunk]
        log(f"Analyzing chunk {i+1}/{len(chunks)} (pages {page_nums})...")

        try:
            user_prompt = _build_qp_user_prompt(chunk)
            qp_pages, _ = _call_groq_with_retries(
                client, QP_SYSTEM_PROMPT, user_prompt,
                _parse_qp_llm_response, budget, log
            )
        except Exception as e:
            log(f"WARNING: chunk {i+1} failed: {e}")
            continue

        recovered_pages = []
        for pn in qp_pages:
            if pn in valid_page_numbers:
                recovered_pages.append(pn)
            else:
                split_result = _try_split_concatenated_page_number(
                    pn, valid_page_numbers, max_page_number
                )
                if split_result:
                    log(f"Recovered concatenated: {pn} -> {split_result}")
                    recovered_pages.extend(split_result)

        chunk_results.append((sorted(set(recovered_pages)), []))

    all_qp_pages = set()
    for qp_pages, _ in chunk_results:
        all_qp_pages.update(qp_pages)

    qp_page_indices = sorted(p - 1 for p in all_qp_pages)
    log(f"Question paper pages: {[p+1 for p in qp_page_indices]}")

    qp_pages = [pages[i] for i in qp_page_indices]
    questions = extract_canonical_questions(qp_pages, status_callback)

    log(f"Extracted {len(questions)} canonical question(s)")
    return qp_page_indices, questions


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
    return "Here are the OCR'd pages:\n\n" + "\n\n".join(blocks)


def _try_split_concatenated_page_number(n: int, valid_page_numbers: set, max_page: int) -> list:
    if n in valid_page_numbers:
        return []

    s = str(n)
    max_digits = len(str(max_page))
    
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
    for num_parts in range(2, min(len(s) + 1, 4)):
        for widths in __import__('itertools').product(range(1, max_digits + 1), repeat=num_parts):
            if sum(widths) != len(s):
                continue
            result = split_attempt(s, widths)
            if result:
                candidates.append(result)

    return candidates[0] if candidates else []


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
        raise Exception("GROQ_API_KEY not found")

    client = Groq(api_key=api_key)
    budget = _TokenBudgetTracker()

    user_prompt = _build_canonical_questions_prompt(qp_pages)
    log(f"Extracting canonical questions from {len(qp_pages)} page(s)...")

    try:
        questions = _call_groq_with_retries(
            client, QUESTION_PAPER_ONLY_SYSTEM_PROMPT, user_prompt,
            _parse_canonical_questions_response, budget, log
        )
    except Exception as e:
        log(f"WARNING: canonical extraction failed: {e}")
        return []

    log(f"Extracted {len(questions)} canonical question(s)")
    return questions


def _build_canonical_questions_prompt(qp_pages: list) -> str:
    blocks = [f"--- PAGE {p['page_number']} ---\n{p['raw_text']}" for p in qp_pages]
    return "Complete question paper text:\n\n" + "\n\n".join(blocks)


def _parse_canonical_questions_response(content: str) -> list:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if not isinstance(data, dict) or "questions" not in data:
        raise ValueError(f"Missing 'questions' key")

    questions = data["questions"]
    if not isinstance(questions, list):
        raise ValueError(f"'questions' must be a list")

    return [str(q).strip() for q in questions if str(q).strip()]


# =========================================================
# COMPLETE PIPELINE
# =========================================================

@_diagnose_tuple_errors
def process_pdf(file_input, status_callback=None):
    """Complete PDF processing pipeline with FIX for long answers."""
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    # Step 1: Normalize input
    file_bytes, file_name = _normalize_file_input(file_input, default_name="document.pdf")
    
    # Step 2: Preprocess PDF
    log("Preprocessing PDF...")
    file_bytes = preprocess_pdf(file_bytes, dpi=250)
    
    # Step 3: OCR
    pages = run_ocr(file_bytes, file_name, status_callback)
    
    # Step 4: Build OCR JSON
    log("Building OCR JSON...")
    ocr_json = build_ocr_json(pages)
    
    # Step 5: Identify questions
    qp_page_indices, official_questions = identify_questions_with_llm(pages, status_callback)
    log(f"Found {len(official_questions)} questions on {len(qp_page_indices)} page(s)")
    
    if not official_questions:
        raise Exception("No questions extracted from the document.")
    
    # Step 6: Extract answer pages
    answer_page_indices = [i for i in range(len(pages)) if i not in qp_page_indices]
    answer_pages = [pages[i] for i in answer_page_indices]
    log(f"Found {len(answer_pages)} answer page(s)")
    
    # Step 7: Flatten answer lines - KEEP EVERYTHING
    answer_lines = []
    for page in answer_pages:
        lines = page["raw_text"].split("\n")
        for line in lines:
            line = clean_ocr_text(line)
            if line:  # Keep all non-empty lines
                answer_lines.append(line)
    
    log(f"Flattened {len(answer_lines)} answer lines")
    
    # Step 8: Validate answer pages
    if not _sanity_check_answer_pages(answer_lines, len(official_questions), log):
        log("WARNING: Answer pages seem too short. Continuing anyway...")
    
    # Step 9: Map answers - USING COMPLETE TEXT
    log("Mapping answers to questions (processing ALL text at once)...")
    qa_map = map_answers_with_llm(answer_lines, official_questions, status_callback)
    
    # Step 10: Build QA pairs
    qa_pairs = []
    matched_count = 0
    total_answer_chars = 0
    
    for q in official_questions:
        answer = qa_map.get(q, "")
        if answer:
            matched_count += 1
            total_answer_chars += len(answer)
        qa_pairs.append({
            "question": q,
            "answer": answer,
            "matched": bool(answer)
        })
    
    log(f"Matched {matched_count} of {len(official_questions)} questions")
    log(f"Total answer text length: {total_answer_chars} characters")
    
    if matched_count == 0:
        raise Exception("No answers could be matched to questions.")
    
    return ocr_json, qa_pairs


def _sanity_check_answer_pages(answer_lines: list, num_questions: int, log=print) -> bool:
    total_chars = sum(len(line) for line in answer_lines)
    avg_chars_per_question = total_chars / max(num_questions, 1)
    
    # Very conservative - only warn if truly empty
    if avg_chars_per_question < 50 and total_chars < 500:
        log(f"WARNING: Only {total_chars} chars for {num_questions} questions.")
        return False
    
    return True


def save_outputs(ocr_json: dict, qa_pairs: list, output_dir: str = ".",
                  base_name: str = "document") -> tuple:
    """Save outputs to JSON files."""
    ocr_path = os.path.join(output_dir, f"{base_name}_ocr.json")
    qa_path = os.path.join(output_dir, f"{base_name}_qa_pairs.json")
    
    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(ocr_json, f, ensure_ascii=False, indent=2)
    
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    return ocr_path, qa_path


# =========================================================
# NOISE DETECTION
# =========================================================

NOISE_RE = re.compile(
    r'(?:Teacher\'?s?\s*Signature'
    r'|Tancher\'?s?\s*Signature'
    r'|PAGE\s*NO'
    r'|^\s*DATE\b)',
    re.IGNORECASE
)


def is_noise(line: str) -> bool:
    return bool(NOISE_RE.search(line))


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"Processing: {pdf_path}")
        
        try:
            ocr_json, qa_pairs = process_pdf(pdf_path)
            ocr_path, qa_path = save_outputs(ocr_json, qa_pairs, base_name="output")
            
            print(f"\n✅ Done! Outputs saved to:")
            print(f"  - OCR: {ocr_path}")
            print(f"  - QA Pairs: {qa_path}")
            print(f"\n📊 Summary:")
            print(f"  - Total pages: {ocr_json['total_pages']}")
            print(f"  - Questions: {len(qa_pairs)}")
            print(f"  - Matched: {sum(1 for p in qa_pairs if p['matched'])}")
            
            # Show sample
            print("\n📝 Sample Output:")
            for i, pair in enumerate(qa_pairs[:3]):
                if pair['matched']:
                    print(f"\nQ{i+1}: {pair['question'][:100]}...")
                    print(f"A{i+1}: {pair['answer'][:150]}...")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Usage: python script.py <path_to_pdf>")
