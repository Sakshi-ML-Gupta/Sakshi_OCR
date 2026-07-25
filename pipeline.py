import os
import io
import re
import json
import time
import threading
import fitz
import httpx
from pathlib import Path

# Optional LangChain splitter with pure-python fallback
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

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
# INPUT NORMALIZATION & DIAGNOSTICS
# =========================================================

def _normalize_file_input(file_input, default_name="document.pdf"):
    if isinstance(file_input, tuple):
        if len(file_input) < 2:
            raise ValueError(f"Tuple file_input must have at least 2 items, got {len(file_input)}")
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
            raise TypeError(f"file_input.read() returned {type(data).__name__}, expected bytes.")
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
                raise TypeError(f"[DIAGNOSTIC] Error in {func.__name__}(): {e}") from e
            raise
    return wrapper


_groq_call_lock = threading.Lock()


# =========================================================
# PREPROCESS & OCR
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

    log("WARNING: Page breaks not found cleanly. Using full document as single page.")
    return [markdown.strip()]


def run_ocr(file_content: bytes, file_name: str, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    file_name = _coerce_name(file_name, default_name="document.pdf")
    api_key = get_api_key("DATALAB_API_KEY")
    if not api_key:
        raise Exception("DATALAB_API_KEY not found in secrets")

    headers = {"X-API-Key": api_key}
    log(f"Submitting document to Datalab OCR... ({len(file_content)/(1024*1024):.1f}MB)")

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
    check_url = data["request_check_url"]
    log("Document submitted -- polling OCR...")

    for attempt in range(150):
        poll_resp = httpx.get(check_url, headers=headers, timeout=60)
        result = poll_resp.json()
        if result.get("status") == "complete":
            log("OCR complete.")
            break
        if result.get("status") == "failed":
            raise Exception(f"OCR failed: {result.get('error')}")
        time.sleep(2)
    else:
        raise Exception("OCR timed out")

    markdown = result.get("markdown") or ""
    page_texts = _split_paginated_markdown(markdown, result.get("page_count"), log=log)

    return [{"page_number": idx + 1, "raw_text": text} for idx, text in enumerate(page_texts)]


def build_ocr_json(pages: list) -> dict:
    return {
        "total_pages": len(pages),
        "pages": [{"page_number": p["page_number"], "text": p["raw_text"]} for p in pages]
    }


# =========================================================
# TOKEN BUDGET TRACKER (FIXED LOOP ISSUE)
# =========================================================

GROQ_MODEL = "openai/gpt-oss-120b"
TPM_LIMIT = 8000
TPM_SAFETY_FRACTION = 0.80

def _estimate_tokens(text: str) -> int:
    return int(len(text) / 2.5) + 1


class _TokenBudgetTracker:
    def __init__(self, tpm_limit=TPM_LIMIT, safety_fraction=TPM_SAFETY_FRACTION):
        self.tpm_limit = tpm_limit
        self.safe_limit = tpm_limit * safety_fraction
        self.events = []

    def _prune(self):
        now = time.monotonic()
        self.events = [(ts, tok) for ts, tok in self.events if now - ts < 60]

    def wait_if_needed(self, upcoming_tokens: int, log=print):
        self._prune()
        used = sum(tok for _, tok in self.events)
        
        if (used + upcoming_tokens) > self.safe_limit:
            log("Token limit buffer reached. Waiting 12s for cooling...")
            time.sleep(12)
            self._prune()

    def record_usage(self, tokens: int):
        self.events.append((time.monotonic(), tokens))

    def reset_window(self):
        self.events.clear()


def _call_groq_with_retries(client, system_prompt: str, user_prompt: str,
                              response_parser, budget: "_TokenBudgetTracker",
                              log, max_retries: int = 4):
    import groq

    estimated_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt) + 500

    for attempt in range(1, max_retries + 2):
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
            return response_parser(response.choices[0].message.content)

        except (groq.RateLimitError, groq.BadRequestError) as e:
            budget.reset_window()
            wait_time = 15.0 * attempt
            log(f"Rate limit hit. Sleeping {wait_time}s before retry...")
            time.sleep(wait_time)

        except Exception as e:
            log(f"Groq exception: {e}")
            time.sleep(3.0)

    raise Exception("Groq API call failed after retries.")


# =========================================================
# QUESTION PAPER IDENTIFICATION
# =========================================================

QP_SYSTEM_PROMPT = """Analyze student exam assignment booklet pages.
Return ONLY JSON:
{
  "question_paper_pages": [1, 2],
  "questions": ["1.(i) Question text...", "1.(ii) Question text..."]
}"""


def identify_questions_with_llm(pages: list, status_callback=None) -> tuple:
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    from groq import Groq
    api_key = get_api_key("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    budget = _TokenBudgetTracker()

    qp_pages_found = set()
    
    # Process 2 pages at a time for QP detection
    for i in range(0, len(pages), 2):
        chunk_pages = pages[i:i+2]
        user_prompt = "\n\n".join([f"--- PAGE {p['page_number']} ---\n{p['raw_text']}" for p in chunk_pages])
        
        def parse_qp(content):
            c = re.sub(r'^```(?:json)?\s*\n?|\n?```\s*$', '', content.strip())
            d = json.loads(c)
            return d.get("question_paper_pages", []), d.get("questions", [])

        try:
            qp_p, _ = _call_groq_with_retries(client, QP_SYSTEM_PROMPT, user_prompt, parse_qp, budget, log)
            for p in qp_p:
                if 1 <= p <= len(pages):
                    qp_pages_found.add(p - 1)
        except Exception as e:
            log(f"QP detection error on pages {i+1}-{i+2}: {e}")

    qp_indices = sorted(list(qp_pages_found))
    qp_pages = [pages[idx] for idx in qp_indices]

    if not qp_pages:
        return [], []

    # Canonical Question Extraction Pass
    qp_prompt = "\n\n".join([f"--- PAGE {p['page_number']} ---\n{p['raw_text']}" for p in qp_pages])
    
    def parse_canonical(content):
        c = re.sub(r'^```(?:json)?\s*\n?|\n?```\s*$', '', content.strip())
        return json.loads(c).get("questions", [])

    canonical_questions = _call_groq_with_retries(
        client, 
        "Extract all individual official questions from the Question Paper pages. Return JSON: {\"questions\": [\"1. ...\", \"2. ...\"]}", 
        qp_prompt, 
        parse_canonical, 
        budget, 
        log
    )

    return qp_indices, canonical_questions


# =========================================================
# CHUNK-BASED ANSWER MAPPING (LANGCHAIN / RECURSIVE CHUNKING)
# =========================================================

NOISE_RE = re.compile(r'(?:Teacher\'?s?\s*Signature|PAGE\s*NO|^\s*DATE\b|^\s*\d{1,3}\s*$)', re.IGNORECASE)

def is_noise(line: str) -> bool:
    return bool(NOISE_RE.search(line))


ANSWER_CHUNK_SYSTEM_PROMPT = """You are mapping chunks of student answers to official questions.
Find which official question REF (e.g. REF-A, REF-B) this text belongs to.

Return ONLY JSON:
{
  "mapped_answers": [
    {
      "ref": "REF-A",
      "verbatim_text": "Exact text segment corresponding to this question from the chunk"
    }
  ]
}"""


def _split_text_with_overlap(text: str, chunk_size=3500, chunk_overlap=250) -> list:
    """Uses LangChain RecursiveCharacterTextSplitter if installed, else python fallback."""
    if HAS_LANGCHAIN:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)
    
    # Fallback Recursive Splitter Logic
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunks.append(text[start:].strip())
            break
        
        # Try to break at paragraph or line break
        break_pos = text.rfind("\n\n", start, end)
        if break_pos == -1:
            break_pos = text.rfind("\n", start, end)
        if break_pos == -1 or break_pos < start + (chunk_size // 2):
            break_pos = end

        chunks.append(text[start:break_pos].strip())
        start = max(start + 1, break_pos - chunk_overlap)
        
    return [c for c in chunks if c]


def map_answers_with_llm(answer_text: str, questions: list, status_callback=None) -> dict:
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    from groq import Groq
    api_key = get_api_key("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    budget = _TokenBudgetTracker()

    ref_to_question = {f"REF-{chr(65+i)}": q for i, q in enumerate(questions)}
    questions_block = "\n".join(f"[REF-{chr(65+i)}] {q}" for i, q in enumerate(questions))

    # Split long answer text into manageable overlapping chunks (~1500 tokens / 3500 chars)
    chunks = _split_text_with_overlap(answer_text, chunk_size=3500, chunk_overlap=250)
    log(f"Answer text divided into {len(chunks)} chunk(s) with 250-char overlap.")

    qa_map = {q: [] for q in questions}

    for idx, chunk in enumerate(chunks):
        log(f"Processing Answer Chunk {idx+1}/{len(chunks)}...")
        user_prompt = f"OFFICIAL QUESTIONS:\n{questions_block}\n\nSTUDENT ANSWER CHUNK:\n{chunk}"

        def parse_chunk_res(content):
            c = re.sub(r'^```(?:json)?\s*\n?|\n?```\s*$', '', content.strip())
            data = json.loads(c)
            return data.get("mapped_answers", [])

        try:
            mapped_segments = _call_groq_with_retries(
                client, ANSWER_CHUNK_SYSTEM_PROMPT, user_prompt,
                parse_chunk_res, budget, log
            )
            for item in mapped_segments:
                ref = str(item.get("ref", "")).strip().upper()
                verbatim = str(item.get("verbatim_text", "")).strip()
                
                if ref in ref_to_question and verbatim:
                    q_title = ref_to_question[ref]
                    # Avoid duplicate overlapping text insertions
                    if not any(verbatim in existing for existing in qa_map[q_title]):
                        qa_map[q_title].append(verbatim)

        except Exception as e:
            log(f"Error mapping chunk {idx+1}: {e}")

    # Format final outputs
    final_qa_map = {}
    for q, text_list in qa_map.items():
        if text_list:
            final_qa_map[q] = "\n\n".join(text_list).strip()
        else:
            final_qa_map[q] = ""

    return final_qa_map


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
    ocr_json = build_ocr_json(pages)

    log("Identifying Question Paper pages and extracting canonical questions...")
    qp_indices, official_questions = identify_questions_with_llm(pages, status_callback)

    if not official_questions:
        raise Exception("Could not detect official questions from Question Paper.")

    log(f"Found {len(official_questions)} question(s). Extracting answers...")

    # Filter out QP pages to isolate Student Answers
    answer_pages = [pages[i] for i in range(len(pages)) if i not in qp_indices]

    answer_lines = []
    for page in answer_pages:
        for line in page["raw_text"].split("\n"):
            if not is_noise(line):
                answer_lines.append(line)

    full_answer_text = "\n".join(answer_lines)

    log("Mapping student answers chunk-by-chunk...")
    qa_map = map_answers_with_llm(full_answer_text, official_questions, status_callback)

    qa_pairs = [
        {
            "question": q,
            "answer": qa_map.get(q, ""),
            "matched": bool(qa_map.get(q, "").strip())
        }
        for q in official_questions
    ]

    return ocr_json, qa_pairs


def save_outputs(ocr_json: dict, qa_pairs: list, output_dir: str = ".", base_name: str = "document") -> tuple:
    ocr_path = os.path.join(output_dir, f"{base_name}_ocr.json")
    qa_path = os.path.join(output_dir, f"{base_name}_qa_pairs.json")

    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(ocr_json, f, ensure_ascii=False, indent=2)

    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    return ocr_path, qa_path
