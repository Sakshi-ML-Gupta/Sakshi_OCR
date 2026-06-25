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

# =========================================================
# LLM-BASED QUESTION PAPER / QUESTION DETECTION (Groq)
# =========================================================

GROQ_MODEL = "openai/gpt-oss-120b"

TPM_LIMIT = 8000
TPM_SAFETY_FRACTION = 0.85
CHARS_PER_TOKEN_ESTIMATE = 2.0

MAX_CHARS_PER_CHUNK = 15000  # Increased for longer content
CHUNK_OVERLAP_PAGES = 4

QP_SYSTEM_PROMPT = """You are analyzing OCR text extracted from a scanned student exam assignment booklet.

Your task: read the pages shown and return ONLY valid JSON (no markdown fences, no commentary) in EXACTLY this shape:

{
  "question_paper_pages": [14, 16, 18],
  "questions": ["1. Example question text. (10)", "2. Another example question. (10)"]
}

CRITICAL RULES:
- Extract questions EXACTLY as they appear on the question paper pages - word for word.
- IMPORTANT: Each sub-question like (i), (ii), (iii), (iv) must be a SEPARATE question entry.
- Format sub-questions as: "1. (i) Sub-question text" (with the main number and sub-part)
- For example, if you see "1. a) Renaissance" and "1. b) Amoretti", extract them as:
  "1. a) Renaissance" and "1. b) Amoretti" as SEPARATE entries.
- Do NOT merge multiple sub-questions into one entry.
- Preserve the EXACT original text -- do not paraphrase, do not fix OCR errors.
- Output ONLY the JSON object. No prose before or after."""


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

    # FIX: Ensure sub-questions are properly split
    # If any question contains multiple sub-parts like "(i)", "(ii)", "(iii)" 
    # that were merged, split them
    final_questions = []
    for q in questions:
        # Check if this question contains multiple sub-parts
        sub_parts = re.findall(r'\([ivx]+\)\s*[^\(]*?(?=\s*\([ivx]+\)|\Z)', q, re.IGNORECASE)
        if len(sub_parts) > 1:
            # This question has multiple sub-parts - split them
            for sub in sub_parts:
                if sub.strip():
                    final_questions.append(sub.strip())
        else:
            final_questions.append(q)
    
    return qp_pages, final_questions


# =========================================================
# FIX: SPLIT SUB-QUESTIONS IN THE ANSWER MAPPING TOO
# =========================================================

def map_answers_with_llm(answer_lines: list, questions: list, status_callback=None) -> dict:
    """
    Maps each official question to its verbatim answer text.
    Each sub-question gets its own separate mapping.
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

    # Deterministic REF label <-> question index mapping
    ref_to_question = {f"REF-{chr(65+i)}": q for i, q in enumerate(questions)}

    numbered_lines = list(enumerate(answer_lines))
    chunks = _chunk_lines_by_char_budget(numbered_lines)
    log(f"Split {len(answer_lines)} answer line(s) into {len(chunks)} LLM chunk(s) for answer mapping")

    all_ranges = []

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
            continue

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

    # Deduplicate: keep the longest range for each REF
    best_by_ref = {}
    for r in all_ranges:
        existing = best_by_ref.get(r["ref"])
        if existing is None or (r["end_line"] - r["start_line"]) > (existing["end_line"] - existing["start_line"]):
            best_by_ref[r["ref"]] = r

    deduped_ranges = list(best_by_ref.values())
    resolved_ranges = _resolve_overlapping_answer_ranges(deduped_ranges)

    log(f"Final answer mapping: {len(resolved_ranges)} of {len(questions)} question(s) matched")

    # Slice the ORIGINAL answer_lines verbatim using the resolved ranges
    qa_map = {}
    for r in resolved_ranges:
        start, end = r["start_line"], r["end_line"]
        verbatim_lines = [
            answer_lines[j] for j in range(start, end + 1)
            if 0 <= j < len(answer_lines) and answer_lines[j].strip() and not is_noise(answer_lines[j])
        ]
        original_question = ref_to_question[r["ref"]]
        answer_text = " ".join(verbatim_lines).strip()
        qa_map[original_question] = strip_question_restatement(answer_text)

    return qa_map


def split_merged_answers(qa_pairs: list) -> list:
    """
    FIX: Splits merged answers into separate Q&A pairs for each sub-question.
    If an answer contains multiple sub-answers like "1.(i)" and "1.(ii)" merged,
    this splits them into separate entries.
    """
    final_pairs = []
    
    for pair in qa_pairs:
        question = pair["question"]
        answer = pair["answer"]
        
        # Check if the answer contains multiple sub-answers
        # Look for patterns like "1.(i)", "1.(ii)" etc. in the answer
        sub_answer_pattern = re.compile(r'(\d+\.?\s*\([ivx]+\))\s*[^\n]*?(?=\s*\d+\.?\s*\([ivx]+\)|\Z)', re.IGNORECASE | re.DOTALL)
        sub_matches = list(sub_answer_pattern.finditer(answer))
        
        if len(sub_matches) > 1:
            # Multiple sub-answers found - split them
            for match in sub_matches:
                sub_text = match.group(0).strip()
                # Try to find which sub-question this belongs to
                sub_label = match.group(1).strip()
                
                # Find the corresponding question
                for q in final_pairs:
                    if sub_label in q["question"]:
                        # Append to existing
                        q["answer"] += "\n" + sub_text
                        break
                else:
                    # Create new entry
                    final_pairs.append({
                        "question": sub_label,
                        "answer": sub_text,
                        "matched": True
                    })
        else:
            # No sub-answers found - keep as is
            final_pairs.append(pair)
    
    return final_pairs


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

    # Extract questions and question paper pages using LLM
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

    # Map each question to its answer independently
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

    # Build Q&A pairs in official question order
    qa_pairs = []
    for q in official_questions:
        qa_pairs.append({
            "question": q,
            "answer": qa_map.get(q, ""),
            "matched": q in qa_map,
        })

    # FIX: Split merged answers into separate Q&A pairs for evaluation
    log("Splitting merged sub-answers into separate Q&A pairs...")
    qa_pairs = split_merged_answers(qa_pairs)

    log(f"Done -- {len(qa_pairs)} Q-A pairs ({matched_count} matched)")

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
