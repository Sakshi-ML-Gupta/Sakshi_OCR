import os
import io
import re
import json
import time
import fitz
import httpx
from pathlib import Path

# =========================================================
# API KEY
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
# OCR — Tesseract (local, no API needed)
#
# Renders each PDF page as a high-DPI image via PyMuPDF,
# then runs Tesseract OCR with Hindi+English language support.
# Returns per-page text, same structure as the old Datalab path.
# =========================================================

def run_ocr(file_bytes: bytes, file_name: str, status_callback=None, dpi: int = 300):
    """
    Run Tesseract OCR on a PDF.

    Requirements:
        - System: tesseract-ocr  (apt install tesseract-ocr tesseract-ocr-hin)
        - Python: pip install pytesseract Pillow

    The `dpi` parameter controls image resolution — 300 is a good balance
    of accuracy vs. speed for handwritten exam booklets.
    """
    import pytesseract
    from PIL import Image

    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    size_mb = len(file_bytes) / (1024 * 1024)
    log(f"Opening PDF for Tesseract OCR... ({size_mb:.1f}MB)")

    src_doc = fitz.open(stream=file_bytes, filetype="pdf")
    total_pages = len(src_doc)
    log(f"PDF has {total_pages} page(s) — running Tesseract at {dpi} DPI")

    # Determine language: default to Hindi+English for Indian exam papers.
    # Users can override with TESS_LANG env var (e.g. "eng", "hin+eng", "tam+eng").
    lang = os.getenv("TESS_LANG", "hin+eng")
    log(f"Tesseract language: {lang}")

    pages = []
    for page_num in range(total_pages):
        page = src_doc[page_num]
        pix = page.get_pixmap(dpi=dpi)

        # Convert PyMuPDF pixmap to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Run Tesseract
        text = pytesseract.image_to_string(img, lang=lang, config="--psm 6")

        pages.append({
            "page_number": page_num + 1,
            "raw_text": text.strip()
        })

        if (page_num + 1) % 5 == 0 or (page_num + 1) == total_pages:
            log(f"  OCR progress: {page_num + 1}/{total_pages} pages")

    src_doc.close()
    log(f"Tesseract OCR done — {len(pages)} page(s) extracted")
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
# REFERENCE BOOK OCR — Tesseract handles page-by-page natively
# =========================================================

def process_reference(file_input, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    if isinstance(file_input, (str, Path)):
        file_bytes = Path(file_input).read_bytes()
        file_name  = Path(file_input).name
    else:
        file_bytes = file_input.read()
        file_name  = getattr(file_input, "name", "reference.pdf")

    pages = run_ocr(file_bytes, file_name, status_callback)
    log(f"Reference OCR complete — {len(pages)} page(s)")
    return build_ocr_json(pages)


# =========================================================
# DETECT QUESTION PAPER PAGE
# The page that contains the printed list of questions
# Identified by having numbered questions >= 5 items
# =========================================================

def find_question_paper_pages(pages: list, min_questions: int = 2) -> list:
    """
    Returns list of page indices (0-based) that look like GENUINE exam
    question paper pages — as opposed to:
    - ID card / registration / admin pages (numbered instructions,
      terms and conditions)
    - Handwritten answer pages where the student restates the question
      number before writing their answer

    Strategy: a real question paper page must satisfy ALL of:
    1. Has 2+ lines matching a question-start pattern (numbered or lettered)
    2. Has at least one STRONG exam-paper signal:
       - mark allocation pattern like "10", "X2=10", "3X20=60", "5X2"
       - a SECTION/PART header (SECTION-A, भाग-1, PART B)
       - explicit instruction phrase ("answer all questions",
         "सभी प्रश्न अनिवार्य", "TMA", "assignment code", course code pattern)
    3. Does NOT contain admin/ID-card signals (enrolment number, IGNOU
       student identity, regional centre, "produced on demand", QR code
       instructions) — these are registration pages, not exam papers
    4. Does NOT contain handwritten-answer signals (उत्तर, Ans-, Teacher's
       Signature, PAGE NO/DATE handwritten template, A.15- style answer
       labels)
    """
    Q_LINE_NUM   = re.compile(r'^\s*\d+[\.\)]\s+.{15,}')
    Q_LINE_LATIN = re.compile(r'^\s*[a-d]\)\s+.{5,}', re.IGNORECASE)
    Q_LINE_DEVA  = re.compile(r'^\s*[क-घ]\)\s+.{5,}')

    MARK_ALLOCATION = re.compile(
        r'(?:\d+\s*[xX]\s*\d+\s*=?\s*\d*|\b\d{1,3}\s*$|\(\s*\d+\s*\))',
    )
    SECTION_HEADER = re.compile(
        r'(?:SECTION\s*[-–]?\s*[A-Z]|PART\s*[-–]?\s*\d|भाग\s*[-–]?\s*\d|भाग\s*[-–]?\s*[१-९])',
        re.IGNORECASE
    )
    EXAM_INSTRUCTION = re.compile(
        r'(?:answer\s+all\s+questions|all\s+questions\s+are\s+compulsory'
        r'|सभी\s*प्रश्न\s*अनिवार्य|assignment\s*code|TMA\b|कुल\s*अंक'
        r'|words?\s+each|शब्दों\s*में)',
        re.IGNORECASE
    )

    ADMIN_PAGE_MARKERS = re.compile(
        r'(?:enrolment\s*number|enrollment\s*no|student\s*identity\s*card'
        r'|regional\s*cent[er]+|study\s*cent[er]+|produced\s+on\s+demand'
        r'|qr\s*code|registration\s*details|admission\s*status'
        r'|father.?s\s*name|programme\s*registered|IGNOU\s*-\s*Student)',
        re.IGNORECASE
    )

    ANSWER_PAGE_MARKERS = re.compile(
        r'(?:उत्तर\s*[\-\:]|Ans\.?\s*[\-\:]|A\.\d|A\d+\s*[\-\:]'
        r'|Teacher.?s\s*Signature|PAGE\s*NO[\.\:]?\s*\d*\s*DATE)',
        re.IGNORECASE
    )

    candidate_pages = []
    weak_pages = []   # pages with question-like lines but no strong signal

    for i, page in enumerate(pages):
        text  = page["raw_text"]
        lines = text.split("\n")

        q_count = sum(
            1 for line in lines
            if Q_LINE_NUM.match(line.strip())
            or Q_LINE_LATIN.match(line.strip())
            or Q_LINE_DEVA.match(line.strip())
        )

        if q_count < min_questions:
            continue

        if ADMIN_PAGE_MARKERS.search(text):
            continue   # ID card / registration page

        if ANSWER_PAGE_MARKERS.search(text):
            continue   # student's handwritten answer page

        has_strong_signal = bool(
            MARK_ALLOCATION.search(text)
            or SECTION_HEADER.search(text)
            or EXAM_INSTRUCTION.search(text)
        )

        if has_strong_signal:
            candidate_pages.append(i)
        else:
            weak_pages.append(i)

    # Promote weak pages adjacent to a confirmed question-paper page
    confirmed_set = set(candidate_pages)
    for i in weak_pages:
        if (i - 1) in confirmed_set or (i + 1) in confirmed_set:
            candidate_pages.append(i)
            confirmed_set.add(i)

    return sorted(candidate_pages)


# =========================================================
# EXTRACT OFFICIAL QUESTIONS — scans across MULTIPLE pages
# =========================================================

def extract_official_questions_multi_page(pages: list, qp_page_indices: list) -> list:
    """
    Extracts numbered questions across all detected question-paper pages.
    Handles standard numbered questions, multi-line continuations, and
    lettered sub-parts (Latin a-d / Devanagari क-घ).
    """
    all_questions = []
    pending_parent = None

    Q_START   = re.compile(r'^\s*(\d+)[\.\)]\s+(.+)')
    SUB_LATIN = re.compile(r'^\s*\(?([a-d])\)\s*(.+)', re.IGNORECASE)
    SUB_DEVA  = re.compile(r'^\s*\(?([क-घ])\)\s*(.+)')
    SKIP      = re.compile(r'^#+\s|^भाग|^PART|^\s*$')

    for page_idx in qp_page_indices:
        lines   = pages[page_idx]["raw_text"].split("\n")
        current = None

        for line in lines:
            stripped = line.strip()

            if not stripped or SKIP.match(stripped):
                if current:
                    all_questions.append({"text": current.strip(), "parent": None})
                    current = None
                continue

            sub_m = SUB_LATIN.match(stripped) or SUB_DEVA.match(stripped)
            if sub_m:
                if current:
                    all_questions.append({"text": current.strip(), "parent": None})
                    current = None
                label = sub_m.group(1)
                body  = sub_m.group(2)
                parent_text = pending_parent if pending_parent else ""
                combined = f"{parent_text} {label}) {body}".strip()
                all_questions.append({"text": combined, "parent": pending_parent})
                continue

            m = Q_START.match(stripped)
            if m:
                if current:
                    all_questions.append({"text": current.strip(), "parent": None})
                current = stripped
                if re.search(r'(?:लिखिए|following|:)\s*$', stripped, re.IGNORECASE):
                    pending_parent = stripped
                else:
                    pending_parent = None
                continue

            if current:
                current += " " + stripped

        if current:
            all_questions.append({"text": current.strip(), "parent": None})

    # Split questions that have 2+ inline sub-parts
    final_questions = []
    SUBPART_RE = re.compile(r'(?:^|\s)\(?([a-zक-घ])\)\s', re.UNICODE)

    for q in all_questions:
        text = q["text"]
        matches = list(SUBPART_RE.finditer(text))

        if len(matches) >= 2:
            preamble = text[:matches[0].start()].strip()
            for idx, m in enumerate(matches):
                start = m.start(1)
                end   = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
                part_text = text[start:end].strip()
                full_q = f"{preamble} {part_text}".strip() if preamble else part_text
                final_questions.append(full_q)
        else:
            final_questions.append(text)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in final_questions:
        key = re.sub(r'\s+', ' ', q.lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(q)

    return unique


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
# FIND QUESTION BOUNDARIES IN ANSWER PAGES — similarity based
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
    """Strip leading numbering like '1.', 'Q1.', 'प्र.2', '20.', 'a)', '(a)', 'क)', '(क)' etc."""
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
    """
    Scans answer lines for restated questions matching official questions.
    Uses sliding window to join multi-line question restatements.
    Enforces that boundaries appear in the SAME ORDER as the official
    questions list.
    """
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
    """
    For each boundary, answer = raw lines starting AFTER the full matched
    question span, up to the next boundary. Pure text slicing, zero LLM.
    """
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

# =========================================================
# OPENAI-BASED Q&A LINE DETECTION
#
# OpenAI's ONLY job: read numbered lines and report which line
# numbers correspond to question starts and answer starts.
# It NEVER outputs question text or answer text — only integers.
# This makes it layout-agnostic (works on any PDF format) while
# remaining hallucination-safe, because every line number it
# returns is validated against the real document before use.
# =========================================================

def get_openai_client():
    from openai import OpenAI
    key = get_api_key("OPENAI_API_KEY")
    if not key:
        raise Exception("OPENAI_API_KEY not found")
    return OpenAI(api_key=key)


def build_numbered_line_dump(pages: list) -> list:
    """
    Flattens the whole document into a single list of lines,
    each tagged with its global line number and source page number.
    """
    line_index = []
    for page in pages:
        for line in page["raw_text"].split("\n"):
            line_index.append({
                "line_number": len(line_index),
                "page_number": page["page_number"],
                "text": line
            })
    return line_index


def ask_openai_for_qa_lines(line_index: list, status_callback=None, chunk_size: int = 350):
    """
    Sends the numbered line dump to OpenAI in chunks and asks ONLY
    for line numbers — never content.
    Returns a list of {question_id, question_start_line, answer_start_line}.
    Raises on any API/parsing failure so the caller can fall back.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    client = get_openai_client()
    total = len(line_index)
    all_results = []

    num_chunks = (total + chunk_size - 1) // chunk_size

    for ci in range(num_chunks):
        start = ci * chunk_size
        end   = min(start + chunk_size, total)
        chunk = line_index[start:end]

        numbered_text = "\n".join(
            f"[L{e['line_number']}] {e['text']}" for e in chunk
        )

        log(f"OpenAI scanning lines {start}-{end} (chunk {ci+1}/{num_chunks})...")

        prompt = f"""You are scanning OCR text from a scanned exam answer booklet.
The document may be in ANY language (Hindi, English, mixed) and the
layout varies between documents — do not assume a fixed structure.

Each line below is tagged with its line number like [L42].

Find every QUESTION and the START of its ANSWER in this chunk.
A question is a numbered/lettered exam prompt the student must answer
(e.g. "1.", "Q.1", "क)", "(a)", "9.", roman numerals, etc — in any
language). The answer is the student's handwritten response that
follows, often after a marker like "Ans-", "उत्तर-", "A.1-", or with
no marker at all (answer just starts on the next line).

Return ONLY this JSON, nothing else:
{{
  "items": [
    {{
      "question_id": "<a short label for this question, e.g. Q1, Q9-a>",
      "question_start_line": <integer line number where the question starts>,
      "answer_start_line": <integer line number where the answer starts, or null if not found in this chunk>
    }}
  ]
}}

If no questions are found in this chunk, return {{"items": []}}.
Do NOT include question text or answer text in your response — line numbers only.

NUMBERED LINES:
{numbered_text}"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You identify line numbers only. Never output document content. Return valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )

        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)
        items = data.get("items", [])
        log(f"  Chunk {ci+1}: OpenAI reported {len(items)} item(s)")
        all_results.extend(items)

    return all_results


def validate_and_clean_llm_items(raw_items: list, line_index: list, status_callback=None) -> list:
    """
    Validates every LLM-reported item against the real document.
    Drops anything that:
    - has an out-of-range line number
    - has a question_start_line >= answer_start_line (nonsensical)
    - points to an empty/too-short line for the question start
    - duplicates a question_start_line already accepted
    Enforces overall increasing order of question_start_line.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    total_lines = len(line_index)
    cleaned = []
    seen_start_lines = set()

    raw_items_sorted = sorted(
        [it for it in raw_items if isinstance(it.get("question_start_line"), int)],
        key=lambda it: it["question_start_line"]
    )

    last_start = -1
    for it in raw_items_sorted:
        qsl = it.get("question_start_line")
        asl = it.get("answer_start_line")

        if qsl is None or not (0 <= qsl < total_lines):
            continue
        if qsl in seen_start_lines:
            continue
        if qsl <= last_start:
            continue

        q_line_text = line_index[qsl]["text"].strip()
        if len(q_line_text) < 2:
            continue

        if asl is not None:
            if not isinstance(asl, int) or not (0 <= asl < total_lines) or asl <= qsl:
                asl = None

        cleaned.append({
            "question_id":        it.get("question_id", f"Q{len(cleaned)+1}"),
            "question_start_line": qsl,
            "answer_start_line":   asl
        })
        seen_start_lines.add(qsl)
        last_start = qsl

    log(f"Validation: {len(cleaned)} of {len(raw_items)} OpenAI items passed checks")
    return cleaned


def slice_qa_from_line_items(line_index: list, items: list) -> list:
    """
    Pure text slicing — zero LLM. Given validated {question_start_line,
    answer_start_line} pairs, slices the question and answer directly
    from the raw OCR line index.
    """
    qa_pairs = []

    for i, item in enumerate(items):
        q_start = item["question_start_line"]
        a_start = item["answer_start_line"]
        if a_start is None:
            a_start = q_start + 1

        a_end = items[i + 1]["question_start_line"] if i + 1 < len(items) else len(line_index)

        q_lines = [
            line_index[j]["text"] for j in range(q_start, min(a_start, len(line_index)))
            if line_index[j]["text"].strip()
        ]
        a_lines = [
            line_index[j]["text"] for j in range(a_start, max(a_end, a_start))
            if line_index[j]["text"].strip() and not is_noise(line_index[j]["text"])
        ]

        qa_pairs.append({
            "question": " ".join(q_lines).strip(),
            "answer":   " ".join(a_lines).strip()
        })

    return qa_pairs


def try_openai_pipeline(pages: list, status_callback=None):
    """
    Attempts the OpenAI line-identification pipeline end to end.
    Returns (qa_pairs, None) on success, or (None, reason_string)
    if anything fails/looks unreliable.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    key_present = bool(get_api_key("OPENAI_API_KEY"))
    log(f"OPENAI_API_KEY found in secrets/env: {key_present}")
    if not key_present:
        return None, "OPENAI_API_KEY is not set in Streamlit secrets or environment"

    try:
        line_index = build_numbered_line_dump(pages)
        log(f"Built line index: {len(line_index)} total lines")

        raw_items = ask_openai_for_qa_lines(line_index, status_callback)
        if not raw_items:
            return None, "OpenAI returned zero items across all chunks"

        cleaned = validate_and_clean_llm_items(raw_items, line_index, status_callback)
        if len(cleaned) < 2:
            return None, f"Only {len(cleaned)} of {len(raw_items)} OpenAI items passed validation (need >=2)"

        qa_pairs = slice_qa_from_line_items(line_index, cleaned)
        non_empty = [p for p in qa_pairs if p["answer"].strip()]
        if len(non_empty) < len(qa_pairs) * 0.5:
            return None, f"Only {len(non_empty)} of {len(qa_pairs)} sliced answers were non-empty"

        log(f"OpenAI pipeline succeeded — {len(qa_pairs)} Q-A pairs")
        return qa_pairs, None

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log(f"OpenAI pipeline exception: {type(e).__name__}: {e}")
        log(tb)
        return None, f"{type(e).__name__}: {e}"


def run_regex_pipeline(pages: list, status_callback=None):
    """
    The original regex/similarity-based pipeline, used as a fallback
    when the OpenAI pipeline is unavailable or produces unreliable results.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    qp_page_indices = find_question_paper_pages(pages)
    log(f"[fallback] Question paper pages detected: {[p+1 for p in qp_page_indices] if qp_page_indices else 'none'}")

    if not qp_page_indices:
        raise Exception(
            "Could not detect any question paper pages in this document, "
            "and the OpenAI-based detection also failed.\n"
            f"Page 1 preview:\n{pages[0]['raw_text'][:500]}"
        )

    official_questions = extract_official_questions_multi_page(pages, qp_page_indices)
    log(f"[fallback] Official questions extracted: {len(official_questions)}")

    if not official_questions:
        raise Exception(
            "Question paper pages were found, but no questions could be parsed from them.\n"
            f"Detected pages: {[p+1 for p in qp_page_indices]}"
        )

    answer_page_indices = [i for i in range(len(pages)) if i not in qp_page_indices]
    answer_pages = [pages[i] for i in answer_page_indices]
    log(f"[fallback] Answer pages: {[i+1 for i in answer_page_indices]}")

    answer_lines = []
    for page in answer_pages:
        for line in page["raw_text"].split("\n"):
            if not is_noise(line):
                answer_lines.append(line)

    log(f"[fallback] Flattened {len(answer_lines)} answer lines")

    boundaries = find_question_boundaries_by_similarity(answer_lines, official_questions)
    log(f"[fallback] Matched {len(boundaries)} of {len(official_questions)} questions")

    matched_qs = {b["question"] for b in boundaries}
    for q in official_questions:
        if q not in matched_qs:
            log(f"[fallback] WARNING: No match found for: {q[:60]}")

    if not boundaries:
        raise Exception(
            "Could not match any questions in answer pages (fallback also failed).\n"
            f"Official questions: {official_questions}"
        )

    qa_pairs = slice_raw_answers_by_boundaries(answer_lines, boundaries)
    return qa_pairs


def process_pdf(file_input, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    if isinstance(file_input, (str, Path)):
        file_bytes = Path(file_input).read_bytes()
        file_name  = Path(file_input).name
    else:
        file_bytes = file_input.read()
        file_name  = getattr(file_input, "name", "document.pdf")

    # Step 1: OCR — Tesseract (local, no API call)
    pages = run_ocr(file_bytes, file_name, status_callback)

    # Step 2: Build OCR JSON
    log("Building OCR JSON...")
    ocr_json = build_ocr_json(pages)
    log(f"Total pages: {ocr_json['total_pages']}")

    # Step 3: Try OpenAI-based line identification first
    log("Attempting OpenAI-based question/answer line detection...")
    qa_pairs, openai_fail_reason = try_openai_pipeline(pages, status_callback)

    if qa_pairs is not None:
        log(f"Done — {len(qa_pairs)} Q-A pairs (via OpenAI)")
        return ocr_json, qa_pairs

    log(f"OpenAI pipeline did not produce results. Reason: {openai_fail_reason}")
    log("Falling back to regex/similarity pipeline...")

    try:
        qa_pairs = run_regex_pipeline(pages, status_callback)
    except Exception as regex_error:
        raise Exception(
            f"Both detection pipelines failed.\n\n"
            f"OpenAI pipeline failed because: {openai_fail_reason}\n\n"
            f"Regex fallback failed because: {regex_error}"
        )

    log(f"Done — {len(qa_pairs)} Q-A pairs (via regex fallback)")
    return ocr_json, qa_pairs
