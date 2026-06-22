import os
import io
import re
import json
import fitz
from pathlib import Path
from PIL import Image
import pytesseract

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
# OCR — Tesseract (local, free, no rate limits)
#
# Rasterizes each PDF page to an image, then runs Tesseract OCR
# on it directly. Supports multiple languages via the `lang`
# parameter (e.g. "eng+hin" for English+Hindi mixed documents).
# Tesseract must have the relevant language packs installed
# (tesseract-ocr-hin for Hindi, etc) — see requirements/setup notes.
# =========================================================

# Default language set — covers English + Hindi (Devanagari).
# Extend this string (e.g. "eng+hin+tam") if other languages appear
# in your documents; Tesseract language packs must be installed
# for any code added here.
TESSERACT_LANGS = os.environ.get("TESSERACT_LANGS", "eng+hin")


def run_ocr(file_content: bytes, file_name: str, status_callback=None, dpi: int = 300):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    log(f"Running local Tesseract OCR ({TESSERACT_LANGS})...")

    src_doc     = fitz.open(stream=file_content, filetype="pdf")
    total_pages = len(src_doc)
    log(f"Document has {total_pages} page(s)")

    pages = []

    for page_num in range(total_pages):
        log(f"OCR: page {page_num + 1} of {total_pages}...")

        page = src_doc[page_num]
        pix  = page.get_pixmap(dpi=dpi)

        img_bytes = pix.tobytes("png")
        img       = Image.open(io.BytesIO(img_bytes))

        try:
            text = pytesseract.image_to_string(img, lang=TESSERACT_LANGS)
        except Exception as e:
            log(f"  Page {page_num + 1} OCR failed: {e}")
            text = ""

        pages.append({
            "page_number": page_num + 1,
            "raw_text":    text
        })

    src_doc.close()

    log(f"OCR done — {len(pages)} page(s) extracted")
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
# REFERENCE BOOK OCR — same Tesseract pipeline, full document
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
    weak_pages = []   # pages with question-like lines but no strong signal — possible continuations

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
            continue   # ID card / registration page — never a question paper

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
            # No strong signal on its own — could be a continuation page
            # (e.g. a parent question's lettered sub-parts spilling onto
            # the next page). Only counts if adjacent to a confirmed page.
            weak_pages.append(i)

    # Promote weak pages that are immediately adjacent to a confirmed
    # question-paper page (continuation of the same question paper),
    # rather than requiring every single page to repeat the strong signal.
    confirmed_set = set(candidate_pages)
    for i in weak_pages:
        if (i - 1) in confirmed_set or (i + 1) in confirmed_set:
            candidate_pages.append(i)
            confirmed_set.add(i)

    return sorted(candidate_pages)


# =========================================================
# EXTRACT OFFICIAL QUESTIONS — scans across MULTIPLE pages
# Also captures lettered sub-questions (क/ख/ग/घ, a/b/c/d)
# that appear as a standalone list after a parent question
# like "Q.9 निम्नलिखित पर टिप्पणी लिखिए" on a DIFFERENT page
# than where the sub-options are printed.
# =========================================================

def extract_official_questions_multi_page(pages: list, qp_page_indices: list) -> list:
    """
    Extracts numbered questions across all detected question-paper pages,
    in page order. Handles:
    - Standard numbered questions: "1. text"
    - Multi-line questions (continuation lines joined)
    - Lettered sub-parts within a question: a) b) c) / क) ख) ग) घ)
    - Sub-parts that appear on a later page than their parent question
      (common when "Q.9 Write notes on:" is followed by a), b), c), d)
      printed on the next page)
    """
    all_questions = []
    pending_parent = None   # holds a parent question awaiting sub-parts from next page

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

            # Lettered sub-part (Latin a-d or Devanagari क-घ)
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
                # Track this as a potential parent for sub-parts on a later page
                # (e.g. ends with "टिप्पणी लिखिए" / "following" / colon)
                if re.search(r'(?:लिखिए|following|:)\s*$', stripped, re.IGNORECASE):
                    pending_parent = stripped
                else:
                    pending_parent = None
                continue

            if current:
                current += " " + stripped

        if current:
            all_questions.append({"text": current.strip(), "parent": None})

    # Now split any question that has 2+ inline sub-parts (a)/b)/c) on one line block)
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
# Works for Hindi, English, any language
# Matches student-written question restatements against the
# official question paper text using word-overlap similarity
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

    Correctness guarantees:
    1. Tracks how many lines (`span`) the matched question text occupies,
       so the answer slice can start AFTER the full question text.
    2. Enforces that boundaries appear in the SAME ORDER as the official
       questions list. If a question's best-scoring candidate would break
       order (e.g. a false-positive shares vocabulary with an earlier
       question), the NEXT best-scoring candidate for that same question
       is tried, and so on, rather than dropping the question entirely.
    """
    candidates_by_question = {}   # question -> list of candidates, sorted by score desc

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

    # Sort each question's candidates by score, descending
    for q in candidates_by_question:
        candidates_by_question[q].sort(key=lambda c: -c["score"])

    # Walk questions in official order. For each, try candidates from
    # highest score downward, accepting the first one that comes after
    # the previously accepted boundary's line_index.
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
    question span (boundary["span"] lines), up to the next boundary.
    Pure text slicing, zero LLM.
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
# GEMINI-BASED Q&A LINE DETECTION
#
# Gemini's ONLY job: read numbered lines and report which line
# numbers correspond to question starts and answer starts.
# It NEVER outputs question text or answer text — only integers.
# This makes it layout-agnostic (works on any PDF format) while
# remaining hallucination-safe, because every line number it
# returns is validated against the real document before use:
#   - must be a real, in-range line index
#   - must be in increasing order
#   - the line at that index must actually look like the kind
#     of content claimed (a question line / a non-empty line)
# If validation fails for too many entries, or Gemini is unavailable,
# the caller falls back to the regex/similarity pipeline.
# =========================================================

def get_gemini_client():
    from google import genai
    key = get_api_key("GEMINI_API_KEY")
    if not key:
        raise Exception("GEMINI_API_KEY not found")
    return genai.Client(api_key=key)


def build_numbered_line_dump(pages: list) -> list:
    """
    Flattens the whole document into a single list of lines,
    each tagged with its global line number and source page number.
    This numbering is what gets shown to Gemini and is the same
    numbering used afterward for validation and slicing.
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


def ask_llm_for_qa_lines(line_index: list, status_callback=None, chunk_size: int = 350):
    """
    Sends the numbered line dump to Gemini in chunks (to stay under
    token limits) and asks ONLY for line numbers — never content.
    Returns a list of {question_id, question_start_line, answer_start_line}.
    Raises on any API/parsing failure so the caller can fall back.
    """
    from google.genai import types

    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    client = get_gemini_client()
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

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

        log(f"Gemini scanning lines {start}-{end} (chunk {ci+1}/{num_chunks})...")

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

        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=2000,
                response_mime_type="application/json",
                system_instruction="You identify line numbers only. Never output document content. Return valid JSON only."
            )
        )

        raw  = (resp.text or "").strip()
        data = json.loads(raw)
        items = data.get("items", [])
        log(f"  Chunk {ci+1}: Gemini reported {len(items)} item(s)")
        all_results.extend(items)

    return all_results


def validate_and_clean_llm_items(raw_items: list, line_index: list, status_callback=None) -> list:
    """
    Validates every Gemini-reported item against the real document.
    Drops anything that:
    - has an out-of-range line number
    - has a question_start_line >= answer_start_line (nonsensical)
    - points to an empty/too-short line for the question start
    - duplicates a question_start_line already accepted
    Also enforces overall increasing order of question_start_line,
    dropping any item that goes backwards (same safeguard as the
    regex pipeline's order enforcement).
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    total_lines = len(line_index)
    cleaned = []
    seen_start_lines = set()

    # Sort by question_start_line first so order-enforcement is meaningful
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
            continue   # out of order — likely hallucinated, skip

        q_line_text = line_index[qsl]["text"].strip()
        if len(q_line_text) < 2:
            continue   # points to a blank/near-empty line — not a real question

        # Validate answer_start_line if provided
        if asl is not None:
            if not isinstance(asl, int) or not (0 <= asl < total_lines) or asl <= qsl:
                asl = None   # invalid — will be treated as "not found", filled in later

        cleaned.append({
            "question_id":        it.get("question_id", f"Q{len(cleaned)+1}"),
            "question_start_line": qsl,
            "answer_start_line":   asl
        })
        seen_start_lines.add(qsl)
        last_start = qsl

    log(f"Validation: {len(cleaned)} of {len(raw_items)} Gemini items passed checks")
    return cleaned


def slice_qa_from_line_items(line_index: list, items: list) -> list:
    """
    Pure text slicing — zero LLM. Given validated {question_start_line,
    answer_start_line} pairs, slices the question and answer directly
    from the raw OCR line index. If answer_start_line is missing for
    an item, defaults to question_start_line + 1 (next line).
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


def try_llm_pipeline(pages: list, status_callback=None):
    """
    Attempts the Gemini line-identification pipeline end to end.
    Returns (qa_pairs, None) on success, or (None, reason_string)
    if anything fails/looks unreliable — signalling the caller to
    fall back to regex, with a human-readable reason.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    # Diagnostic: confirm the key is actually visible BEFORE attempting
    # anything else, so a missing-secret problem is obvious immediately.
    key_present = bool(get_api_key("GEMINI_API_KEY"))
    log(f"GEMINI_API_KEY found in secrets/env: {key_present}")
    if not key_present:
        return None, "GEMINI_API_KEY is not set in Streamlit secrets or environment"

    try:
        line_index = build_numbered_line_dump(pages)
        log(f"Built line index: {len(line_index)} total lines")

        raw_items = ask_llm_for_qa_lines(line_index, status_callback)
        if not raw_items:
            return None, "Gemini returned zero items across all chunks"

        cleaned = validate_and_clean_llm_items(raw_items, line_index, status_callback)
        if len(cleaned) < 2:
            return None, f"Only {len(cleaned)} of {len(raw_items)} Gemini items passed validation (need >=2)"

        qa_pairs = slice_qa_from_line_items(line_index, cleaned)
        non_empty = [p for p in qa_pairs if p["answer"].strip()]
        if len(non_empty) < len(qa_pairs) * 0.5:
            return None, f"Only {len(non_empty)} of {len(qa_pairs)} sliced answers were non-empty"

        log(f"Gemini pipeline succeeded — {len(qa_pairs)} Q-A pairs")
        return qa_pairs, None

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log(f"Gemini pipeline exception: {type(e).__name__}: {e}")
        log(tb)
        return None, f"{type(e).__name__}: {e}"


def run_regex_pipeline(pages: list, status_callback=None):
    """
    The original regex/similarity-based pipeline, used as a fallback
    when the Gemini pipeline is unavailable or produces unreliable results.
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
            "and the Gemini-based detection also failed.\n"
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

    # Step 1: OCR — send the original PDF directly.
    pages = run_ocr(file_bytes, file_name, status_callback)

    # Step 2: Build OCR JSON
    log("Building OCR JSON...")
    ocr_json = build_ocr_json(pages)
    log(f"Total pages: {ocr_json['total_pages']}")

    # Step 3: Try Gemini-based line identification first — layout-agnostic,
    # works regardless of where the question paper appears in the PDF.
    # Gemini only ever returns line numbers; all text is sliced by Python
    # from the raw OCR afterward, so answer content is never LLM-touched.
    log("Attempting Gemini-based question/answer line detection...")
    qa_pairs, llm_fail_reason = try_llm_pipeline(pages, status_callback)

    if qa_pairs is not None:
        log(f"Done — {len(qa_pairs)} Q-A pairs (via Gemini)")
        return ocr_json, qa_pairs

    log(f"Gemini pipeline did not produce results. Reason: {llm_fail_reason}")
    log("Falling back to regex/similarity pipeline...")

    try:
        qa_pairs = run_regex_pipeline(pages, status_callback)
    except Exception as regex_error:
        raise Exception(
            f"Both detection pipelines failed.\n\n"
            f"Gemini pipeline failed because: {llm_fail_reason}\n\n"
            f"Regex fallback failed because: {regex_error}"
        )

    log(f"Done — {len(qa_pairs)} Q-A pairs (via regex fallback)")
    return ocr_json, qa_pairs
