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
# OCR — Datalab (Chandra model) via /convert endpoint
#
# Datalab's /convert is async: submit -> poll request_check_url
# until status == "complete". paginate=True returns markdown with
# page-break markers so we can split back into per-page text,
# matching the page-based structure the rest of the pipeline needs.
# =========================================================

DATALAB_BASE_URL = "https://www.datalab.to"

# Marker for page breaks when paginate=True
# Datalab inserts a horizontal rule with the page number between pages.
PAGE_BREAK_RE = re.compile(
    r'\n?-{3,}\s*\n+\s*\{(\d+)\}-{3,}\s*\n?|\n?\{(\d+)\}-{3,}\s*\n?',
)


def _split_paginated_markdown(markdown: str, total_pages_hint: int = None) -> list:
    """
    Datalab paginated markdown separates pages with a horizontal rule
    containing the page number, e.g.:
        page 1 content
        ------- Page 1 -------
        page 2 content
    Exact format can vary slightly by version, so we fall back to a
    generic split on form-feed / page-marker patterns, and if no
    markers are found at all, return the whole text as one page.
    """
    # Try splitting on common Datalab page break patterns
    generic_break = re.compile(r'\n-{3,}\s*Page\s*\d+\s*-{3,}\n', re.IGNORECASE)

    parts = generic_break.split(markdown)
    if len(parts) > 1:
        return [p.strip() for p in parts]

    # Fallback: no recognizable page breaks — return as single block
    return [markdown.strip()]


def run_ocr(file_content: bytes, file_name: str, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    api_key = get_api_key("DATALAB_API_KEY")
    if not api_key:
        raise Exception("DATALAB_API_KEY not found in secrets or environment")

    # Guard against oversized uploads — fail with a clear message
    # instead of a cryptic Cloudflare 413 HTML page.
    size_mb = len(file_content) / (1024 * 1024)
    MAX_MB  = 45   # conservative margin under typical Cloudflare limits
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
            "mode": "accurate",     # highest accuracy — best for handwriting
            "paginate": "true"      # keep page boundaries in the output
        },
        timeout=120
    )

    if resp.status_code != 200:
        raise Exception(f"Datalab submit error {resp.status_code}: {resp.text}")

    data = resp.json()

    if not data.get("success", True):
        raise Exception(f"Datalab submit failed: {data.get('error')}")

    check_url = data["request_check_url"]
    log("Document submitted — polling for OCR result...")

    # ── Poll until complete ─────────────────────────────────
    max_polls = 150          # ~150 * 2s = 5 minutes max wait
    poll_interval = 2

    result = None
    for attempt in range(max_polls):
        poll_resp = httpx.get(check_url, headers=headers, timeout=60)

        if poll_resp.status_code != 200:
            raise Exception(f"Datalab poll error {poll_resp.status_code}: {poll_resp.text}")

        result = poll_resp.json()
        status = result.get("status")

        if status == "complete":
            log("OCR complete — parsing pages...")
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
    page_texts = _split_paginated_markdown(markdown, page_count_hint)

    pages = []
    for idx, text in enumerate(page_texts):
        pages.append({
            "page_number": idx + 1,
            "raw_text":    text
        })

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
# REFERENCE BOOK OCR — Datalab handles full multi-page PDFs
# natively, so no manual page-splitting is needed here.
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
# GROQ-BASED Q&A LINE DETECTION
#
# Groq's ONLY job: read numbered lines and report which line
# numbers correspond to question starts and answer starts.
# It NEVER outputs question text or answer text — only integers.
# This makes it layout-agnostic (works on any PDF format) while
# remaining hallucination-safe, because every line number it
# returns is validated against the real document before use:
#   - must be a real, in-range line index
#   - must be in increasing order
#   - the line at that index must actually look like the kind
#     of content claimed (a question line / a non-empty line)
# If validation fails for too many entries, or Groq is unavailable,
# the caller falls back to the regex/similarity pipeline.
# =========================================================

def get_groq_client():
    from groq import Groq
    key = get_api_key("GROQ_API_KEY")
    if not key:
        raise Exception("GROQ_API_KEY not found")
    return Groq(api_key=key)


def build_numbered_line_dump(pages: list) -> list:
    """
    Flattens the whole document into a single list of lines,
    each tagged with its global line number and source page number.
    This numbering is what gets shown to Groq and is the same
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


def ask_groq_for_qa_lines(line_index: list, status_callback=None, chunk_size: int = 350):
    """
    Sends the numbered line dump to Groq in chunks (to stay under
    token limits) and asks ONLY for line numbers — never content.
    Returns a list of {question_id, question_start_line, answer_start_line}.
    Raises on any API/parsing failure so the caller can fall back.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    groq = get_groq_client()
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

        log(f"Groq scanning lines {start}-{end} (chunk {ci+1}/{num_chunks})...")

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

        resp = groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
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
        log(f"  Chunk {ci+1}: Groq reported {len(items)} item(s)")
        all_results.extend(items)

    return all_results


def validate_and_clean_groq_items(raw_items: list, line_index: list, status_callback=None) -> list:
    """
    Validates every Groq-reported item against the real document.
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

    log(f"Validation: {len(cleaned)} of {len(raw_items)} Groq items passed checks")
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


def try_groq_pipeline(pages: list, status_callback=None):
    """
    Attempts the Groq line-identification pipeline end to end.
    Returns qa_pairs on success, or None if anything fails/looks
    unreliable — signalling the caller to fall back to regex.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    try:
        line_index = build_numbered_line_dump(pages)
        log(f"Built line index: {len(line_index)} total lines")

        raw_items = ask_groq_for_qa_lines(line_index, status_callback)
        if not raw_items:
            log("Groq returned no items — falling back")
            return None

        cleaned = validate_and_clean_groq_items(raw_items, line_index, status_callback)
        if len(cleaned) < 2:
            log("Too few validated items from Groq — falling back")
            return None

        qa_pairs = slice_qa_from_line_items(line_index, cleaned)
        non_empty = [p for p in qa_pairs if p["answer"].strip()]
        if len(non_empty) < len(qa_pairs) * 0.5:
            log("More than half the answers came out empty — falling back")
            return None

        log(f"Groq pipeline succeeded — {len(qa_pairs)} Q-A pairs")
        return qa_pairs

    except Exception as e:
        log(f"Groq pipeline failed ({e}) — falling back to regex pipeline")
        return None


def run_regex_pipeline(pages: list, status_callback=None):
    """
    The original regex/similarity-based pipeline, used as a fallback
    when the Groq pipeline is unavailable or produces unreliable results.
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
            "and the Groq-based detection also failed.\n"
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

    # Step 3: Try Groq-based line identification first — layout-agnostic,
    # works regardless of where the question paper appears in the PDF.
    # Groq only ever returns line numbers; all text is sliced by Python
    # from the raw OCR afterward, so answer content is never LLM-touched.
    log("Attempting Groq-based question/answer line detection...")
    qa_pairs = try_groq_pipeline(pages, status_callback)

    if qa_pairs is None:
        log("Falling back to regex/similarity pipeline...")
        qa_pairs = run_regex_pipeline(pages, status_callback)

    log(f"Done — {len(qa_pairs)} Q-A pairs")
    return ocr_json, qa_pairs

