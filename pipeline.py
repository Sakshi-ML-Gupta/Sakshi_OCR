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
# INPUT NORMALIZATION
# Accepts str/Path, bytes, file-like objects, or (filename, bytes)
# / (filename, bytes, content_type) tuples without crashing.
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

# FIX (round 3): a single page-break regex is too brittle -- different
# documents have produced different marker shapes in practice (this is
# why some real uploads above collapsed to "1 page" even though they
# were clearly multi-page assignments). We now try several known marker
# shapes and prefer whichever split result matches Datalab's own
# page_count, if provided. We also log enough diagnostic info that if
# a brand new marker shape appears, it's immediately visible in the
# logs instead of silently collapsing to 1 page again.
PAGE_BREAK_PATTERNS = [
    re.compile(r'\n?\{(\d+)\}-{3,}\n?'),                            # {0}------------------
    re.compile(r'\n?-{2,}\{(\d+)\}-{2,}\n?'),                       # ---{0}---
    re.compile(r'\n-{3,}\s*Page\s*\d+\s*-{3,}\n', re.IGNORECASE),   # ----- Page 1 -----
    re.compile(r'\n\s*\[PAGE\s*(\d+)\]\s*\n', re.IGNORECASE),       # [PAGE 1]
    re.compile(r'\n\s*<!--\s*page\s*(\d+)\s*-->\s*\n', re.IGNORECASE),  # <!-- page 1 -->
]


def _split_paginated_markdown(markdown: str, page_count_hint: int = None, log=print) -> list:
    """
    Tries every known Datalab page-break marker shape. If multiple
    patterns produce a valid split, prefers the one whose page count
    matches Datalab's own page_count_hint (when available); otherwise
    takes whichever pattern produced the most pages (more pages found
    is generally more correct than fewer, since failing to split at
    all is the known failure mode).
    """
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

    # Form-feed fallback
    if '\f' in markdown:
        parts = [p.strip() for p in markdown.split('\f') if p.strip()]
        if len(parts) > 1:
            return parts

    # Nothing worked -- log diagnostics so the NEXT failure is fixable
    # in one look instead of needing another round of guessing.
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

    # Extra diagnostic: if we collapsed to 1 page on a sizeable file,
    # warn loudly so it's obvious in logs that the split failed rather
    # than the document genuinely being one page.
    if len(pages) == 1 and size_mb > 1.0:
        log(
            f"WARNING: Only 1 page extracted from a {size_mb:.1f}MB file. "
            f"This usually means the page-break marker format was not "
            f"recognized. Markdown length: {len(markdown)} chars."
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
# DETECT QUESTION PAPER PAGE
# =========================================================

def find_question_paper_pages(pages: list, min_questions: int = 2) -> list:
    Q_LINE_NUM   = re.compile(r'^\s*[0-9]+[\.\)]\s+.{15,}')
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
    weak_pages = []

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
            continue

        if ANSWER_PAGE_MARKERS.search(text):
            continue

        has_strong_signal = bool(
            MARK_ALLOCATION.search(text)
            or SECTION_HEADER.search(text)
            or EXAM_INSTRUCTION.search(text)
        )

        if has_strong_signal:
            candidate_pages.append(i)
        else:
            weak_pages.append(i)

    confirmed_set = set(candidate_pages)
    for i in weak_pages:
        if (i - 1) in confirmed_set or (i + 1) in confirmed_set:
            candidate_pages.append(i)
            confirmed_set.add(i)

    return sorted(candidate_pages)


# =========================================================
# FIX (round 3): instruction-line filter
#
# Numbered lines that tell the student HOW to answer (word counts,
# "answer the following questions", marks distribution instructions)
# were being extracted as if they were real question content. They
# can never be matched against an answer because students don't
# restate instructions -- they restate the actual question. This
# filter removes them from the official question list before matching
# is ever attempted, so they stop showing up as permanent false
# "no match found" warnings and stop occupying a "question slot" that
# a real question should have filled.
# =========================================================

INSTRUCTION_LINE_RE = re.compile(
    r'(?:answer\s+(?:the\s+)?(?:following|all)\s+questions'
    r'|in\s+about\s+\d+\s+words'
    r'|each\s+question\s+carries'
    r'|attempt\s+any'
    r'|all\s+questions\s+are\s+compulsory'
    r'|सभी\s*प्रश्न\s*अनिवार्य'
    r'|किसी\s*भी.*?उत्तर\s*दें'
    r'|शब्दों\s*में\s*उत्तर)',
    re.IGNORECASE
)


def is_instruction_line(text: str) -> bool:
    """True if text is a meta-instruction about HOW to answer, not an
    actual question to be answered."""
    return bool(INSTRUCTION_LINE_RE.search(text))


# =========================================================
# EXTRACT OFFICIAL QUESTIONS -- scans across MULTIPLE pages
#
# FIX (round 3): Q_START is now restricted to ASCII digits [0-9]
# only. Devanagari numerals (१,२,३...) were previously being matched
# by \d (Unicode-aware by default), which caused numbered sub-headings
# inside topic lists / long answers (e.g. "१. द्वंद्व समास") to be
# extracted as if they were standalone exam questions. Real question
# papers in this pipeline's domain (IGNOU-style assignments) number
# questions with Western numerals even in Hindi-medium papers, so this
# restriction removes a major source of phantom "questions" that could
# never be matched to a real student answer.
# =========================================================

def extract_official_questions_multi_page(pages: list, qp_page_indices: list) -> list:
    all_questions = []
    pending_parent = None

    Q_START   = re.compile(r'^\s*([0-9]+)[\.\)]\s+(.+)')
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

    final_questions = []
    SUBPART_RE = re.compile(r'(?:^|\s)\(?([a-z])\)\s', re.UNICODE)
    SUBPART_DEVA_RE = re.compile(r'(?:^|\s)\(?([क-घ])\)\s', re.UNICODE)

    for q in all_questions:
        text = q["text"]
        matches = sorted(
            list(SUBPART_RE.finditer(text)) + list(SUBPART_DEVA_RE.finditer(text)),
            key=lambda m: m.start()
        )

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

    # FIX: drop instruction lines before dedup/return -- they are never
    # real questions and can never be matched to a student answer.
    final_questions = [q for q in final_questions if not is_instruction_line(q)]

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
# FIND QUESTION BOUNDARIES IN ANSWER PAGES -- similarity based
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


# =========================================================
# COMPLETE PIPELINE
# =========================================================

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

    qp_page_indices = find_question_paper_pages(pages)
    log(f"Question paper pages detected: {[p+1 for p in qp_page_indices] if qp_page_indices else 'none'}")

    if not qp_page_indices:
        raise Exception(
            "Could not detect any question paper pages in this document.\n"
            "This usually means the document has a different layout than expected, "
            "OR the OCR output collapsed to fewer pages than expected (check the "
            "logs above for a 'WARNING: No page-break marker recognized' message).\n"
            f"Page 1 preview:\n{pages[0]['raw_text'][:500]}"
        )

    official_questions = extract_official_questions_multi_page(pages, qp_page_indices)
    log(f"Official questions extracted: {len(official_questions)}")

    if not official_questions:
        raise Exception(
            "Question paper pages were found, but no questions could be parsed from them.\n"
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

    log("Matching questions via similarity (works for Hindi/English/any format)...")
    boundaries = find_question_boundaries_by_similarity(answer_lines, official_questions)
    log(f"Matched {len(boundaries)} of {len(official_questions)} questions")

    matched_qs = {b["question"] for b in boundaries}
    for q in official_questions:
        if q not in matched_qs:
            log(f"WARNING: No match found for: {q[:60]}")

    if not boundaries:
        raise Exception(
            "Could not match any questions in answer pages.\n"
            f"Official questions: {official_questions}\n"
            f"First 10 answer lines: {answer_lines[:10]}"
        )

    log("Slicing raw answers...")
    qa_pairs = slice_raw_answers_by_boundaries(answer_lines, boundaries)

    log(f"Done -- {len(qa_pairs)} Q-A pairs")
    return ocr_json, qa_pairs
