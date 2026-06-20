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
# OCR — Datalab (Chandra model) via /convert endpoint
# =========================================================

DATALAB_BASE_URL = "https://www.datalab.to"

def _split_paginated_markdown(markdown: str, total_pages_hint: int = None) -> list:
    """
    Robust multi-fallback approach to split Datalab's paginated markdown.
    """
    # Fallback 1: Standard explicit page text
    parts = re.split(r'\n-{3,}\s*Page\s*\d+\s*-{3,}\n', markdown, flags=re.IGNORECASE)
    if len(parts) > 1:
        return [p.strip() for p in parts if p.strip()]

    # Fallback 2: Line-by-line scan for dash sequences ignoring embedded images
    lines = markdown.split('\n')
    split_indices = []
    
    for i, line in enumerate(lines):
        clean_line = re.sub(r'!\[.*?\]\(.*?\)', '', line).strip()
        if len(clean_line) >= 3 and clean_line.replace('-', '').strip() == '':
            split_indices.append(i)
            
    if len(split_indices) > 0:
        pages = []
        start = 0
        for idx in split_indices:
            pages.append("\n".join(lines[start:idx]).strip())
            start = idx + 1
        pages.append("\n".join(lines[start:]).strip())
        return [p for p in pages if p.strip()]

    # Fallback 3: Form feed character
    if '\f' in markdown:
        parts = markdown.split('\f')
        return [p.strip() for p in parts if p.strip()]

    # Fallback 4: Un-paginated text - split evenly by paragraphs if hint provided
    if total_pages_hint and total_pages_hint > 1:
        blocks = re.split(r'\n\s*\n', markdown)
        if len(blocks) > total_pages_hint:
            chunk_size = len(blocks) // total_pages_hint
            pages = []
            for i in range(total_pages_hint):
                start = i * chunk_size
                end = start + chunk_size if i < total_pages_hint - 1 else len(blocks)
                pages.append("\n\n".join(blocks[start:end]).strip())
            return pages

    return [markdown.strip()]


def run_ocr(file_content: bytes, file_name: str, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

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
    log("Document submitted — polling for OCR result...")

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
# REFERENCE BOOK OCR
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
# Pure structural pattern matching. No layout assumptions.
# =========================================================

NEGATIVE_FINGERPRINTS = re.compile(
    r'(?:'
    r'identity\s*card|id\s*card'                         
    r'|this\s+card\s+should\s+be\s+produced'              
    r'|student\s+name|father\s+name|enrolment\s+no'       
    r'|programme\s*code|reg\.\s*no|study\s+centre'        
    r'|signature\s*of\s+the\s+student'                    
    r'|date\s*of\s*issue|valid\s*upto'                    
    r')',
    re.IGNORECASE
)

ANSWER_PAGE_FINGERPRINTS = re.compile(
    r'(?:'
    r'उत्तर\s*[\-\:]|Ans\.?\s*[\-\:]|A\.\d|A\d+\s*[\-\:]' 
    r'|\bAns\b\s*$'                                        
    r')',
    re.IGNORECASE
)

STRONG_EXAM_SIGNALS = [
    re.compile(r'\b\d+\s*[xX×]\s*\d+\s*=\s*\d+\b'),
    re.compile(r'(?:\(|\[|\s)\d{2}\s*(?:\)|\]|\s|$)'),      
    re.compile(r'\bSECTION\s*[\-–]?\s*[A-D]\b', re.IGNORECASE),
    re.compile(r'\bPART\s*[\-–]?\s*[A-D]\b', re.IGNORECASE),
    re.compile(r'\bखंड\s*[\-–]?\s*[अ-ज]\b'),               
    re.compile(r'\b[A-Z]{2,4}\s*[-–]\s*\d{2,4}\b'),
    re.compile(r'(?:Time|Duration|समय)\s*[:\-]?\s*\d+\s*(?:Hours|Hrs|मिनट|घंटे)', re.IGNORECASE),
    re.compile(r'(?:Maximum\s*Marks|कुल\s*अंक)\s*[:\-]?\s*\d+', re.IGNORECASE),
    re.compile(r'(?:attempt|explain|define|describe|discuss|write\s+notes|compare|analyze|evaluate|illustrate)', re.IGNORECASE),
]

def find_question_paper_pages(pages: list, min_questions: int = 2, min_score: int = 3) -> list:
    Q_LINE_NUM   = re.compile(r'^\s*\d+[\.\)]\s+.{20,}')
    Q_LINE_LATIN = re.compile(r'^\s*[a-d]\)\s+.{8,}', re.IGNORECASE)
    Q_LINE_DEVA  = re.compile(r'^\s*[क-घ]\)\s+.{8,}')

    candidate_pages = []

    for i, page in enumerate(pages):
        text  = page["raw_text"]
        lines = text.split("\n")

        if NEGATIVE_FINGERPRINTS.search(text):
            continue

        if ANSWER_PAGE_FINGERPRINTS.search(text):
            continue

        score = 0

        for signal_re in STRONG_EXAM_SIGNALS:
            if signal_re.search(text):
                score += 1

        q_count = 0
        for line in lines:
            stripped = line.strip()
            is_q = (Q_LINE_NUM.match(stripped) 
                    or Q_LINE_LATIN.match(stripped) 
                    or Q_LINE_DEVA.match(stripped))
            if is_q:
                if not re.match(r'^(?:Ans|उत्तर|A\.)', stripped, re.IGNORECASE):
                    q_count += 1
                    
        if q_count >= min_questions:
            score += 1

        if score >= min_score:
            candidate_pages.append(i)

    return candidate_pages


# =========================================================
# EXTRACT OFFICIAL QUESTIONS
# =========================================================

def extract_official_questions_multi_page(pages: list, qp_page_indices: list) -> list:
    all_questions = []
    pending_parent = None

    Q_START   = re.compile(r'^\s*(\d+)[\.\)]\s+(.+)')
    SUB_LATIN = re.compile(r'^\s*([a-d])\)\s*(.+)', re.IGNORECASE)
    SUB_DEVA  = re.compile(r'^\s*([क-घ])\)\s*(.+)')
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
    SUBPART_RE = re.compile(r'(?:^|\s)([a-zक-घ])\)\s', re.UNICODE)

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
# FIND QUESTION BOUNDARIES IN ANSWER PAGES
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
    text = re.sub(r'^[a-z]\)\s*', '', text, flags=re.IGNORECASE)
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

    if isinstance(file_input, (str, Path)):
        file_bytes = Path(file_input).read_bytes()
        file_name  = Path(file_input).name
    else:
        file_bytes = file_input.read()
        file_name  = getattr(file_input, "name", "document.pdf")

    pages = run_ocr(file_bytes, file_name, status_callback)

    log("Building OCR JSON...")
    ocr_json = build_ocr_json(pages)
    log(f"Total pages: {ocr_json['total_pages']}")

    qp_page_indices = find_question_paper_pages(pages)
    log(f"Question paper pages detected: {[p+1 for p in qp_page_indices] if qp_page_indices else 'none'}")

    if not qp_page_indices:
        raise Exception(
            "Could not detect any question paper pages in this document.\n"
            "This usually means the document has a different layout than expected — "
            "no page was found with strong exam structural signals (Marks, Sections, Course Codes).\n"
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

    log(f"Done — {len(qa_pairs)} Q-A pairs")
    return ocr_json, qa_pairs
