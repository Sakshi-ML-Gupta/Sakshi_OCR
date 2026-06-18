import os
import io
import re
import json
import fitz
import base64
import httpx
from pathlib import Path
from mistralai import Mistral

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
# OCR
# =========================================================

def run_ocr(file_content: bytes, file_name: str, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    api_key = get_api_key("MISTRAL_API_KEY")
    client  = Mistral(api_key=api_key)

    log("Uploading to Mistral OCR...")
    uploaded = client.files.upload(
        file={"file_name": file_name, "content": file_content},
        purpose="ocr"
    )
    signed = client.files.get_signed_url(file_id=uploaded.id, expiry=1)
    log("Running OCR...")

    resp = httpx.post(
        "https://api.mistral.ai/v1/ocr",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": signed.url
            },
            "include_image_base64": False
        },
        timeout=180
    )

    if resp.status_code != 200:
        raise Exception(f"OCR error {resp.status_code}: {resp.text}")

    try:
        client.files.delete(file_id=uploaded.id)
    except Exception:
        pass

    pages = []
    for page in resp.json().get("pages", []):
        pages.append({
            "page_number": page.get("index", 0) + 1,
            "raw_text":    page.get("markdown", "")
        })

    log(f"OCR done — {len(pages)} pages")
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
# REFERENCE BOOK — page by page, base64 inline
# =========================================================

def ocr_page_base64(page_b64: str, api_key: str) -> str:
    resp = httpx.post(
        "https://api.mistral.ai/v1/ocr",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{page_b64}"
            },
            "include_image_base64": False
        },
        timeout=120
    )
    if resp.status_code == 200:
        ocr_pages = resp.json().get("pages", [])
        return ocr_pages[0].get("markdown", "") if ocr_pages else ""
    raise Exception(f"OCR error {resp.status_code}: {resp.text[:200]}")


def process_reference(file_input, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    if isinstance(file_input, (str, Path)):
        file_bytes = Path(file_input).read_bytes()
    else:
        file_bytes = file_input.read()

    api_key     = get_api_key("MISTRAL_API_KEY")
    src_doc     = fitz.open(stream=file_bytes, filetype="pdf")
    total_pages = len(src_doc)
    src_doc.close()

    log(f"Reference book: {total_pages} page(s)")
    pages_output = []

    for page_num in range(total_pages):
        log(f"Page {page_num + 1} of {total_pages}...")
        try:
            src_doc  = fitz.open(stream=file_bytes, filetype="pdf")
            one_page = fitz.open()
            one_page.insert_pdf(src_doc, from_page=page_num, to_page=page_num)
            src_doc.close()

            pix      = one_page[0].get_pixmap(dpi=200)
            out_doc  = fitz.open()
            new_page = out_doc.new_page(width=pix.width, height=pix.height)
            new_page.insert_image(new_page.rect, pixmap=pix)

            buf      = io.BytesIO()
            out_doc.save(buf)
            page_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            one_page.close()
            out_doc.close()
            buf.close()

            text = ocr_page_base64(page_b64, api_key)
        except Exception as e:
            log(f"  Page {page_num + 1} failed: {e}")
            text = ""

        pages_output.append({"page_number": page_num + 1, "raw_text": text})

    log(f"Reference OCR complete — {total_pages} pages")
    return build_ocr_json(pages_output)


# =========================================================
# DETECT QUESTION PAPER PAGE
# The page that contains the printed list of questions
# Identified by having numbered questions >= 5 items
# =========================================================

def find_question_paper_pages(pages: list, min_questions: int = 2) -> list:
    """
    Returns list of page indices (0-based) that look like question paper
    pages — i.e. pages dominated by printed numbered/lettered questions
    rather than handwritten answers.

    Unlike picking a single "best" page, this scans every page, since
    question papers can span multiple pages or be split into sections
    (Part 1 / Part 2 / Part 3, or a parent question like "Q.9 Write notes
    on:" followed by lettered sub-parts a)/b)/c)/d) or क)/ख)/ग)/घ) that
    may land on their own page).

    A page counts as a question-paper page if it has at least
    `min_questions` lines matching EITHER a numbered question pattern
    OR a lettered sub-part pattern, AND does not contain answer-marker
    text (Ans-, उत्तर, A.15- etc) — answer pages sometimes contain
    numbered sub-points which would otherwise be miscounted as questions.
    """
    Q_LINE_NUM   = re.compile(r'^\s*\d+[\.\)]\s+.{15,}')
    Q_LINE_LATIN = re.compile(r'^\s*[a-d]\)\s+.{5,}', re.IGNORECASE)
    Q_LINE_DEVA  = re.compile(r'^\s*[क-घ]\)\s+.{5,}')
    ANSWER_MARKER = re.compile(
        r'(?:उत्तर\s*[\-\:]|Ans\.?\s*[\-\:]|A\.\d|A\d+\s*[\-\:])',
        re.IGNORECASE
    )

    candidate_pages = []

    for i, page in enumerate(pages):
        text  = page["raw_text"]
        lines = text.split("\n")

        q_count = sum(
            1 for line in lines
            if Q_LINE_NUM.match(line.strip())
            or Q_LINE_LATIN.match(line.strip())
            or Q_LINE_DEVA.match(line.strip())
        )

        has_answer_marker = bool(ANSWER_MARKER.search(text))

        if q_count >= min_questions and not has_answer_marker:
            candidate_pages.append(i)

    return candidate_pages


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
    """Strip leading numbering like '1.', 'Q1.', 'प्र.2', '20.', 'a)' etc."""
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

    # Step 1: Preprocess
    log("Preprocessing PDF...")
    processed = preprocess_pdf(file_bytes)

    # Step 2: OCR
    pages = run_ocr(processed, file_name, status_callback)

    # Step 3: Build OCR JSON
    log("Building OCR JSON...")
    ocr_json = build_ocr_json(pages)
    log(f"Total pages: {ocr_json['total_pages']}")

    # Step 4: Scan ALL pages for question-paper-like pages
    # (handles question papers split across multiple pages/sections,
    #  rather than assuming a single page holds everything)
    qp_page_indices = find_question_paper_pages(pages)
    log(f"Question paper pages detected: {[p+1 for p in qp_page_indices] if qp_page_indices else 'none'}")

    if not qp_page_indices:
        raise Exception(
            "Could not detect any question paper pages in this document.\n"
            "This usually means the document has a different layout than expected — "
            "no page was found with multiple numbered question lines.\n"
            f"Page 1 preview:\n{pages[0]['raw_text'][:500]}"
        )

    official_questions = extract_official_questions_multi_page(pages, qp_page_indices)
    log(f"Official questions extracted: {len(official_questions)}")

    if not official_questions:
        raise Exception(
            "Question paper pages were found, but no questions could be parsed from them.\n"
            f"Detected pages: {[p+1 for p in qp_page_indices]}"
        )

    # Step 5: Answer pages = every page NOT identified as a question paper page
    answer_page_indices = [i for i in range(len(pages)) if i not in qp_page_indices]
    answer_pages = [pages[i] for i in answer_page_indices]

    log(f"Answer pages: {[i+1 for i in answer_page_indices]}")

    # Step 6: Flatten answer page lines
    answer_lines = []
    for page in answer_pages:
        for line in page["raw_text"].split("\n"):
            if not is_noise(line):
                answer_lines.append(line)

    log(f"Flattened {len(answer_lines)} answer lines")

    # Step 7: Similarity-based matching — works for any language/format
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

    # Step 8: Slice raw answers — zero LLM, pure text slicing
    log("Slicing raw answers...")
    qa_pairs = slice_raw_answers_by_boundaries(answer_lines, boundaries)

    log(f"Done — {len(qa_pairs)} Q-A pairs")
    return ocr_json, qa_pairs
