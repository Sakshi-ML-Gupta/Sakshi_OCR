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
# REFERENCE BOOK OCR
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
# QUESTION PAPER DETECTION
# =========================================================

def find_question_paper_page(pages: list) -> int:
    """
    Returns index (0-based) of the page that is the printed question
    paper. Looks for the page with the most numbered/lettered label
    lines AND short per-label content (a list, not prose answers).
    """
    best_idx, best_count = -1, 0
    for i, page in enumerate(pages):
        count = sum(
            1 for line in page["raw_text"].split("\n")
            if TOP_LEVEL_RE.match(line.strip())
        )
        if count > best_count:
            best_count = count
            best_idx = i
    return best_idx if best_count >= 3 else -1


# =========================================================
# OFFICIAL QUESTION EXTRACTION — WITH SUB-PARTS
#
# Produces a FLAT list of question units in document order, where
# each unit knows its own label/text AND, if it's a sub-part, which
# parent question it belongs to. This is the key structural fix:
# sub-parts (a/b/c, i/ii/iii, क/ख/ग/घ) are extracted as independent
# question units, not swallowed into their parent's answer blob.
#
# Each unit: {"label": str, "text": str, "level": "top"|"sub",
#             "parent_label": str|None}
# =========================================================

TOP_LEVEL_RE = re.compile(r'^\s*(\d{1,2})[\.\)]\s+(.+)')

SUB_PART_PATTERNS = [
    re.compile(r'^\s*\(([a-zA-Z])\)\s*(.+)'),                 # (a) (b) (i) (ii)
    re.compile(r'^\s*([a-zA-Z])\)\s*(.+)'),                    # a) b) i) ii) without parens
    re.compile(r'^\s*\(([\u0915-\u0939])\)\s*(.+)'),           # (क) (ख) (ग) (घ)
]

# Lines that are SECTION/INSTRUCTION headers, not real questions.
# These describe a *group* of questions ("Answer the following
# questions in about 800 words each") rather than asking one
# specific thing — they should never become a Q&A pair of their own.
INSTRUCTION_ONLY_RE = re.compile(
    r'^(?:answer the following|write short notes on the following|'
    r'निम्नलिखित (?:पर|के) |सभी प्रश्न|note\s*[:.]|section\b|भाग[\-\s]?\d|'
    r'part\s*[-\s]?\w)',
    re.I
)

SKIP_LINE_RE = re.compile(r'^#+\s|^भाग|^PART|^\s*$|^Section\b', re.I)


def _is_instruction_only(text: str) -> bool:
    """
    True if this looks like a section/group instruction rather than a
    question with its own distinct answer — e.g. "Answer the following
    questions in about 800 words each" (no specific content asked),
    as opposed to "Discuss the major themes in the play Dr. Faustus"
    (a specific, answerable prompt) even though both start similarly.
    """
    stripped = text.strip()
    # If the line is ONLY the instructional phrase with no further
    # specific content (short, generic, ends right after marks/word
    # count), treat as instruction. If it's long / contains a specific
    # question after a colon, treat as real.
    if not INSTRUCTION_ONLY_RE.match(stripped):
        return False
    # Real questions that happen to start with "Answer the following"
    # but then specify content are longer / contain "?" or a colon
    # followed by substantial text. Pure instructions are short and
    # end in word-count / marks notation.
    word_count = len(stripped.split())
    return word_count <= 18


def extract_official_questions(page_text: str) -> list:
    """
    Walks the printed question-paper page and produces a flat,
    ordered list of question units, correctly capturing sub-parts
    as independent entries tied to their parent.

    Handles:
      - top-level numbered questions: "1. text", "2. text"
      - lettered/roman sub-parts following a top-level question:
        "a) text", "(b) text", "(i) text"
      - Devanagari lettered sub-parts: "(क) text"
      - multi-line continuations (text wraps to next line before the
        next label appears)
      - section instruction lines that should NOT become their own
        Q&A pair (e.g. "Answer the following questions in about 800
        words each.") — these are dropped, and their *contained*
        sub-items (the actual numbered questions that follow) become
        the real top-level units for that section
      - duplicate numbering across sections (e.g. Section A's "1.,
        2., 3." vs Section C's own "1., 2., 3.") by tracking section
        boundaries via instruction lines and re-keying duplicate
        labels with a section-aware suffix so they never collide
    """
    lines = page_text.split("\n")

    units = []
    current_top = None        # currently open top-level unit (dict) or None
    pending_parent_label = None  # label that subsequent sub-parts (a,b,c) should attach to,
                                   # even if the parent line itself was instruction-only
                                   # and never became its own unit
    section_counter = 0        # increments every time we see an instruction line

    def flush_top():
        if current_top is not None:
            current_top["text"] = current_top["text"].strip()
            units.append(current_top)

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if SKIP_LINE_RE.match(stripped) and not TOP_LEVEL_RE.match(stripped):
            continue

        # --- Top-level numbered line? ---
        m_top = TOP_LEVEL_RE.match(stripped)
        if m_top:
            num, rest = m_top.group(1), m_top.group(2)
            label = f"S{section_counter}-{num}" if section_counter else num

            if _is_instruction_only(rest):
                # This is a section header like "Answer the following
                # questions in about 800 words each." Close out any
                # open unit, bump the section counter so subsequent
                # "1./2./3." labels get a unique section-aware key,
                # and do NOT create a Q&A pair for this line itself —
                # but DO remember its label so sub-parts that follow
                # (a) b) c)) can still attach to it.
                flush_top()
                current_top = None
                pending_parent_label = label
                section_counter += 1
                continue

            # Real top-level question. Close the previous one first.
            flush_top()
            current_top = {
                "label": label,
                "display_label": f"{num}.",
                "text": rest,
                "level": "top",
                "parent_label": None,
            }
            pending_parent_label = label
            continue

        # --- Sub-part line (a) b) (क) etc — attaches to whichever
        # parent label is currently pending (whether or not that
        # parent became its own answerable unit). ---
        matched_sub = None
        for pat in SUB_PART_PATTERNS:
            m = pat.match(stripped)
            if m:
                matched_sub = m
                break

        if matched_sub and pending_parent_label is not None:
            letter, rest = matched_sub.group(1), matched_sub.group(2)

            # First sub-part under this parent: the parent itself
            # stops accumulating free text (it's now just an umbrella),
            # so flush/clear current_top if it matches this parent.
            if current_top is not None and current_top["label"] == pending_parent_label:
                flush_top()
                current_top = None

            units.append({
                "label": f"{pending_parent_label}{letter})",
                "display_label": f"{letter})",
                "text": rest,
                "level": "sub",
                "parent_label": pending_parent_label,
            })
            continue

        # --- Continuation line: append to whichever unit is open ---
        if current_top is not None:
            current_top["text"] += " " + stripped
        elif units:
            units[-1]["text"] += " " + stripped
        # else: stray line before any question started — ignore

    flush_top()

    # Build final question text list. If a top-level question has
    # sub-parts, it has no independent answer of its own — only its
    # sub-parts are answerable questions. We mark this by NOT including
    # bare top-level questions that were immediately consumed into subs
    # (they were never appended above since we set current_top=None and
    # didn't flush them as a standalone unit — flush_top() found
    # current_top None and added nothing extra, so they're naturally
    # excluded). Units list now contains exactly the answerable items.

    return [{"label": u["label"], "display_label": u["display_label"],
             "text": u["text"].strip(), "level": u["level"],
             "parent_label": u["parent_label"]} for u in units]


# =========================================================
# NOISE FILTERING FOR ANSWER PAGES
# =========================================================

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
    r'|Experiment\s*Name'
    r'|KLA(?:SS|ES|SE)?(?:NOTE|ENOTE|ENSTE|ENCTE|SCHOTE)?'
    r'|KLEBENOTE|KILKEENOTE|KIASSNOTE|KIASENOTE|KIRENNOTE'
    r'|!\[img[\-\d]*\.jpeg?\]\([^)]*\)'
    r'|^\s*\*{1,3}\s*$'
    r'|^\s*\d{1,3}\s*$)',
    re.IGNORECASE
)


def is_noise(line: str) -> bool:
    return bool(NOISE_RE.search(line))


# =========================================================
# SIMILARITY MATCHING — find where each question is restated/
# answered in the handwritten pages
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
    text = re.sub(r'^\(?[a-zA-Z\u0915-\u0939]\)\s*', '', text)
    return text.strip()


def find_question_boundaries_by_similarity(
    answer_lines: list,
    question_units: list,
    similarity_threshold: float = 0.28,
    window: int = 4
) -> list:
    """
    Scans answer lines for restated questions matching official
    question units (top-level AND sub-parts treated identically —
    each is just a "question" to match against). Sub-parts use a
    lower window since they're typically short labels like "a) Renaissance".
    """
    candidates = []

    for i in range(len(answer_lines)):
        line_i = answer_lines[i].strip()
        if len(line_i) < 3:
            continue

        for w in range(1, window + 1):
            if i + w > len(answer_lines):
                break

            combined = " ".join(
                answer_lines[i + k].strip()
                for k in range(w) if answer_lines[i + k].strip()
            )
            if len(combined) < 6:
                continue

            combined_clean = strip_leading_label(combined)

            for unit in question_units:
                q_text = unit["text"]
                q_clean = strip_leading_label(q_text)

                s1 = similarity(combined, q_text)
                s2 = similarity(combined_clean, q_clean)

                # Sub-part labels are often very short (e.g. "Renaissance",
                # "Amoretti") — boost exact-label-line matches.
                s3 = 0.0
                if unit["level"] == "sub":
                    label_only = unit["display_label"].rstrip(")")
                    # does the answer line literally start with this
                    # sub label's first content word?
                    first_word = q_clean.split()[0] if q_clean.split() else ""
                    if first_word and first_word.lower() in normalize(combined).split():
                        s3 = 0.5

                score = max(s1, s2, s3)

                if score >= similarity_threshold:
                    candidates.append({
                        "question": unit["text"],
                        "label": unit["label"],
                        "display_label": unit["display_label"],
                        "level": unit["level"],
                        "parent_label": unit["parent_label"],
                        "line_index": i,
                        "span": w,
                        "score": score
                    })

    # Best candidate per question unit (by label, since text might repeat)
    best_per_label = {}
    for c in candidates:
        key = c["label"]
        if key not in best_per_label or c["score"] > best_per_label[key]["score"]:
            best_per_label[key] = c

    # Enforce document order matches official question order
    final = []
    last_line_index = -1
    for unit in question_units:
        c = best_per_label.get(unit["label"])
        if c is None:
            continue
        if c["line_index"] <= last_line_index:
            continue
        final.append(c)
        last_line_index = c["line_index"]

    return final


def slice_raw_answers_by_boundaries(answer_lines: list, boundaries: list) -> list:
    """
    For each boundary, answer = raw lines starting AFTER the matched
    question span, up to the next boundary. Pure text slicing.
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
            "question": f"{b['display_label']} {b['question']}".strip(),
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

    log("Preprocessing PDF...")
    processed = preprocess_pdf(file_bytes)

    pages = run_ocr(processed, file_name, status_callback)

    log("Building OCR JSON...")
    ocr_json = build_ocr_json(pages)
    log(f"Total pages: {ocr_json['total_pages']}")

    qp_idx = find_question_paper_page(pages)
    log(f"Question paper detected on page: {qp_idx + 1 if qp_idx >= 0 else 'not found'}")

    if qp_idx >= 0:
        official_questions = extract_official_questions(pages[qp_idx]["raw_text"])
        log(f"Official question units extracted (incl. sub-parts): {len(official_questions)}")
        for u in official_questions:
            log(f"  [{u['level']}] {u['display_label']} {u['text'][:60]}")
    else:
        official_questions = []
        log("No question paper page found")

    if not official_questions:
        raise Exception(
            "Could not extract official questions from the question paper page.\n"
            f"Preview of detected page:\n{pages[max(qp_idx,0)]['raw_text'][:500]}"
        )

    answer_start = qp_idx + 1 if qp_idx >= 0 else 0
    answer_pages = pages[answer_start:]

    answer_lines = []
    for page in answer_pages:
        for line in page["raw_text"].split("\n"):
            if not is_noise(line):
                answer_lines.append(line)

    log(f"Flattened {len(answer_lines)} answer lines")

    log("Matching questions via similarity (top-level + sub-parts)...")
    boundaries = find_question_boundaries_by_similarity(answer_lines, official_questions)
    log(f"Matched {len(boundaries)} of {len(official_questions)} question units")

    matched_labels = {b["label"] for b in boundaries}
    for u in official_questions:
        if u["label"] not in matched_labels:
            log(f"WARNING: No match found for [{u['level']}] {u['display_label']} {u['text'][:60]}")

    if not boundaries:
        raise Exception(
            "Could not match any questions in answer pages.\n"
            f"Official questions: {[u['text'] for u in official_questions]}\n"
            f"First 10 answer lines: {answer_lines[:10]}"
        )

    log("Slicing raw answers...")
    qa_pairs = slice_raw_answers_by_boundaries(answer_lines, boundaries)

    log(f"Done — {len(qa_pairs)} Q-A pairs")
    return ocr_json, qa_pairs
