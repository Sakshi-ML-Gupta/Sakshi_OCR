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
# LINE RECOVERY  ***CRITICAL FIX***
#
# Mistral OCR markdown does NOT reliably emit "\n" between logical
# lines/paragraphs within a page's text. Instead, paragraph breaks
# are represented as runs of 2+ whitespace characters. A naive
# `text.split("\n")` therefore collapses an entire page into ONE
# string, which silently breaks every line-based regex in the
# pipeline (question detection, sub-part detection, noise filtering)
# without raising any error — this was the root cause of multiple
# previous bugs that looked like "matching" problems but were
# actually "there were no lines to match against" problems.
#
# This function reconstructs logical lines from raw OCR text:
#   1. Normalise actual "\n" characters to the same separator.
#   2. Split on runs of 2+ whitespace (paragraph/markdown breaks).
#   3. Within each resulting chunk, further split immediately before
#      any INLINE sub-part label (a) b) (i) (क) etc.) or inline
#      top-level "N. " that appears after other text on the same
#      chunk — these are real new logical lines that got fused
#      together because they weren't separated by 2+ whitespace in
#      the source (common when OCR reads a sequence of short labels
#      as one flowing line, e.g. "a) Renaissance b) Amoretti").
# =========================================================

_PARA_SPLIT_RE     = re.compile(r'\s{2,}')
_INLINE_SUBPART_RE = re.compile(r'(?<=\S)\s+(?=\(?[a-zA-Z]\)\s)')
_INLINE_SUBPART_DEVA_RE = re.compile(r'(?<=\S)\s+(?=\([\u0915-\u0939]\)\s)')
_INLINE_TOPLEVEL_RE = re.compile(r'(?<=\S)\s+(?=\d{1,2}\.\s+[A-Z\u0900-\u097F"\'])')


def recover_lines(raw_text: str) -> list:
    """Returns a list of logical lines reconstructed from raw OCR text."""
    if not raw_text:
        return []

    normalised = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = []
    for part in normalised.split("\n"):
        chunks.extend(_PARA_SPLIT_RE.split(part))

    lines = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        pieces = _INLINE_SUBPART_RE.split(chunk)
        for piece in pieces:
            sub_pieces = _INLINE_SUBPART_DEVA_RE.split(piece)
            for sp in sub_pieces:
                final_pieces = _INLINE_TOPLEVEL_RE.split(sp)
                lines.extend(p.strip() for p in final_pieces if p.strip())

    return lines


# =========================================================
# QUESTION PAPER DETECTION
# =========================================================

TOP_LEVEL_RE = re.compile(r'^\s*(\d{1,2})[\.\)]\s+(.+)')


def find_question_paper_page(pages: list) -> int:
    """
    Returns index (0-based) of the page that is the printed question
    paper. Looks for the page (after line recovery) with the most
    top-level numbered label lines.
    """
    best_idx, best_count = -1, 0
    for i, page in enumerate(pages):
        lines = recover_lines(page["raw_text"])
        count = sum(1 for line in lines if TOP_LEVEL_RE.match(line))
        if count > best_count:
            best_count = count
            best_idx = i
    return best_idx if best_count >= 3 else -1


# =========================================================
# OFFICIAL QUESTION EXTRACTION — WITH SUB-PARTS
# =========================================================

SUB_PART_PATTERNS = [
    re.compile(r'^\s*\(([a-zA-Z])\)\s*(.+)'),                 # (a) (b) (i) (ii)
    re.compile(r'^\s*([a-zA-Z])\)\s*(.+)'),                    # a) b) i) ii)
    re.compile(r'^\s*\(([\u0915-\u0939])\)\s*(.+)'),           # (क) (ख) (ग) (घ)
]

INSTRUCTION_ONLY_RE = re.compile(
    r'^(?:answer the following|write short notes on the following|'
    r'निम्नलिखित (?:पर|के) |सभी प्रश्न|note\s*[:.]|section\b|भाग[\-\s]?\d|'
    r'part\s*[-\s]?\w)',
    re.I
)

SKIP_LINE_RE = re.compile(
    r'^#+\s*$|^#+\s*(?:section|भाग|part)\b|^भाग|^PART|^\s*$|^Section\b|'
    r'^Max\.?\s*Marks|^Course Code|^\S+ ALL SOLVED|^openeducation',
    re.I
)


def _is_instruction_only(text: str) -> bool:
    stripped = text.strip()
    if not INSTRUCTION_ONLY_RE.match(stripped):
        return False
    word_count = len(stripped.split())
    return word_count <= 18


def extract_official_questions(page_text: str) -> list:
    """
    Walks the printed question-paper page (after line recovery) and
    produces a flat, ordered list of question units, correctly
    capturing sub-parts as independent entries tied to their parent,
    and excluding pure section/instruction lines from becoming their
    own (unanswerable) Q&A pair.
    """
    lines = recover_lines(page_text)

    units = []
    current_top = None
    pending_parent_label = None
    section_counter = 0

    def flush_top():
        if current_top is not None:
            current_top["text"] = current_top["text"].strip()
            units.append(current_top)

    for stripped in lines:
        if not stripped:
            continue
        if SKIP_LINE_RE.match(stripped) and not TOP_LEVEL_RE.match(stripped):
            continue

        m_top = TOP_LEVEL_RE.match(stripped)
        if m_top:
            num, rest = m_top.group(1), m_top.group(2)
            label = f"S{section_counter}-{num}" if section_counter else num

            if _is_instruction_only(rest):
                flush_top()
                current_top = None
                pending_parent_label = label
                section_counter += 1
                continue

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

        matched_sub = None
        for pat in SUB_PART_PATTERNS:
            m = pat.match(stripped)
            if m:
                matched_sub = m
                break

        if matched_sub and pending_parent_label is not None:
            letter, rest = matched_sub.group(1), matched_sub.group(2)

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

        # Continuation line: append to whichever unit is open
        if current_top is not None:
            current_top["text"] += " " + stripped
        elif units:
            units[-1]["text"] += " " + stripped

    flush_top()

    return [{"label": u["label"], "display_label": u["display_label"],
             "text": u["text"].strip(), "level": u["level"],
             "parent_label": u["parent_label"]} for u in units]


# =========================================================
# NOISE FILTERING FOR ANSWER LINES
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
    r'|^\s*\d{1,3}\s*$'
    r'|^ALL SOLVED ASSIGNMENT'
    r'|openeducation\.co\.in'
    r'|^Acknowledgment$'
    r'|^Enrolment Number'
    r'|^RC Code'
    r'|isms\.ignou\.ac\.in)',
    re.IGNORECASE
)


def is_noise(line: str) -> bool:
    return bool(NOISE_RE.search(line))


def _is_junk_page(page_text: str) -> bool:
    """
    Whole-page junk: covers, admit cards, web admin screenshots,
    acknowledgments — pages with no real Q&A content at all.
    """
    JUNK_SIGNALS = re.compile(
        r'Acknowledgment|acknowledgement|isms\.ignou|'
        r'KNOW YOUR ADMISSION|REGISTRATION DETAILS|'
        r'Student Identity Card|RC Code|Father\'?s Name|'
        r'ENROLMENT NUMBER\s*:|PROGRAMME (?:TITLE|CODE)\s*:|'
        r'sincere gratitude',
        re.I
    )
    lines = recover_lines(page_text)
    if not lines:
        return True
    hits = sum(1 for l in lines if JUNK_SIGNALS.search(l))
    return hits / max(len(lines), 1) > 0.25


# =========================================================
# SIMILARITY MATCHING
# =========================================================

def normalize(text: str) -> str:
    """
    Normalise text for word-overlap similarity matching.

    CRITICAL FIX: Python's `\\w` (even with re.UNICODE) does NOT
    include Devanagari combining marks — matras (ा ि ी ु ू ृ े ै ो ौ)
    and the virama/halant (्) all fail `\\w`. The original regex
    `[^\\w\\s]` therefore replaced every matra with a space, shattering
    Hindi words into their individual base consonants (e.g.
    "वैज्ञानिक" became "व ज ञ न क" — five fragments instead of one
    word). This silently destroyed similarity matching for ANY Hindi
    text, which is why Hindi document Q&A boundary detection was
    unreliable. We explicitly whitelist the Devanagari combining mark
    ranges alongside \\w so whole words survive intact.
    """
    text = text.lower()
    # \w (word chars) + \s (whitespace) + Devanagari combining marks:
    #   U+0900-0903: candrabindu, anusvara, visarga (nasalisation/aspiration signs)
    #   U+093A-094F: dependent vowel signs (matras) + virama/halant + nukta
    #   U+0951-0957: stress/accent marks + extra vowel signs
    #   U+0962-0963: vocalic l/ll vowel signs
    # Deliberately EXCLUDES U+0964 (danda) and U+0965 (double danda) —
    # these are Devanagari sentence-ending punctuation and should
    # still be treated as separators, same as a Latin period.
    text = re.sub(
        r'[^\w\s\u0900-\u0903\u093A-\u094F\u0951-\u0957\u0962\u0963]',
        ' ', text, flags=re.UNICODE
    )
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def similarity(a: str, b: str) -> float:
    wa = set(normalize(a).split())
    wb = set(normalize(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def _char_trigrams(word: str) -> set:
    word = f"  {word}  "
    return {word[i:i+3] for i in range(len(word) - 2)}


def fuzzy_word_similarity(a: str, b: str) -> float:
    """
    Character-trigram overlap between two short strings — used ONLY
    as a matching signal to LOCATE where a question is answered, never
    to alter any text. Handles OCR misspellings of proper nouns (e.g.
    question says "Amoretti", student's OCR'd handwriting reads
    "Amosetti") where plain word-overlap similarity scores 0 because
    no word is spelled identically.
    """
    ta, tb = _char_trigrams(a.lower()), _char_trigrams(b.lower())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def strip_leading_label(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^(?:Ans(?:wer)?[.\s]+)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^(?:उत्तर)\s*[\-\:\s]*', '', text)
    text = re.sub(r'^(?:प्र|प्रो|प्रश्न)[\.\s]*\d*[\.\s]*', '', text)
    text = re.sub(r'^[१-९०][०-९]*[\.\-\s]*', '', text)
    text = re.sub(r'^(?:Q\.?\s*)?\d+[.)]\s*', '', text)
    text = re.sub(r'^\(?[a-zA-Z\u0915-\u0939]\)\s*', '', text)
    return text.strip()


# A line that's just the question being restated (e.g. the student
# rewrote "(a) Renaissance" or "(b) Amoretti =" before actually
# answering) — short, label-prefixed, little independent content.
# We use this both to prefer non-echo boundary matches and to skip
# leading echo lines so an answer slice begins at the real prose.
def _is_echo_line(line: str, max_words: int = 5) -> bool:
    """
    True if this line looks like a BARE restatement of a question
    label/title with no real content attached — e.g. "(a) Renaissance"
    or "b) Amoretti =" — as opposed to "(a) Renaissance: It was in
    Italy that..." where a colon/dash introduces actual answer prose
    on the same line. The presence of a colon followed by more than a
    couple words means this is NOT a bare echo, even if otherwise short.
    """
    stripped = line.strip()

    # If there's a colon/dash separator followed by substantial text,
    # this line already contains real content — never treat as echo.
    m = re.search(r'[:\-]\s*(\S.*)$', stripped)
    if m and len(m.group(1).split()) >= 3:
        return False

    word_count = len(stripped.split())
    if word_count > max_words:
        return False
    starts_with_label = bool(
        re.match(r'^\s*\(?[a-zA-Z0-9\u0915-\u0939]{1,3}[\.\)]', stripped)
    )
    return starts_with_label


# Markers that explicitly signal "this is where the answer begins"
# when found right after a repeated sub-part letter — e.g. the
# question is restated as (a)(b)(c), then answered later as a
# SEPARATE (a)=>...(b)=>...(c)=>... block. Detecting these markers
# lets us prefer the true answer occurrence over the bare restatement.
ANSWER_MARKER_AFTER_LABEL_RE = re.compile(
    r'^\s*\(?[a-zA-Z\u0915-\u0939]\)?\s*[\.\)]?\s*(?:⇒|=>|:-|Ans(?:wer)?\s*[\-:]|Reference\s*:|उत्तर\s*[\-:]|→)',
    re.I
)


def _has_answer_marker(line: str) -> bool:
    return bool(ANSWER_MARKER_AFTER_LABEL_RE.match(line.strip()))


def _find_group_anchor_line(answer_lines: list, parent_label: str, sibling_units: list) -> int:
    """
    For a group of sub-parts sharing the same parent (e.g. 1a, 1b),
    find the line index where this group's question block is restated
    (the cluster of bare "(a)", "(b)" labels, or the first sub-part's
    high-confidence text match). Returns -1 if not found.
    Used to scope answer-marker matching to AFTER this point, so
    letter labels reused by a different section's question never
    collide with this group's answers.
    """
    best_line = -1
    for unit in sibling_units:
        if unit["parent_label"] != parent_label:
            continue
        q_clean = strip_leading_label(unit["text"])
        q_words = q_clean.split()
        if not q_words:
            continue
        for i, line in enumerate(answer_lines):
            s = similarity(line, unit["text"])
            if s >= 0.4:
                if best_line == -1 or i < best_line:
                    best_line = i
                break
    return best_line


def find_question_boundaries_by_similarity(
    answer_lines: list,
    question_units: list,
    similarity_threshold: float = 0.28,
    window: int = 4
) -> list:
    candidates = []

    # Precompute, for each parent group, the line range where its
    # question block is restated and (presumably) answered — used to
    # scope answer-marker (s4) matching so reused letter labels
    # (a)/(b)/(c) across different sections never collide.
    parent_labels_in_order = []
    seen_pl = set()
    for u in question_units:
        if u["parent_label"] and u["parent_label"] not in seen_pl:
            parent_labels_in_order.append(u["parent_label"])
            seen_pl.add(u["parent_label"])

    raw_anchor = {
        pl: _find_group_anchor_line(answer_lines, pl, question_units)
        for pl in parent_labels_in_order
    }
    # Bound each group's window to end where the NEXT group (in
    # question-paper order) begins, so a letter match deep inside a
    # later section's answers never gets credited to an earlier
    # group's same-letter sub-part.
    group_window = {}
    for idx, pl in enumerate(parent_labels_in_order):
        start = raw_anchor.get(pl, -1)
        next_starts = [
            raw_anchor[parent_labels_in_order[j]]
            for j in range(idx + 1, len(parent_labels_in_order))
            if raw_anchor.get(parent_labels_in_order[j], -1) != -1
        ]
        end = min(next_starts) if next_starts else len(answer_lines)
        group_window[pl] = (start, end)

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

                s3 = 0.0
                s3_anchor = False
                if unit["level"] == "sub":
                    first_word = q_clean.split()[0] if q_clean.split() else ""
                    first_line_norm = normalize(answer_lines[i].strip()).split()
                    if first_word and first_word.lower() in first_line_norm:
                        s3 = 0.5
                        s3_anchor = True
                    elif first_word and len(first_word) >= 5:
                        # Fuzzy fallback for OCR-misspelled proper nouns
                        # (e.g. question "Amoretti" vs OCR "Amosetti").
                        # Only applied to longer words to avoid noisy
                        # false positives on short common words.
                        best_fuzzy = max(
                            (fuzzy_word_similarity(first_word, w) for w in first_line_norm),
                            default=0.0
                        )
                        if best_fuzzy >= 0.6:
                            s3 = 0.45
                            s3_anchor = True

                s4 = 0.0
                if unit["level"] == "sub" and _has_answer_marker(answer_lines[i]):
                    m_label = re.match(
                        r'^\s*\(?([a-zA-Z\u0915-\u0939])\)?', answer_lines[i].strip()
                    )
                    unit_letter = unit["display_label"].rstrip(") ").lstrip("(")
                    win_start, win_end = group_window.get(unit["parent_label"], (-1, -1))
                    if (m_label and m_label.group(1).lower() == unit_letter.lower()
                            and win_start != -1 and win_start <= i < win_end):
                        s4 = 0.6

                score = max(s1, s2, s3, s4)
                winning_signal = "s1_s2"
                if score > 0 and score == s4:
                    winning_signal = "s4"
                elif score > 0 and score == s3:
                    winning_signal = "s3"

                # When s3 or s4 (single-line anchors) produce the
                # winning score, the answer slice should start AT
                # this line — it likely contains both the label/
                # marker and the start of real prose (e.g. "(a)
                # Renaissance: It was in Italy..." or "(a) ⇒
                # Reference: The given lines are...") — not after it.
                effective_span = w
                if winning_signal == "s4":
                    effective_span = 0
                elif winning_signal == "s3" and s3_anchor:
                    effective_span = 0

                if score >= similarity_threshold:
                    candidates.append({
                        "question": unit["text"],
                        "label": unit["label"],
                        "display_label": unit["display_label"],
                        "level": unit["level"],
                        "parent_label": unit["parent_label"],
                        "line_index": i,
                        "span": effective_span,
                        "score": score,
                        "via": winning_signal,
                    })

    # Non-echo candidates (real content) always beat echo candidates
    # (bare label restatement) for the same question label, regardless
    # of score — a perfect-score match on a bare "(a) Renaissance"
    # label is less useful than an 0.85-score match on the line that
    # actually starts the real answer. Only fall back to an echo
    # candidate if NO non-echo candidate exists for that label (e.g.
    # OCR garbled the proper noun so badly the real-answer line never
    # scored high enough to become a candidate at all).
    def _candidate_rank(c):
        line_text = answer_lines[c["line_index"]] if c["line_index"] < len(answer_lines) else ""
        is_echo = _is_echo_line(line_text)
        return (1 if is_echo else 0, -round(c["score"], 2))

    best_per_label = {}
    for c in candidates:
        key = c["label"]
        if key not in best_per_label or _candidate_rank(c) < _candidate_rank(best_per_label[key]):
            best_per_label[key] = c

    # Second pass: for any unit where the winning candidate is itself
    # a restatement of the question (very high s1/s2 word-overlap,
    # occurring at/near its group's anchor line) AND a genuine
    # answer-marker (s4) candidate exists for the same unit further
    # along, prefer the s4 candidate — restating the quote is not
    # the same as answering it.
    s4_candidates_by_label = {}
    for c in candidates:
        if c.get("via") == "s4":
            key = c["label"]
            if key not in s4_candidates_by_label or c["score"] > s4_candidates_by_label[key]["score"]:
                s4_candidates_by_label[key] = c

    for label, s4_cand in s4_candidates_by_label.items():
        current = best_per_label.get(label)
        if current is None:
            best_per_label[label] = s4_cand
            continue
        if current.get("via") != "s4":
            # Current winner came from raw word-overlap (likely the
            # bare restatement). Swap to the s4 (explicit answer
            # marker) candidate, which is structurally more reliable
            # for this "restated-then-answered-separately" pattern.
            best_per_label[label] = s4_cand

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
    qa_pairs = []
    for i, b in enumerate(boundaries):
        span    = b.get("span", 1)
        a_start = b["line_index"] + span
        a_end   = boundaries[i + 1]["line_index"] if i + 1 < len(boundaries) else len(answer_lines)

        # Skip leading echo lines (restated question labels) so the
        # answer starts at the real prose, not a repeated question.
        while a_start < a_end and _is_echo_line(answer_lines[a_start]):
            a_start += 1

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
        if _is_junk_page(page["raw_text"]):
            log(f"  Skipping junk page {page['page_number']}")
            continue
        for line in recover_lines(page["raw_text"]):
            if not is_noise(line):
                answer_lines.append(line)

    log(f"Flattened {len(answer_lines)} answer lines (after line recovery)")

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
