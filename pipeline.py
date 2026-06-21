import os
import io
import re
import json
import time
import fitz
from pathlib import Path
from PIL import Image

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


def get_gemini_model():
    import google.generativeai as genai
    api_key = get_api_key("GEMINI_API_KEY")
    if not api_key:
        raise Exception("GEMINI_API_KEY not found. Get a free one at https://aistudio.google.com/apikey")
    
    genai.configure(api_key=api_key)
    # Flash 1.5 is extremely fast, free, and supports all Indian languages natively
    return genai.GenerativeModel('gemini-2.0-flash')


# =========================================================
# OCR — Google Gemini 1.5 Flash (Vision)
#
# Supports ALL Indian languages natively. Zero cost.
# Pure Python SDK — no binaries, works instantly on Streamlit Cloud.
# =========================================================

def run_ocr(file_bytes: bytes, file_name: str, status_callback=None, dpi: int = 200):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    model = get_gemini_model()
    
    size_mb = len(file_bytes) / (1024 * 1024)
    log(f"Opening PDF for Gemini Vision OCR... ({size_mb:.1f}MB)")

    src_doc = fitz.open(stream=file_bytes, filetype="pdf")
    total_pages = len(src_doc)
    log(f"PDF has {total_pages} page(s)")

    pages = []

    for page_num in range(total_pages):
        page = src_doc[page_num]
        pix = page.get_pixmap(dpi=dpi)

        # Convert directly to PIL Image (Gemini SDK accepts PIL natively)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        log(f"  Scanning page {page_num + 1}/{total_pages}...")

        prompt = (
            "Extract ALL text from this image exactly as written. "
            "Rules:\n"
            "- Preserve every line break exactly as it appears.\n"
            "- Do NOT translate, summarize, correct, or modify any text.\n"
            "- Keep original numbering (1., 2., a), b), क), அ), etc.).\n"
            "- The text may be in ANY Indian language or English or mixed — output it exactly as written.\n"
            "- Output ONLY the extracted text, nothing else."
        )

        response = model.generate_content([prompt, img])
        text = response.text.strip()

        pages.append({
            "page_number": page_num + 1,
            "raw_text": text
        })

        # Gemini free tier is 15 requests per minute. 
        # Adding a 4.5s sleep ensures we never hit a rate limit, even on large PDFs.
        if page_num < total_pages - 1:
            time.sleep(4.5)

    src_doc.close()
    log(f"Gemini Vision OCR done — {len(pages)} page(s) extracted")
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
# =========================================================

def find_question_paper_pages(pages: list, min_questions: int = 2) -> list:
    Q_LINE_NUM   = re.compile(r'^\s*\d+[\.\)]\s+.{15,}')
    Q_LINE_LATIN = re.compile(r'^\s*[a-d]\)\s+.{5,}', re.IGNORECASE)
    Q_LINE_DEVA  = re.compile(r'^\s*[क-घ]\)\s+.{5,}')

    MARK_ALLOCATION = re.compile(r'(?:\d+\s*[xX]\s*\d+\s*=?\s*\d*|\b\d{1,3}\s*$|\(\s*\d+\s*\))')
    SECTION_HEADER = re.compile(r'(?:SECTION\s*[-–]?\s*[A-Z]|PART\s*[-–]?\s*\d|भाग\s*[-–]?\s*\d|भाग\s*[-–]?\s*[१-९])', re.IGNORECASE)
    EXAM_INSTRUCTION = re.compile(r'(?:answer\s+all\s+questions|all\s+questions\s+are\s+compulsory|सभी\s*प्रश्न\s*अनिवार्य|assignment\s*code|TMA\b|कुल\s*अंक|words?\s+each|शब्दों\s*में)', re.IGNORECASE)
    ADMIN_PAGE_MARKERS = re.compile(r'(?:enrolment\s*number|enrollment\s*no|student\s*identity\s*card|regional\s*cent[er]+|study\s*cent[er]+|produced\s+on\s+demand|qr\s*code|registration\s*details|admission\s*status|father.?s\s*name|programme\s*registered|IGNOU\s*-\s*Student)', re.IGNORECASE)
    ANSWER_PAGE_MARKERS = re.compile(r'(?:उत्तर\s*[\-\:]|Ans\.?\s*[\-\:]|A\.\d|A\d+\s*[\-\:]|Teacher.?s\s*Signature|PAGE\s*NO[\.\:]?\s*\d*\s*DATE)', re.IGNORECASE)

    candidate_pages = []
    weak_pages = []

    for i, page in enumerate(pages):
        text  = page["raw_text"]
        lines = text.split("\n")

        q_count = sum(1 for line in lines if Q_LINE_NUM.match(line.strip()) or Q_LINE_LATIN.match(line.strip()) or Q_LINE_DEVA.match(line.strip()))

        if q_count < min_questions: continue
        if ADMIN_PAGE_MARKERS.search(text): continue
        if ANSWER_PAGE_MARKERS.search(text): continue

        has_strong_signal = bool(MARK_ALLOCATION.search(text) or SECTION_HEADER.search(text) or EXAM_INSTRUCTION.search(text))

        if has_strong_signal: candidate_pages.append(i)
        else: weak_pages.append(i)

    confirmed_set = set(candidate_pages)
    for i in weak_pages:
        if (i - 1) in confirmed_set or (i + 1) in confirmed_set:
            candidate_pages.append(i)
            confirmed_set.add(i)

    return sorted(candidate_pages)


# =========================================================
# EXTRACT OFFICIAL QUESTIONS
# =========================================================

def extract_official_questions_multi_page(pages: list, qp_page_indices: list) -> list:
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
                if current: all_questions.append({"text": current.strip(), "parent": None}); current = None
                continue

            sub_m = SUB_LATIN.match(stripped) or SUB_DEVA.match(stripped)
            if sub_m:
                if current: all_questions.append({"text": current.strip(), "parent": None}); current = None
                label = sub_m.group(1); body  = sub_m.group(2)
                parent_text = pending_parent if pending_parent else ""
                all_questions.append({"text": f"{parent_text} {label}) {body}".strip(), "parent": pending_parent})
                continue

            m = Q_START.match(stripped)
            if m:
                if current: all_questions.append({"text": current.strip(), "parent": None})
                current = stripped
                pending_parent = stripped if re.search(r'(?:लिखिए|following|:)\s*$', stripped, re.IGNORECASE) else None
                continue

            if current: current += " " + stripped

        if current: all_questions.append({"text": current.strip(), "parent": None})

    final_questions = []
    SUBPART_RE = re.compile(r'(?:^|\s)\(?([a-zक-घ])\)\s', re.UNICODE)

    for q in all_questions:
        text = q["text"]; matches = list(SUBPART_RE.finditer(text))
        if len(matches) >= 2:
            preamble = text[:matches[0].start()].strip()
            for idx, m in enumerate(matches):
                start = m.start(1); end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
                part_text = text[start:end].strip()
                final_questions.append(f"{preamble} {part_text}".strip() if preamble else part_text)
        else: final_questions.append(text)

    seen = set(); unique = []
    for q in final_questions:
        key = re.sub(r'\s+', ' ', q.lower().strip())
        if key not in seen: seen.add(key); unique.append(q)
    return unique


NOISE_RE = re.compile(r'(?:Teacher\'?s?\s*Signature|Tancher\'?s?\s*Signature|Facebook\'?s?\s*Signature|PAGE\s*NO|^\s*DATE\b|Neel?\s*Kamal|Neal?\s*Kamal|Need?\s*Komal|Nod\s*Komal|TAKMA\s*SINAN|^\s*\d{1,3}\s*$)', re.IGNORECASE)

def is_noise(line: str) -> bool:
    return bool(NOISE_RE.search(line))


# =========================================================
# SIMILARITY & BOUNDARY LOGIC
# =========================================================

def normalize(text: str) -> str:
    text = text.lower(); text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE); text = re.sub(r'\s+', ' ', text)
    return text.strip()

def similarity(a: str, b: str) -> float:
    wa = set(normalize(a).split()); wb = set(normalize(b).split())
    if not wa or not wb: return 0.0
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

def find_question_boundaries_by_similarity(answer_lines: list, questions: list, similarity_threshold: float = 0.30, window: int = 4) -> list:
    candidates_by_question = {}
    for i in range(len(answer_lines)):
        line_i = answer_lines[i].strip()
        if len(line_i) < 8: continue
        for w in range(1, window + 1):
            if i + w > len(answer_lines): break
            combined = " ".join(answer_lines[i + k].strip() for k in range(w) if answer_lines[i + k].strip())
            if len(combined) < 10: continue
            combined_clean = strip_leading_label(combined)
            for q in questions:
                q_clean = strip_leading_label(q)
                score = max(similarity(combined, q), similarity(combined_clean, q_clean))
                if score >= similarity_threshold:
                    candidates_by_question.setdefault(q, []).append({"question": q, "line_index": i, "span": w, "score": score})

    for q in candidates_by_question: candidates_by_question[q].sort(key=lambda c: -c["score"])

    final = []; last_line_index = -1
    for q in questions:
        for c in candidates_by_question.get(q, []):
            if c["line_index"] > last_line_index:
                final.append(c); last_line_index = c["line_index"]; break
    return final

def slice_raw_answers_by_boundaries(answer_lines: list, boundaries: list) -> list:
    qa_pairs = []
    for i, b in enumerate(boundaries):
        a_start = b["line_index"] + b.get("span", 1)
        a_end   = boundaries[i + 1]["line_index"] if i + 1 < len(boundaries) else len(answer_lines)
        raw = [answer_lines[j] for j in range(a_start, a_end) if answer_lines[j].strip() and not is_noise(answer_lines[j])]
        qa_pairs.append({"question": b["question"], "answer": " ".join(raw).strip()})
    return qa_pairs


# =========================================================
# GEMINI-BASED Q&A LINE DETECTION
# =========================================================

def build_numbered_line_dump(pages: list) -> list:
    line_index = []
    for page in pages:
        for line in page["raw_text"].split("\n"):
            line_index.append({"line_number": len(line_index), "page_number": page["page_number"], "text": line})
    return line_index

def ask_gemini_for_qa_lines(line_index: list, status_callback=None, chunk_size: int = 350):
    import google.generativeai as genai
    
    def log(msg):
        print(msg)
        if status_callback: status_callback(msg)

    model = get_gemini_model()
    total = len(line_index)
    all_results = []
    num_chunks = (total + chunk_size - 1) // chunk_size

    for ci in range(num_chunks):
        start = ci * chunk_size; end = min(start + chunk_size, total)
        chunk = line_index[start:end]
        numbered_text = "\n".join(f"[L{e['line_number']}] {e['text']}" for e in chunk)

        log(f"Gemini scanning lines {start}-{end} (chunk {ci+1}/{num_chunks})...")

        prompt = f"""You are scanning OCR text from a scanned exam answer booklet. The document may be in ANY Indian language (Hindi, Tamil, Telugu, etc.) or English or mixed.

Each line below is tagged with its line number like [L42].
Find every QUESTION and the START of its ANSWER in this chunk. 
A question is a numbered/lettered exam prompt (e.g. "1.", "Q.1", "क)", "(a)"). The answer is the handwritten response that follows (often after "Ans-", "उत्तर-", or just the next line).

Return ONLY valid JSON:
{{
  "items": [
    {{
      "question_id": "<e.g. Q1, Q9-a>",
      "question_start_line": <integer>,
      "answer_start_line": <integer or null>
    }}
  ]
}}
If no questions found, return {{"items": []}}. Do NOT output text content.

NUMBERED LINES:
{numbered_text}"""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        
        data = json.loads(response.text)
        items = data.get("items", [])
        log(f"  Chunk {ci+1}: Gemini reported {len(items)} item(s)")
        all_results.extend(items)
        
        if ci < num_chunks - 1:
            time.sleep(4.5) # Stay safely under free tier rate limits

    return all_results

def validate_and_clean_llm_items(raw_items: list, line_index: list, status_callback=None) -> list:
    def log(msg):
        print(msg)
        if status_callback: status_callback(msg)

    total_lines = len(line_index); cleaned = []; seen_start_lines = set()
    raw_items_sorted = sorted([it for it in raw_items if isinstance(it.get("question_start_line"), int)], key=lambda it: it["question_start_line"])
    last_start = -1

    for it in raw_items_sorted:
        qsl = it.get("question_start_line"); asl = it.get("answer_start_line")
        if qsl is None or not (0 <= qsl < total_lines): continue
        if qsl in seen_start_lines or qsl <= last_start: continue
        if len(line_index[qsl]["text"].strip()) < 2: continue
        if asl is not None and (not isinstance(asl, int) or not (0 <= asl < total_lines) or asl <= qsl): asl = None
        
        cleaned.append({"question_id": it.get("question_id", f"Q{len(cleaned)+1}"), "question_start_line": qsl, "answer_start_line": asl})
        seen_start_lines.add(qsl); last_start = qsl

    log(f"Validation: {len(cleaned)} of {len(raw_items)} Gemini items passed checks")
    return cleaned

def slice_qa_from_line_items(line_index: list, items: list) -> list:
    qa_pairs = []
    for i, item in enumerate(items):
        q_start = item["question_start_line"]; a_start = item["answer_start_line"] or q_start + 1
        a_end = items[i + 1]["question_start_line"] if i + 1 < len(items) else len(line_index)
        
        q_lines = [line_index[j]["text"] for j in range(q_start, min(a_start, len(line_index))) if line_index[j]["text"].strip()]
        a_lines = [line_index[j]["text"] for j in range(a_start, max(a_end, a_start)) if line_index[j]["text"].strip() and not is_noise(line_index[j]["text"])]
        
        qa_pairs.append({"question": " ".join(q_lines).strip(), "answer": " ".join(a_lines).strip()})
    return qa_pairs

def try_gemini_pipeline(pages: list, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback: status_callback(msg)

    if not get_api_key("GEMINI_API_KEY"): return None, "GEMINI_API_KEY is not set"

    try:
        line_index = build_numbered_line_dump(pages)
        log(f"Built line index: {len(line_index)} total lines")

        raw_items = ask_gemini_for_qa_lines(line_index, status_callback)
        if not raw_items: return None, "Gemini returned zero items"

        cleaned = validate_and_clean_llm_items(raw_items, line_index, status_callback)
        if len(cleaned) < 2: return None, f"Only {len(cleaned)} Gemini items passed validation"

        qa_pairs = slice_qa_from_line_items(line_index, cleaned)
        non_empty = [p for p in qa_pairs if p["answer"].strip()]
        if len(non_empty) < len(qa_pairs) * 0.5: return None, f"Too many empty answers sliced"

        log(f"Gemini pipeline succeeded — {len(qa_pairs)} Q-A pairs")
        return qa_pairs, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def run_regex_pipeline(pages: list, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback: status_callback(msg)

    qp_page_indices = find_question_paper_pages(pages)
    if not qp_page_indices: raise Exception("Could not detect question paper pages.")
    
    official_questions = extract_official_questions_multi_page(pages, qp_page_indices)
    if not official_questions: raise Exception("No questions parsed from question paper pages.")

    answer_page_indices = [i for i in range(len(pages)) if i not in qp_page_indices]
    answer_lines = [line for page in [pages[i] for i in answer_page_indices] for line in page["raw_text"].split("\n") if not is_noise(line)]

    boundaries = find_question_boundaries_by_similarity(answer_lines, official_questions)
    if not boundaries: raise Exception("Could not match any questions in answer pages.")
    
    return slice_raw_answers_by_boundaries(answer_lines, boundaries)


# =========================================================
# COMPLETE PIPELINE
# =========================================================

def process_pdf(file_input, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback: status_callback(msg)

    if isinstance(file_input, (str, Path)):
        file_bytes = Path(file_input).read_bytes(); file_name  = Path(file_input).name
    else:
        file_bytes = file_input.read(); file_name  = getattr(file_input, "name", "document.pdf")

    # Step 1: OCR (Gemini Vision - Free, All Indian Languages)
    pages = run_ocr(file_bytes, file_name, status_callback)

    # Step 2: Build OCR JSON
    log("Building OCR JSON...")
    ocr_json = build_ocr_json(pages)
    log(f"Total pages: {ocr_json['total_pages']}")

    # Step 3: Try Gemini-based line identification
    log("Attempting Gemini-based question/answer line detection...")
    qa_pairs, gemini_fail_reason = try_gemini_pipeline(pages, status_callback)

    if qa_pairs is not None:
        log(f"Done — {len(qa_pairs)} Q-A pairs (via Gemini)")
        return ocr_json, qa_pairs

    log(f"Gemini pipeline did not produce results. Reason: {gemini_fail_reason}")
    log("Falling back to regex/similarity pipeline...")

    try:
        qa_pairs = run_regex_pipeline(pages, status_callback)
    except Exception as regex_error:
        raise Exception(f"Both pipelines failed.\nGemini: {gemini_fail_reason}\nRegex: {regex_error}")

    log(f"Done — {len(qa_pairs)} Q-A pairs (via regex fallback)")
    return ocr_json, qa_pairs
