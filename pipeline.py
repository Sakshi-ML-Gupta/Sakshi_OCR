import os
import io
import re
import json
import time
import threading
import fitz
import httpx
from pathlib import Path

# =========================================================
# API KEYS & SETUP
# =========================================================

def get_api_key(name):
    try:
        import streamlit as st
        return st.secrets[name]
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv(name)


def _normalize_file_input(file_input, default_name="document.pdf"):
    if isinstance(file_input, tuple):
        if len(file_input) < 2:
            raise ValueError(f"Tuple input requires at least 2 items")
        return bytes(file_input[1]), Path(str(file_input[0])).name
    if isinstance(file_input, (bytes, bytearray)):
        return bytes(file_input), default_name
    if isinstance(file_input, (str, Path)):
        p = Path(file_input)
        return p.read_bytes(), p.name
    if hasattr(file_input, "read"):
        return bytes(file_input.read()), getattr(file_input, "name", default_name)
    raise TypeError(f"Unsupported file_input type: {type(file_input).__name__}")


_groq_call_lock = threading.Lock()
GROQ_MODEL = "openai/gpt-oss-120b"


# =========================================================
# OCR ENGINES (DATALAB CHANDRA)
# =========================================================

DATALAB_BASE_URL = "https://www.datalab.to"
PAGE_BREAK_PATTERNS = [
    re.compile(r'\n?\{(\d+)\}-{3,}\n?'),
    re.compile(r'\n?-{2,}\{(\d+)\}-{2,}\n?'),
    re.compile(r'\n-{3,}\s*Page\s*\d+\s*-{3,}\n', re.IGNORECASE),
    re.compile(r'\n\s*\[PAGE\s*(\d+)\]\s*\n', re.IGNORECASE),
]

def run_ocr(file_content: bytes, file_name: str, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    api_key = get_api_key("DATALAB_API_KEY")
    if not api_key:
        raise Exception("DATALAB_API_KEY missing.")

    headers = {"X-API-Key": api_key}
    log(f"Submitting PDF to Datalab OCR ({len(file_content)/(1024*1024):.1f}MB)...")

    resp = httpx.post(
        f"{DATALAB_BASE_URL}/api/v1/convert",
        headers=headers,
        files={"file": (file_name, file_content, "application/pdf")},
        data={"output_format": "markdown", "mode": "accurate", "paginate": "true"},
        timeout=120
    )

    if resp.status_code != 200:
        raise Exception(f"Datalab Error {resp.status_code}: {resp.text}")

    check_url = resp.json()["request_check_url"]
    log("Polling OCR progress...")

    for _ in range(150):
        poll = httpx.get(check_url, headers=headers, timeout=60).json()
        if poll.get("status") == "complete":
            break
        if poll.get("status") == "failed":
            raise Exception(f"OCR failed: {poll.get('error')}")
        time.sleep(2)
    else:
        raise Exception("OCR timed out.")

    markdown = poll.get("markdown") or ""
    
    # Extract page splits accurately
    parts = []
    start = 0
    for pattern in PAGE_BREAK_PATTERNS:
        matches = list(pattern.finditer(markdown))
        if matches:
            for m in matches:
                parts.append(markdown[start:m.start()].strip())
                start = m.end()
            parts.append(markdown[start:].strip())
            break
            
    if not parts:
        parts = [p.strip() for p in markdown.split('\f') if p.strip()] or [markdown.strip()]

    log(f"OCR Complete: {len(parts)} page(s) extracted.")
    return [{"page_number": idx + 1, "raw_text": text} for idx, text in enumerate(parts)]


# =========================================================
# CONTROLLED LLM ENGINE WITH RATE LIMITING
# =========================================================

def _call_groq_safe(system_prompt: str, user_prompt: str, log=print):
    from groq import Groq
    client = Groq(api_key=get_api_key("GROQ_API_KEY"))

    for attempt in range(1, 5):
        try:
            with _groq_call_lock:
                res = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
            # Mandatory 2.5s breather to respect Groq TPM
            time.sleep(2.5)
            content = res.choices[0].message.content.strip()
            content = re.sub(r'^```(?:json)?\s*\n?|\n?```\s*$', '', content)
            return json.loads(content)

        except Exception as e:
            wait = attempt * 12
            log(f"Groq API backoff (Attempt {attempt}): {e}. Waiting {wait}s...")
            time.sleep(wait)

    raise Exception("Groq execution failed after retries.")


# =========================================================
# QUESTION PAPER IDENTIFICATION & EXTRACTION
# =========================================================

QP_IDENTIFY_PROMPT = """Analyze booklet pages and detect Question Paper pages.
Return JSON:
{
  "qp_page_numbers": [1, 2]
}"""

QP_EXTRACT_PROMPT = """You are analyzing Question Paper pages. Extract all official printed questions in order.
Separate sub-questions (e.g., Q1(a), Q1(b)) into clear individual list items.

Return JSON:
{
  "questions": [
    "Q1(a) Question text...",
    "Q1(b) Question text...",
    "Q2 Question text..."
  ]
}"""

def extract_questions(pages: list, log=print) -> tuple:
    # Stage 1: Identify QP Pages
    all_pages_text = "\n\n".join([f"--- PAGE {p['page_number']} ---\n{p['raw_text'][:600]}" for p in pages])
    res = _call_groq_safe(QP_IDENTIFY_PROMPT, f"Pages overview:\n{all_pages_text}", log=log)
    
    qp_nums = set(res.get("qp_page_numbers", []))
    qp_indices = [p - 1 for p in qp_nums if 1 <= p <= len(pages)]

    if not qp_indices:
        # Fallback: Assume Page 1 & 2 are QP
        qp_indices = [0] if len(pages) == 1 else [0, 1]

    log(f"Question Paper identified on Page(s): {[i+1 for i in qp_indices]}")

    # Stage 2: Extract Canonical Question Titles
    qp_text = "\n\n".join([f"--- PAGE {pages[i]['page_number']} ---\n{pages[i]['raw_text']}" for i in qp_indices])
    q_res = _call_groq_safe(QP_EXTRACT_PROMPT, f"Question Paper Text:\n{qp_text}", log=log)
    
    questions = q_res.get("questions", [])
    log(f"Extracted Total Canonical Questions: {len(questions)}")
    
    return qp_indices, questions


# =========================================================
# PAGE-BY-PAGE FULL COVERAGE ANSWER MAPPING
# =========================================================

ANSWER_MAPPING_PROMPT = """You are an answer sheet evaluator. Map the provided student answer text to the official question titles.

Rules:
1. Extract verbatim text written by student for each matched question.
2. A single question's answer can span across multiple pages.
3. Ignore header/footer/noise (e.g. Page Nos, Teacher Signatures).

Return JSON:
{
  "answers": [
    {
      "question_index": 0,
      "verbatim_answer": "Student answer text for Question 0..."
    }
  ]
}"""

def map_all_answers_page_batching(pages: list, qp_indices: list, questions: list, log=print) -> dict:
    answer_pages = [p for i, p in enumerate(pages) if i not in qp_indices]
    
    # Initialize empty slots for EVERY single extracted question
    full_answers = {q: [] for q in questions}
    
    q_list_formatted = "\n".join([f"INDEX [{idx}]: {q_text}" for idx, q_text in enumerate(questions)])

    # Batch process 3 pages at a time (Sequential Page Preservation)
    BATCH_SIZE = 3
    for i in range(0, len(answer_pages), BATCH_SIZE):
        batch = answer_pages[i : i + BATCH_SIZE]
        batch_page_nums = [p["page_number"] for p in batch]
        
        log(f"Processing Student Answer Batch: Pages {batch_page_nums}...")

        batch_text = "\n\n".join([f"=== PAGE {p['page_number']} ===\n{p['raw_text']}" for p in batch])
        
        user_prompt = f"OFFICIAL QUESTIONS:\n{q_list_formatted}\n\nSTUDENT ANSWER PAGES:\n{batch_text}"

        try:
            mapped_res = _call_groq_safe(ANSWER_MAPPING_PROMPT, user_prompt, log=log)
            
            for item in mapped_res.get("answers", []):
                q_idx = item.get("question_index")
                ans_text = str(item.get("verbatim_answer", "")).strip()

                if isinstance(q_idx, int) and 0 <= q_idx < len(questions) and ans_text:
                    q_title = questions[q_idx]
                    full_answers[q_title].append(ans_text)

        except Exception as e:
            log(f"Error on Batch Pages {batch_page_nums}: {e}")

    # Combine text snippets collected across all page batches
    final_qa_map = {}
    for q_title, snippets in full_answers.items():
        if snippets:
            # Combine snippets cleanly avoiding absolute duplicates
            combined = "\n\n".join(dict.fromkeys(snippets))
            final_qa_map[q_title] = combined
        else:
            final_qa_map[q_title] = ""

    return final_qa_map


# =========================================================
# MAIN PIPELINE
# =========================================================

def process_pdf(file_input, status_callback=None):
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    file_bytes, file_name = _normalize_file_input(file_input)
    
    # Step 1: Run OCR
    pages = run_ocr(file_bytes, file_name, status_callback)
    ocr_json = {"total_pages": len(pages), "pages": [{"page_number": p["page_number"], "text": p["raw_text"]} for p in pages]}

    # Step 2: Extract Question Paper and List of Questions
    qp_indices, questions = extract_questions(pages, log=log)

    if not questions:
        raise Exception("Failed to extract questions from the Question Paper.")

    # Step 3: Run Full Page-Batch Answer Mapping
    qa_map = map_all_answers_page_batching(pages, qp_indices, questions, log=log)

    # Step 4: Build Comprehensive Q-A Pairs
    qa_pairs = []
    for q in questions:
        ans = qa_map.get(q, "")
        qa_pairs.append({
            "question": q,
            "answer": ans,
            "matched": bool(ans.strip())
        })

    log(f"Process Finished: {sum(1 for x in qa_pairs if x['matched'])}/{len(questions)} Questions matched with Answers.")
    return ocr_json, qa_pairs
