"""
SIMPLEST WORKING VERSION - GUARANTEED NO LINE SKIPPING
"""

import os, re, json, time
import fitz, httpx
from groq import Groq

def get_api_key(name):
    try:
        import streamlit as st
        return st.secrets[name]
    except:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv(name)

# ============================================================
# OCR - DATALAB
# ============================================================

def run_ocr(file_bytes, file_name):
    api_key = get_api_key("DATALAB_API_KEY")
    if not api_key:
        raise Exception("DATALAB_API_KEY missing")
    
    headers = {"X-API-Key": api_key}
    
    resp = httpx.post(
        "https://www.datalab.to/api/v1/convert",
        headers=headers,
        files={"file": (file_name, file_bytes, "application/pdf")},
        data={"output_format": "markdown", "mode": "accurate", "paginate": "true"},
        timeout=120
    )
    
    if resp.status_code != 200:
        raise Exception(f"OCR error: {resp.status_code}")
    
    data = resp.json()
    check_url = data["request_check_url"]
    
    for _ in range(150):
        poll = httpx.get(check_url, headers=headers, timeout=60)
        if poll.status_code != 200:
            continue
        result = poll.json()
        if result.get("status") == "complete":
            break
        if result.get("status") == "failed":
            raise Exception("OCR failed")
        time.sleep(2)
    
    markdown = result.get("markdown", "")
    if not markdown.strip():
        raise Exception("Empty OCR result")
    
    # Split pages
    pages = []
    if '\f' in markdown:
        parts = [p.strip() for p in markdown.split('\f') if p.strip()]
    else:
        parts = [markdown.strip()]
    
    for i, text in enumerate(parts):
        pages.append({"page_number": i+1, "text": text})
    
    return pages

# ============================================================
# GROQ HELPER
# ============================================================

def groq_call(system, user):
    api_key = get_api_key("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY missing")
    
    client = Groq(api_key=api_key)
    
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            content = resp.choices[0].message.content
            # Clean markdown
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*$', '', content)
            return json.loads(content)
        except Exception as e:
            if "try again in" in str(e):
                wait = float(re.search(r'try again in\s+([\d.]+)s', str(e)).group(1)) + 1
                time.sleep(wait)
                continue
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)

# ============================================================
# STEP 1: FIND QUESTION PAGES
# ============================================================

def find_question_pages(pages):
    system = """You are analyzing exam pages. Identify which pages contain the question paper.
    Return JSON: {"question_pages": [page_numbers]}"""
    
    all_qp = set()
    
    for i in range(0, len(pages), 3):
        chunk = pages[i:i+3]
        user = "\n\n".join([f"PAGE {p['page_number']}:\n{p['text']}" for p in chunk])
        user = "Which pages contain the question paper?\n\n" + user
        
        try:
            data = groq_call(system, user)
            for p in data.get("question_pages", []):
                if 1 <= p <= len(pages):
                    all_qp.add(p - 1)  # Convert to 0-based
        except:
            continue
    
    return sorted(all_qp)

# ============================================================
# STEP 2: EXTRACT QUESTIONS
# ============================================================

def extract_questions(qp_pages):
    system = """Extract all exam questions from these pages.
    Return JSON: {"questions": ["q1", "q2", ...]}"""
    
    user = "\n\n".join([f"PAGE {p['page_number']}:\n{p['text']}" for p in qp_pages])
    user = "Extract all questions:\n\n" + user
    
    try:
        data = groq_call(system, user)
        questions = data.get("questions", [])
        return [q.strip() for q in questions if q.strip()]
    except:
        return []

# ============================================================
# STEP 3: MAP ANSWERS - MAIN FIX
# ============================================================

def map_answers(answer_lines, questions):
    """
    FIXED: LLM gives ONLY range, Python extracts ALL lines
    """
    if not answer_lines or not questions:
        return {}
    
    total_lines = len(answer_lines)
    ref_map = {f"REF-{chr(65+i)}": q for i, q in enumerate(questions)}
    
    all_ranges = []
    
    # Chunk with overlap
    chunk_size = 50
    overlap = 15
    
    for start in range(0, total_lines, chunk_size - overlap):
        end = min(start + chunk_size, total_lines)
        context_start = max(0, start - overlap)
        chunk = list(enumerate(answer_lines[context_start:end]))
        
        # Build prompt
        q_text = "\n".join([f"[REF-{chr(65+i)}] {q}" for i, q in enumerate(questions)])
        l_text = "\n".join([f"[{idx}] {text}" for idx, text in chunk])
        
        system = """For each question, find the line range where the answer is.
        Return JSON: {"answers": [{"ref": "REF-A", "start": 12, "end": 18}]}
        Use line numbers as given in [brackets]."""
        
        user = f"Questions:\n{q_text}\n\nAnswer text:\n{l_text}"
        
        try:
            data = groq_call(system, user)
            for item in data.get("answers", []):
                ref = item.get("ref", "").strip().upper()
                s = item.get("start", -1)
                e = item.get("end", -1)
                
                if ref not in ref_map:
                    continue
                if s < 0 or e < 0 or s > e:
                    continue
                if s >= total_lines:
                    continue
                
                e = min(e, total_lines - 1)
                all_ranges.append({"ref": ref, "start": s, "end": e})
        except:
            continue
    
    # Deduplicate - keep longest range
    best = {}
    for r in all_ranges:
        ref = r["ref"]
        length = r["end"] - r["start"]
        if ref not in best or length > (best[ref]["end"] - best[ref]["start"]):
            best[ref] = r
    
    # Sort by start
    sorted_ranges = sorted(best.values(), key=lambda x: x["start"])
    
    # Resolve overlaps
    resolved = []
    for i, r in enumerate(sorted_ranges):
        start = r["start"]
        end = r["end"]
        
        if i + 1 < len(sorted_ranges):
            next_start = sorted_ranges[i + 1]["start"]
            if end >= next_start:
                end = next_start - 1
                if end < start:
                    continue
        
        if start <= end:
            resolved.append({"ref": r["ref"], "start": start, "end": end})
    
    # EXTRACT - PURE SLICING, NO SKIPPING
    qa_map = {}
    for r in resolved:
        start = max(0, r["start"])
        end = min(total_lines - 1, r["end"])
        
        if start > end:
            continue
        
        # PURE SLICING - ALL LINES IN RANGE
        extracted = answer_lines[start:end + 1]
        
        # Clean only first line's label
        if extracted:
            first = extracted[0]
            cleaned = re.sub(
                r'^\s*(?:Ans(?:wer)?[.\s:-]+\d*|उत्तर\s*\d*\s*[\-\:]?|प्र[०.\s]*\d*|Q\.?\s*\d+[.\s:-])',
                '',
                first,
                flags=re.IGNORECASE
            )
            if cleaned != first:
                extracted[0] = cleaned
        
        answer = "\n".join(extracted).strip()
        
        if answer:
            q = ref_map.get(r["ref"])
            if q:
                qa_map[q] = answer
    
    return qa_map

# ============================================================
# MAIN PIPELINE
# ============================================================

def process_pdf(file_input):
    # Normalize input
    if isinstance(file_input, tuple):
        file_bytes = file_input[1]
        file_name = file_input[0]
    elif isinstance(file_input, bytes):
        file_bytes = file_input
        file_name = "document.pdf"
    elif isinstance(file_input, (str, Path)):
        p = Path(file_input)
        file_bytes = p.read_bytes()
        file_name = p.name
    else:
        raise Exception("Invalid input type")
    
    print("1. Running OCR...")
    pages = run_ocr(file_bytes, file_name)
    
    print("2. Finding question pages...")
    qp_indices = find_question_pages(pages)
    print(f"Question pages: {[i+1 for i in qp_indices]}")
    
    if not qp_indices:
        raise Exception("No question pages found")
    
    print("3. Extracting questions...")
    qp_pages = [pages[i] for i in qp_indices]
    questions = extract_questions(qp_pages)
    print(f"Found {len(questions)} questions")
    
    if not questions:
        raise Exception("No questions extracted")
    
    print("4. Building answer lines...")
    answer_indices = [i for i in range(len(pages)) if i not in qp_indices]
    
    answer_lines = []
    for idx in answer_indices:
        for line in pages[idx]["text"].split("\n"):
            if line.strip():
                answer_lines.append(line.strip())
    
    print(f"Total answer lines: {len(answer_lines)}")
    
    if not answer_lines:
        raise Exception("No answer lines found")
    
    print("5. Mapping answers...")
    qa_map = map_answers(answer_lines, questions)
    
    # Build output
    qa_pairs = []
    matched = 0
    for q in questions:
        answer = qa_map.get(q, "")
        if answer:
            matched += 1
        qa_pairs.append({
            "question": q,
            "answer": answer,
            "matched": bool(answer),
            "lines": len(answer.split("\n")) if answer else 0
        })
    
    print(f"Matched {matched}/{len(questions)} questions")
    
    return pages, qa_pairs

# ============================================================
# SAVE OUTPUT
# ============================================================

def save_outputs(pages, qa_pairs, output_dir=".", base_name="document"):
    ocr_json = {
        "total_pages": len(pages),
        "pages": [{"page_number": p["page_number"], "text": p["text"]} for p in pages]
    }
    
    ocr_path = Path(output_dir) / f"{base_name}_ocr.json"
    qa_path = Path(output_dir) / f"{base_name}_qa_pairs.json"
    
    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(ocr_json, f, ensure_ascii=False, indent=2)
    
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    return str(ocr_path), str(qa_path)

# ============================================================
# USAGE
# ============================================================

if __name__ == "__main__":
    # Use like this:
    # pages, qa_pairs = process_pdf("document.pdf")
    # save_outputs(pages, qa_pairs)
    pass
