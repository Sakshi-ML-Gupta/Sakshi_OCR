import streamlit as st
import os
import json
import re
import subprocess
import shutil
import logging
from pypdf import PdfReader
from groq import Groq

# Suppress noisy standard library PDF warnings in terminal
logging.getLogger("pypdf").setLevel(logging.ERROR)

# ----------------------------------------------------------------------
# Page Setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="Automated Assignment Parser", page_icon="📝", layout="wide")

st.title("Automated University Assignment Q&A Pipeline")
st.write(
    "Upload a scanned student assignment booklet to run the entire OCR, "
    "classification, extraction, and answer-mapping process automatically."
)

# ----------------------------------------------------------------------
# Sidebar Controls
# ----------------------------------------------------------------------
st.sidebar.header("Pipeline Settings")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
groq_model = st.sidebar.selectbox(
    "Groq Model",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    index=0,
    help="Llama 3.3 70B is highly recommended for structured boundary mapping reasoning."
)

ocr_mode = st.sidebar.radio(
    "OCR System Mode",
    ["Datalab Chandra (Local CLI)", "PyPDF Reader Fallback"],
    index=1,
    help="Datalab Chandra will run if installed. Choose Fallback for standard PDF text reading."
)

# ----------------------------------------------------------------------
# Core Pipeline Processing Functions
# ----------------------------------------------------------------------

def run_fallback_ocr(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page_num": i + 1, "text": text})
    return pages


def run_chandra_ocr(pdf_bytes, filename, status):
    temp_pdf_path = f"temp_{filename}"
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_bytes)
        
    if "Fallback" in ocr_mode:
        status.write("🔄 OCR: Running PyPDF extraction fallback...")
        return run_fallback_ocr(temp_pdf_path)
        
    status.write("🔄 OCR: Launching local Datalab Chandra OCR model...")
    output_dir = "chandra_temp_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = ["chandra", temp_pdf_path, output_dir, "--batch-size", "1"]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        md_file_path = None
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".md"):
                    md_file_path = os.path.join(root, file)
                    break
        
        if md_file_path and os.path.exists(md_file_path):
            with open(md_file_path, "r", encoding="utf-8") as f:
                full_md_content = f.read()
            pages = full_md_content.split("\x0c")
            if len(pages) <= 1:
                pages = re.split(r'\n-+\n|\n==+=\n', full_md_content)
                
            processed_pages = []
            for i, page_text in enumerate(pages):
                processed_pages.append({"page_num": i + 1, "text": page_text.strip()})
            return processed_pages
        else:
            raise FileNotFoundError("Chandra execution finished but output file was missing.")
    except Exception as e:
        status.write(f"⚠️ Chandra execution failed: {str(e)}. Defaulting to standard reader.")
        return run_fallback_ocr(temp_pdf_path)
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


def classify_pages(pages, client, model, status):
    status.write("🔄 Classification: Categorizing booklet pages...")
    classified_pages = []
    
    for page in pages:
        page_num = page['page_num']
        text_content = page['text']
        # Optimization: Limit context tokens to speed up classification
        truncated_text = text_content[:1200]
        
        prompt = f"""
You are an expert document classifier. Your job is to classify this single page from a university student assignment/exam booklet.
Classes:
1. "Question Paper" - Contains printed list of official questions, marks, and instructions.
2. "Admin/Cover" - Student info, registration forms, evaluator marking sheets, feed back pages.
3. "Answer Page" - Contains student's handwritten answers.

Page Content:
---
{truncated_text}
---

Respond ONLY with a JSON object in this format:
{{
  "classification": "Question Paper" | "Admin/Cover" | "Answer Page",
  "reason": "Brief reason"
}}
"""
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            res_json = json.loads(response.choices[0].message.content)
            classified_pages.append({
                "page_num": page_num,
                "text": text_content,
                "classification": res_json.get("classification", "Answer Page"),
                "reason": res_json.get("reason", "")
            })
        except Exception:
            classified_pages.append({
                "page_num": page_num,
                "text": text_content,
                "classification": "Answer Page",
                "reason": "Error fallback"
            })
    return classified_pages


def extract_questions(classified_pages, client, model, status):
    status.write("🔄 Extraction: Identifying canonical question list...")
    
    # Identify pages classified as Question Paper
    qp_pages = [p for p in classified_pages if p['classification'] == "Question Paper"]
    
    # AUTOMATED FALLBACK: If LLM missed Question Paper page labels, assume Pages 1 and 2 are the target source
    if not qp_pages:
        status.write("⚠️ Heuristic Fallback: No 'Question Paper' pages classified. Analyzing pages 1-2 as fallback...")
        qp_pages = [p for p in classified_pages if p['page_num'] in [1, 2]]
        
    qp_text = "\n\n".join([f"--- Page {p['page_num']} ---\n{p['text']}" for p in qp_pages])
    
    prompt = f"""
Identify and list all distinct questions and sub-questions from the text.
Rules:
1. Sub-questions must be extracted as separate list items (e.g. split Q1 into Q1.(a), Q1.(b)). Do not group them.
2. Maintain the original question text.
3. Respond ONLY with a JSON object containing a "questions" array.

Document Text:
{qp_text}

JSON Output Format:
{{
  "questions": [
    {{ "id": "1.(a)", "text": "Question detail text..." }}
  ]
}}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        res_json = json.loads(response.choices[0].message.content)
        questions = res_json.get("questions", [])
        status.write(f"✅ Extracted {len(questions)} individual questions/sub-questions.")
        return questions
    except Exception as e:
        status.write(f"⚠️ Extraction failed: {str(e)}")
        return []


def verify_integrity(sliced_lines, start_idx, end_idx, all_lines):
    expected_lines = all_lines[start_idx : end_idx + 1]
    if len(sliced_lines) != len(expected_lines):
        return False
    for sl, el in zip(sliced_lines, expected_lines):
        if sl['idx'] != el['idx'] or sl['text'] != el['text'] or sl['page_num'] != el['page_num']:
            return False
    return True


def map_answers(classified_pages, questions, client, model, status):
    status.write("🔄 Mapping: Aligning answers to extracted questions...")
    
    # Isolate actual answer pages
    answer_pages = [p for p in classified_pages if p['classification'] == "Answer Page"]
    if not answer_pages:
        status.write("⚠️ No 'Answer Page' directories available to map.")
        return []
        
    # Build continuous line coordinate canvas
    all_lines = []
    global_idx = 0
    for page in answer_pages:
        page_num = page['page_num']
        lines = page['text'].split('\n')
        for line in lines:
            all_lines.append({"idx": global_idx, "text": line, "page_num": page_num})
            global_idx += 1
            
    # Remove blank lines to compress context token size
    llm_lines_repr = []
    for line_obj in all_lines:
        cleaned_text = line_obj['text'].strip()
        if cleaned_text:
            llm_lines_repr.append(f"L{line_obj['idx']}: {cleaned_text}")
            
    llm_document_context = "\n".join(llm_lines_repr)
    
    qa_results = []
    last_end_index = 0
    
    for i, q in enumerate(questions):
        q_id = q.get("id", f"Q{i+1}")
        q_text = q.get("text", "")
        
        status.write(f"➡️ Locating boundaries for Question {q_id}...")
        
        prompt = f"""
Find the exact start and end line indices corresponding to the student's handwritten answer.
- Rely on line numbers ('L<number>').
- The student likely answered this near or after index {last_end_index} (sequential mapping).
- Respond ONLY with JSON coordinates. Do not summarize or paraphrase the answer text.

Target Question:
"{q_text}"

Student OCR Transcript lines:
{llm_document_context}

JSON Output Format:
{{
  "start_index": <integer_index>,
  "end_index": <integer_index>,
  "explanation": "Brief rationale"
}}
"""
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            res_json = json.loads(response.choices[0].message.content)
            
            start_idx = int(res_json.get("start_index", -1))
            end_idx = int(res_json.get("end_index", -1))
            
            if (start_idx == -1 or end_idx == -1 or start_idx > end_idx or 
                start_idx >= len(all_lines) or end_idx >= len(all_lines)):
                qa_results.append({
                    "question_id": q_id,
                    "question": q_text,
                    "answer": "[No valid student answer bounds identified]",
                    "pages": [],
                    "confidence": "out_of_bounds"
                })
                continue
                
            # Perform Python Slicing on the target raw data
            sliced_lines = all_lines[start_idx : end_idx + 1]
            
            # Non-negotiable Integrity Check Verification
            if not verify_integrity(sliced_lines, start_idx, end_idx, all_lines):
                qa_results.append({
                    "question_id": q_id,
                    "question": q_text,
                    "answer": "[Error: Slicing integrity validation failed]",
                    "pages": [],
                    "confidence": "integrity_failed"
                })
                continue
                
            raw_answer_text = "\n".join([line['text'] for line in sliced_lines]).strip()
            spanned_pages = list(sorted(set([line['page_num'] for line in sliced_lines])))
            
            qa_results.append({
                "question_id": q_id,
                "question": q_text,
                "answer": raw_answer_text,
                "pages": spanned_pages,
                "start_line": start_idx,
                "end_line": end_idx,
                "confidence": "high",
                "explanation": res_json.get("explanation", "")
            })
            
            # Step the index tracker sequentially
            last_end_index = end_idx
            
        except Exception as e:
            qa_results.append({
                "question_id": q_id,
                "question": q_text,
                "answer": f"[Mapping error occurred: {str(e)}]",
                "pages": [],
                "confidence": "failed"
            })
            
    return qa_results

# ----------------------------------------------------------------------
# Pipeline Execution Flow
# ----------------------------------------------------------------------
if uploaded_file:
    if not groq_api_key:
        st.warning("⚠️ Please provide a Groq API Key in the sidebar settings to begin.")
    else:
        client = Groq(api_key=groq_api_key)
        
        # Use a single automated status widget to process end-to-end with no intermediate button clicks
        with st.status("Processing Pipeline (Running Automatically)...", expanded=True) as status:
            
            # Stage 1: OCR
            raw_pages = run_chandra_ocr(uploaded_file.getvalue(), uploaded_file.name, status)
            
            # Stage 2: Page Classification
            classified_pages = classify_pages(raw_pages, client, groq_model, status)
            
            # Stage 2b: Question Extraction (with auto-fallback logic inside)
            extracted_questions = extract_questions(classified_pages, client, groq_model, status)
            
            # Stage 3: Sequential Mapping
            if extracted_questions:
                mapped_results = map_answers(classified_pages, extracted_questions, client, groq_model, status)
            else:
                mapped_results = []
                status.write("❌ Pipeline finished early: No questions were extracted to match.")
                
            status.update(label="Pipeline Processing Complete!", state="complete", expanded=False)

        # Display output panels once the pipeline concludes
        if mapped_results:
            st.success("Automated Processing Completed Successfully!")
            
            tab1, tab2, tab3 = st.tabs(["Mapped Q&A Results", "Classification Diagnostics", "Raw OCR Data"])
            
            with tab1:
                st.subheader("Extracted Q&A Pairs")
                for item in mapped_results:
                    with st.expander(f"Question {item['question_id']} (Pages Spanned: {item['pages']})"):
                        st.markdown(f"**Question:**\n`{item['question']}`")
                        st.markdown("**Mapped Raw Answer Text:**")
                        st.code(item['answer'], language="text")
                        
            with tab2:
                st.subheader("Page Classifications")
                st.json(classified_pages)
                
            with tab3:
                st.subheader("Raw Page Transcripts")
                st.json(raw_pages)
                
            # Download actions
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Q&A Pairs (JSON)",
                    data=json.dumps(mapped_results, indent=2),
                    file_name="extracted_qa_pairs.json",
                    mime="application/json"
                )
            with col2:
                st.download_button(
                    label="Download OCR Text (JSON)",
                    data=json.dumps(raw_pages, indent=2),
                    file_name="ocr_raw_pages.json",
                    mime="application/json"
                )
