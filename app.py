import streamlit as st

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="Automated Assignment Parser", page_icon="📝", layout="wide")

import os
import json
import re
import subprocess
import shutil
import logging
from pypdf import PdfReader
from groq import Groq

# Suppress noisy standard library PDF warnings
logging.getLogger("pypdf").setLevel(logging.ERROR)

# ----------------------------------------------------------------------
# Stage 1: OCR Utility Functions
# ----------------------------------------------------------------------

def run_fallback_ocr(pdf_path):
    """Fallback standard PDF text extraction when local Chandra is absent."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({
            "page_num": i + 1,
            "text": text
        })
    return pages


def run_chandra_ocr(pdf_bytes, filename, status_bar):
    """Runs Stage 1 OCR via Datalab Chandra CLI, falls back on errors."""
    temp_pdf_path = f"temp_{filename}"
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_bytes)
        
    if "Fallback" in ocr_mode:
        status_bar.write("🔄 OCR Stage: Running PyPDF extraction fallback...")
        return run_fallback_ocr(temp_pdf_path)
        
    status_bar.write("🔄 OCR Stage: Running local Datalab Chandra OCR model...")
    output_dir = "chandra_temp_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the Chandra CLI tool (locally on your VM/Hosted Server)
    cmd = ["chandra", temp_pdf_path, output_dir, "--batch-size", "1"]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Locate the output Markdown file
        md_file_path = None
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".md"):
                    md_file_path = os.path.join(root, file)
                    break
        
        if md_file_path and os.path.exists(md_file_path):
            with open(md_file_path, "r", encoding="utf-8") as f:
                full_md_content = f.read()
                
            # Chandra separates pages using form feeds
            pages = full_md_content.split("\x0c")
            if len(pages) <= 1:
                # Markdown horizontal rule split fallback
                pages = re.split(r'\n-+\n|\n==+=\n', full_md_content)
                
            processed_pages = []
            for i, page_text in enumerate(pages):
                processed_pages.append({
                    "page_num": i + 1,
                    "text": page_text.strip()
                })
            return processed_pages
        else:
            raise FileNotFoundError("Chandra completed successfully but no output .md was found.")
            
    except Exception as e:
        status_bar.write(f"⚠️ Local Chandra OCR execution not available: {str(e)}. Defaulting to standard reader.")
        return run_fallback_ocr(temp_pdf_path)
    finally:
        # Clean up temporary files
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

# ----------------------------------------------------------------------
# Stage 2: Page Classification & Question Extraction
# ----------------------------------------------------------------------

def classify_pages(pages, client, model, status_bar):
    """Stage 2: Classify pages (Question Paper vs Admin/Cover vs Answer Page)"""
    status_bar.write("🔄 Stage 2: Classifying pages...")
    classified_pages = []
    
    for page in pages:
        page_num = page['page_num']
        text_content = page['text']
        
        # Token Optimization: Limit classification input text to speed up processing
        truncated_text = text_content[:1200]
        
        prompt = f"""
You are an expert document classifier. Your job is to classify this single page from a university student assignment/exam booklet.
The three possible classifications are:
1. "Question Paper" - Contains the list of printed official questions to be answered, with marks and instructions.
2. "Admin/Cover" - Cover sheets, grade logs, student verification forms.
3. "Answer Page" - Written answers of the student.

Page Content:
---
{truncated_text}
---

Respond ONLY with a JSON object in this exact format:
{{
  "classification": "Question Paper" | "Admin/Cover" | "Answer Page",
  "reason": "Brief explanation"
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
                "reason": "Fallback due to API error"
            })
    return classified_pages


def extract_questions(classified_pages, client, model, status_bar):
    """Stage 2b: Extracts the flat list of questions/sub-questions."""
    status_bar.write("🔄 Stage 2b: Identifying canonical question list...")
    
    # Identify pages classified as Question Paper
    qp_pages = [p for p in classified_pages if p['classification'] == "Question Paper"]
    
    # Automated Fallback: If no pages are classified as Question Paper, assume Pages 1 and 2 contain them
    if not qp_pages:
        status_bar.write("⚠️ Fallback: No 'Question Paper' pages classified. Scanning pages 1 and 2 for questions...")
        qp_pages = [p for p in classified_pages if p['page_num'] in [1, 2]]
        
    qp_text = "\n\n".join([f"--- Page {p['page_num']} ---\n{p['text']}" for p in qp_pages])
    
    prompt = f"""
Analyze the following text and extract all distinct questions and sub-questions.
Rules:
1. Sub-questions must be extracted as separate list entries (e.g. split Q1 into Q1.(a), Q1.(b)). Do not group them.
2. Retain original question marks/weights if present.
3. Respond ONLY with a JSON object containing a "questions" array.

Document Text:
{qp_text}

JSON Output Format:
{{
  "questions": [
    {{ "id": "1.(a)", "text": "Question text here..." }}
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
        status_bar.write(f"✅ Extracted {len(questions)} individual questions/sub-questions.")
        return questions
    except Exception as e:
        status_bar.write(f"⚠️ Question extraction failed: {str(e)}")
        return []

# ----------------------------------------------------------------------
# Stage 3: Sequential Answer Mapping & Slicing
# ----------------------------------------------------------------------

def verify_integrity(sliced_lines, start_idx, end_idx, all_lines):
    """Core Integrity Check: Validates that Python sliced array matches indices strictly."""
    expected_lines = all_lines[start_idx : end_idx + 1]
    if len(sliced_lines) != len(expected_lines):
        return False
    for sl, el in zip(sliced_lines, expected_lines):
        if sl['idx'] != el['idx'] or sl['text'] != el['text'] or sl['page_num'] != el['page_num']:
            return False
    return True


def map_answers(classified_pages, questions, client, model, status_bar):
    """Stage 3: Sequential Answer Mapping using LLM Line Coordinates & Python Slicing."""
    status_bar.write("🔄 Stage 3: Mapping answers...")
    
    # Isolate answer pages
    answer_pages = [p for p in classified_pages if p['classification'] == "Answer Page"]
    if not answer_pages:
        status_bar.write("⚠️ No 'Answer Page' pages detected to extract answers from.")
        return []
        
    # Build continuous line coordinate canvas
    all_lines = []
    global_idx = 0
    for page in answer_pages:
        page_num = page['page_num']
        lines = page['text'].split('\n')
        for line in lines:
            all_lines.append({
                "idx": global_idx,
                "text": line,
                "page_num": page_num
            })
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
        
        status_bar.write(f"➡️ Mapping Answer for Question {q_id}...")
        
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
                    "answer": "[No valid student answer boundaries matched]",
                    "pages": [],
                    "confidence": "out_of_bounds"
                })
                continue
                
            # Perform Python Slicing on raw list
            sliced_lines = all_lines[start_idx : end_idx + 1]
            
            # Non-negotiable Integrity Check
            if not verify_integrity(sliced_lines, start_idx, end_idx, all_lines):
                qa_results.append({
                    "question_id": q_id,
                    "question": q_text,
                    "answer": "[Error: Slicing integrity verification failed]",
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
            
            # Set tracking point sequentially
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
# Sidebar Controls & Initialization (Declared globally before flow)
# ----------------------------------------------------------------------
st.sidebar.header("Configuration")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
groq_model = st.sidebar.selectbox(
    "Groq Model",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    index=0
)
ocr_mode = st.sidebar.radio(
    "OCR Mode",
    ["Datalab Chandra (Local CLI)", "PyPDF Reader Fallback"],
    index=1
)

# Initialize the global file uploader widget cleanly
uploaded_file = st.file_uploader("Upload Scanned PDF Answer Booklet", type=["pdf"])

# ----------------------------------------------------------------------
# Automated Execution Flow Block
# ----------------------------------------------------------------------
if uploaded_file:
    if not groq_api_key:
        st.warning("⚠️ Please provide a Groq API Key in the sidebar settings to begin.")
    else:
        client = Groq(api_key=groq_api_key)
        
        # Runs end-to-end automatically without intermediate button presses
        with st.status("Processing Pipeline...", expanded=True) as status:
            
            # Stage 1: Run OCR on upload
            raw_pages = run_chandra_ocr(uploaded_file.getvalue(), uploaded_file.name, status)
            
            # Stage 2: Page Classification
            classified_pages = classify_pages(raw_pages, client, groq_model, status)
            
            # Stage 2b: Question Extraction (with fallback context handler)
            extracted_questions = extract_questions(classified_pages, client, groq_model, status)
            
            # Stage 3: Sequential Mapping
            if extracted_questions:
                mapped_results = map_answers(classified_pages, extracted_questions, client, groq_model, status)
            else:
                mapped_results = []
                status.write("❌ Processing interrupted: No questions were extracted to evaluate.")
                
            status.update(label="Pipeline Processing Complete!", state="complete", expanded=False)

        # Output Tab Panels
        if mapped_results:
            st.success("Automated processing complete!")
            
            tab1, tab2, tab3 = st.tabs(["Mapped Q&A Results", "Classification Log", "Raw OCR Data"])
            
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
