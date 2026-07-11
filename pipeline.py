import streamlit as st
import os
import json
import re
import subprocess
import shutil
from pypdf import PdfReader
from groq import Groq

# ----------------------------------------------------------------------
# Page Configurations & Setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="Assignment Q&A Parser", page_icon="📝", layout="wide")

# Install missing libraries instructions
# pip install streamlit groq pypdf chandra-ocr

st.title("University Assignment booklet Q&A Processing Pipeline")
st.write(
    "Upload scanned PDF answer booklets to process them using **Datalab Chandra OCR** and **Groq LLM**."
)

# ----------------------------------------------------------------------
# Sidebar / Settings Panel
# ----------------------------------------------------------------------
st.sidebar.header("Configuration Settings")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
groq_model = st.sidebar.selectbox(
    "Groq Model Selection",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    index=0,
    help="Llama 3.3 70B is recommended for complex reasoning steps like answer mapping."
)

ocr_mode = st.sidebar.radio(
    "OCR Model Mode",
    ["Datalab Chandra (Local CLI)", "PyPDF Fallback (For CPU/Local Testing)"],
    index=1,
    help="Choose Datalab Chandra if you have chandra-ocr set up. Select Fallback to test without model weight downloads."
)

# ----------------------------------------------------------------------
# Core Functions
# ----------------------------------------------------------------------

def update_status(logs_container, message, level="info"):
    """Callback function to push clean progress updates to the UI."""
    if level == "info":
        logs_container.markdown(f"ℹ️ {message}")
    elif level == "warning":
        logs_container.markdown(f"⚠️ **{message}**")
    elif level == "success":
        logs_container.markdown(f"✅ **{message}**")
    elif level == "error":
        logs_container.markdown(f"❌ **{message}**")


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


def run_chandra_ocr(pdf_bytes, filename, logs_callback):
    """Runs Stage 1 OCR via Datalab Chandra CLI, falls back on errors."""
    temp_pdf_path = f"temp_{filename}"
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_bytes)
        
    if "Fallback" in ocr_mode:
        logs_callback("Using standard PDF extraction fallback mode.", "info")
        return run_fallback_ocr(temp_pdf_path)
        
    logs_callback("Invoking local Datalab Chandra OCR model...", "info")
    output_dir = "chandra_temp_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the Chandra CLI tool (designed to run locally on your VM or Hosted Server)
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
                
            # Chandra separates pages using form feeds or markdown line dividers
            pages = full_md_content.split("\x0c") # Form feed standard split
            if len(pages) <= 1:
                # Try Markdown horizontal rules fallback split
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
        logs_callback(f"Chandra OCR local runtime error or not found: {str(e)}. Defaulting to standard text reader.", "warning")
        return run_fallback_ocr(temp_pdf_path)
    finally:
        # Cleanup temporary files
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


def classify_pages(pages, client, model, logs_callback):
    """Stage 2: Page Classification (Question Paper vs Admin/Cover vs Answer Page)"""
    classified_pages = []
    logs_callback("Starting page classification task...", "info")
    
    for page in pages:
        page_num = page['page_num']
        text_content = page['text']
        
        # Optimization: Truncate context to classification indicators (first 1200 chars)
        # This reduces token costs and execution delays significantly.
        truncated_text = text_content[:1200]
        
        prompt = f"""
You are an expert document classifier. Your job is to classify this page of a university student assignment/exam booklet.
The three possible classifications are:
1. "Question Paper" - Contains the list of printed official questions to be answered, with marks and exam instructions.
2. "Admin/Cover" - Contains student info sheets, roll number, course code, grade feedback forms, or blank spaces.
3. "Answer Page" - Contains student's handwritten answers to the assignment.

Analyze the page content:
---
{truncated_text}
---

Respond ONLY with a valid JSON object in this exact schema:
{{
  "classification": "Question Paper" | "Admin/Cover" | "Answer Page",
  "reason": "Brief explanation for the decision."
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
            classification = res_json.get("classification", "Answer Page")
            reason = res_json.get("reason", "")
            
            classified_pages.append({
                "page_num": page_num,
                "text": text_content,
                "classification": classification,
                "reason": reason
            })
            logs_callback(f"Page {page_num} classified as: **{classification}** ({reason})", "info")
        except Exception as e:
            classified_pages.append({
                "page_num": page_num,
                "text": text_content,
                "classification": "Answer Page",
                "reason": f"Fallback due to error: {str(e)}"
            })
            logs_callback(f"Page {page_num} set to default 'Answer Page' due to an LLM error.", "warning")
            
    return classified_pages


def extract_questions(classified_pages, client, model, logs_callback):
    """Stage 2: Extracts the flat canonical list of questions/sub-questions."""
    logs_callback("Starting Question Paper analysis...", "info")
    
    qp_pages = [p for p in classified_pages if p['classification'] == "Question Paper"]
    if not qp_pages:
        logs_callback("No Question Paper pages detected. Manual entry required or PDF did not contain QP.", "warning")
        return []
        
    qp_text = "\n\n".join([f"--- QP Page {p['page_num']} ---\n{p['text']}" for p in qp_pages])
    
    prompt = f"""
Analyze the following university Question Paper text and extract a structured list of ALL questions.

Rules:
1. Sub-questions must be split into separate entries (e.g., split Q1 into Q1.(i), Q1.(ii) etc. as unique items). Do not merge them!
2. Preserve marks, weights, and detailed texts.
3. Return ONLY a JSON object containing a flat "questions" array.

Question Paper Text:
{qp_text}

JSON Output Format:
{{
  "questions": [
    {{ "id": "1.(a)", "text": "Discuss the central theme of realism. (10 marks)" }},
    {{ "id": "1.(b)", "text": "Explain elements of modernism. (10 marks)" }}
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
        logs_callback(f"Extracted {len(questions)} distinct question/sub-question entries.", "success")
        return questions
    except Exception as e:
        logs_callback(f"Question extraction failed: {str(e)}", "error")
        return []


def verify_integrity(sliced_lines, start_idx, end_idx, all_lines):
    """Core Integrity Check: Validates that Python sliced array matches indices strictly."""
    expected_lines = all_lines[start_idx : end_idx + 1]
    if len(sliced_lines) != len(expected_lines):
        return False
    for sl, el in zip(sliced_lines, expected_lines):
        if sl['idx'] != el['idx'] or sl['text'] != el['text'] or sl['page_num'] != el['page_num']:
            return False
    return True


def map_answers(classified_pages, questions, client, model, logs_callback):
    """Stage 3: Sequential Answer Mapping using LLM Line Coordinates & Python Slicing."""
    logs_callback("Starting Stage 3: Sequential Answer Mapping...", "info")
    
    answer_pages = [p for p in classified_pages if p['classification'] == "Answer Page"]
    if not answer_pages:
        logs_callback("No Student Answer Pages were found to map.", "error")
        return []
        
    # Assemble continuous line dictionary structure across the answer pages
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
            
    # Optimization: Filter empty lines from the prompt payload to compress tokens
    llm_lines_repr = []
    for line_obj in all_lines:
        cleaned_text = line_obj['text'].strip()
        if cleaned_text:
            llm_lines_repr.append(f"L{line_obj['idx']}: {cleaned_text}")
            
    llm_document_context = "\n".join(llm_lines_repr)
    
    logs_callback(f"Created a search canvas of {len(all_lines)} total lines (Compressed to {len(llm_lines_repr)} non-empty lines for LLM context).", "info")
    
    qa_results = []
    last_end_index = 0
    
    for i, q in enumerate(questions):
        q_id = q.get("id", f"Q{i+1}")
        q_text = q.get("text", "")
        
        logs_callback(f"Matching Answer boundaries for: **{q_id}**...", "info")
        
        prompt = f"""
You are an academic processing system. Your task is to identify where the student's answer for the following question starts and ends.

Instructions:
1. Identify the starting line index and ending line index for the target question.
2. Return only the raw integers corresponding to the line IDs "L<number>".
3. Because student booklets are completed sequentially, the answer to this question most likely starts at or after index {last_end_index}. Use this as context, but search earlier lines if the student wrote answers out of order.
4. Do not summarize or synthesize the text. Only provide indices.

Target Question:
"{q_text}"

OCR Transcript lines:
{llm_document_context}

Respond ONLY with a JSON object in this format:
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
            explanation = res_json.get("explanation", "")
            
            # Validation checks on LLM index output
            if (start_idx == -1 or end_idx == -1 or start_idx > end_idx or 
                start_idx >= len(all_lines) or end_idx >= len(all_lines)):
                logs_callback(f"Indices received [{start_idx}:{end_idx}] are invalid for {q_id}. Leaving blank.", "warning")
                qa_results.append({
                    "question_id": q_id,
                    "question": q_text,
                    "answer": "[No matching boundaries found]",
                    "pages": [],
                    "confidence": "out_of_bounds"
                })
                continue
                
            # Perform Python Slicing on the raw dataset
            sliced_lines = all_lines[start_idx : end_idx + 1]
            
            # Run Non-negotiable Slicing Integrity Check
            if not verify_integrity(sliced_lines, start_idx, end_idx, all_lines):
                logs_callback(f"Integrity alignment mismatch for {q_id}. Slice aborted.", "error")
                qa_results.append({
                    "question_id": q_id,
                    "question": q_text,
                    "answer": "[Error: Integrity Slice Check Failed]",
                    "pages": [],
                    "confidence": "integrity_failed"
                })
                continue
                
            # Extract raw sliced text strictly from local slicing list
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
                "explanation": explanation
            })
            
            # Sequential search step adjustment
            last_end_index = end_idx
            logs_callback(f"Mapped {q_id} to page(s) {spanned_pages} (Lines {start_idx} to {end_idx})", "success")
            
        except Exception as e:
            logs_callback(f"Error mapping {q_id}: {str(e)}", "error")
            qa_results.append({
                "question_id": q_id,
                "question": q_text,
                "answer": f"[Mapping generation failed: {str(e)}]",
                "pages": [],
                "confidence": "failed"
            })
            
    return qa_results

# ----------------------------------------------------------------------
# Application Flow UI Layout
# ----------------------------------------------------------------------

uploaded_file = st.file_uploader("Upload Scanned PDF Student Answer Booklet", type=["pdf"])

if uploaded_file:
    if not groq_api_key:
        st.warning("Please provide your Groq API Key in the sidebar settings to process.")
    else:
        # Initializing groq client
        client = Groq(api_key=groq_api_key)
        
        st.subheader("Processing Console")
        logs_box = st.container()
        
        with logs_box:
            st.write("---")
            log_st = st.empty()
            
            # Step 1: OCR Stage
            raw_pages = run_chandra_ocr(
                uploaded_file.getvalue(), 
                uploaded_file.name, 
                lambda msg, lvl="info": update_status(log_st, msg, lvl)
            )
            
            # Step 2: Classification Stage
            classified = classify_pages(
                raw_pages, 
                client, 
                groq_model, 
                lambda msg, lvl="info": update_status(log_st, msg, lvl)
            )
            
            # Step 2b: Question extraction
            questions = extract_questions(
                classified, 
                client, 
                groq_model, 
                lambda msg, lvl="info": update_status(log_st, msg, lvl)
            )
            
            # Step 3: Sequential Mapping
            mapping_results = []
            if questions:
                mapping_results = map_answers(
                    classified, 
                    questions, 
                    client, 
                    groq_model, 
                    lambda msg, lvl="info": update_status(log_st, msg, lvl)
                )
            else:
                update_status(log_st, "No questions found to start mapping.", "warning")
                
            st.write("---")
            
        # Display Outputs
        if mapping_results:
            st.success("Pipeline Execution Complete!")
            
            tab1, tab2, tab3 = st.tabs(["Mapped Q&A Results", "Classification Logs", "Raw OCR Metadata"])
            
            with tab1:
                st.subheader("Structured Q&A Output Blocks")
                for item in mapping_results:
                    with st.expander(f"Question {item['question_id']} (Pages: {item['pages']})"):
                        st.markdown(f"**Question Text:** \n`{item['question']}`")
                        st.markdown("**Mapped Raw Answer Text:**")
                        st.code(item['answer'], language="text")
                        
            with tab2:
                st.subheader("Page Classification Records")
                st.json(classified)
                
            with tab3:
                st.subheader("Complete Page Content JSON")
                st.json(raw_pages)
                
            # Download Buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download OCR Text JSON",
                    data=json.dumps(raw_pages, indent=2),
                    file_name="ocr_metadata.json",
                    mime="application/json"
                )
            with col2:
                st.download_button(
                    label="Download Q&A Pairs JSON",
                    data=json.dumps(mapping_results, indent=2),
                    file_name="qa_pairs_extracted.json",
                    mime="application/json"
                )
