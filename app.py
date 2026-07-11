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
st.set_page_config(page_title="Resilient Assignment Parser", page_icon="📝", layout="wide")

st.title("Resilient University Assignment Q&A Pipeline")
st.write(
    "Process scanned booklets with Datalab Chandra OCR & Groq. "
    "Includes **Manual Overrides** to correct LLM mistakes before mapping."
)

# Initialize Session State to track pipeline steps
if "raw_pages" not in st.session_state:
    st.session_state.raw_pages = None
if "classified_pages" not in st.session_state:
    st.session_state.classified_pages = None
if "extracted_questions" not in st.session_state:
    st.session_state.extracted_questions = None
if "mapped_results" not in st.session_state:
    st.session_state.mapped_results = None

# ----------------------------------------------------------------------
# Sidebar Settings
# ----------------------------------------------------------------------
st.sidebar.header("Configuration Settings")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
groq_model = st.sidebar.selectbox(
    "Groq Model Selection",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    index=0
)

ocr_mode = st.sidebar.radio(
    "OCR Model Mode",
    ["Datalab Chandra (Local CLI)", "PyPDF Fallback (For CPU/Local Testing)"],
    index=1
)

# Reset pipeline if a new file is uploaded
def reset_pipeline():
    st.session_state.raw_pages = None
    st.session_state.classified_pages = None
    st.session_state.extracted_questions = None
    st.session_state.mapped_results = None

uploaded_file = st.file_uploader(
    "Upload Scanned PDF Student Answer Booklet", 
    type=["pdf"], 
    on_change=reset_pipeline
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


def run_chandra_ocr(pdf_bytes, filename):
    temp_pdf_path = f"temp_{filename}"
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_bytes)
        
    if "Fallback" in ocr_mode:
        return run_fallback_ocr(temp_pdf_path)
        
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
            raise FileNotFoundError("Chandra completed successfully but no output .md was found.")
    except Exception as e:
        st.warning(f"Chandra OCR local runtime error: {str(e)}. Defaulting to standard text reader.")
        return run_fallback_ocr(temp_pdf_path)
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


def classify_pages(pages, client, model):
    classified_pages = []
    for page in pages:
        page_num = page['page_num']
        text_content = page['text']
        truncated_text = text_content[:1200]
        
        prompt = f"""
You are an expert document classifier. Your job is to classify this page of a university student assignment/exam booklet.
Classifications:
1. "Question Paper" - Contains printed questions, marks, or instructions.
2. "Admin/Cover" - Cover sheets, grade card, registration forms.
3. "Answer Page" - Written answers.

Page Content:
---
{truncated_text}
---

Respond ONLY with a JSON object:
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
                "reason": "Fallback due to LLM error"
            })
    return classified_pages


def extract_questions_from_text(qp_text, client, model):
    prompt = f"""
Analyze the following Question Paper text and extract all questions.
Rules:
1. Sub-questions must be individual list entries (e.g. split Q1 into Q1.(a), Q1.(b)). Do not merge them.
2. Respond ONLY with a JSON object.

Question Paper Text:
{qp_text}

JSON Output Format:
{{
  "questions": [
    {{ "id": "1.(a)", "text": "Question details here..." }}
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
        return res_json.get("questions", [])
    except Exception as e:
        st.error(f"Failed to parse questions: {str(e)}")
        return []


def verify_integrity(sliced_lines, start_idx, end_idx, all_lines):
    expected_lines = all_lines[start_idx : end_idx + 1]
    if len(sliced_lines) != len(expected_lines):
        return False
    for sl, el in zip(sliced_lines, expected_lines):
        if sl['idx'] != el['idx'] or sl['text'] != el['text'] or sl['page_num'] != el['page_num']:
            return False
    return True


def map_answers(classified_pages, questions, client, model, status_placeholder):
    answer_pages = [p for p in classified_pages if p['classification'] == "Answer Page"]
    if not answer_pages:
        return []
        
    all_lines = []
    global_idx = 0
    for page in answer_pages:
        page_num = page['page_num']
        lines = page['text'].split('\n')
        for line in lines:
            all_lines.append({"idx": global_idx, "text": line, "page_num": page_num})
            global_idx += 1
            
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
        
        status_placeholder.write(f"🔄 Mapping Answer for Question **{q_id}**...")
        
        prompt = f"""
Identify the start and end line index for the target question.
Because booklets are sequential, the answer most likely starts at/after index {last_end_index}. 
Only return the raw line indices in JSON.

Target Question:
"{q_text}"

OCR Transcript lines:
{llm_document_context}

Respond ONLY with a JSON object:
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
                    "answer": "[Mapping boundaries out of range]",
                    "pages": [],
                    "confidence": "failed"
                })
                continue
                
            sliced_lines = all_lines[start_idx : end_idx + 1]
            
            if not verify_integrity(sliced_lines, start_idx, end_idx, all_lines):
                qa_results.append({
                    "question_id": q_id,
                    "question": q_text,
                    "answer": "[Error: Integrity verification failed]",
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
            last_end_index = end_idx
            
        except Exception as e:
            qa_results.append({
                "question_id": q_id,
                "question": q_text,
                "answer": f"[Error mapping answer: {str(e)}]",
                "pages": [],
                "confidence": "failed"
            })
            
    return qa_results

# ----------------------------------------------------------------------
# Interactive Execution Flow
# ----------------------------------------------------------------------
if uploaded_file:
    if not groq_api_key:
        st.warning("Please enter your Groq API Key in the sidebar to proceed.")
    else:
        client = Groq(api_key=groq_api_key)

        # STEP 1: OCR (Run automatically on upload if not done yet)
        if st.session_state.raw_pages is None:
            with st.spinner("Stage 1: Running OCR on the PDF document..."):
                st.session_state.raw_pages = run_chandra_ocr(uploaded_file.getvalue(), uploaded_file.name)
            st.success("Stage 1: OCR Processing completed!")

        # STEP 2: Classification (Run automatically after OCR)
        if st.session_state.classified_pages is None:
            with st.spinner("Stage 2: Running Page Classification..."):
                st.session_state.classified_pages = classify_pages(st.session_state.raw_pages, client, groq_model)
            st.success("Stage 2: Initial classification completed!")

        # --- MANUAL OVERRIDE INTERFACE ---
        st.header("Step-by-Step Document Adjustment Panel")
        st.write("Review and adjust the page types to ensure the pipeline proceeds without failure.")

        # Let user override classifications in real-time
        with st.expander("📝 View & Adjust Page Classifications"):
            cols = st.columns(4)
            for i, page in enumerate(st.session_state.classified_pages):
                col_idx = i % 4
                with cols[col_idx]:
                    st.write(f"**Page {page['page_num']}**")
                    new_val = st.selectbox(
                        f"Type for P.{page['page_num']}",
                        ["Question Paper", "Admin/Cover", "Answer Page"],
                        index=["Question Paper", "Admin/Cover", "Answer Page"].index(page['classification']),
                        key=f"class_{page['page_num']}"
                    )
                    # Update configuration in session state
                    st.session_state.classified_pages[i]['classification'] = new_val

        # STEP 3: Question Extraction Trigger
        if st.button("Extract Questions From Classified Pages"):
            qp_pages = [p for p in st.session_state.classified_pages if p['classification'] == "Question Paper"]
            if qp_pages:
                qp_text = "\n\n".join([f"--- QP Page {p['page_num']} ---\n{p['text']}" for p in qp_pages])
                with st.spinner("Analyzing Question Paper pages..."):
                    st.session_state.extracted_questions = extract_questions_from_text(qp_text, client, groq_model)
            else:
                st.session_state.extracted_questions = []

        # --- QUESTION OVERRIDE / MANUAL FALLBACK WRITER ---
        if st.session_state.extracted_questions is not None:
            st.subheader("Extracted Questions List")
            
            if len(st.session_state.extracted_questions) == 0:
                st.warning("⚠️ No questions found. You can manually enter or edit questions below.")
            
            # Interactive JSON editor or text-based question input
            q_list_raw = json.dumps(st.session_state.extracted_questions, indent=2)
            edited_q_json = st.text_area(
                "Manually Edit/Add Questions (JSON format):", 
                value=q_list_raw, 
                height=250,
                help="You can manually specify questions here if the LLM missed them or if the PDF did not contain the question paper."
            )
            try:
                st.session_state.extracted_questions = json.loads(edited_q_json)
            except Exception as e:
                st.error("Invalid JSON syntax in manual editor. Please fix the formatting.")

            # STEP 4: Execution of Sequential Mapping
            if st.button("Run Sequential Answer Mapping", type="primary"):
                if not st.session_state.extracted_questions:
                    st.error("Cannot run mapping without a valid list of questions.")
                else:
                    status_placeholder = st.empty()
                    with st.spinner("Mapping questions to exact raw handwritten answer text..."):
                        st.session_state.mapped_results = map_answers(
                            st.session_state.classified_pages,
                            st.session_state.extracted_questions,
                            client,
                            groq_model,
                            status_placeholder
                        )
                    status_placeholder.empty()
                    st.success("Mapping Completed successfully!")

        # --- SHOW RESULTS ---
        if st.session_state.mapped_results:
            st.header("Pipeline Output Results")
            tab1, tab2 = st.tabs(["Mapped Q&A pairs", "Metadata & Debug Export"])
            
            with tab1:
                for item in st.session_state.mapped_results:
                    with st.expander(f"Question {item['question_id']} (Pages: {item['pages']})"):
                        st.markdown(f"**Question:**\n`{item['question']}`")
                        st.markdown("**Mapped Raw Answer Text:**")
                        st.code(item['answer'], language="text")
                        
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download Final Q&A JSON",
                        data=json.dumps(st.session_state.mapped_results, indent=2),
                        file_name="final_qa_mapping.json",
                        mime="application/json"
                    )
                with col2:
                    st.download_button(
                        label="Download Full Document States",
                        data=json.dumps({
                            "pages": st.session_state.raw_pages,
                            "classification": st.session_state.classified_pages,
                            "questions": st.session_state.extracted_questions
                        }, indent=2),
                        file_name="pipeline_debug_export.json",
                        mime="application/json"
                    )
