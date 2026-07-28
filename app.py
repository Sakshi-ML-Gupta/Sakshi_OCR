"""
Streamlit app for OCR'ing mixed printed/handwritten assignment PDFs
and extracting clean Question -> Answer pairs.

Run with:
    streamlit run app.py
"""

import json
import traceback
# THIS MUST BE THE VERY FIRST CUSTOM IMPORT
import tuple_error_diagnostic  # noqa: F401  (imported for its side effect)

import streamlit as st
from pipeline import process_pdf

st.set_page_config(page_title="Assignment OCR + Q&A Extractor", layout="wide")
st.title("Assignment OCR + Question/Answer Extractor")
st.caption(
    "Upload a scanned assignment PDF (mixed printed questions + "
    "handwritten answers). The pipeline OCRs the document, identifies "
    "the real exam questions, and matches them to the student's answers."
)

# =========================================================
# SESSION STATE -- THE RERUN GUARD
# =========================================================

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None
if "pending_file_bytes" not in st.session_state:
    st.session_state.pending_file_bytes = None
if "pending_file_name" not in st.session_state:
    st.session_state.pending_file_name = None


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload assignment PDF",
    type=["pdf"],
    disabled=st.session_state.is_processing,
)

process_clicked = st.button(
    "Process document",
    disabled=st.session_state.is_processing or uploaded_file is None,
)

# Step 1: Capture file bytes and set state
if process_clicked and not st.session_state.is_processing:
    st.session_state.pending_file_bytes = uploaded_file.getvalue()
    st.session_state.pending_file_name = uploaded_file.name
    st.session_state.is_processing = True
    st.session_state.result = None
    st.session_state.error = None
    st.rerun()

# Step 2: Long-running processing call
if (
    st.session_state.is_processing
    and st.session_state.result is None
    and st.session_state.error is None
):
    status_box = st.empty()
    log_lines = []

    def status_callback(msg):
        log_lines.append(msg)
        status_box.text("\n".join(log_lines[-12:]))

    try:
        file_bytes = st.session_state.pending_file_bytes
        file_name = st.session_state.pending_file_name

        ocr_json, qa_pairs = process_pdf(
            (file_name, file_bytes),
            status_callback=status_callback,
        )
        st.session_state.result = (ocr_json, qa_pairs)

    except Exception as e:
        st.session_state.error = f"{e}\n\n```\n{traceback.format_exc()}\n```"

    finally:
        st.session_state.is_processing = False
        st.session_state.pending_file_bytes = None
        st.session_state.pending_file_name = None
        st.rerun()


# =========================================================
# RESULTS / ERROR DISPLAY
# =========================================================

if st.session_state.error:
    st.error(st.session_state.error)
    if st.button("Dismiss"):
        st.session_state.error = None
        st.rerun()

if st.session_state.result:
    ocr_json, qa_pairs = st.session_state.result

    st.success(
        f"Done — {len(qa_pairs)} Q&A pair(s) extracted "
        f"from {ocr_json.get('total_pages', 0)} page(s)."
    )

    # Safe display loop handling both String & Dictionary formats
    for i, qa in enumerate(qa_pairs, start=1):
        if isinstance(qa, str):
            q_title = qa[:90]
            q_text = qa
            a_text = ""
        elif isinstance(qa, dict):
            q_text = qa.get("question", "")
            q_title = q_text[:90] if q_text else f"Question {i}"
            a_text = qa.get("answer", "")
        else:
            q_text = str(qa)
            q_title = q_text[:90]
            a_text = ""

        with st.expander(f"Q{i}: {q_title}"):
            st.markdown("**Question:**")
            st.write(q_text)
            if a_text:
                st.markdown("**Answer:**")
                st.write(a_text)

    st.divider()

    # JSON Download & Reset Actions (Out of the loop)
    result_json = json.dumps(qa_pairs, ensure_ascii=False, indent=2)
    st.download_button(
        "Download Q&A pairs as JSON",
        data=result_json,
        file_name="qa_pairs.json",
        mime="application/json",
    )

    if st.button("Process another file"):
        st.session_state.result = None
        st.rerun()

# =========================================================
# FULL PIPELINE WRAPPER FOR APP.PY
# =========================================================
@_diagnose_tuple_errors
def process_pdf(file_input, status_callback=None):
    """
    Complete end-to-end processing wrapper for app.py:
    1. Runs OCR on PDF
    2. Identifies question paper & canonical questions
    3. Sequentially maps answers
    4. Returns tuple: (ocr_json, qa_pairs)
    """
    file_bytes, file_name = _normalize_file_input(file_input, default_name="assignment.pdf")

    # 1. Run OCR
    pages_raw = run_ocr_cached(file_bytes, file_name, status_callback)
    ocr_json = build_ocr_json(pages_raw)

    pages = [{"page_number": p["page_number"], "raw_text": p["raw_text"]} for p in pages_raw]

    # 2. Extract Questions & Classify Pages
    qp_indices, questions, admin_indices = identify_questions_with_llm(pages, status_callback)

    # 3. Extract Answer Lines
    answer_pages = [p for i, p in enumerate(pages) if i not in qp_indices and i not in admin_indices]
    answer_lines = []
    for p in answer_pages:
        answer_lines.extend(p["raw_text"].splitlines())

    # 4. Map Answers Sequentially
    ranges = map_answers_sequential(answer_lines, questions, status_callback)

    # 5. Format into Q&A Pairs Dictionary
    qa_pairs = []
    for r in ranges:
        ref = r["ref"]
        # Convert REF-A -> 0, REF-B -> 1 ...
        q_idx = ord(ref.split("-")[-1]) - ord("A")
        
        q_text = questions[q_idx] if q_idx < len(questions) else f"Question {q_idx + 1}"
        ans_lines = answer_lines[r["start_line"] : r["end_line"] + 1]
        ans_text = "\n".join(ans_lines).strip()

        qa_pairs.append({
            "question": q_text,
            "answer": ans_text if ans_text else "Answer text not found."
        })

    return ocr_json, qa_pairs
