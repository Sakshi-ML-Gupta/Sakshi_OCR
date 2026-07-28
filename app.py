"""
Streamlit app for OCR'ing mixed printed/handwritten assignment PDFs
and extracting clean Question -> Answer pairs.

Run with:
    streamlit run app.py

Requires these files in the SAME folder:
    - pipeline_llm_v4.py          (the OCR + LLM pipeline module)
    - tuple_error_diagnostic.py   (diagnostic instrumentation)

Required environment variables / st.secrets entries:
    - DATALAB_API_KEY
    - GROQ_API_KEY
"""

# THIS MUST BE THE VERY FIRST IMPORT -- before streamlit, before
# anything else. It installs diagnostic instrumentation that will
# print the exact file/line responsible if the
# "expected str, bytes or os.PathLike object, not tuple" error ever
# occurs anywhere in this process, instead of it surfacing as an
# unexplained crash deep inside Streamlit's own error handling.
import tuple_error_diagnostic  # noqa: F401  (imported for its side effect)

import streamlit as st
#from pipeline import process_pdf
from pipeline import process_reference as process_pdf


st.set_page_config(page_title="Assignment OCR + Q&A Extractor", layout="wide")
st.title("Assignment OCR + Question/Answer Extractor")
st.caption(
    "Upload a scanned assignment PDF (mixed printed questions + "
    "handwritten answers). The pipeline OCRs the document, identifies "
    "the real exam questions, and matches them to the student's answers."
)

# =========================================================
# SESSION STATE -- THE RERUN GUARD
#
# Streamlit reruns the ENTIRE script top-to-bottom on almost any
# interaction (button click, widget change, even a websocket
# reconnect in some environments). Without this guard, a rerun that
# happens WHILE process_pdf() is still running (it can take minutes:
# OCR + several paced LLM calls) would start a SECOND, fully
# independent run -- both competing for the same shared Groq token
# budget at once. This is the exact cause of the doubled log lines
# seen in earlier debugging sessions ("Asking LLM to analyze chunk
# 2/8" printed twice back-to-back) and the reason the daily token
# quota got burned twice as fast as it should have.
#
# st.session_state persists across reruns WITHIN one user's session,
# so this flag correctly blocks a second run from starting no matter
# what triggers the rerun -- not just a second click of the button.
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

# Step 1: button click -- capture the file's bytes NOW (while we still
# have direct access to the upload widget's value) and set the
# processing flag, then rerun immediately so the disabled button state
# renders before the long-running call starts.
if process_clicked and not st.session_state.is_processing:
    st.session_state.pending_file_bytes = uploaded_file.getvalue()
    st.session_state.pending_file_name = uploaded_file.name
    st.session_state.is_processing = True
    st.session_state.result = None
    st.session_state.error = None
    st.rerun()

# Step 2: the ACTUAL long-running call, gated purely on the
# is_processing flag (not on the button's one-shot click state) --
# this is what makes the guard effective against ANY rerun trigger,
# not just the button itself.
if (
    st.session_state.is_processing
    and st.session_state.result is None
    and st.session_state.error is None
):
    status_box = st.empty()
    log_lines = []

    def status_callback(msg):
        log_lines.append(msg)
        # Show the last several lines so the box doesn't grow unbounded
        status_box.text("\n".join(log_lines[-12:]))

    try:
        file_bytes = st.session_state.pending_file_bytes
        file_name = st.session_state.pending_file_name

        # process_pdf accepts (filename, bytes) tuples directly --
        # this matches pipeline_llm_v4.py's documented input shapes
        # exactly, so no manual file-handle juggling is needed here.
        ocr_json, qa_pairs = process_pdf(
            (file_name, file_bytes),
            status_callback=status_callback,
        )
        st.session_state.result = (ocr_json, qa_pairs)

    except Exception as e:
        import traceback
        st.session_state.error = f"{e}\n\n```\n{traceback.format_exc()}\n```"

    finally:
        st.session_state.is_processing = False
        st.session_state.pending_file_bytes = None
        st.session_state.pending_file_name = None
        st.rerun()  # final rerun: show results/error, re-enable controls


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

    st.success(f"Done — {len(qa_pairs)} Q&A pair(s) extracted "
               f"from {ocr_json['total_pages']} page(s).")

    for i, qa in enumerate(qa_pairs, start=1):
        with st.expander(f"Q{i}: {qa['question'][:90]}"):
            st.markdown("**Question:**")
            st.write(qa["question"])
            st.markdown("**Answer:**")
            st.write(qa["answer"] if qa["answer"] else "_(no answer text matched)_")

    import json
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
