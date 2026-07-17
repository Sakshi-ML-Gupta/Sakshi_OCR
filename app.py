"""
Streamlit front-end for pipeline.py

KEY DESIGN CHOICES (these directly fix the bugs from the previous crash):

1. NO background threads for process_pdf(). It's called synchronously,
   inside the same script-run that the button click triggers. Streamlit's
   ScriptRunner already gives each run its own thread; spawning ANOTHER
   thread for a long network job is what caused the previous
   "RuntimeError: Event loop is closed" crash -- when the user re-ran the
   the script (e.g. by touching a widget) while the OCR polling was still
   running in a leftover thread, that old thread eventually tried to talk
   to a Streamlit event loop that had already been torn down.

2. A file-hash + session_state guard prevents the SAME uploaded file from
   being submitted to Datalab more than once across reruns. This is what
   caused "Submitting document... (9.4MB)" to print three times for one
   click -- Streamlit re-executes the whole script on every rerun, and
   without a guard, process_pdf() was being called again each time.

3. The "Process" button is disabled (via session_state) while a job is
   already in flight for the current file, so a double-click / accidental
   re-click cannot fire a second, overlapping run.
"""

import hashlib
import json
import traceback

import streamlit as st

import pipeline


st.set_page_config(page_title="Answer Booklet OCR + Mapper", layout="wide")
st.title("Answer Booklet OCR + Question Mapper")


# ---------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None          # hash of the last-processed file
if "result" not in st.session_state:
    st.session_state.result = None             # (ocr_json, qa_pairs) once done
if "processing" not in st.session_state:
    st.session_state.processing = False        # true while a job is in flight
if "error" not in st.session_state:
    st.session_state.error = None
if "logs" not in st.session_state:
    st.session_state.logs = []


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reset_for_new_file():
    st.session_state.result = None
    st.session_state.error = None
    st.session_state.logs = []
    st.session_state.processing = False


# ---------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload the scanned answer booklet (PDF)",
    type=["pdf"],
    key="uploader",
)

reference_file = st.file_uploader(
    "Optional: reference book / marking scheme (PDF)",
    type=["pdf"],
    key="reference_uploader",
)

if uploaded_file is not None:
    current_hash = _hash_bytes(uploaded_file.getvalue())

    # A genuinely NEW file was uploaded -- clear any old result/job state
    # for the previous file so we don't accidentally show stale results
    # or think a job is still running for a file that's no longer selected.
    if st.session_state.file_hash != current_hash:
        st.session_state.file_hash = current_hash
        _reset_for_new_file()

    already_done = st.session_state.result is not None
    col1, col2 = st.columns([1, 3])
    with col1:
        process_clicked = st.button(
            "Process document",
            disabled=st.session_state.processing or already_done,
        )
    with col2:
        if st.session_state.processing:
            st.info("A job is already running for this file -- please wait for it to finish.")
        elif already_done:
            st.success("This file has already been processed below. Upload a different file to run again.")

    # ---------------------------------------------------------------
    # Run the pipeline SYNCHRONOUSLY (no thread) exactly once per file.
    # ---------------------------------------------------------------
    if process_clicked and not st.session_state.processing and not already_done:
        st.session_state.processing = True
        st.session_state.error = None
        st.session_state.logs = []

        log_box = st.status("Processing document...", expanded=True)

        def _status_callback(msg: str):
            st.session_state.logs.append(msg)
            log_box.write(msg)

        try:
            file_bytes = uploaded_file.getvalue()
            file_name = uploaded_file.name

            ocr_json, qa_pairs = pipeline.process_pdf(
                (file_name, file_bytes),
                status_callback=_status_callback,
            )
            st.session_state.result = (ocr_json, qa_pairs)
            log_box.update(label="Processing complete", state="complete", expanded=False)

        except Exception as e:
            st.session_state.error = f"{e}\n\n{traceback.format_exc()}"
            log_box.update(label="Processing failed", state="error", expanded=True)

        finally:
            st.session_state.processing = False
            # Trigger a clean rerun now that state is settled, so the
            # button/result UI above reflects the final state immediately.
            st.rerun()

    # ---------------------------------------------------------------
    # Reference book (independent of the main pipeline; also synchronous)
    # ---------------------------------------------------------------
    if reference_file is not None:
        ref_hash_key = f"ref_hash_{_hash_bytes(reference_file.getvalue())}"
        if ref_hash_key not in st.session_state:
            if st.button("Process reference book"):
                try:
                    with st.spinner("Running OCR on reference book..."):
                        ref_json = pipeline.process_reference(
                            (reference_file.name, reference_file.getvalue()),
                            status_callback=st.write,
                        )
                    st.session_state[ref_hash_key] = ref_json
                    st.success(f"Reference OCR complete -- {ref_json['total_pages']} page(s)")
                except Exception as e:
                    st.error(f"Reference OCR failed: {e}")
        else:
            st.success("Reference book already processed.")


# ---------------------------------------------------------------------
# Error display
# ---------------------------------------------------------------------
if st.session_state.error:
    st.error("Processing failed. See details below.")
    with st.expander("Error details"):
        st.code(st.session_state.error)


# ---------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------
if st.session_state.result:
    ocr_json, qa_pairs = st.session_state.result

    matched = sum(1 for p in qa_pairs if p["matched"])
    st.subheader(f"Results -- {matched} / {len(qa_pairs)} questions matched")

    for p in qa_pairs:
        with st.expander(f"{p['ref']}: {p['question'][:90]}", expanded=not p["matched"]):
            if p["matched"]:
                st.markdown(f"**Lines:** {p['start_line']}–{p['end_line']}  "
                            f"**Pages:** {p.get('start_page')}–{p.get('end_page')}")
                st.write(p["answer"])
            else:
                st.warning("No answer text was matched for this question.")

    st.divider()
    st.subheader("Downloads")

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Download OCR JSON",
            data=json.dumps(ocr_json, ensure_ascii=False, indent=2),
            file_name="document_ocr.json",
            mime="application/json",
        )
    with col_b:
        st.download_button(
            "Download Q&A JSON",
            data=json.dumps(qa_pairs, ensure_ascii=False, indent=2),
            file_name="document_qa_pairs.json",
            mime="application/json",
        )

    with st.expander("Full processing log"):
        st.code("\n".join(st.session_state.logs))
