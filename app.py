import json
import threading
import traceback
from datetime import datetime

import streamlit as st

from pipeline import (
    process_pdf,
    process_reference,
    save_outputs,
)

st.set_page_config(
    page_title="Assignment OCR & Question-Answer Mapper",
    page_icon="📝",
    layout="wide",
)

# =========================================================
# SESSION STATE
# =========================================================

if "logs" not in st.session_state:
    st.session_state.logs = []
if "result" not in st.session_state:
    st.session_state.result = None  # (ocr_json, qa_pairs)
if "reference_result" not in st.session_state:
    st.session_state.reference_result = None
if "error" not in st.session_state:
    st.session_state.error = None


def reset_run_state():
    st.session_state.logs = []
    st.session_state.result = None
    st.session_state.error = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("⚙️ Settings")
    st.caption(
        "This app OCRs a scanned exam assignment booklet, detects the "
        "official question paper pages, and maps each question to the "
        "student's answer."
    )

    mode = st.radio(
        "What do you want to process?",
        options=["Assignment booklet (Q&A mapping)", "Reference book (OCR only)"],
        index=0,
    )

    st.divider()
    st.caption(
        "Required secrets/env vars:\n\n"
        "- `MISTRAL_API_KEY`\n"
        "- `GROQ_API_KEY`"
    )

    missing_keys = []
    try:
        from pdf_processor import get_api_key
        if not get_api_key("MISTRAL_API_KEY"):
            missing_keys.append("MISTRAL_API_KEY")
        if not get_api_key("GROQ_API_KEY"):
            missing_keys.append("GROQ_API_KEY")
    except Exception:
        pass

    if missing_keys:
        st.warning(f"Missing: {', '.join(missing_keys)}. Add them to `.streamlit/secrets.toml` or your `.env` file.")
    else:
        st.success("API keys detected ✅")


# =========================================================
# MAIN AREA
# =========================================================

st.title("📝 Assignment OCR & Question-Answer Mapper")

if mode == "Assignment booklet (Q&A mapping)":
    st.write(
        "Upload a scanned exam assignment PDF (question paper + handwritten "
        "answers mixed together). The app will OCR it, find the question "
        "paper pages, extract the canonical question list, and map each "
        "question to the student's verbatim answer."
    )

    uploaded_file = st.file_uploader(
        "Upload assignment PDF", type=["pdf"], key="assignment_uploader"
    )

    col_run, col_clear = st.columns([1, 1])
    run_clicked = col_run.button("🚀 Process document", type="primary", disabled=uploaded_file is None)
    clear_clicked = col_clear.button("🗑️ Clear results")

    if clear_clicked:
        reset_run_state()
        st.rerun()

    if run_clicked and uploaded_file is not None:
        reset_run_state()

        log_placeholder = st.empty()
        _log_lock = threading.Lock()

        def status_callback(msg: str):
            # NOTE: with the concurrent chunk processing in
            # pdf_processor.py, this callback can now be invoked from
            # multiple worker threads at roughly the same time. A lock
            # keeps the session-state append and the placeholder redraw
            # atomic per call, avoiding interleaved/garbled log output.
            with _log_lock:
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.logs.append(f"[{timestamp}] {msg}")
                log_placeholder.code("\n".join(st.session_state.logs[-200:]), language=None)

        with st.spinner("Processing... this can take a few minutes for large documents"):
            try:
                file_bytes = uploaded_file.getvalue()
                ocr_json, qa_pairs = process_pdf(
                    (uploaded_file.name, file_bytes), status_callback=status_callback
                )
                st.session_state.result = (ocr_json, qa_pairs)
            except Exception as e:
                st.session_state.error = f"{e}\n\n{traceback.format_exc()}"

    # ---- Show logs even after the run (from session state) ----
    if st.session_state.logs and st.session_state.result is None and st.session_state.error is None:
        st.code("\n".join(st.session_state.logs[-200:]), language=None)

    # ---- Error ----
    if st.session_state.error:
        st.error("Processing failed. See details below.")
        with st.expander("Show full error / logs", expanded=True):
            st.code(st.session_state.error)
            if st.session_state.logs:
                st.text("--- Log trail ---")
                st.code("\n".join(st.session_state.logs[-200:]), language=None)

    # ---- Result ----
    if st.session_state.result:
        ocr_json, qa_pairs = st.session_state.result
        matched = sum(1 for qa in qa_pairs if qa.get("matched"))
        total = len(qa_pairs)

        st.success(f"Done! Matched {matched} of {total} questions.")

        tab_qa, tab_ocr, tab_logs = st.tabs(["📋 Q&A Pairs", "📄 Raw OCR", "🪵 Logs"])

        with tab_qa:
            for i, qa in enumerate(qa_pairs, start=1):
                status_icon = "✅" if qa.get("matched") else "⚠️ no match found"
                with st.expander(f"Q{i}. {qa['question'][:100]}{'...' if len(qa['question']) > 100 else ''}  —  {status_icon}"):
                    st.markdown("**Question:**")
                    st.write(qa["question"])
                    st.markdown("**Answer:**")
                    if qa.get("answer"):
                        st.write(qa["answer"])
                    else:
                        st.info("No answer text was matched for this question.")

        with tab_ocr:
            st.caption(f"Total pages OCR'd: {ocr_json.get('total_pages', 0)}")
            for page in ocr_json.get("pages", []):
                with st.expander(f"Page {page['page_number']}"):
                    st.text(page["text"])

        with tab_logs:
            st.code("\n".join(st.session_state.logs), language=None)

        st.divider()
        st.subheader("⬇️ Downloads")

        base_name = (uploaded_file.name.rsplit(".", 1)[0] if uploaded_file else "document")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download OCR JSON",
                data=json.dumps(ocr_json, ensure_ascii=False, indent=2),
                file_name=f"{base_name}_ocr.json",
                mime="application/json",
            )
        with col2:
            st.download_button(
                "Download Q&A Pairs JSON",
                data=json.dumps(qa_pairs, ensure_ascii=False, indent=2),
                file_name=f"{base_name}_qa_pairs.json",
                mime="application/json",
            )

else:  # Reference book (OCR only)
    st.write(
        "Upload a reference book PDF. The app will OCR every page and "
        "return the full text as JSON (no question/answer mapping)."
    )

    ref_file = st.file_uploader("Upload reference PDF", type=["pdf"], key="reference_uploader")

    col_run, col_clear = st.columns([1, 1])
    ref_run_clicked = col_run.button("🚀 OCR reference book", type="primary", disabled=ref_file is None)
    ref_clear_clicked = col_clear.button("🗑️ Clear results", key="ref_clear")

    if ref_clear_clicked:
        st.session_state.reference_result = None
        st.rerun()

    if ref_run_clicked and ref_file is not None:
        log_placeholder = st.empty()
        logs = []

        def ref_status_callback(msg: str):
            timestamp = datetime.now().strftime("%H:%M:%S")
            logs.append(f"[{timestamp}] {msg}")
            log_placeholder.code("\n".join(logs[-200:]), language=None)

        with st.spinner("Running OCR..."):
            try:
                file_bytes = ref_file.getvalue()
                ocr_json = process_reference(
                    (ref_file.name, file_bytes), status_callback=ref_status_callback
                )
                st.session_state.reference_result = ocr_json
            except Exception as e:
                st.error(f"OCR failed: {e}")
                st.code(traceback.format_exc())

    if st.session_state.reference_result:
        ocr_json = st.session_state.reference_result
        st.success(f"OCR complete — {ocr_json.get('total_pages', 0)} page(s)")

        for page in ocr_json.get("pages", []):
            with st.expander(f"Page {page['page_number']}"):
                st.text(page["text"])

        base_name = (ref_file.name.rsplit(".", 1)[0] if ref_file else "reference")
        st.download_button(
            "Download OCR JSON",
            data=json.dumps(ocr_json, ensure_ascii=False, indent=2),
            file_name=f"{base_name}_ocr.json",
            mime="application/json",
        )
