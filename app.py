"""
app.py
======
Streamlit front-end for the Exam Evaluator OCR module.

Flow:
1. User uploads an assignment/answer-booklet PDF (handwritten/printed,
   any language mix, any page structure).
2. pipeline.run_ocr_pipeline() OCRs every page in parallel -> full OCR JSON.
3. pipeline.extract_qa_pairs() maps question paper text to matching answer
   text, verbatim, from the OCR output -> Q/A JSON.
4. Both JSON files are made available as downloads. The Q/A JSON is the
   final artifact; the full OCR JSON is generated last but is the
   authoritative record of everything read off the page.

Run with:
    export ANTHROPIC_API_KEY=sk-ant-...
    streamlit run app.py
"""

import io
import json
import time

import streamlit as st

import fitz  # PyMuPDF

from pipeline import (
    run_ocr_pipeline,
    extract_qa_pairs,
    auto_tune_batch_size,
    TARGET_MAX_SECONDS,
)

st.set_page_config(page_title="Exam Evaluator — OCR", page_icon="📝", layout="wide")

st.title("📝 Exam Evaluator — OCR Module")
st.caption(
    "Upload a scanned assignment/answer booklet (handwritten and/or printed, "
    "any language, any page layout). The pipeline transcribes every page "
    "verbatim, then maps questions to their answers."
)

with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input(
        "GEMINI_API_KEY",
        type="password",
        help="Free key from https://aistudio.google.com/apikey. Only needed here "
             "if it isn't already set as an environment variable.",
    )
    if api_key_input:
        import os
        os.environ["GEMINI_API_KEY"] = api_key_input
    st.markdown("---")
    st.markdown(
        "**Notes**\n"
        "- Runs on the free Gemini API tier — no billing required.\n"
        "- No spelling/grammar correction is applied — transcription is verbatim.\n"
        "- Grader red-ink marks are ignored automatically.\n"
        "- Pages are batched + rate-limited to stay within free-tier quotas "
        "(tune `GEMINI_RPM` / `OCR_PAGES_PER_BATCH` env vars if you hit 429s "
        "or want to go faster on a higher-tier key)."
    )

uploaded_file = st.file_uploader("Upload assignment PDF", type=["pdf"])

run = st.button("Run OCR", type="primary", disabled=uploaded_file is None)

if run and uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    filename = uploaded_file.name

    page_count = fitz.open(stream=pdf_bytes, filetype="pdf").page_count
    batch_size, est_seconds, feasible = auto_tune_batch_size(page_count)

    if feasible:
        st.caption(
            f"{page_count} pages · auto-tuned batch size {batch_size} · "
            f"estimated time ≈ {est_seconds:.0f}s (target ≤ {TARGET_MAX_SECONDS:.0f}s)"
        )
    else:
        st.warning(
            f"{page_count} pages at your current GEMINI_RPM can't fit under "
            f"{TARGET_MAX_SECONDS:.0f}s even at the largest safe batch size "
            f"(estimated ≈ {est_seconds:.0f}s / {est_seconds/60:.1f} min). "
            f"This is a free-tier rate-limit floor, not something the code "
            f"can work around without a higher-RPM key. Proceeding anyway "
            f"at the fastest configuration available."
        )

    progress_bar = st.progress(0, text="Starting OCR...")
    status = st.empty()

    def _progress(done, total):
        progress_bar.progress(done / total, text=f"OCR'd {done}/{total} pages")

    t0 = time.time()
    try:
        with status.status("Running OCR on all pages...", expanded=False):
            full_ocr_json = run_ocr_pipeline(
                pdf_bytes, filename=filename, progress_cb=_progress, batch_size=batch_size
            )
        ocr_time = time.time() - t0

        with status.status("Mapping questions to answers...", expanded=False):
            qa_pairs = extract_qa_pairs(full_ocr_json)
        total_time = time.time() - t0

    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()

    progress_bar.progress(1.0, text="Done")
    st.success(
        f"Completed {full_ocr_json['page_count']} pages in {total_time:.1f}s "
        f"(OCR: {ocr_time:.1f}s, mapping: {total_time - ocr_time:.1f}s)"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Q-A Pairs")
        st.caption(f"{len(qa_pairs)} question(s) matched")
        st.json(qa_pairs, expanded=False)
    with col2:
        st.subheader("Full OCR (per page)")
        st.json(full_ocr_json, expanded=False)

    qa_bytes = json.dumps(qa_pairs, ensure_ascii=False, indent=2).encode("utf-8")
    full_ocr_bytes = json.dumps(full_ocr_json, ensure_ascii=False, indent=2).encode("utf-8")

    st.markdown("### Downloads")
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.download_button(
            "⬇️ Download Q-A JSON",
            data=qa_bytes,
            file_name=f"{filename.rsplit('.', 1)[0]}_qa_pairs.json",
            mime="application/json",
        )
    with dcol2:
        # Full OCR JSON is generated/offered last, as required.
        st.download_button(
            "⬇️ Download Full OCR JSON",
            data=full_ocr_bytes,
            file_name=f"{filename.rsplit('.', 1)[0]}_full_ocr.json",
            mime="application/json",
        )
elif uploaded_file is None:
    st.info("Upload a PDF to begin.")
