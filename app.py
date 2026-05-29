import json
import traceback
import streamlit as st

from pipeline import process_pdf

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="OCR QA Extractor",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("📘 OCR Question Answer Extractor")

st.markdown(
    """
Upload a PDF file and extract:

1. Full OCR JSON
2. Question-Answer JSON
"""
)

# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# =========================================================
# PROCESS BUTTON
# =========================================================

if uploaded_file is not None:

    st.success("✅ PDF Uploaded Successfully")

    st.write("Filename:", uploaded_file.name)

    if st.button("🚀 Run OCR Pipeline"):

        try:

            # =====================================================
            # RESET FILE POINTER
            # VERY IMPORTANT
            # =====================================================

            uploaded_file.seek(0)

            # =====================================================
            # LOADING
            # =====================================================

            progress = st.progress(0)

            status = st.empty()

            status.info("Starting OCR Pipeline...")

            progress.progress(10)

            # =====================================================
            # PROCESS PDF
            # =====================================================

            status.info("Running OCR and extracting text...")

            ocr_json, final_json = process_pdf(uploaded_file)

            progress.progress(80)

            status.info("Generating outputs...")

            progress.progress(100)

            status.success("Pipeline Completed Successfully")

            st.success("✅ OCR + QA Extraction Finished")

            # =====================================================
            # OCR JSON OUTPUT
            # =====================================================

            st.divider()

            st.subheader("📄 OCR JSON Output")

            st.json(ocr_json)

            st.download_button(
                label="⬇ Download OCR JSON",
                data=json.dumps(
                    ocr_json,
                    ensure_ascii=False,
                    indent=4
                ),
                file_name="ocr_output.json",
                mime="application/json"
            )

            # =====================================================
            # FINAL QA OUTPUT
            # =====================================================

            st.divider()

            st.subheader("🧠 Final Question Answer JSON")

            st.json(final_json)

            st.download_button(
                label="⬇ Download QA JSON",
                data=json.dumps(
                    final_json,
                    ensure_ascii=False,
                    indent=4
                ),
                file_name="qa_output.json",
                mime="application/json"
            )

        except Exception as e:

            st.error("❌ Pipeline Failed")

            st.code(str(e))

            st.subheader("Full Error Trace")

            st.code(traceback.format_exc())