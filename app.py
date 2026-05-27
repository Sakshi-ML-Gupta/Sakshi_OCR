import json
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

# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# =========================================================
# PROCESS
# =========================================================

if uploaded_file:

    st.success("PDF Uploaded Successfully")

    if st.button("Run OCR Pipeline"):

        with st.spinner("Processing PDF..."):

            try:

                ocr_json, final_json = process_pdf(
                    uploaded_file
                )

                st.success("Pipeline Completed")

                # =====================================================
                # OCR JSON
                # =====================================================

                st.subheader("OCR JSON")

                st.json(ocr_json)

                st.download_button(
                    label="Download OCR JSON",
                    data=json.dumps(
                        ocr_json,
                        ensure_ascii=False,
                        indent=4
                    ),
                    file_name="ocr_output.json",
                    mime="application/json"
                )

                # =====================================================
                # FINAL QA JSON
                # =====================================================

                st.subheader("Final QA JSON")

                st.json(final_json)

                st.download_button(
                    label="Download QA JSON",
                    data=json.dumps(
                        final_json,
                        ensure_ascii=False,
                        indent=4
                    ),
                    file_name="qa_output.json",
                    mime="application/json"
                )

            except Exception as e:

                st.error(str(e))