# =========================================================
# app.py
# =========================================================

import json
import streamlit as st

from pipeline import process_pdf

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Universal OCR QA Extractor",
    layout="wide"
)

# =========================================================
# UI
# =========================================================

st.title("📘 Universal OCR QA Extractor")

st.write(
    "Upload any assignment PDF and extract "
    "questions + answers automatically."
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# =========================================================
# PROCESS
# =========================================================

if uploaded_file is not None:

    st.success("PDF Uploaded Successfully")

    if st.button("Run OCR Pipeline"):

        with st.spinner("Processing PDF..."):

            try:

                output_path = process_pdf(
                    uploaded_file
                )

                with open(
                    output_path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    data = json.load(f)

                st.success("OCR Completed")

                st.json(data)

                st.download_button(
                    label="Download JSON",
                    data=json.dumps(
                        data,
                        indent=4,
                        ensure_ascii=False
                    ),
                    file_name="output.json",
                    mime="application/json"
                )

            except Exception as e:

                st.error(str(e))