import os
import json
import streamlit as st

from pipeline import process_pdf

st.set_page_config(
    page_title="OCR QA Extractor",
    layout="wide"
)

st.title("📘 OCR QA Extractor")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    os.makedirs("uploads", exist_ok=True)

    pdf_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF Uploaded")

    if st.button("Run OCR"):

        with st.spinner("Processing..."):

            try:

                output_path = process_pdf(pdf_path)

                with open(output_path, "r", encoding="utf-8") as f:

                    data = json.load(f)

                st.success("Done")

                st.json(data)

                st.download_button(
                    label="Download JSON",
                    data=json.dumps(data, indent=4),
                    file_name="output.json",
                    mime="application/json"
                )

            except Exception as e:

                st.error(str(e))