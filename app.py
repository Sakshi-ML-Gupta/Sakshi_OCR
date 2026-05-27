import os
import streamlit as st
from pipeline import process_pdf

st.set_page_config(
    page_title="OCR QA Extractor",
    layout="wide"
)

st.title("📘 OCR QA Extraction Pipeline")

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

    if st.button("Run OCR Pipeline"):

        with st.spinner("Processing PDF..."):

            try:

                final_json_path = process_pdf(pdf_path)

                st.success("Pipeline Complete")

                with open(final_json_path, "r", encoding="utf-8") as f:
                    data = f.read()

                st.json(data)

                st.download_button(
                    label="Download JSON",
                    data=data,
                    file_name=os.path.basename(final_json_path),
                    mime="application/json"
                )

            except Exception as e:

                st.error(str(e))