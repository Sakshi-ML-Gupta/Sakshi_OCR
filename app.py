import streamlit as st
import tempfile
import json

from pipeline import process_pdf

st.set_page_config(
    page_title="Universal OCR Pipeline",
    layout="wide"
)

st.title("📄 OCR Pipeline")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    st.success("PDF uploaded successfully")

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())

        pdf_path = tmp_file.name

    if st.button("Run OCR Pipeline"):

        with st.spinner("Processing PDF..."):

            try:

                output_path = process_pdf(pdf_path)

                with open(output_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                st.success("Processing Complete")

                st.json(data)

                st.download_button(
                    label="Download JSON",
                    data=json.dumps(data, indent=4),
                    file_name="final_output.json",
                    mime="application/json"
                )

            except Exception as e:

                st.error(str(e))