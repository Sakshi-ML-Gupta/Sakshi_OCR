import os
import re
import uuid
from pathlib import Path

import streamlit as st

from pipeline import process_pdf


st.set_page_config(page_title="OCR QA Extractor", layout="wide")
st.title("OCR QA Extraction Pipeline")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    original_name = Path(uploaded_file.name).name
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(original_name).stem).strip("._")
    safe_stem = safe_stem or "upload"
    pdf_path = upload_dir / f"{safe_stem}_{uuid.uuid4().hex[:8]}.pdf"

    with open(pdf_path, "xb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded")

    if st.button("Run OCR Pipeline"):
        with st.spinner("Processing PDF..."):
            try:
                final_json_path = process_pdf(str(pdf_path))
                st.success("Pipeline complete")

                with open(final_json_path, "r", encoding="utf-8") as f:
                    data = f.read()

                st.json(data)
                st.download_button(
                    label="Download JSON",
                    data=data,
                    file_name=os.path.basename(final_json_path),
                    mime="application/json",
                )
            except Exception as e:
                st.error(str(e))
