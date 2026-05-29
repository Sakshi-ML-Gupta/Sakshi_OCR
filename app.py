import json
import traceback
import streamlit as st

from pipeline import process_pdf

st.set_page_config(page_title="OCR QA Extractor", layout="wide")

st.title("📘 OCR Question Answer Extractor")

st.markdown("""
Upload a PDF file and extract:
1. Full OCR JSON
2. Question-Answer JSON
""")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:

    st.success("✅ PDF Uploaded Successfully")
    st.write("Filename:", uploaded_file.name)

    if st.button("🚀 Run OCR Pipeline"):

        try:
            uploaded_file.seek(0)

            progress = st.progress(0)
            status = st.empty()
            log_box = st.empty()
            log_lines = []

            def update_status(msg):
                log_lines.append(msg)
                log_box.code("\n".join(log_lines), language="text")

            status.info("🔄 Starting OCR Pipeline...")
            progress.progress(5)

            update_status("📤 Uploading PDF to Mistral OCR...")

            # ── pass status callback into process_pdf ──
            ocr_json, final_json = process_pdf(
                uploaded_file,
                status_callback=update_status
            )

            progress.progress(100)
            status.success("✅ Pipeline Completed Successfully")

            # ── OCR JSON ──
            st.divider()
            st.subheader("📄 OCR JSON Output")
            st.json(ocr_json)
            st.download_button(
                label="⬇ Download OCR JSON",
                data=json.dumps(ocr_json, ensure_ascii=False, indent=4),
                file_name="ocr_output.json",
                mime="application/json"
            )

            # ── QA JSON ──
            st.divider()
            st.subheader("🧠 Final Question Answer JSON")
            st.json(final_json)
            st.download_button(
                label="⬇ Download QA JSON",
                data=json.dumps(final_json, ensure_ascii=False, indent=4),
                file_name="qa_output.json",
                mime="application/json"
            )

        except Exception as e:
            st.error("❌ Pipeline Failed")
            st.code(str(e))
            st.subheader("Full Error Trace")
            st.code(traceback.format_exc())