import json
import streamlit as st

# NOTE: change this import if your pipeline module has a different filename.
# This assumes the module you shared is saved as `pipeline.py` in the same folder.
import pipeline
pipeline.process_pdf(file_input)  # This should work if the file is named pipeline.py


st.set_page_config(page_title="Exam Answer Extractor", layout="wide")
st.title("📝 Exam Booklet -> Question/Answer Extractor")

st.markdown(
    "Upload a scanned exam booklet (PDF). The app will OCR it, detect the "
    "question paper pages, extract the official questions, and map each "
    "question to the student's answer."
)

uploaded_file = st.file_uploader("Upload exam booklet PDF", type=["pdf"])

if "log_lines" not in st.session_state:
    st.session_state.log_lines = []

if uploaded_file is not None:
    if st.button("Process document", type="primary"):
        st.session_state.log_lines = []
        log_box = st.empty()

        def status_callback(msg: str):
            st.session_state.log_lines.append(msg)
            # show last ~25 log lines so the box doesn't grow forever
            log_box.code("\n".join(st.session_state.log_lines[-25:]))

        file_bytes = uploaded_file.read()

        try:
            with st.spinner("Processing... this can take a few minutes for large documents"):
                ocr_json, qa_pairs = pipeline.process_pdf(
                    (uploaded_file.name, file_bytes),
                    status_callback=status_callback,
                )
        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.stop()

        st.session_state.ocr_json = ocr_json
        st.session_state.qa_pairs = qa_pairs
        st.success("Done!")

if "qa_pairs" in st.session_state:
    qa_pairs = st.session_state.qa_pairs
    ocr_json = st.session_state.ocr_json

    matched = sum(1 for p in qa_pairs if p["matched"])
    st.subheader(f"Results — {matched} of {len(qa_pairs)} questions matched")

    for i, p in enumerate(qa_pairs, start=1):
        with st.expander(f"Q{i}: {p['question'][:90]}", expanded=False):
            st.markdown(f"**Question:**\n\n{p['question']}")
            if p["matched"]:
                pages_info = ""
                if p.get("start_page") is not None:
                    pages_info = f" (pages {p['start_page']}–{p['end_page']})"
                st.markdown(f"**Answer (lines {p['start_line']}–{p['end_line']}{pages_info}):**")
                st.write(p["answer"])
                with st.popover("Show raw OCR (unedited)"):
                    st.write(p["answer_raw"])
            else:
                st.warning("No matching answer found for this question.")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download OCR JSON",
            data=json.dumps(ocr_json, ensure_ascii=False, indent=2),
            file_name="ocr.json",
            mime="application/json",
        )
    with col2:
        st.download_button(
            "Download Q&A pairs JSON",
            data=json.dumps(qa_pairs, ensure_ascii=False, indent=2),
            file_name="qa_pairs.json",
            mime="application/json",
        )

st.divider()
st.caption(
    "Requires DATALAB_API_KEY and GROQ_API_KEY set in Streamlit secrets "
    "(.streamlit/secrets.toml) or as environment variables."
)
