"""
Streamlit frontend.

Run with:  streamlit run app.py
Requires DATALAB_API_KEY and GROQ_API_KEY as environment variables
(or a .env file loaded via python-dotenv — see README).
"""
import os
import time
import traceback

import streamlit as st

from pipeline import run_pipeline, save_outputs

st.set_page_config(page_title="Assignment Booklet Q&A Extractor", layout="wide")
st.title("📄 Assignment Booklet → Structured Q&A Pairs")
st.caption(
    "Upload a scanned, handwritten assignment/exam booklet. The pipeline OCRs it, "
    "figures out which pages are question paper / cover / answers, and matches every "
    "question to its raw answer text — no manual copy-paste."
)

with st.sidebar:
    st.subheader("Status")
    log_box = st.empty()
    st.caption("Live progress from the pipeline appears here.")

uploaded = st.file_uploader("Upload booklet (PDF)", type=["pdf"])

if "logs" not in st.session_state:
    st.session_state.logs = []


def status_callback(message: str) -> None:
    st.session_state.logs.append(f"{time.strftime('%H:%M:%S')} — {message}")
    log_box.code("\n".join(st.session_state.logs[-30:]), language=None)


if uploaded is not None:
    if st.button("Run pipeline", type="primary"):
        st.session_state.logs = []
        missing = [k for k in ("DATALAB_API_KEY", "GROQ_API_KEY") if not os.environ.get(k)]
        if missing:
            st.error(f"Missing environment variable(s): {', '.join(missing)}. Set them and reload.")
        else:
            pdf_bytes = uploaded.getvalue()
            try:
                with st.spinner("Processing…"):
                    result = run_pipeline(pdf_bytes, filename=uploaded.name, status_cb=status_callback)
                st.success("Done!")

                # ---- summary ----
                qa_pairs = result["qa_pairs"]
                matched = sum(1 for r in qa_pairs if r["found"])
                col1, col2, col3 = st.columns(3)
                col1.metric("Pages OCR'd", result["ocr"]["page_count"])
                col2.metric("Questions found", len(result["questions"]))
                col3.metric("Answers matched", f"{matched}/{len(qa_pairs)}")

                st.subheader("Q&A Pairs")
                for r in qa_pairs:
                    with st.expander(f"{r['question']}  {'✅' if r['found'] else '❌ not matched'}"):
                        st.markdown(f"**Question:** {r['question']}")
                        if r["found"]:
                            st.markdown("**Answer (raw OCR):**")
                            st.text(r["answer"])
                            st.caption(
                                f"Source page(s): {r['source_pages']} · "
                                f"lines {r['start_line']}–{r['end_line']} · "
                                f"confidence: {r['confidence']}"
                            )
                        else:
                            st.warning("No confident match found for this question.")

                base_name = os.path.splitext(uploaded.name)[0]
                paths = save_outputs(result, out_dir="outputs", base_name=base_name)

                st.subheader("Downloads")
                dl1, dl2 = st.columns(2)
                with open(paths["ocr_json"], "rb") as f:
                    dl1.download_button("⬇️ Download OCR JSON", f, file_name=os.path.basename(paths["ocr_json"]))
                with open(paths["qa_pairs_json"], "rb") as f:
                    dl2.download_button(
                        "⬇️ Download Q&A Pairs JSON", f, file_name=os.path.basename(paths["qa_pairs_json"])
                    )

            except Exception as exc:  # noqa: BLE001
                st.error(f"Pipeline failed: {exc}")
                st.code(traceback.format_exc())
else:
    st.info("Upload a PDF to get started.")
