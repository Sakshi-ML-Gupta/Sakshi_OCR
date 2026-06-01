import json
import traceback
import streamlit as st
from pipeline import process_pdf, run_ocr, build_ocr_json, preprocess_pdf

st.set_page_config(page_title="OCR QA Extractor", layout="wide")
st.title("📘 OCR Question Answer Extractor")

# =========================================================
# MODE SELECTION
# =========================================================

mode = st.radio(
    "Select Mode",
    options=[
        "Mode 1 — Assignment PDF only (OCR + Q-A)",
        "Mode 2 — Reference Book only (OCR only)",
        "Mode 3 — Assignment + Reference Book (OCR + OCR + Q-A)",
    ],
    index=0
)

st.divider()

# =========================================================
# FILE UPLOADERS — shown based on mode
# =========================================================

assignment_file = None
reference_file  = None

if mode.startswith("Mode 1"):
    assignment_file = st.file_uploader("Upload Assignment PDF", type=["pdf"])

elif mode.startswith("Mode 2"):
    reference_file = st.file_uploader("Upload Reference Book PDF", type=["pdf"])

elif mode.startswith("Mode 3"):
    col1, col2 = st.columns(2)
    with col1:
        assignment_file = st.file_uploader("Upload Assignment PDF", type=["pdf"])
    with col2:
        reference_file = st.file_uploader("Upload Reference Book PDF", type=["pdf"])

# =========================================================
# VALIDATE UPLOADS BEFORE SHOWING BUTTON
# =========================================================

ready = False

if mode.startswith("Mode 1") and assignment_file:
    st.success(f"✅ Assignment: {assignment_file.name}")
    ready = True

elif mode.startswith("Mode 2") and reference_file:
    st.success(f"✅ Reference: {reference_file.name}")
    ready = True

elif mode.startswith("Mode 3") and assignment_file and reference_file:
    st.success(f"✅ Assignment: {assignment_file.name}")
    st.success(f"✅ Reference:  {reference_file.name}")
    ready = True

elif mode.startswith("Mode 3") and (assignment_file or reference_file):
    st.warning("⚠️ Please upload both PDFs to proceed.")

# =========================================================
# RUN BUTTON
# =========================================================

if ready and st.button("🚀 Run Pipeline"):

    try:
        progress  = st.progress(0)
        status    = st.empty()
        log_box   = st.empty()
        log_lines = []

        def update_status(msg):
            log_lines.append(msg)
            log_box.code("\n".join(log_lines), language="text")

        status.info("🔄 Starting...")
        progress.progress(5)

        # ──────────────────────────────────────────────────
        # MODE 1 — Assignment only
        # ──────────────────────────────────────────────────

        if mode.startswith("Mode 1"):

            assignment_file.seek(0)
            update_status("Processing assignment PDF...")

            ocr_json, qa_pairs = process_pdf(
                assignment_file,
                status_callback=update_status
            )

            progress.progress(100)
            status.success("✅ Done!")

            # OCR JSON
            st.divider()
            st.subheader("📄 Assignment OCR JSON")
            st.json(ocr_json)
            st.download_button(
                label="⬇ Download OCR JSON",
                data=json.dumps(ocr_json, ensure_ascii=False, indent=2),
                file_name="assignment_ocr.json",
                mime="application/json"
            )

            # QA JSON
            st.divider()
            st.subheader("🧠 Q-A Pairs")
            st.json(qa_pairs)
            st.download_button(
                label="⬇ Download QA JSON",
                data=json.dumps(qa_pairs, ensure_ascii=False, indent=2),
                file_name="qa_output.json",
                mime="application/json"
            )

        # ──────────────────────────────────────────────────
        # MODE 2 — Reference book only
        # ──────────────────────────────────────────────────

        elif mode.startswith("Mode 2"):

            reference_file.seek(0)
            update_status("Processing reference book PDF...")

            ref_bytes = reference_file.read()
            ref_name  = reference_file.name

            update_status("Preprocessing reference book...")
            ref_processed = preprocess_pdf(ref_bytes)

            update_status("Running OCR on reference book...")
            ref_pages = run_ocr(
                ref_processed,
                ref_name,
                status_callback=update_status
            )

            ref_ocr_json = build_ocr_json(ref_pages)

            progress.progress(100)
            status.success("✅ Done!")

            # Reference OCR JSON
            st.divider()
            st.subheader("📚 Reference Book OCR JSON")
            st.json(ref_ocr_json)
            st.download_button(
                label="⬇ Download Reference OCR JSON",
                data=json.dumps(ref_ocr_json, ensure_ascii=False, indent=2),
                file_name="reference_ocr.json",
                mime="application/json"
            )

        # ──────────────────────────────────────────────────
        # MODE 3 — Assignment + Reference
        # ──────────────────────────────────────────────────

        elif mode.startswith("Mode 3"):

            # -- Assignment --
            assignment_file.seek(0)
            update_status("Processing assignment PDF...")

            ocr_json, qa_pairs = process_pdf(
                assignment_file,
                status_callback=update_status
            )

            progress.progress(50)

            # -- Reference --
            reference_file.seek(0)
            update_status("Processing reference book PDF...")

            ref_bytes = reference_file.read()
            ref_name  = reference_file.name

            update_status("Preprocessing reference book...")
            ref_processed = preprocess_pdf(ref_bytes)

            update_status("Running OCR on reference book...")
            ref_pages = run_ocr(
                ref_processed,
                ref_name,
                status_callback=update_status
            )

            ref_ocr_json = build_ocr_json(ref_pages)

            progress.progress(100)
            status.success("✅ Done!")

            # Assignment OCR JSON
            st.divider()
            st.subheader("📄 Assignment OCR JSON")
            st.json(ocr_json)
            st.download_button(
                label="⬇ Download Assignment OCR JSON",
                data=json.dumps(ocr_json, ensure_ascii=False, indent=2),
                file_name="assignment_ocr.json",
                mime="application/json"
            )

            # Reference OCR JSON
            st.divider()
            st.subheader("📚 Reference Book OCR JSON")
            st.json(ref_ocr_json)
            st.download_button(
                label="⬇ Download Reference OCR JSON",
                data=json.dumps(ref_ocr_json, ensure_ascii=False, indent=2),
                file_name="reference_ocr.json",
                mime="application/json"
            )

            # QA JSON
            st.divider()
            st.subheader("🧠 Q-A Pairs")
            st.json(qa_pairs)
            st.download_button(
                label="⬇ Download QA JSON",
                data=json.dumps(qa_pairs, ensure_ascii=False, indent=2),
                file_name="qa_output.json",
                mime="application/json"
            )

    except Exception as e:
        st.error("❌ Pipeline Failed")
        st.code(str(e))
        st.subheader("Full Error Trace")
        st.code(traceback.format_exc())