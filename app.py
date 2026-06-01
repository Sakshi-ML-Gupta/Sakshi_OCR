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
        "📄 Assignment PDF only  →  OCR JSON + QA JSON",
        "📚 Reference Book only  →  OCR JSON only",
        "📄 + 📚 Assignment + Reference  →  OCR JSON (both) + QA JSON",
    ],
    index=0
)

st.divider()

# =========================================================
# FILE UPLOADERS based on mode
# =========================================================

assignment_file = None
reference_file  = None

if mode.startswith("📄 Assignment PDF only"):
    assignment_file = st.file_uploader(
        "Upload Assignment PDF", type=["pdf"], key="assignment"
    )

elif mode.startswith("📚 Reference Book only"):
    reference_file = st.file_uploader(
        "Upload Reference Book PDF", type=["pdf"], key="reference"
    )

elif mode.startswith("📄 + 📚"):
    col1, col2 = st.columns(2)
    with col1:
        assignment_file = st.file_uploader(
            "Upload Assignment PDF", type=["pdf"], key="assignment"
        )
    with col2:
        reference_file = st.file_uploader(
            "Upload Reference Book PDF", type=["pdf"], key="reference"
        )

# =========================================================
# VALIDATE UPLOADS
# =========================================================

ready = False

if mode.startswith("📄 Assignment PDF only") and assignment_file:
    st.success(f"✅ Assignment: {assignment_file.name}")
    ready = True

elif mode.startswith("📚 Reference Book only") and reference_file:
    st.success(f"✅ Reference: {reference_file.name}")
    ready = True

elif mode.startswith("📄 + 📚") and assignment_file and reference_file:
    st.success(f"✅ Assignment: {assignment_file.name}")
    st.success(f"✅ Reference: {reference_file.name}")
    ready = True

elif mode.startswith("📄 + 📚") and (assignment_file or reference_file):
    st.warning("⚠️ Please upload both files to continue")

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

        status.info("🔄 Starting pipeline...")
        progress.progress(5)

        # =====================================================
        # MODE 1 — Assignment only
        # =====================================================

        if mode.startswith("📄 Assignment PDF only"):

            assignment_file.seek(0)
            ocr_json, qa_pairs = process_pdf(
                assignment_file,
                status_callback=update_status
            )

            progress.progress(100)
            status.success("✅ Done!")

            st.divider()
            st.subheader("📄 Assignment OCR JSON")
            st.json(ocr_json)
            st.download_button(
                label="⬇ Download Assignment OCR JSON",
                data=json.dumps(ocr_json, ensure_ascii=False, indent=2),
                file_name="assignment_ocr.json",
                mime="application/json"
            )

            st.divider()
            st.subheader("🧠 Q-A Pairs")
            st.json(qa_pairs)
            st.download_button(
                label="⬇ Download QA JSON",
                data=json.dumps(qa_pairs, ensure_ascii=False, indent=2),
                file_name="qa_output.json",
                mime="application/json"
            )

        # =====================================================
        # MODE 2 — Reference Book only
        # =====================================================

        elif mode.startswith("📚 Reference Book only"):

            reference_file.seek(0)
            file_bytes = reference_file.read()
            file_name  = reference_file.name

            update_status("Preprocessing reference book...")
            processed = preprocess_pdf(file_bytes)
            progress.progress(20)

            pages = run_ocr(
                processed,
                file_name,
                status_callback=update_status
            )
            progress.progress(90)

            ref_ocr_json = build_ocr_json(pages)
            progress.progress(100)
            status.success("✅ Done!")

            st.divider()
            st.subheader("📚 Reference Book OCR JSON")
            st.json(ref_ocr_json)
            st.download_button(
                label="⬇ Download Reference OCR JSON",
                data=json.dumps(ref_ocr_json, ensure_ascii=False, indent=2),
                file_name="reference_ocr.json",
                mime="application/json"
            )

        # =====================================================
        # MODE 3 — Assignment + Reference
        # =====================================================

        elif mode.startswith("📄 + 📚"):

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
            file_bytes = reference_file.read()
            file_name  = reference_file.name

            update_status("Preprocessing reference book...")
            processed = preprocess_pdf(file_bytes)

            pages = run_ocr(
                processed,
                file_name,
                status_callback=update_status
            )
            progress.progress(90)

            ref_ocr_json = build_ocr_json(pages)
            progress.progress(100)
            status.success("✅ Done!")

            # -- Output 1: Assignment OCR --
            st.divider()
            st.subheader("📄 Assignment OCR JSON")
            st.json(ocr_json)
            st.download_button(
                label="⬇ Download Assignment OCR JSON",
                data=json.dumps(ocr_json, ensure_ascii=False, indent=2),
                file_name="assignment_ocr.json",
                mime="application/json"
            )

            # -- Output 2: Reference OCR --
            st.divider()
            st.subheader("📚 Reference Book OCR JSON")
            st.json(ref_ocr_json)
            st.download_button(
                label="⬇ Download Reference OCR JSON",
                data=json.dumps(ref_ocr_json, ensure_ascii=False, indent=2),
                file_name="reference_ocr.json",
                mime="application/json"
            )

            # -- Output 3: QA --
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