import io
import json
import zipfile
import traceback
import streamlit as st
from pipeline import process_pdf, process_reference

st.set_page_config(page_title="OCR QA Extractor", layout="wide")
st.title("📘 OCR Question Answer Extractor")

# =========================================================
# HELPERS
# =========================================================

def clean_json(obj):
    """Recursively replace \\n with space in all string values."""
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json(i) for i in obj]
    if isinstance(obj, str):
        return obj.replace("\n", " ").strip()
    return obj


def to_json_bytes(obj):
    return json.dumps(clean_json(obj), ensure_ascii=False, indent=2).encode("utf-8")


def make_zip(files: dict) -> bytes:
    """files = {filename: bytes}"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    buf.seek(0)
    return buf.read()


def show_download(label, data_bytes, file_name):
    st.download_button(
        label=label,
        data=data_bytes,
        file_name=file_name,
        mime="application/json"
    )

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
# FILE UPLOADERS
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
# VALIDATE
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
# RUN PIPELINE
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

            ocr_bytes = to_json_bytes(ocr_json)
            qa_bytes  = to_json_bytes(qa_pairs)

            st.divider()
            st.subheader("📄 Assignment OCR JSON")
            st.json(clean_json(ocr_json))
            show_download("⬇ Download Assignment OCR JSON", ocr_bytes, "assignment_ocr.json")

            st.divider()
            st.subheader("🧠 Q-A Pairs")
            st.json(clean_json(qa_pairs))
            show_download("⬇ Download QA JSON", qa_bytes, "qa_output.json")

            st.divider()
            st.subheader("📦 Download All")
            zip_bytes = make_zip({
                "assignment_ocr.json": ocr_bytes,
                "qa_output.json": qa_bytes
            })
            st.download_button(
                label="⬇ Download All as ZIP",
                data=zip_bytes,
                file_name="assignment_output.zip",
                mime="application/zip"
            )

        # =====================================================
        # MODE 2 — Reference Book only
        # =====================================================

        elif mode.startswith("📚 Reference Book only"):

            reference_file.seek(0)
            ref_ocr_json = process_reference(
                reference_file,
                status_callback=update_status
            )
            progress.progress(100)
            status.success("✅ Done!")

            ref_bytes = to_json_bytes(ref_ocr_json)

            st.divider()
            st.subheader("📚 Reference Book OCR JSON")
            st.json(clean_json(ref_ocr_json))
            show_download("⬇ Download Reference OCR JSON", ref_bytes, "reference_ocr.json")

        # =====================================================
        # MODE 3 — Assignment + Reference
        # =====================================================

        elif mode.startswith("📄 + 📚"):

            assignment_file.seek(0)
            update_status("Processing assignment PDF...")
            ocr_json, qa_pairs = process_pdf(
                assignment_file,
                status_callback=update_status
            )
            progress.progress(50)

            reference_file.seek(0)
            update_status("Processing reference book...")
            ref_ocr_json = process_reference(
                reference_file,
                status_callback=update_status
            )
            progress.progress(100)
            status.success("✅ Done!")

            ocr_bytes = to_json_bytes(ocr_json)
            qa_bytes  = to_json_bytes(qa_pairs)
            ref_bytes = to_json_bytes(ref_ocr_json)

            st.divider()
            st.subheader("📄 Assignment OCR JSON")
            st.json(clean_json(ocr_json))
            show_download("⬇ Download Assignment OCR JSON", ocr_bytes, "assignment_ocr.json")

            st.divider()
            st.subheader("📚 Reference Book OCR JSON")
            st.json(clean_json(ref_ocr_json))
            show_download("⬇ Download Reference OCR JSON", ref_bytes, "reference_ocr.json")

            st.divider()
            st.subheader("🧠 Q-A Pairs")
            st.json(clean_json(qa_pairs))
            show_download("⬇ Download QA JSON", qa_bytes, "qa_output.json")

            st.divider()
            st.subheader("📦 Download All")
            zip_bytes = make_zip({
                "assignment_ocr.json": ocr_bytes,
                "reference_ocr.json": ref_bytes,
                "qa_output.json":     qa_bytes
            })
            st.download_button(
                label="⬇ Download All as ZIP",
                data=zip_bytes,
                file_name="all_outputs.zip",
                mime="application/zip"
            )

    except Exception as e:
        st.error("❌ Pipeline Failed")
        st.code(str(e))
        st.subheader("Full Error Trace")
        st.code(traceback.format_exc())