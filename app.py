"""
app.py — Streamlit UI for the Assignment PDF -> Q-A pair extractor.

Run locally with:
  streamlit run app.py

For a hosted Streamlit Cloud app, set MISTRAL_API_KEY / GROQ_API_KEY in
Settings -> Secrets (see the sidebar note below) so users don't have to
paste their own keys, or leave the sidebar fields for users to enter theirs.

All extraction/pairing logic lives in pipeline.py — this file is just the UI.
"""
from __future__ import annotations
import json
import tempfile
from pathlib import Path

import streamlit as st

from pipeline import run_pipeline

st.set_page_config(page_title="Assignment PDF → Q-A Extractor", page_icon="📝", layout="wide")

st.title("📝 Assignment PDF → Q-A Pair Extractor")
st.caption(
    "Upload one or more assignment PDFs. Text is extracted natively where possible "
    "(Mistral OCR fallback for scans/handwriting), then split into verbatim Q-A pairs "
    "(regex first, Groq LLM fallback for unstructured sections). Nothing is paraphrased "
    "or invented — every pair is checked back against the source text."
)

# ---------------------------------------------------------------------------
# Sidebar: API keys & options
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    # Falls back to Streamlit Cloud secrets if configured, so the app can be
    # deployed without asking every user to paste a key.
    default_mistral = st.secrets.get("MISTRAL_API_KEY", "") if hasattr(st, "secrets") else ""
    default_groq = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""

    mistral_key = st.text_input(
        "Mistral API key (OCR)", value=default_mistral, type="password",
        help="Used for scanned/handwritten pages. Leave blank to skip OCR — "
             "only pages with a real text layer will extract well.",
    )
    groq_key = st.text_input(
        "Groq API key (LLM fallback)", value=default_groq, type="password",
        help="Used to segment unstructured/ambiguous sections into Q-A pairs. "
             "Leave blank to only extract regex-confident pairs.",
    )
    groq_model = st.text_input("Groq model", value="llama-3.3-70b-versatile")
    minimal_output = st.checkbox("Minimal JSON (just q/a, no metadata)", value=False)

    st.divider()
    st.caption(
        "On Streamlit Cloud: add these under **Settings → Secrets** as\n\n"
        "```\nMISTRAL_API_KEY = \"...\"\nGROQ_API_KEY = \"...\"\n```\n\n"
        "and this page will pick them up automatically."
    )

# ---------------------------------------------------------------------------
# Main: upload + run
# ---------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload assignment PDF(s)", type=["pdf"], accept_multiple_files=True
)

run_clicked = st.button("Extract Q-A pairs", type="primary", disabled=not uploaded_files)

if run_clicked and uploaded_files:
    for uf in uploaded_files:
        st.subheader(f"📄 {uf.name}")

        # run_pipeline needs a real file path, so persist the upload to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name

        try:
            with st.spinner(f"Extracting from {uf.name}..."):
                result = run_pipeline(
                    tmp_path,
                    mistral_key or None,
                    groq_key or None,
                    groq_model,
                )
        except Exception as e:
            st.error(f"Failed to process {uf.name}: {e}")
            continue
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if not result.qa_pairs:
            st.warning(
                "No Q-A pairs were extracted. If this is a scanned/handwritten PDF, "
                "add a Mistral API key. If the format is unstructured, add a Groq API key."
            )
            continue

        output = result.to_minimal() if minimal_output else json.loads(result.model_dump_json())

        col1, col2, col3 = st.columns(3)
        col1.metric("Q-A pairs found", len(result.qa_pairs))
        col2.metric("Pages", result.num_pages)
        col3.metric("Flagged low-confidence", result.unmatched_low_confidence)

        # Quick readable view
        for pair in result.qa_pairs:
            flag = "" if pair.verified else " ⚠️ low-confidence"
            with st.expander(f"Q{pair.id}: {pair.q[:80]}{'...' if len(pair.q) > 80 else ''}{flag}"):
                st.markdown(f"**Question:** {pair.q}")
                st.markdown(f"**Answer:** {pair.a if pair.a else '_(no answer found)_'}")
                st.caption(
                    f"page {pair.page_start}-{pair.page_end} · source: {pair.source} · "
                    f"confidence: {pair.confidence}"
                )

        st.download_button(
            f"⬇️ Download JSON for {uf.name}",
            data=json.dumps(output, ensure_ascii=False, indent=2),
            file_name=f"{Path(uf.name).stem}_qa.json",
            mime="application/json",
            key=f"dl_{uf.name}",
        )

        st.divider()

elif not uploaded_files:
    st.info("Upload a PDF above to get started.")
