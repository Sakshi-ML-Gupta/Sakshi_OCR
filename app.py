"""
app.py — CLI entry point for the Assignment PDF -> Q-A pair extractor.

Usage:
  python app.py assignment.pdf \
      --mistral-key $MISTRAL_API_KEY \
      --groq-key $GROQ_API_KEY \
      --out result.json \
      [--minimal]

Either key can be omitted:
  - No Mistral key: OCR is skipped; scanned/handwritten pages will be low quality.
  - No Groq key: only regex-confident pairs are returned; unstructured
    portions of the document are skipped rather than guessed at.

All extraction/pairing logic lives in pipeline.py — this file is just the CLI.
"""
from __future__ import annotations
import argparse
import json
import sys

from pipeline import run_pipeline


def main():
    ap = argparse.ArgumentParser(description="OCR + Q-A pair extraction for assignment PDFs")
    ap.add_argument("pdf", help="path to the input PDF")
    ap.add_argument("--mistral-key", default=None, help="Mistral API key (OCR for scanned/handwritten pages)")
    ap.add_argument("--groq-key", default=None, help="Groq API key (LLM fallback for unstructured Q/A segments)")
    ap.add_argument("--groq-model", default="llama-3.3-70b-versatile")
    ap.add_argument("--out", default="qa_output.json")
    ap.add_argument("--minimal", action="store_true", help="output only [{q,a}, ...] with no metadata")
    args = ap.parse_args()

    result = run_pipeline(args.pdf, args.mistral_key, args.groq_key, args.groq_model)

    output = result.to_minimal() if args.minimal else json.loads(result.model_dump_json())
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(result.qa_pairs)} Q-A pairs "
          f"({result.unmatched_low_confidence} flagged low-confidence) -> {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
