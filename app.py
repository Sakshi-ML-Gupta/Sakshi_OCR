"""
app.py — CLI entry point for the Assignment PDF -> Q-A pair extractor.

Usage (path given explicitly):
  python app.py assignment.pdf \
      --mistral-key $MISTRAL_API_KEY \
      --groq-key $GROQ_API_KEY \
      --out result.json \
      [--minimal]

Usage (no path given): the script will look for PDFs in --input-dir
(default "input/") and process each one, or prompt interactively if that
folder doesn't exist or is empty.

Either API key can be omitted:
  - No Mistral key: OCR is skipped; scanned/handwritten pages will be low quality.
  - No Groq key: only regex-confident pairs are returned; unstructured
    portions of the document are skipped rather than guessed at.

All extraction/pairing logic lives in pipeline.py — this file is just the CLI.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

from pipeline import run_pipeline


def resolve_pdf_paths(pdf_arg: str | None, input_dir: str) -> list[str]:
    """
    Figures out which PDF(s) to process:
      1. explicit path given on the command line -> just that file
      2. no path given, --input-dir exists and has PDFs -> all of them
      3. no path given, nothing found -> prompt interactively for a path
    """
    if pdf_arg:
        if not os.path.isfile(pdf_arg):
            sys.exit(f"Error: '{pdf_arg}' is not a file that exists.")
        return [pdf_arg]

    if os.path.isdir(input_dir):
        found = sorted(glob.glob(os.path.join(input_dir, "*.pdf")))
        if found:
            print(f"No PDF path given — found {len(found)} PDF(s) in '{input_dir}/'.", file=sys.stderr)
            return found

    # Nothing given, nothing found automatically — ask for it directly.
    entered = input("Enter path to the assignment PDF: ").strip().strip('"').strip("'")
    if not entered or not os.path.isfile(entered):
        sys.exit(f"Error: '{entered}' is not a file that exists.")
    return [entered]


def process_one(pdf_path: str, args) -> None:
    result = run_pipeline(pdf_path, args.mistral_key, args.groq_key, args.groq_model)
    output = result.to_minimal() if args.minimal else json.loads(result.model_dump_json())

    if args.out:
        out_path = args.out
    else:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        out_path = f"{base}_qa.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[{pdf_path}] Extracted {len(result.qa_pairs)} Q-A pairs "
          f"({result.unmatched_low_confidence} flagged low-confidence) -> {out_path}",
          file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="OCR + Q-A pair extraction for assignment PDFs")
    ap.add_argument("pdf", nargs="?", default=None,
                     help="path to the input PDF (optional — if omitted, "
                          "reads from --input-dir or prompts for a path)")
    ap.add_argument("--input-dir", default="input",
                     help="folder to scan for PDFs when no path is given (default: 'input/')")
    ap.add_argument("--mistral-key", default=None, help="Mistral API key (OCR for scanned/handwritten pages)")
    ap.add_argument("--groq-key", default=None, help="Groq API key (LLM fallback for unstructured Q/A segments)")
    ap.add_argument("--groq-model", default="llama-3.3-70b-versatile")
    ap.add_argument("--out", default=None,
                     help="output JSON path (default: '<pdf_name>_qa.json'; ignored when "
                          "processing multiple PDFs from --input-dir, which always uses that pattern)")
    ap.add_argument("--minimal", action="store_true", help="output only [{q,a}, ...] with no metadata")
    args = ap.parse_args()

    pdf_paths = resolve_pdf_paths(args.pdf, args.input_dir)

    if len(pdf_paths) > 1:
        args.out = None  # force per-file naming when batch-processing a folder

    for pdf_path in pdf_paths:
        process_one(pdf_path, args)


if __name__ == "__main__":
    main()
