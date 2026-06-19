import io
import json
import os
import re
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

try:
    import streamlit as st
except Exception:
    st = None

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

try:
    import fitz
except ImportError:
    fitz = None

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None


load_dotenv()

OUTPUT_OCR_DIR = Path("outputs/ocr_json")
OUTPUT_FINAL_DIR = Path("outputs/final_json")

NOISE_RE = re.compile(
    r"(?:^TOPIC\s*$|^DATE\s*$|^PAGE\s*NO|Teacher'?s?\s*Signature|^\d{1,3}$)",
    re.IGNORECASE,
)


def get_config(name, default=None):
    value = os.getenv(name)

    if value:
        return value

    if st is not None:
        try:
            if name in st.secrets:
                return str(st.secrets[name])
        except Exception:
            pass

    return default


OCR_PROVIDER = get_config("OCR_PROVIDER", "surya").strip().lower()
GEMINI_API_KEY = get_config("GEMINI_API_KEY")
GEMINI_MODEL = get_config("GEMINI_MODEL", "gemini-3.5-flash")

PADDLEOCR_LANG = get_config("PADDLEOCR_LANG", "hi")
SURYA_LANGS = [
    item.strip()
    for item in get_config(
        "SURYA_LANGS",
        "en,hi,bn,ta,te,gu,kn,ml,mr,pa,or",
    ).split(",")
    if item.strip()
]


def require_config(name, value):
    if not value:
        raise RuntimeError(f"Missing {name}. Add it to Streamlit secrets or .env.")
    return value.strip()


def clean_ocr_lines(text):
    lines = []
    seen = set()

    for raw_line in str(text).splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()

        if not line or NOISE_RE.search(line):
            continue

        if line not in seen:
            seen.add(line)
            lines.append(line)

    return lines


def pdf_to_images(pdf_bytes, dpi=220):
    if fitz is None:
        raise RuntimeError("PyMuPDF missing. Add pymupdf to requirements.txt.")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    images = []

    try:
        for page_number, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            images.append((page_number, image))
    finally:
        doc.close()

    return images


def extract_text_from_any_ocr_result(result):
    lines = []

    def walk(value):
        if value is None:
            return

        if isinstance(value, str):
            if value.strip():
                lines.append(value)
            return

        if isinstance(value, dict):
            for key in ("text", "rec_text", "label"):
                if key in value and str(value[key]).strip():
                    lines.append(str(value[key]))

            for key in ("rec_texts", "texts"):
                if key in value and isinstance(value[key], list):
                    lines.extend(str(item) for item in value[key] if str(item).strip())

            for child in value.values():
                walk(child)
            return

        if hasattr(value, "text") and str(value.text).strip():
            lines.append(str(value.text))

        if hasattr(value, "text_lines"):
            walk(value.text_lines)

        if hasattr(value, "dict"):
            try:
                walk(value.dict())
            except Exception:
                pass

        if isinstance(value, tuple):
            if len(value) >= 2 and isinstance(value[1], (int, float)):
                lines.append(str(value[0]))
            else:
                for child in value:
                    walk(child)
            return

        if isinstance(value, list):
            if len(value) >= 2 and isinstance(value[1], tuple):
                lines.append(str(value[1][0]))
                return

            for child in value:
                walk(child)

    walk(result)
    return clean_ocr_lines("\n".join(lines))


def run_paddle_ocr(pdf_bytes):
    if PaddleOCR is None:
        raise RuntimeError(
            "PaddleOCR missing. Add paddleocr and paddlepaddle to requirements.txt."
        )

    print(f"Loading PaddleOCR lang={PADDLEOCR_LANG}...")
    ocr = PaddleOCR(use_angle_cls=True, lang=PADDLEOCR_LANG, show_log=False)

    pages = []

    for page_number, image in pdf_to_images(pdf_bytes):
        print(f"OCR page {page_number} with PaddleOCR...")
        result = ocr.ocr(np.array(image), cls=True)
        lines = extract_text_from_any_ocr_result(result)

        pages.append({
            "page_number": page_number,
            "text": lines,
        })

    return {
        "total_pages": len(pages),
        "pages": pages,
    }


def run_surya_ocr(pdf_bytes):
    try:
        from surya.detection import DetectionPredictor
        from surya.ocr import run_ocr
        from surya.recognition import RecognitionPredictor
    except ImportError as e:
        raise RuntimeError("Surya OCR missing. Add surya-ocr to requirements.txt.") from e

    print(f"Loading Surya OCR langs={','.join(SURYA_LANGS)}...")

    images_with_numbers = pdf_to_images(pdf_bytes)
    page_numbers = [item[0] for item in images_with_numbers]
    images = [item[1] for item in images_with_numbers]

    recognition_predictor = RecognitionPredictor()
    detection_predictor = DetectionPredictor()
    lang_lists = [SURYA_LANGS for _ in images]

    print("Running Surya OCR...")
    predictions = run_ocr(
        images,
        lang_lists,
        recognition_predictor,
        detection_predictor,
    )

    pages = []

    for page_number, prediction in zip(page_numbers, predictions):
        lines = extract_text_from_any_ocr_result(prediction)

        pages.append({
            "page_number": page_number,
            "text": lines,
        })

    return {
        "total_pages": len(pages),
        "pages": pages,
    }


def run_ocr(pdf_bytes):
    if OCR_PROVIDER == "surya":
        return run_surya_ocr(pdf_bytes)

    if OCR_PROVIDER == "paddle":
        return run_paddle_ocr(pdf_bytes)

    raise RuntimeError("OCR_PROVIDER must be either surya or paddle.")


def extract_json_object(text):
    text = str(text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        raise RuntimeError(f"Gemini did not return JSON. Response was: {text[:500]}")

    return json.loads(match.group(0))


def gemini_json(system_prompt, user_payload):
    api_key = require_config("GEMINI_API_KEY", GEMINI_API_KEY)

    prompt = f"""
{system_prompt}

Input JSON:
{json.dumps(user_payload, ensure_ascii=False)}

Return valid JSON only. No markdown. No explanation.
"""

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{GEMINI_MODEL}:generateContent"
    )

    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    }

    try:
        with httpx.Client(timeout=240) as client:
            response = client.post(
                url,
                params={"key": api_key},
                json=body,
            )
    except httpx.RequestError as e:
        raise RuntimeError(f"Could not reach Gemini. Details: {e}") from e

    if response.status_code >= 400:
        raise RuntimeError(f"Gemini error {response.status_code}: {response.text[:700]}")

    data = response.json()

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        raise RuntimeError(f"Unexpected Gemini response: {data}") from e

    return extract_json_object(text)


def pages_for_prompt(pages):
    return [
        {
            "page_number": page["page_number"],
            "text": "\n".join(page["text"]),
        }
        for page in pages
    ]


def normalize_questions(questions):
    normalized = []
    seen = set()

    for index, question in enumerate(questions, start=1):
        qid = str(
            question.get("question_id")
            or question.get("id")
            or f"Q{index}"
        ).strip()

        qtext = re.sub(
            r"\s+",
            " ",
            str(question.get("question") or question.get("text") or ""),
        ).strip()

        if not qid or not qtext:
            continue

        key = qid.lower()

        if key in seen:
            suffix = 2
            while f"{key}.{suffix}" in seen:
                suffix += 1
            qid = f"{qid}.{suffix}"
            key = qid.lower()

        seen.add(key)

        normalized.append({
            "question_id": qid,
            "question": qtext,
            "page_number": question.get("page_number"),
        })

    if not normalized:
        raise RuntimeError("No usable questions extracted.")

    return normalized


def extract_official_questions_with_llm(pages):
    system_prompt = """
You extract the printed question paper from OCR text for exam revaluation.

Return only JSON:
{
  "questions": [
    {
      "question_id": "Q1",
      "question": "question text",
      "page_number": 1
    }
  ]
}

Rules:
- Follow the question paper order exactly.
- If a parent question has sub-questions, create one row per sub-question.
- Use IDs like Q1, Q1.a, Q1.b, Q2.i, A1(i), B2 depending on the paper.
- Preserve section labels if printed.
- Do not include student answers as official questions.
- Do not invent missing questions.
"""

    payload = gemini_json(
        system_prompt,
        {"pages": pages_for_prompt(pages)},
    )

    return normalize_questions(payload.get("questions", []))


def flatten_answer_lines(pages, question_page_numbers):
    lines = []
    line_id = 1

    answer_pages = [
        page for page in pages
        if page["page_number"] not in question_page_numbers
    ]

    if not answer_pages:
        answer_pages = pages

    for page in answer_pages:
        for text in page["text"]:
            clean = re.sub(r"\s+", " ", text).strip()

            if not clean or NOISE_RE.search(clean):
                continue

            lines.append({
                "line_id": line_id,
                "page_number": page["page_number"],
                "text": clean,
            })

            line_id += 1

    return lines


def map_answer_spans_with_llm(questions, answer_lines):
    system_prompt = """
You map student answer OCR lines to official exam questions.

Return only JSON:
{
  "answer_spans": [
    {
      "question_id": "Q1.a",
      "start_line": 1,
      "end_line": 25,
      "confidence": 0.92,
      "notes": "short reason"
    }
  ]
}

Rules:
- question_id must exactly match one official question_id.
- Use official question order as the backbone.
- Map sub-questions separately.
- If the student writes labels like Q1.a, 1(a), A1(i), match that exact official question.
- Do not create spans for unanswered questions.
- Do not overlap spans.
- Prefer line positions. Do not rewrite the answer.
"""

    payload = gemini_json(
        system_prompt,
        {
            "official_questions": questions,
            "answer_lines": answer_lines,
        },
    )

    return payload.get("answer_spans", [])


def slice_answers(questions, answer_lines, spans):
    line_by_id = {
        line["line_id"]: line
        for line in answer_lines
    }

    question_by_id = {
        q["question_id"]: q
        for q in questions
    }

    qa_pairs = []
    used_question_ids = set()

    for span in sorted(spans, key=lambda item: item.get("start_line", 10**9)):
        qid = span.get("question_id")

        if qid not in question_by_id:
            continue

        if qid in used_question_ids:
            continue

        start = int(span.get("start_line", 0))
        end = int(span.get("end_line", 0))

        if start <= 0 or end < start:
            continue

        raw_lines = [
            line_by_id[line_id]["text"]
            for line_id in range(start, end + 1)
            if line_id in line_by_id
        ]

        answer = re.sub(r"\s+", " ", " ".join(raw_lines)).strip()

        if not answer:
            continue

        qa_pairs.append({
            "question_id": qid,
            "question": question_by_id[qid]["question"],
            "answer": answer,
            "mapping_confidence": span.get("confidence"),
        })

        used_question_ids.add(qid)

    return qa_pairs


def build_final_json(qa_pairs):
    return {
        "total_qa_pairs": len(qa_pairs),
        "qa_pairs": qa_pairs,
    }


def process_pdf(pdf_path):
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise RuntimeError("PDF not found")

    OUTPUT_OCR_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FINAL_DIR.mkdir(parents=True, exist_ok=True)

    pdf_bytes = pdf_file.read_bytes()

    ocr_json = run_ocr(pdf_bytes)

    ocr_output_path = OUTPUT_OCR_DIR / f"{pdf_file.stem}_ocr.json"

    with open(ocr_output_path, "w", encoding="utf-8") as f:
        json.dump(ocr_json, f, ensure_ascii=False, indent=4)

    pages = ocr_json["pages"]

    questions = extract_official_questions_with_llm(pages)

    question_page_numbers = {
        int(q["page_number"])
        for q in questions
        if str(q.get("page_number") or "").isdigit()
    }

    answer_lines = flatten_answer_lines(
        pages,
        question_page_numbers,
    )

    spans = map_answer_spans_with_llm(
        questions,
        answer_lines,
    )

    qa_pairs = slice_answers(
        questions,
        answer_lines,
        spans,
    )

    final_json = build_final_json(qa_pairs)

    final_output_path = OUTPUT_FINAL_DIR / f"{pdf_file.stem}_final.json"

    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)

    return str(final_output_path)
