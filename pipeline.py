import io
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None


load_dotenv()

CHANDRA_OCR_URL = os.getenv("CHANDRA_OCR_URL")
CHANDRA_API_KEY = os.getenv("CHANDRA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

OUTPUT_OCR_DIR = Path("outputs/ocr_json")
OUTPUT_FINAL_DIR = Path("outputs/final_json")

NOISE_RE = re.compile(
    r"(?:^TOPIC\s*$|^DATE\s*$|^PAGE\s*NO|Teacher'?s?\s*Signature|^\d{1,3}$)",
    re.IGNORECASE,
)


def require_env(name: str, value: str | None) -> str:
    if not value:
        raise RuntimeError(f"Missing {name}. Add it to .env before running the pipeline.")
    return value


def require_url(name: str, value: str | None) -> str:
    url = require_env(name, value).strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(f"{name} must be a full URL, for example https://api.example.com/ocr.")
    if "example" in parsed.netloc or "your-" in url:
        raise RuntimeError(f"{name} is still a placeholder. Replace it with your real Chandra OCR endpoint.")
    return url


def preprocess_pdf(file_path: str, dpi: int = 300) -> bytes:
    print(f"Preprocessing {file_path}...")
    if convert_from_path is None:
        print("pdf2image is not installed, using original PDF.")
        return Path(file_path).read_bytes()

    try:
        images = convert_from_path(file_path, dpi=dpi)
        if images:
            pdf_bytes = io.BytesIO()
            images[0].save(pdf_bytes, format="PDF", save_all=True, append_images=images[1:])
            pdf_bytes.seek(0)
            print("Converted to image-based PDF")
            return pdf_bytes.read()
    except Exception as e:
        print(f"Preprocessing failed, using original PDF: {e}")

    return Path(file_path).read_bytes()


def extract_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(extract_text(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "markdown", "raw_text", "content", "value"):
            if key in value:
                text = extract_text(value[key])
                if text.strip():
                    return text
        return "\n".join(extract_text(item) for item in value.values())
    return str(value)


def parse_chandra_response(payload) -> list[dict]:
    if isinstance(payload, dict):
        for key in ("pages", "data", "result", "results", "documents"):
            if key in payload:
                return parse_chandra_response(payload[key])
        text = extract_text(payload)
        return [{"page_number": 1, "text": clean_ocr_lines(text)}] if text.strip() else []

    if isinstance(payload, list):
        pages = []
        for index, item in enumerate(payload, start=1):
            text = extract_text(item)
            page_number = item.get("page_number") or item.get("page") or index if isinstance(item, dict) else index
            pages.append({"page_number": int(page_number), "text": clean_ocr_lines(text)})
        return pages

    text = extract_text(payload)
    return [{"page_number": 1, "text": clean_ocr_lines(text)}] if text.strip() else []


def clean_ocr_lines(text: str) -> list[str]:
    lines = []
    seen_on_page = set()
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or NOISE_RE.search(line):
            continue
        if line not in seen_on_page:
            seen_on_page.add(line)
            lines.append(line)
    return lines


def run_ocr(file_content: bytes, file_name: str) -> dict:
    url = require_url("CHANDRA_OCR_URL", CHANDRA_OCR_URL)
    api_key = require_env("CHANDRA_API_KEY", CHANDRA_API_KEY)

    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": (file_name, file_content, "application/pdf")}
    data = {"output_format": "json"}

    print("Running Chandra OCR...")
    try:
        with httpx.Client(timeout=300) as client:
            response = client.post(url, headers=headers, files=files, data=data)
    except httpx.RequestError as e:
        raise RuntimeError(
            "Could not reach Chandra OCR. Check CHANDRA_OCR_URL in .env; "
            f"the configured host could not be contacted. Details: {e}"
        ) from e

    if response.status_code >= 400:
        raise RuntimeError(f"Chandra OCR error {response.status_code}: {response.text[:500]}")

    try:
        payload = response.json()
    except json.JSONDecodeError:
        payload = response.text

    pages = parse_chandra_response(payload)
    if not pages:
        raise RuntimeError("Chandra OCR returned no readable pages.")

    return {"total_pages": len(pages), "pages": pages}


def groq_json(system_prompt: str, user_payload: dict, temperature: float = 0.0) -> dict:
    api_key = require_env("GROQ_API_KEY", GROQ_API_KEY)
    body = {
        "model": GROQ_MODEL,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        with httpx.Client(timeout=180) as client:
            response = client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body)
    except httpx.RequestError as e:
        raise RuntimeError(
            "Could not reach Groq. Check your internet connection and GROQ_API_KEY. "
            f"Details: {e}"
        ) from e

    if response.status_code >= 400:
        raise RuntimeError(f"Groq error {response.status_code}: {response.text[:500]}")

    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def pages_for_prompt(pages: list[dict]) -> list[dict]:
    return [
        {
            "page_number": page["page_number"],
            "text": "\n".join(page["text"]),
        }
        for page in pages
    ]


def extract_official_questions_with_groq(pages: list[dict]) -> list[dict]:
    system_prompt = """
You extract the printed question paper from OCR pages for exam revaluation.
Return only JSON with key "questions".
Each question must have:
- question_id: stable exam id like Q1, Q1.a, Q1.i, A1(i), B2. Preserve section labels if printed.
- question: exact printed question text, cleaned only for obvious OCR line breaks.
- page_number.
Rules:
- Follow the question paper order.
- If a parent question has sub-questions, create one row per sub-question.
- Do not include student answers as questions.
- Do not invent missing questions.
"""
    payload = groq_json(system_prompt, {"pages": pages_for_prompt(pages)})
    questions = payload.get("questions", [])
    if not questions:
        raise RuntimeError("Groq could not identify official questions from the OCR text.")
    return normalize_questions(questions)


def normalize_questions(questions: list[dict]) -> list[dict]:
    normalized = []
    seen = set()

    for index, question in enumerate(questions, start=1):
        qid = str(question.get("question_id") or question.get("id") or f"Q{index}").strip()
        qtext = re.sub(r"\s+", " ", str(question.get("question") or question.get("text") or "")).strip()
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
        normalized.append(
            {
                "question_id": qid,
                "question": qtext,
                "page_number": question.get("page_number"),
            }
        )

    if not normalized:
        raise RuntimeError("Groq returned questions, but none had usable IDs and text.")

    return normalized


def flatten_answer_lines(pages: list[dict], question_page_numbers: set[int]) -> list[dict]:
    lines = []
    line_id = 1
    answer_pages = [page for page in pages if page["page_number"] not in question_page_numbers]
    if not answer_pages:
        answer_pages = pages

    for page in answer_pages:
        for text in page["text"]:
            clean = re.sub(r"\s+", " ", text).strip()
            if not clean or NOISE_RE.search(clean):
                continue
            lines.append({"line_id": line_id, "page_number": page["page_number"], "text": clean})
            line_id += 1
    return lines


def map_answer_spans_with_groq(questions: list[dict], answer_lines: list[dict]) -> list[dict]:
    system_prompt = """
You map handwritten/student answers to official exam questions.
Return only JSON with key "answer_spans".
Each span must have:
- question_id: exactly one of the provided question IDs.
- start_line: first OCR line of that answer, not the printed question paper.
- end_line: last OCR line of that answer.
- confidence: number from 0 to 1.
- notes: short reason.
Rules:
- Use the official question order as the backbone.
- If the student writes sub-question labels like Q1.a, 1(a), A1(i), map to that exact provided ID.
- If an answer starts with a copied question, start_line may be that copied-question line.
- Do not create spans for unanswered questions.
- Do not overlap spans.
- Prefer raw line positions over rewriting text.
"""
    payload = groq_json(
        system_prompt,
        {
            "official_questions": questions,
            "answer_lines": answer_lines,
        },
    )
    return payload.get("answer_spans", [])


def slice_answers(questions: list[dict], answer_lines: list[dict], spans: list[dict]) -> list[dict]:
    line_by_id = {line["line_id"]: line for line in answer_lines}
    question_by_id = {q["question_id"]: q for q in questions}
    qa_pairs = []

    for span in sorted(spans, key=lambda item: item.get("start_line", 10**9)):
        qid = span.get("question_id")
        if qid not in question_by_id:
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

        qa_pairs.append(
            {
                "question_id": qid,
                "question": question_by_id[qid]["question"],
                "answer": answer,
                "mapping_confidence": span.get("confidence"),
            }
        )

    return qa_pairs


def build_final_json(qa_pairs: list[dict]) -> dict:
    return {"total_qa_pairs": len(qa_pairs), "qa_pairs": qa_pairs}


def process_pdf(pdf_path):
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise RuntimeError("PDF not found")

    OUTPUT_OCR_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FINAL_DIR.mkdir(parents=True, exist_ok=True)

    pdf_bytes = preprocess_pdf(str(pdf_file))
    ocr_json = run_ocr(pdf_bytes, pdf_file.name)

    ocr_output_path = OUTPUT_OCR_DIR / f"{pdf_file.stem}_ocr.json"
    with open(ocr_output_path, "w", encoding="utf-8") as f:
        json.dump(ocr_json, f, ensure_ascii=False, indent=4)

    pages = ocr_json["pages"]
    questions = extract_official_questions_with_groq(pages)
    question_page_numbers = {
        int(q["page_number"])
        for q in questions
        if str(q.get("page_number") or "").isdigit()
    }
    answer_lines = flatten_answer_lines(pages, question_page_numbers)
    spans = map_answer_spans_with_groq(questions, answer_lines)
    qa_pairs = slice_answers(questions, answer_lines, spans)

    final_json = build_final_json(qa_pairs)
    final_output_path = OUTPUT_FINAL_DIR / f"{pdf_file.stem}_final.json"
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)

    return str(final_output_path)
