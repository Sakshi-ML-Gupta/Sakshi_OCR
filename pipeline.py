"""
pipeline.py
============
OCR + Question-Answer mapping engine for the Exam Evaluator project.

Uses Google's Gemini API FREE TIER (no paid API required) as the OCR/mapping
engine, via the current unified `google-genai` SDK.

Why Gemini free tier specifically:
- Free-tier vision models handle cursive/mixed-script handwriting (e.g.
  Devanagari + English mixed on the same page) far better than classical
  OCR engines (Tesseract/EasyOCR), which is what real exam scripts need.
- Genuinely free (no credit card) via a Google AI Studio API key.

Free tier constraint this file is explicitly built around:
- The binding limits on the free tier are REQUESTS PER MINUTE and
  REQUESTS PER DAY, not tokens (token budget is generous, ~250k TPM).
  So the design goal is "minimize request COUNT", not "minimize tokens".
  -> Multiple page images are BATCHED into a single API call.
  -> A hard rate limiter paces every call so we never exceed the
     configured requests-per-minute budget (default conservative: 8 RPM,
     safely under every current free-tier model's floor).
  -> Retries back off hard specifically on 429 / RESOURCE_EXHAUSTED.

Design goals (per spec):
- Handles handwritten AND printed content, in ANY mix of languages/scripts,
  with no hard-coded language assumption -> naturally adaptive.
- STRICT verbatim transcription: no spelling fixes, no grammar fixes, no
  paraphrasing, no summarizing. What is written is what comes out.
- Ignores grader/evaluator red-ink marks, circles, ticks and marks-out-of-X
  annotations -> those are NOT part of the student's answer.
- Works on UNSTRUCTURED pdfs: no fixed page layout is assumed. A page can
  be a cover page, an ID card, a printed question paper, or N handwritten
  answer pages in any order. Page type + language are detected per page.
- Two-stage pipeline:
    Stage 1 (vision, batched+rate-limited)  -> raw verbatim OCR JSON
    Stage 2 (text-only, single call)        -> Q/A pairs extracted *only*
                                                from the Stage-1 verbatim
                                                text (never re-reads the
                                                images, never invents
                                                new wording)
  This keeps stage 2 fast/cheap (one extra request, text tokens only) and
  guarantees the "no modification" requirement because stage 2 is
  instructed to copy spans from stage 1's output rather than compose
  new text.

Only public entry points you need:
    run_ocr_pipeline(pdf_bytes, filename, progress_cb) -> full OCR dict
    extract_qa_pairs(full_ocr_json)                    -> list[{"question","answer"}]
"""

import os
import io
import re
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
from PIL import Image

from google import genai
from google.genai import types

# --------------------------------------------------------------------------
# Config (all overridable via environment variables, no code changes needed)
# --------------------------------------------------------------------------
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")  # try gemini-2.5-flash-lite for higher RPM/RPD
RENDER_DPI = int(os.environ.get("OCR_RENDER_DPI", "220"))         # PDF -> image quality
MAX_IMAGE_EDGE = int(os.environ.get("OCR_MAX_IMAGE_EDGE", "1568"))  # vision-optimal edge
JPEG_QUALITY = int(os.environ.get("OCR_JPEG_QUALITY", "90"))

# Free-tier pacing. Keep RPM conservative by default -- raise it only if
# your AI Studio dashboard confirms your model's tier allows more.
GEMINI_RPM = float(os.environ.get("GEMINI_RPM", "8"))
PAGES_PER_BATCH = int(os.environ.get("OCR_PAGES_PER_BATCH", "4"))  # fewer requests = fewer RPD hits
MAX_WORKERS = int(os.environ.get("OCR_MAX_WORKERS", "3"))          # concurrency on top of the limiter
MAX_RETRIES = int(os.environ.get("OCR_MAX_RETRIES", "5"))
OCR_MAX_TOKENS = int(os.environ.get("OCR_MAX_TOKENS", "8000"))
QA_MAX_TOKENS = int(os.environ.get("QA_MAX_TOKENS", "8000"))

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
OCR_SYSTEM_PROMPT = """You are a forensic OCR transcription engine used inside an exam-evaluation \
pipeline. You will be shown one or more scanned page images from the SAME document, each \
preceded by a text label "PAGE <n>:". Transcribe EACH page independently and EXACTLY as written.

NON-NEGOTIABLE RULES:
1. Transcribe verbatim. Do NOT fix spelling. Do NOT fix grammar. Do NOT rephrase, \
summarize, translate, or "clean up" anything. If a word is misspelled or a sentence \
is grammatically broken in the original, reproduce it exactly as written, mistakes and all.
2. Preserve the original script/language of every line exactly as it appears (e.g. \
Devanagari stays Devanagari, English stays English, mixed-script lines stay mixed). \
Never transliterate or translate.
3. Transcribe ONLY content written by the original author of the page (printed question \
paper text, or the student's own handwritten pen work in blue/black/pencil ink).
4. IGNORE and DO NOT transcribe grader/evaluator annotations: red-ink circles, ticks, \
crosses, underlines/scribbles used for marking, marks-out-of-X numbers, rubber-stamp \
text, or any red-ink handwriting that is clearly an evaluator's mark rather than the \
student's own answer content. Do not mention these marks in the "text" field at all.
5. Preserve paragraph/line breaks and question numbering/labels as they appear \
(e.g. "1.", "Q1", numbered headings, part/section labels) since these are essential for \
downstream question-answer matching. Keep them inline in the transcribed text.
6. If part of a page is illegible, write [illegible] at that spot rather than guessing or \
inventing text. Never fabricate content that is not actually on the page.
7. Do not add commentary, explanations, translations, or corrections anywhere in output.
8. Every page you are shown must produce exactly one object in the output array, in the \
same order the pages were given to you, each tagged with its correct page_number.

CLASSIFY each page as one of:
- "question_paper": printed exam/assignment question paper text
- "answer_sheet": handwritten (or typed) student answer content
- "cover_page": title/cover/enrollment/signature page with no question or answer content
- "id_card": ID card / photo page with no question or answer content
- "other": anything not fitting above (blank page, unrelated content)

OUTPUT FORMAT: respond with ONLY a single JSON array, no markdown fences, no prose, one \
object per page shown, in order:
[
  {
    "page_number": <int, matching the PAGE <n> label>,
    "page_type": "question_paper" | "answer_sheet" | "cover_page" | "id_card" | "other",
    "languages": ["<ISO 639-1 or script name for every language/script present>"],
    "text": "<verbatim transcription of all question/answer content on the page, using \
\\n for line breaks; empty string if page has no question/answer content>"
  },
  ...
]"""

QA_SYSTEM_PROMPT = """You are the question-answer mapping stage of an exam-evaluation pipeline. \
You receive the ALREADY-OCR'd verbatim text of every page of a scanned assignment/answer \
booklet, tagged by page number and page_type. The document is UNSTRUCTURED: pages can be \
in any order, question numbering conventions can vary page to page, and a single question's \
answer may span multiple consecutive answer_sheet pages.

YOUR JOB: reconstruct question-answer pairs.
- Take each distinct question from the question_paper page(s) (verbatim, as already \
transcribed — do not reword it).
- Find the matching answer content in the answer_sheet page(s), matched by question \
number/label. Numbering formats will often differ between the question paper and the \
answer booklet — match them by their logical order and any numeric/label cues, not by \
exact string match.
- If an answer spans multiple pages/blocks for the same question, concatenate them in the \
correct reading order, verbatim, separated by a single newline. Do not merge unrelated \
questions together.
- Copy question and answer text EXACTLY as given to you in the input (character for \
character) — you are only segmenting and pairing existing text, never rewriting, \
correcting, translating, summarizing, or adding to it.
- Ignore cover_page / id_card / other page content entirely — it is never part of a question \
or an answer.
- If a question exists but no matching answer content can be found, include it with \
"answer": "" — do not invent an answer.
- If answer content exists that cannot be confidently matched to any listed question, \
omit it rather than guessing.

OUTPUT FORMAT: respond with ONLY a single JSON array, no markdown fences, no prose:
[
  {"question": "<verbatim question text>", "answer": "<verbatim matched answer text>"},
  ...
]"""


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------
def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/apikey and export it before running the app."
        )
    return genai.Client(api_key=api_key)


# --------------------------------------------------------------------------
# Rate limiter -- keeps us under the free tier's requests-per-minute cap
# regardless of how many worker threads are calling the API.
# --------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, rpm: float):
        self.min_interval = 60.0 / max(0.001, rpm)
        self._lock = threading.Lock()
        self._last_call = 0.0

    def acquire(self):
        with self._lock:
            now = time.time()
            wait = self._last_call + self.min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()


_rate_limiter = RateLimiter(GEMINI_RPM)


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()


def _call_with_retry(fn, *, max_retries: int = MAX_RETRIES):
    """Runs fn() respecting the global rate limiter, retrying with backoff.
    Backs off harder specifically on rate-limit/quota errors."""
    last_err = None
    for attempt in range(max_retries):
        _rate_limiter.acquire()
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - want to retry on any transient error
            last_err = e
            if attempt < max_retries - 1:
                if _is_rate_limit_error(e):
                    time.sleep(min(60, 8 * (attempt + 1)))  # hard backoff on 429/quota
                else:
                    time.sleep(1.5 * (attempt + 1))
    raise last_err


# --------------------------------------------------------------------------
# PDF -> images
# --------------------------------------------------------------------------
def pdf_to_images(pdf_bytes: bytes, dpi: int = RENDER_DPI):
    """Render every page of the PDF to a PIL RGB image at the given DPI."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def _resize_for_vision(img: Image.Image) -> Image.Image:
    """Downscale so the longest edge is MAX_IMAGE_EDGE -- keeps handwriting
    legible while minimizing upload size / latency / token cost."""
    w, h = img.size
    longest = max(w, h)
    if longest <= MAX_IMAGE_EDGE:
        return img
    scale = MAX_IMAGE_EDGE / longest
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def _image_to_part(img: Image.Image) -> types.Part:
    buf = io.BytesIO()
    _resize_for_vision(img).convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


# --------------------------------------------------------------------------
# JSON parsing helpers (models occasionally wrap JSON in fences/prose)
# --------------------------------------------------------------------------
def _strip_fences(text: str) -> str:
    return _JSON_FENCE_RE.sub("", text.strip()).strip()


def _safe_parse_array(text: str) -> list:
    text = _strip_fences(text or "")
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\[.*\]", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return []


def _empty_page(page_num: int, error: str = "") -> dict:
    d = {"page_number": page_num, "page_type": "other", "languages": [], "text": ""}
    if error:
        d["error"] = error
    return d


# --------------------------------------------------------------------------
# Stage 1: batched, rate-limited OCR
# --------------------------------------------------------------------------
def _ocr_batch(client: genai.Client, batch: list) -> list:
    """batch: list of (page_num, PIL.Image). Returns list of page dicts."""
    contents = []
    for page_num, img in batch:
        contents.append(f"PAGE {page_num}:")
        contents.append(_image_to_part(img))
    contents.append(
        f"Transcribe all {len(batch)} page(s) above per your instructions. JSON array only."
    )

    def _do_call():
        resp = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=OCR_SYSTEM_PROMPT,
                response_mime_type="application/json",
                max_output_tokens=OCR_MAX_TOKENS,
            ),
        )
        return resp.text

    try:
        text = _call_with_retry(_do_call)
        parsed = _safe_parse_array(text)
    except Exception as e:
        err = str(e)
        return [_empty_page(pn, error=err) for pn, _ in batch]

    # Map results back by page_number when present, else fall back to order.
    by_num = {}
    for item in parsed:
        if isinstance(item, dict) and "page_number" in item:
            by_num[item["page_number"]] = item

    results = []
    for i, (page_num, _) in enumerate(batch):
        item = by_num.get(page_num)
        if item is None and i < len(parsed) and isinstance(parsed[i], dict):
            item = parsed[i]  # positional fallback if model omitted page_number
        if item is None:
            item = _empty_page(page_num, error="missing from model response")
        else:
            item = dict(item)
            item["page_number"] = page_num
            item.setdefault("page_type", "other")
            item.setdefault("languages", [])
            item.setdefault("text", "")
        results.append(item)
    return results


def _chunk(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def run_ocr_pipeline(pdf_bytes: bytes, filename: str = "document.pdf", progress_cb=None) -> dict:
    """
    Full OCR of every page. Pages are grouped into small batches (fewer
    requests -> respects free-tier RPD) and batches run concurrently, each
    call paced by the global rate limiter (respects free-tier RPM).

    progress_cb: optional callable(done:int, total:int) for UI progress bars.
    Returns the FULL OCR JSON structure (verbatim content for every page).
    """
    client = get_client()
    images = pdf_to_images(pdf_bytes)
    total = len(images)
    numbered = list(enumerate(images, start=1))
    batches = list(_chunk(numbered, PAGES_PER_BATCH))

    results_by_page = {}
    done_pages = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max(1, MAX_WORKERS)) as executor:
        future_to_batch = {executor.submit(_ocr_batch, client, b): b for b in batches}
        for fut in as_completed(future_to_batch):
            batch = future_to_batch[fut]
            page_results = fut.result()
            with lock:
                for pd in page_results:
                    results_by_page[pd["page_number"]] = pd
                done_pages += len(batch)
                if progress_cb:
                    progress_cb(min(done_pages, total), total)

    pages = [results_by_page.get(i, _empty_page(i, error="not processed"))
             for i in range(1, total + 1)]

    return {
        "filename": filename,
        "page_count": total,
        "pages": pages,
    }


# --------------------------------------------------------------------------
# Stage 2: Q-A mapping (single text-only call over Stage-1 output)
# --------------------------------------------------------------------------
def _build_pages_context(full_ocr_json: dict) -> str:
    blocks = []
    for p in full_ocr_json.get("pages", []):
        ptype = p.get("page_type", "other")
        if ptype in ("cover_page", "id_card"):
            continue  # never relevant to Q/A, skip to save tokens
        blocks.append(
            f"=== PAGE {p.get('page_number')} | type={ptype} | "
            f"languages={p.get('languages')} ===\n{p.get('text', '')}"
        )
    return "\n\n".join(blocks)


def extract_qa_pairs(full_ocr_json: dict, client: genai.Client = None) -> list:
    """
    Stage 2: derive [{"question": ..., "answer": ...}, ...] purely from the
    verbatim Stage-1 OCR text (no re-reading of images, no new wording).
    Costs exactly ONE extra request against the free-tier quota.
    """
    client = client or get_client()
    context = _build_pages_context(full_ocr_json)
    if not context.strip():
        return []

    def _do_call():
        resp = client.models.generate_content(
            model=MODEL,
            contents=context,
            config=types.GenerateContentConfig(
                system_instruction=QA_SYSTEM_PROMPT,
                response_mime_type="application/json",
                max_output_tokens=QA_MAX_TOKENS,
            ),
        )
        return resp.text

    text = _call_with_retry(_do_call)
    return _safe_parse_array(text)


# --------------------------------------------------------------------------
# Convenience: run both stages end to end
# --------------------------------------------------------------------------
def run_full_pipeline(pdf_bytes: bytes, filename: str = "document.pdf", progress_cb=None):
    """Returns (full_ocr_json, qa_pairs_list)."""
    full_ocr_json = run_ocr_pipeline(pdf_bytes, filename=filename, progress_cb=progress_cb)
    qa_pairs = extract_qa_pairs(full_ocr_json)
    return full_ocr_json, qa_pairs
