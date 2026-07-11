"""
Stage 1 — OCR (Datalab Chandra, hosted API)

Optimization note: instead of rasterizing the PDF into images ourselves and
firing one OCR call per page, we submit the *whole PDF* to Datalab's
/convert endpoint once, with paginate=True. Datalab handles page splitting
internally and returns a single markdown blob with page-delimiter markers
("\n\n{N}" + 48 dashes + "\n\n"). That means:
  - 1 HTTP submission + a handful of polls, regardless of page count
  - no local rasterization dependency (pypdfium2/pdf2image) to maintain
  - Hindi/English mixed handwriting handled by mode="accurate"
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable

import requests

from config import settings
from utils import cache_get, cache_set, file_hash, retry_with_backoff, status

PAGE_DELIM_RE = re.compile(r"\n\n\{(\d+)\}-{40,}\n\n")


@dataclass
class OcrPage:
    page_number: int  # 0-indexed, matches Datalab's page_range convention
    text: str


@dataclass
class OcrResult:
    pages: list[OcrPage]
    page_count: int
    raw_markdown: str
    parse_quality_score: float | None = None


class DatalabError(RuntimeError):
    pass


def _submit(pdf_bytes: bytes, filename: str) -> str:
    if not settings.DATALAB_API_KEY:
        raise DatalabError("DATALAB_API_KEY is not set.")

    def _do() -> str:
        resp = requests.post(
            f"{settings.DATALAB_BASE_URL}/convert",
            headers={"X-API-Key": settings.DATALAB_API_KEY},
            files={"file": (filename, pdf_bytes, "application/pdf")},
            data={
                "output_format": "markdown",
                "mode": settings.DATALAB_MODE,
                "paginate": "true",
            },
            timeout=120,
        )
        if resp.status_code == 429:
            raise DatalabError("rate_limited")
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", True) and "request_check_url" not in data:
            raise DatalabError(f"Datalab submit failed: {data}")
        return data["request_check_url"]

    return retry_with_backoff(_do, retryable_exceptions=(DatalabError, requests.RequestException))


def _poll(check_url: str, status_cb: Callable[[str], None] | None) -> dict:
    headers = {"X-API-Key": settings.DATALAB_API_KEY}
    deadline = time.time() + settings.DATALAB_POLL_TIMEOUT_S
    attempt = 0
    while time.time() < deadline:
        resp = requests.get(check_url, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "complete":
            if not data.get("success", True):
                raise DatalabError(f"Datalab conversion failed: {data.get('error')}")
            return data
        if data.get("status") == "failed":
            raise DatalabError(f"Datalab conversion failed: {data.get('error')}")
        attempt += 1
        status(status_cb, f"OCR in progress… (poll #{attempt})")
        time.sleep(settings.DATALAB_POLL_INTERVAL)
    raise DatalabError("Datalab polling timed out.")


def _split_pages(markdown: str, page_count_hint: int | None) -> list[OcrPage]:
    """Split Datalab's paginated markdown on its page-delimiter marker."""
    matches = list(PAGE_DELIM_RE.finditer(markdown))
    if not matches:
        # No delimiters found (e.g. single-page doc, or Datalab changed the
        # format) — degrade gracefully to a single page rather than losing
        # the OCR text.
        return [OcrPage(page_number=0, text=markdown.strip())]

    pages: list[OcrPage] = []
    # Text before the first delimiter belongs to page 0.
    first_chunk = markdown[: matches[0].start()].strip()
    if first_chunk:
        pages.append(OcrPage(page_number=0, text=first_chunk))

    for i, m in enumerate(matches):
        page_num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        pages.append(OcrPage(page_number=page_num, text=markdown[start:end].strip()))

    return pages


def run_ocr(
    pdf_bytes: bytes,
    filename: str = "document.pdf",
    status_cb: Callable[[str], None] | None = None,
) -> OcrResult:
    """
    Single entry point for Stage 1. Cached by file content hash so a
    Streamlit rerun on the same upload never re-bills OCR.
    """
    key = file_hash(pdf_bytes)
    cached = cache_get("ocr", key)
    if cached:
        status(status_cb, "OCR cache hit — skipping Datalab call.")
        pages = [OcrPage(**p) for p in cached["pages"]]
        return OcrResult(
            pages=pages,
            page_count=cached["page_count"],
            raw_markdown=cached["raw_markdown"],
            parse_quality_score=cached.get("parse_quality_score"),
        )

    status(status_cb, "Submitting scanned booklet to Chandra OCR…")
    check_url = _submit(pdf_bytes, filename)
    data = _poll(check_url, status_cb)

    markdown = data.get("markdown", "") or ""
    page_count = data.get("page_count", 0)
    pages = _split_pages(markdown, page_count)

    result = {
        "pages": [{"page_number": p.page_number, "text": p.text} for p in pages],
        "page_count": page_count or len(pages),
        "raw_markdown": markdown,
        "parse_quality_score": data.get("parse_quality_score"),
    }
    cache_set("ocr", key, result)

    status(status_cb, f"OCR complete — {len(pages)} pages extracted.")
    return OcrResult(
        pages=pages,
        page_count=result["page_count"],
        raw_markdown=markdown,
        parse_quality_score=result["parse_quality_score"],
    )
