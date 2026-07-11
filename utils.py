"""
Shared, dependency-light utilities used across all three stages.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from typing import Any, Callable, TypeVar

from config import settings

T = TypeVar("T")


# --------------------------------------------------------------------------
# Retry / backoff (handles Groq 429s and Datalab transient errors without
# pulling in an extra dependency like tenacity)
# --------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_retries: int = settings.GROQ_MAX_RETRIES,
    base_delay: float = 1.0,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except retryable_exceptions as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == max_retries:
                break
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            if on_retry:
                on_retry(attempt, exc)
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


# --------------------------------------------------------------------------
# On-disk caching keyed by content hash. This is what makes re-runs (e.g.
# a Streamlit rerun after a UI tweak, or a retry after a crash on stage 3)
# essentially free instead of re-billing OCR/LLM calls.
# --------------------------------------------------------------------------
def _cache_path(namespace: str, key: str) -> str:
    os.makedirs(os.path.join(settings.CACHE_DIR, namespace), exist_ok=True)
    return os.path.join(settings.CACHE_DIR, namespace, f"{key}.json")


def cache_get(namespace: str, key: str) -> Any | None:
    if not settings.CACHE_ENABLED:
        return None
    path = _cache_path(namespace, key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def cache_set(namespace: str, key: str, value: Any) -> None:
    if not settings.CACHE_ENABLED:
        return
    path = _cache_path(namespace, key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False)


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:24]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


# --------------------------------------------------------------------------
# Line numbering — the backbone of the "LLM only returns line numbers"
# integrity guarantee. Every stage-3 search operates on this array, never
# on raw prose, so there is nothing for the model to paraphrase.
# --------------------------------------------------------------------------
def to_numbered_lines(text: str) -> list[str]:
    """Split into lines, preserving blank lines (they matter for slicing)."""
    return text.split("\n")


def render_numbered_block(lines: list[str], start: int, end: int) -> str:
    """
    Render lines[start:end] (end exclusive) as 'N: content' for the prompt.
    Using explicit numeric prefixes is what lets the LLM reply with a bare
    integer range instead of copying text — this is the main token saver
    in stage 3, since the model never has to echo back answer content.
    """
    out = []
    for i in range(start, min(end, len(lines))):
        out.append(f"{i}: {lines[i]}")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Rough token estimate (no tokenizer dependency). Good enough to decide
# batch sizes / whether to shrink a window before we hit a provider limit.
# ~4 chars/token is the standard rule of thumb for English; Hindi/Devanagari
# text runs closer to ~2-3 chars/token, so we bias conservatively.
# --------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3)


def status(cb: Callable[[str], None] | None, message: str) -> None:
    """Fire-and-forget status callback used by the Streamlit frontend."""
    if cb:
        cb(message)
