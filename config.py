"""
Central configuration. Everything tunable lives here so the rest of the
codebase never hardcodes a model name, batch size, or URL.
"""
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    # ---------------- Datalab (Chandra OCR) ----------------
    DATALAB_API_KEY: str = os.environ.get("DATALAB_API_KEY", "")
    DATALAB_BASE_URL: str = "https://www.datalab.to/api/v1"
    # "accurate" is worth the extra latency for messy handwriting; use
    # "balanced" if you need to cut cost/time and scans are clean.
    DATALAB_MODE: str = os.environ.get("DATALAB_MODE", "accurate")
    DATALAB_POLL_INTERVAL: float = float(os.environ.get("DATALAB_POLL_INTERVAL", 2.0))
    DATALAB_POLL_TIMEOUT_S: int = int(os.environ.get("DATALAB_POLL_TIMEOUT_S", 900))

    # ---------------- Groq (LLM stages) ----------------
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
    # Small/fast model for the cheap, high-volume task: page classification.
    GROQ_CLASSIFY_MODEL: str = os.environ.get("GROQ_CLASSIFY_MODEL", "llama-3.1-8b-instant")
    # Bigger model for tasks where a mistake is expensive: pulling the
    # canonical question list, and locating answer boundaries.
    GROQ_EXTRACT_MODEL: str = os.environ.get("GROQ_EXTRACT_MODEL", "llama-3.3-70b-versatile")
    GROQ_MAP_MODEL: str = os.environ.get("GROQ_MAP_MODEL", "llama-3.3-70b-versatile")
    GROQ_TEMPERATURE: float = 0.0
    GROQ_MAX_RETRIES: int = 4

    # ---------------- Optimization knobs ----------------
    # How many OCR'd pages get bundled into a single classification call.
    # Bigger = fewer round trips (less fixed per-call overhead) but risks
    # hitting context limits / harder-to-parse batches. 6-10 is a sweet
    # spot for a page that's ~150-400 tokens of markdown.
    CLASSIFY_BATCH_PAGES: int = int(os.environ.get("CLASSIFY_BATCH_PAGES", 8))
    # Parallel workers for classification batches (independent of each
    # other, safe to run concurrently) and for OCR polling of multi-file runs.
    MAX_WORKERS: int = int(os.environ.get("MAX_WORKERS", 4))
    # Sequential mapping (Stage 3) is intentionally NOT parallelized across
    # questions for accuracy reasons (each search benefits from knowing
    # where the previous answer ended). This is where we cut cost instead
    # by shrinking the search window every step (see mapping_stage.py).
    MAPPING_LOOKBACK_LINES: int = int(os.environ.get("MAPPING_LOOKBACK_LINES", 3))

    CACHE_DIR: str = os.environ.get("PIPELINE_CACHE_DIR", ".pipeline_cache")
    CACHE_ENABLED: bool = os.environ.get("PIPELINE_CACHE_ENABLED", "1") != "0"


settings = Settings()
