# Assignment Booklet → Structured Q&A Pipeline

Turns a scanned, handwritten (Hindi/English) university assignment or exam
booklet into clean `{question, answer}` pairs — no manual copy-paste, no
hardcoded assumptions about layout, subject, or language.

## Setup

```bash
pip install -r requirements.txt
export DATALAB_API_KEY="your_datalab_key"   # https://www.datalab.to/app/keys
export GROQ_API_KEY="your_groq_key"         # https://console.groq.com/keys
streamlit run app.py
```

## Architecture

```
PDF upload
   │
   ▼
Stage 1 — ocr_stage.py        Datalab Chandra /convert (mode=accurate, paginate=true)
   │                          ONE HTTP submit + poll for the whole PDF.
   ▼
Stage 2 — classify_stage.py   (a) classify_pages(): batched Groq calls (small/fast
   │                              model, N/batch_size round trips, run concurrently)
   │                          (b) extract_questions(): ONE call on just the
   │                              question-paper pages (bigger model), sub-parts
   │                              split into separate entries (1.i, 1.ii, ...)
   ▼
Stage 3 — mapping_stage.py    One Groq call PER question, searching only forward
   │                          from where the previous answer ended (cursor window).
   │                          LLM returns {start_line, end_line} only — never text.
   │                          Python slices the raw OCR lines; a slice equality
   │                          check ("integrity check") verifies zero drift.
   ▼
pipeline.py                   Wires the three stages, serializes ocr.json + qa.json
   │
   ▼
app.py                        Streamlit UI: upload, live status log, JSON downloads
```

## Files

| File | Responsibility |
|---|---|
| `config.py` | Every tunable (models, batch size, timeouts) in one place |
| `utils.py` | Retry/backoff, disk cache, line-numbering helpers |
| `ocr_stage.py` | Stage 1: Datalab Chandra OCR |
| `classify_stage.py` | Stage 2: page classification + question extraction |
| `mapping_stage.py` | Stage 3: sequential answer mapping + integrity check |
| `pipeline.py` | Orchestrator + output serialization |
| `app.py` | Streamlit frontend |

## Why it's fast / token-efficient

1. **OCR is one API call, not one-per-page.** Datalab's `/convert` handles
   the whole PDF server-side with `paginate=true`; we split the returned
   markdown on its page-delimiter marker locally (regex, free).
2. **Classification is batched.** Pages are grouped (`CLASSIFY_BATCH_PAGES`,
   default 8) into single calls instead of one call per page, run
   concurrently with a thread pool. This turns an O(pages) call count into
   O(pages / batch_size), with a cheap model (`llama-3.1-8b-instant`) since
   the task is simple.
3. **Question extraction only reads question-paper pages.** Admin/cover
   and answer pages are filtered out by Stage 2a before Stage 2b ever sees
   them, so the bigger model gets a small, focused prompt.
4. **Sequential mapping uses a shrinking search window, not a growing one.**
   Because answers appear in the same order as questions, each search only
   scans from a few lines before where the previous answer ended onward —
   not the entire answer text every time. On an N-question, L-line booklet
   this changes total scanned tokens from ~N×L (fixed full-doc search) down
   to roughly L + a small overlap per question. It also structurally rules
   out matching a later question to an earlier, already-consumed answer.
5. **Everything is disk-cached by content hash** (`utils.cache_get/set`),
   keyed on file bytes / page text / prompt content. Re-running the same
   PDF (e.g. a Streamlit rerun, or resuming after Stage 3 fails halfway)
   costs nothing for the parts already done.

## Non-negotiables enforced in code

- **Raw answers only**: `mapping_stage._verify_slice` re-derives the answer
  from `lines[start:end+1]` and compares byte-for-byte against what's
  stored; the LLM's job is exclusively picking `start_line`/`end_line`.
- **Sub-questions never merged**: `extract_questions`'s system prompt
  requires each sub-part to get its own `id` (`"3.i"`, `"3.ii"`, ...), and
  nothing downstream re-merges by base question number.
- **Traceability**: every `QAPair` carries `source_pages` (derived from the
  line→page map built in Stage 3), plus the exact `start_line`/`end_line`
  used, so any answer can be manually checked against the scan.
- **Generalization**: no subject/university-specific strings anywhere in
  the prompts or code — labels, schemas, and prompts operate purely on
  page role (question paper / admin / answer) and line position.

## Output shape (`*_qa_pairs.json`)

```json
[
  {
    "question_id": "1.ii",
    "question": "1.ii. Discuss the theme of X. (10)",
    "answer": "raw OCR text, verbatim, question restatement stripped",
    "found": true,
    "confidence": 0.92,
    "source_pages": [7, 8],
    "start_line": 142,
    "end_line": 168,
    "integrity_ok": true
  }
]
```

## Tuning knobs (env vars, all optional)

- `DATALAB_MODE` (`fast`/`balanced`/`accurate`, default `accurate` — keep
  `accurate` for handwriting)
- `GROQ_CLASSIFY_MODEL`, `GROQ_EXTRACT_MODEL`, `GROQ_MAP_MODEL`
- `CLASSIFY_BATCH_PAGES` (default 8)
- `MAX_WORKERS` (default 4)
- `MAPPING_LOOKBACK_LINES` (default 3)
- `PIPELINE_CACHE_ENABLED` (`1`/`0`, default on)
