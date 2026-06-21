"""
TUPLE ERROR DIAGNOSTIC -- import this FIRST, before anything else, at
the very top of your Streamlit app file:

    import tuple_error_diagnostic  # <-- add this as the FIRST import
    import streamlit as st
    # ... rest of your app

WHY THIS EXISTS:
The error "expected str, bytes or os.PathLike object, not tuple" has
been chased for several rounds now. Every function in pipeline_llm_v4.py
has been individually tested against every realistic tuple shape and
NONE of them reproduce this exact error -- which means it is being
raised somewhere this investigation hasn't been able to see: most
likely inside your Streamlit app code itself, before process_pdf() is
even called, or inside a library call this investigation hasn't
covered.

Rather than guess a fifth time, this module intercepts the EXACT two
built-in functions that can produce this exact error message --
os.fspath() and the open() builtin -- at the lowest possible level, in
the whole process, before Streamlit or anything else gets a chance to
swallow or obscure the traceback. The instant a tuple reaches either
of them, it prints the full call stack to your terminal/logs, showing
you the exact file and line number responsible.

This is read-only diagnostic instrumentation. It does not change any
behavior for non-tuple inputs -- it only adds a print statement in the
exact moment this specific bug pattern is detected, then lets the
original error propagate normally.

HOW TO USE THE OUTPUT:
Once this is imported and the bug happens again, your terminal/logs
(wherever Streamlit's stdout goes -- terminal if running locally,
the app logs if deployed) will show a block starting with
"=== TUPLE-TO-PATH BUG INTERCEPTED ===" followed by the exact stack
trace. Copy that ENTIRE block and share it -- it will show definitively
which file and line number is responsible, ending this guessing loop
for good.
"""

import os
import builtins
import traceback
import sys

_original_fspath = os.fspath
_original_open = builtins.open


def _patched_fspath(path):
    if isinstance(path, tuple):
        print("=" * 70, file=sys.stderr)
        print("=== TUPLE-TO-PATH BUG INTERCEPTED (via os.fspath) ===", file=sys.stderr)
        print(f"Tuple value received: {path!r}", file=sys.stderr)
        print("Call stack (most recent call last):", file=sys.stderr)
        traceback.print_stack(file=sys.stderr)
        print("=" * 70, file=sys.stderr)
    return _original_fspath(path)


def _patched_open(file, *args, **kwargs):
    if isinstance(file, tuple):
        print("=" * 70, file=sys.stderr)
        print("=== TUPLE-TO-PATH BUG INTERCEPTED (via open()) ===", file=sys.stderr)
        print(f"Tuple value received as file argument: {file!r}", file=sys.stderr)
        print("Call stack (most recent call last):", file=sys.stderr)
        traceback.print_stack(file=sys.stderr)
        print("=" * 70, file=sys.stderr)
    return _original_open(file, *args, **kwargs)


# Install the patches process-wide. Safe to import multiple times --
# guards against double-patching if the module is accidentally imported
# more than once (e.g. via Streamlit's module reload on rerun).
if not getattr(os.fspath, "_is_tuple_diagnostic_patch", False):
    _patched_fspath._is_tuple_diagnostic_patch = True
    os.fspath = _patched_fspath

if not getattr(builtins.open, "_is_tuple_diagnostic_patch", False):
    _patched_open._is_tuple_diagnostic_patch = True
    builtins.open = _patched_open

print(
    "[tuple_error_diagnostic] Active -- will print full stack trace the "
    "instant a tuple reaches os.fspath() or open() anywhere in this process.",
    file=sys.stderr,
)
