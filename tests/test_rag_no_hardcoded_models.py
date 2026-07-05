# Purpose: Lane C regression guard — no hardcoded model IDs under app/rag/.
# Called-by: pytest
# Depends-on: stdlib only (scans source text, imports nothing from app)
# Last-renovated: 2026-07-01
# tests/test_rag_no_hardcoded_models.py
"""
Regression guard: no literal model ID strings under app/rag/.

Why: until 2026-07-01 the RAG answerer hardcoded its tier→model map
(gpt-4.1-mini / claude-sonnet-4-5-20250929 / claude-opus-4-5). All three
were stale, superseded models by the time they were removed — hardcoded
model IDs rot. Every model choice must resolve from .env.

Detection: any *quoted string literal* beginning with a model-ID prefix
(gpt-, claude-, gemini-). Mentions inside docstrings/comments (history
notes, provider-inference heuristics like startswith("claude")) don't
start a quoted string with the prefix, so they pass.
"""

import re
from pathlib import Path

RAG_ROOT = Path(__file__).parent.parent / "app" / "rag"

# A quote immediately followed by a model-ID prefix = a literal model string.
_MODEL_LITERAL = re.compile(r"""["'](?:gpt-|claude-|gemini-)""")

# Whitelist: relative posix path → allowed regex-match substrings, each entry
# must carry a justification comment. Deliberately EMPTY — everything under
# app/rag/ resolves models from .env (or from the embeddings provider's own
# constant, which lives outside app/rag/). Add entries only with a reason.
WHITELIST: dict = {}


def _iter_rag_sources():
    for path in sorted(RAG_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def test_rag_root_exists():
    assert RAG_ROOT.is_dir(), f"app/rag not found at {RAG_ROOT}"


def test_no_hardcoded_model_literals_under_app_rag():
    offenders = []
    for path in _iter_rag_sources():
        rel = path.relative_to(RAG_ROOT.parent.parent).as_posix()
        source = path.read_text(encoding="utf-8", errors="replace")
        allowed = WHITELIST.get(rel, ())
        for lineno, line in enumerate(source.splitlines(), 1):
            m = _MODEL_LITERAL.search(line)
            if not m:
                continue
            if any(a in line for a in allowed):
                continue
            offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Hardcoded model ID literal(s) found under app/rag/ — model IDs "
        "must resolve from .env, never source:\n  " + "\n  ".join(offenders)
    )


def test_known_stale_models_are_gone():
    """The exact stale IDs the 2026-07-01 de-hardcode removed never return
    as quoted literals (they may appear in docstrings as history notes)."""
    stale = ("\"gpt-4.1-mini\"", "'gpt-4.1-mini'",
             "\"claude-sonnet-4-5-20250929\"", "'claude-sonnet-4-5-20250929'",
             "\"claude-opus-4-5\"", "'claude-opus-4-5'")
    offenders = []
    for path in _iter_rag_sources():
        source = path.read_text(encoding="utf-8", errors="replace")
        for s in stale:
            if s in source:
                rel = path.relative_to(RAG_ROOT.parent.parent).as_posix()
                offenders.append(f"{rel}: {s}")
    assert not offenders, "Stale hardcoded model returned:\n  " + "\n  ".join(offenders)
