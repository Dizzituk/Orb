# FILE: app/rag/_answerer_context_block.py
# Purpose: Non-terminal codebase context block for prompt injection (Lane C contract).
# Called-by: app.rag.answerer (build_codebase_context)
# Depends-on: app.rag.answerer (search internals), app.rag._answerer_text_search
# Last-renovated: 2026-07-01
"""
Non-terminal codebase context block for prompt injection.

Runs the answerer's grounded retrieval but, instead of producing a
user-facing answer, formats the evidence (relevant files, key findings,
short snippets) as a delimited context block that the conversational
layer injects into ITS prompt and speaks from in its own voice.

This exists because the RAG answerer used to be a TERMINAL reply path:
its raw grounded output (file paths, structured findings, another
model's voice) streamed straight to the user. This module is the
context-supplier alternative.

Contract (Lane A wires a guarded call against this, sight unseen):
    - ≤3000 characters, plain text
    - delimited [CODEBASE_CONTEXT] … [/CODEBASE_CONTEXT]
    - "" when retrieval finds nothing relevant
    - never raises

No LLM is called here — the block is assembled deterministically from
retrieval output, so there is no model to configure and nothing to
hardcode.

v1.0 (2026-07-01): Lane C — non-terminal context contract.
"""

import logging

logger = logging.getLogger(__name__)

# Hard cap from the Lane A↔C contract — not tunable via env on purpose:
# Lane A's prompt budget assumes it.
MAX_CONTEXT_BLOCK_CHARS = 3000

_HEADER = "[CODEBASE_CONTEXT]"
_FOOTER = "[/CODEBASE_CONTEXT]"

_MAX_FILES_LISTED = 10
_DOCSTRING_SNIPPET_CHARS = 100
_TEXT_SEARCH_TOKEN_BUDGET = 300  # ~1200 chars, only for count/find questions


async def build_context_block(question: str, db) -> str:
    """
    Build the [CODEBASE_CONTEXT] block for a question. Never raises.

    Args:
        question: The user's message (or a distilled question from it).
        db:       SQLAlchemy session for the architecture index.

    Returns:
        Delimited context block ≤3000 chars, or "" on no results/any error.
    """
    try:
        return _build(question, db)
    except Exception as e:
        logger.warning(
            "[rag:context_block] Build failed (returning empty): %s", e
        )
        return ""


def _build(question: str, db) -> str:
    # Lazy import — answerer.py imports this module lazily too, so
    # neither direction can create an import cycle at module load.
    from app.rag.answerer import _search_chunks_hybrid

    chunks, search_mode = _search_chunks_hybrid(db, question, use_embeddings=True)
    if not chunks:
        logger.info(
            "[rag:context_block] No relevant chunks for: %.60s", question
        )
        return ""

    lines = [
        _HEADER,
        f"Grounded code evidence for this question (search mode: {search_mode}).",
        "",
        "Relevant files:",
    ]

    # Deduped, order-preserving file list (chunks arrive ranked)
    seen_files: list = []
    for chunk in chunks:
        fp = getattr(chunk, "file_path", "") or ""
        if fp and fp not in seen_files:
            seen_files.append(fp)
    for fp in seen_files[:_MAX_FILES_LISTED]:
        lines.append(f"  - {fp}")
    if len(seen_files) > _MAX_FILES_LISTED:
        lines.append(f"  … and {len(seen_files) - _MAX_FILES_LISTED} more files")

    lines.append("")
    lines.append("Key findings:")

    budget = MAX_CONTEXT_BLOCK_CHARS - len(_FOOTER) - 1
    used = sum(len(line) + 1 for line in lines)

    for chunk in chunks:
        entry = _format_chunk(chunk)
        if not entry:
            continue
        if used + len(entry) + 1 > budget:
            break
        lines.append(entry)
        used += len(entry) + 1

    # For count/find-style questions, add the text-search evidence too
    text_section = _text_search_section(question)
    if text_section and used + len(text_section) + 2 <= budget:
        lines.append("")
        lines.append(text_section)

    lines.append(_FOOTER)
    block = "\n".join(lines)

    # Belt and braces — the cap is a contract, so enforce it absolutely
    if len(block) > MAX_CONTEXT_BLOCK_CHARS:
        cut = MAX_CONTEXT_BLOCK_CHARS - len(_FOOTER) - 1
        block = block[:cut].rstrip() + "\n" + _FOOTER

    logger.info(
        "[rag:context_block] Built block: %d chunks, %d files, %d chars",
        len(chunks), len(seen_files), len(block),
    )
    return block


def _format_chunk(chunk) -> str:
    """One evidence line per chunk: path :: kind `name` (Lstart) — snippet."""
    fp = getattr(chunk, "file_path", "") or ""
    if not fp:
        return ""

    parts = [f"  - {fp}"]

    name = getattr(chunk, "chunk_name", "") or ""
    kind = getattr(chunk, "chunk_type", "") or ""
    if name:
        parts.append(f" :: {kind} `{name}`" if kind else f" :: `{name}`")

    start = getattr(chunk, "start_line", None)
    if start:
        parts.append(f" (L{start})")

    signature = getattr(chunk, "signature", "") or ""
    docstring = (getattr(chunk, "docstring", "") or "").strip()
    snippet = signature or docstring
    if snippet:
        snippet = " ".join(snippet.split())[:_DOCSTRING_SNIPPET_CHARS]
        parts.append(f" — {snippet}")

    return "".join(parts)


def _text_search_section(question: str) -> str:
    """Text-search evidence for count/find questions; "" otherwise or on error."""
    try:
        from app.rag._answerer_text_search import (
            extract_search_term,
            search_codebase_text,
        )

        if not extract_search_term(question):
            return ""
        return search_codebase_text(
            question, token_budget=_TEXT_SEARCH_TOKEN_BUDGET
        )
    except Exception as e:
        logger.warning("[rag:context_block] Text search failed: %s", e)
        return ""
