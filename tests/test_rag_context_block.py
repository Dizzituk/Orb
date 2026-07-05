# Purpose: Lane C tests — build_codebase_context() non-terminal contract.
# Called-by: pytest
# Depends-on: app.rag.answerer, app.rag._answerer_context_block
# Last-renovated: 2026-07-01
# tests/test_rag_context_block.py
"""
Contract tests for build_codebase_context (Lane C, 2026-07-01).

The contract (Lane A wires a guarded call against this sight unseen):
    async def build_codebase_context(question: str, db) -> str
    - ≤3000 characters, plain text, [CODEBASE_CONTEXT]…[/CODEBASE_CONTEXT]
    - "" when retrieval finds nothing relevant
    - NEVER raises (broken db, broken search, broken import — all "")
"""

import asyncio
import inspect
import sys
import types
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest

import app.rag.answerer as answerer_mod
from app.rag.answerer import build_codebase_context
from app.rag._answerer_context_block import MAX_CONTEXT_BLOCK_CHARS


def _run(coro):
    return asyncio.run(coro)


def _fake_chunk(i: int):
    return types.SimpleNamespace(
        file_path=f"app/some/very/long/module/path/segment_{i}/handler_file_{i}.py",
        chunk_name=f"handle_case_number_{i}",
        chunk_type="function",
        signature=f"def handle_case_number_{i}(db, payload, retries=3) -> dict",
        docstring=(
            f"Handles case {i} of the synthetic workload. " * 6
        ),
        start_line=10 * i + 1,
        end_line=10 * i + 9,
    )


class _BrokenDB:
    """Session stand-in whose every query explodes."""

    def query(self, *args, **kwargs):
        raise RuntimeError("db exploded")


# ---------------------------------------------------------------------------
# Signature contract
# ---------------------------------------------------------------------------

def test_contract_signature_exact():
    assert inspect.iscoroutinefunction(build_codebase_context)
    sig = inspect.signature(build_codebase_context)
    assert list(sig.parameters) == ["question", "db"]
    assert sig.parameters["question"].annotation is str
    assert sig.return_annotation is str


# ---------------------------------------------------------------------------
# Empty / failure paths — always "" and never a raise
# ---------------------------------------------------------------------------

def test_returns_empty_string_on_no_results(monkeypatch):
    monkeypatch.setattr(
        answerer_mod, "_search_chunks_hybrid", lambda db, q, use_embeddings=True: ([], "none")
    )
    assert _run(build_codebase_context("anything at all", db=None)) == ""


def test_never_raises_on_broken_db():
    result = _run(build_codebase_context("how does the router work?", _BrokenDB()))
    assert result == ""


def test_never_raises_when_search_itself_raises(monkeypatch):
    def _boom(db, q, use_embeddings=True):
        raise RuntimeError("search exploded")

    monkeypatch.setattr(answerer_mod, "_search_chunks_hybrid", _boom)
    assert _run(build_codebase_context("q", db=None)) == ""


def test_never_raises_when_leaf_module_is_broken(monkeypatch):
    # Simulate the context-block module failing to import at call time.
    monkeypatch.setitem(sys.modules, "app.rag._answerer_context_block", None)
    assert _run(build_codebase_context("q", db=None)) == ""


def test_never_raises_on_garbage_question(monkeypatch):
    monkeypatch.setattr(
        answerer_mod, "_search_chunks_hybrid", lambda db, q, use_embeddings=True: ([], "none")
    )
    assert _run(build_codebase_context("", db=None)) == ""


# ---------------------------------------------------------------------------
# Happy path — well-formed, capped block
# ---------------------------------------------------------------------------

def test_block_is_well_formed_and_grounded(monkeypatch):
    chunks = [_fake_chunk(i) for i in range(3)]
    monkeypatch.setattr(
        answerer_mod, "_search_chunks_hybrid",
        lambda db, q, use_embeddings=True: (chunks, "semantic"),
    )
    block = _run(build_codebase_context("how are cases handled?", db=None))
    assert block.startswith("[CODEBASE_CONTEXT]")
    assert block.rstrip().endswith("[/CODEBASE_CONTEXT]")
    assert chunks[0].file_path in block
    assert "handle_case_number_0" in block


def test_block_never_exceeds_3000_chars(monkeypatch):
    # 200 chunks with long paths/docstrings — raw evidence far exceeds the cap
    chunks = [_fake_chunk(i) for i in range(200)]
    monkeypatch.setattr(
        answerer_mod, "_search_chunks_hybrid",
        lambda db, q, use_embeddings=True: (chunks, "semantic"),
    )
    block = _run(build_codebase_context("summarise every handler", db=None))
    assert MAX_CONTEXT_BLOCK_CHARS == 3000  # the A↔C contract number
    assert 0 < len(block) <= 3000
    assert block.rstrip().endswith("[/CODEBASE_CONTEXT]")


def test_block_is_plain_text_str(monkeypatch):
    chunks = [_fake_chunk(1)]
    monkeypatch.setattr(
        answerer_mod, "_search_chunks_hybrid",
        lambda db, q, use_embeddings=True: (chunks, "keyword"),
    )
    block = _run(build_codebase_context("one chunk", db=None))
    assert isinstance(block, str)


# ---------------------------------------------------------------------------
# The terminal Q&A entry points still exist unchanged (addition, not a swap)
# ---------------------------------------------------------------------------

def test_terminal_entry_points_still_exist():
    assert hasattr(answerer_mod, "ask_architecture_async")
    assert hasattr(answerer_mod, "ask_architecture")
    assert inspect.iscoroutinefunction(answerer_mod.ask_architecture_async)
