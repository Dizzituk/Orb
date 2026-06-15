# FILE: app/rag/retrieval/chat_injection.py
# Purpose: Repo-scan chunks -> conversational memory injection, with staleness labels.
# Called-by: app.llm.routing.memory_injection
# Depends-on: app.rag.retrieval.arch_search, app.rag.models, app.llm.routing.rag_fallback
# Last-renovated: 2026-06-12
"""
Repo-scan chunks -> conversational memory injection.

Before 2026-06-12 the 9,100+ arch_code_chunk embeddings were write-only for
normal chat: the semantic retrieval channel dropped their source_type, the
dedicated arch search filtered on a constant that matched zero rows, and no
injection path existed. This module closes the loop:

  - is_codebase_query(): deterministic gate — topic-tagger code/architecture
    tags OR the rag_fallback architecture-question patterns. Everyday chat
    must NOT be polluted with code chunks.
  - build_repo_context(): searches arch chunks (embedding channel first,
    keyword fallback when embeddings are unavailable), formats the top hits,
    and prefixes an honest staleness header derived from the scan run's
    `source` + scan date. Sandbox-sourced snapshots get an explicit
    "sandbox currently offline — last snapshot dated X" marker when
    192.168.250.2 is unreachable, so ASTRA phrases answers accordingly.

Never raises into the caller — all failures degrade to "no repo context".
"""
from __future__ import annotations

import logging
import os
import socket
import time
from typing import List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Topic-tagger tags that indicate the conversation is about code/architecture
CODE_TAGS = {"code", "architecture", "astra_dev", "python", "testing", "debugging"}

# Injection budget
MAX_CHUNKS = 6
MAX_CONTEXT_CHARS = 4500
MIN_SIMILARITY = 0.45

# Reachability probe cache (per process — "per session" for a single-user app)
_PROBE_TTL_SECONDS = 300
_probe_cache = {"ts": 0.0, "reachable": None}


def _enabled() -> bool:
    return os.getenv("ASTRA_REPO_CHAT_RETRIEVAL", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


# =========================================================================
# Gate
# =========================================================================

def is_codebase_query(message: str, query_tags: Optional[List[str]] = None) -> bool:
    """True when the message is about ASTRA's own code/architecture.

    Two signals, either fires:
      1. topic_tagger tags intersect CODE_TAGS (catches literal code talk)
      2. rag_fallback.is_architecture_query() (catches natural questions
         like "where is retrieval implemented?" that the tagger labels
         'general')
    """
    if query_tags and set(query_tags) & CODE_TAGS:
        return True
    try:
        from app.llm.routing.rag_fallback import is_architecture_query
        return is_architecture_query(message or "")
    except Exception:
        return False


# =========================================================================
# Sandbox reachability (short timeout, cached)
# =========================================================================

def _sandbox_host_port() -> Tuple[str, int]:
    """Resolve the sandbox controller host/port without importing heavy deps."""
    url = None
    try:
        from app.sandbox.client import DEFAULT_SANDBOX_URL, CONTROLLER_ADDR_FILE
        try:
            from pathlib import Path
            p = Path(CONTROLLER_ADDR_FILE)
            if p.exists():
                cached = p.read_text(encoding="utf-8").strip()
                if cached.startswith("http"):
                    url = cached
        except Exception:
            pass
        url = url or DEFAULT_SANDBOX_URL
    except Exception:
        url = "http://192.168.250.2:8765"
    parsed = urlparse(url)
    return parsed.hostname or "192.168.250.2", parsed.port or 8765


def check_sandbox_reachable(timeout_seconds: float = 1.5, force: bool = False) -> bool:
    """TCP-connect probe with a short timeout, cached for _PROBE_TTL_SECONDS.

    The cache means one slow probe per session window, not one per message.
    """
    now = time.time()
    if not force and _probe_cache["reachable"] is not None and (
        now - _probe_cache["ts"] < _PROBE_TTL_SECONDS
    ):
        return _probe_cache["reachable"]

    host, port = _sandbox_host_port()
    reachable = False
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            reachable = True
    except OSError:
        reachable = False
    except Exception:
        reachable = False

    _probe_cache["ts"] = now
    _probe_cache["reachable"] = reachable
    if not reachable:
        logger.info("[repo_chat] sandbox %s:%s unreachable (cached %ss)",
                    host, port, _PROBE_TTL_SECONDS)
    return reachable


def reset_probe_cache() -> None:
    """Testing hook."""
    _probe_cache["ts"] = 0.0
    _probe_cache["reachable"] = None


# =========================================================================
# Staleness header
# =========================================================================

def _staleness_header(db, scan_ids: List[int]) -> str:
    """Build the honest framing line from the scan runs behind the hits."""
    try:
        from app.rag.models import ArchScanRun
        runs = db.query(ArchScanRun).filter(ArchScanRun.id.in_(scan_ids)).all()
    except Exception:
        runs = []

    if not runs:
        return "[CODEBASE SNAPSHOT — scan date unknown]"

    lines = []
    for run in runs:
        scanned = run.completed_at or run.started_at
        date_str = scanned.strftime("%Y-%m-%d %H:%M") if scanned else "unknown date"
        source = getattr(run, "source", None) or "host"
        if source == "sandbox":
            if check_sandbox_reachable():
                lines.append(
                    f"[CODEBASE SNAPSHOT — sandbox scan dated {date_str}; "
                    f"sandbox is online, a fresh scan may supersede this]"
                )
            else:
                lines.append(
                    f"[CODEBASE SNAPSHOT — sandbox currently OFFLINE; this is the "
                    f"last snapshot, dated {date_str}. Phrase answers accordingly: "
                    f"the code may have changed since.]"
                )
        else:
            lines.append(
                f"[CODEBASE SNAPSHOT — host repo scan dated {date_str}; "
                f"treat as current unless told otherwise]"
            )
    return "\n".join(lines)


# =========================================================================
# Search + formatting
# =========================================================================

def _format_chunk(chunk) -> str:
    loc = f"{chunk.file_path}"
    if chunk.start_line:
        loc += f":{chunk.start_line}"
        if chunk.end_line:
            loc += f"-{chunk.end_line}"
    parts = [f"• {chunk.chunk_type} `{chunk.chunk_name}` — {loc}"]
    if chunk.signature:
        parts.append(f"  {chunk.signature.strip()[:200]}")
    if chunk.docstring:
        doc = " ".join(chunk.docstring.split())[:240]
        parts.append(f"  {doc}")
    return "\n".join(parts)


def _embedding_hits(db, message: str):
    """Embedding-channel search. Returns list[ArchCodeChunk] or []."""
    try:
        from app.rag.retrieval.arch_search import ArchitectureSearch
        from app.rag.models import ArchCodeChunk, SourceType
        response = ArchitectureSearch(db).search(
            message, top_k_dirs=0, top_k_chunks=MAX_CHUNKS * 2,
            source_types=[SourceType.ARCH_CHUNK],
        )
        chunks = []
        for r in response.results:
            if r.score < MIN_SIMILARITY:
                continue
            chunk = db.query(ArchCodeChunk).get(r.source_id)
            if chunk is not None and chunk.status == "active":
                chunks.append(chunk)
        return chunks
    except Exception as exc:
        logger.debug("[repo_chat] embedding search unavailable: %s", exc)
        return []


# Common question words that LIKE-match half the codebase's docstrings —
# useless as search keys (the answerer's naive search suffered from this:
# 'where' alone filled the result limit with noise).
_STOPWORDS = {
    "where", "what", "when", "which", "does", "this", "that", "there",
    "the", "and", "for", "with", "from", "into", "implemented", "defined",
    "located", "how", "work", "works", "code", "codebase", "file", "show",
    "find", "tell", "about", "explain",
}


def _keyword_hits(db, message: str):
    """Ranked keyword fallback: name hits beat path hits beat docstring hits."""
    try:
        import re as _re
        from sqlalchemy import or_, func
        from app.rag.models import ArchCodeChunk

        words = [w.lower() for w in _re.findall(r"[\w_]{4,}", message)]
        keywords = [w for w in words if w not in _STOPWORDS][:6]
        if not keywords:
            return []

        filters = []
        for kw in keywords:
            filters.append(func.lower(ArchCodeChunk.chunk_name).contains(kw))
            filters.append(func.lower(ArchCodeChunk.file_path).contains(kw))
            filters.append(func.lower(ArchCodeChunk.docstring).contains(kw))
        rows = (
            db.query(ArchCodeChunk)
            .filter(ArchCodeChunk.status == "active")
            .filter(or_(*filters))
            .limit(120)
            .all()
        )

        def _score(chunk) -> int:
            name = (chunk.chunk_name or "").lower()
            path = (chunk.file_path or "").lower()
            doc = (chunk.docstring or "").lower()
            score = 0
            for kw in keywords:
                if kw in name:
                    score += 4
                if kw in path:
                    score += 2
                if kw in doc:
                    score += 1
            return score

        ranked = sorted(rows, key=_score, reverse=True)
        return [c for c in ranked if _score(c) > 0][:MAX_CHUNKS * 2]
    except Exception as exc:
        logger.debug("[repo_chat] keyword search unavailable: %s", exc)
        return []


def build_repo_context(
    db,
    message: str,
    query_tags: Optional[List[str]] = None,
) -> str:
    """Return a formatted repo-context block for injection, or "".

    Empty string when: feature disabled, gate says not a code query,
    no scan data, or every channel failed. Never raises.
    """
    try:
        if not _enabled() or not message or not message.strip():
            return ""
        if not is_codebase_query(message, query_tags):
            return ""

        chunks = _embedding_hits(db, message)
        used_channel = "embedding"
        if not chunks:
            chunks = _keyword_hits(db, message)
            used_channel = "keyword"
        if not chunks:
            return ""

        # Dedupe by id, cap count
        seen = set()
        selected = []
        for c in chunks:
            if c.id in seen:
                continue
            seen.add(c.id)
            selected.append(c)
            if len(selected) >= MAX_CHUNKS:
                break

        scan_ids = sorted({c.scan_id for c in selected if c.scan_id})
        header = _staleness_header(db, scan_ids)

        body_lines = []
        total = len(header)
        for c in selected:
            entry = _format_chunk(c)
            if total + len(entry) > MAX_CONTEXT_CHARS:
                break
            body_lines.append(entry)
            total += len(entry)

        if not body_lines:
            return ""

        logger.info(
            "[repo_chat] injecting %d repo chunk(s) via %s channel",
            len(body_lines), used_channel,
        )
        return (
            f"{header}\n"
            "Relevant code from ASTRA's own repositories:\n"
            + "\n".join(body_lines)
        )
    except Exception as exc:
        logger.warning("[repo_chat] context build failed: %s", exc)
        return ""


__all__ = [
    "is_codebase_query",
    "build_repo_context",
    "check_sandbox_reachable",
    "reset_probe_cache",
    "CODE_TAGS",
]
