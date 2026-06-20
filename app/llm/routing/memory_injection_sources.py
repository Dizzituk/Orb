# FILE: app/llm/routing/memory_injection_sources.py
# Purpose: Source collectors + cross-source de-dup for the single memory front door.
# Called-by: app.llm.routing.memory_injection
# Depends-on: app.memory.summary_injection, app.memory.spine_service, app.memory.integration
# Last-renovated: 2026-06-15 (Job 2: log_block_stats per-turn observability)
"""
Source collectors and de-dup for the memory front door (MEMORY_MAP.md §2).

`memory_injection.build_memory_context` is the single assembler. To keep that
file thin (and under the 30KB cap), the conversation-level sources (rolling
summary + within-conversation recall) and the legacy MemoryRouter keyword
source live here, alongside the cross-source de-dup that guarantees a single
fact appears at most once in the assembled block.

Every collector is failure-proof: any error returns "" and is logged at debug.
None of these run when the front door's D0-D4 gate decides D0 (inject nothing)
— the gate is applied once, in build_memory_context, before these are called.
"""
from __future__ import annotations

import logging
import re
from typing import List, Sequence, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# =============================================================================
# Source collectors (thin, defensive wrappers over the existing builders)
# =============================================================================

def recent_window_ids(db: Session, project_id: int, limit: int = 20) -> List[int]:
    """Recent message ids for `project_id`, used to exclude the live window
    from recall so recall only ever surfaces scrolled-out history. The
    streaming/phone path doesn't carry the window ids, so the front door
    derives them here (mirrors what endpoints/chat.py passes explicitly)."""
    try:
        from app.memory import service as memory_service
        msgs = memory_service.list_messages(db, project_id, limit=limit)
        return [m.id for m in msgs if getattr(m, "id", None) is not None]
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("[mem_sources] window id load failed: %s", exc)
        return []


def collect_summary(db: Session, project_id: int) -> str:
    """Rolling conversation summary block ([CONVERSATION_SUMMARY])."""
    try:
        from app.memory.summary_injection import build_summary_context
        return build_summary_context(db, project_id) or ""
    except Exception as exc:
        logger.debug("[mem_sources] summary skipped: %s", exc)
        return ""


def collect_recall(
    db: Session,
    project_id: int,
    user_message: str,
    window_message_ids: Sequence[int] = (),
) -> str:
    """Within-conversation recall block ([CONVERSATION_RECALL]) — scrolled-out
    messages on the current topic. Empty when the question carries no specific
    topic or nothing older than the window matches."""
    try:
        from app.memory.spine_service import build_conversation_recall
        return build_conversation_recall(
            db, project_id, user_message, window_message_ids,
        ) or ""
    except Exception as exc:
        logger.debug("[mem_sources] recall skipped: %s", exc)
        return ""


def collect_coverage(db: Session, project_id: int, user_message: str) -> str:
    """Conversation coverage block ([CONVERSATION_COVERAGE]) — Nat Job 1.

    A cheap, non-decaying enumeration of every topic discussed this session,
    read from the keyword trail. Only fires on a recall question ("what did we
    talk about", "summarise our chat"); empty otherwise. Pairs with the lossy
    summary: summary synthesises, coverage enumerates."""
    try:
        from app.memory.nat_jobs.coverage import build_coverage_block
        return build_coverage_block(db, project_id, user_message) or ""
    except Exception as exc:
        logger.debug("[mem_sources] coverage skipped: %s", exc)
        return ""


def collect_router(user_message: str, project_id) -> str:
    """Legacy MemoryRouter keyword block ([MEMORY CONTEXT]).

    Kept as a source so the dominant chat path loses no content during the
    front-door consolidation; de-dup (below) drops any line it shares with the
    semantic facts. Reliance on the deprecated unified_query path is a
    documented follow-on (see MEMORY_MAP.md §3) — this goes through the
    MemoryRouter singleton, not unified_query directly."""
    try:
        from app.memory.integration import inject_memory_context
        return inject_memory_context(
            query=user_message, project_id=str(project_id), limit=10,
        ) or ""
    except Exception as exc:
        logger.debug("[mem_sources] router skipped: %s", exc)
        return ""


# =============================================================================
# Cross-source de-dup + assembly
# =============================================================================

_BULLET_RE = re.compile(r"^[\s>#\-\*•]+")
_TAGPREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")  # one leading [TYPE] / [domain]
_LABEL_RE = re.compile(r"^[^:]{1,30}:\s+")    # one short leading "Label: " lead
_MIN_DEDUP_LEN = 15  # don't de-dup short labels/headers, only real content


def _norm_line(line: str) -> str:
    """Normalise a content line so verbatim/near-verbatim repeats across
    sources collapse to the same key.

    Strips the bullet, one leading [TYPE]/[domain] tag, and one short leading
    "Label: " so the SAME fact surfaced two ways — e.g. the semantic facts'
    "• [preference] TTS: <fact>" and the MemoryRouter's "[domain] <fact>" —
    normalise to the same key. Only the content core is compared."""
    s = line.strip()
    s = _BULLET_RE.sub("", s)
    s = _TAGPREFIX_RE.sub("", s)
    s = s.lower()
    # Strip a short "label: " lead only when real content remains after it.
    core = _LABEL_RE.sub("", s)
    if len(core) >= _MIN_DEDUP_LEN:
        s = core
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .!:;-–—")


def _is_structural(line: str) -> bool:
    """Marker/header/empty lines are never de-duped (they frame the block)."""
    s = line.strip()
    if not s:
        return True
    if s.startswith("[") and s.endswith("]"):      # [CONVERSATION_SUMMARY] etc.
        return True
    if s.startswith("<") and s.endswith(">"):       # <user_preferences> etc.
        return True
    if s.startswith("###") or s.startswith("=="):   # section headers
        return True
    return False


def _dedup_block(text: str, seen: set) -> str:
    """Drop content lines whose normalised form was already emitted earlier.
    Mutates `seen`. Structural and short lines pass through untouched."""
    out: List[str] = []
    for line in text.split("\n"):
        if _is_structural(line):
            out.append(line)
            continue
        norm = _norm_line(line)
        if len(norm) >= _MIN_DEDUP_LEN:
            if norm in seen:
                continue
            seen.add(norm)
        out.append(line)
    return "\n".join(out)


def _has_content(text: str) -> bool:
    """True if any non-structural line survived de-dup."""
    return any(not _is_structural(ln) for ln in text.split("\n") if ln.strip())


def assemble_block(
    *,
    preferences_text: str = "",
    facts_text: str = "",
    graph_text: str = "",
    summary_text: str = "",
    recall_text: str = "",
    coverage_text: str = "",
    enrichment_text: str = "",
    repo_context: str = "",
    router_text: str = "",
) -> str:
    """Assemble the single memory block in priority order, de-duping content
    lines across sources (earlier source wins). Returns "" if nothing survives.

    Priority (highest first): preferences → semantic facts (+nudges/sentinel/
    proposals) → cross-domain graph links → conversation summary → recall →
    repo → MemoryRouter. The first two are wrapped in their <tags>; graph/
    summary/recall/router already carry their own [MARKERS]; repo is wrapped in
    <codebase_memory>. Graph links sit right after the semantic facts they
    extend (the relationship channel beside the similarity channel)."""
    seen: set = set()
    parts: List[str] = []

    # (raw_inner, wrapper_open, wrapper_close) — empty wrappers = pre-marked block
    segments: List[Tuple[str, str, str]] = [
        (preferences_text, "<user_preferences>\n", "\n</user_preferences>"),
        (facts_text, "<memory_context>\n", "\n</memory_context>"),
        (graph_text, "", ""),
        (summary_text, "", ""),
        (recall_text, "", ""),
        # Nat Job 1 (coverage) + Job 2 (enrichment) — already carry their own
        # [MARKER]s, so empty wrappers. Coverage sits with the conversation
        # sources; enrichment (fetched real data) right after it.
        (coverage_text, "", ""),
        (enrichment_text, "", ""),
        (repo_context, "<codebase_memory>\n", "\n</codebase_memory>"),
        (router_text, "", ""),
    ]

    for raw, open_tag, close_tag in segments:
        if not raw or not raw.strip():
            continue
        deduped = _dedup_block(raw, seen)
        if not _has_content(deduped):
            continue
        parts.append(f"{open_tag}{deduped.strip()}{close_tag}" if open_tag else deduped.strip())

    return "\n\n".join(parts)


# =============================================================================
# Per-turn observability (Job 2, task 3)
# =============================================================================

def log_block_stats(
    mode: str,
    project_id,
    block: str,
    *,
    summary: bool = False,
    recall: bool = False,
    facts=None,
    depth=None,
    graph=None,
) -> None:
    """Log one line per chat turn describing what the front door assembled.

    The reported voice/phone "lost-middle" bug — a long session where earlier
    topics never re-enter context — shows up here as a run of turns whose block
    carries no recall and no summary: `recall=n summary=n block=0chars`, turn
    after turn. Logged at INFO so it surfaces in normal backend logs without a
    debug flag, on every project-aware path (desktop streaming, phone, /chat).
    Never raises — observability must not break a reply.
    """
    try:
        parts = [
            f"project={project_id}",
            f"mode={mode}",
            f"block={len(block or '')}chars",
            f"recall={'Y' if recall else 'n'}",
            f"summary={'Y' if summary else 'n'}",
        ]
        if facts is not None:
            parts.append(f"facts={'Y' if facts else 'n'}")
        if graph is not None:
            parts.append(f"graph={'Y' if graph else 'n'}")
        if depth is not None:
            parts.append(f"depth={depth}")
        logger.info("[memory_frontdoor] %s", " ".join(parts))
    except Exception:  # pragma: no cover - defensive
        pass


__all__ = [
    "recent_window_ids",
    "collect_summary",
    "collect_recall",
    "collect_coverage",
    "collect_router",
    "assemble_block",
    "log_block_stats",
]
