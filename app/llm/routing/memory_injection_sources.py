# FILE: app/llm/routing/memory_injection_sources.py
# Purpose: Source collectors + cross-source de-dup for the single memory front door.
# Called-by: app.llm.routing.memory_injection
# Depends-on: app.memory.summary_injection, app.memory.spine_service, app.memory.integration
# Last-renovated: 2026-06-25 (assemble_block: hard total-size budget on the injected block)
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
import os
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


def collect_recurring(db: Session, project_id: int, user_message: str) -> str:
    """Recurring-theme block ([RECURRING THEMES] / [RECURRING THEME]).

    Cross-CONVERSATION pattern recognition: the subjects the user returns to
    across many separate sessions. Fires on a recall question (overview) or when
    the current turn's own topic is itself a recurring theme. Complements coverage
    (within-day) and the graph walk (cross-domain links) with the missing
    cross-time recurrence channel."""
    try:
        from app.memory.nat_jobs.recurring_themes import build_recurring_themes_block
        return build_recurring_themes_block(db, project_id, user_message) or ""
    except Exception as exc:
        logger.debug("[mem_sources] recurring themes skipped: %s", exc)
        return ""


_BROAD_ARCH_PATTERNS = [
    # Capability-scoping questions ("what exists / what needs building for X")
    re.compile(r"\bwhat(?:'s| is| are)? (?:already )?(?:exists?|built|in place|missing)\b", re.IGNORECASE),
    re.compile(r"\b(?:what|which)\b.{0,50}\bneeds? (?:to be )?(?:built|building|added|created|done|happen)\b", re.IGNORECASE),
    re.compile(r"\bwhat would (?:it take|need|have to happen)\b", re.IGNORECASE),
    re.compile(r"\b(?:could|can|would|will) (?:you|astra)\b.{0,60}\b(?:do|run|track|watch|monitor|handle|build|research|manage)\b", re.IGNORECASE),
    re.compile(r"\b(?:idle|background)\b.{0,40}\b(?:tasks?|work|jobs?|time)\b", re.IGNORECASE),
    re.compile(r"\b(?:your|astra'?s?) (?:architecture|codebase|subsystems?|systems?|capabilit\w*|components?|memory system|pipeline|stack)\b", re.IGNORECASE),
    re.compile(r"\bwhat (?:do|can) you (?:know about yourself|do|actually do)\b", re.IGNORECASE),
    re.compile(r"\b(?:overview|map|picture) of (?:the |your )?(?:system|codebase|architecture)\b", re.IGNORECASE),
    re.compile(r"\bhow does (?:a |the )?(?:message|request|chat|voice|speech|audio|build)\b.{0,60}\b(?:flow|reach|travel|get|come back|end to end)\b", re.IGNORECASE),
    re.compile(r"\b(?:phone app|bridge app|the app|desktop)\b.{0,60}\b(?:reach|talk to|connect|flow|come back)\b.{0,40}\b(?:backend|astra|server)\b", re.IGNORECASE),
]

_ARCH_CARD_IDS = ("system-overview", "capability-ledger-core")


def _arch_cards_max_chars() -> int:
    try:
        return max(0, int(os.getenv("ASTRA_ARCH_CARDS_MAX_CHARS", "8000")))
    except Exception:
        return 8000


def is_broad_architecture_question(message: str) -> bool:
    """Broad / self-referential system questions — "what could run during idle
    time and what exists for that", "what would need building for X", "how does
    a phone message become speech". These are ALTITUDE questions: file-level
    code chunks answer them badly, the system-level cards answer them well.
    Deterministic and cheap (regex), mirroring chat_injection's gate style."""
    if not message:
        return False
    return any(p.search(message) for p in _BROAD_ARCH_PATTERNS)


def collect_architecture_cards(db: Session, user_message: str, query_tags=None) -> str:
    """Enriched-card block for broad architecture questions (2026-07-02).

    When the question is broad/self-referential, inject the SYSTEM_OVERVIEW
    card and the CAPABILITY_LEDGER card (generated from the HOST tree,
    .architecture/enriched/) AHEAD of file-level chunks — the caller prepends
    this to repo_context inside <codebase_memory>. Bounded by
    ASTRA_ARCH_CARDS_MAX_CHARS (default 8000); trimmed at line boundaries.
    Empty when the gate doesn't fire or the cards aren't ingested yet."""
    try:
        if not is_broad_architecture_question(user_message):
            return ""
        from app.rag.models import ArchCodeChunk, ChunkType
        rows = (
            db.query(ArchCodeChunk)
            .filter(
                ArchCodeChunk.chunk_type == ChunkType.ARCHITECTURE_CARD,
                ArchCodeChunk.chunk_name.in_(_ARCH_CARD_IDS),
                ArchCodeChunk.status == "active",
            )
            .all()
        )
        if not rows:
            return ""
        order = {name: i for i, name in enumerate(_ARCH_CARD_IDS)}
        rows.sort(key=lambda c: order.get(c.chunk_name, 99))
        budget = _arch_cards_max_chars()
        # Equal share per card: the overview alone can fill the whole budget,
        # and for "what exists / what needs building" the LEDGER is the
        # answer-bearing card — both must survive trimming.
        share = max(1500, budget // max(len(rows), 1))
        parts: List[str] = []
        used = 0
        for card in rows:
            body = (card.docstring or "").strip()
            if not body:
                continue
            room = min(share, budget - used)
            if room < 400:
                break
            if len(body) > room:
                body = body[:room].rsplit("\n", 1)[0].rstrip() + "\n[...card trimmed to fit budget...]"
            block = f"[ARCHITECTURE CARD: {card.chunk_name} — host-generated self-knowledge]\n{body}"
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)
    except Exception as exc:
        logger.debug("[mem_sources] architecture cards skipped: %s", exc)
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


# ── Hard total-size budget (2026-06-25) ────────────────────────────────
# Per-source caps (memory_injection._format_preferences / _format_retrieved_records)
# bound each segment; this bounds the assembled WHOLE so the injected block can
# never blow past budget. Trims by priority: segments are added high-priority
# first (preferences -> facts -> ... -> repo -> router), and when the budget is
# hit the overflowing segment is clipped at a line boundary and the lower-priority
# tail is dropped. Env-configurable; ASTRA_MEMORY_BLOCK_MAX_CHARS=0 disables it.
_BUDGET_MARK = "\n[...memory trimmed to fit budget...]"
_MIN_SEGMENT_KEEP = 200  # don't clip a segment down to a uselessly small stub


def _block_max_chars() -> int:
    try:
        return max(0, int(os.getenv("ASTRA_MEMORY_BLOCK_MAX_CHARS", "16000")))
    except Exception:
        return 16000


def _join_within_budget(parts: List[str]) -> str:
    """Join priority-ordered segments under the hard total budget. No-op (and
    byte-identical) when the assembled block already fits, so small blocks are
    untouched; only oversized blocks are trimmed, tail-first."""
    budget = _block_max_chars()
    full = "\n\n".join(parts)
    if not budget or len(full) <= budget:
        return full

    kept: List[str] = []
    used = 0
    for seg in parts:
        sep = 2 if kept else 0
        if used + sep + len(seg) <= budget:
            kept.append(seg)
            used += sep + len(seg)
            continue
        # Overflow: clip this segment at a line boundary if a useful piece fits,
        # then stop (drop all lower-priority segments after it).
        room = budget - used - sep - len(_BUDGET_MARK)
        if room >= _MIN_SEGMENT_KEEP:
            clipped = seg[:room].rsplit("\n", 1)[0].rstrip()
            if clipped:
                kept.append(clipped + _BUDGET_MARK)
        break

    result = "\n\n".join(kept)
    logger.info(
        "[memory_frontdoor] block over budget: %d -> %d chars "
        "(budget=%d, kept %d/%d sources)",
        len(full), len(result), budget, len(kept), len(parts),
    )
    return result


def assemble_block(
    *,
    preferences_text: str = "",
    facts_text: str = "",
    graph_text: str = "",
    recurring_text: str = "",
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
        # Cross-conversation recurring themes sit beside the cross-domain graph
        # links — both are synthesis channels above the raw conversation sources.
        (recurring_text, "", ""),
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
        inner = deduped.strip()
        parts.append(f"{open_tag}{inner}{close_tag}" if open_tag else inner)

    # Hard total-budget backstop: bounds the assembled whole (drops/clips the
    # low-priority tail first). No-op when the block already fits.
    return _join_within_budget(parts)


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
    "collect_recurring",
    "collect_router",
    "collect_architecture_cards",
    "is_broad_architecture_question",
    "assemble_block",
    "log_block_stats",
]
