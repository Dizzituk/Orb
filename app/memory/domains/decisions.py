# FILE: app/memory/domains/decisions.py
# Purpose: Tier 4 — Design Decisions Log.
# Called-by: app.memory.domains, app.memory.seed_tiers, app.memory.startup
# Depends-on: app.db, app.memory.rag_entries_model
# Last-renovated: 2026-06-11
"""
Tier 4 — Design Decisions Log.

Append-only log of architectural decisions with rationale.
Each decision records what was decided, why, what alternatives
were considered, and what component it affects.

Stored in rag_entries with domain='architecture' and tier='T4'.
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.db import get_db_session
from app.memory.rag_entries_model import RAGEntry

logger = logging.getLogger(__name__)

DOMAIN = "architecture"
PROJECT = "astra-core"
TIER = "T4"


# =========================================================================
# DecisionStore
# =========================================================================

class DecisionStore:
    """
    Append-only design decisions log.

    Each decision is stored as a rag_entries row with structured text.
    Decisions are never updated or deleted — only appended.
    """

    def add_decision(
        self,
        title: str,
        decision: str,
        rationale: str,
        alternatives: Optional[str] = None,
        component: Optional[str] = None,
        project_id: str = PROJECT,
    ) -> int:
        """
        Add a new design decision.

        Args:
            title: Short name (e.g. "quarantine-before-purge")
            decision: What was decided
            rationale: Why this choice was made
            alternatives: What other options were considered
            component: Which subsystem this affects

        Returns:
            The rag_entries row ID.
        """
        text = _format_decision(title, decision, rationale, alternatives, component)

        db = get_db_session()
        try:
            entry = RAGEntry(
                project_id=project_id,
                domain=DOMAIN,
                chunk_text=text,
                status="ACTIVE",
                ingest_source="decision_log",
                indexed_at=datetime.utcnow(),
            )
            db.add(entry)
            db.commit()
            db.refresh(entry)
            logger.info(f"[decisions] Added decision: {title} (id={entry.id})")
            return entry.id
        finally:
            db.close()

    def query_decisions(
        self,
        search_text: Optional[str] = None,
        component: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Query decisions by keyword or component.

        Returns list of dicts with id, title, text.
        """
        db = get_db_session()
        try:
            query = db.query(RAGEntry).filter(
                RAGEntry.domain == DOMAIN,
                RAGEntry.status == "ACTIVE",
                RAGEntry.chunk_text.like(f"[{TIER}:%"),
            )

            if search_text:
                query = query.filter(
                    RAGEntry.chunk_text.ilike(f"%{search_text}%")
                )
            if component:
                query = query.filter(
                    RAGEntry.chunk_text.ilike(f"%COMPONENT: {component}%")
                )

            rows = query.order_by(RAGEntry.indexed_at.desc()).limit(limit).all()
            return [_parse_decision(r) for r in rows]
        finally:
            db.close()


# =========================================================================
# Formatting helpers
# =========================================================================

def _format_decision(
    title: str,
    decision: str,
    rationale: str,
    alternatives: Optional[str],
    component: Optional[str],
) -> str:
    """Format a decision as structured text for storage."""
    parts = [
        f"[{TIER}:{title}] DESIGN DECISION: {title}",
        f"DECISION: {decision}",
        f"RATIONALE: {rationale}",
    ]
    if alternatives:
        parts.append(f"ALTERNATIVES CONSIDERED: {alternatives}")
    if component:
        parts.append(f"COMPONENT: {component}")
    parts.append(f"RECORDED: {datetime.utcnow().isoformat()}")
    return "\n\n".join(parts)


def _parse_decision(entry: RAGEntry) -> dict:
    """Parse a decision entry back into a dict."""
    return {
        "id": entry.id,
        "text": entry.chunk_text,
        "indexed_at": entry.indexed_at,
    }


# =========================================================================
# Seed data
# =========================================================================

SEED_DECISIONS = [
    {
        "title": "deterministic-tier0-skip-llm",
        "decision": "Tier 0 translation uses deterministic pattern matching with no LLM calls. "
            "Only ambiguous intents escalate to Tier 1 (LLM classification).",
        "rationale": "LLM calls are slow and expensive. Most commands (scan, refactor, status) "
            "can be reliably detected with regex and keyword matching. Reserving LLM for "
            "genuinely ambiguous cases keeps response times low.",
        "alternatives": "All-LLM classification (too slow), hybrid with confidence threshold "
            "(chosen for Tier 1, but Tier 0 is purely deterministic).",
        "component": "translation_layer",
    },
    {
        "title": "quarantine-before-purge",
        "decision": "When code is refactored, old entries are quarantined (hidden from queries "
            "but retained in DB) before permanent deletion. Purge is a separate manual step.",
        "rationale": "Immediate deletion prevents rollback. Quarantine provides a safety window "
            "where refactored code can be restored if the new package fails boot/test. "
            "Purge only runs after the user confirms stability.",
        "alternatives": "Direct delete (no rollback), soft-delete with TTL (arbitrary timeouts), "
            "version branches (too complex for SQLite).",
        "component": "rag",
    },
    {
        "title": "confidence-time-decay",
        "decision": "Confidence scores for phrase-to-intent mappings decay over time. "
            "Recency factor: 1.0 if used within 30 days, then 0.95^(days_since_30).",
        "rationale": "User language evolves. A phrase that meant 'refactor' three months ago "
            "might mean something different now. Decay ensures stale mappings lose influence "
            "and the system adapts to current usage patterns.",
        "alternatives": "No decay (stale scores persist forever), hard expiry (loses all "
            "history), session-scoped only (no long-term learning).",
        "component": "confidence",
    },
    {
        "title": "evidence-first-spec-generation",
        "decision": "SpecGate must gather codebase evidence (file contents, function signatures, "
            "existing patterns) BEFORE generating any specification. Evidence bundle is a "
            "required input, not an optional enhancement.",
        "rationale": "LLMs hallucinate file paths, function names, and API shapes when generating "
            "from description alone. Grounding specs in actual codebase evidence eliminates "
            "entire categories of spec errors. Evidence-first is non-negotiable.",
        "alternatives": "Generate-then-verify (catches errors too late), user-supplied evidence "
            "(burdens the user), no evidence (hallucination city).",
        "component": "specgate",
    },
    {
        "title": "bridge-not-replace-rag-unification",
        "decision": "The unified memory system bridges existing tables (arch_code_chunks, "
            "astra_preferences) via MemoryRouter rather than migrating data into a single "
            "rag_entries table.",
        "rationale": "6,172 embedded chunks with working lifecycle ops and architecture scan "
            "integrations would be at risk during migration. Bridge approach gives the spec's "
            "unified interface without touching working data. New domains use rag_entries; "
            "existing domains stay in their proven tables.",
        "alternatives": "Full migration into rag_entries (risk of data loss and broken integrations), "
            "keep everything separate (no unified query surface).",
        "component": "memory",
    },
    {
        "title": "file-size-20kb-30kb-limits",
        "decision": "Logic files target 20KB with a hard max of 30KB. Data-heavy files "
            "(constants, templates, schemas) can exceed if logic content is minimal. "
            "Files exceeding limits are split into cooperating submodules automatically.",
        "rationale": "Monolith files are unreadable, hard to navigate, and create massive "
            "blast radius on changes. Small files enable easy debugging, low-risk edits, "
            "and efficient AI-assisted iteration (fits in context windows).",
        "alternatives": "No limit (monoliths), stricter limits (excessive file count), "
            "class-level splitting only (doesn't catch utility bloat).",
        "component": "all",
    },
    {
        "title": "sandbox-first-execution",
        "decision": "All generated or modified code executes in Docker sandbox containers "
            "before touching the main system. Build and test verification required.",
        "rationale": "LLM-generated code can contain subtle bugs, break imports, or introduce "
            "security issues. Sandbox execution catches these before they affect the live "
            "system. The cost of spinning up a container is always preferable to debugging "
            "a broken production environment.",
        "alternatives": "Direct execution (dangerous), manual review only (doesn't catch "
            "runtime errors), virtual environments (insufficient isolation).",
        "component": "sandbox",
    },
    {
        "title": "multi-provider-llm-fallback",
        "decision": "LLM routing uses a primary provider (Google Gemini) with automatic "
            "fallback chains to OpenAI and Anthropic. Provider selection considers cost, "
            "latency, and task complexity.",
        "rationale": "Single-provider dependency creates outage risk. API rate limits, "
            "downtime, and model-specific weaknesses are mitigated by having multiple "
            "providers available. Each provider has different strengths.",
        "alternatives": "Single provider (fragile), user-selected provider only (burdens user), "
            "parallel calls (expensive and complex).",
        "component": "llm",
    },
    {
        "title": "manual-decomposition-over-automated-extraction",
        "decision": "Large orchestrator files with single monolithic functions require the "
            "context-object + phase-extraction pattern for decomposition. Automated surgical "
            "extraction cannot reach inside function bodies — it only splits at the "
            "function/class boundary level.",
        "rationale": "The segment_loop decomposition (135KB → 10.5KB + 10 modules) proved that "
            "files with one giant function need a fundamentally different approach. The "
            "context object captures shared state, then each phase is extracted as a "
            "separate function that receives the context. Automated tools can't infer "
            "phase boundaries inside a function — that requires human/LLM understanding.",
        "alternatives": "Automated AST splitting (can't split inside functions), "
            "leaving as monolith (unreadable), arbitrary line-count splits (breaks logic).",
        "component": "refactor",
    },
]


def seed_decisions() -> int:
    """
    Seed initial design decisions. Idempotent — skips existing.

    Returns count of decisions inserted.
    """
    db = get_db_session()
    store = DecisionStore()
    count = 0

    try:
        for d in SEED_DECISIONS:
            # Check if already exists by title marker
            marker = f"[{TIER}:{d['title']}]"
            existing = db.query(RAGEntry).filter(
                RAGEntry.domain == DOMAIN,
                RAGEntry.chunk_text.like(f"{marker}%"),
            ).first()

            if existing:
                logger.debug(f"[decisions] Decision '{d['title']}' already seeded")
                continue

            text = _format_decision(
                d["title"], d["decision"], d["rationale"],
                d.get("alternatives"), d.get("component"),
            )
            entry = RAGEntry(
                project_id=PROJECT,
                domain=DOMAIN,
                chunk_text=text,
                status="ACTIVE",
                ingest_source="decision_log",
                indexed_at=datetime.utcnow(),
            )
            db.add(entry)
            count += 1

        if count > 0:
            db.commit()
            logger.info(f"[decisions] Seeded {count} design decisions")
        return count
    finally:
        db.close()
