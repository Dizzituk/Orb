# FILE: app/memory/domains/tier_content.py
# Purpose: Curated content for architecture memory Tiers 1 and 2.
# Called-by: app.memory.seed_tiers
# Depends-on: app.db, app.memory.rag_entries_model
# Last-renovated: 2026-06-11
"""
Curated content for architecture memory Tiers 1 and 2.

Tier 1: System Overview — single document covering ASTRA's purpose,
    architecture, pipeline flow, and design philosophy.

Tier 2: Subsystem Contracts — one document per major subsystem defining
    inputs, outputs, responsibilities, and integration points.

These are static curated documents stored in rag_entries with
domain='architecture'. Updated manually when the system changes
significantly.

Content is stored as Python constants. The seed function inserts
them into rag_entries if not already present (idempotent).
"""

import logging
from datetime import datetime

from sqlalchemy.orm import Session

from app.db import get_db_session
from app.memory.rag_entries_model import RAGEntry

logger = logging.getLogger(__name__)

# Metadata constants
DOMAIN = "architecture"
PROJECT = "astra-core"
SOURCE = "manual"


# =========================================================================
# TIER 1 — System Overview
# =========================================================================

T1_SYSTEM_OVERVIEW = """ASTRA (Autonomous System for Task Routing and Architecture) is an AI-powered \
coding assistant and autonomous software engineering platform. It consists of a Python FastAPI backend \
(Orb/) serving as the AI brain and an Electron + React desktop client (orb-desktop/) providing the \
user-facing chat interface.

CORE PIPELINE FLOW:
User message → Translation Layer (Tier 0 deterministic rules → Tier 1 LLM classification) → \
Intent Classification → Route to Handler:
  - Simple chat → Weaver (conversational streaming engine)
  - Code generation/refactor → SpecGate (specification generation) → Critical Pipeline → \
Segmented Implementation → Cohesion Check → Sandbox Verification → Delivery
  - Architecture tasks → Overwatcher (governance + orchestration)

TECHNOLOGY STACK:
Backend: Python 3 / FastAPI / SQLite (orb_memory.db) / Google Gemini (primary LLM) / OpenAI + \
Anthropic (fallback) / SSE streaming
Frontend: Electron / React 18 / TypeScript / Vite
Infrastructure: Docker sandboxes / NDJSON job ledgers / AES encryption at rest / Token auth

KEY DATABASES:
- orb_memory.db: Primary database — projects, messages, architecture scans, RAG entries, \
embeddings, preferences, experience patterns, job history
- Filesystem: .architecture/ (scan outputs), jobs/ (job ledgers), data/ (files, logs, auth)

DESIGN PHILOSOPHY:
1. Evidence-first: Gather evidence before generating. Never guess. SpecGate requires grounded \
evidence from the codebase before producing specifications.
2. Best way, not quickest: Thorough analysis and complete implementations over quick patches. \
Multi-pass validation over single-shot generation.
3. Sandbox-first: Never execute untested code on the main system. All code changes run in Docker \
sandboxes with build/test verification before delivery.
4. Quarantine-before-purge: Replaced code is quarantined (hidden from queries but recoverable) \
before permanent deletion. Supports rollback.
5. Modular files: 20KB target, 30KB max for logic files. Separate concerns into cooperating \
submodules rather than expanding single files.
6. Multi-provider resilience: Primary LLM (Gemini) with automatic fallback chains to OpenAI \
and Anthropic. No single point of failure.

AGENTIC PIPELINE:
For complex tasks, ASTRA uses a multi-stage pipeline:
1. Translation Layer classifies intent (deterministic Tier 0 → LLM Tier 1 if ambiguous)
2. SpecGate generates formal specifications with evidence from codebase
3. Critical Pipeline decomposes spec into segments
4. Implementer executes each segment in sandbox
5. Cohesion checker verifies segments integrate correctly
6. Overwatcher supervises the entire process, enforcing governance policies
7. Final delivery with build/test verification

MEMORY SYSTEMS:
- RAG: 6,172 embedded code chunks from architecture scans (arch_code_chunks)
- Unified rag_entries table for knowledge, context, decisions, ingested content
- Preference system (astra_preferences) with evidence ledger and confidence scoring
- Hot index for fast D0/D1 retrieval
- MemoryRouter as single entry point for all memory operations
"""


# =========================================================================
# TIER 2 — Subsystem Contracts
# =========================================================================

T2_CONTRACTS = {
    "weaver": {
        "name": "Weaver",
        "purpose": "Primary conversational streaming engine. Handles all chat-mode interactions, "
            "generating natural language responses with context awareness.",
        "inputs": "User message, conversation history, memory context (RAG results, preferences), "
            "translation result (intent classification), active mode state.",
        "outputs": "Streamed SSE response chunks, extracted facts for context memory, "
            "optional tool calls (web search, file operations).",
        "key_files": "app/llm/weaver_stream.py (30.4KB — main engine), "
            "app/llm/weaver_stream_core.py (core logic), app/llm/weaver_simple.py (quick responses), "
            "app/llm/weaver_memory.py (memory integration).",
        "dependencies": "Translation layer (intent), LLM clients (Gemini/OpenAI/Anthropic), "
            "RAG retriever (context), preference system, streaming infrastructure.",
        "integration": "Called by stream_router when intent is chat/conversational. "
            "Feeds extracted facts to context memory. Can escalate to SpecGate if "
            "it detects a build request mid-conversation.",
        "constraints": "Must stream tokens incrementally via SSE. Must not trigger code "
            "execution directly — escalates to pipeline for that. Respects user preferences "
            "for tone, verbosity, and formatting.",
    },
    "specgate": {
        "name": "SpecGate",
        "purpose": "Specification generation gate. Transforms natural language requirements into "
            "formal, grounded specifications before any code is written.",
        "inputs": "User requirement (from translation), codebase evidence (from RAG + file discovery), "
            "existing spec history, project context.",
        "outputs": "Pot spec (formal specification document), evidence bundle, "
            "confidence score, clarification questions if ambiguous.",
        "key_files": "app/llm/spec_gate_stream.py (35.8KB — streaming spec generation), "
            "app/pot_spec/grounded/spec_runner.py (spec execution), "
            "app/pot_spec/grounded/file_discovery.py (evidence gathering), "
            "app/pot_spec/grounded/segmentation.py (spec decomposition).",
        "dependencies": "RAG system (evidence retrieval), file discovery (codebase scanning), "
            "LLM clients (spec generation), architecture scan data.",
        "integration": "Called after translation classifies intent as build/refactor. "
            "Output feeds into Critical Pipeline for segmented implementation. "
            "Can return to user for clarification if evidence is insufficient.",
        "constraints": "Must gather evidence BEFORE generating spec (evidence-first rule). "
            "Spec must reference actual files and line numbers. Must not hallucinate "
            "file paths or function signatures — all references grounded in scan data.",
    },
    "critical_pipeline": {
        "name": "Critical Pipeline",
        "purpose": "Executes high-stakes code generation and modification tasks through a "
            "multi-stage critique-revise loop.",
        "inputs": "Pot spec (from SpecGate), segmentation plan, codebase context, "
            "sandbox environment reference.",
        "outputs": "Generated/modified code files, critique results, revision history, "
            "build/test verification results.",
        "key_files": "app/llm/critical_pipeline_stream.py (streaming), "
            "app/llm/pipeline/high_stakes.py (33.6KB — execution), "
            "app/llm/pipeline/critique.py (LLM-based review), "
            "app/llm/pipeline/critique_schemas.py (severity levels).",
        "dependencies": "SpecGate (spec input), sandbox controller (Docker execution), "
            "LLM clients (code generation + critique), orchestrator (segment management).",
        "integration": "Receives segmented spec from SpecGate. Each segment executed in "
            "sandbox, critiqued, revised if needed. Final output goes through cohesion "
            "check before delivery. Overwatcher monitors throughout.",
        "constraints": "All code execution happens in sandbox — never on main system. "
            "Critique loop runs up to 3 iterations per segment. Blocker-level critique "
            "findings halt the pipeline until resolved.",
    },
    "overwatcher": {
        "name": "Overwatcher",
        "purpose": "Governance and supervision layer. Enforces policies, manages strikes, "
            "monitors pipeline execution quality, and orchestrates complex multi-step tasks.",
        "inputs": "Pipeline events, user messages, governance policy rules, strike state, "
            "cost metrics, execution history.",
        "outputs": "Policy decisions (allow/deny/warn), strike records, execution logs, "
            "quality scores, intervention commands.",
        "key_files": "app/overwatcher/overwatcher_command.py (34KB — command processor), "
            "app/overwatcher/architecture_executor/orchestrator.py (95KB — main orchestrator), "
            "app/overwatcher/conduct_policy.py (governance rules), "
            "app/overwatcher/strike_state.py (strike tracking).",
        "dependencies": "All pipeline components (monitors them), LLM clients (for analysis), "
            "job system, translation layer.",
        "integration": "Wraps around the entire pipeline. Receives events from SpecGate, "
            "Critical Pipeline, and Orchestrator. Can halt execution, issue strikes, "
            "or force re-planning. Also handles direct architecture commands "
            "(scan, refactor, update).",
        "constraints": "Must not be bypassed — all pipeline-level operations route through "
            "Overwatcher. Strike thresholds are hard-coded. Cost guards prevent runaway "
            "LLM spending. The orchestrator.py at 95KB is a known monolith — decomposition "
            "planned but complex due to deep internal state.",
    },
    "implementer": {
        "name": "Implementer (Orchestrator)",
        "purpose": "Manages segmented implementation of spec'd tasks. Coordinates the "
            "segment-by-segment execution loop with compilation, cohesion checking, "
            "and integration verification.",
        "inputs": "Segmentation plan (from SpecGate), sandbox reference, "
            "per-segment instructions, codebase state.",
        "outputs": "Implemented code per segment, cohesion check results, "
            "integration test results, final compiled output.",
        "key_files": "app/orchestrator/ (~100+ files — heavily decomposed), "
            "key entry points in the segment loop and compilation modules.",
        "dependencies": "SpecGate (segmentation), Critical Pipeline (per-segment execution), "
            "sandbox controller, RAG (codebase context).",
        "integration": "Sits between SpecGate and delivery. Iterates through segments, "
            "delegates each to Critical Pipeline, verifies cohesion between segments, "
            "runs integration checks, and compiles final output.",
        "constraints": "Segments must be executed in dependency order. Each segment's "
            "output must pass cohesion check against prior segments before proceeding. "
            "Failed cohesion triggers re-implementation of the offending segment.",
    },
    "translation_layer": {
        "name": "Translation Layer",
        "purpose": "First gate for all user input. Classifies intent, detects commands, "
            "manages operating modes, and routes messages to the appropriate handler.",
        "inputs": "Raw user message, conversation history, active mode state, "
            "phrase cache, confidence scores.",
        "outputs": "Structured intent (chat/code/refactor/search/command/etc.), "
            "confidence score, detected mode transitions, extracted parameters.",
        "key_files": "app/translation/intents.py (40.5KB — core intent classifier), "
            "app/translation/tier0_rules.py (30.7KB — deterministic rules), "
            "app/translation/tier1_classifier.py (LLM classification), "
            "app/translation/gates.py (routing gates), "
            "app/translation/translator.py (orchestrator).",
        "dependencies": "Phrase cache, confidence scores (future — Job 5), "
            "LLM client (Tier 1 only), mode state.",
        "integration": "Every user message passes through translation first. "
            "Output determines which handler receives the message: Weaver (chat), "
            "SpecGate (build), Overwatcher (commands), direct LLM, web search, etc.",
        "constraints": "Tier 0 (deterministic) must be fast — no LLM calls. "
            "Tier 1 (LLM) only fires when Tier 0 is ambiguous. "
            "Must never misclassify destructive commands as chat.",
    },
    "rag": {
        "name": "RAG System",
        "purpose": "Retrieval-Augmented Generation. Indexes the codebase into searchable "
            "chunks with embeddings for semantic search, providing relevant context "
            "to every LLM call.",
        "inputs": "Architecture scan output (SIGNATURES, INDEX), user queries, "
            "embedding requests, lifecycle events (quarantine/purge).",
        "outputs": "Relevant code chunks with relevance scores, embedded vectors, "
            "architecture context for LLM prompts, lifecycle status changes.",
        "key_files": "app/rag/models.py (data models), app/rag/lifecycle.py (status management), "
            "app/rag/indexer.py (chunk indexing), app/rag/retriever.py (search), "
            "app/rag/descriptors/descriptor_gen.py (embeddable text generation).",
        "dependencies": "Architecture scan pipeline, embedding service (OpenAI embeddings), "
            "SQLite database, MemoryRouter.",
        "integration": "Fed by architecture scans. Queried by Weaver, SpecGate, "
            "and Critical Pipeline for codebase context. Lifecycle managed by "
            "refactor pipeline (quarantine on refactor, purge on confirmation). "
            "Bridged into MemoryRouter via ArchitectureStore.",
        "constraints": "All queries filter by status='active' — quarantined entries "
            "are invisible. Embeddings use OpenAI text-embedding-3-small. "
            "Chunk size targets ~500 tokens for optimal retrieval.",
    },
    "segment_loop": {
        "name": "Segment Loop",
        "purpose": "Core iteration loop for segmented task execution. Manages the "
            "cycle of: take next segment → implement → verify → cohesion check → advance.",
        "inputs": "Ordered list of segments from segmentation plan, sandbox state, "
            "prior segment outputs (for cohesion).",
        "outputs": "Per-segment implementation results, cohesion verdicts, "
            "loop completion status, accumulated errors.",
        "key_files": "app/orchestrator/ segment loop modules (decomposed from original "
            "135KB monolith into 10+ cooperating modules ~10.5KB each).",
        "dependencies": "Critical Pipeline (per-segment implementation), "
            "cohesion checker, sandbox controller, orchestrator state.",
        "integration": "Called by the Implementer/Orchestrator. Each iteration "
            "delegates to Critical Pipeline, checks cohesion, and either advances "
            "or triggers re-implementation. Reports progress to Overwatcher.",
        "constraints": "Segments execute in dependency order. Maximum 3 retries "
            "per segment. Cohesion failure on segment N can trigger rollback to "
            "segment N-1. The original 135KB monolith was surgically decomposed — "
            "files with single large functions needed context-object + phase-extraction "
            "pattern rather than automated splitting.",
    },
}


# =========================================================================
# Seeding Functions
# =========================================================================

def _entry_exists(db: Session, tier: str, key: str) -> bool:
    """Check if a tier entry already exists (by matching tier+key in chunk_text prefix)."""
    marker = f"[{tier}:{key}]"
    return db.query(RAGEntry).filter(
        RAGEntry.domain == DOMAIN,
        RAGEntry.project_id == PROJECT,
        RAGEntry.chunk_text.like(f"{marker}%"),
    ).first() is not None


def seed_tier1(db: Session) -> int:
    """Seed Tier 1 — System Overview. Returns 1 if inserted, 0 if exists."""
    if _entry_exists(db, "T1", "system_overview"):
        logger.debug("[tier_content] T1 system_overview already seeded")
        return 0

    entry = RAGEntry(
        project_id=PROJECT,
        domain=DOMAIN,
        chunk_text=f"[T1:system_overview] {T1_SYSTEM_OVERVIEW}",
        status="ACTIVE",
        ingest_source=SOURCE,
        indexed_at=datetime.utcnow(),
    )
    db.add(entry)
    db.commit()
    logger.info("[tier_content] Seeded T1 system_overview")
    return 1


def seed_tier2(db: Session) -> int:
    """Seed Tier 2 — Subsystem Contracts. Returns count of entries inserted."""
    count = 0
    for key, contract in T2_CONTRACTS.items():
        if _entry_exists(db, "T2", key):
            logger.debug(f"[tier_content] T2 {key} already seeded")
            continue

        # Format contract as structured text
        text = (
            f"[T2:{key}] SUBSYSTEM CONTRACT: {contract['name']}\n\n"
            f"PURPOSE: {contract['purpose']}\n\n"
            f"INPUTS: {contract['inputs']}\n\n"
            f"OUTPUTS: {contract['outputs']}\n\n"
            f"KEY FILES: {contract['key_files']}\n\n"
            f"DEPENDENCIES: {contract['dependencies']}\n\n"
            f"INTEGRATION: {contract['integration']}\n\n"
            f"CONSTRAINTS: {contract['constraints']}"
        )

        entry = RAGEntry(
            project_id=PROJECT,
            domain=DOMAIN,
            chunk_text=text,
            status="ACTIVE",
            ingest_source=SOURCE,
            indexed_at=datetime.utcnow(),
        )
        db.add(entry)
        count += 1

    if count > 0:
        db.commit()
        logger.info(f"[tier_content] Seeded {count} T2 subsystem contracts")
    return count


def seed_tiers_1_and_2() -> dict:
    """
    Seed Tiers 1 and 2 into rag_entries. Idempotent.

    Returns summary of what was inserted.
    """
    db = get_db_session()
    try:
        t1 = seed_tier1(db)
        t2 = seed_tier2(db)
        return {"tier1_inserted": t1, "tier2_inserted": t2}
    finally:
        db.close()
