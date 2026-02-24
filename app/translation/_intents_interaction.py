# FILE: app/translation/_intents_interaction.py
"""
User-facing interaction intent definitions.
Chat, Web Search, Deep Research, Memory Ingest/Store.
"""
from __future__ import annotations
from typing import Dict
from .schemas import CanonicalIntent, IntentDefinition


INTERACTION_INTENTS: Dict[CanonicalIntent, IntentDefinition] = {

    # -------------------------------------------------------------------------
    # CHAT (no action)
    # -------------------------------------------------------------------------

    CanonicalIntent.CHAT_ONLY: IntentDefinition(
        intent=CanonicalIntent.CHAT_ONLY,
        trigger_phrases=[],
        trigger_patterns=[],
        requires_context=[],
        requires_confirmation=False,
        description="Normal conversation - no backend actions",
        behavior=(
            "Chat mode - NO backend actions.\n"
            "Used for:\n"
            "- Normal conversation\n"
            "- Thinking\n"
            "- Explanations\n"
            "- Planning\n"
            "- Storytelling\n"
            "- Meta-discussion"
        ),
    ),

    # -------------------------------------------------------------------------
    # WEB SEARCH (v2.1)
    # -------------------------------------------------------------------------

    CanonicalIntent.WEB_SEARCH: IntentDefinition(
        intent=CanonicalIntent.WEB_SEARCH,
        trigger_phrases=[],
        trigger_patterns=[],
        requires_context=[],
        requires_confirmation=False,
        description="Search the web and return synthesised results with sources",
        behavior=(
            "Web search — read-only, no confirmation required.\n"
            "\n"
            "Process:\n"
            "1. Extract search query from message\n"
            "2. Query Brave Search API (DuckDuckGo fallback)\n"
            "3. Fetch top source pages for evidence\n"
            "4. LLM synthesises answer from sources\n"
            "5. Return answer with cited sources"
        ),
    ),

    # -------------------------------------------------------------------------
    # DEEP RESEARCH (v2.1)
    # -------------------------------------------------------------------------

    CanonicalIntent.DEEP_RESEARCH: IntentDefinition(
        intent=CanonicalIntent.DEEP_RESEARCH,
        trigger_phrases=[],
        trigger_patterns=[],
        requires_context=[],
        requires_confirmation=False,
        description="Iterative multi-round web research with gap analysis and synthesis",
        behavior=(
            "Deep research — thorough, multi-round investigation.\n"
            "\n"
            "Process:\n"
            "1. LLM plans 3-5 targeted search queries\n"
            "2. Runs all queries via Brave/DDG\n"
            "3. Fetches and tags top pages by credibility\n"
            "4. LLM identifies information gaps\n"
            "5. Generates follow-up queries to fill gaps\n"
            "6. Repeats up to 3 rounds\n"
            "7. Final synthesis from all evidence with citations"
        ),
    ),

    # -------------------------------------------------------------------------
    # MEMORY INGEST (v2.1)
    # -------------------------------------------------------------------------

    CanonicalIntent.MEMORY_INGEST: IntentDefinition(
        intent=CanonicalIntent.MEMORY_INGEST,
        trigger_phrases=[],
        trigger_patterns=[],
        requires_context=[],
        requires_confirmation=False,
        description="Ingest bulk data into ASTRA memory (files, exports, uploads)",
        behavior=(
            "Memory ingest — processes uploaded data through the ingest pipeline.\n"
            "\n"
            "Process:\n"
            "1. Parse uploaded file (JSON, text, CSV, etc.)\n"
            "2. Extract discrete knowledge items\n"
            "3. Classify domain, memory tier, confidence\n"
            "4. Deduplicate against existing memory\n"
            "5. Store high-confidence items, queue low-confidence for review\n"
            "6. Report summary: stored, duplicates, review queue"
        ),
    ),

    CanonicalIntent.MEMORY_STORE: IntentDefinition(
        intent=CanonicalIntent.MEMORY_STORE,
        trigger_phrases=[],
        trigger_patterns=[],
        requires_context=[],
        requires_confirmation=False,
        description="Remember a specific fact or preference from conversation",
        behavior=(
            "Memory store — saves a specific fact or preference.\n"
            "\n"
            "Process:\n"
            "1. Extract the fact/preference from message\n"
            "2. Classify as preference, biographical, or knowledge\n"
            "3. Route to appropriate capture channel\n"
            "4. Confirm storage to user"
        ),
    ),
}
