# FILE: app/translation/_intents_conversational.py
# Purpose: Intent definitions: Conversational domain.
# Called-by: app.translation.intents
# Depends-on: app.translation.schemas
# Last-renovated: 2026-06-11
"""
Intent definitions: Conversational domain.
Chat, web search, deep research, memory ingest, memory store.
"""
from __future__ import annotations
from typing import Dict
from .schemas import CanonicalIntent, IntentDefinition


CONVERSATIONAL_INTENTS: Dict[CanonicalIntent, IntentDefinition] = {

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

    # -------------------------------------------------------------------------
    # IMAGE GENERATION (v3.1)
    # -------------------------------------------------------------------------

    CanonicalIntent.GENERATE_IMAGE: IntentDefinition(
        intent=CanonicalIntent.GENERATE_IMAGE,
        trigger_phrases=[
            "generate an image",
            "Generate an image",
            "create an image",
            "Create an image",
            "make an image",
            "Make an image",
            "make me an image",
            "Make me an image",
            "draw me",
            "Draw me",
            "draw a",
            "create a picture",
            "Create a picture",
            "generate a picture",
            "Generate a picture",
            "make a picture",
            "Make a picture",
            "make me a picture",
            "generate image",
            "create image",
            "make image",
            "create a chart",
            "Create a chart",
            "make a chart",
            "Make a chart",
            "generate a chart",
            "Generate a chart",
            "create a graph",
            "Create a graph",
            "make a graph",
            "Make a graph",
            "create an infographic",
            "Create an infographic",
            "make an infographic",
            "Make an infographic",
            "compile an image",
            "Compile an image",
            "visualise",
            "Visualise",
            "visualize",
            "Visualize",
            "make a thumbnail",
            "create a thumbnail",
            "generate a thumbnail",
            "make me a logo",
            "create a logo",
            "design a logo",
            "make a banner",
            "create a banner",
            "make an icon",
            "create an icon",
            "AI image",
            "generate artwork",
            "make artwork",
            "create artwork",
        ],
        trigger_patterns=[
            r"\b(generate|create|make|draw|design|produce)\s+(me\s+)?(an?\s+)?(image|picture|graphic|illustration|visual|thumbnail|banner|icon|logo|avatar|chart|graph|infographic)",
            r"\b(image|picture|graphic|illustration|chart|graph|infographic)\s+(of|for|showing|with|comparing|on)\b",
            r"\bI\s+(want|need)\s+(an?\s+)?(image|picture|graphic|illustration|chart|graph|infographic)",
            r"\b(chart|graph|plot|visualise|visualize)\s+(the|my|this|some|latest|recent|current)",
            r"\b(compile|build|put together)\s+(an?\s+)?(image|chart|graph|infographic|visual)",
            r"\b(?:make|create|generate|draw|design)\s+(?:me\s+)?(?:an?\s+)?(?:image|picture|photo|thumbnail|logo|banner|icon|artwork|illustration|graphic|visual)",
            r"\b(?:image|picture|photo|thumbnail)\s+of\b",
            r"\bgenerate\s+(?:a\s+)?(?:new\s+)?(?:image|picture|graphic|visual)",
        ],
        requires_context=[],
        requires_confirmation=False,
        description="Generate an AI image from a text description",
        behavior=(
            "Image generation — creates an image from the user's description.\n"
            "\n"
            "Process:\n"
            "1. Extract the image description from the message\n"
            "2. Run through prompt synthesis (Stage 1 — enriches the prompt)\n"
            "3. Send enriched prompt to image backend (Stage 2 — GPT Image / Nano Banana)\n"
            "4. Return the generated image as a data URI + file info\n"
            "\n"
            "Provider config: IMAGE_GEN_PROVIDER / IMAGE_GEN_MODEL in .env\n"
            "Fallback: IMAGE_GEN_FALLBACK_PROVIDER / IMAGE_GEN_FALLBACK_MODEL"
        ),
    ),

    # -------------------------------------------------------------------------
    # MEMORY STORE (v2.1)
    # -------------------------------------------------------------------------

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
