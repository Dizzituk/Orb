# FILE: app/debug/memory_tool_definitions.py
# Purpose: Tool schemas for the conversation LLM's memory access.
# Called-by: app.debug.tool_definitions
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Tool schemas for the conversation LLM's memory access.

These definitions expose the tiered memory system directly to the chat
model so it can save, update, forget, and search facts DURING the turn
- eliminating the confabulation failure mode where the model says
"saved to memory" without any write actually happening.

Kept separate from tool_definitions.py to keep per-file size sensible.
"""
from __future__ import annotations

from typing import List


# =============================================================================
# save_to_memory
# =============================================================================

SAVE_TO_MEMORY_TOOL = {
    "name": "save_to_memory",
    "description": (
        "Save a durable fact about the user to ASTRA's tiered memory. "
        "Call this when the user tells you something worth remembering "
        "across conversations - biographical facts, preferences, "
        "principles, project context, or learning goals. "
        "NEVER claim to have saved, remembered, or stored anything unless "
        "you have called this tool AND received saved=true in the result. "
        "If the user says 'remember this' or 'don't forget', use weight=5 "
        "and permanence=permanent. If they mention something in passing, "
        "use lower weight. For historical places the user has lived, use "
        "save_residence instead of this tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["biographical", "preference", "philosophy", "project", "learning"],
                "description": (
                    "biographical = personal hard facts (name, DOB, location, nationality, family). "
                    "preference = style/tool/workflow preferences that persist across sessions. "
                    "philosophy = hard rules and principles stated as absolutes (always/never). "
                    "project = stable project context (project names, long-running goals, legal cases). "
                    "learning = durable skill-building patterns or knowledge interests."
                ),
            },
            "key": {
                "type": "string",
                "description": (
                    "Stable snake_case identifier. Use the SAME key when the same "
                    "concept comes up again so reinforcement works. Good: "
                    "'favourite_band', 'code_style', 'd7_visa_plan'. Bad: "
                    "'band_mentioned_today', 'user_said_about_code'."
                ),
            },
            "value": {
                "type": "string",
                "description": "The fact itself, in the user's own words or a faithful paraphrase.",
            },
            "weight": {
                "type": "integer",
                "description": (
                    "1-5 commitment/consequence weight. "
                    "1-2 = mentioned casually, low commitment. "
                    "3 = moderate/default. "
                    "4 = clear commitment or load-bearing context. "
                    "5 = explicit 'remember this' or critical biographical fact."
                ),
            },
            "permanence": {
                "type": "string",
                "enum": ["permanent", "evolving", "ephemeral"],
                "description": (
                    "permanent = biographical / philosophy / hard rules. Tier 1, no decay. "
                    "evolving = preferences and active projects that may change over time. Tier 2. "
                    "ephemeral = current session context or WIP state. Tier 3, decays fast."
                ),
            },
            "reason": {
                "type": "string",
                "description": "Optional short note explaining why this is being saved (audit trail).",
            },
        },
        "required": ["category", "key", "value"],
    },
}


# =============================================================================
# update_memory
# =============================================================================

UPDATE_MEMORY_TOOL = {
    "name": "update_memory",
    "description": (
        "Change the value of an existing memory fact. Use this when the "
        "user corrects or updates something you previously saved - e.g. "
        "'actually I moved last year', 'no, it's X not Y', or "
        "'change my preference to Z'. Prefer this over save_to_memory "
        "when you know the key already exists. If you're not sure whether "
        "the fact exists, call search_memory first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": (
                    "The preference key to update. Can be the bare key "
                    "('favourite_band') or the full form "
                    "('conv_extract:preference:favourite_band')."
                ),
            },
            "new_value": {
                "type": "string",
                "description": "The replacement value.",
            },
            "reason": {
                "type": "string",
                "description": "Why the update - for audit trail (optional).",
            },
        },
        "required": ["key", "new_value"],
    },
}


# =============================================================================
# forget_memory
# =============================================================================

FORGET_MEMORY_TOOL = {
    "name": "forget_memory",
    "description": (
        "Mark a memory fact for expiry. Use this when the user explicitly "
        "says 'forget that' or 'don't remember X any more'. The evidence "
        "ledger is preserved for audit - the fact just stops being "
        "retrieved. Don't use this when the user is CORRECTING a fact "
        "(use update_memory for corrections)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "The preference key to forget (bare or full form).",
            },
            "reason": {
                "type": "string",
                "description": "Why the user wants this forgotten (optional audit note).",
            },
        },
        "required": ["key"],
    },
}


# =============================================================================
# search_memory
# =============================================================================

SEARCH_MEMORY_TOOL = {
    "name": "search_memory",
    "description": (
        "Search ASTRA's stored memory by keyword or category. Use this "
        "when the user asks 'what do you know about X', 'where have I "
        "lived', 'what are my preferences', or any similar question that "
        "requires retrieving stored facts. Also use before update_memory "
        "or forget_memory if you're unsure whether a fact exists. "
        "Returns both preference records and always-injected identity "
        "fields - check both lists in the result."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Free-text search matched against stored values and keys. "
                    "Empty string returns all active facts (optionally filtered "
                    "by category)."
                ),
            },
            "category": {
                "type": "string",
                "enum": ["biographical", "preference", "philosophy", "project", "learning"],
                "description": "Optional filter - limit results to one category.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 20, cap 50).",
            },
        },
        "required": [],
    },
}


# =============================================================================
# save_residence
# =============================================================================

SAVE_RESIDENCE_TOOL = {
    "name": "save_residence",
    "description": (
        "Append or update an entry in the user's residence history. Use "
        "this when the user mentions a PAST place they've lived - "
        "'I grew up in X', 'from 14 to 18 I lived in the Algarve', "
        "'I lived in Exeter for 22 years'. This writes to the "
        "always-injected identity block so ASTRA remembers the user's "
        "residence timeline across conversations. "
        "For CURRENT location (where the user lives now), use "
        "save_to_memory with category=biographical and key=current_location. "
        "Deduplicates by place name - calling twice with the same place "
        "merges updates rather than duplicating."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "place": {
                "type": "string",
                "description": (
                    "Location name, as specific as the user gave it "
                    "(e.g. 'Exeter', 'Algarve, Portugal', 'London')."
                ),
            },
            "from_year": {
                "type": "integer",
                "description": "Year moved in (4-digit). Omit if unknown.",
            },
            "to_year": {
                "type": "integer",
                "description": "Year left. Omit if user still lives there or unknown.",
            },
            "duration_years": {
                "type": "integer",
                "description": (
                    "Explicit duration when years aren't known "
                    "('lived there for 22 years' without specific dates)."
                ),
            },
            "notes": {
                "type": "string",
                "description": "Short context note (e.g. 'teenage years', 'with family').",
            },
        },
        "required": ["place"],
    },
}


# =============================================================================
# bundles
# =============================================================================

def get_memory_tools() -> List[dict]:
    """All memory tools - injected for every chat model that supports tools."""
    return [
        SAVE_TO_MEMORY_TOOL,
        UPDATE_MEMORY_TOOL,
        FORGET_MEMORY_TOOL,
        SEARCH_MEMORY_TOOL,
        SAVE_RESIDENCE_TOOL,
    ]
