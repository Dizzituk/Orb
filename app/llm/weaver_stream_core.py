# FILE: app/llm/weaver_stream_core.py
"""
Weaver Stream Core - Prompt building and response parsing for ASTRA Weaver.

v2.2 (2026-01-22): WEAVER HARDENING - Bug 4 Fix
- Added execution_mode field to schema (backward compatible)
- Supports "discussion only", "planning phase", "no code yet" modes
- Field is optional - downstream won't break if missing

v2.1 (2026-01-04): Content Preservation Fix
- Added CONTENT_PRESERVATION_DIRECTIVE to prevent "Chinese whispers" content drift
- Added content_verbatim, location, scope_constraints fields to output schema
- Improved instructions for small/cheap LLMs (GPT-5 mini, etc.)
- Better few-shot examples for verbatim extraction

v2.0: Original implementation with incremental weaving support.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.git_utils import get_current_commit
from app.memory import service as memory_service

from app.specs.schema import (
from app.llm._weaver_stream_core_utils import WEAVER_DELTA_FETCH_MULTIPLIER, WEAVER_MAX_OUTPUT_TOKENS, _estimate_tokens, _get_last_consumed_message_id_from_spec, _to_jsonable, build_spec_from_dict, build_weaver_prompt, parse_weaver_response
    SpecConstraints,
    SpecMetadata,
    SpecProvenance,
    SpecRequirements,
    SpecSafety,
    Spec as SpecSchema,
)

try:
    from app.llm.weaver_incremental import format_conversation_for_prompt
    _INCREMENTAL_HELPERS_AVAILABLE = True
except Exception:
    format_conversation_for_prompt = None
    _INCREMENTAL_HELPERS_AVAILABLE = False

logger = logging.getLogger(__name__)
MAX_MESSAGES_FOR_SPEC = int(os.getenv("WEAVER_MAX_MESSAGES", "50"))
MAX_TOKENS_FOR_CONTEXT = int(os.getenv("WEAVER_MAX_CONTEXT_TOKENS", str(WEAVER_MAX_OUTPUT_TOKENS)))


def _is_control_message(role: str, content: str) -> bool:
    c = (content or "").strip()
    rl = (role or "").strip().lower()
    if not c:
        return True
    if rl == "user":
        lc = c.lower()
        if lc.startswith("astra, command:") or lc.startswith("astra command:") or lc.startswith("astra, cmd:"):
            return True
    if rl in ("assistant", "orb"):
        markers = (
            "🧵 weaving spec from conversation",
            "📋 spec created:",
            "📋 spec saved",
            "shall i send this to spec gate",
            "say yes to proceed",
            "⚠️ weak spots to address",
            "provenance",
        )
        lc = c.lower()
        if any(m in lc for m in markers):
            return True
    if rl == "system":
        return True
    return False


@dataclass
class WeaverContext:
    messages: List[Dict[str, Any]]
    message_ids: List[int]
    token_estimate: int
    timestamp_start: Optional[datetime]
    timestamp_end: Optional[datetime]
    commit_hash: Optional[str]


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def gather_weaver_context(
    db: Session,
    project_id: int,
    max_messages: int = MAX_MESSAGES_FOR_SPEC,
    max_tokens: int = MAX_TOKENS_FOR_CONTEXT,
    since_spec_id: Optional[str] = None,
) -> WeaverContext:
    _ = since_spec_id
    messages_raw = memory_service.list_messages(db, project_id, limit=max_messages)
    messages_raw = list(reversed(messages_raw))

    messages: List[Dict[str, Any]] = []
    message_ids: List[int] = []
    total_tokens = 0
    timestamp_start: Optional[datetime] = None
    timestamp_end: Optional[datetime] = None

    for msg in messages_raw:
        role = getattr(msg, "role", "user")
        content = getattr(msg, "content", "") or ""
        if _is_control_message(role, content):
            continue
        tokens = estimate_tokens(content)
        if total_tokens + tokens > max_tokens:
            break
        messages.append({
            "role": role,
            "content": content,
            "id": msg.id,
            "created_at": msg.created_at.isoformat() if msg.created_at else None,
        })
        if msg.id is not None:
            message_ids.append(msg.id)
        total_tokens += tokens
        if msg.created_at:
            if timestamp_start is None or msg.created_at < timestamp_start:
                timestamp_start = msg.created_at
            if timestamp_end is None or msg.created_at > timestamp_end:
                timestamp_end = msg.created_at

    commit_result = get_current_commit()
    commit_hash = commit_result.value if commit_result.success else None

    logger.info("[weaver_context] Gathered %d raw, filtered to %d messages, %d tokens",
                len(messages_raw), len(messages), total_tokens)

    return WeaverContext(
        messages=messages,
        message_ids=message_ids,
        token_estimate=total_tokens,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        commit_hash=commit_hash,
    )


def gather_weaver_delta_context(
    db: Session,
    project_id: int,
    since_message_id: int,
    max_messages: int = MAX_MESSAGES_FOR_SPEC,
    max_tokens: int = MAX_TOKENS_FOR_CONTEXT,
) -> WeaverContext:
    fetch_limit = max(max_messages, max_messages * max(1, WEAVER_DELTA_FETCH_MULTIPLIER))
    messages_raw = memory_service.list_messages(db, project_id, limit=fetch_limit)
    messages_raw = list(reversed(messages_raw))

    messages: List[Dict[str, Any]] = []
    message_ids: List[int] = []
    total_tokens = 0
    timestamp_start: Optional[datetime] = None
    timestamp_end: Optional[datetime] = None

    for msg in messages_raw:
        if msg.id <= since_message_id:
            continue
        role = getattr(msg, "role", "user")
        content = getattr(msg, "content", "") or ""
        if _is_control_message(role, content):
            continue
        tokens = estimate_tokens(content)
        if total_tokens + tokens > max_tokens:
            break
        messages.append({
            "role": role,
            "content": content,
            "id": msg.id,
            "created_at": msg.created_at.isoformat() if msg.created_at else None,
        })
        if msg.id is not None:
            message_ids.append(msg.id)
        total_tokens += tokens
        if msg.created_at:
            if timestamp_start is None or msg.created_at < timestamp_start:
                timestamp_start = msg.created_at
            if timestamp_end is None or msg.created_at > timestamp_end:
                timestamp_end = msg.created_at

    commit_result = get_current_commit()
    commit_hash = commit_result.value if commit_result.success else None

    logger.info("[weaver_delta] Found %d messages after id=%d", len(messages), since_message_id)

    return WeaverContext(
        messages=messages,
        message_ids=message_ids,
        token_estimate=total_tokens,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        commit_hash=commit_hash,
    )


# =============================================================================
# CONTENT PRESERVATION DIRECTIVE (v2.1 - fixes Chinese Whispers)
# =============================================================================

CONTENT_PRESERVATION_DIRECTIVE = """
## CRITICAL: Content Preservation Rules (MUST FOLLOW)

You MUST preserve EXACT content when users specify it. DO NOT summarize, paraphrase, or shorten.

### Rule 1: Verbatim File Content
When user says "write file with content X", "file should say X", "content: X", or "saying X":
- Extract X EXACTLY as written, character-for-character
- Put it in the "content_verbatim" field
- DO NOT simplify, summarize, truncate, or rephrase

Examples:
- User: "write 'Hello world'" → content_verbatim: "Hello world"
- User: "file saying hello" → content_verbatim: "hello"
- User: "content: You cannot go out of scope" → content_verbatim: "You cannot go out of scope"

### Rule 2: Exact Locations (PRESERVE TERMINOLOGY)
When user specifies a location, preserve their EXACT words:
- "Sandbox Desktop" → location: "Sandbox Desktop" (NOT "Desktop")
- "the test folder" → "test" folder (NOT "test directory")
- Include the full path exactly as user specified

### Rule 3: Scope Constraints
When user says "only inside X", "do not touch Y", or similar:
- Put these EXACTLY in the "scope_constraints" array
- Example: "only inside Sandbox Desktop" → scope_constraints: ["Only operate inside Sandbox Desktop"]
"""


def build_weaver_update_prompt(
    previous_spec_core: Dict[str, Any],
    previous_weak_spots: List[str],
    delta_context: WeaverContext,
) -> str:
    instructions = f"""
You are ASTRA Weaver in UPDATE mode.

Given:
1) PREVIOUS SPEC (JSON)
2) PREVIOUS weak spots
3) NEW MESSAGES since last weave

Your job:
- Update spec with new information
- Incorporate answers to weak spots and REMOVE resolved ones
- Add new weak spots only if delta introduces ambiguity

{CONTENT_PRESERVATION_DIRECTIVE}

## UPDATE RULES

1. PRESERVE content_verbatim if set (unless user changes it)
2. PRESERVE location if set (unless user changes it)
3. PRESERVE scope_constraints and ADD new ones
4. UPDATE steps if user provided clarification
5. REMOVE resolved weak_spots
6. ADD new weak_spots if new ambiguities

Intent spec ONLY. Do NOT output scripts or invent details.
Return ONLY the updated JSON object.
""".strip()

    try:
        prev_json = json.dumps(_to_jsonable(previous_spec_core), indent=2, sort_keys=True, ensure_ascii=False)
    except Exception:
        prev_json = json.dumps({"_error": str(previous_spec_core)}, indent=2)

    prev_weak = "\n".join([f"- {w}" for w in (previous_weak_spots or [])]) or "(none)"

    if _INCREMENTAL_HELPERS_AVAILABLE and format_conversation_for_prompt:
        delta_text = format_conversation_for_prompt(delta_context.messages)
    else:
        lines: List[str] = []
        for msg in delta_context.messages:
            lines.append(f"[{msg.get('role', 'user').upper()}] {msg.get('content', '')}")
        delta_text = "\n\n".join(lines)

    return f"""{instructions}

PREVIOUS SPEC:
{prev_json}

PREVIOUS WEAK SPOTS:
{prev_weak}

NEW MESSAGES:
{delta_text}

Output the UPDATED JSON spec."""