# FILE: app/memory/summary_generator.py
"""
Generates and updates conversation summaries using a cheap/fast LLM.

Strategy:
  1. Load existing summary (if any) + unsummarised messages.
  2. Build a structured prompt asking the model to merge/update.
  3. Call the configured SUMMARY_MODEL (default: gemini-2.5-flash-lite).
  4. Parse the JSON response into SummaryContent schema.
  5. Persist as a new ConversationSummary version.

The generator runs as a background task (asyncio.create_task) — it must
never block the chat response. Failures are logged and swallowed.

Part of CONV-MEMORY-001: Phase 2 (Summary Generation Service).
"""

import json
import logging
import os
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session as DbSession

from app.memory.conversation_models import ConversationSession
from app.memory.conversation_schemas import SummaryContent, SummaryCreate
from app.memory.conversation_service import (
    get_latest_summary,
    create_summary,
)

logger = logging.getLogger(__name__)

# =========================================================================
# Configuration
# =========================================================================

def _get_summary_model() -> tuple[str, str]:
    """
    Return (provider, model) for summary generation.

    Reads from env vars with cheap defaults:
      SUMMARY_PROVIDER = google
      SUMMARY_MODEL    = gemini-2.5-flash-lite
    """
    provider = os.getenv("SUMMARY_PROVIDER", "google")
    model = os.getenv("SUMMARY_MODEL", "gemini-2.5-flash-lite")
    return provider, model


# =========================================================================
# Prompt construction
# =========================================================================

_SUMMARY_SYSTEM_PROMPT = """\
You are a conversation summariser. Your job is to produce a structured JSON \
summary of a conversation between a user and an AI assistant.

RULES:
- Output ONLY valid JSON — no markdown fences, no preamble, no explanation.
- Preserve factual accuracy. Do not invent information.
- Keep the summary concise — under 800 words total across all fields.
- If an existing summary is provided, UPDATE it — merge new information, \
  replace outdated items, keep items that are still relevant.
- Lists should contain at most 10 items. Drop the oldest/least relevant if needed.

OUTPUT SCHEMA:
{
  "current_topic": "What the conversation is currently about",
  "key_decisions": ["Decision 1", "Decision 2"],
  "open_items": ["Unresolved question or task"],
  "technical_context": "Languages, frameworks, files, architecture discussed",
  "attachments_described": ["filename.ext — brief description of what it contained"],
  "user_preferences_expressed": ["Preference or style instruction"],
  "biographical_facts": ["Durable personal facts the user stated: DOB, birthplace, current location, family, job changes, relationships, nationality, contact details. ONLY include if the user explicitly stated it in this session. Quote the user's phrasing where possible so downstream extraction has a clean signal."],
  "models_involved": ["provider:model identifiers that participated"],
  "message_range": {"from_id": <int>, "to_id": <int>}
}

BIOGRAPHICAL PRESERVATION RULE:
  If the user says anything like "I was born in X", "I live in Y", "my
  name is Z", "I moved to A", "my birthday is B", "I work as a C",
  preserve it in biographical_facts VERBATIM or near-verbatim. Do not
  compress these into topic summaries — they are hard facts that later
  stages depend on.
"""


def _build_user_prompt(
    existing_summary: Optional[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    from_id: int,
    to_id: int,
) -> str:
    """
    Build the user prompt that includes existing summary + new messages.
    """
    parts = []

    if existing_summary:
        parts.append("EXISTING SUMMARY (update this):")
        parts.append(json.dumps(existing_summary, indent=2))
        parts.append("")

    parts.append(f"NEW MESSAGES (IDs {from_id}–{to_id}):")
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        msg_id = msg.get("id", "?")
        # Truncate very long messages to keep prompt manageable
        if len(content) > 2000:
            content = content[:1800] + "\n[...TRUNCATED...]"
        parts.append(f"[{role}] (#{msg_id}): {content}")

    parts.append("")
    parts.append(
        "Produce an updated JSON summary covering the ENTIRE conversation "
        "so far (existing summary + new messages). Output ONLY the JSON."
    )

    return "\n".join(parts)


# =========================================================================
# Response parsing
# =========================================================================

def _parse_summary_response(raw: str) -> SummaryContent:
    """
    Parse LLM response into SummaryContent.

    Handles common LLM quirks:
      - Markdown fences (```json ... ```)
      - Leading/trailing whitespace
      - Partial JSON (returns defaults for missing fields)
    """
    cleaned = raw.strip()

    # Strip markdown fences
    if cleaned.startswith("```"):
        # Remove first line (```json) and last line (```)
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(
            "[summary_gen] Failed to parse summary JSON: %s — raw[:200]: %s",
            e, cleaned[:200],
        )
        # Return a minimal valid summary rather than failing entirely
        return SummaryContent(
            current_topic="[summary parse error — will retry on next trigger]",
        )

    return SummaryContent(**{
        k: v for k, v in data.items()
        if k in SummaryContent.model_fields
    })


def _estimate_tokens(summary: SummaryContent) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    text = json.dumps(summary.model_dump(), ensure_ascii=False)
    return len(text) // 4


# =========================================================================
# Core generation
# =========================================================================

async def generate_summary(
    db: DbSession,
    session: ConversationSession,
) -> Optional[SummaryContent]:
    """
    Generate or update the conversation summary for a session.

    This is the main entry point called by the background task.

    Returns the SummaryContent on success, None on failure.
    """
    from app.memory.models import Message

    # 1. Load existing summary
    latest = get_latest_summary(db, session.id)
    existing_json = latest.summary_json if latest else None

    # 2. Load unsummarised messages
    query = (
        db.query(Message)
        .filter(Message.session_id == session.id)
        .order_by(Message.id.asc())
    )
    if latest and latest.summarised_through_message_id:
        query = query.filter(
            Message.id > latest.summarised_through_message_id,
        )

    unsummarised = query.all()

    if not unsummarised:
        logger.debug(
            "[summary_gen] No unsummarised messages for session %d",
            session.id,
        )
        return None

    # Convert to dicts for prompt
    msg_dicts = [
        {
            "id": msg.id,
            "role": msg.role,
            "content": msg.content or "",
        }
        for msg in unsummarised
    ]

    from_id = unsummarised[0].id
    to_id = unsummarised[-1].id

    # 3. Build prompt
    user_prompt = _build_user_prompt(existing_json, msg_dicts, from_id, to_id)

    # 4. Call LLM
    provider, model = _get_summary_model()
    logger.info(
        "[summary_gen] Generating summary for session %d "
        "(%d new messages, %s/%s)",
        session.id, len(unsummarised), provider, model,
    )

    try:
        from app.llm._streaming_utils_3 import call_llm_text
        raw_response = await call_llm_text(
            provider=provider,
            model=model,
            system_prompt=_SUMMARY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=2000,
            timeout_seconds=30,
        )
    except Exception as e:
        logger.error(
            "[summary_gen] LLM call failed for session %d: %s",
            session.id, e,
        )
        return None

    # 5. Parse response
    summary_content = _parse_summary_response(raw_response)

    # Ensure message_range is set
    summary_content.message_range = {"from_id": from_id, "to_id": to_id}

    # Merge models_involved from existing + any new models
    if existing_json and "models_involved" in existing_json:
        existing_models = set(existing_json["models_involved"])
        new_models = set(summary_content.models_involved)
        summary_content.models_involved = sorted(
            existing_models | new_models
        )

    # 6. Persist
    token_est = _estimate_tokens(summary_content)

    create_summary(
        db,
        SummaryCreate(
            session_id=session.id,
            project_id=session.project_id,
            summary_json=summary_content.model_dump(),
            summarised_through_message_id=to_id,
            token_estimate=token_est,
        ),
    )

    logger.info(
        "[summary_gen] Summary v%s created for session %d "
        "(through msg %d, ~%d tokens)",
        "new", session.id, to_id, token_est,
    )

    return summary_content
