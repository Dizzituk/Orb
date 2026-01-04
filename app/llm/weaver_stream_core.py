# FILE: app/llm/weaver_stream_core.py
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

# IMPORTANT:
# Use direct module import paths (not package re-exports) to avoid import-time failures.
from app.specs.schema import (  # type: ignore
    SpecConstraints,
    SpecMetadata,
    SpecProvenance,
    SpecRequirements,
    SpecSafety,
    Spec as SpecSchema,
)

# Optional: incremental helpers (safe, dependency-light)
try:
    from app.llm.weaver_incremental import format_conversation_for_prompt

    _INCREMENTAL_HELPERS_AVAILABLE = True
except Exception:
    format_conversation_for_prompt = None  # type: ignore
    _INCREMENTAL_HELPERS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Env-driven context limits (kept consistent with weaver_stream.py defaults)
WEAVER_MAX_OUTPUT_TOKENS = int(os.getenv("WEAVER_MAX_OUTPUT_TOKENS", "15000"))
MAX_MESSAGES_FOR_SPEC = int(os.getenv("WEAVER_MAX_MESSAGES", "50"))
MAX_TOKENS_FOR_CONTEXT = int(os.getenv("WEAVER_MAX_CONTEXT_TOKENS", str(WEAVER_MAX_OUTPUT_TOKENS)))
WEAVER_DELTA_FETCH_MULTIPLIER = int(os.getenv("WEAVER_DELTA_FETCH_MULTIPLIER", "4"))


def _to_jsonable(obj: Any) -> Any:
    """
    Convert nested objects into JSON-serializable primitives.

    Defensive: previous spec cores can include Pydantic models (e.g., SpecConstraints),
    dataclasses, datetimes, enums, etc.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, datetime):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]

    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))

    # pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return _to_jsonable(obj.model_dump())  # type: ignore[attr-defined]
        except Exception:
            pass

    # pydantic v1
    if hasattr(obj, "dict"):
        try:
            return _to_jsonable(obj.dict())  # type: ignore[attr-defined]
        except Exception:
            pass

    # enum-like
    if hasattr(obj, "value"):
        try:
            return _to_jsonable(obj.value)  # type: ignore[attr-defined]
        except Exception:
            pass

    # last resort: public __dict__
    if hasattr(obj, "__dict__"):
        try:
            data = {k: v for k, v in vars(obj).items() if not str(k).startswith("_")}
            return _to_jsonable(data)
        except Exception:
            pass

    return str(obj)


def _is_control_message(role: str, content: str) -> bool:
    """
    Control-plane messages must NOT be woven into specs.

    This is the key fix for "Found 20 messages" even when nothing new was typed:
    every weaver trigger and weaver output is stored in memory and was being re-ingested.

    We filter:
    - the user trigger command lines
    - weaver's own assistant "spec created/saved" outputs (common markers)
    """
    c = (content or "").strip()
    rl = (role or "").strip().lower()

    if not c:
        return True  # ignore empty noise

    # user triggers
    if rl == "user":
        lc = c.lower()
        if lc.startswith("astra, command:") or lc.startswith("astra command:") or lc.startswith("astra, cmd:"):
            return True

    # assistant weaver outputs (high-signal markers; keeps real assistant chat intact)
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

    # system messages should never be part of weaving context
    if rl == "system":
        return True

    return False


def _get_last_consumed_message_id_from_spec(db_spec: Any) -> Optional[int]:
    """
    Extract the last consumed message ID from a spec's stored metadata.
    
    Checks (in order):
    1. content_json["metadata"]["weaver_last_consumed_message_id"]
    2. Fallback to max(source_message_ids) if available
    
    Returns None if no checkpoint data exists.
    """
    try:
        # Try content_json metadata first
        if hasattr(db_spec, "content_json") and db_spec.content_json:
            if isinstance(db_spec.content_json, dict):
                metadata = db_spec.content_json.get("metadata", {})
                if metadata and "weaver_last_consumed_message_id" in metadata:
                    val = metadata["weaver_last_consumed_message_id"]
                    if isinstance(val, int):
                        return val
                    if isinstance(val, str) and val.isdigit():
                        return int(val)
        
        # Fallback: use max source_message_id from provenance
        if hasattr(db_spec, "source_message_ids") and db_spec.source_message_ids:
            ids = [int(x) for x in db_spec.source_message_ids if x]
            if ids:
                return max(ids)
        
        return None
    except Exception as e:
        logger.warning("[weaver_core] Failed to extract last_consumed_message_id: %s", e)
        return None


@dataclass
class WeaverContext:
    messages: List[Dict[str, Any]]
    message_ids: List[int]
    token_estimate: int
    timestamp_start: Optional[datetime]
    timestamp_end: Optional[datetime]
    commit_hash: Optional[str]


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# Backwards-compat alias (older code used _estimate_tokens)
def _estimate_tokens(text: str) -> int:
    return estimate_tokens(text)


def gather_weaver_context(
    db: Session,
    project_id: int,
    max_messages: int = MAX_MESSAGES_FOR_SPEC,
    max_tokens: int = MAX_TOKENS_FOR_CONTEXT,
    since_spec_id: Optional[str] = None,  # backwards-compat; unused
) -> WeaverContext:
    _ = since_spec_id

    messages_raw = memory_service.list_messages(db, project_id, limit=max_messages)
    messages_raw = list(reversed(messages_raw))  # chronological

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

        messages.append(
            {
                "role": role,
                "content": content,
                "id": msg.id,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
            }
        )
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

    logger.info("[weaver_context] Gathered %d raw messages from DB", len(messages_raw))
    logger.info("[weaver_context] Filtered to %d non-control messages", len(messages))
    logger.info("[weaver_context] Message IDs: %s", message_ids)
    logger.info("[weaver_context] Total tokens: %d", total_tokens)
    
    if len(messages) == 0:
        logger.warning("[weaver_context] ⚠️ NO MESSAGES after filtering! This will produce empty prompt.")
        logger.warning("[weaver_context] Raw message count was %d - all were filtered as control messages", len(messages_raw))

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
    # Over-fetch then filter to ensure we can find enough "real" delta messages
    fetch_limit = max(max_messages, max_messages * max(1, WEAVER_DELTA_FETCH_MULTIPLIER))
    messages_raw = memory_service.list_messages(db, project_id, limit=fetch_limit)
    messages_raw = list(reversed(messages_raw))  # chronological

    messages: List[Dict[str, Any]] = []
    message_ids: List[int] = []
    total_tokens = 0
    timestamp_start: Optional[datetime] = None
    timestamp_end: Optional[datetime] = None

    for msg in messages_raw:
        # Only consider messages AFTER the checkpoint
        if msg.id <= since_message_id:
            continue

        role = getattr(msg, "role", "user")
        content = getattr(msg, "content", "") or ""

        if _is_control_message(role, content):
            continue

        tokens = estimate_tokens(content)
        if total_tokens + tokens > max_tokens:
            break

        messages.append(
            {
                "role": role,
                "content": content,
                "id": msg.id,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
            }
        )
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

    logger.info("[weaver_delta] Fetched %d raw messages from DB (limit=%d)", len(messages_raw), fetch_limit)
    logger.info("[weaver_delta] Looking for messages AFTER message_id=%d", since_message_id)
    logger.info("[weaver_delta] Found %d messages after filtering control messages", len(messages))
    logger.info("[weaver_delta] Delta message IDs: %s", message_ids)
    logger.info("[weaver_delta] Total tokens: %d", total_tokens)
    
    if len(messages) == 0:
        logger.info("[weaver_delta] No new messages since checkpoint - this triggers spec reuse")

    return WeaverContext(
        messages=messages,
        message_ids=message_ids,
        token_estimate=total_tokens,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        commit_hash=commit_hash,
    )


def build_weaver_prompt(context: WeaverContext) -> str:
    instructions = """
You are ASTRA Weaver.

Your task: synthesize a clear, structured Point-of-Truth spec from the conversation below.

Output a JSON object with this schema:

{
  "title": "...",
  "summary": "...",
  "objective": "...",
  "requirements": {
    "functional": ["..."],
    "non_functional": ["..."]
  },
  "constraints": {
    "budget": "... or null",
    "latency": "... or null",
    "platform": "... or null",
    "integrations": ["..."],
    "compliance": ["..."]
  },
  "safety": {
    "risks": ["..."],
    "mitigations": ["..."],
    "runtime_guards": ["..."]
  },
  "acceptance_criteria": ["..."],
  "dependencies": ["..."],
  "non_goals": ["..."],
  "metadata": {
    "priority": "low|medium|high",
    "owner": "... or null",
    "tags": ["..."]
  },
  "weak_spots": ["areas that need clarification or are ambiguous"]
}

Conciseness rules (token discipline):
- Keep each list short (prefer <= 8 bullets per list).
- Avoid cross-platform variations unless the USER explicitly requests multi-OS support.
- Do not include prose essays. Use crisp bullet statements.

Quality rules:
- Requirements MUST be concrete and testable.
- Acceptance criteria MUST be verifiable end-to-end checks.
- Only include information implied by the conversation.
- If something is unclear/missing, do NOT assume; put it in weak_spots.

Formatting rules:
- DO NOT wrap the JSON in backticks.
- DO NOT add commentary before or after the JSON.
""".strip()

    if _INCREMENTAL_HELPERS_AVAILABLE and format_conversation_for_prompt:
        conversation_text = format_conversation_for_prompt(context.messages)
    else:
        lines: List[str] = []
        for msg in context.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"[{role.upper()}] {content}")
        conversation_text = "\n\n".join(lines)

    return f"""{instructions}

Below is the recent conversation:

{conversation_text}

Now, produce the JSON spec as described above."""


def build_weaver_update_prompt(
    previous_spec_core: Dict[str, Any],
    previous_weak_spots: List[str],
    delta_context: WeaverContext,
) -> str:
    instructions = """
You are ASTRA Weaver in UPDATE mode.

You will be given:
1) The PREVIOUS SPEC (JSON core fields)
2) The PREVIOUS weak spots list
3) Only the NEW MESSAGES since the last weave (delta)

Your job:
- Update the spec to reflect any new information from the delta messages.
- If the delta messages answer prior weak spots, incorporate the answers into the spec and REMOVE those weak spots.
- Add NEW weak spots only if the delta introduces new ambiguity.

CRITICAL RULES (scope + anti-drift):
- Intent spec ONLY (what/where/constraints/how-to-verify). Do NOT output scripts or OS command blocks.
- Do NOT invent filenames, directories, or side effects that are not explicitly requested.
- If file vs folder is ambiguous, keep it as a weak spot; do not guess.
- Ignore assistant meta/disclaimer text unless the USER explicitly adopts it.

Conciseness rules:
- Keep lists short (prefer <= 8 bullets per list).
- Prefer small, incremental edits; preserve existing spec details unless contradicted or clarified.

Output format:
Return ONLY the updated JSON object with the same shape as the previous spec schema.
Do not add extra wrappers or commentary.
""".strip()

    try:
        prev_json = json.dumps(_to_jsonable(previous_spec_core), indent=2, sort_keys=True, ensure_ascii=False)
    except Exception:
        prev_json = json.dumps(
            {"_unserializable_previous_spec_core": str(previous_spec_core)},
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )

    prev_weak = "\n".join([f"- {w}" for w in (previous_weak_spots or [])]) or "(none)"

    if _INCREMENTAL_HELPERS_AVAILABLE and format_conversation_for_prompt:
        delta_text = format_conversation_for_prompt(delta_context.messages)
    else:
        lines: List[str] = []
        for msg in delta_context.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"[{role.upper()}] {content}")
        delta_text = "\n\n".join(lines)

    return f"""{instructions}

PREVIOUS SPEC (core JSON fields):
{prev_json}

PREVIOUS WEAK SPOTS:
{prev_weak}

NEW MESSAGES SINCE LAST WEAVE:
{delta_text}

Now, output the UPDATED JSON spec (same schema)."""


def parse_weaver_response(response_text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    response_text = (response_text or "").strip()

    json_block_start = response_text.find("```json")
    if json_block_start != -1:
        json_block_end = response_text.find("```", json_block_start + 7)
        if json_block_end != -1:
            json_str = response_text[json_block_start + 7 : json_block_end].strip()
        else:
            return None, "Could not find closing ``` for JSON block"
    else:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start == -1 or json_end <= json_start:
            return None, "Could not find JSON in response"
        json_str = response_text[json_start:json_end].strip()

    try:
        spec_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"

    if not isinstance(spec_dict, dict):
        return None, "Top-level JSON must be an object"

    summary_text = ""
    for marker in ("**Summary:**", "Summary:"):
        idx = response_text.find(marker)
        if idx != -1:
            summary_text = response_text[idx + len(marker) :].strip()
            break

    return spec_dict, summary_text


def build_spec_from_dict(
    spec_dict: Dict[str, Any],
    context: WeaverContext,
    project_id: Optional[int] = None,
    conversation_id: Optional[str] = None,
    generator_model: Optional[str] = None,
) -> SpecSchema:
    """
    Backwards compatible builder.

    Older weaver_stream.py calls: build_spec_from_dict(spec_dict, context, project_id, conversation_id)
    Newer callers may pass generator_model explicitly.
    """
    _ = project_id  # SpecSchema doesn't require project_id, but keep signature compatible.

    if not generator_model:
        generator_model = os.getenv("WEAVER_MODEL") or "weaver"

    provenance = SpecProvenance(
        conversation_id=conversation_id,
        source_message_ids=context.message_ids,
        commit_hash=context.commit_hash,
        generator_model=str(generator_model),
        token_count=context.token_estimate,
        timestamp_start=context.timestamp_start.isoformat() if context.timestamp_start else None,
        timestamp_end=context.timestamp_end.isoformat() if context.timestamp_end else None,
    )

    req_data = spec_dict.get("requirements", {}) or {}
    requirements = SpecRequirements(
        functional=req_data.get("functional", []) or [],
        non_functional=req_data.get("non_functional", []) or [],
    )

    con_data = spec_dict.get("constraints", {}) or {}
    constraints = SpecConstraints(
        budget=con_data.get("budget"),
        latency=con_data.get("latency"),
        platform=con_data.get("platform"),
        integrations=con_data.get("integrations", []) or [],
        compliance=con_data.get("compliance", []) or [],
    )

    safety_data = spec_dict.get("safety", {}) or {}
    safety = SpecSafety(
        risks=safety_data.get("risks", []) or [],
        mitigations=safety_data.get("mitigations", []) or [],
        runtime_guards=safety_data.get("runtime_guards", []) or [],
    )

    meta_data = spec_dict.get("metadata", {}) or {}
    metadata = SpecMetadata(
        priority=meta_data.get("priority", "medium"),
        owner=meta_data.get("owner"),
        tags=meta_data.get("tags", []) or [],
    )

    return SpecSchema(
        title=spec_dict.get("title", "Untitled Spec"),
        summary=spec_dict.get("summary", ""),
        objective=spec_dict.get("objective", ""),
        requirements=requirements,
        constraints=constraints,
        safety=safety,
        acceptance_criteria=spec_dict.get("acceptance_criteria", []) or [],
        dependencies=spec_dict.get("dependencies", []) or [],
        non_goals=spec_dict.get("non_goals", []) or [],
        metadata=metadata,
        provenance=provenance,
    )