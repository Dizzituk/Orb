# FILE: app/llm/weaver_stream_core.py
"""
Weaver Stream Core - Prompt building and response parsing for ASTRA Weaver.

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

WEAVER_MAX_OUTPUT_TOKENS = int(os.getenv("WEAVER_MAX_OUTPUT_TOKENS", "15000"))
MAX_MESSAGES_FOR_SPEC = int(os.getenv("WEAVER_MAX_MESSAGES", "50"))
MAX_TOKENS_FOR_CONTEXT = int(os.getenv("WEAVER_MAX_CONTEXT_TOKENS", str(WEAVER_MAX_OUTPUT_TOKENS)))
WEAVER_DELTA_FETCH_MULTIPLIER = int(os.getenv("WEAVER_DELTA_FETCH_MULTIPLIER", "4"))


def _to_jsonable(obj: Any) -> Any:
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
    if hasattr(obj, "model_dump"):
        try:
            return _to_jsonable(obj.model_dump())
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return _to_jsonable(obj.dict())
        except Exception:
            pass
    if hasattr(obj, "value"):
        try:
            return _to_jsonable(obj.value)
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            data = {k: v for k, v in vars(obj).items() if not str(k).startswith("_")}
            return _to_jsonable(data)
        except Exception:
            pass
    return str(obj)


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


def _get_last_consumed_message_id_from_spec(db_spec: Any) -> Optional[int]:
    try:
        if hasattr(db_spec, "content_json") and db_spec.content_json:
            if isinstance(db_spec.content_json, dict):
                metadata = db_spec.content_json.get("metadata", {})
                if metadata and "weaver_last_consumed_message_id" in metadata:
                    val = metadata["weaver_last_consumed_message_id"]
                    if isinstance(val, int):
                        return val
                    if isinstance(val, str) and val.isdigit():
                        return int(val)
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
    if not text:
        return 0
    return max(1, len(text) // 4)


def _estimate_tokens(text: str) -> int:
    return estimate_tokens(text)


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


def build_weaver_prompt(context: WeaverContext) -> str:
    instructions = f"""
You are ASTRA Weaver.

Your task: Extract a structured specification from the conversation below.

{CONTENT_PRESERVATION_DIRECTIVE}

## Output Format (JSON)

Return a JSON object with this schema:

{{
  "title": "Short descriptive title (max 10 words)",
  "summary": "One sentence describing what to do",
  "objective": "Detailed description of the goal",
  "content_verbatim": "EXACT file content if user specified (copy character-for-character), or null",
  "location": "EXACT path/location as user specified, or null",
  "scope_constraints": ["List of boundaries - what CAN and CANNOT be touched"],
  "outputs": [
    {{"name": "artifact name", "path": "exact/path", "description": "what it is"}}
  ],
  "steps": [
    "S1: First concrete action",
    "S2: Second concrete action",
    "S3: Third concrete action",
    "S4: Verification step"
  ],
  "requirements": {{
    "functional": ["What the system must do"],
    "non_functional": ["Performance, security, etc."]
  }},
  "constraints": {{
    "budget": null,
    "latency": null,
    "platform": null,
    "integrations": [],
    "compliance": []
  }},
  "safety": {{
    "risks": [],
    "mitigations": [],
    "runtime_guards": []
  }},
  "acceptance_criteria": ["How to verify success - must be testable"],
  "dependencies": [],
  "non_goals": [],
  "metadata": {{
    "priority": "medium",
    "owner": null,
    "tags": []
  }},
  "weak_spots": ["Areas needing clarification"]
}}

## Few-Shot Examples

### Example 1: Simple file creation
Conversation:
[USER] Find the test folder on Sandbox Desktop and write a file inside saying hello

Correct output:
{{
  "title": "Write hello file to test folder on Sandbox Desktop",
  "content_verbatim": "hello",
  "location": "Sandbox Desktop/test",
  "scope_constraints": ["Only operate inside Sandbox Desktop", "Only write to test folder"],
  "outputs": [{{"name": "text file", "path": "Sandbox Desktop/test/", "description": "file containing hello"}}],
  "steps": [
    "S1: Locate test folder on Sandbox Desktop",
    "S2: Create text file inside test folder",
    "S3: Write exact content 'hello' to the file",
    "S4: Verify file exists with correct content"
  ],
  "acceptance_criteria": ["File exists in Sandbox Desktop/test", "File content is exactly 'hello'"],
  "weak_spots": ["Exact filename not specified"]
}}

## Critical Rules

1. content_verbatim: EXACT words if user specified file content (HIGHEST PRIORITY)
2. location: EXACT path/location terminology from user
3. steps: Minimum 3-4 concrete steps numbered S1, S2, S3...
4. outputs: At least 1 artifact if creating/modifying something
5. acceptance_criteria: At least 1 testable criterion
6. If unclear, add to weak_spots (do NOT guess)

DO NOT wrap JSON in backticks. Return ONLY the JSON object.
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

## Conversation to Analyze

{conversation_text}

Now produce the JSON spec. CRITICAL: content_verbatim must be EXACTLY what user said."""


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

    # Ensure required fields with defaults
    spec_dict.setdefault("steps", [])
    spec_dict.setdefault("weak_spots", [])
    spec_dict.setdefault("scope_constraints", [])
    spec_dict.setdefault("outputs", [])
    spec_dict.setdefault("acceptance_criteria", [])
    
    # CRITICAL: Ensure output file info is in acceptance_criteria
    # (acceptance_criteria survives DB serialization, outputs may not)
    outputs = spec_dict.get("outputs", [])
    content_verbatim = spec_dict.get("content_verbatim", "")
    location = spec_dict.get("location", "")
    
    if outputs:
        for out in outputs:
            name = out.get("name", "") if isinstance(out, dict) else str(out)
            if name:
                # Add an acceptance criterion that describes this output
                criterion = f"Output file '{name}'"
                if location:
                    criterion += f" at {location}"
                if content_verbatim:
                    criterion += f" contains: {content_verbatim[:100]}"
                # Avoid duplicates
                if criterion not in spec_dict["acceptance_criteria"]:
                    spec_dict["acceptance_criteria"].append(criterion)
    elif content_verbatim and location:
        # No outputs but we have content and location - synthesize acceptance criterion
        criterion = f"File at {location} contains exactly: {content_verbatim}"
        if criterion not in spec_dict["acceptance_criteria"]:
            spec_dict["acceptance_criteria"].append(criterion)

    # Log content preservation for debugging
    if spec_dict.get("content_verbatim"):
        logger.info("[weaver_core] ✓ content_verbatim: '%s'", spec_dict["content_verbatim"][:80])
    if spec_dict.get("location"):
        logger.info("[weaver_core] ✓ location: '%s'", spec_dict["location"])

    summary_text = ""
    for marker in ("**Summary:**", "Summary:"):
        idx = response_text.find(marker)
        if idx != -1:
            summary_text = response_text[idx + len(marker):].strip()
            break

    return spec_dict, summary_text


def build_spec_from_dict(
    spec_dict: Dict[str, Any],
    context: WeaverContext,
    project_id: Optional[int] = None,
    conversation_id: Optional[str] = None,
    generator_model: Optional[str] = None,
) -> SpecSchema:
    _ = project_id

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
    # v2.1: Include content preservation fields in metadata
    meta_data["content_verbatim"] = spec_dict.get("content_verbatim")
    meta_data["location"] = spec_dict.get("location")
    meta_data["scope_constraints"] = spec_dict.get("scope_constraints", [])
    meta_data["outputs"] = spec_dict.get("outputs", [])
    meta_data["steps"] = spec_dict.get("steps", [])
    meta_data["weak_spots"] = spec_dict.get("weak_spots", [])

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