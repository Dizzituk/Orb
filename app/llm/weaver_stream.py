# FILE: app/llm/weaver_stream.py
"""
Weaver Stream Handler for ASTRA.

Weaves conversation/ramble into a coherent candidate spec.
Uses a non-frontier model configured via environment (WEAVER_PROVIDER/WEAVER_MODEL) for spec generation.

Flow:
1. Pull recent conversation from message history
2. Call LLM to build structured spec JSON
3. Validate and parse spec
4. Store in database
5. Stream markdown summary to user
6. Register flow state for Spec Gate routing

INVARIANT: Weaver uses a non-frontier model configured via WEAVER_PROVIDER/WEAVER_MODEL.
INVARIANT: Every spec stores source_message_ids for reproducibility.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from sqlalchemy.orm import Session

from app.git_utils import get_current_commit
from app.llm.audit_logger import RoutingTrace
from app.memory import service as memory_service
from app.specs import (
    SpecConstraints,
    SpecMetadata,
    SpecProvenance,
    SpecRequirements,
    SpecSafety,
    SpecSchema,
    create_spec,
    spec_to_markdown,
    validate_spec,
)

# Flow state management (optional)
try:
    from app.llm.spec_flow_state import start_weaver_flow

    _FLOW_STATE_AVAILABLE = True
except ImportError:
    start_weaver_flow = None
    _FLOW_STATE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Weaver model configuration (ENV-driven; no hard-coded model IDs)
# These MUST be provided via environment; if missing, Weaver will emit an error event.
WEAVER_PROVIDER = os.getenv("WEAVER_PROVIDER")  # e.g. "google", "openai", "anthropic"
WEAVER_MODEL = os.getenv("WEAVER_MODEL")        # e.g. "gemini-2.0-flash", "gpt-5.2"
WEAVER_MAX_OUTPUT_TOKENS = int(os.getenv("WEAVER_MAX_OUTPUT_TOKENS", "15000"))
WEAVER_TIMEOUT_SECONDS = int(os.getenv("WEAVER_TIMEOUT_SECONDS", "60"))

# Context limits
MAX_MESSAGES_FOR_SPEC = 50
MAX_TOKENS_FOR_CONTEXT = WEAVER_MAX_OUTPUT_TOKENS  # rough estimate; keep context <= max output


@dataclass
class WeaverContext:
    """Context gathered for spec building."""
    messages: List[Dict[str, Any]]
    message_ids: List[int]
    token_estimate: int
    timestamp_start: Optional[datetime]
    timestamp_end: Optional[datetime]
    commit_hash: Optional[str]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def gather_weaver_context(
    db: Session,
    project_id: int,
    max_messages: int = MAX_MESSAGES_FOR_SPEC,
    max_tokens: int = MAX_TOKENS_FOR_CONTEXT,
    since_spec_id: Optional[str] = None,  # kept for future, currently unused
) -> WeaverContext:
    """
    Gather conversation context for spec building.

    Strategy:
    1. Get recent messages up to max_messages.
    2. Respect a rough token budget (max_tokens).
    3. Include message IDs & timestamps for provenance.
    """
    # Use memory_service API (don't touch ORM models directly)
    messages_raw = memory_service.list_messages(db, project_id, limit=max_messages)

    # Reverse to chronological order (oldest first)
    messages_raw = list(reversed(messages_raw))

    messages: List[Dict[str, Any]] = []
    message_ids: List[int] = []
    total_tokens = 0
    timestamp_start: Optional[datetime] = None
    timestamp_end: Optional[datetime] = None

    for msg in messages_raw:
        content = msg.content or ""
        tokens = _estimate_tokens(content)

        if total_tokens + tokens > max_tokens:
            break

        messages.append(
            {
                "role": msg.role,
                "content": content,
                "id": msg.id,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
            }
        )
        message_ids.append(msg.id)
        total_tokens += tokens

        if msg.created_at:
            if timestamp_start is None or msg.created_at < timestamp_start:
                timestamp_start = msg.created_at
            if timestamp_end is None or msg.created_at > timestamp_end:
                timestamp_end = msg.created_at

    # Get commit hash
    commit_result = get_current_commit()
    commit_hash = commit_result.value if commit_result.success else None

    return WeaverContext(
        messages=messages,
        message_ids=message_ids,
        token_estimate=total_tokens,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        commit_hash=commit_hash,
    )


def build_weaver_prompt(context: WeaverContext) -> str:
    """
    Build the prompt for Weaver.

    The prompt instructs the model to:
    - Analyze the recent conversation
    - Extract project/system requirements
    - Produce a structured spec JSON according to SpecSchema shape.
    """
    instructions = """
You are ASTRA Weaver, an expert specification-weaving assistant.

Your job:
- Take the recent conversation between user and assistant
- Infer the *current* project or feature they are working on
- Synthesize a structured, *implementable* specification JSON

Output format:
Return **ONLY** a single JSON object with the following shape:

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

Rules:
- DO NOT wrap the JSON in backticks.
- DO NOT add commentary before or after the JSON.
- Only include information actually implied by the conversation.
- If something is unclear or missing, set a sensible default and mention it in weak_spots.
"""

    conversation_text_lines: List[str] = []
    for msg in context.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        conversation_text_lines.append(f"[{role.upper()}] {content}")

    conversation_text = "\n\n".join(conversation_text_lines)

    prompt = f"""{instructions.strip()}

Below is the recent conversation:

{conversation_text}

Now, produce the JSON spec as described above."""
    return prompt


def parse_weaver_response(
    response_text: str,
) -> tuple[Optional[Dict[str, Any]], str]:
    """
    Parse Weaver LLM response into spec JSON and a free-form summary (if any).

    Returns:
        (spec_dict, summary_text) or (None, error_message)
    """
    response_text = response_text.strip()

    # Try fenced ```json block first
    json_block_start = response_text.find("```json")
    if json_block_start != -1:
        json_block_end = response_text.find("```", json_block_start + 7)
        if json_block_end != -1:
            json_str = response_text[json_block_start + 7 : json_block_end].strip()
        else:
            return None, "Could not find closing ``` for JSON block"
    else:
        # Fallback: raw JSON from first '{' to last '}'
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

    # Summary: everything after the JSON block (if present)
    summary_text = ""
    summary_marker = "**Summary:**"
    idx = response_text.find(summary_marker)
    if idx != -1:
        summary_text = response_text[idx + len(summary_marker) :].strip()
    else:
        # Generic "Summary:" fallback
        summary_marker = "Summary:"
        idx = response_text.find(summary_marker)
        if idx != -1:
            summary_text = response_text[idx + len(summary_marker) :].strip()

    return spec_dict, summary_text


def build_spec_from_dict(
    spec_dict: Dict[str, Any],
    context: WeaverContext,
    project_id: int,
    conversation_id: Optional[str] = None,
) -> SpecSchema:
    """Build a SpecSchema from parsed dict and context."""

    provenance = SpecProvenance(
        conversation_id=conversation_id,
        source_message_ids=context.message_ids,
        commit_hash=context.commit_hash,
        generator_model=f"weaver-v1-{WEAVER_MODEL}",
        token_count=context.token_estimate,
        timestamp_start=context.timestamp_start.isoformat()
        if context.timestamp_start
        else None,
        timestamp_end=context.timestamp_end.isoformat()
        if context.timestamp_end
        else None,
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

    spec = SpecSchema(
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

    return spec


async def generate_weaver_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
    conversation_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Generate SSE stream for Weaver spec building.

    1. Gathers conversation context
    2. Calls LLM to build spec
    3. Parses and validates spec
    4. Stores in database
    5. Streams markdown summary to user
    """
    try:
        # Ensure provider/model are configured
        if not WEAVER_PROVIDER or not WEAVER_MODEL:
            err = "WEAVER_PROVIDER/WEAVER_MODEL must be set in environment for Weaver to run"
            logger.error("[weaver] %s", err)
            yield "data: " + json.dumps({"type": "error", "error": err}) + "\n\n"
            return

        # Intro
        yield "data: " + json.dumps(
            {
                "type": "token",
                "content": "🧵 **Weaving spec from conversation...**\n\n",
            }
        ) + "\n\n"
        await asyncio.sleep(0.01)

        # Gather context
        yield "data: " + json.dumps(
            {"type": "token", "content": "📋 Gathering conversation context...\n"}
        ) + "\n\n"
        context = gather_weaver_context(db, project_id)

        if not context.messages:
            msg = (
                "I couldn't find any recent conversation to weave into a spec.\n\n"
                "Please describe your project or paste your notes first, "
                "then ask me to weave them into a spec.\n"
            )
            yield "data: " + json.dumps({"type": "token", "content": msg}) + "\n\n"
            yield "data: " + json.dumps(
                {
                    "type": "done",
                    "provider": "weaver",
                    "model": "context_check",
                }
            ) + "\n\n"
            return

        yield "data: " + json.dumps(
            {
                "type": "token",
                "content": f"Found {len(context.messages)} messages ({context.token_estimate} tokens) in recent context.\n\n",
            }
        ) + "\n\n"

        # Build prompt
        prompt = build_weaver_prompt(context)

        # Call LLM
        yield "data: " + json.dumps(
            {"type": "token", "content": "🤖 Generating spec structure...\n\n"}
        ) + "\n\n"

        from app.llm.streaming import stream_llm

        full_response = ""
        async for chunk in stream_llm(
            provider=WEAVER_PROVIDER,
            model=WEAVER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are the ASTRA Weaver, a precise spec-building assistant.",
        ):
            if isinstance(chunk, dict):
                if chunk.get("type") == "token":
                    content = chunk.get("text", "") or chunk.get("content", "")
                else:
                    content = ""
            else:
                content = str(chunk)

            if not content:
                continue

            full_response += content
            # We deliberately do not stream partial JSON to the user.

        # Parse response
        spec_dict, summary_text = parse_weaver_response(full_response)
        if spec_dict is None:
            error_msg = summary_text or "Failed to parse Weaver response"
            logger.warning("[weaver] Failed to parse spec: %s", error_msg)
            yield "data: " + json.dumps(
                {
                    "type": "token",
                    "content": f"\n⚠️ Failed to parse spec: {error_msg}\n",
                }
            ) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "done", "provider": "weaver", "model": WEAVER_MODEL}
            ) + "\n\n"
            if trace:
                trace.finalize(success=False, error_message=error_msg)
            return

        # Build spec schema
        spec_schema = build_spec_from_dict(
            spec_dict, context, project_id, conversation_id
        )

        # Validate
        validation = validate_spec(spec_schema)

        # Store in database
        yield "data: " + json.dumps(
            {"type": "token", "content": "💾 Saving spec...\n\n"}
        ) + "\n\n"
        create_spec(
            db, project_id, spec_schema, generator_model=f"weaver-v1-{WEAVER_MODEL}"
        )

        # Generate markdown for display
        markdown = spec_to_markdown(spec_schema)

        # Stream the formatted output
        yield "data: " + json.dumps(
            {"type": "token", "content": "---\n\n"}
        ) + "\n\n"

        chunk_size = 100
        for i in range(0, len(markdown), chunk_size):
            chunk = markdown[i : i + chunk_size]
            yield "data: " + json.dumps(
                {"type": "token", "content": chunk}
            ) + "\n\n"
            await asyncio.sleep(0.01)

        yield "data: " + json.dumps(
            {"type": "token", "content": "\n\n---\n\n"}
        ) + "\n\n"

        # Add summary and weak spots
        if summary_text:
            yield "data: " + json.dumps(
                {"type": "token", "content": "## Weaver Notes\n\n"}
            ) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "token", "content": summary_text + "\n\n"}
            ) + "\n\n"

        weak_spots = spec_dict.get("weak_spots", []) or []
        if weak_spots:
            yield "data: " + json.dumps(
                {"type": "token", "content": "### ⚠️ Weak Spots to Address\n\n"}
            ) + "\n\n"
            for spot in weak_spots:
                yield "data: " + json.dumps(
                    {"type": "token", "content": f"- {spot}\n"}
                ) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "token", "content": "\n"}
            ) + "\n\n"

        # Validation warnings (if your validate_spec returns them)
        if hasattr(validation, "warnings") and validation.warnings:
            yield "data: " + json.dumps(
                {"type": "token", "content": "### 📋 Validation Notes\n\n"}
            ) + "\n\n"
            for warning in validation.warnings:
                yield "data: " + json.dumps(
                    {"type": "token", "content": f"- {warning}\n"}
                ) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "token", "content": "\n"}
            ) + "\n\n"

        # Next steps - prompt for Spec Gate
        yield "data: " + json.dumps(
            {"type": "token", "content": "---\n\n"}
        ) + "\n\n"
        yield "data: " + json.dumps(
            {
                "type": "token",
                "content": f"📋 **Spec saved** (ID: `{spec_schema.spec_id}`)\n\n",
            }
        ) + "\n\n"
        yield "data: " + json.dumps(
            {
                "type": "token",
                "content": "**Shall I send this to Spec Gate for validation?**\n\n",
            }
        ) + "\n\n"
        yield "data: " + json.dumps(
            {
                "type": "token",
                "content": "_Say **Yes** to proceed, or refine the spec first._\n",
            }
        ) + "\n\n"

        # Save assistant + user messages to memory
        from app.memory import schemas as memory_schemas

        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="user",
                content=message,
                provider="local",
            ),
        )
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content=(
                    f"📋 Spec created: {spec_schema.title}\n\n"
                    f"Spec ID: {spec_schema.spec_id}\n\n"
                    "Shall I send this to Spec Gate for validation?"
                ),
                provider=WEAVER_PROVIDER,
                model=f"weaver-{WEAVER_MODEL}",
            ),
        )

        # Register flow state for command flow continuity
        if _FLOW_STATE_AVAILABLE and start_weaver_flow:
            try:
                start_weaver_flow(project_id, spec_schema.spec_id)
                logger.debug(
                    "[weaver] Registered flow state for project %s, spec %s",
                    project_id,
                    spec_schema.spec_id,
                )
            except Exception as e:
                logger.warning("[weaver] Failed to register flow state: %s", e)

        if trace:
            trace.finalize(success=True)

        yield "data: " + json.dumps(
            {
                "type": "done",
                "provider": WEAVER_PROVIDER,
                "model": f"weaver-{WEAVER_MODEL}",
                "spec_id": spec_schema.spec_id,
            }
        ) + "\n\n"

    except Exception as e:
        logger.exception("[weaver] Stream failed: %s", e)
        if trace:
            try:
                trace.finalize(success=False, error_message=str(e))
            except Exception:
                # Best-effort only; don't crash Weaver on trace issues
                pass
        yield "data: " + json.dumps({"type": "error", "error": str(e)}) + "\n\n"
