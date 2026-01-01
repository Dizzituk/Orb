# FILE: app/llm/weaver_stream.py
"""
Weaver Stream Handler for ASTRA.

Weaves conversation/ramble into a coherent candidate spec.
Uses GPT-5.2 latest (NOT a frontier model) for spec generation.

Flow:
1. Pull recent conversation from message history
2. Call LLM to build structured spec JSON
3. Validate and parse spec
4. Store in database
5. Stream markdown summary to user

INVARIANT: Weaver uses GPT-5.2 latest, NOT frontier models.
INVARIANT: Every spec stores source_message_ids for reproducibility.
"""
from __future__ import annotations
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, AsyncGenerator
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.memory import service as memory_service
from app.git_utils import get_current_commit
from app.specs import (
    SpecSchema,
    SpecProvenance,
    SpecRequirements,
    SpecConstraints,
    SpecSafety,
    SpecMetadata,
    create_spec,
    validate_spec,
    spec_to_markdown,
    get_latest_draft_spec,
)
from app.llm.audit_logger import RoutingTrace

logger = logging.getLogger(__name__)

# Weaver model configuration
WEAVER_PROVIDER = "openai"
WEAVER_MODEL = "gpt-5.2"  # GPT-5.2 latest, NOT frontier

# Context limits
MAX_MESSAGES_FOR_SPEC = 50  # Max messages to include
MAX_TOKENS_FOR_CONTEXT = 8000  # Rough token limit for context


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
    """Rough token estimate (4 chars per token)."""
    return len(text) // 4


def gather_weaver_context(
    db: Session,
    project_id: int,
    max_messages: int = MAX_MESSAGES_FOR_SPEC,
    max_tokens: int = MAX_TOKENS_FOR_CONTEXT,
    since_spec_id: Optional[str] = None,
) -> WeaverContext:
    """
    Gather conversation context for spec building.
    
    Strategy:
    1. If since_spec_id provided, get messages since that spec was created
    2. Else, get recent messages up to max_messages
    3. Respect token limit
    4. Include message IDs for provenance
    """
    # Get messages
    messages_raw = memory_service.list_messages(db, project_id, limit=max_messages)
    
    # Reverse to chronological order (oldest first)
    messages_raw = list(reversed(messages_raw))
    
    # Filter and build context
    messages = []
    message_ids = []
    total_tokens = 0
    timestamp_start = None
    timestamp_end = None
    
    for msg in messages_raw:
        content = msg.content or ""
        tokens = _estimate_tokens(content)
        
        if total_tokens + tokens > max_tokens:
            break
        
        messages.append({
            "role": msg.role,
            "content": content,
            "id": msg.id,
            "created_at": msg.created_at.isoformat() if msg.created_at else None,
        })
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
    """Build the prompt for the Weaver LLM."""
    
    # Format conversation
    conversation_text = ""
    for msg in context.messages:
        role = msg["role"].upper()
        content = msg["content"]
        conversation_text += f"[{role}]: {content}\n\n"
    
    prompt = f"""You are the ASTRA Weaver. Your job is to consolidate a conversation into a structured spec.

Review the following conversation and extract a coherent specification.

<conversation>
{conversation_text}
</conversation>

Based on this conversation, create a structured spec in JSON format. Extract:
- The main objective (what the user wants to achieve)
- Functional requirements (features/behaviors)
- Non-functional requirements (performance, UX, etc.)
- Constraints (budget, platform, integrations)
- Safety considerations (risks, mitigations)
- Acceptance criteria (how to know it's done)

Output a JSON object with this structure:
{{
    "title": "Brief descriptive title",
    "summary": "2-3 sentence summary of the spec",
    "objective": "Clear statement of what the user wants",
    "requirements": {{
        "functional": ["requirement 1", "requirement 2"],
        "non_functional": ["performance requirement", "UX requirement"]
    }},
    "constraints": {{
        "budget": "budget constraint or null",
        "latency": "latency requirement or null",
        "platform": "target platform or null",
        "integrations": ["external system 1"],
        "compliance": []
    }},
    "safety": {{
        "risks": ["identified risk 1"],
        "mitigations": ["mitigation for risk 1"],
        "runtime_guards": []
    }},
    "acceptance_criteria": ["criterion 1", "criterion 2"],
    "inputs": [],
    "outputs": [],
    "steps": [
        {{"id": "1", "description": "Step description", "dependencies": [], "notes": null}}
    ],
    "dependencies": [],
    "non_goals": ["explicitly out of scope item"],
    "metadata": {{
        "priority": "medium",
        "owner": null,
        "tags": []
    }}
}}

Rules:
1. Only include information that was actually discussed in the conversation
2. If something wasn't mentioned, leave it empty or null
3. Be specific - use the user's own words where appropriate
4. Flag obvious gaps or contradictions in a "weak_spots" field (array of strings)
5. The JSON must be valid and parseable

After the JSON, provide a brief natural language summary pointing out:
- What the spec covers
- Any obvious weak spots or things the user should clarify
- Suggested next steps

Output format:
```json
<your spec JSON here>
```

**Summary:**
<your natural language summary here>"""
    
    return prompt


def parse_weaver_response(response_text: str) -> tuple[Optional[Dict[str, Any]], str]:
    """
    Parse Weaver LLM response into spec JSON and summary.
    
    Returns:
        (spec_dict, summary_text) or (None, error_message)
    """
    # Try to extract JSON from markdown code block
    json_start = response_text.find("```json")
    json_end = response_text.find("```", json_start + 7) if json_start >= 0 else -1
    
    if json_start >= 0 and json_end > json_start:
        json_str = response_text[json_start + 7:json_end].strip()
    else:
        # Try to find raw JSON
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
        else:
            return None, "Could not find JSON in response"
    
    # Parse JSON
    try:
        spec_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    
    # Extract summary (everything after JSON block)
    summary_start = response_text.find("**Summary:**")
    if summary_start < 0:
        summary_start = response_text.find("Summary:")
    if summary_start < 0:
        summary_start = json_end + 3 if json_end > 0 else 0
    
    summary_text = response_text[summary_start:].strip()
    if summary_text.startswith("**Summary:**"):
        summary_text = summary_text[12:].strip()
    elif summary_text.startswith("Summary:"):
        summary_text = summary_text[8:].strip()
    
    return spec_dict, summary_text


def build_spec_from_dict(
    spec_dict: Dict[str, Any],
    context: WeaverContext,
    project_id: int,
    conversation_id: Optional[str] = None,
) -> SpecSchema:
    """Build a SpecSchema from parsed dict and context."""
    
    # Build provenance
    provenance = SpecProvenance(
        conversation_id=conversation_id,
        source_message_ids=context.message_ids,
        commit_hash=context.commit_hash,
        generator_model=f"weaver-v1-{WEAVER_MODEL}",
        token_count=context.token_estimate,
        timestamp_start=context.timestamp_start.isoformat() if context.timestamp_start else None,
        timestamp_end=context.timestamp_end.isoformat() if context.timestamp_end else None,
    )
    
    # Build requirements
    req_data = spec_dict.get("requirements", {})
    requirements = SpecRequirements(
        functional=req_data.get("functional", []),
        non_functional=req_data.get("non_functional", []),
    )
    
    # Build constraints
    con_data = spec_dict.get("constraints", {})
    constraints = SpecConstraints(
        budget=con_data.get("budget"),
        latency=con_data.get("latency"),
        platform=con_data.get("platform"),
        integrations=con_data.get("integrations", []),
        compliance=con_data.get("compliance", []),
    )
    
    # Build safety
    safety_data = spec_dict.get("safety", {})
    safety = SpecSafety(
        risks=safety_data.get("risks", []),
        mitigations=safety_data.get("mitigations", []),
        runtime_guards=safety_data.get("runtime_guards", []),
    )
    
    # Build metadata
    meta_data = spec_dict.get("metadata", {})
    metadata = SpecMetadata(
        priority=meta_data.get("priority", "medium"),
        owner=meta_data.get("owner"),
        tags=meta_data.get("tags", []),
    )
    
    # Build full spec
    spec = SpecSchema(
        title=spec_dict.get("title", "Untitled Spec"),
        summary=spec_dict.get("summary", ""),
        objective=spec_dict.get("objective", ""),
        requirements=requirements,
        constraints=constraints,
        safety=safety,
        acceptance_criteria=spec_dict.get("acceptance_criteria", []),
        dependencies=spec_dict.get("dependencies", []),
        non_goals=spec_dict.get("non_goals", []),
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
        yield "data: " + json.dumps({"type": "token", "content": "🧵 **Weaving spec from conversation...**\n\n"}) + "\n\n"
        await asyncio.sleep(0.01)
        
        # Gather context
        yield "data: " + json.dumps({"type": "token", "content": "📋 Gathering conversation context...\n"}) + "\n\n"
        context = gather_weaver_context(db, project_id)
        
        if not context.messages:
            yield "data: " + json.dumps({"type": "token", "content": "\n⚠️ No conversation history found. Please discuss your requirements first, then ask me to weave them into a spec.\n"}) + "\n\n"
            yield "data: " + json.dumps({"type": "done", "provider": "weaver", "model": "context_check"}) + "\n\n"
            return
        
        yield "data: " + json.dumps({"type": "token", "content": f"Found {len(context.messages)} messages ({context.token_estimate} tokens)\n\n"}) + "\n\n"
        
        # Build prompt
        prompt = build_weaver_prompt(context)
        
        # Call LLM
        yield "data: " + json.dumps({"type": "token", "content": "🤖 Generating spec structure...\n\n"}) + "\n\n"
        
        from app.llm.streaming import stream_llm
        
        full_response = ""
        async for chunk in stream_llm(
            provider=WEAVER_PROVIDER,
            model=WEAVER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are the ASTRA Weaver, a precise spec-building assistant.",
        ):
            if isinstance(chunk, dict):
                content = chunk.get("text", "") or chunk.get("content", "")
            else:
                content = str(chunk)
            
            if content:
                full_response += content
                # Don't stream raw JSON to user - we'll format it nicely
        
        # Parse response
        spec_dict, summary_text = parse_weaver_response(full_response)
        
        if spec_dict is None:
            yield "data: " + json.dumps({"type": "token", "content": f"\n⚠️ Failed to parse spec: {summary_text}\n"}) + "\n\n"
            yield "data: " + json.dumps({"type": "done", "provider": "weaver", "model": WEAVER_MODEL}) + "\n\n"
            return
        
        # Build spec schema
        spec_schema = build_spec_from_dict(spec_dict, context, project_id, conversation_id)
        
        # Validate
        validation = validate_spec(spec_schema)
        
        # Store in database
        yield "data: " + json.dumps({"type": "token", "content": "💾 Saving spec...\n\n"}) + "\n\n"
        db_spec = create_spec(db, project_id, spec_schema, generator_model=f"weaver-v1-{WEAVER_MODEL}")
        
        # Generate markdown for display
        markdown = spec_to_markdown(spec_schema)
        
        # Stream the formatted output
        yield "data: " + json.dumps({"type": "token", "content": "---\n\n"}) + "\n\n"
        
        # Stream markdown in chunks
        chunk_size = 100
        for i in range(0, len(markdown), chunk_size):
            chunk = markdown[i:i + chunk_size]
            yield "data: " + json.dumps({"type": "token", "content": chunk}) + "\n\n"
            await asyncio.sleep(0.01)
        
        yield "data: " + json.dumps({"type": "token", "content": "\n\n---\n\n"}) + "\n\n"
        
        # Add summary and weak spots
        if summary_text:
            yield "data: " + json.dumps({"type": "token", "content": "## Weaver Notes\n\n"}) + "\n\n"
            yield "data: " + json.dumps({"type": "token", "content": summary_text + "\n\n"}) + "\n\n"
        
        # Weak spots from LLM
        weak_spots = spec_dict.get("weak_spots", [])
        if weak_spots:
            yield "data: " + json.dumps({"type": "token", "content": "### ⚠️ Weak Spots to Address\n\n"}) + "\n\n"
            for spot in weak_spots:
                yield "data: " + json.dumps({"type": "token", "content": f"- {spot}\n"}) + "\n\n"
            yield "data: " + json.dumps({"type": "token", "content": "\n"}) + "\n\n"
        
        # Validation warnings
        if validation.warnings:
            yield "data: " + json.dumps({"type": "token", "content": "### 📋 Validation Notes\n\n"}) + "\n\n"
            for warning in validation.warnings:
                yield "data: " + json.dumps({"type": "token", "content": f"- {warning}\n"}) + "\n\n"
            yield "data: " + json.dumps({"type": "token", "content": "\n"}) + "\n\n"
        
        # Next steps - prompt for Spec Gate
        yield "data: " + json.dumps({"type": "token", "content": "---\n\n"}) + "\n\n"
        yield "data: " + json.dumps({"type": "token", "content": f"📋 **Spec saved** (ID: `{spec_schema.spec_id}`)\n\n"}) + "\n\n"
        yield "data: " + json.dumps({"type": "token", "content": "**Shall I send this to Spec Gate for validation?**\n\n"}) + "\n\n"
        yield "data: " + json.dumps({"type": "token", "content": "_Say **Yes** to proceed, or refine the spec first._\n"}) + "\n\n"
        
        # Save assistant message
        response_content = f"📋 Spec created: {spec_schema.title}\n\nSpec ID: {spec_schema.spec_id}\n\nShall I send this to Spec Gate for validation?"
        
        from app.memory import schemas as memory_schemas
        memory_service.create_message(db, memory_schemas.MessageCreate(
            project_id=project_id, role="user", content=message, provider="local"
        ))
        memory_service.create_message(db, memory_schemas.MessageCreate(
            project_id=project_id, role="assistant", content=response_content,
            provider=WEAVER_PROVIDER, model=f"weaver-{WEAVER_MODEL}"
        ))
        
        if trace:
            trace.finalize(success=True)
        
        yield "data: " + json.dumps({
            "type": "done",
            "provider": WEAVER_PROVIDER,
            "model": f"weaver-{WEAVER_MODEL}",
            "spec_id": spec_schema.spec_id,
        }) + "\n\n"
        
    except Exception as e:
        logger.exception("[weaver] Stream failed: %s", e)
        if trace:
            trace.finalize(success=False, error_message=str(e))
        yield "data: " + json.dumps({"type": "error", "error": str(e)}) + "\n\n"
