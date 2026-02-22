from __future__ import annotations
import json
import logging
import os
from app.specs.schema import SpecConstraints, SpecMetadata, SpecProvenance, SpecRequirements, SpecSafety, SpecSchema
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
_INCREMENTAL_HELPERS_AVAILABLE = True
format_conversation_for_prompt = None
logger = logging.getLogger(__name__)


WEAVER_MAX_OUTPUT_TOKENS = int(os.getenv("WEAVER_MAX_OUTPUT_TOKENS", "15000"))

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

def _estimate_tokens(text: str) -> int:
    from .weaver_stream_core import estimate_tokens
    return estimate_tokens(text)

def build_weaver_prompt(context: WeaverContext) -> str:
    from .weaver_stream_core import CONTENT_PRESERVATION_DIRECTIVE, WeaverContext
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
  "execution_mode": "Pipeline control mode if specified (e.g., 'Discussion only', 'No coding yet', 'Planning phase'), or null",
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
    spec_dict.setdefault("execution_mode", None)  # v2.2: Bug 4 fix - backward compatible
    
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
    # v2.2: Log execution_mode for debugging
    if spec_dict.get("execution_mode"):
        logger.info("[weaver_core] ✓ execution_mode: '%s'", spec_dict["execution_mode"])

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
    from .weaver_stream_core import WeaverContext
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
    # v2.2: Bug 4 fix - execution_mode for pipeline control
    meta_data["execution_mode"] = spec_dict.get("execution_mode")

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
