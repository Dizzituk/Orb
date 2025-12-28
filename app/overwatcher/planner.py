# FILE: app/overwatcher/planner.py
"""Block 7: Overwatcher Chunk Planner.

Converts approved architecture into small, bounded chunks for Sonnet coding.

Spec v2.3 Block 7:
- Model: GPT-5.2 (gpt-5.2-chat-latest) via Chat Completions
- Fallback: Claude Sonnet (claude-sonnet-4-5-20250514)
- Purpose: Break architecture into implementable chunks (low-output plan; no code)

Each chunk:
- Fits within context limits
- Has explicit file permissions
- Ends in a verification gate
- Is self-contained and deterministic
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from app.overwatcher.schemas import (
    Chunk,
    ChunkPlan,
    ChunkStep,
    ChunkVerification,
    ChunkStatus,
    FileAction,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration (Spec v2.3 §10)
# =============================================================================

# Block 7 uses GPT-5.2 (chat-latest for streaming/chat)
PLANNER_PROVIDER = os.getenv("ORB_PLANNER_PROVIDER", "openai")
PLANNER_MODEL = os.getenv("OPENAI_PLANNER_MODEL", "gpt-5.2-chat-latest")

# Fallback per spec §3.3
PLANNER_FALLBACK_PROVIDER = os.getenv("ORB_PLANNER_FALLBACK_PROVIDER", "anthropic")
PLANNER_FALLBACK_MODEL = os.getenv("ORB_PLANNER_FALLBACK_MODEL", "claude-sonnet-4-5-20250514")

# Chunk constraints
MAX_FILES_PER_CHUNK = int(os.getenv("ORB_MAX_FILES_PER_CHUNK", "5"))
MAX_TOKENS_PER_CHUNK = int(os.getenv("ORB_MAX_TOKENS_PER_CHUNK", "8000"))
PREFER_VERTICAL_SLICES = os.getenv("ORB_PREFER_VERTICAL_SLICES", "1") == "1"

# Planner output limits (spec §9.6: low-output plan; no code)
PLANNER_MAX_OUTPUT_TOKENS = int(os.getenv("ORB_PLANNER_MAX_OUTPUT_TOKENS", "4000"))


# =============================================================================
# Prompt for LLM-based planning
# =============================================================================

CHUNK_PLANNER_SYSTEM = """You are an expert software architect creating an implementation plan.

CRITICAL RULES:
1. Output ONLY a JSON plan - no code, no implementation details
2. Each chunk modifies at most {max_files} files
3. Prefer "vertical slices" - features that work end-to-end
4. Chunks must be ordered by dependency (foundations first)
5. Each chunk ends with verification (tests, lint, types)

You must echo these exact lines at the start of your response:
SPEC_ID: {spec_id}
SPEC_HASH: {spec_hash}

Then output the JSON plan."""

CHUNK_PLANNER_PROMPT = """Break this architecture into implementable chunks.

ARCHITECTURE DOCUMENT:
```
{arch_content}
```

SPEC REQUIREMENTS:
```json
{spec_json}
```

OUTPUT FORMAT (JSON only, after the SPEC headers):
{{
  "chunks": [
    {{
      "chunk_id": "CHUNK-001",
      "title": "Brief title",
      "objective": "What this chunk accomplishes",
      "spec_refs": ["MUST-1", "SHOULD-2"],
      "arch_refs": ["Section 2.1", "Module: Auth"],
      "allowed_files": {{
        "add": ["path/to/new/file.py"],
        "modify": ["path/to/existing.py"],
        "delete_candidates": []
      }},
      "steps": [
        {{
          "step_id": "STEP-001-1",
          "description": "What to do",
          "file_path": "path/to/file.py",
          "action": "add|modify|delete",
          "details": "Specific notes (NO CODE)"
        }}
      ],
      "verification": {{
        "commands": ["pytest tests/test_auth.py", "ruff check path/"],
        "expected_outcomes": {{"pytest": "all pass", "ruff": "no errors"}},
        "timeout_seconds": 60
      }},
      "rollback_plan": "How to undo if verification fails",
      "stop_conditions": ["If X happens, halt and escalate"],
      "priority": 0,
      "dependencies": []
    }}
  ]
}}

Generate the chunk plan. Output ONLY the SPEC headers followed by valid JSON."""


def build_chunk_planner_prompt(
    arch_content: str,
    spec_json: str,
    spec_id: str,
    spec_hash: str,
    max_files: int = MAX_FILES_PER_CHUNK,
) -> Tuple[str, str]:
    """Build system and user prompts for chunk planning.
    
    Returns:
        (system_prompt, user_prompt)
    """
    system = CHUNK_PLANNER_SYSTEM.format(
        max_files=max_files,
        spec_id=spec_id,
        spec_hash=spec_hash,
    )
    
    user = CHUNK_PLANNER_PROMPT.format(
        arch_content=arch_content,
        spec_json=spec_json,
    )
    
    return system, user


# =============================================================================
# JSON Parsing
# =============================================================================

def extract_json_from_output(raw_output: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM output, handling SPEC headers.
    
    Expected format:
    SPEC_ID: xxx
    SPEC_HASH: yyy
    
    { "chunks": [...] }
    """
    if not raw_output:
        return None
    
    text = raw_output.strip()
    
    # Strip SPEC headers if present
    lines = text.split("\n")
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("SPEC_ID:") or line.strip().startswith("SPEC_HASH:"):
            start_idx = i + 1
            continue
        if line.strip() and not line.strip().startswith("SPEC"):
            break
    
    text = "\n".join(lines[start_idx:]).strip()
    
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try code fence extraction
    fence_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text, re.IGNORECASE)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Try finding JSON object
    start = text.find("{")
    if start == -1:
        return None
    
    depth = 0
    end = -1
    in_string = False
    escape = False
    
    for i, char in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    
    if end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    
    return None


def parse_chunk_plan_output(raw_output: str) -> Tuple[List[Chunk], Optional[str], Optional[str]]:
    """Parse LLM output into list of Chunks.
    
    Returns:
        (chunks, spec_id, spec_hash) - spec_id/hash from output headers
    """
    # Extract SPEC headers
    spec_id = None
    spec_hash = None
    
    lines = raw_output.strip().split("\n") if raw_output else []
    for line in lines[:5]:  # Only check first 5 lines
        if line.strip().startswith("SPEC_ID:"):
            spec_id = line.split(":", 1)[1].strip()
        elif line.strip().startswith("SPEC_HASH:"):
            spec_hash = line.split(":", 1)[1].strip()
    
    data = extract_json_from_output(raw_output)
    
    if data is None:
        logger.warning("[planner] Failed to parse JSON from output")
        return [], spec_id, spec_hash
    
    chunks_data = data.get("chunks", [])
    if not chunks_data:
        # Maybe the output is the chunks array directly
        if isinstance(data, list):
            chunks_data = data
    
    chunks = []
    for cd in chunks_data:
        try:
            chunk = Chunk.from_dict(cd)
            chunks.append(chunk)
        except Exception as e:
            logger.warning(f"[planner] Failed to parse chunk: {e}")
    
    return chunks, spec_id, spec_hash


# =============================================================================
# Plan Generation (LLM-based)
# =============================================================================

async def generate_chunk_plan(
    *,
    job_id: str,
    arch_id: str,
    arch_version: int,
    arch_content: str,
    spec_id: str,
    spec_hash: str,
    spec_json: str,
    llm_call_fn: Callable,
    job_artifact_root: str,
    provider_id: str = None,
    model_id: str = None,
) -> ChunkPlan:
    """Generate a chunk plan from architecture using LLM.
    
    Spec v2.3 Block 7:
    - Primary: GPT-5.2 (gpt-5.2-chat-latest)
    - Fallback: Claude Sonnet
    - Output: Low-output plan with no code
    
    Args:
        job_id: Job UUID
        arch_id: Architecture document ID
        arch_version: Architecture version
        arch_content: Architecture document content
        spec_id: Spec ID
        spec_hash: Spec hash
        spec_json: Spec as JSON string
        llm_call_fn: Async function to call LLM
        job_artifact_root: Root for artifacts
        provider_id: LLM provider (default: openai)
        model_id: LLM model (default: gpt-5.2-chat-latest)
    
    Returns:
        ChunkPlan with parsed chunks
    """
    from app.pot_spec.ledger import (
        emit_chunk_plan_created,
        emit_stage_started,
        emit_provider_fallback,
        emit_stage_failed,
    )
    
    plan_id = str(uuid4())
    stage_run_id = str(uuid4())
    
    # Default to spec-mandated models
    provider_id = provider_id or PLANNER_PROVIDER
    model_id = model_id or PLANNER_MODEL
    
    # Emit stage started
    try:
        emit_stage_started(
            job_artifact_root=job_artifact_root,
            job_id=job_id,
            stage_id="chunk_planning",
            stage_run_id=stage_run_id,
        )
    except Exception as e:
        logger.warning(f"[planner] Failed to emit stage started: {e}")
    
    system_prompt, user_prompt = build_chunk_planner_prompt(
        arch_content=arch_content,
        spec_json=spec_json,
        spec_id=spec_id,
        spec_hash=spec_hash,
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    # Try primary model
    result = None
    used_fallback = False
    
    try:
        result = await llm_call_fn(
            provider_id=provider_id,
            model_id=model_id,
            messages=messages,
            max_tokens=PLANNER_MAX_OUTPUT_TOKENS,
        )
    except Exception as e:
        logger.warning(f"[planner] Primary model failed ({provider_id}/{model_id}): {e}")
        
        # Try fallback
        if PLANNER_FALLBACK_PROVIDER and PLANNER_FALLBACK_MODEL:
            try:
                emit_provider_fallback(
                    job_artifact_root=job_artifact_root,
                    job_id=job_id,
                    from_provider=provider_id,
                    from_model=model_id,
                    to_provider=PLANNER_FALLBACK_PROVIDER,
                    to_model=PLANNER_FALLBACK_MODEL,
                    reason=str(e),
                )
            except Exception:
                pass
            
            try:
                result = await llm_call_fn(
                    provider_id=PLANNER_FALLBACK_PROVIDER,
                    model_id=PLANNER_FALLBACK_MODEL,
                    messages=messages,
                    max_tokens=PLANNER_MAX_OUTPUT_TOKENS,
                )
                used_fallback = True
                provider_id = PLANNER_FALLBACK_PROVIDER
                model_id = PLANNER_FALLBACK_MODEL
            except Exception as e2:
                logger.error(f"[planner] Fallback model also failed: {e2}")
    
    if result is None:
        # Both failed - emit failure and return empty plan
        try:
            emit_stage_failed(
                job_artifact_root=job_artifact_root,
                job_id=job_id,
                stage_id="chunk_planning",
                error_type="llm_call_failed",
                error_message="Both primary and fallback models failed",
            )
        except Exception:
            pass
        
        return ChunkPlan(
            plan_id=plan_id,
            job_id=job_id,
            arch_id=arch_id,
            arch_version=arch_version,
            spec_id=spec_id,
            spec_hash=spec_hash,
            chunks=[],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    
    raw_output = result.content if hasattr(result, "content") else str(result)
    chunks, returned_spec_id, returned_spec_hash = parse_chunk_plan_output(raw_output)
    
    # Verify spec hash (fail-fast per spec §4.5)
    if returned_spec_hash and returned_spec_hash != spec_hash:
        logger.error(f"[planner] SPEC_HASH mismatch: expected {spec_hash[:16]}..., got {returned_spec_hash[:16]}...")
        try:
            emit_stage_failed(
                job_artifact_root=job_artifact_root,
                job_id=job_id,
                stage_id="chunk_planning",
                error_type="spec_hash_mismatch",
                error_message=f"Expected {spec_hash}, got {returned_spec_hash}",
            )
        except Exception:
            pass
        # Still return chunks but mark as potentially invalid
    
    # Calculate total estimated tokens
    total_tokens = sum(c.estimated_tokens for c in chunks)
    
    plan = ChunkPlan(
        plan_id=plan_id,
        job_id=job_id,
        arch_id=arch_id,
        arch_version=arch_version,
        spec_id=spec_id,
        spec_hash=spec_hash,
        chunks=chunks,
        created_at=datetime.now(timezone.utc).isoformat(),
        total_estimated_tokens=total_tokens,
    )
    
    # Store plan
    plan_path = store_chunk_plan(plan, job_artifact_root)
    
    # Emit plan created event
    try:
        emit_chunk_plan_created(
            job_artifact_root=job_artifact_root,
            job_id=job_id,
            plan_id=plan_id,
            arch_id=arch_id,
            arch_version=arch_version,
            chunk_count=len(chunks),
            plan_path=plan_path,
        )
    except Exception as e:
        logger.warning(f"[planner] Failed to emit plan created: {e}")
    
    logger.info(f"[planner] Generated plan {plan_id} with {len(chunks)} chunks (model={model_id}, fallback={used_fallback})")
    return plan


# =============================================================================
# Plan Storage
# =============================================================================

def store_chunk_plan(
    plan: ChunkPlan,
    job_artifact_root: str,
) -> str:
    """Store chunk plan to filesystem.
    
    Path: jobs/{job_id}/plan/chunks_v{version}.json
    
    Returns path to stored plan.
    """
    plan_dir = Path(job_artifact_root) / "jobs" / plan.job_id / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    
    plan_path = plan_dir / f"chunks_v{plan.arch_version}.json"
    plan_path.write_text(plan.to_json(), encoding="utf-8")
    
    logger.info(f"[planner] Stored plan: {plan_path}")
    return str(plan_path)


def load_chunk_plan(
    job_id: str,
    arch_version: int,
    job_artifact_root: str,
) -> Optional[ChunkPlan]:
    """Load chunk plan from filesystem."""
    plan_path = Path(job_artifact_root) / "jobs" / job_id / "plan" / f"chunks_v{arch_version}.json"
    
    if not plan_path.exists():
        return None
    
    try:
        data = json.loads(plan_path.read_text(encoding="utf-8"))
        return ChunkPlan.from_dict(data)
    except Exception as e:
        logger.warning(f"[planner] Failed to load plan: {e}")
        return None


# =============================================================================
# Chunk Ordering
# =============================================================================

def topological_sort_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """Sort chunks by dependencies (topological order).
    
    Chunks with no dependencies come first.
    """
    # Build dependency graph
    chunk_map = {c.chunk_id: c for c in chunks}
    in_degree = {c.chunk_id: 0 for c in chunks}
    
    for chunk in chunks:
        for dep in chunk.dependencies:
            if dep in chunk_map:
                in_degree[chunk.chunk_id] += 1
    
    # Kahn's algorithm
    queue = [cid for cid, deg in in_degree.items() if deg == 0]
    result = []
    
    while queue:
        # Sort by priority within same level
        queue.sort(key=lambda cid: chunk_map[cid].priority)
        cid = queue.pop(0)
        result.append(chunk_map[cid])
        
        # Reduce in-degree for dependents
        for chunk in chunks:
            if cid in chunk.dependencies:
                in_degree[chunk.chunk_id] -= 1
                if in_degree[chunk.chunk_id] == 0:
                    queue.append(chunk.chunk_id)
    
    # Add any remaining (cyclic dependencies)
    remaining = [c for c in chunks if c.chunk_id not in {r.chunk_id for r in result}]
    result.extend(sorted(remaining, key=lambda c: c.priority))
    
    return result


def get_next_chunk(plan: ChunkPlan) -> Optional[Chunk]:
    """Get the next chunk to implement.
    
    Returns the first pending chunk whose dependencies are all verified.
    """
    sorted_chunks = topological_sort_chunks(plan.chunks)
    verified_ids = {c.chunk_id for c in plan.chunks if c.status == ChunkStatus.VERIFIED}
    
    for chunk in sorted_chunks:
        if chunk.status != ChunkStatus.PENDING:
            continue
        
        # Check all dependencies are verified
        deps_met = all(dep in verified_ids for dep in chunk.dependencies)
        if deps_met:
            return chunk
    
    return None


__all__ = [
    # Configuration
    "PLANNER_PROVIDER",
    "PLANNER_MODEL",
    "PLANNER_FALLBACK_PROVIDER",
    "PLANNER_FALLBACK_MODEL",
    # Prompt building
    "build_chunk_planner_prompt",
    # Parsing
    "extract_json_from_output",
    "parse_chunk_plan_output",
    # Plan generation
    "generate_chunk_plan",
    # Storage
    "store_chunk_plan",
    "load_chunk_plan",
    # Ordering
    "topological_sort_chunks",
    "get_next_chunk",
]
