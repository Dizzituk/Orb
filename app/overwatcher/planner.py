# FILE: app/overwatcher/planner.py
"""Block 7: Overwatcher Chunk Planner.

Converts approved architecture into small, bounded chunks for Sonnet coding.

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
from typing import Any, Dict, List, Optional, Tuple
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

# Configuration
MAX_FILES_PER_CHUNK = int(os.getenv("ORB_MAX_FILES_PER_CHUNK", "5"))
MAX_TOKENS_PER_CHUNK = int(os.getenv("ORB_MAX_TOKENS_PER_CHUNK", "8000"))
PREFER_VERTICAL_SLICES = os.getenv("ORB_PREFER_VERTICAL_SLICES", "1") == "1"


# =============================================================================
# Prompt for LLM-based planning
# =============================================================================

CHUNK_PLANNER_PROMPT = """You are an expert software architect creating an implementation plan.

Given an approved architecture document, break it into small, bounded implementation chunks.

RULES:
1. Each chunk should modify at most {max_files} files
2. Prefer "vertical slices" - features that work end-to-end over scattered refactors
3. Each chunk MUST end with verification (tests pass, lint clean, types check)
4. Chunks should be ordered by dependency (foundations first)
5. Each chunk must be self-contained enough to implement in isolation

OUTPUT FORMAT (JSON):
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
          "details": "Specific implementation notes"
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
      "dependencies": ["CHUNK-000"]
    }}
  ]
}}

ARCHITECTURE DOCUMENT:
```
{arch_content}
```

SPEC REQUIREMENTS:
```json
{spec_json}
```

Generate the chunk plan. Output ONLY valid JSON."""


def build_chunk_planner_prompt(
    arch_content: str,
    spec_json: str,
    max_files: int = MAX_FILES_PER_CHUNK,
) -> str:
    """Build the prompt for chunk planning."""
    return CHUNK_PLANNER_PROMPT.format(
        max_files=max_files,
        arch_content=arch_content,
        spec_json=spec_json,
    )


# =============================================================================
# JSON Parsing
# =============================================================================

def extract_json_from_output(raw_output: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM output."""
    if not raw_output:
        return None
    
    text = raw_output.strip()
    
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


def parse_chunk_plan_output(raw_output: str) -> List[Chunk]:
    """Parse LLM output into list of Chunks."""
    data = extract_json_from_output(raw_output)
    
    if data is None:
        logger.warning("[planner] Failed to parse JSON from output")
        return []
    
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
    
    return chunks


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
    llm_call_fn,  # async callable for LLM
    provider_id: str = "anthropic",
    model_id: str = "claude-sonnet-4-20250514",
) -> ChunkPlan:
    """Generate a chunk plan from architecture using LLM.
    
    Args:
        job_id: Job UUID
        arch_id: Architecture document ID
        arch_version: Architecture version
        arch_content: Architecture document content
        spec_id: Spec ID
        spec_hash: Spec hash
        spec_json: Spec as JSON string
        llm_call_fn: Async function to call LLM
        provider_id: LLM provider
        model_id: LLM model
    
    Returns:
        ChunkPlan with parsed chunks
    """
    plan_id = str(uuid4())
    
    prompt = build_chunk_planner_prompt(
        arch_content=arch_content,
        spec_json=spec_json,
    )
    
    messages = [
        {"role": "system", "content": "You are an expert at breaking down architecture into implementation chunks. Output only valid JSON."},
        {"role": "user", "content": prompt},
    ]
    
    try:
        result = await llm_call_fn(
            provider_id=provider_id,
            model_id=model_id,
            messages=messages,
        )
        
        raw_output = result.content if hasattr(result, "content") else str(result)
        chunks = parse_chunk_plan_output(raw_output)
        
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
        
        logger.info(f"[planner] Generated plan {plan_id} with {len(chunks)} chunks")
        return plan
        
    except Exception as e:
        logger.error(f"[planner] Failed to generate chunk plan: {e}")
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


# =============================================================================
# Plan Storage
# =============================================================================

def store_chunk_plan(
    plan: ChunkPlan,
    job_artifact_root: str,
) -> str:
    """Store chunk plan to filesystem.
    
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
    "build_chunk_planner_prompt",
    "parse_chunk_plan_output",
    "generate_chunk_plan",
    "store_chunk_plan",
    "load_chunk_plan",
    "topological_sort_chunks",
    "get_next_chunk",
]
