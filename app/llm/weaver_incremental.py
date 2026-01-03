# FILE: app/llm/weaver_incremental.py
"""
Weaver Incremental Helpers for ASTRA.

This module provides small, dependency-light helpers to enable "incremental weaving":
- Track a checkpoint (last spec id + last seen message id + last weak spots)
- Gather delta-only messages since the checkpoint
- Build create/update prompts that:
  - avoid invention (no made-up filenames/paths/OS scripts)
  - ignore assistant meta-noise
  - keep output concise for small models
- Provide safe conversion utilities for SpecSchema objects

Design goals:
- Backwards compatible: if any optional dependency is missing, callers can fall back to
  full-window weaving without breaking existing behavior.
- Low risk: helpers avoid touching DB directly; callers pass in the data they already have.

This file intentionally contains no side effects and no global state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class WeaverCheckpoint:
    """State needed to compute "delta context" for the next weave."""
    last_weaver_spec_id: Optional[str] = None
    last_seen_message_id: Optional[int] = None
    last_weak_spots: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.last_weak_spots is None:
            self.last_weak_spots = []


def extract_spec_core(spec_schema: Any) -> Dict[str, Any]:
    """
    Extract the core, prompt-relevant fields from a SpecSchema-like object.

    We intentionally exclude large provenance blocks and any DB-only fields.
    This keeps prompts small and avoids confusing the model.

    Supports Pydantic (model_dump/dict) and dataclass-like objects.
    """
    def _get(obj: Any, name: str, default: Any = None) -> Any:
        return getattr(obj, name, default)

    # Try pydantic-style dumps first
    data: Dict[str, Any] = {}
    if hasattr(spec_schema, "model_dump"):
        try:
            data = spec_schema.model_dump()  # type: ignore[attr-defined]
        except Exception:
            data = {}
    elif hasattr(spec_schema, "dict"):
        try:
            data = spec_schema.dict()  # type: ignore[attr-defined]
        except Exception:
            data = {}

    # If we couldn't dump cleanly, extract attributes directly
    if not data:
        data = {
            "title": _get(spec_schema, "title", ""),
            "summary": _get(spec_schema, "summary", ""),
            "objective": _get(spec_schema, "objective", ""),
            "requirements": _get(spec_schema, "requirements", None),
            "constraints": _get(spec_schema, "constraints", None),
            "safety": _get(spec_schema, "safety", None),
            "acceptance_criteria": _get(spec_schema, "acceptance_criteria", []),
            "dependencies": _get(spec_schema, "dependencies", []),
            "non_goals": _get(spec_schema, "non_goals", []),
            "metadata": _get(spec_schema, "metadata", None),
        }

    # Normalize nested Pydantic objects
    def _normalize(v: Any) -> Any:
        if hasattr(v, "model_dump"):
            try:
                return v.model_dump()  # type: ignore[attr-defined]
            except Exception:
                return str(v)
        if hasattr(v, "dict"):
            try:
                return v.dict()  # type: ignore[attr-defined]
            except Exception:
                return str(v)
        return v

    core: Dict[str, Any] = {
        "title": data.get("title", ""),
        "summary": data.get("summary", ""),
        "objective": data.get("objective", ""),
        "requirements": _normalize(data.get("requirements", {})),
        "constraints": _normalize(data.get("constraints", {})),
        "safety": _normalize(data.get("safety", {})),
        "acceptance_criteria": data.get("acceptance_criteria", []) or [],
        "dependencies": data.get("dependencies", []) or [],
        "non_goals": data.get("non_goals", []) or [],
        "metadata": _normalize(data.get("metadata", {})),
    }
    return core


def compute_weak_spots_delta(previous: Sequence[str], current: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Return (resolved, newly_added) weak spots based on simple string equality."""
    prev_set = set([p.strip() for p in previous if p and p.strip()])
    curr_set = set([c.strip() for c in current if c and c.strip()])
    resolved = sorted(list(prev_set - curr_set))
    newly_added = sorted(list(curr_set - prev_set))
    return resolved, newly_added


def is_assistant_meta_noise(content: str) -> bool:
    """
    Heuristic: detect assistant meta/disclaimer lines that should not steer the spec.
    Only used to *downweight* assistant messages in prompts, never to drop user intent.
    """
    if not content:
        return False
    c = content.lower()
    patterns = [
        "i can't access your filesystem",
        "i cannot access your filesystem",
        "tell me which os",
        "i will give exact commands",
        "i can't create files directly",
        "i cannot create files directly",
    ]
    return any(p in c for p in patterns)


def format_conversation_for_prompt(messages: Sequence[Dict[str, Any]]) -> str:
    """
    Create a compact, model-friendly conversation transcript.

    - Always include user messages.
    - Include assistant messages unless they look like meta-noise.
    """
    lines: List[str] = []
    for msg in messages:
        role = (msg.get("role") or "user").lower()
        content = (msg.get("content") or "").strip()
        if not content:
            continue

        if role == "assistant" and is_assistant_meta_noise(content):
            # skip meta-noise to reduce drift/bloat
            continue

        tag = "USER" if role == "user" else role.upper()
        lines.append(f"[{tag}] {content}")
    return "\n\n".join(lines)


def truncate_markdown(markdown: str, max_chars: int) -> str:
    """Truncate markdown output to keep UI readable and preserve context headroom."""
    if max_chars <= 0 or len(markdown) <= max_chars:
        return markdown
    return markdown[: max_chars].rstrip() + "\n\n…(truncated for brevity)…\n"
