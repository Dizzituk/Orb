# FILE: app/debug/orchestrator/decomposer.py
# Purpose: Decomposer — converts a free-text bug list into a list of scoped SubagentBriefs.
# Called-by: app.debug.orchestrator.loop_controller
# Depends-on: app.debug.orchestrator.schemas, app.pipeline_v2.target_registry
# Last-renovated: 2026-06-11
"""
Decomposer — converts a free-text bug list into a list of scoped SubagentBriefs.

Uses a single planning-model call (e.g. gpt-5.4 Thinking) to:
  1. Identify the distinct problems in the bug list
  2. Infer which target_project each problem belongs to
  3. Group related problems into investigation briefs
  4. Propose starting-point files where obvious

The decomposer never touches code or tools — it is pure LLM synthesis over
the bug list + a summary of registered projects from target_registry.

v1.0 (2026-04-13): Initial implementation.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from app.debug.orchestrator.schemas import (
    DecompositionResult,
    SubagentBrief,
    SubagentRole,
)

logger = logging.getLogger(__name__)

_MAX_BRIEFS = 8
# Was 15; we observed a serial single-model run hit 30 rounds to fix 4 concerns,
# so each parallel subagent needs plenty of headroom to investigate its slice.
_DEFAULT_MAX_TOOL_CALLS_PER_BRIEF = 150


# ---------------------------------------------------------------------------
# Project registry summary
# ---------------------------------------------------------------------------

def _project_summary() -> str:
    """Build a compact summary of registered projects for the decomposer prompt."""
    try:
        from app.pipeline_v2.target_registry import list_profiles
        profiles = list_profiles()
    except Exception as e:
        logger.warning("[decomposer] Could not load target registry: %s", e)
        return "(project registry unavailable — use your best judgement for target_project)"

    if not profiles:
        return "(no projects registered)"

    lines = []
    for p in profiles:
        lines.append(
            f"- {p.project_name} (id={p.project_name}) "
            f"[{p.language}/{p.framework}] root={p.project_root}"
        )
    return "\n".join(lines)


def _valid_project_ids() -> List[str]:
    try:
        from app.pipeline_v2.target_registry import list_profiles
        return [p.project_name for p in list_profiles()]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_DECOMPOSER_SYSTEM = """You are ASTRA's Debug Decomposer. Your job is to read a \
free-text list of bugs (written by the developer, possibly via voice-to-text \
so expect typos and incomplete sentences) and break them into a small set of \
INVESTIGATION BRIEFS that parallel subagents can work on.

Principles:
- Prefer FEWER briefs that each cover related bugs over many tiny briefs.
- One brief = one coherent investigation question, scoped to ONE target_project.
- If bugs span multiple projects, create one brief per project involved.
- Every brief is role=investigator at this stage. Executors come later.
- Suggest 1-3 likely starting files per brief if obvious; leave empty otherwise.
- If a bug cannot be mapped to a known project, note it in unaddressed_bugs.

You MUST respond with a single JSON object and nothing else."""

_DECOMPOSER_RESPONSE_SCHEMA = """\
{
  "rationale": "Plain-English explanation of how you broke the problem up",
  "briefs": [
    {
      "brief_id": "inv_01",
      "task": "Scoped investigation question in plain English",
      "target_project": "must match one of the registered project IDs",
      "context_files": ["optional/path/hint.py"],
      "related_bugs": ["verbatim or paraphrased bug text this covers"]
    }
  ],
  "unaddressed_bugs": ["bug text that could not be mapped"]
}"""


def _build_user_prompt(bug_list: str, target_project_hint: Optional[str]) -> str:
    registered = _project_summary()
    hint = (
        f"\n## Hint\nThe developer has indicated target_project='{target_project_hint}'. "
        "Prefer this unless a bug clearly belongs to a different project.\n"
        if target_project_hint else ""
    )
    return (
        f"## Bug List (verbatim)\n{bug_list}\n\n"
        f"## Registered Projects\n{registered}\n"
        f"{hint}\n"
        f"## Required Response Shape\n{_DECOMPOSER_RESPONSE_SCHEMA}\n\n"
        f"Max briefs: {_MAX_BRIEFS}. "
        "If the bug list is already narrow, ONE brief is fine."
    )


# ---------------------------------------------------------------------------
# LLM call (OpenAI)
# ---------------------------------------------------------------------------

async def _call_decomposer_llm(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int = 12000,
) -> str:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("openai package not installed")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    # Generous timeout; thinking models can legitimately take 60-120s on high effort
    client = AsyncOpenAI(api_key=api_key, timeout=300.0)

    # Enable reasoning for thinking-capable models. Silently ignored by non-thinking models.
    call_kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    # GPT-5 family supports reasoning_effort; pass high for decomposer/planner quality.
    try:
        call_kwargs["reasoning_effort"] = "high"
        resp = await client.chat.completions.create(**call_kwargs)
    except Exception as e:
        # Fall back without reasoning_effort if the model rejects it
        err_str = str(e).lower()
        if "reasoning_effort" in err_str or "unrecognized" in err_str or "unsupported" in err_str:
            logger.info("[decomposer] model=%s does not support reasoning_effort, retrying without", model)
            call_kwargs.pop("reasoning_effort", None)
            resp = await client.chat.completions.create(**call_kwargs)
        else:
            raise
    return resp.choices[0].message.content or "{}"


# ---------------------------------------------------------------------------
# JSON parsing with tolerance
# ---------------------------------------------------------------------------

def _parse_decomposer_output(raw: str) -> Dict[str, Any]:
    """Parse the LLM response, tolerant to code fences etc."""
    if not raw:
        return {}
    t = raw.strip()
    for fence in ("```json", "```JSON", "```"):
        if fence in t:
            parts = t.split(fence)
            for chunk in parts:
                chunk = chunk.strip()
                if chunk.startswith("{") and chunk.rstrip().endswith("}"):
                    t = chunk
                    break
    if not t.startswith("{"):
        s = t.find("{")
        e = t.rfind("}")
        if s >= 0 and e > s:
            t = t[s:e + 1]
    try:
        return json.loads(t)
    except Exception as err:
        logger.warning("[decomposer] JSON parse failed: %s", err)
        return {}


def _normalise_id(s: str) -> str:
    """Normalise a project id for tolerant matching (case/space/punct invariant)."""
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def _match_project_id(candidate: str, valid_ids: List[str]) -> Optional[str]:
    """Return the canonical id if candidate matches any valid id loosely."""
    if not candidate:
        return None
    norm = _normalise_id(candidate)
    for vid in valid_ids:
        if _normalise_id(vid) == norm:
            return vid
    # Partial prefix/contains match as a last resort
    for vid in valid_ids:
        if norm and (norm in _normalise_id(vid) or _normalise_id(vid) in norm):
            return vid
    return None


def _validate_and_coerce_briefs(
    raw: Dict[str, Any],
    fallback_target: Optional[str],
) -> DecompositionResult:
    """Turn raw JSON into typed DecompositionResult, clamping invalid values."""
    valid_ids_list = _valid_project_ids()
    briefs_raw = raw.get("briefs", []) or []
    briefs: List[SubagentBrief] = []
    dropped: List[str] = []

    logger.info(
        "[decomposer] validating %d raw brief(s), valid targets: %r",
        len(briefs_raw), valid_ids_list,
    )

    for i, b in enumerate(briefs_raw[:_MAX_BRIEFS]):
        try:
            brief_id = str(b.get("brief_id") or f"inv_{i+1:02d}")
            task = str(b.get("task", "")).strip()
            if not task:
                dropped.append(f"{brief_id}: empty task")
                continue

            raw_target = str(b.get("target_project", "")).strip()
            target = _match_project_id(raw_target, valid_ids_list)

            if target is None and valid_ids_list:
                # Fallback: use hint if it resolves, else first valid id
                if fallback_target:
                    target = _match_project_id(fallback_target, valid_ids_list)
                if target is None:
                    target = valid_ids_list[0]
                logger.info(
                    "[decomposer] brief %s: target %r not matched, using %r",
                    brief_id, raw_target, target,
                )
            elif target is None:
                target = raw_target or "unknown"

            ctx_files = [str(p) for p in (b.get("context_files") or []) if p]
            related = [str(x) for x in (b.get("related_bugs") or []) if x]

            briefs.append(SubagentBrief(
                brief_id=brief_id,
                role=SubagentRole.INVESTIGATOR,
                task=task,
                target_project=target,
                context_files=ctx_files[:5],  # cap at 5 hints
                related_bugs=related[:10],
                depends_on=[],
                max_tool_calls=_DEFAULT_MAX_TOOL_CALLS_PER_BRIEF,
            ))
        except Exception as e:
            logger.warning("[decomposer] Skipped invalid brief at idx=%d: %s", i, e)

    rationale = str(raw.get("rationale", "")).strip() or "(no rationale provided)"
    unaddressed = [str(x) for x in (raw.get("unaddressed_bugs") or []) if x]

    return DecompositionResult(
        briefs=briefs,
        rationale=rationale,
        unaddressed_bugs=unaddressed,
    )


# ---------------------------------------------------------------------------
# Fallback: single all-encompassing brief if LLM returns nothing usable
# ---------------------------------------------------------------------------

def _fallback_decomposition(bug_list: str, target_project: Optional[str]) -> DecompositionResult:
    valid_ids = _valid_project_ids()
    target = target_project or (valid_ids[0] if valid_ids else "unknown")
    return DecompositionResult(
        briefs=[SubagentBrief(
            brief_id="inv_01",
            role=SubagentRole.INVESTIGATOR,
            task=(
                "Investigate all issues listed in the bug list. "
                "Identify root causes, cite files/lines, return structured findings."
            ),
            target_project=target,
            context_files=[],
            related_bugs=[bug_list[:500]],
            depends_on=[],
            max_tool_calls=_DEFAULT_MAX_TOOL_CALLS_PER_BRIEF * 3,  # single brief: give more budget
        )],
        rationale="Decomposer fallback: LLM response unusable, investigating holistically.",
        unaddressed_bugs=[],
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def decompose(
    bug_list: str,
    target_project: Optional[str],
    model: str,
) -> DecompositionResult:
    """Decompose a free-text bug list into investigation briefs.

    Args:
        bug_list: Raw developer-authored text describing what's broken.
        target_project: Optional hint; if valid, preferred unless LLM
            clearly identifies a different project in the bug text.
        model: Model name (e.g. "gpt-5.4").

    Returns:
        DecompositionResult. If the LLM fails entirely, returns a
        single-brief fallback so the loop can still proceed.
    """
    t_start = time.time()
    system_prompt = _DECOMPOSER_SYSTEM
    user_prompt = _build_user_prompt(bug_list, target_project)

    try:
        raw_text = await _call_decomposer_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
        )
    except Exception as e:
        logger.exception("[decomposer] LLM call failed: %s", e)
        return _fallback_decomposition(bug_list, target_project)

    raw = _parse_decomposer_output(raw_text)
    if not raw:
        logger.warning(
            "[decomposer] LLM response did not parse as JSON (%d chars); first 500: %r",
            len(raw_text or ""), (raw_text or "")[:500],
        )
    elif not raw.get("briefs"):
        logger.warning(
            "[decomposer] LLM returned JSON with no briefs key. Raw keys: %r; raw: %r",
            list(raw.keys()) if isinstance(raw, dict) else None,
            str(raw)[:500],
        )
    result = _validate_and_coerce_briefs(raw, fallback_target=target_project)

    if not result.briefs:
        logger.warning(
            "[decomposer] No valid briefs produced — using fallback. Raw response first 2000 chars: %r",
            (raw_text or "")[:2000],
        )
        result = _fallback_decomposition(bug_list, target_project)

    logger.info(
        "[decomposer] Produced %d brief(s) in %dms, unaddressed=%d",
        len(result.briefs), int((time.time() - t_start) * 1000),
        len(result.unaddressed_bugs),
    )
    return result
