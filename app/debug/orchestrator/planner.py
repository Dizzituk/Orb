# FILE: app/debug/orchestrator/planner.py
# Purpose: Planner — converts investigation reports into an ordered DebugPlan.
# Called-by: app.debug.orchestrator.loop_controller
# Depends-on: app.debug.orchestrator.schemas
# Last-renovated: 2026-06-11
"""
Planner — converts investigation reports into an ordered DebugPlan.

Takes all SubagentReports from the investigation phase and calls a
reasoning model (e.g. gpt-5.4 Thinking) to:
  1. Synthesise a single root cause across the reports
  2. Surface contradictions between subagents
  3. Produce an ordered list of FixSteps with dependencies
  4. Mark which steps can run in parallel (non-overlapping file scopes)

Pure LLM synthesis — no tool use, no code changes.

v1.0 (2026-04-13): Initial implementation.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set

from app.debug.orchestrator.schemas import (
    DebugPlan,
    FixStep,
    SubagentReport,
)

logger = logging.getLogger(__name__)

_MAX_STEPS = 10


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """You are ASTRA's Debug Planner. You receive investigation \
reports from multiple subagents and produce a single ordered FIX PLAN.

# Hard rules (non-negotiable)

1. If the investigation found one or more bugs, root causes, or unfinished
   code paths, you MUST produce concrete fix steps. "Needs manual review",
   "already mostly done", or "unclear" are NOT acceptable outputs if the
   investigators surfaced specific file paths and code issues — propose
   the fix, even if you're only 60 percent sure. Err on the side of action.

2. If you genuinely cannot produce ANY fix step from the findings, every
   single one of those bugs must appear in unresolved_bugs with a reason.
   An empty steps array + empty unresolved_bugs is a malformed response.

3. Fix steps describe what to DO, not what the bug is:
   - BAD:  "Login flow lacks idle-based expiry"
   - GOOD: "Add interaction-based session expiry check to ConnectionManager.isLoggedIn()"

4. Each step targets specific files. The executors are scoped to those
   files and cannot go outside them, so get this right.

# Your synthesis job

a) Identify the single most likely root cause, woven from all the reports.
b) Note contradictions between subagents (findings that disagree).
c) Produce fix steps, ordered by dependency (if B needs A done first, mark
   depends_on). Prefer fewer larger steps over many tiny ones, but never
   bundle unrelated concerns.
d) Mark steps parallelisable_with ONLY when their target_files do not
   overlap AND they have no logical ordering dependency. When in doubt,
   leave them serial.
e) Any bug you cannot translate into a fix step goes in unresolved_bugs
   with the reason why.

Respond with a single JSON object. No prose outside it."""

_PLANNER_RESPONSE_SCHEMA = """\
{
  "root_cause": "Single-paragraph synthesis of the root cause.",
  "confidence": 0.0-1.0,
  "contradictions": ["Plain-English statements of conflicting findings"],
  "steps": [
    {
      "step_id": "fix_01",
      "description": "What this step does, plain English.",
      "target_project": "project_id matching registered target",
      "target_files": ["path/relative/to/repo.py"],
      "depends_on": ["step_ids that must complete first"],
      "parallelisable_with": ["step_ids with non-overlapping files"],
      "rationale": "Which findings support this step"
    }
  ],
  "unresolved_bugs": ["Bugs the plan does not address"]
}"""


def _report_summary_for_prompt(report: SubagentReport) -> Dict[str, Any]:
    """Compact representation of a SubagentReport for the planner prompt."""
    return {
        "brief_id": report.brief_id,
        "role": report.role.value,
        "status": report.status.value,
        "findings": [
            {
                "summary": f.summary,
                "confidence": round(f.confidence, 2),
                "tags": f.tags,
                "evidence_count": len(f.evidence),
                "evidence_preview": f.evidence[:3] if f.evidence else [],
            }
            for f in report.findings
        ],
        "dead_ends": report.dead_ends,
        "files_read": report.files_read[:20],  # cap — planner doesn't need all of them
        "error": report.error,
    }


def _build_user_prompt(
    bug_list: str,
    reports: List[SubagentReport],
    prior_iteration_summary: Optional[str] = None,
) -> str:
    reports_json = json.dumps(
        [_report_summary_for_prompt(r) for r in reports],
        indent=2,
    )
    prior = (
        f"\n## Prior Iteration Summary (if this is a re-plan)\n{prior_iteration_summary}\n"
        if prior_iteration_summary else ""
    )
    return (
        f"## Original Bug List\n{bug_list}\n\n"
        f"## Investigation Reports ({len(reports)} subagent(s))\n{reports_json}\n"
        f"{prior}\n"
        f"## Required Response Shape\n{_PLANNER_RESPONSE_SCHEMA}\n\n"
        f"Max steps: {_MAX_STEPS}. Prefer fewer, larger steps over many tiny ones, "
        "but never bundle unrelated concerns into one step."
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

async def _call_planner_llm(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int = 20000,
) -> str:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("openai package not installed")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    # Planner synthesises across all investigation reports; give it room to think.
    client = AsyncOpenAI(api_key=api_key, timeout=300.0)
    call_kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    try:
        call_kwargs["reasoning_effort"] = "high"
        resp = await client.chat.completions.create(**call_kwargs)
    except Exception as e:
        err_str = str(e).lower()
        if "reasoning_effort" in err_str or "unrecognized" in err_str or "unsupported" in err_str:
            logger.info("[planner] model=%s does not support reasoning_effort, retrying without", model)
            call_kwargs.pop("reasoning_effort", None)
            resp = await client.chat.completions.create(**call_kwargs)
        else:
            raise
    return resp.choices[0].message.content or "{}"


# ---------------------------------------------------------------------------
# JSON parsing + validation
# ---------------------------------------------------------------------------

def _parse_planner_output(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    t = raw.strip()
    for fence in ("```json", "```JSON", "```"):
        if fence in t:
            for chunk in t.split(fence):
                chunk = chunk.strip()
                if chunk.startswith("{") and chunk.rstrip().endswith("}"):
                    t = chunk
                    break
    if not t.startswith("{"):
        s, e = t.find("{"), t.rfind("}")
        if s >= 0 and e > s:
            t = t[s:e + 1]
    try:
        return json.loads(t)
    except Exception as err:
        logger.warning("[planner] JSON parse failed: %s", err)
        return {}


def _sanitise_parallelism(steps: List[FixStep]) -> List[FixStep]:
    """Strip parallelisable_with entries where target_files actually overlap."""
    by_id = {s.step_id: s for s in steps}
    for s in steps:
        safe = []
        s_files: Set[str] = set(s.target_files)
        for pid in s.parallelisable_with:
            p = by_id.get(pid)
            if not p:
                continue
            if p.step_id in s.depends_on:
                continue
            if s.step_id in p.depends_on:
                continue
            if s_files.intersection(set(p.target_files)):
                logger.info(
                    "[planner] Dropping parallelism %s <-> %s (file overlap)",
                    s.step_id, pid,
                )
                continue
            safe.append(pid)
        s.parallelisable_with = safe
    return steps


def _validate_plan(raw: Dict[str, Any]) -> DebugPlan:
    steps_raw = raw.get("steps", []) or []
    steps: List[FixStep] = []

    for i, s in enumerate(steps_raw[:_MAX_STEPS]):
        try:
            step = FixStep(
                step_id=str(s.get("step_id") or f"fix_{i+1:02d}"),
                description=str(s.get("description", "")).strip() or "(no description)",
                target_project=str(s.get("target_project", "unknown")),
                target_files=[str(p) for p in (s.get("target_files") or []) if p][:10],
                depends_on=[str(d) for d in (s.get("depends_on") or []) if d],
                parallelisable_with=[str(p) for p in (s.get("parallelisable_with") or []) if p],
                rationale=str(s.get("rationale", "")).strip() or "(no rationale)",
            )
            steps.append(step)
        except Exception as e:
            logger.warning("[planner] Skipped invalid step idx=%d: %s", i, e)

    # Sanitise parallelism markers against actual file overlaps
    steps = _sanitise_parallelism(steps)

    confidence_raw = raw.get("confidence", 0.5)
    try:
        confidence = max(0.0, min(1.0, float(confidence_raw)))
    except Exception:
        confidence = 0.5

    return DebugPlan(
        root_cause=str(raw.get("root_cause", "")).strip() or "(root cause not identified)",
        steps=steps,
        contradictions=[str(x) for x in (raw.get("contradictions") or []) if x],
        confidence=confidence,
        unresolved_bugs=[str(x) for x in (raw.get("unresolved_bugs") or []) if x],
    )


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def _fallback_plan(reports: List[SubagentReport]) -> DebugPlan:
    """Planner failed — produce a manual-review plan rather than crashing."""
    findings_all = []
    for r in reports:
        for f in r.findings:
            findings_all.append(f.summary)
    return DebugPlan(
        root_cause=(
            "Planner LLM could not produce a structured plan. "
            "Manual review required. "
            + (f"Collected findings: {'; '.join(findings_all[:5])}" if findings_all else "")
        ),
        steps=[],
        contradictions=[],
        confidence=0.0,
        unresolved_bugs=["Plan generation failed"],
    )


# ---------------------------------------------------------------------------
# Dependency ordering for executors
# ---------------------------------------------------------------------------

def topological_batches(plan: DebugPlan) -> List[List[FixStep]]:
    """Group steps into batches that can execute in parallel.

    Each batch contains steps whose dependencies are all satisfied by
    earlier batches. Within a batch, steps may run concurrently as long
    as their parallelisable_with markers agree (already sanitised).

    Returns a list of batches, in execution order.
    """
    remaining = {s.step_id: s for s in plan.steps}
    completed: Set[str] = set()
    batches: List[List[FixStep]] = []

    safety = len(remaining) + 2
    while remaining and safety > 0:
        safety -= 1
        ready = [
            s for s in remaining.values()
            if all(d in completed for d in s.depends_on)
        ]
        if not ready:
            # Circular dep or dangling reference — flush the rest serially
            logger.warning(
                "[planner] Circular/dangling deps detected; flushing %d steps serially",
                len(remaining),
            )
            for s in list(remaining.values()):
                batches.append([s])
                completed.add(s.step_id)
                del remaining[s.step_id]
            break

        batches.append(ready)
        for s in ready:
            completed.add(s.step_id)
            del remaining[s.step_id]

    return batches


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def plan_fixes(
    bug_list: str,
    reports: List[SubagentReport],
    model: str,
    prior_iteration_summary: Optional[str] = None,
) -> DebugPlan:
    """Synthesise investigation reports into an ordered fix plan."""
    t_start = time.time()

    if not reports:
        logger.warning("[planner] No reports provided")
        return DebugPlan(
            root_cause="No investigation reports to synthesise.",
            steps=[],
            contradictions=[],
            confidence=0.0,
            unresolved_bugs=["Investigation phase produced no reports"],
        )

    user_prompt = _build_user_prompt(bug_list, reports, prior_iteration_summary)

    try:
        raw_text = await _call_planner_llm(
            system_prompt=_PLANNER_SYSTEM,
            user_prompt=user_prompt,
            model=model,
        )
    except Exception as e:
        logger.exception("[planner] LLM call failed: %s", e)
        return _fallback_plan(reports)

    raw = _parse_planner_output(raw_text)
    if not raw:
        logger.warning(
            "[planner] LLM response did not parse as JSON (%d chars); first 500: %r",
            len(raw_text or ""), (raw_text or "")[:500],
        )
    else:
        step_count_in = len(raw.get("steps", []) or [])
        logger.info(
            "[planner] raw response parsed ok. Claimed steps=%d, keys=%r, root_cause_len=%d",
            step_count_in,
            list(raw.keys()) if isinstance(raw, dict) else None,
            len(str(raw.get("root_cause", ""))),
        )
    plan = _validate_plan(raw)

    if not plan.steps and not plan.unresolved_bugs:
        logger.warning(
            "[planner] Plan has no steps and no unresolved bugs — using fallback. "
            "Raw response first 2000 chars: %r",
            (raw_text or "")[:2000],
        )
        plan = _fallback_plan(reports)

    logger.info(
        "[planner] Produced %d step(s), confidence=%.2f, contradictions=%d in %dms",
        len(plan.steps), plan.confidence, len(plan.contradictions),
        int((time.time() - t_start) * 1000),
    )
    return plan
