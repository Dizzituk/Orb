# FILE: app/pot_spec/grounded/_simple_create_review.py
# Purpose: SpecGate CREATE self-review pass.
# Called-by: app.pot_spec.grounded._simple_create_utils_17
# Depends-on: app.pot_spec.grounded.simple_create
# Last-renovated: 2026-06-11
"""
SpecGate CREATE self-review pass.

After the evidence loop has produced a draft spec, this runs one structured
self-audit: the model re-reads the original intent and its own draft, lists
every requirement and every EXISTING thing the draft claims to reuse, and
verifies each is covered and actually grounded (read), not guessed. Any core
reuse-target it only assumed triggers a fresh EVIDENCE_REQUEST, so the real
file gets read before the spec is finalised.

The review is given the SAME real, discovered file paths and project roots that
the main analysis loop gets, so it copies exact paths instead of fabricating
them from the prose in the brief. It is never hardcoded to a codebase - the
paths come from whatever target was actually discovered for this job.

This is the "list everything, check it, fill the gaps, then finalise" gate that
lives inside SpecGate itself. The live pipeline is Weaver -> SpecGate -> Builder
-> Verifier with no separate critique stage, so the rigour has to sit in-stage.

Single responsibility: the review prompt + the review-and-verify orchestration.
"""
from __future__ import annotations

import logging
from app.pot_spec.grounded._stage_reasoning import spec_gate_reasoning, spec_gate_timeout_seconds
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


def _format_known_paths(integration_points: Optional[List[Any]]) -> str:
    """Render discovered integration points as a list of real absolute paths."""
    if not integration_points:
        return ""
    lines: List[str] = []
    for p in integration_points:
        fp = None
        if hasattr(p, "file_path"):
            fp = getattr(p, "file_path", None)
        elif isinstance(p, dict):
            fp = p.get("file_path") or p.get("path")
        elif isinstance(p, str):
            fp = p
        if fp:
            lines.append(f"- {fp}")
    return "\n".join(lines)


def _build_review_prompt(
    goal: str,
    what_to_do: str,
    draft: str,
    integration_points: Optional[List[Any]] = None,
    project_paths: Optional[List[str]] = None,
) -> str:
    """Construct the self-review user prompt for the model."""
    roots_block = "\n".join(f"- {r}" for r in (project_paths or [])) or "(none provided)"
    known = _format_known_paths(integration_points)
    known_block = known if known else "(none discovered - see PATH RULES; do not invent paths)"

    return f"""SELF-REVIEW PASS - audit your own draft before it is finalised.

You previously produced the draft spec below. Before it can be accepted, audit it
against the ORIGINAL INTENT and confirm it is fully grounded. Do NOT simply
restate it.

--- ORIGINAL INTENT ---
Feature Request:
{goal}

Full Description / Requirements:
{what_to_do}
--- END INTENT ---

--- YOUR CURRENT DRAFT SPEC ---
{draft}
--- END DRAFT ---

PROJECT ROOT(S) (the real base path(s) of the target codebase):
{roots_block}

VERIFIED EXISTING FILES (these exist on disk - copy these EXACT paths when you
need to read one):
{known_block}

PATH RULES (CRITICAL - a wrong path wastes an entire round):
- Every EVIDENCE_REQUEST file_path MUST be a real absolute path.
- Copy an exact path from the VERIFIED EXISTING FILES list above whenever you can.
- If you need a file that is not on that list, build its path under one of the
  PROJECT ROOT(S) above, following the project's real folder structure. NEVER
  invent a root, and NEVER put "..." or any placeholder segment in a path.
- Do not reconstruct a path from prose in the brief - use the roots/paths given.
- If you cannot determine a real path for something you must reuse, do NOT guess:
  mark it as [HUMAN_REQUIRED: <what you need and why>] in the draft instead.

Work through this audit step by step:

1. REQUIREMENTS COVERAGE
   List every distinct requirement and constraint in the intent. For each, state
   whether the draft actually addresses it. If any is missing or only partly
   handled, fix the draft so it is covered.

2. REUSE VERIFICATION (CRITICAL)
   List every EXISTING file, class, function, endpoint, or pattern the draft
   claims to reuse, call, or hook into - for example an existing chat-send path,
   an existing offline queue or drainer, an existing persistence primitive, or an
   existing endpoint. For EACH, decide:
     - VERIFIED: you actually read the real file and the signature/behaviour you
       depend on is confirmed.
     - UNVERIFIED: you guessed or assumed it, or marked it [VERIFY] /
       [DECISION_ALLOWED] without reading the real code.
   Any reuse-target the feature's correctness depends on that is UNVERIFIED is NOT
   acceptable. For each one, emit a targeted EVIDENCE_REQUEST now to read the real
   file (obeying the PATH RULES above). A "safe default" does NOT apply to
   something you must reuse: if, after reading, you genuinely cannot confirm it,
   mark it [HUMAN_REQUIRED] - never [DECISION_ALLOWED].

3. CONSISTENCY
   Confirm the Files to Modify and New Files lists are consistent with the
   Architecture Overview and the Acceptance Criteria, and that nothing
   contradicts a stated constraint or introduces an undeclared dependency.

Then output the corrected, COMPLETE draft spec - every section (Architecture
Overview, Files to Modify, New Files to Create, Acceptance Criteria) - reflecting
the audit. You MAY emit EVIDENCE_REQUEST blocks AFTER the draft to verify
unresolved reuse-targets; the orchestrator will read those files and re-prompt
you. Never emit only EVIDENCE_REQUESTs with no draft."""


async def run_spec_self_review(
    llm_analysis: str,
    provider_id: str,
    model_id: str,
    llm_call_func: Optional[Callable],
    project_paths: List[str],
    goal: str = "",
    what_to_do: str = "",
    integration_points: Optional[List[Any]] = None,
) -> str:
    """Run one self-review pass over the draft, then resolve any new evidence.

    The review is handed the real discovered file paths and project roots (via
    integration_points / project_paths) so it copies exact paths instead of
    inventing them. Returns the reviewed (and evidence-fulfilled) analysis. On
    any failure the original draft is returned unchanged - the review must never
    degrade output.
    """
    if not llm_analysis or not llm_call_func or not provider_id or not model_id:
        return llm_analysis

    try:
        from .simple_create import (
            CREATE_ANALYSIS_SYSTEM_PROMPT,
            _fulfil_evidence_requests,
        )
    except Exception as exc:  # import guard - never break the spec flow
        logger.warning("[SPEC_GATE_REVIEW] imports unavailable, skipping review: %s", exc)
        return llm_analysis

    review_prompt = _build_review_prompt(
        goal, what_to_do, llm_analysis, integration_points, project_paths
    )

    logger.info("[SPEC_GATE_REVIEW] Starting self-review pass (%s/%s)", provider_id, model_id)
    print("[SPEC_GATE_REVIEW] Self-review: auditing draft vs intent + verifying reuse-targets")

    try:
        result = await llm_call_func(
            provider_id=provider_id,
            model_id=model_id,
            messages=[{"role": "user", "content": review_prompt}],
            system_prompt=CREATE_ANALYSIS_SYSTEM_PROMPT,
            temperature=1.0,  # stripped by registry when the model requires it
            max_tokens=16384,  # floored upward by registry per effort level
            timeout_seconds=spec_gate_timeout_seconds(),
            reasoning=spec_gate_reasoning(),  # v2.8: env-overridable (SPEC_GATE_REASONING_EFFORT)
        )
    except Exception as exc:
        logger.warning("[SPEC_GATE_REVIEW] review call failed, keeping original draft: %s", exc)
        return llm_analysis

    is_ok = False
    try:
        is_ok = bool(result.is_success()) and bool(result.content)
    except Exception:
        is_ok = False

    if not is_ok:
        logger.warning("[SPEC_GATE_REVIEW] review returned no usable content, keeping original draft")
        return llm_analysis

    reviewed = result.content.strip()
    logger.info("[SPEC_GATE_REVIEW] Review produced %d chars", len(reviewed))

    # If the review raised new evidence requests (reuse-targets it had only
    # guessed), resolve them through the existing fulfilment loop so the real
    # files get read before the spec is finalised.
    if "EVIDENCE_REQUEST" in reviewed:
        logger.info("[SPEC_GATE_REVIEW] Review emitted EVIDENCE_REQUESTs - verifying reuse-targets")
        print("[SPEC_GATE_REVIEW] Review flagged unverified reuse-targets - reading the real files")
        try:
            reviewed = await _fulfil_evidence_requests(
                llm_analysis=reviewed,
                provider_id=provider_id,
                model_id=model_id,
                llm_call_func=llm_call_func,
                project_paths=project_paths,
                goal=goal,
                what_to_do=what_to_do,
            )
        except Exception as exc:
            logger.warning("[SPEC_GATE_REVIEW] evidence fulfilment after review failed: %s", exc)
            # 'reviewed' still holds the audited draft; return it rather than failing.

    print(f"[SPEC_GATE_REVIEW] Self-review complete: {len(reviewed)} chars")
    return reviewed
