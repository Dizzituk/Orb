# FILE: app/pot_spec/spec_gate.py
"""
Spec Gate - Converts user intent into a PoT spec DRAFT as a single JSON object.

This is the "singular point of truth" before the critical pipeline and overwatcher.

Key addition (2026-01):
- Adds optional "deliverables" field to the spec JSON so Overwatcher can deterministically
  choose a target file / action without guessing.

Key addition (2026-01-03):
- Can operate in "Weaver validation" mode when the caller supplies Weaver JSON via
  constraints_hint["weaver_spec_json"]. In this mode Spec Gate must:
    - Treat Weaver JSON as the source of truth
    - Ask only job-relevant clarification questions (not generic architecture questions)
    - On Round 3, return a final spec JSON even if gaps remain, recording gaps in open_issues
      (and leaving open_questions empty).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Small helpers / infra
# =============================================================================

def _artifact_root() -> str:
    return os.path.abspath(os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"))


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_event(job_root: str, job_id: str, event: dict) -> None:
    """Append ledger event (best-effort)."""
    try:
        path = os.path.join(job_root, "jobs", job_id, "events.ndjson")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("[spec_gate] failed to append event")


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_spec_file(job_root: str, job_id: str, spec_payload: dict) -> tuple[str, str]:
    """
    Writes spec JSON to job_root/jobs/<job_id>/spec/spec_v1.json and returns (spec_id, spec_hash).

    IMPORTANT:
    - spec_hash is computed over the canonical JSON of the payload *excluding* the spec_hash field itself.
    - The file written to disk includes spec_hash.
    """
    spec_id = spec_payload.get("spec_id") or os.urandom(16).hex()
    spec_payload["spec_id"] = spec_id
    spec_payload["spec_version"] = str(spec_payload.get("spec_version") or "1")

    # Ensure timestamp exists (use created_at if present)
    if not spec_payload.get("timestamp"):
        spec_payload["timestamp"] = spec_payload.get("created_at") or _utc_ts()

    # Compute hash excluding spec_hash (avoid self-referential hashing)
    payload_for_hash = dict(spec_payload)
    payload_for_hash.pop("spec_hash", None)
    spec_hash = _sha256_hex(_canonical_json_bytes(payload_for_hash))
    spec_payload["spec_hash"] = spec_hash

    spec_dir = os.path.join(job_root, "jobs", job_id, "spec")
    os.makedirs(spec_dir, exist_ok=True)
    spec_path = os.path.join(spec_dir, "spec_v1.json")

    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec_payload, f, ensure_ascii=False, indent=2)

    return spec_id, spec_hash


# =============================================================================
# Draft schema (model output)
# =============================================================================

@dataclass
class _SpecGateDraft:
    title: str
    goal: str
    summary: str
    deliverables: list[dict[str, Any]]
    requirements: dict
    constraints: dict
    acceptance_tests: list
    open_questions: list
    open_issues: list
    recommendations: list
    repo_snapshot: Optional[dict]


def _is_draft_meaningful(draft: _SpecGateDraft) -> bool:
    if draft.goal and isinstance(draft.goal, str) and draft.goal.strip():
        return True
    if draft.title and isinstance(draft.title, str) and draft.title.strip():
        return True
    if draft.deliverables and isinstance(draft.deliverables, list):
        return True
    return False


def _parse_json_from_text(text: str) -> dict:
    """
    Extract JSON object from model output.
    Accepts:
      - raw JSON
      - fenced ```json ... ```
    """
    s = (text or "").strip()

    if "```json" in s:
        start = s.find("```json") + len("```json")
        end = s.find("```", start)
        if end == -1:
            raise ValueError("Missing closing ``` for json block")
        s = s[start:end].strip()

    # If there is extra text, try slice from first { to last }
    if not s.startswith("{"):
        a = s.find("{")
        b = s.rfind("}")
        if a == -1 or b == -1 or b <= a:
            raise ValueError("No JSON object found")
        s = s[a : b + 1]

    return json.loads(s)


def parse_spec_gate_output(text: str) -> _SpecGateDraft:
    """
    Parse model output into draft.

    Expected shape (JSON object):
    {
      "title": "...",
      "goal": "...",
      "summary": "...",
      "deliverables": [...],
      "requirements": {"must": [...], "should": [...], "can": [...]},
      "constraints": {...},
      "acceptance_tests": [...],
      "open_questions": [...],
      "open_issues": [...],
      "recommendations": [...],
      "repo_snapshot": {...}
    }

    Back-compat:
    - acceptance_criteria is accepted as a synonym for acceptance_tests
    - open_gaps is accepted as a synonym for open_issues
    """
    obj = _parse_json_from_text(text)
    if not isinstance(obj, dict):
        raise ValueError("Spec Gate output must be a JSON object")

    title = (obj.get("title") or "").strip()
    goal = (obj.get("goal") or obj.get("objective") or "").strip()
    summary = (obj.get("summary") or "").strip()

    deliverables = obj.get("deliverables") or []
    if not isinstance(deliverables, list):
        deliverables = []

    requirements = obj.get("requirements") or {}
    constraints = obj.get("constraints") or {}

    acceptance_tests = (
        obj.get("acceptance_tests")
        or obj.get("acceptance_criteria")
        or obj.get("acceptanceCriteria")
        or []
    )

    open_questions = obj.get("open_questions") or []
    open_issues = obj.get("open_issues") or obj.get("open_gaps") or []

    recommendations = obj.get("recommendations") or []
    repo_snapshot = obj.get("repo_snapshot")

    if not isinstance(requirements, dict):
        requirements = {}
    if not isinstance(constraints, dict):
        constraints = {}
    if not isinstance(acceptance_tests, list):
        acceptance_tests = []
    if not isinstance(open_questions, list):
        open_questions = []
    if not isinstance(open_issues, list):
        open_issues = []
    if not isinstance(recommendations, list):
        recommendations = []
    if repo_snapshot is not None and not isinstance(repo_snapshot, dict):
        repo_snapshot = None

    return _SpecGateDraft(
        title=title,
        goal=goal,
        summary=summary,
        deliverables=deliverables,
        requirements=requirements,
        constraints=constraints,
        acceptance_tests=acceptance_tests,
        open_questions=open_questions,
        open_issues=open_issues,
        recommendations=recommendations,
        repo_snapshot=repo_snapshot,
    )


# =============================================================================
# Question detection (required by engine/high_stakes_stream imports)
# =============================================================================

def detect_user_questions(text: str) -> bool:
    """
    Return True if `text` appears to contain user-facing questions that require an answer.

    This is intentionally conservative and safe:
    - If the text is Spec Gate JSON and contains a non-empty `open_questions`, returns True.
    - Otherwise checks for obvious question markers (?, "Clarification Needed", "please provide", etc.)
    """
    s = (text or "").strip()
    if not s:
        return False

    # 1) If it's JSON with open_questions, treat that as authoritative
    try:
        obj = _parse_json_from_text(s)
        if isinstance(obj, dict):
            oq = obj.get("open_questions")
            if isinstance(oq, list) and len(oq) > 0:
                return True
    except Exception:
        pass

    # 2) Heuristics
    markers = [
        "?\n",
        "? ",
        "?\r",
        "Clarification Needed",
        "Please answer",
        "please answer",
        "please provide",
        "Please provide",
    ]
    if "?" in s:
        return True
    return any(m in s for m in markers)


# =============================================================================
# Main Spec Gate entry point
# =============================================================================

async def run_spec_gate(
    db: Any,
    job_id: str,
    user_intent: str,
    provider_id: str,
    model_id: str,
    repo_snapshot: Optional[dict] = None,
    constraints_hint: Optional[dict] = None,
    **kwargs,
) -> Tuple[str, str, List[str]]:
    """
    Generates spec JSON, writes to artifacts, emits events, returns (spec_id, spec_hash, open_questions).

    IMPORTANT:
    - If constraints_hint includes `weaver_spec_json`, Spec Gate must validate/refine that spec,
      NOT invent a different job or ask generic architecture questions.
    """
    job_root = _artifact_root()
    constraints_hint = constraints_hint if isinstance(constraints_hint, dict) else {}

    has_weaver = isinstance(constraints_hint.get("weaver_spec_json"), (dict, list, str))
    mode_line = "MODE: WEAVER_VALIDATION" if has_weaver else "MODE: SPEC_FROM_INTENT"

    # Build prompt (kept deterministic and low-token)
    system_prompt = (
        "You are Spec Gate.\n"
        f"{mode_line}\n\n"
        "Your job is to output ONE JSON object that is a clear, machine-actionable specification.\n"
        "Return ONLY JSON. No prose. No Markdown. No backticks.\n\n"
        "Relevance rules for questions:\n"
        "- Ask questions ONLY if the answer materially changes deliverables, requirements, constraints, or acceptance tests.\n"
        "- If the task is a simple local action (files/folders/text), do NOT ask architecture, scaling, compliance, cloud, CI/CD, or team questions.\n"
        "- Prefer 0–3 questions max per round; if unsure, record gaps in open_issues instead of asking.\n\n"
        "Round-3 rule:\n"
        "- If you are instructed this is Round 3 (final), open_questions MUST be an empty list.\n"
        "- Still output a complete spec JSON. Put remaining unknowns/gaps in open_issues.\n"
        "- Do NOT invent details. Do NOT write assumptions.\n\n"
        "OUTPUT SHAPE (JSON):\n"
        "{\n"
        '  "title": "Short title",\n'
        '  "goal": "Clear, specific description of what needs to be accomplished",\n'
        '  "summary": "1-3 lines",\n'
        '  "deliverables": [\n'
        "    {\n"
        '      "type": "file",\n'
        '      "target": "DESKTOP|REPO|ARTIFACTS",\n'
        '      "filename": "relative/path/or/name",\n'
        '      "action": "add|modify|delete",\n'
        '      "content": "string or null",\n'
        '      "must_exist": true,\n'
        '      "allow_create": false\n'
        "    }\n"
        "  ],\n"
        '  "requirements": {"must": ["..."], "should": ["..."], "can": ["..."]},\n'
        '  "constraints": {"...": "..."},\n'
        '  "acceptance_tests": ["..."],\n'
        '  "open_questions": ["..."],\n'
        '  "open_issues": ["..."],\n'
        '  "recommendations": ["..."],\n'
        '  "repo_snapshot": {}\n'
        "}\n"
    )

    # User prompt: keep minimal; the heavy context travels via constraints_hint/weaver_spec_json
    prompt = (
        "USER_INTENT (may include clarification answers):\n"
        f"{(user_intent or '').strip()}\n\n"
        "If constraints_hint contains weaver_spec_json, treat it as the source-of-truth spec to validate/refine.\n"
        "Output only the JSON object in the shape described.\n"
    )

    from app.llm.streaming import call_llm_text  # must exist in repo

    _append_event(job_root, job_id, {
        "event": "SPEC_GATE_START",
        "job_id": job_id,
        "provider": provider_id,
        "model": model_id,
        "ts": _utc_ts(),
        "status": "ok",
    })

    llm_text = await call_llm_text(
        provider=provider_id,
        model=model_id,
        system_prompt=system_prompt,
        user_prompt=prompt,
        repo_snapshot=repo_snapshot,
        constraints_hint=constraints_hint,
        **kwargs,
    )

    draft = parse_spec_gate_output(llm_text)

    if not _is_draft_meaningful(draft):
        _append_event(job_root, job_id, {
            "event": "SPEC_GATE_DRAFT_EMPTY",
            "job_id": job_id,
            "ts": _utc_ts(),
            "status": "error",
        })
        raise RuntimeError("Spec Gate produced empty draft")

    created_at = datetime.now(timezone.utc).isoformat()

    spec_payload = {
        "spec_id": None,  # filled by _write_spec_file
        "spec_version": "1",
        "title": draft.title,
        "goal": draft.goal,
        "summary": draft.summary,
        "deliverables": draft.deliverables,
        "requirements": draft.requirements,
        "constraints": draft.constraints,
        "acceptance_criteria": draft.acceptance_tests,
        "open_questions": draft.open_questions,
        "open_issues": draft.open_issues,
        "recommendations": draft.recommendations,
        "repo_snapshot": draft.repo_snapshot or repo_snapshot,
        "timestamp": _utc_ts(),
        "created_at": created_at,
        "generator_model": f"{provider_id}/{model_id}",
        "created_by_model": f"{provider_id}/{model_id}",
        "provenance": {
            "source": "weaver" if has_weaver else "user_intent",
            "project_id": constraints_hint.get("project_id"),
        },
    }

    spec_id, spec_hash = _write_spec_file(job_root, job_id, spec_payload)

    _append_event(job_root, job_id, {
        "event": "SPEC_CREATED",
        "job_id": job_id,
        "spec_id": spec_id,
        "spec_hash": spec_hash,
        "open_questions_count": len(draft.open_questions),
        "has_deliverables": bool(draft.deliverables),
        "ts": _utc_ts(),
        "status": "ok",
    })

    return spec_id, spec_hash, draft.open_questions


__all__ = [
    "run_spec_gate",
    "detect_user_questions",
    "_artifact_root",
    "_append_event",
    "_utc_ts",
    "parse_spec_gate_output",
]
