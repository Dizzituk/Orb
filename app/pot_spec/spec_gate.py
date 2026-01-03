# FILE: app/pot_spec/spec_gate.py
"""
Spec Gate - Converts user intent into a PoT spec DRAFT as a single JSON object.

This is the "singular point of truth" before the critical pipeline and overwatcher.

Key addition (2026-01):
- Adds optional "deliverables" field to the spec JSON so Overwatcher can deterministically
  choose a target file / action without guessing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# Artifact paths / ledger helpers
# =============================================================================

def _artifact_root() -> str:
    return os.path.abspath(os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"))

def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _append_event(job_root: str, job_id: str, event: dict) -> None:
    try:
        os.makedirs(os.path.join(job_root, "jobs", job_id), exist_ok=True)
        path = os.path.join(job_root, "jobs", job_id, "events.ndjson")
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
    """
    spec_id = spec_payload.get("spec_id") or os.urandom(16).hex()
    spec_payload["spec_id"] = spec_id
    spec_payload["spec_version"] = str(spec_payload.get("spec_version") or "1")

    spec_dir = os.path.join(job_root, "jobs", job_id, "spec")
    os.makedirs(spec_dir, exist_ok=True)
    spec_path = os.path.join(spec_dir, "spec_v1.json")

    raw = _canonical_json_bytes(spec_payload)
    spec_hash = _sha256_hex(raw)

    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec_payload, f, ensure_ascii=False, indent=2)

    return spec_id, spec_hash


# =============================================================================
# Draft schema
# =============================================================================

@dataclass
class _SpecGateDraft:
    goal: str
    deliverables: list[dict[str, Any]]
    requirements: dict
    constraints: dict
    acceptance_tests: list
    open_questions: list
    recommendations: list
    repo_snapshot: Optional[dict]


def _is_draft_meaningful(draft: _SpecGateDraft) -> bool:
    if draft.goal and isinstance(draft.goal, str) and draft.goal.strip():
        return True
    # If goal is missing but deliverables are present, still meaningful.
    if isinstance(draft.deliverables, list) and len(draft.deliverables) > 0:
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
            raise ValueError("No JSON object found in output")
        s = s[a : b + 1].strip()

    return json.loads(s)


def parse_spec_gate_output(text: str) -> _SpecGateDraft:
    """
    Parse model output into draft.

    Base shape (existing):
    {
      "goal": "...",
      "requirements": {"must": [...], "should": [...], "can": [...]},
      "constraints": {...},
      "acceptance_tests": [...],
      "open_questions": [...],
      "recommendations": [...],
      "repo_snapshot": {...}
    }

    NEW (preferred by Overwatcher):
      "deliverables": [
        {
          "type": "file",
          "target": "DESKTOP|REPO|ARTIFACTS",
          "filename": "relative/path/or/name",
          "action": "add|modify|delete",
          "content": "string or null",
          "must_exist": true,
          "allow_create": false
        }
      ]
    """
    data = _parse_json_from_text(text)

    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object")

    goal = str(data.get("goal") or "").strip()
    # Note: we no longer require non-empty goal here; validation happens in _is_draft_meaningful

    # Optional: deliverables (preferred by Overwatcher)
    deliverables_raw = data.get("deliverables")
    deliverables: list[dict[str, Any]] = []
    if isinstance(deliverables_raw, list):
        for d in deliverables_raw:
            if isinstance(d, dict):
                deliverables.append(d)
    # If missing, we keep [] and let downstream stages decide whether that is acceptable.

    requirements = data.get("requirements") or {}
    constraints = data.get("constraints") or {}
    acceptance_tests = data.get("acceptance_tests") or []
    open_questions = data.get("open_questions") or []
    recommendations = data.get("recommendations") or []
    repo_snapshot = data.get("repo_snapshot")

    if not isinstance(requirements, dict):
        requirements = {}
    if not isinstance(constraints, dict):
        constraints = {}
    if not isinstance(acceptance_tests, list):
        acceptance_tests = []
    if not isinstance(open_questions, list):
        open_questions = []
    if not isinstance(recommendations, list):
        recommendations = []
    if repo_snapshot is not None and not isinstance(repo_snapshot, dict):
        repo_snapshot = None

    return _SpecGateDraft(
        goal=goal,
        deliverables=deliverables,
        requirements=requirements,
        constraints=constraints,
        acceptance_tests=acceptance_tests,
        open_questions=open_questions,
        recommendations=recommendations,
        repo_snapshot=repo_snapshot,
    )


# =============================================================================
# Question detection (required by engine/high_stakes_stream imports)
# =============================================================================

_QUESTION_LINE_RE = re.compile(r"(^|\n)\s*(?:[-*]\s*)?(?:Q[:\s]|Question[:\s])", re.IGNORECASE)
_PLEASE_PROVIDE_RE = re.compile(r"\b(please\s+(?:provide|clarify|confirm)|can\s+you\s+provide|need\s+to\s+know)\b", re.IGNORECASE)

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
            if isinstance(oq, list) and any(str(x).strip() for x in oq):
                return True
    except Exception:
        pass

    # 2) Text heuristics
    lower = s.lower()
    if "clarification needed" in lower:
        return True
    if "spec gate was unable" in lower and "clarify" in lower:
        return True
    if "open_questions" in lower:
        # Some stages echo the key name even without valid JSON.
        return True

    if "?" in s:
        # Quick win: any explicit question mark
        return True

    if _QUESTION_LINE_RE.search(s):
        return True
    if _PLEASE_PROVIDE_RE.search(s):
        return True

    return False


# =============================================================================
# LLM call wrapper (your project already has this)
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
    """
    job_root = _artifact_root()

    # Build prompt (unchanged, only add deliverables in output schema)
    system_prompt = (
        "You are Spec Gate.\n"
        "Your job is to convert the user's intent into a PoT spec DRAFT as a single JSON object.\n\n"
        "IMPORTANT:\n"
        "- Return ONLY JSON, no prose.\n"
        "- Do not wrap in backticks.\n"
        "- If ambiguous, ask questions via open_questions.\n\n"
        "OUTPUT SHAPE (JSON):\n"
        "{\n"
        '  "goal": "Clear, specific description of what needs to be accomplished",\n'
        '  "deliverables": [\n'
        '    {\n'
        '      "type": "file",\n'
        '      "target": "DESKTOP|REPO|ARTIFACTS",\n'
        '      "filename": "relative/path/or/name",\n'
        '      "action": "add|modify|delete",\n'
        '      "content": "string or null",\n'
        '      "must_exist": true,\n'
        '      "allow_create": false\n'
        '    }\n'
        '  ],\n'
        '  "requirements": {\n'
        '    "must": ["..."],\n'
        '    "should": ["..."],\n'
        '    "can": ["..."]\n'
        '  },\n'
        '  "constraints": { "platform": "...", "budget": "...", "latency": "...", "compliance": ["..."] },\n'
        '  "acceptance_tests": ["..."],\n'
        '  "open_questions": ["..."],\n'
        '  "recommendations": ["..."],\n'
        '  "repo_snapshot": { ... } \n'
        "}\n"
    )

    prompt = user_intent

    # NOTE: you already have an LLM call in your repo. Keep your existing implementation.
    from app.llm.streaming import call_llm_text  # (example) your project must already supply something like this

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
        # Hard fail: no goal and no deliverables means Overwatcher cannot act.
        _append_event(job_root, job_id, {
            "event": "SPEC_GATE_DRAFT_EMPTY",
            "job_id": job_id,
            "ts": _utc_ts(),
            "status": "error",
        })
        raise ValueError("Spec Gate produced an empty/meaningless draft (no goal, no deliverables)")

    spec_payload = {
        "spec_version": "1",
        "spec_id": None,  # filled in by _write_spec_file
        "goal": draft.goal,
        "deliverables": draft.deliverables,
        "requirements": draft.requirements,
        "constraints": draft.constraints,
        "acceptance_tests": draft.acceptance_tests,
        "open_questions": draft.open_questions,
        "recommendations": draft.recommendations,
        "repo_snapshot": draft.repo_snapshot or repo_snapshot,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by_model": f"{provider_id}/{model_id}",
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
