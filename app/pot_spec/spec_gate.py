# FILE: app/pot_spec/spec_gate.py
"""PoT Spec Gate (Block 2)

Purpose
- Turn a user's request into a small, machine-checkable PoT spec draft (JSON)
- Persist spec_v1.json under jobs/<job_id>/spec/
- Emit ledger events under jobs/<job_id>/ledger/events.ndjson
- Return (spec_id, spec_hash, open_questions)

Design constraints
- Must not require JSON-only outputs from downstream stages.
- Spec Gate itself SHOULD output JSON and be robust to "almost JSON" (code fences, pre/post text).
- On provider failure / unparsable output: still write a fallback spec_v1.json so the system does not stall.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

# Optional imports (keep Spec Gate resilient if other modules move)
try:
    from sqlalchemy.orm import Session  # type: ignore
except Exception:  # pragma: no cover
    Session = Any  # type: ignore

try:
    from app.providers.registry import llm_call  # type: ignore
except Exception:  # pragma: no cover
    llm_call = None  # type: ignore

try:
    # Preferred: use shared artifact root helper if present
    from app.pot_spec.service import get_job_artifact_root  # type: ignore
except Exception:  # pragma: no cover
    get_job_artifact_root = None  # type: ignore


# =============================================================================
# Helpers
# =============================================================================

def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _artifact_root() -> str:
    if callable(get_job_artifact_root):
        try:
            return str(get_job_artifact_root())
        except Exception:
            pass
    # Safe fallback: match service.py default
    return os.path.abspath(os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs").strip() or "jobs")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _append_event(job_artifact_root: str, job_id: str, event: dict[str, Any]) -> None:
    """Append a ledger event. Uses app.pot_spec.ledger.append_event if available, else writes NDJSON directly."""
    try:
        from app.pot_spec.ledger import append_event  # type: ignore

        append_event(job_artifact_root=job_artifact_root, job_id=job_id, event=event)
        return
    except Exception:
        pass

    ledger_dir = os.path.join(job_artifact_root, job_id, "ledger")
    _ensure_dir(ledger_dir)
    ledger_file = os.path.join(ledger_dir, "events.ndjson")
    with open(ledger_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _canonical_json_bytes(obj: Any) -> bytes:
    # Canonical bytes: stable key order, no whitespace variance.
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_spec_file(job_artifact_root: str, job_id: str, spec_version: int, payload: dict[str, Any]) -> tuple[str, str]:
    spec_dir = os.path.join(job_artifact_root, job_id, "spec")
    _ensure_dir(spec_dir)
    path = os.path.join(spec_dir, f"spec_v{spec_version}.json")

    raw = _canonical_json_bytes(payload)
    with open(path, "wb") as f:
        f.write(raw)

    spec_hash = hashlib.sha256(raw).hexdigest()
    return path, spec_hash


def _extract_json_object(text: str) -> Optional[str]:
    """Extract the first JSON object from text (handles code fences and pre/post text)."""
    if not text:
        return None

    # Remove common fences
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)

    # Fast path: whole string is JSON
    if t.startswith("{") and t.endswith("}"):
        return t

    # Heuristic: find the first {...} block (balanced-ish, best effort)
    start = t.find("{")
    if start == -1:
        return None

    # Scan to find a plausible matching closing brace.
    depth = 0
    for i in range(start, len(t)):
        ch = t[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = t[start : i + 1]
                return candidate
    return None


# =============================================================================
# Public API
# =============================================================================

def detect_user_questions(text: str) -> bool:
    """Heuristic only: true if the user text likely contains questions to resolve before proceeding."""
    if not text:
        return False
    # '?' is the strongest signal; keep it simple and stable.
    return "?" in text


@dataclass
class _SpecGateDraft:
    goal: str
    requirements: dict[str, list[str]]
    constraints: dict[str, Any]
    acceptance_tests: list[str]
    open_questions: list[str]
    recommendations: list[str]
    repo_snapshot: Optional[dict[str, Any]] = None


def parse_spec_gate_output(text: str) -> _SpecGateDraft:
    """Parse model output into a draft.

    Expected JSON object keys (minimal):
      goal: str
      requirements: { must: [..], should: [..], can: [..] }
      constraints: { ... }
      acceptance_tests: [..]
      open_questions: [..]
      recommendations: [..]
      repo_snapshot: optional object
    """
    raw = _extract_json_object(text)
    if not raw:
        raise ValueError("Spec Gate output was not valid JSON")

    data = json.loads(raw)

    goal = str(data.get("goal") or "").strip()
    if not goal:
        raise ValueError("Spec Gate JSON missing non-empty 'goal'")

    req = data.get("requirements") or {}
    requirements = {
        "must": list(req.get("must") or []),
        "should": list(req.get("should") or []),
        "can": list(req.get("can") or []),
    }

    constraints = dict(data.get("constraints") or {})
    acceptance_tests = list(data.get("acceptance_tests") or [])
    open_questions = list(data.get("open_questions") or [])
    recommendations = list(data.get("recommendations") or [])
    repo_snapshot = data.get("repo_snapshot")
    if repo_snapshot is not None and not isinstance(repo_snapshot, dict):
        repo_snapshot = None

    return _SpecGateDraft(
        goal=goal,
        requirements=requirements,
        constraints=constraints,
        acceptance_tests=acceptance_tests,
        open_questions=open_questions,
        recommendations=recommendations,
        repo_snapshot=repo_snapshot,
    )


async def run_spec_gate(
    db: Session,
    job_id: str,
    user_intent: str,
    provider_id: str,
    model_id: str,
    repo_snapshot: Optional[dict[str, Any]] = None,
    constraints_hint: Optional[dict[str, Any]] = None,
    downstream_output_excerpt: Optional[str] = None,
    reroute_reason: Optional[str] = None,
) -> tuple[str, str, list[str]]:
    """Run Spec Gate and persist spec_v1.json.

    Always writes a spec_v1.json (fallback if provider fails/unparsable output).
    Returns (spec_id, spec_hash, open_questions).
    """
    job_root = _artifact_root()
    spec_id = str(uuid4())
    spec_version = 1
    created_at_iso = datetime.now(timezone.utc).isoformat()
    created_by_model = f"{provider_id}/{model_id}"

    _append_event(
        job_root,
        job_id,
        {
            "event": "SPEC_GATE_STARTED",
            "job_id": job_id,
            "provider": provider_id,
            "model": model_id,
            "inputs": {"reroute_reason": reroute_reason, "user_intent_chars": len(user_intent or "")},
            "status": "ok",
            "ts": _utc_ts(),
        },
    )

    # Default constraints: keep consistent with existing behavior.
    constraints: dict[str, Any] = {"stability_accuracy": "high", "allowed_tools": "free_only"}
    if isinstance(constraints_hint, dict):
        # Shallow-merge hints, but never remove required keys.
        constraints.update({k: v for k, v in constraints_hint.items() if v is not None})

    draft: Optional[_SpecGateDraft] = None
    raw_text: str = ""

    if llm_call is not None:
        system = (
            "You are Spec Gate. Convert the user's intent into a PoT spec DRAFT as a single JSON object.\n"
            "Output ONLY JSON (no markdown), matching this shape:\n"
            "{\n"
            '  "goal": "…",\n'
            '  "requirements": {"must": [...], "should": [...], "can": [...]},\n'
            '  "constraints": {...},\n'
            '  "acceptance_tests": [...],\n'
            '  "open_questions": [...],\n'
            '  "recommendations": [...]\n'
            "}\n"
            "Rules:\n"
            "- Keep lists short and precise.\n"
            "- If the user's intent is already clear, open_questions MUST be [].\n"
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_intent or ""},
        ]

        try:
            result = await llm_call(
                provider_id=provider_id,
                model_id=model_id,
                messages=messages,
                temperature=0.7,  # keep default-ish; some models reject non-default values anyway
                max_tokens=1200,
                timeout_seconds=120,
            )

            # Be tolerant to enum/string status forms
            status = getattr(result, "status", None)
            status_str = str(status).lower() if status is not None else ""
            raw_text = (getattr(result, "content", "") or "").strip()

            if not raw_text:
                _append_event(
                    job_root,
                    job_id,
                    {
                        "event": "SPEC_GATE_EMPTY_OUTPUT",
                        "job_id": job_id,
                        "provider": provider_id,
                        "model": model_id,
                        "provider_status": status_str,
                        "raw_excerpt": "",
                        "error": "Empty output from provider",
                        "status": "error",
                        "ts": _utc_ts(),
                    },
                )
            else:
                try:
                    draft = parse_spec_gate_output(raw_text)
                except Exception as e:
                    _append_event(
                        job_root,
                        job_id,
                        {
                            "event": "SPEC_GATE_PARSE_FAILED",
                            "job_id": job_id,
                            "provider": provider_id,
                            "model": model_id,
                            "error": str(e),
                            "raw_excerpt": raw_text[:2000],
                            "status": "error",
                            "ts": _utc_ts(),
                        },
                    )
        except Exception as e:
            _append_event(
                job_root,
                job_id,
                {
                    "event": "SPEC_GATE_CALL_FAILED",
                    "job_id": job_id,
                    "provider": provider_id,
                    "model": model_id,
                    "error": str(e),
                    "status": "error",
                    "ts": _utc_ts(),
                },
            )

    # Fallback draft if model failed / unparsable.
    if draft is None:
        draft = _SpecGateDraft(
            goal=(user_intent or "").strip() or "Create a PoT spec for the requested work.",
            requirements={"must": [], "should": [], "can": []},
            constraints=constraints,
            acceptance_tests=[],
            open_questions=[],
            recommendations=[],
            repo_snapshot=repo_snapshot if isinstance(repo_snapshot, dict) else None,
        )
    else:
        # Ensure constraints exist and contain required keys.
        draft.constraints = dict(draft.constraints or {})
        draft.constraints.setdefault("stability_accuracy", constraints.get("stability_accuracy", "high"))
        draft.constraints.setdefault("allowed_tools", constraints.get("allowed_tools", "free_only"))
        if draft.repo_snapshot is None and isinstance(repo_snapshot, dict):
            draft.repo_snapshot = repo_snapshot

    payload: dict[str, Any] = {
        "job_id": job_id,
        "spec_id": spec_id,
        "spec_version": spec_version,
        "parent_spec_id": None,
        "created_at": created_at_iso,
        "created_by_model": created_by_model,
        "goal": draft.goal,
        "requirements": draft.requirements,
        "constraints": draft.constraints,
        "acceptance_tests": draft.acceptance_tests,
        "open_questions": draft.open_questions,
        "recommendations": draft.recommendations,
        "repo_snapshot": draft.repo_snapshot,
    }

    spec_path, spec_hash = _write_spec_file(job_root, job_id, spec_version, payload)

    _append_event(
        job_root,
        job_id,
        {
            "event": "SPEC_CREATED",
            "job_id": job_id,
            "spec_id": spec_id,
            "spec_hash": spec_hash,
            "spec_version": spec_version,
            "parent_spec_id": None,
            "created_by_model": created_by_model,
            "inputs": {
                "draft_present": bool(draft),
                "open_questions_count": len(draft.open_questions or []),
                "repo_snapshot_present": bool(payload.get("repo_snapshot")),
            },
            "outputs": {"spec_file": os.path.relpath(spec_path, job_root)},
            "status": "ok",
            "ts": _utc_ts(),
        },
    )

    if draft.open_questions:
        _append_event(
            job_root,
            job_id,
            {
                "event": "SPEC_QUESTIONS_EMITTED",
                "job_id": job_id,
                "spec_id": spec_id,
                "spec_hash": spec_hash,
                "open_questions_count": len(draft.open_questions),
                "error": None,
                "status": "ok",
                "ts": _utc_ts(),
            },
        )

    return spec_id, spec_hash, list(draft.open_questions or [])
