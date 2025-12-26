# FILE: app/pot_spec/spec_gate.py
from __future__ import annotations

import json
import os
import re
from typing import Optional, Tuple, Any

from sqlalchemy.orm import Session

from app.pot_spec.schemas import PoTSpecDraft
from app.pot_spec.service import create_spec_from_draft
from app.pot_spec.ledger import append_event
from app.pot_spec.service import get_job_artifact_root

# Provider registry (non-stream)
from app.providers.registry import llm_call


_SPEC_GATE_SYSTEM_PROMPT = """You are Spec Gate.
You MUST output a single JSON object and NOTHING ELSE.

Purpose:
- Convert the user's intent into a precise implementation spec.
- The ONLY place allowed to ask the user questions is via open_questions[] in this JSON.

Rules:
- Output must be valid JSON (no trailing commas, no comments).
- Keep language plain and direct.
- Separate open_questions (things the user must answer) from recommendations (optional suggestions).
- If the request is underspecified, put the missing info into open_questions[].
- If reroute_reason is provided, you MUST include at least one item in open_questions[].

Schema (JSON object):
{
  "goal": "string",
  "requirements": { "must": ["..."], "should": ["..."], "can": ["..."] },
  "constraints": { ... },
  "acceptance_tests": ["..."],
  "open_questions": ["..."],
  "recommendations": ["..."],
  "repo_snapshot": { ... } | null
}
"""


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract first JSON object from model output."""
    if not text:
        raise ValueError("empty model output")

    s = text.strip()

    # Fast-path: starts with JSON object
    if s.startswith("{") and s.endswith("}"):
        return json.loads(s)

    # Otherwise, attempt to find first {...} block
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        raise ValueError("no JSON object found in output")
    return json.loads(m.group(0))


def parse_spec_gate_output(text: str) -> PoTSpecDraft:
    obj = _extract_json_object(text)
    return PoTSpecDraft(**obj)


def detect_user_questions(text: str) -> bool:
    """Hard policy: any question mark in assistant output counts as asking the user a question."""
    if not text:
        return False
    return "?" in text


async def run_spec_gate(
    db: Session,
    *,
    job_id: str,
    user_intent: str,
    provider_id: str = "openai",
    model_id: str,
    repo_snapshot: Optional[dict[str, Any]] = None,
    constraints_hint: Optional[dict[str, Any]] = None,
    reroute_reason: Optional[str] = None,
    downstream_output_excerpt: Optional[str] = None,
) -> Tuple[str, str, list[str]]:
    """Run Spec Gate via provider registry, persist spec, and emit ledger events.

    Returns: (spec_id, spec_hash, open_questions)
    """
    job_root = get_job_artifact_root()
    append_event(
        job_artifact_root=job_root,
        job_id=job_id,
        event={
            "event": "SPEC_GATE_STARTED",
            "job_id": job_id,
            "provider": provider_id,
            "model": model_id,
            "inputs": {
                "user_intent_chars": len(user_intent or ""),
                "reroute_reason": reroute_reason,
            },
            "status": "ok",
        },
    )

    # Build user message for Spec Gate
    user_parts = [
        "USER_INTENT:\n" + (user_intent or "").strip(),
    ]
    if constraints_hint:
        user_parts.append("CONSTRAINTS_HINT:\n" + json.dumps(constraints_hint, ensure_ascii=False, indent=2))
    if repo_snapshot:
        user_parts.append("REPO_SNAPSHOT:\n" + json.dumps(repo_snapshot, ensure_ascii=False, indent=2))
    if reroute_reason:
        user_parts.append("REROUTE_REASON:\n" + reroute_reason.strip())
    if downstream_output_excerpt:
        user_parts.append("DOWNSTREAM_OUTPUT_EXCERPT:\n" + downstream_output_excerpt.strip())

    messages = [
        {"role": "user", "content": "\n\n".join(user_parts)}
    ]

    result = await llm_call(
        provider_id=provider_id,
        model_id=model_id,
        messages=messages,
        system_prompt=_SPEC_GATE_SYSTEM_PROMPT,
        temperature=float(os.getenv("SPEC_GATE_TEMPERATURE", "1.0")),
        max_tokens=1200,
        timeout_seconds=120,
        data_sensitivity_constraint="internal",
        allowed_tools=[],
        forbidden_tools=[],
    )

    # If provider failed, produce a minimal draft with a single question
    if getattr(result, "status", None) is None or str(result.status) != "success":
        draft = PoTSpecDraft(
            goal=(user_intent or "").strip(),
            requirements={"must": [], "should": [], "can": []},
            constraints=constraints_hint or {},
            acceptance_tests=[],
            open_questions=[
                "Spec Gate failed to produce a valid response. Please re-run the request, or paste the error output."
            ],
            recommendations=[],
            repo_snapshot=repo_snapshot,
        )
        created_by = f"{provider_id}/{model_id}"
        spec = create_spec_from_draft(db, job_id, draft, created_by_model=created_by)
        append_event(
            job_artifact_root=job_root,
            job_id=job_id,
            event={
                "event": "SPEC_QUESTIONS_EMITTED",
                "job_id": job_id,
                "spec_id": spec.spec_id,
                "spec_hash": spec.spec_hash,
                "open_questions_count": len(draft.open_questions),
                "status": "error",
                "error": getattr(result, "error_message", "unknown_error"),
            },
        )
        return spec.spec_id, spec.spec_hash, draft.open_questions

    raw_text = (getattr(result, "content", None) or "").strip()

    try:
        draft = parse_spec_gate_output(raw_text)
    except Exception as e:
        draft = PoTSpecDraft(
            goal=(user_intent or "").strip(),
            requirements={"must": [], "should": [], "can": []},
            constraints=constraints_hint or {},
            acceptance_tests=[],
            open_questions=[
                f"Spec Gate returned invalid JSON. Please re-run. Parse error: {type(e).__name__}"
            ],
            recommendations=[],
            repo_snapshot=repo_snapshot,
        )

    # Enforce reroute rule
    if reroute_reason and not (draft.open_questions or []):
        draft.open_questions = [
            "Downstream stage attempted to ask user questions. Please answer any missing details needed to proceed."
        ]

    created_by = f"{provider_id}/{model_id}"
    spec = create_spec_from_draft(db, job_id, draft, created_by_model=created_by)

    if draft.open_questions:
        append_event(
            job_artifact_root=job_root,
            job_id=job_id,
            event={
                "event": "SPEC_QUESTIONS_EMITTED",
                "job_id": job_id,
                "spec_id": spec.spec_id,
                "spec_hash": spec.spec_hash,
                "open_questions_count": len(draft.open_questions),
                "status": "ok",
            },
        )
    else:
        append_event(
            job_artifact_root=job_root,
            job_id=job_id,
            event={
                "event": "SPEC_FINALIZED",
                "job_id": job_id,
                "spec_id": spec.spec_id,
                "spec_hash": spec.spec_hash,
                "status": "ok",
            },
        )

    return spec.spec_id, spec.spec_hash, list(draft.open_questions or [])
