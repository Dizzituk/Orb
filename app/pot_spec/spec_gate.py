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
- On provider failure / unparsable output: retry with fallback providers, then pause with questions if all fail.

Reliability improvements (2025-12):
- Retry with fallback providers (openai → anthropic → google) on empty/invalid output
- Validate spec quality before writing: require meaningful goal, requirements, or real questions
- If all retries fail, create spec with system-generated clarification questions (never silent empty success)

v1.2 (2025-12-28):
- Fixed: detect_user_questions now uses smarter heuristics (density, code block stripping)
- Fixed: Long architecture docs no longer trigger false positive question detection

v1.1 (2025-12-28):
- Fixed: Prompt now requires questions for vague/incomplete requests
- Added: Debug logging for artifact write tracing
- Fixed: No silent failures on artifact write
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, List, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# Optional imports (keep Spec Gate resilient if other modules move)
try:
    from sqlalchemy.orm import Session  # type: ignore
except Exception:  # pragma: no cover
    Session = Any  # type: ignore

try:
    from app.providers.registry import llm_call, is_provider_available  # type: ignore
except Exception:  # pragma: no cover
    llm_call = None  # type: ignore
    is_provider_available = None  # type: ignore

try:
    # Preferred: use shared artifact root helper if present
    from app.pot_spec.service import get_job_artifact_root  # type: ignore
except Exception:  # pragma: no cover
    get_job_artifact_root = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

# Fallback provider chain for retries (order matters)
FALLBACK_PROVIDERS: List[Tuple[str, str]] = [
    ("openai", os.getenv("OPENAI_SPEC_GATE_MODEL", "gpt-5.2-chat-latest")),
    ("anthropic", os.getenv("ANTHROPIC_SPEC_GATE_MODEL", "claude-opus-4-5-20251101")),
    ("google", os.getenv("GOOGLE_SPEC_GATE_MODEL", "gemini-3-pro-preview")),
]

# Maximum retry attempts across providers
MAX_SPEC_GATE_ATTEMPTS = 3


# =============================================================================
# Helpers
# =============================================================================

def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _artifact_root() -> str:
    if callable(get_job_artifact_root):
        try:
            root = str(get_job_artifact_root())
            logger.debug(f"[spec_gate] _artifact_root from service: {root}")
            return root
        except Exception as e:
            logger.warning(f"[spec_gate] get_job_artifact_root() failed: {e}")
    # Safe fallback: match service.py default
    fallback = os.path.abspath(os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs").strip() or "jobs")
    logger.debug(f"[spec_gate] _artifact_root fallback: {fallback}")
    return fallback


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

    # FIX: Added "jobs" segment to path
    ledger_dir = os.path.join(job_artifact_root, "jobs", job_id, "ledger")
    _ensure_dir(ledger_dir)
    ledger_file = os.path.join(ledger_dir, "events.ndjson")
    with open(ledger_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _canonical_json_bytes(obj: Any) -> bytes:
    # Canonical bytes: stable key order, no whitespace variance.
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_spec_file(job_artifact_root: str, job_id: str, spec_version: int, payload: dict[str, Any]) -> tuple[str, str]:
    # FIX: Added "jobs" segment to path
    spec_dir = os.path.join(job_artifact_root, "jobs", job_id, "spec")
    logger.debug(f"[spec_gate] _write_spec_file: spec_dir={spec_dir}")
    
    try:
        _ensure_dir(spec_dir)
    except Exception as e:
        logger.error(f"[spec_gate] Failed to create spec directory {spec_dir}: {e}")
        raise
    
    path = os.path.join(spec_dir, f"spec_v{spec_version}.json")
    logger.debug(f"[spec_gate] Writing spec to: {path}")

    raw = _canonical_json_bytes(payload)
    try:
        with open(path, "wb") as f:
            f.write(raw)
    except Exception as e:
        logger.error(f"[spec_gate] Failed to write spec file {path}: {e}")
        raise
    
    # Verify write succeeded
    if not os.path.exists(path):
        err_msg = f"[spec_gate] Spec file not created after write: {path}"
        logger.error(err_msg)
        raise IOError(err_msg)
    
    file_size = os.path.getsize(path)
    logger.info(f"[spec_gate] Spec file written successfully: {path} ({file_size} bytes)")

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


def _safe_excerpt(text: str, max_len: int = 2000) -> str:
    """Return a bounded excerpt suitable for logging."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... [truncated, {len(text)} total chars]"


# =============================================================================
# Public API
# =============================================================================

def detect_user_questions(text: str) -> bool:
    """Heuristic: true if text contains direct questions TO the user.
    
    Ignores:
    - Question marks inside code blocks
    - Rhetorical/documentation questions in large documents
    - Low question density (< 1 question per 2000 chars)
    
    Triggers on:
    - High question density in short text
    - Explicit user-directed phrases like "please clarify", "Q1:", etc.
    """
    if not text:
        return False
    
    # Strip code blocks (```...```) to avoid false positives from code comments
    text_no_code = re.sub(r'```[\s\S]*?```', '', text)
    text_no_code = re.sub(r'`[^`]+`', '', text_no_code)
    
    # Count question marks
    question_count = text_no_code.count('?')
    
    if question_count == 0:
        return False
    
    # For long documents (>5000 chars), require high question density
    # Architecture docs often have 1-2 "?" in comments/docs - that's not "asking questions"
    text_len = len(text_no_code)
    if text_len > 5000:
        # Require at least 1 question per 2000 chars to trigger
        density_threshold = text_len / 2000
        if question_count < density_threshold:
            return False
    
    # Check for explicit user-directed question phrases
    user_question_patterns = [
        r'\bplease (answer|clarify|confirm|specify|choose|select)\b',
        r'\bwhich (would you|do you|option)\b',
        r'\bdo you (want|need|prefer|require)\b',
        r'\bwould you (like|prefer)\b',
        r'\bcan you (clarify|specify|confirm)\b',
        r'\bwhat (would you|do you)\b',
        r'\b(Q\d+|Question \d+):\s*\w',  # Numbered questions like "Q1:" or "Question 1:"
    ]
    
    text_lower = text_no_code.lower()
    for pattern in user_question_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # For short text (<2000 chars), any question mark is suspicious
    if text_len < 2000 and question_count >= 1:
        return True
    
    # For medium text (2000-5000 chars), require 3+ questions
    if text_len < 5000 and question_count >= 3:
        return True
    
    return False


@dataclass
class _SpecGateDraft:
    goal: str
    requirements: dict[str, list[str]]
    constraints: dict[str, Any]
    acceptance_tests: list[str]
    open_questions: list[str]
    recommendations: list[str]
    repo_snapshot: Optional[dict[str, Any]] = None


def _is_draft_meaningful(draft: _SpecGateDraft) -> bool:
    """Check if a draft has enough content to be useful.
    
    A draft is meaningful if it has:
    - A non-trivial goal (more than just whitespace/boilerplate), OR
    - At least one requirement in must/should/can, OR
    - At least one real open question
    
    This prevents silently succeeding with completely empty specs.
    """
    # Check goal: must be non-empty and not just the user intent echoed back
    goal = (draft.goal or "").strip()
    if len(goal) < 10:
        # Too short to be a real spec goal
        has_goal = False
    else:
        has_goal = True
    
    # Check requirements
    reqs = draft.requirements or {}
    has_requirements = bool(
        reqs.get("must") or 
        reqs.get("should") or 
        reqs.get("can")
    )
    
    # Check open questions (real questions, not empty placeholders)
    questions = [q for q in (draft.open_questions or []) if q and len(q.strip()) > 5]
    has_questions = bool(questions)
    
    return has_goal or has_requirements or has_questions


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
    
    Raises ValueError if output is not valid JSON or missing required fields.
    """
    raw = _extract_json_object(text)
    if not raw:
        raise ValueError("Spec Gate output was not valid JSON (no JSON object found)")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Spec Gate output was not valid JSON: {e}")

    goal = str(data.get("goal") or "").strip()
    # Note: we no longer require non-empty goal here; validation happens in _is_draft_meaningful

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


def _get_available_providers() -> List[Tuple[str, str]]:
    """Return list of (provider_id, model_id) that are currently available."""
    available = []
    for provider_id, model_id in FALLBACK_PROVIDERS:
        if is_provider_available is not None and callable(is_provider_available):
            if is_provider_available(provider_id):
                available.append((provider_id, model_id))
        else:
            # If we can't check, assume available
            available.append((provider_id, model_id))
    return available


async def _attempt_spec_gate_call(
    provider_id: str,
    model_id: str,
    user_intent: str,
    job_id: str,
    job_root: str,
    attempt_number: int,
) -> Tuple[Optional[_SpecGateDraft], str, Optional[str]]:
    """Make a single Spec Gate LLM call attempt.
    
    Returns: (draft_or_none, raw_text, error_message_or_none)
    """
    if llm_call is None:
        return None, "", "llm_call not available"
    
    system = (
        "You are Spec Gate. Convert the user's intent into a PoT spec DRAFT as a single JSON object.\n"
        "Output ONLY JSON (no markdown fences, no explanation), matching this shape:\n"
        "{\n"
        '  "goal": "Clear, specific description of what needs to be accomplished",\n'
        '  "requirements": {\n'
        '    "must": ["list of absolute requirements"],\n'
        '    "should": ["list of recommended requirements"],\n'
        '    "can": ["list of optional nice-to-haves"]\n'
        "  },\n"
        '  "constraints": {"key": "value constraints"},\n'
        '  "acceptance_tests": ["list of testable acceptance criteria"],\n'
        '  "open_questions": ["questions you MUST ask if any critical info is missing"],\n'
        '  "recommendations": ["optional suggestions"]\n'
        "}\n\n"
        "CRITICAL RULES FOR HIGH-STAKES ARCHITECTURE:\n"
        "1. HANDLING USER ANSWERS:\n"
        "   - If the input contains 'Previous conversation:' with Q&A, the user is ANSWERING questions.\n"
        "   - Extract all answers and incorporate them into requirements/constraints.\n"
        "   - If all questions are answered satisfactorily, set open_questions to [].\n"
        "   - Only add NEW questions if answers revealed new ambiguities.\n\n"
        "2. INITIAL REQUESTS (no previous conversation):\n"
        "   - You MUST ask clarifying questions in open_questions if:\n"
        "     * Platform/language/framework is not specified\n"
        "     * Scale/performance requirements are unclear\n"
        "     * User type/audience is not defined\n"
        "     * Core features are ambiguous\n"
        "     * Deployment environment is unknown\n"
        "   - A vague request like 'create Tetris' REQUIRES questions about platform, language, features.\n\n"
        "3. GENERAL RULES:\n"
        "   - Extract EXPLICIT requirements from user intent into must/should/can lists.\n"
        "   - If user specified what they want clearly, populate requirements and leave open_questions empty.\n"
        "   - Output valid JSON only, no markdown code fences, no prose before/after.\n"
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
            temperature=0,  # Deterministic for reproducible spec hashes
            max_tokens=1500,
            timeout_seconds=120,
            reasoning={"effort": "none"},  # GPT-5.x: no reasoning overhead for extraction
        )

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
                    "attempt": attempt_number,
                    "provider_status": status_str,
                    "raw_excerpt": "",
                    "error": "Empty output from provider",
                    "status": "error",
                    "ts": _utc_ts(),
                },
            )
            return None, "", f"Empty output from {provider_id}/{model_id}"

        try:
            draft = parse_spec_gate_output(raw_text)
            
            # Validate draft quality
            if not _is_draft_meaningful(draft):
                _append_event(
                    job_root,
                    job_id,
                    {
                        "event": "SPEC_GATE_DRAFT_INSUFFICIENT",
                        "job_id": job_id,
                        "provider": provider_id,
                        "model": model_id,
                        "attempt": attempt_number,
                        "raw_excerpt": _safe_excerpt(raw_text),
                        "error": "Draft parsed but has no meaningful content (empty goal, requirements, and questions)",
                        "draft_summary": {
                            "goal_len": len(draft.goal or ""),
                            "must_count": len(draft.requirements.get("must", [])),
                            "should_count": len(draft.requirements.get("should", [])),
                            "can_count": len(draft.requirements.get("can", [])),
                            "questions_count": len(draft.open_questions or []),
                        },
                        "status": "error",
                        "ts": _utc_ts(),
                    },
                )
                return None, raw_text, f"Draft has no meaningful content from {provider_id}/{model_id}"
            
            return draft, raw_text, None
            
        except Exception as e:
            _append_event(
                job_root,
                job_id,
                {
                    "event": "SPEC_GATE_PARSE_FAILED",
                    "job_id": job_id,
                    "provider": provider_id,
                    "model": model_id,
                    "attempt": attempt_number,
                    "error": str(e),
                    "raw_excerpt": _safe_excerpt(raw_text),
                    "status": "error",
                    "ts": _utc_ts(),
                },
            )
            return None, raw_text, f"Parse failed: {e}"

    except Exception as e:
        _append_event(
            job_root,
            job_id,
            {
                "event": "SPEC_GATE_CALL_FAILED",
                "job_id": job_id,
                "provider": provider_id,
                "model": model_id,
                "attempt": attempt_number,
                "error": str(e),
                "status": "error",
                "ts": _utc_ts(),
            },
        )
        return None, "", f"LLM call failed: {e}"


async def run_spec_gate(
    db: Session,
    *args: Any,
    **kwargs: Any,
) -> tuple[str, str, list[str]]:
    """Run Spec Gate and persist spec_v1.json.

    This function sits between a user's free-form request and the high-stakes pipeline. It turns a user
    intent into a canonical PoT spec (spec_v1.json) and returns:

      (spec_id, spec_hash, open_questions)

    Compatibility:
      - Accepts both keyword-style calls:
          run_spec_gate(db, job_id=..., user_intent=..., provider_id=..., model_id=...)
      - And legacy positional calls:
          run_spec_gate(db, job_id, user_intent, provider_id, model_id, ...)
      - If both positional and keyword are provided for the same field, the keyword wins and a ledger
        warning event is written.

    Reliability:
      - Retries across fallback providers if primary fails (empty output, parse error, etc.)
      - Validates draft quality before accepting
      - If all retries fail, creates spec with system-generated clarification questions
      - Never silently succeeds with empty/useless spec
    """

    # -------------------------------------------------------------------------
    # Parameter normalization (avoid "got multiple values for argument" issues)
    # -------------------------------------------------------------------------
    job_id_kw = kwargs.pop("job_id", None)
    user_intent_kw = kwargs.pop("user_intent", None)
    provider_id_kw = kwargs.pop("provider_id", None)
    model_id_kw = kwargs.pop("model_id", None)

    repo_snapshot = kwargs.pop("repo_snapshot", None)
    constraints_hint = kwargs.pop("constraints_hint", None)
    downstream_output_excerpt = kwargs.pop("downstream_output_excerpt", None)
    reroute_reason = kwargs.pop("reroute_reason", None)

    pos = list(args)
    conflicts: dict[str, dict[str, Any]] = {}

    def _pick(name: str, pos_val: Any, kw_val: Any) -> Any:
        if kw_val is None:
            return pos_val
        if pos_val is None or pos_val == kw_val:
            return kw_val
        conflicts[name] = {"positional": pos_val, "keyword": kw_val}
        return kw_val

    job_id = _pick("job_id", pos[0] if len(pos) > 0 else None, job_id_kw)
    user_intent = _pick("user_intent", pos[1] if len(pos) > 1 else None, user_intent_kw)
    provider_id = _pick("provider_id", pos[2] if len(pos) > 2 else None, provider_id_kw)
    model_id = _pick("model_id", pos[3] if len(pos) > 3 else None, model_id_kw)

    extra_positional = pos[4:] if len(pos) > 4 else []
    extra_kw_keys = list(kwargs.keys())

    if job_id is None or user_intent is None or provider_id is None or model_id is None:
        raise ValueError(
            "run_spec_gate requires job_id, user_intent, provider_id, model_id (positional or keyword)"
        )

    job_root = _artifact_root()

    if conflicts or extra_positional or extra_kw_keys:
        # Best-effort: write a warning to the ledger, but never fail Spec Gate for this.
        _append_event(
            job_root,
            str(job_id),
            {
                "event": "SPEC_GATE_CALL_SHAPE",
                "status": "warn",
                "conflicts": conflicts,
                "extra_positional": extra_positional,
                "extra_kw_keys": extra_kw_keys,
                "ts": _utc_ts(),
            },
        )

    # Normalize types (avoid surprising JSON serialization issues)
    job_id = str(job_id)
    user_intent = str(user_intent)
    provider_id = str(provider_id)
    model_id = str(model_id)

    job_root = _artifact_root()
    spec_id = str(uuid4())
    spec_version = 1
    created_at_iso = datetime.now(timezone.utc).isoformat()

    # Default constraints: keep consistent with existing behavior.
    constraints: dict[str, Any] = {"stability_accuracy": "high", "allowed_tools": "free_only"}
    if isinstance(constraints_hint, dict):
        # Shallow-merge hints, but never remove required keys.
        constraints.update({k: v for k, v in constraints_hint.items() if v is not None})

    # -------------------------------------------------------------------------
    # Build provider attempt order: requested provider first, then fallbacks
    # -------------------------------------------------------------------------
    providers_to_try: List[Tuple[str, str]] = [(provider_id, model_id)]
    available_fallbacks = _get_available_providers()
    for fb_provider, fb_model in available_fallbacks:
        if fb_provider != provider_id:  # Don't duplicate
            providers_to_try.append((fb_provider, fb_model))
    
    # Cap total attempts
    providers_to_try = providers_to_try[:MAX_SPEC_GATE_ATTEMPTS]

    _append_event(
        job_root,
        job_id,
        {
            "event": "SPEC_GATE_STARTED",
            "job_id": job_id,
            "primary_provider": provider_id,
            "primary_model": model_id,
            "fallback_chain": [f"{p}/{m}" for p, m in providers_to_try],
            "inputs": {"reroute_reason": reroute_reason, "user_intent_chars": len(user_intent or "")},
            "status": "ok",
            "ts": _utc_ts(),
        },
    )

    # -------------------------------------------------------------------------
    # Attempt Spec Gate calls with retry/fallback
    # -------------------------------------------------------------------------
    draft: Optional[_SpecGateDraft] = None
    last_raw_text: str = ""
    all_errors: List[str] = []
    successful_provider: Optional[str] = None
    successful_model: Optional[str] = None

    for attempt_idx, (try_provider, try_model) in enumerate(providers_to_try, start=1):
        _append_event(
            job_root,
            job_id,
            {
                "event": "SPEC_GATE_ATTEMPT",
                "job_id": job_id,
                "attempt": attempt_idx,
                "provider": try_provider,
                "model": try_model,
                "status": "started",
                "ts": _utc_ts(),
            },
        )
        
        attempt_draft, raw_text, error_msg = await _attempt_spec_gate_call(
            provider_id=try_provider,
            model_id=try_model,
            user_intent=user_intent,
            job_id=job_id,
            job_root=job_root,
            attempt_number=attempt_idx,
        )
        
        if raw_text:
            last_raw_text = raw_text
        
        if attempt_draft is not None:
            draft = attempt_draft
            successful_provider = try_provider
            successful_model = try_model
            _append_event(
                job_root,
                job_id,
                {
                    "event": "SPEC_GATE_ATTEMPT_SUCCESS",
                    "job_id": job_id,
                    "attempt": attempt_idx,
                    "provider": try_provider,
                    "model": try_model,
                    "status": "ok",
                    "ts": _utc_ts(),
                },
            )
            break
        else:
            all_errors.append(f"Attempt {attempt_idx} ({try_provider}/{try_model}): {error_msg}")
            _append_event(
                job_root,
                job_id,
                {
                    "event": "SPEC_GATE_ATTEMPT_FAILED",
                    "job_id": job_id,
                    "attempt": attempt_idx,
                    "provider": try_provider,
                    "model": try_model,
                    "error": error_msg,
                    "status": "error",
                    "ts": _utc_ts(),
                },
            )

    # -------------------------------------------------------------------------
    # Handle all-attempts-failed case: create degraded spec with questions
    # -------------------------------------------------------------------------
    created_by_model: str
    is_degraded = False
    
    if draft is None:
        is_degraded = True
        # All retries failed - create spec with system-generated clarification questions
        system_questions = [
            "Spec Gate was unable to parse your request into a structured specification. Please clarify:",
            "1. What is the primary goal or outcome you want to achieve?",
            "2. What are the key requirements (must-have features)?",
            "3. Are there any specific constraints or limitations to consider?",
        ]
        
        if last_raw_text:
            system_questions.append(
                f"(The system received this response but couldn't parse it: {_safe_excerpt(last_raw_text, 500)})"
            )
        
        draft = _SpecGateDraft(
            goal=f"[SPEC GATE FAILED] Original request: {_safe_excerpt(user_intent, 500)}",
            requirements={"must": [], "should": [], "can": []},
            constraints=constraints,
            acceptance_tests=[],
            open_questions=system_questions,
            recommendations=[],
            repo_snapshot=repo_snapshot if isinstance(repo_snapshot, dict) else None,
        )
        created_by_model = "spec_gate/degraded_fallback"
        
        _append_event(
            job_root,
            job_id,
            {
                "event": "SPEC_GATE_ALL_ATTEMPTS_FAILED",
                "job_id": job_id,
                "total_attempts": len(providers_to_try),
                "errors": all_errors,
                "fallback_action": "created_degraded_spec_with_questions",
                "open_questions_count": len(system_questions),
                "status": "degraded",
                "ts": _utc_ts(),
            },
        )
    else:
        created_by_model = f"{successful_provider}/{successful_model}"
        # Ensure constraints exist and contain required keys.
        draft.constraints = dict(draft.constraints or {})
        draft.constraints.setdefault("stability_accuracy", constraints.get("stability_accuracy", "high"))
        draft.constraints.setdefault("allowed_tools", constraints.get("allowed_tools", "free_only"))
        if draft.repo_snapshot is None and isinstance(repo_snapshot, dict):
            draft.repo_snapshot = repo_snapshot

    # -------------------------------------------------------------------------
    # Build and write spec payload
    # -------------------------------------------------------------------------
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
    
    # Add degraded flag if applicable
    if is_degraded:
        payload["_spec_gate_degraded"] = True
        payload["_spec_gate_errors"] = all_errors

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
            "is_degraded": is_degraded,
            "inputs": {
                "draft_present": bool(draft),
                "open_questions_count": len(draft.open_questions or []),
                "repo_snapshot_present": bool(payload.get("repo_snapshot")),
                "attempts_made": len(providers_to_try) if is_degraded else (
                    providers_to_try.index((successful_provider, successful_model)) + 1 
                    if successful_provider else 1
                ),
            },
            "outputs": {"spec_file": os.path.relpath(spec_path, job_root)},
            "status": "degraded" if is_degraded else "ok",
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
                "is_system_generated": is_degraded,
                "error": None,
                "status": "ok",
                "ts": _utc_ts(),
            },
        )

    return spec_id, spec_hash, list(draft.open_questions or [])