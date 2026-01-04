# FILE: app/pot_spec/spec_gate_v2.py
"""
Spec Gate v2 (ASTRA)

Purpose
- Take Weaver's PoT draft (JSON) + the user's latest clarification message.
- Ask ONLY job-relevant clarification questions (max 3 rounds) when required fields are missing.
- Otherwise, emit a restart-safe SPoT (Singular Point of Truth) spec and persist it to the Specs DB.

Key fixes in this rewrite
1) HARD STOP bug fix (list-of-lists / non-dict items):
   - The previous crash came from code paths that assumed each output/step item is a dict
     and called `.get(...)`. In real runs, Weaver/user content can include strings, lists,
     or other shapes. This version normalizes everything into dict-shaped items before
     constructing SpecSchema.

2) DB persistence bug fix (create_spec signature mismatch / wrong payload):
   - Spec ID None / Spec Hash None means persistence either failed or returned a type we
     didn’t parse correctly.
   - This version:
       * builds BOTH content_dict + content_json
       * introspects create_spec(...) signature
       * tries common parameter names (content_json/content/spec/spec_schema/etc.)
       * parses return values (ORM object / dict / tuple)
       * only marks ready_for_pipeline=True if persisted AND spec_id is real

3) Redundant/repeated questions reduction:
   - The gate parses "output/steps/verify" sections when present.
   - If the user replies without section headers, it uses a small heuristic classifier
     to treat lines as outputs/steps/acceptance so it can progress instead of repeating.

Imported by app/llm/spec_gate_stream.py as:
    from app.pot_spec.spec_gate_v2 import run_spec_gate_v2, SpecGateResult
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies (keep import-safe)
# ---------------------------------------------------------------------------

try:
    from app.specs import service as specs_service
except Exception:  # pragma: no cover
    specs_service = None  # type: ignore

try:
    # In this codebase, SpecSchema is typically the dataclass / model named "Spec".
    from app.specs.schema import Spec as SpecSchema
except Exception:  # pragma: no cover
    SpecSchema = None  # type: ignore

try:
    from app.specs.schema import SpecStatus
except Exception:  # pragma: no cover
    SpecStatus = None  # type: ignore


# ---------------------------------------------------------------------------
# Public result type (imported by spec_gate_stream.py)
# ---------------------------------------------------------------------------

@dataclass
class SpecGateResult:
    # Control flow
    ready_for_pipeline: bool = False
    open_questions: List[str] = None  # type: ignore

    # Output
    spot_markdown: Optional[str] = None

    # Persistence
    db_persisted: bool = False
    spec_id: Optional[str] = None
    spec_hash: Optional[str] = None
    spec_version: Optional[int] = None  # spec-gate round marker (1..3)

    # Hard stop (should be rare)
    hard_stopped: bool = False
    hard_stop_reason: Optional[str] = None

    # Debug
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if self.open_questions is None:
            self.open_questions = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMMAND_RE = re.compile(r"(?im)^\s*astra,\s*command\s*:\s*.*$")

_SECTION_RE = re.compile(
    r"(?im)^\s*(outputs?|steps?|verify|verification|acceptance|evidence)\s*:\s*$"
)

_BULLET_RE = re.compile(r"(?m)^\s*(?:[-*]|(\d+)[.)])\s+(.*\S)\s*$")

_VERBISH_RE = re.compile(
    r"(?i)\b(go to|open|create|make|write|save|close|run|execute|verify|check|ensure|confirm)\b"
)

_ARTIFACTISH_RE = re.compile(r"(?i)\b(file|folder|directory|path|artifact|output)\b")

_VERIFYISH_RE = re.compile(r"(?i)\b(verify|verification|acceptance|evidence|check|confirm|should|must)\b")


def _strip_astra_command(text: str) -> str:
    """Remove lines like `Astra, command: ...` so parsing is stable."""
    if not text:
        return ""
    cleaned = _COMMAND_RE.sub("", text)
    return cleaned.strip()


def _extract_weaver_spec(constraints_hint: Optional[dict]) -> Tuple[Optional[dict], dict]:
    """
    Pull Weaver PoT JSON from constraints_hint["weaver_spec_json"].

    Returns (weaver_spec_json_or_none, provenance_dict).
    """
    constraints_hint = constraints_hint or {}
    weaver = constraints_hint.get("weaver_spec_json")
    prov: dict = {}
    for k in ("weaver_spec_id", "weaver_spec_hash", "weaver_spec_version"):
        if constraints_hint.get(k):
            prov[k] = constraints_hint[k]
    if isinstance(weaver, dict):
        return weaver, prov
    return None, prov


def _best_effort_title_and_objective(weaver_spec: Optional[dict], user_text: str) -> Tuple[str, str]:
    """Derive a stable title/objective without calling an LLM."""
    title = ""
    objective = ""

    if isinstance(weaver_spec, dict):
        for key in ("title", "name", "job_title"):
            v = weaver_spec.get(key)
            if isinstance(v, str) and v.strip():
                title = v.strip()
                break

        for key in ("objective", "goal", "summary", "intent", "description"):
            v = weaver_spec.get(key)
            if isinstance(v, str) and v.strip():
                objective = v.strip()
                break

    if not title:
        title = "Spec Gate Job"

    if not objective:
        lines = [ln.strip() for ln in (user_text or "").splitlines() if ln.strip()]
        objective = lines[0] if lines else "Define and execute the requested job."

    return title, objective


def _split_sections(user_text: str) -> Dict[str, str]:
    """
    Extract sections like:
      output: ...
      steps: ...
      verify: ...
    """
    text = (user_text or "").strip()
    if not text:
        return {}

    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return {}

    sections: Dict[str, str] = {}
    for i, m in enumerate(matches):
        key = m.group(1).lower()
        if key.startswith("output"):
            key = "output"
        elif key.startswith("step"):
            key = "steps"
        elif key.startswith("verif"):
            key = "verify"
        elif key.startswith("accept"):
            key = "acceptance"
        elif key.startswith("evid"):
            key = "evidence"

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections[key] = body

    return sections


def _bullets(block: str) -> List[str]:
    """Extract bullet/numbered lines; fallback to non-empty lines if none."""
    if not block:
        return []
    out: List[str] = []
    for m in _BULLET_RE.finditer(block):
        out.append(m.group(2).strip())
    if out:
        return out
    return [ln.strip() for ln in block.splitlines() if ln.strip()]


def _guess_output_name(line: str) -> str:
    """Best-effort extraction of an artifact name from a line."""
    if not line:
        return "artifact"

    m = re.search(r"(?i)\bcalled\s+([A-Za-z0-9_.-]+)\b", line)
    if m:
        return m.group(1)

    m = re.search(r"(?i)\bnamed\s+([A-Za-z0-9_.-]+)\b", line)
    if m:
        return m.group(1)

    m = re.search(r"(?i)\bfile\s+([A-Za-z0-9_.-]+)\b", line)
    if m:
        return m.group(1)

    return "artifact"


def _to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (list, tuple)):
        return " ".join(_to_text(i) for i in x if _to_text(i))
    if isinstance(x, dict):
        for k in ("description", "text", "name", "value"):
            if isinstance(x.get(k), str) and x.get(k).strip():
                return x.get(k).strip()
        return str(x)
    return str(x).strip()


def _coerce_output_items(items: Any) -> List[dict]:
    """
    Normalize outputs into a list of dicts:
      {"name","type","description","example","acceptance_criteria"}
    """
    if not items:
        return []

    if isinstance(items, dict):
        items = [items]
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, list):
        items = [items]

    out: List[dict] = []
    for it in items:
        if isinstance(it, dict):
            desc = _to_text(it.get("description")) or _to_text(it)
            name = _to_text(it.get("name")) or _guess_output_name(desc)
            typ = _to_text(it.get("type")) or ("file" if re.search(r"(?i)\bfile\b", desc) else "artifact")
            out.append(
                {
                    "name": name or "artifact",
                    "type": typ or "artifact",
                    "description": desc or "",
                    "example": it.get("example", None),
                    "acceptance_criteria": list(it.get("acceptance_criteria") or []),
                }
            )
            continue

        desc = _to_text(it)
        if not desc:
            continue
        name = _guess_output_name(desc)
        typ = "file" if re.search(r"(?i)\bfile\b", desc) else "artifact"
        out.append(
            {
                "name": name or "artifact",
                "type": typ,
                "description": desc,
                "example": None,
                "acceptance_criteria": [],
            }
        )

    return out


def _coerce_step_items(items: Any) -> List[dict]:
    """
    Normalize steps into a list of dicts:
      {"id","description","dependencies","notes"}
    """
    if not items:
        return []

    if isinstance(items, dict):
        items = [items]
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, list):
        items = [items]

    out: List[dict] = []
    n = 1
    for it in items:
        if isinstance(it, dict):
            desc = _to_text(it.get("description")) or _to_text(it)
            sid = _to_text(it.get("id")) or f"S{n}"
            out.append(
                {
                    "id": sid,
                    "description": desc or "",
                    "dependencies": list(it.get("dependencies") or []),
                    "notes": it.get("notes", None),
                }
            )
            n += 1
            continue

        desc = _to_text(it)
        if not desc:
            continue
        out.append({"id": f"S{n}", "description": desc, "dependencies": [], "notes": None})
        n += 1

    return out


def _coerce_acceptance_items(items: Any) -> List[str]:
    if not items:
        return []
    if isinstance(items, str):
        items = [items]
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        items = [items]
    out: List[str] = []
    for it in items:
        t = _to_text(it)
        if t:
            out.append(t)
    return out


def _parse_user_clarification(user_text: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (output_lines, step_lines, verify_lines).
    """
    sections = _split_sections(user_text)
    if sections:
        output_lines = _bullets(sections.get("output", ""))
        step_lines = _bullets(sections.get("steps", ""))
        verify_lines = (
            _bullets(sections.get("verify", ""))
            or _bullets(sections.get("acceptance", ""))
            or _bullets(sections.get("evidence", ""))
        )
        return output_lines, step_lines, verify_lines

    # Heuristic mode
    lines = _bullets(user_text)
    output_lines: List[str] = []
    step_lines: List[str] = []
    verify_lines: List[str] = []

    for ln in lines:
        if not ln:
            continue
        if _VERIFYISH_RE.search(ln):
            verify_lines.append(ln)
            continue

        if _ARTIFACTISH_RE.search(ln) and not _VERBISH_RE.search(ln):
            output_lines.append(ln)
            continue

        step_lines.append(ln)

    return output_lines, step_lines, verify_lines


def _derive_required_fields(
    weaver_spec: Optional[dict],
    output_lines: List[str],
    step_lines: List[str],
    verify_lines: List[str],
) -> List[str]:
    """Decide which job-relevant questions are still needed."""
    weaver_has_outputs = isinstance(weaver_spec, dict) and bool(weaver_spec.get("outputs"))
    weaver_has_steps = isinstance(weaver_spec, dict) and bool(weaver_spec.get("steps"))
    weaver_has_acceptance = isinstance(weaver_spec, dict) and bool(
        weaver_spec.get("acceptance_criteria") or weaver_spec.get("acceptance")
    )

    need_outputs = not (weaver_has_outputs or bool(output_lines))
    need_steps = not (weaver_has_steps or bool(step_lines))
    need_acceptance = not (weaver_has_acceptance or bool(verify_lines))

    questions: List[str] = []
    if need_outputs:
        questions.append("What exact output artifacts should exist when the job is done (file/folder names + locations)?")
    if need_steps:
        questions.append("What exact steps should the system take, in order? (keep it short)")
    if need_acceptance:
        questions.append("How should we verify success? (e.g., exact file content, exact path, overwrite behaviour)")

    return questions


def _build_spot_markdown(
    spec_title: str,
    objective: str,
    outputs: List[dict],
    steps: List[dict],
    acceptance: List[str],
    open_issues: List[str],
) -> str:
    md: List[str] = []
    md.append(f"# {spec_title}".strip())
    md.append("")
    md.append("## Objective")
    md.append(objective.strip())
    md.append("")
    md.append("## Outputs")
    if outputs:
        for o in outputs:
            name = _to_text(o.get("name")) or "artifact"
            desc = _to_text(o.get("description")) or ""
            md.append(f"- **{name}** — {desc}".strip())
    else:
        md.append("- (none specified)")
    md.append("")
    md.append("## Steps")
    if steps:
        for s in steps:
            sid = _to_text(s.get("id")) or ""
            desc = _to_text(s.get("description")) or ""
            md.append(f"- {sid}: {desc}".strip())
    else:
        md.append("- (none specified)")
    md.append("")
    md.append("## Verification / Acceptance Criteria")
    if acceptance:
        for a in acceptance:
            md.append(f"- {a}")
    else:
        md.append("- (none specified)")
    md.append("")
    md.append("## Open Issues")
    if open_issues:
        for oi in open_issues:
            md.append(f"- {oi}")
    else:
        md.append("- (none)")
    md.append("")
    return "\n".join(md).strip() + "\n"


def _make_spec_schema(
    title: str,
    objective: str,
    context: Dict[str, Any],
    outputs: List[dict],
    steps: List[dict],
    acceptance: List[str],
    round_n: int,
) -> Any:
    """
    Build a SpecSchema instance while avoiding fragile coercers.
    """
    if SpecSchema is None:
        raise RuntimeError("Spec schema module not available (app.specs.schema.Spec).")

    safe_context = dict(context or {})
    safe_context["spec_gate_round"] = round_n  # round marker, not DB spec version

    data = {
        "title": title,
        "objective": objective,
        "context": safe_context,
        "outputs": outputs,
        "steps": steps,
        "acceptance_criteria": acceptance,
        "spec_version": "v1",
    }

    try:
        return SpecSchema(**data)  # type: ignore[arg-type]
    except Exception:
        spec = SpecSchema()  # type: ignore[call-arg]
        for k, v in data.items():
            try:
                setattr(spec, k, v)
            except Exception:
                pass
        return spec


def _spec_to_dict(spec_schema: Any) -> Dict[str, Any]:
    """
    Produce a plain dict for persistence, regardless of schema implementation.
    """
    if spec_schema is None:
        return {}
    if isinstance(spec_schema, dict):
        return spec_schema
    for attr in ("to_dict", "model_dump", "dict"):
        fn = getattr(spec_schema, attr, None)
        if callable(fn):
            try:
                d = fn()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
    # last resort
    try:
        return {
            "title": getattr(spec_schema, "title", None),
            "objective": getattr(spec_schema, "objective", None),
            "context": getattr(spec_schema, "context", None),
            "outputs": getattr(spec_schema, "outputs", None),
            "steps": getattr(spec_schema, "steps", None),
            "acceptance_criteria": getattr(spec_schema, "acceptance_criteria", None),
        }
    except Exception:
        return {}


def _parse_create_spec_return(ret: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Handle ORM object / dict / tuple returns.
    """
    if ret is None:
        return None, None

    # tuple/list like (spec_id, spec_hash, ...)
    if isinstance(ret, (tuple, list)) and len(ret) >= 2:
        sid = ret[0]
        sh = ret[1]
        return (str(sid) if sid is not None else None, str(sh) if sh is not None else None)

    # dict-like
    if isinstance(ret, dict):
        sid = ret.get("spec_id") or ret.get("id")
        sh = ret.get("spec_hash") or ret.get("hash")
        return (str(sid) if sid is not None else None, str(sh) if sh is not None else None)

    # ORM-like
    sid = getattr(ret, "spec_id", None) or getattr(ret, "id", None)
    sh = getattr(ret, "spec_hash", None) or getattr(ret, "hash", None)
    return (str(sid) if sid is not None else None, str(sh) if sh is not None else None)


def _persist_spec_best_effort(
    db: Session,
    project_id: int,
    spec_schema: Any,
    *,
    job_id: Optional[str] = None,
    provider_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """
    Persist spec via specs_service.create_spec, but only pass kwargs it supports.

    Returns: (persisted_ok, spec_id, spec_hash, error_str)
    """
    if specs_service is None or not hasattr(specs_service, "create_spec"):
        return False, None, None, "specs_service.create_spec not available"

    fn = specs_service.create_spec  # type: ignore[attr-defined]

    # Build portable payloads
    content_dict = _spec_to_dict(spec_schema)
    content_json = json.dumps(content_dict, ensure_ascii=False, indent=2)
    fallback_hash = hashlib.sha256(content_json.encode("utf-8")).hexdigest()

    # Prefer a real enum if present, but don’t crash if not
    status_val = None
    try:
        status_val = SpecStatus.VALIDATED if SpecStatus is not None else "validated"  # type: ignore[attr-defined]
    except Exception:
        status_val = "validated"

    generator_model = None
    if provider_id and model_id:
        generator_model = f"{provider_id}/{model_id}"

    # Signature-driven kwargs builder
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        param_names = set(params.keys())
    except Exception:
        params = {}
        param_names = set()

    # Values we can supply (by name)
    values: Dict[str, Any] = {
        "db": db,
        "project_id": project_id,
        # IMPORTANT: DB spec version should not track rounds; keep it stable unless your schema says otherwise.
        "spec_version": 1,
        "status": status_val,
        "content_json": content_json,
        "content": content_dict,
        "content_dict": content_dict,
        "spec_json": content_dict,
        "schema_json": content_dict,
        "generator_model": generator_model,
        "model": generator_model,
        "source": "spec_gate_v2",
        "created_by": "spec_gate_v2",
        "job_id": job_id,
        "spec_schema": spec_schema,
        "spec": spec_schema,
        "schema": spec_schema,
    }

    def _filtered_kwargs(prefer_keys: List[str]) -> Dict[str, Any]:
        if param_names:
            out: Dict[str, Any] = {}
            for k in prefer_keys:
                if k in param_names and values.get(k) is not None:
                    out[k] = values[k]
            return out
        # no signature -> we can’t filter safely; return only “very common” keys
        out2: Dict[str, Any] = {}
        for k in prefer_keys:
            v = values.get(k)
            if v is not None:
                out2[k] = v
        return out2

    # Candidate call patterns (ordered)
    patterns: List[List[str]] = [
        # Content-first (most common in your earlier logs)
        ["db", "project_id", "spec_version", "status", "content_json", "generator_model", "source", "created_by", "job_id"],
        ["db", "project_id", "spec_version", "status", "content_dict", "generator_model", "source", "created_by", "job_id"],
        ["db", "project_id", "spec_version", "status", "content", "generator_model", "source", "created_by", "job_id"],
        # Schema-object style
        ["db", "project_id", "spec_version", "status", "spec_schema", "generator_model", "source", "created_by", "job_id"],
        ["db", "project_id", "spec_version", "status", "spec", "generator_model", "source", "created_by", "job_id"],
        ["db", "project_id", "spec_version", "status", "schema", "generator_model", "source", "created_by", "job_id"],
        # Minimal fallbacks
        ["db", "project_id", "content_json"],
        ["db", "project_id", "spec_schema"],
        ["db", "project_id", "spec"],
    ]

    last_err: Optional[str] = None

    for keys in patterns:
        kwargs = _filtered_kwargs(keys)

        # If we have a signature and this call would obviously miss required args, skip it.
        if params:
            missing_required = []
            for pname, p in params.items():
                if p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                    # we can satisfy it either via kwargs or via values (if we choose positional later)
                    if pname not in kwargs and pname not in values:
                        missing_required.append(pname)
            if missing_required:
                # We can’t satisfy required params at all; no point trying this pattern.
                continue

        try:
            ret = fn(**kwargs)
            spec_id, spec_hash = _parse_create_spec_return(ret)

            # If DB didn’t return a hash, at least provide a stable one for display/debug.
            if spec_hash is None:
                spec_hash = fallback_hash

            # Require a real id to claim persistence success.
            if spec_id:
                return True, spec_id, spec_hash, None

            last_err = "create_spec returned without a spec id"
        except TypeError as e:
            last_err = f"typeerror: {e}"
            continue
        except Exception as e:
            last_err = f"exception: {e}"
            continue

    return False, None, fallback_hash, last_err


def _update_status_best_effort(db: Session, spec_id: Any) -> None:
    if specs_service is None or not spec_id:
        return
    if not hasattr(specs_service, "update_spec_status"):
        return

    fn = specs_service.update_spec_status  # type: ignore[attr-defined]
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
    except Exception:
        params = {}

    # Prefer enum if available
    status_val = None
    try:
        status_val = SpecStatus.VALIDATED if SpecStatus is not None else "validated"  # type: ignore[attr-defined]
    except Exception:
        status_val = "validated"

    kwargs: Dict[str, Any] = {}
    if "db" in params:
        kwargs["db"] = db
    if "spec_id" in params:
        kwargs["spec_id"] = spec_id
    if "status" in params:
        kwargs["status"] = status_val
    if "validation_result" in params:
        kwargs["validation_result"] = {"spec_gate_v2": True}

    try:
        if kwargs:
            fn(**kwargs)
        else:
            fn(db, spec_id, status_val, {"spec_gate_v2": True})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

async def run_spec_gate_v2(
    db: Session,
    job_id: str,
    user_intent: str,
    provider_id: str,
    model_id: str,
    project_id: int,
    constraints_hint: Optional[dict] = None,
    spec_version: int = 1,
) -> SpecGateResult:
    """
    Spec Gate v2 entrypoint.

    Notes
    - provider_id/model_id are accepted for compatibility with the router.
      This v2 path is deterministic/low-token for now (no LLM calls).
    - Round cap is respected:
        rounds 1-2: ask missing-field questions
        round 3: always finalize and emit full spec (even if some parts are missing)
    """
    try:
        # Round marker (1..3)
        try:
            round_n = int(spec_version or 1)
        except Exception:
            round_n = 1
        round_n = max(1, min(3, round_n))

        weaver_spec, weaver_prov = _extract_weaver_spec(constraints_hint)

        user_text = _strip_astra_command(user_intent or "")
        title, objective = _best_effort_title_and_objective(weaver_spec, user_text)

        output_lines, step_lines, verify_lines = _parse_user_clarification(user_text)

        open_questions = _derive_required_fields(weaver_spec, output_lines, step_lines, verify_lines)

        # Ask questions if still missing AND not at cap
        if open_questions and round_n < 3:
            return SpecGateResult(
                ready_for_pipeline=False,
                open_questions=open_questions,
                spec_version=round_n,
            )

        # Finalize spec (even if incomplete — missing parts become Open Issues)
        context: Dict[str, Any] = {}
        if weaver_spec:
            context["weaver_pot"] = weaver_spec
        if weaver_prov:
            context["weaver_provenance"] = weaver_prov
        if user_text:
            context["latest_user_clarification"] = user_text
        context["job_id"] = job_id
        context["provider_id"] = provider_id
        context["model_id"] = model_id

        # Normalize from user lines first
        outputs = _coerce_output_items(output_lines)
        steps = _coerce_step_items(step_lines)
        acceptance = _coerce_acceptance_items(verify_lines)

        # Fill from weaver only if still missing
        if not outputs and isinstance(weaver_spec, dict) and weaver_spec.get("outputs"):
            outputs = _coerce_output_items(weaver_spec.get("outputs"))
        if not steps and isinstance(weaver_spec, dict) and weaver_spec.get("steps"):
            steps = _coerce_step_items(weaver_spec.get("steps"))
        if not acceptance and isinstance(weaver_spec, dict) and (
            weaver_spec.get("acceptance_criteria") or weaver_spec.get("acceptance")
        ):
            acceptance = _coerce_acceptance_items(
                weaver_spec.get("acceptance_criteria") or weaver_spec.get("acceptance")
            )

        open_issues: List[str] = []
        if not outputs:
            open_issues.append("Outputs not fully specified (expected artifacts/files/folders).")
        if not steps:
            open_issues.append("Steps not fully specified (expected short ordered steps).")
        if not acceptance:
            open_issues.append("Acceptance criteria not fully specified (how to verify success).")

        spot_md = _build_spot_markdown(title, objective, outputs, steps, acceptance, open_issues)

        spec_schema = _make_spec_schema(
            title=title,
            objective=objective,
            context=context,
            outputs=outputs,
            steps=steps,
            acceptance=acceptance,
            round_n=round_n,
        )

        # Persist (must succeed to be pipeline-ready)
        db_persisted, spec_id, spec_hash, persist_err = _persist_spec_best_effort(
            db,
            project_id,
            spec_schema,
            job_id=job_id,
            provider_id=provider_id,
            model_id=model_id,
        )

        if db_persisted and spec_id:
            _update_status_best_effort(db, spec_id)

        return SpecGateResult(
            ready_for_pipeline=bool(db_persisted and spec_id),
            open_questions=[],
            spot_markdown=spot_md,
            db_persisted=bool(db_persisted and spec_id),
            spec_id=spec_id,
            spec_hash=spec_hash,
            spec_version=round_n,
            notes=(None if (db_persisted and spec_id) else f"Spec validated but not persisted: {persist_err}"),
        )

    except Exception as e:
        logger.exception("[spec_gate_v2] HARD STOP: %s", e)
        return SpecGateResult(
            ready_for_pipeline=False,
            open_questions=[],
            hard_stopped=True,
            hard_stop_reason=str(e),
            spec_version=spec_version if isinstance(spec_version, int) else None,
        )


__all__ = ["run_spec_gate_v2", "SpecGateResult"]
