# FILE: app/pot_spec/spec_gate_persistence.py
"""
Spec Gate v2 - Persistence Layer

Contains:
- Filesystem artifact writing
- Database persistence
- Spec schema building
- Markdown generation
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    from app.specs import service as specs_service
except Exception:
    specs_service = None

try:
    from app.specs.schema import Spec as SpecSchema
    from app.specs.schema import SpecProvenance, SpecStatus
except Exception:
    SpecSchema = None
    SpecProvenance = None
    SpecStatus = None


# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------

def artifact_root() -> str:
    """Get root directory for job artifacts."""
    root = os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs")
    return os.path.abspath(root)


def job_dir(job_root: str, job_id: str) -> str:
    """Get job directory path."""
    return os.path.join(job_root, "jobs", job_id)


def write_spec_artifacts(
    *,
    job_id: str,
    spec_version: int,
    spec_payload: Dict[str, Any],
    spot_markdown: str,
    spec_hash: str,
) -> Tuple[bool, Optional[str]]:
    """Write spec artifacts to filesystem."""
    try:
        job_root = artifact_root()
        job_path = job_dir(job_root, job_id)
        spec_dir = os.path.join(job_path, "spec")
        os.makedirs(spec_dir, exist_ok=True)

        json_path = os.path.join(spec_dir, f"spec_v{spec_version}.json")
        md_path = os.path.join(spec_dir, f"spec_v{spec_version}.md")
        sha_path = os.path.join(spec_dir, f"spec_v{spec_version}.sha256")

        with open(json_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(spec_payload, f, ensure_ascii=False, indent=2)

        with open(md_path, "w", encoding="utf-8", newline="\n") as f:
            f.write((spot_markdown or "").rstrip() + "\n")

        with open(sha_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(spec_hash + "\n")

        logger.info("[spec_gate_persistence] Wrote artifacts to %s", spec_dir)
        return True, None
    except Exception as e:
        logger.exception("[spec_gate_persistence] Failed to write artifacts: %s", e)
        return False, str(e)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def compute_spec_hash(spec_payload: Dict[str, Any]) -> str:
    """Compute SHA256 hash of spec payload."""
    payload_for_hash = {k: v for k, v in spec_payload.items() if k != "spec_hash"}
    canonical = json.dumps(
        payload_for_hash, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def build_spot_markdown(
    title: str,
    objective: str,
    outputs: List[Dict[str, str]],
    steps: List[str],
    acceptance: List[str],
    open_issues: List[str],
    spec_id: str,
    spec_hash: str,
    spec_version: int,
    blocking_issues: List[str] = None,
) -> str:
    """Build SPoT markdown document."""
    lines: List[str] = []
    lines.append(f"**{title}**")
    lines.append("")
    
    if blocking_issues:
        lines.append("## ⛔ BLOCKING ISSUES")
        for bi in blocking_issues:
            lines.append(f"- {bi}")
        lines.append("")
    
    lines.append("## Objective")
    lines.append(objective.strip() if objective else "(not specified)")
    lines.append("")
    
    lines.append("## Outputs")
    if outputs:
        for o in outputs:
            name = (o.get("name") or "").strip()
            path = (o.get("path") or "").strip()
            notes = (o.get("notes") or "").strip()
            suffix = f" — `{path}`" if path else ""
            extra = f" ({notes})" if notes else ""
            lines.append(f"- {name}{suffix}{extra}".rstrip())
    else:
        lines.append("- ⚠️ (not specified) - BLOCKING")
    lines.append("")
    
    lines.append("## Steps")
    if steps:
        for i, s in enumerate(steps, start=1):
            lines.append(f"S{i}: {s}")
    else:
        lines.append("S1: ⚠️ (not specified) - BLOCKING")
    lines.append("")
    
    lines.append("## Verification / Acceptance Criteria")
    if acceptance:
        for a in acceptance:
            lines.append(f"- {a}")
    else:
        lines.append("- ⚠️ (not specified) - BLOCKING")
    
    if open_issues:
        lines.append("")
        lines.append("## Open Issues")
        for oi in open_issues:
            lines.append(f"- {oi}")
    
    lines.append("")
    lines.append(f"**Spec ID:** `{spec_id}`")
    lines.append(f"**Spec Hash:** `{spec_hash[:16]}...`")
    lines.append(f"**Round:** {spec_version}")
    lines.append("")
    return "\n".join(lines)


def safe_summary_from_objective(objective: str) -> str:
    """Create safe summary from objective."""
    s = (objective or "").strip().replace("\n", " ")
    if len(s) > 180:
        s = s[:177].rstrip() + "..."
    return s or "SPoT spec generated by Spec Gate v2."


# ---------------------------------------------------------------------------
# Database persistence
# ---------------------------------------------------------------------------

def build_spec_schema(
    *,
    spec_id: str,
    title: str,
    summary: str,
    objective: str,
    outputs: List[Dict[str, str]],
    steps: List[str],
    acceptance: List[str],
    context: Dict[str, Any],
    job_id: str,
    provider_id: str,
    model_id: str,
    # v2.2: Grounding data for Critical Pipeline classification
    grounding_data: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Build SpecSchema for database persistence.
    
    v2.2: Added grounding_data parameter to include sandbox fields
    (sandbox_input_path, sandbox_output_path, sandbox_generated_reply, etc.)
    so Critical Pipeline can properly classify micro vs architecture jobs.
    """
    if SpecSchema is None:
        logger.warning("[spec_gate_persistence] SpecSchema not available")
        return None

    try:
        if SpecProvenance is not None:
            provenance = SpecProvenance(
                job_id=job_id,
                generator_model=f"{provider_id}/{model_id}",
                timestamp_start=datetime.now(timezone.utc).isoformat(),
                timestamp_end=datetime.now(timezone.utc).isoformat(),
            )
        else:
            provenance = None

        spec_data = {
            "spec_id": spec_id,
            "title": title,
            "summary": summary,
            "objective": objective,
            "acceptance_criteria": acceptance,
            "outputs": [
                {"name": o.get("name", ""), "type": "file", "example": o.get("path", "")}
                for o in outputs
            ],
            "steps": [
                {"id": f"S{i+1}", "description": s}
                for i, s in enumerate(steps)
            ],
        }
        
        # v2.2: Include grounding data for Critical Pipeline job classification
        # This is CRITICAL for micro vs architecture routing
        if grounding_data:
            spec_data.update({
                # v1.9: Job classification (Critical Pipeline MUST obey these)
                "job_kind": grounding_data.get("job_kind", "unknown"),
                "job_kind_confidence": grounding_data.get("job_kind_confidence", 0.0),
                "job_kind_reason": grounding_data.get("job_kind_reason", ""),
                # Sandbox resolution fields
                "sandbox_input_path": grounding_data.get("sandbox_input_path"),
                "sandbox_output_path": grounding_data.get("sandbox_output_path"),
                "sandbox_generated_reply": grounding_data.get("sandbox_generated_reply"),
                "sandbox_discovery_used": grounding_data.get("sandbox_discovery_used", False),
                "sandbox_input_excerpt": grounding_data.get("sandbox_input_excerpt"),
                "sandbox_selected_type": grounding_data.get("sandbox_selected_type"),
                "sandbox_folder_path": grounding_data.get("sandbox_folder_path"),
                "sandbox_discovery_status": grounding_data.get("sandbox_discovery_status"),
                # Grounding metadata
                "goal": grounding_data.get("goal"),
                "what_exists": grounding_data.get("what_exists", []),
                "what_missing": grounding_data.get("what_missing", []),
                "constraints_from_repo": grounding_data.get("constraints_from_repo", []),
                "constraints_from_intent": grounding_data.get("constraints_from_intent", []),
                "proposed_steps": grounding_data.get("proposed_steps", []),
                "acceptance_tests": grounding_data.get("acceptance_tests", []),
            })
            logger.info(
                "[spec_gate_persistence] v2.2 Including grounding data: job_kind=%s, sandbox_discovery=%s, input=%s, output=%s",
                grounding_data.get("job_kind"),
                grounding_data.get("sandbox_discovery_used"),
                bool(grounding_data.get("sandbox_input_path")),
                bool(grounding_data.get("sandbox_output_path")),
            )

        try:
            spec_schema = SpecSchema.from_dict(spec_data)
        except Exception:
            spec_schema = SpecSchema(
                spec_id=spec_id,
                title=title,
                summary=summary,
                objective=objective,
                acceptance_criteria=acceptance,
            )

        if provenance is not None and hasattr(spec_schema, 'provenance'):
            spec_schema.provenance = provenance
        
        # v2.2: Attach grounding data to schema for JSON serialization
        # This ensures content_json includes all grounding fields
        if grounding_data and hasattr(spec_schema, '__dict__'):
            for key, value in grounding_data.items():
                if value is not None and not hasattr(spec_schema, key):
                    setattr(spec_schema, key, value)

        return spec_schema

    except Exception as e:
        logger.exception("[spec_gate_persistence] Failed to build SpecSchema: %s", e)
        return None


def persist_spec(
    db: Session,
    project_id: int,
    spec_schema: Any,
    *,
    provider_id: str,
    model_id: str,
) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """
    Persist spec to database AND update status to validated.
    
    v1.6 (2026-01-21): CRITICAL FIX - Now calls update_spec_status() to set
    status to 'validated' after creation. Previously, specs were created with
    DRAFT status and never updated, causing Critical Pipeline to fail with
    "No validated specification found".
    """
    if specs_service is None or not hasattr(specs_service, "create_spec"):
        return False, None, None, "specs_service.create_spec not available"

    if spec_schema is None:
        return False, None, None, "spec_schema is None"

    try:
        generator_model = f"{provider_id}/{model_id}"
        
        logger.info(
            "[spec_gate_persistence] persist_spec START project_id=%s, provider=%s, model=%s",
            project_id, provider_id, model_id
        )
        
        # Step 1: Create spec (initially with DRAFT status)
        db_spec = specs_service.create_spec(
            db=db,
            project_id=project_id,
            spec_schema=spec_schema,
            generator_model=generator_model,
        )

        spec_id = getattr(db_spec, "spec_id", None)
        spec_hash = getattr(db_spec, "spec_hash", None)

        if not spec_id:
            logger.error("[spec_gate_persistence] create_spec returned without spec_id")
            return False, None, None, "create_spec returned without spec_id"
        
        logger.info(
            "[spec_gate_persistence] Spec created with DRAFT status: spec_id=%s, hash=%s",
            spec_id, spec_hash[:16] if spec_hash else "None"
        )
        
        # Step 2: CRITICAL - Update status to VALIDATED
        # This was missing before, causing Critical Pipeline to fail
        status_updated, status_error = update_spec_status(
            db=db,
            spec_id=spec_id,
            provider_id=provider_id,
            model_id=model_id,
        )
        
        if not status_updated:
            logger.error(
                "[spec_gate_persistence] Failed to update spec status to validated: %s",
                status_error
            )
            # Spec exists but not validated - return partial success
            return False, str(spec_id), str(spec_hash) if spec_hash else None, f"status update failed: {status_error}"
        
        logger.info(
            "[spec_gate_persistence] Spec status updated to VALIDATED: spec_id=%s",
            spec_id
        )
        
        # Step 3: Post-persist readback verification
        if hasattr(specs_service, "get_latest_validated_spec"):
            readback = specs_service.get_latest_validated_spec(db, project_id)
            if readback and getattr(readback, "spec_id", None) == spec_id:
                logger.info(
                    "[spec_gate_persistence] POST-PERSIST VERIFIED: spec_id=%s found in validated specs",
                    spec_id
                )
            else:
                logger.warning(
                    "[spec_gate_persistence] POST-PERSIST WARNING: spec_id=%s NOT found in validated specs readback",
                    spec_id
                )
                # Log what was found for debugging
                if readback:
                    logger.warning(
                        "[spec_gate_persistence]   Readback found different spec: %s",
                        getattr(readback, "spec_id", "unknown")
                    )
                else:
                    logger.warning("[spec_gate_persistence]   Readback returned None")
        
        logger.info(
            "[spec_gate_persistence] persist_spec COMPLETE: spec_id=%s, status=validated",
            spec_id
        )
        return True, str(spec_id), str(spec_hash) if spec_hash else None, None

    except Exception as e:
        logger.exception("[spec_gate_persistence] DB persistence failed: %s", e)
        return False, None, None, str(e)


def update_spec_status(
    db: Session,
    spec_id: str,
    *,
    provider_id: str,
    model_id: str,
) -> Tuple[bool, Optional[str]]:
    """Update spec status to validated."""
    if specs_service is None or not hasattr(specs_service, "update_spec_status"):
        return False, "specs_service.update_spec_status not available"

    try:
        validated_status = "validated"
        if SpecStatus is not None:
            try:
                validated_status = SpecStatus.VALIDATED.value
            except Exception:
                pass

        triggered_by = f"spec_gate_v2:{provider_id}/{model_id}"

        specs_service.update_spec_status(
            db=db,
            spec_id=spec_id,
            new_status=validated_status,
            validation_result={"source": "spec_gate_v2", "finalized": True},
            triggered_by=triggered_by,
        )
        logger.info("[spec_gate_persistence] Updated spec status: %s", spec_id)
        return True, None

    except Exception as e:
        logger.warning("[spec_gate_persistence] Failed to update status: %s", e)
        return False, str(e)


__all__ = [
    "artifact_root",
    "job_dir",
    "write_spec_artifacts",
    "compute_spec_hash",
    "build_spot_markdown",
    "safe_summary_from_objective",
    "build_spec_schema",
    "persist_spec",
    "update_spec_status",
]