# FILE: app/llm/_sg_stream_persist.py
"""
Spec Gate stream — persistence helpers.

Extracted from spec_gate_stream.py to reduce file size.
Handles DB persistence of validated specs after SpecGate processing.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
_persist_spec = None
_build_spec_schema = None
_safe_summary_from_objective = None
_specs_service = None
_PERSISTENCE_AVAILABLE = False

try:
    from app.pot_spec.spec_gate_persistence import (
        persist_spec as _ps,
        build_spec_schema as _bss,
        safe_summary_from_objective as _sso,
    )
    _persist_spec = _ps
    _build_spec_schema = _bss
    _safe_summary_from_objective = _sso
    _PERSISTENCE_AVAILABLE = True
except Exception:
    pass

try:
    from app.specs import service as _svc
    _specs_service = _svc
except Exception:
    pass


def persist_validated_spec(
    *,
    db: Session,
    project_id: int,
    result: Any,
    clarification_round: int,
    spec_gate_provider: str,
    spec_gate_model: str,
    job_id: str,
    title_suffix: str = "",
) -> Tuple[bool, Optional[str]]:
    """Persist a validated spec to DB after SpecGate processing.

    Returns:
        (db_persisted: bool, persist_error: Optional[str])
    """
    if not _PERSISTENCE_AVAILABLE or not _persist_spec or not _build_spec_schema:
        return False, "Persistence module not available"

    try:
        spot_md = getattr(result, "spot_markdown", "") or ""
        spec_id = getattr(result, "spec_id", "") or ""
        grounding_data = getattr(result, "grounding_data", None)

        goal = ""
        if grounding_data and grounding_data.get("goal"):
            goal = grounding_data["goal"]
        elif "## Goal" in spot_md:
            goal_start = spot_md.index("## Goal") + len("## Goal")
            goal_end = spot_md.find("##", goal_start)
            if goal_end == -1:
                goal_end = len(spot_md)
            goal = spot_md[goal_start:goal_end].strip()

        summary = (
            _safe_summary_from_objective(goal)
            if _safe_summary_from_objective
            else goal[:180]
        )

        title = f"SPoT Spec (SpecGate v3.0{title_suffix})"
        spec_schema = _build_spec_schema(
            spec_id=spec_id,
            title=title,
            summary=summary,
            objective=goal,
            outputs=[],
            steps=[],
            acceptance=[],
            context={
                "source": "spec_gate_grounded",
                "round": clarification_round,
                **({"segmented": True} if "Segmented" in title_suffix else {}),
            },
            job_id=job_id,
            provider_id=spec_gate_provider,
            model_id=spec_gate_model,
            grounding_data=grounding_data,
        )

        if not spec_schema:
            return False, "build_spec_schema returned None"

        success, db_spec_id, db_spec_hash, error = _persist_spec(
            db=db,
            project_id=project_id,
            spec_schema=spec_schema,
            provider_id=spec_gate_provider,
            model_id=spec_gate_model,
        )

        if success:
            logger.info("[sg_persist] Spec persisted to DB: %s", db_spec_id)
            # v2.4: Overwrite content_markdown with actual POT spec markdown
            if spot_md and len(spot_md) > 100 and _specs_service:
                try:
                    _specs_service.update_spec_content_markdown(db, db_spec_id, spot_md)
                    logger.info(
                        "[sg_persist] v2.4: Updated content_markdown (%d chars)",
                        len(spot_md),
                    )
                except Exception as e:
                    logger.warning("[sg_persist] v2.4: content_markdown update failed: %s", e)
            return True, None
        else:
            return False, error

    except Exception as e:
        logger.exception("[sg_persist] Persistence exception: %s", e)
        return False, str(e)


__all__ = ["persist_validated_spec"]
