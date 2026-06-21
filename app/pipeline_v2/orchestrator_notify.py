# FILE: app/pipeline_v2/orchestrator_notify.py
# Purpose: Orchestrator stage side-effects (narrative push + stage status) — split from orchestrator.py.
# Called-by: app.pipeline_v2.orchestrator, app.pipeline_v2.orchestrator_spec_review
# Depends-on: app.builds.models, app.builds.pipeline_bridge, app.db
# Last-renovated: 2026-06-21
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def _push_narrative(
    stage: str,
    title: str,
    input_summary: str = "",
    output_summary: str = "",
    sections: list = None,
    files_touched: list = None,
    warnings: list = None,
    model_used: str = "",
    duration_ms: int = 0,
    tokens_used: int = 0,
) -> None:
    """Push a narrative entry to the build project's stage log."""
    try:
        from app.db import SessionLocal
        from app.builds.pipeline_bridge import notify_narrative
        from app.builds.models import BuildProject
        db = SessionLocal()
        project = db.query(BuildProject).order_by(BuildProject.updated_at.desc()).first()
        if project:
            notify_narrative(
                db, project.id, stage,
                title=title,
                input_summary=input_summary,
                output_summary=output_summary,
                sections=sections or [],
                files_touched=files_touched or [],
                warnings=warnings or [],
                model_used=model_used,
                duration_ms=duration_ms,
                tokens_used=tokens_used,
            )
        db.close()
    except Exception as e:
        logger.debug("[orchestrator] Narrative push failed: %s", e)


def _update_stage_status(stage: str, status: str, detail: str = "") -> None:
    """Update a pipeline stage status on the current build project."""
    try:
        from app.db import SessionLocal
        from app.builds.pipeline_bridge import (
            notify_stage_start, notify_stage_passed, notify_stage_failed,
        )
        from app.builds.models import BuildProject
        db = SessionLocal()
        project = db.query(BuildProject).order_by(BuildProject.updated_at.desc()).first()
        if project:
            pid = str(project.id)
            if status == "running":
                notify_stage_start(db, pid, stage, detail=detail)
            elif status == "passed":
                notify_stage_passed(db, pid, stage, detail=detail)
            elif status == "failed":
                notify_stage_failed(db, pid, stage, detail=detail)
        db.close()
    except Exception as e:
        logger.debug("[orchestrator] Stage status update failed: %s", e)
