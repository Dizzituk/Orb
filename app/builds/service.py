# FILE: app/builds/service.py
"""
Build project service — CRUD + pipeline stage management.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy.orm import Session

from app.builds.models import (
    BuildProject, BuildStatus, PipelineStage, StageStatus,
)
from app.builds.schemas import (
    BuildProjectResponse, BuildProjectSummary, StageInfo,
    StageLogEntry, PipelineStageEnum, StageStatusEnum,
)

logger = logging.getLogger(__name__)

# Ordered list of pipeline stages for iteration
PIPELINE_STAGES = [
    PipelineStage.weaver,
    PipelineStage.spec_gate,
    PipelineStage.critical_pipeline,
    PipelineStage.overwatcher,
    PipelineStage.implementer,
    PipelineStage.final_checkout,
]

# Map stage enum → model attribute name
_STAGE_STATUS_ATTR = {
    PipelineStage.weaver: "weaver_status",
    PipelineStage.spec_gate: "spec_gate_status",
    PipelineStage.critical_pipeline: "critical_pipeline_status",
    PipelineStage.overwatcher: "overwatcher_status",
    PipelineStage.implementer: "implementer_status",
    PipelineStage.final_checkout: "final_checkout_status",
}


# ── CRUD ──

def list_projects(db: Session) -> List[BuildProject]:
    return db.query(BuildProject).order_by(BuildProject.updated_at.desc()).all()


def get_project(db: Session, project_id: str) -> Optional[BuildProject]:
    return db.query(BuildProject).filter(BuildProject.id == project_id).first()


def create_project(
    db: Session,
    name: str,
    description: Optional[str] = None,
    original_brief: Optional[str] = None,
    chat_project_id: Optional[int] = None,
) -> BuildProject:
    project = BuildProject(
        name=name,
        description=description,
        original_brief=original_brief,
        chat_project_id=chat_project_id,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    logger.info(f"[builds] Created project '{name}' ({project.id})")
    return project


def update_project(
    db: Session,
    project_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
) -> Optional[BuildProject]:
    project = get_project(db, project_id)
    if not project:
        return None
    if name is not None:
        project.name = name
    if description is not None:
        project.description = description
    if status is not None:
        project.status = status
    db.commit()
    db.refresh(project)
    return project


def delete_project(db: Session, project_id: str) -> bool:
    project = get_project(db, project_id)
    if not project:
        return False
    db.delete(project)
    db.commit()
    logger.info(f"[builds] Deleted project {project_id}")
    return True


# ── Pipeline State ──

def advance_stage(
    db: Session,
    project_id: str,
    stage: PipelineStageEnum,
    stage_status: StageStatusEnum,
    detail: Optional[str] = None,
) -> Optional[BuildProject]:
    """Update a specific pipeline stage's status and optionally set current_stage."""
    project = get_project(db, project_id)
    if not project:
        return None

    attr = _STAGE_STATUS_ATTR.get(PipelineStage(stage.value))
    if attr:
        setattr(project, attr, StageStatus(stage_status.value))

    # If a stage is now running, set it as current
    if stage_status == StageStatusEnum.running:
        project.current_stage = PipelineStage(stage.value)
    # If final checkout passed, mark complete
    elif stage_status == StageStatusEnum.passed and stage == PipelineStageEnum.final_checkout:
        project.current_stage = PipelineStage.complete
        project.status = BuildStatus.completed

    # Append to stage log
    log = list(project.stage_log or [])
    log.append({
        "stage": stage.value,
        "event": stage_status.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detail": detail,
    })
    project.stage_log = log

    db.commit()
    db.refresh(project)
    logger.info(f"[builds] Project {project_id}: {stage.value} → {stage_status.value}")
    return project


def update_segments(
    db: Session,
    project_id: str,
    total: Optional[int] = None,
    completed: Optional[int] = None,
) -> Optional[BuildProject]:
    project = get_project(db, project_id)
    if not project:
        return None
    if total is not None:
        project.total_segments = total
    if completed is not None:
        project.completed_segments = completed
    db.commit()
    db.refresh(project)
    return project


def link_spec(db: Session, project_id: str, spec_id: str) -> Optional[BuildProject]:
    project = get_project(db, project_id)
    if not project:
        return None
    project.spec_id = spec_id
    db.commit()
    db.refresh(project)
    return project


def link_job(db: Session, project_id: str, job_id: str) -> Optional[BuildProject]:
    project = get_project(db, project_id)
    if not project:
        return None
    project.job_id = job_id
    db.commit()
    db.refresh(project)
    return project


# ── Narrative ──

def append_narrative(
    db: Session,
    project_id: str,
    stage: str,
    narrative_dict: dict,
) -> Optional[BuildProject]:
    """Append a rich narrative entry to a pipeline stage.

    Each stage can accumulate multiple narrative entries as it executes.
    For example, Critical Pipeline adds one per segment, Implementer
    adds one per file written.

    Args:
        db: Database session
        project_id: Build project ID
        stage: Pipeline stage name (e.g. "weaver", "critical_pipeline")
        narrative_dict: Dict matching StageNarrative schema

    Returns:
        Updated BuildProject or None if not found.
    """
    project = get_project(db, project_id)
    if not project:
        return None

    narratives = dict(project.stage_narratives or {})
    if stage not in narratives:
        narratives[stage] = []
    narratives[stage].append(narrative_dict)
    project.stage_narratives = narratives

    db.commit()
    db.refresh(project)
    logger.info(
        "[builds] Appended narrative to %s.%s (%d entries)",
        project_id, stage, len(narratives[stage]),
    )
    return project


def compile_build_report(
    db: Session,
    project_id: str,
) -> Optional[str]:
    """Compile the full build report as paged JSON.

    Produces a structured report where each pipeline stage is its own
    page, designed for targeted reading by the Debug system or a new
    context window. The io_summary at the top proves sandbox compliance.

    Returns the report as a JSON string, also saves it to the project.
    """
    import json as _json

    project = get_project(db, project_id)
    if not project:
        return None

    narratives = project.stage_narratives or {}
    io_summary = _build_io_summary(project_id)
    pages = []

    # Page 1: Project Scope
    pages.append(_build_scope_page(project))

    # Page 2: Weaver Intent
    pages.append(_build_weaver_page(project, narratives))

    # Page 3: SpecGate Research
    pages.append(_build_specgate_page(narratives))

    # Page 4: Critical Pipeline — Architecture
    pages.append(_build_architecture_page(narratives))

    # Page 5: Implementer — Execution
    pages.append(_build_execution_page(project, narratives))

    # Page 6: Final Checkout — Validation
    pages.append(_build_checkout_page(project, narratives))

    report_obj = {
        "project_id": project.id,
        "project_name": project.name,
        "status": project.status.value,
        "io_summary": io_summary,
        "pages": pages,
    }

    report_json = _json.dumps(report_obj, indent=2, default=str)

    project.build_report = report_json
    db.commit()
    db.refresh(project)

    logger.info(
        "[builds] Compiled paged report for %s (%d chars, %d pages)",
        project_id, len(report_json), len(pages),
    )
    return report_json


# ── Report Page Builders ──

def _build_scope_page(project: BuildProject) -> dict:
    """Page 1: Project Scope — brief + Weaver extraction."""
    extraction = project.weaver_extraction or {}
    return {
        "page": 1,
        "title": "Project Scope",
        "content": {
            "original_brief": project.original_brief or "No brief recorded.",
            "key_requirements": extraction.get("key_requirements", []),
            "design_preferences": extraction.get("design_preferences", []),
            "constraints": extraction.get("constraints", []),
        },
    }


def _build_weaver_page(project: BuildProject, narratives: dict) -> dict:
    """Page 2: Weaver Intent — how the AI interpreted the brief."""
    weaver_entries = narratives.get("weaver", [])
    return {
        "page": 2,
        "title": "Weaver Intent",
        "content": {
            "interpretation": _flatten_narrative_sections(weaver_entries),
            "extraction_summary": project.weaver_extraction or {},
        },
    }


def _build_specgate_page(narratives: dict) -> dict:
    """Page 3: SpecGate Research — codebase analysis summary."""
    sg_entries = narratives.get("spec_gate", [])
    return {
        "page": 3,
        "title": "Research Findings (SpecGate)",
        "content": {
            "analysis": _flatten_narrative_sections(sg_entries),
            "files_examined": _collect_files_touched(sg_entries),
        },
    }


def _build_architecture_page(narratives: dict) -> dict:
    """Page 4: Critical Pipeline — architecture and segmentation."""
    cp_entries = narratives.get("critical_pipeline", [])
    return {
        "page": 4,
        "title": "Architecture & Segmentation",
        "content": {
            "plan": _flatten_narrative_sections(cp_entries),
            "files_planned": _collect_files_touched(cp_entries),
        },
    }


def _build_execution_page(project: BuildProject, narratives: dict) -> dict:
    """Page 5: Implementer — execution summary."""
    impl_entries = narratives.get("implementer", [])
    ow_entries = narratives.get("overwatcher", [])
    return {
        "page": 5,
        "title": "Execution & Implementation",
        "content": {
            "implementer": _flatten_narrative_sections(impl_entries),
            "overwatcher": _flatten_narrative_sections(ow_entries),
            "files_written": _collect_files_touched(impl_entries),
            "segments": {
                "total": project.total_segments,
                "completed": project.completed_segments,
            },
        },
    }


def _build_checkout_page(project: BuildProject, narratives: dict) -> dict:
    """Page 6: Final Checkout — validation results."""
    fc_entries = narratives.get("final_checkout", [])
    # Check stage log for checkout events
    log = project.stage_log or []
    checkout_events = [e for e in log if e.get("stage") == "final_checkout"]

    return {
        "page": 6,
        "title": "Final Checkout",
        "content": {
            "validation": _flatten_narrative_sections(fc_entries),
            "stage_events": checkout_events,
            "files_fixed": _collect_files_touched(fc_entries),
        },
    }


def _build_io_summary(project_id: str) -> dict:
    """Build IO summary from transparency events for this project."""
    summary = {
        "total_reads": 0,
        "total_writes": 0,
        "sandbox_reads": 0,
        "sandbox_writes": 0,
        "host_reads_operational": 0,
        "host_writes_operational": 0,
        "host_reads_violation": 0,
        "host_writes_violation": 0,
    }
    try:
        from app.db import get_db_session
        from app.transparency.models import ReasoningEventModel

        db = get_db_session()
        try:
            rows = (
                db.query(ReasoningEventModel)
                .filter_by(build_project_id=project_id)
                .all()
            )
            for row in rows:
                # io_events are stored as part of evidence_sources JSON
                # or in a dedicated io_events field if the model supports it
                io_events = []
                if hasattr(row, "evidence_sources") and row.evidence_sources:
                    for ev in row.evidence_sources:
                        if isinstance(ev, dict) and ev.get("source_type") == "file_read":
                            summary["total_reads"] += 1
                            summary["sandbox_reads"] += 1
            return summary
        finally:
            db.close()
    except Exception as e:
        logger.debug("[builds] Failed to build IO summary: %s", e)
        return summary


def _flatten_narrative_sections(entries: list) -> list:
    """Flatten narrative entries into a list of section dicts."""
    result = []
    for entry in entries:
        item = {
            "title": entry.get("title", ""),
            "timestamp": entry.get("timestamp", ""),
            "duration_ms": entry.get("duration_ms"),
            "model_used": entry.get("model_used"),
            "input_summary": entry.get("input_summary"),
            "output_summary": entry.get("output_summary"),
            "sections": entry.get("sections", []),
            "warnings": entry.get("warnings", []),
        }
        result.append(item)
    return result


def _collect_files_touched(entries: list) -> list:
    """Collect deduplicated list of files touched across narrative entries."""
    seen = set()
    files = []
    for entry in entries:
        for f in entry.get("files_touched", []):
            if f not in seen:
                seen.add(f)
                files.append(f)
    return files


# ── Response Builders ──

def _get_stages(project: BuildProject) -> List[StageInfo]:
    """Build ordered list of stage info for the response."""
    return [
        StageInfo(stage=PipelineStageEnum(s.value), status=StageStatusEnum(getattr(project, attr).value))
        for s, attr in _STAGE_STATUS_ATTR.items()
    ]


def _parse_stage_log(project: BuildProject) -> List[StageLogEntry]:
    raw = project.stage_log or []
    return [
        StageLogEntry(
            stage=e.get("stage", ""),
            event=e.get("event", ""),
            timestamp=e.get("timestamp", ""),
            detail=e.get("detail"),
        )
        for e in raw
    ]


def to_response(project: BuildProject) -> BuildProjectResponse:
    return BuildProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        status=project.status,
        current_stage=project.current_stage,
        stages=_get_stages(project),
        spec_id=project.spec_id,
        job_id=project.job_id,
        target_path=project.target_path,
        original_brief=project.original_brief,
        weaver_extraction=project.weaver_extraction,
        chat_project_id=project.chat_project_id,
        total_segments=project.total_segments,
        completed_segments=project.completed_segments,
        stage_log=_parse_stage_log(project),
        stage_narratives=project.stage_narratives or {},
        build_report=project.build_report,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


def to_summary(project: BuildProject) -> BuildProjectSummary:
    return BuildProjectSummary(
        id=project.id,
        name=project.name,
        status=project.status,
        current_stage=project.current_stage,
        total_segments=project.total_segments,
        completed_segments=project.completed_segments,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


# ── Startup Recovery ──

def recover_stale_running_stages(db: Session) -> int:
    """v3.2-fix: Reset any stages stuck in 'running' to 'failed' on startup.

    When the app is closed while a pipeline stage is actively running,
    that stage remains in 'running' state in the database. On restart,
    no process is driving it, so the UI shows a permanently stuck spinner
    or stale 'running' badge. This function resets all such stages to
    'failed' so the user can re-trigger them.

    Should be called once during application startup.

    Returns:
        Number of stages that were reset.
    """
    projects = db.query(BuildProject).all()
    reset_count = 0

    for project in projects:
        for stage, attr in _STAGE_STATUS_ATTR.items():
            if getattr(project, attr) == StageStatus.running:
                setattr(project, attr, StageStatus.failed)
                logger.info(
                    "[builds] Startup recovery: %s.%s running → failed (project '%s')",
                    project.id, stage.value, project.name,
                )
                # Append to stage log
                log = list(project.stage_log or [])
                log.append({
                    "stage": stage.value,
                    "event": "failed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "detail": "Reset on startup — was running when app closed",
                })
                project.stage_log = log
                reset_count += 1

        # If the project itself was 'in_progress' and has stale stages, mark it
        if project.status == BuildStatus.in_progress:
            # Check if any stage is still running (shouldn't be after above loop)
            any_running = any(
                getattr(project, attr) == StageStatus.running
                for attr in _STAGE_STATUS_ATTR.values()
            )
            if not any_running:
                # Don't change to failed — user may want to re-run
                pass

    if reset_count > 0:
        db.commit()
        logger.info("[builds] Startup recovery: reset %d stale running stage(s)", reset_count)
    else:
        logger.debug("[builds] Startup recovery: no stale stages found")

    return reset_count
