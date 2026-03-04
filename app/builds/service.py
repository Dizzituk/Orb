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
]

# Map stage enum → model attribute name
_STAGE_STATUS_ATTR = {
    PipelineStage.weaver: "weaver_status",
    PipelineStage.spec_gate: "spec_gate_status",
    PipelineStage.critical_pipeline: "critical_pipeline_status",
    PipelineStage.overwatcher: "overwatcher_status",
    PipelineStage.implementer: "implementer_status",
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
    # If all stages passed, mark complete
    elif stage_status == StageStatusEnum.passed and stage == PipelineStageEnum.implementer:
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
    """Compile the full build report from brief + narratives + deliverables.

    Generates a single markdown document that can be read by a human
    or sent to the Debug workspace for situational awareness.

    Returns the report markdown, also saves it to the project.
    """
    project = get_project(db, project_id)
    if not project:
        return None

    parts = []

    # Header
    parts.append(f"# Build Report: {project.name}")
    parts.append(f"**Project ID:** {project.id}")
    parts.append(f"**Status:** {project.status.value}")
    parts.append(f"**Created:** {project.created_at.isoformat()}")
    if project.job_id:
        parts.append(f"**Job ID:** {project.job_id}")
    if project.spec_id:
        parts.append(f"**Spec ID:** {project.spec_id}")
    parts.append("")

    # Original Brief
    parts.append("## Original Brief")
    parts.append("")
    if project.original_brief:
        parts.append(project.original_brief)
    else:
        parts.append("*No brief recorded.*")
    parts.append("")

    # Stage-by-stage narrative
    stage_order = ["weaver", "spec_gate", "critical_pipeline", "overwatcher", "implementer"]
    stage_labels = {
        "weaver": "Weaver",
        "spec_gate": "SpecGate",
        "critical_pipeline": "Critical Pipeline",
        "overwatcher": "Overwatcher",
        "implementer": "Implementer",
    }
    narratives = project.stage_narratives or {}

    for stage_key in stage_order:
        entries = narratives.get(stage_key, [])
        if not entries:
            continue

        label = stage_labels.get(stage_key, stage_key)
        parts.append(f"## {label}")
        parts.append("")

        for entry in entries:
            title = entry.get("title", "")
            ts = entry.get("timestamp", "")
            duration = entry.get("duration_ms")
            model = entry.get("model_used")
            input_sum = entry.get("input_summary")
            output_sum = entry.get("output_summary")
            sections = entry.get("sections", [])
            files = entry.get("files_touched", [])
            warnings = entry.get("warnings", [])

            if title:
                parts.append(f"### {title}")
            if ts:
                timing = f"*{ts}*"
                if duration:
                    timing += f" ({duration}ms)"
                if model:
                    timing += f" — {model}"
                parts.append(timing)
            parts.append("")

            if input_sum:
                parts.append(f"**Input:** {input_sum}")
                parts.append("")
            if output_sum:
                parts.append(f"**Output:** {output_sum}")
                parts.append("")

            for section in sections:
                heading = section.get("heading", "")
                body = section.get("body", "")
                if heading:
                    parts.append(f"**{heading}**")
                if body:
                    parts.append(body)
                parts.append("")

            if files:
                parts.append(f"**Files:** {', '.join(files)}")
                parts.append("")
            if warnings:
                for w in warnings:
                    parts.append(f"⚠️ {w}")
                parts.append("")

    # Stage Log (timeline)
    log = project.stage_log or []
    if log:
        parts.append("## Stage Log Timeline")
        parts.append("")
        for entry in log:
            ts = entry.get("timestamp", "")[:19]
            stage = entry.get("stage", "")
            event = entry.get("event", "")
            detail = entry.get("detail", "")
            line = f"- **{ts}** {stage} → {event}"
            if detail:
                line += f": {detail}"
            parts.append(line)
        parts.append("")

    # Deliverables
    if project.target_path:
        parts.append("## Deliverables")
        parts.append("")
        parts.append(f"**Target path:** {project.target_path}")
        parts.append("")

    report = "\n".join(parts)

    # Save to project
    project.build_report = report
    db.commit()
    db.refresh(project)

    logger.info(
        "[builds] Compiled build report for %s (%d chars)",
        project_id, len(report),
    )
    return report


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
