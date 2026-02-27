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
