# FILE: app/builds/pipeline_bridge.py
"""
Pipeline Bridge — glue between command dispatch and Build Projects.

When a pipeline intent fires (WEAVER_BUILD_SPEC, SEND_TO_SPEC_GATE, etc.),
this module creates or finds the active Build Project and returns its ID
so stream handlers can report stage progress.

Design:
- One active build project per chat project at a time
- Auto-creates if none exists when a pipeline command fires
- Extracts a name from the brief text
- Stores the original_brief
- Returns build_project_id for handlers to use
"""

import logging
import re
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from app.builds.models import BuildProject, BuildStatus, PipelineStage, StageStatus
from app.builds import service as build_service

logger = logging.getLogger(__name__)

# Max length for auto-extracted project names
_MAX_NAME_LEN = 100


def get_or_create_build_project(
    db: Session,
    chat_project_id: int,
    brief: Optional[str] = None,
) -> BuildProject:
    """
    Find the active build project for this chat project,
    or create one if none exists.

    Args:
        db: Database session
        chat_project_id: The chat project ID that triggered the pipeline
        brief: The user's ramble/brief text (used for naming + storage)

    Returns:
        The active BuildProject instance
    """
    # Look for an existing active project linked to this chat project
    existing = (
        db.query(BuildProject)
        .filter(
            BuildProject.chat_project_id == chat_project_id,
            BuildProject.status == BuildStatus.active,
        )
        .order_by(BuildProject.updated_at.desc())
        .first()
    )

    if existing:
        # Update brief if we have a new one and the existing one is empty
        if brief and not existing.original_brief:
            existing.original_brief = brief
            db.commit()
            db.refresh(existing)
        logger.info(
            "[pipeline_bridge] Found existing build project '%s' (%s) for chat_project=%s",
            existing.name, existing.id, chat_project_id,
        )
        return existing

    # Create a new one
    name = _extract_project_name(brief) if brief else f"Build #{chat_project_id}"

    project = build_service.create_project(
        db,
        name=name,
        original_brief=brief,
        chat_project_id=chat_project_id,
    )
    logger.info(
        "[pipeline_bridge] Created build project '%s' (%s) for chat_project=%s",
        project.name, project.id, chat_project_id,
    )
    return project


def notify_stage_start(
    db: Session,
    build_project_id: str,
    stage: str,
    detail: Optional[str] = None,
) -> Optional[BuildProject]:
    """Notify that a pipeline stage has started running."""
    from app.builds.schemas import PipelineStageEnum, StageStatusEnum
    try:
        return build_service.advance_stage(
            db, build_project_id,
            stage=PipelineStageEnum(stage),
            stage_status=StageStatusEnum.running,
            detail=detail,
        )
    except Exception as e:
        logger.warning("[pipeline_bridge] Failed to notify stage start: %s", e)
        return None


def notify_stage_passed(
    db: Session,
    build_project_id: str,
    stage: str,
    detail: Optional[str] = None,
) -> Optional[BuildProject]:
    """Notify that a pipeline stage has passed."""
    from app.builds.schemas import PipelineStageEnum, StageStatusEnum
    try:
        return build_service.advance_stage(
            db, build_project_id,
            stage=PipelineStageEnum(stage),
            stage_status=StageStatusEnum.passed,
            detail=detail,
        )
    except Exception as e:
        logger.warning("[pipeline_bridge] Failed to notify stage passed: %s", e)
        return None


def notify_stage_failed(
    db: Session,
    build_project_id: str,
    stage: str,
    detail: Optional[str] = None,
) -> Optional[BuildProject]:
    """Notify that a pipeline stage has failed."""
    from app.builds.schemas import PipelineStageEnum, StageStatusEnum
    try:
        return build_service.advance_stage(
            db, build_project_id,
            stage=PipelineStageEnum(stage),
            stage_status=StageStatusEnum.failed,
            detail=detail or "Stage failed",
        )
    except Exception as e:
        logger.warning("[pipeline_bridge] Failed to notify stage failed: %s", e)
        return None


def notify_stage_awaiting(
    db: Session,
    build_project_id: str,
    stage: str,
    detail: Optional[str] = None,
) -> Optional[BuildProject]:
    """Notify that a pipeline stage is awaiting user input."""
    from app.builds.schemas import PipelineStageEnum, StageStatusEnum
    try:
        return build_service.advance_stage(
            db, build_project_id,
            stage=PipelineStageEnum(stage),
            stage_status=StageStatusEnum.awaiting_input,
            detail=detail,
        )
    except Exception as e:
        logger.warning("[pipeline_bridge] Failed to notify stage awaiting: %s", e)
        return None


def _extract_project_name(brief: str) -> str:
    """
    Extract a short project name from the brief text.

    Strategy:
    1. Look for "Build a/an/the X" pattern
    2. Look for "Create a/an/the X" pattern
    3. Fall back to first meaningful sentence fragment
    """
    if not brief:
        return "Untitled Project"

    text = brief.strip()

    # Try "Build/Create/Make a(n) X" patterns
    patterns = [
        # "I want to build out the X" / "I want to create a Y"
        r"(?:I want|I need|we need)\s+to\s+(?:build|create|make|design|implement)\s+(?:out\s+)?(?:a|an|the)\s+(.{5,60}?)(?:\.|,|\n|$)",
        # "Build a/an/the X"
        r"(?:build|create|make|design|implement)\s+(?:out\s+)?(?:a|an|the)\s+(.{5,60}?)(?:\.|,|\n|$)",
        # "Build X" (no article)
        r"(?:build|create|make|design|implement)\s+(?:out\s+)?(.{5,60}?)(?:\.|,|\n|$)",
        # "I want a/an/the X"
        r"(?:I want|I need|we need)\s+(?:a|an|the)\s+(.{5,60}?)(?:\.|,|\n|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Clean up trailing words like "that", "which", "with"
            name = re.sub(r"\s+(that|which|with|for|using|in|on)\s*$", "", name, flags=re.IGNORECASE)
            if len(name) > _MAX_NAME_LEN:
                name = name[:_MAX_NAME_LEN].rsplit(" ", 1)[0]
            return name.strip()

    # Fallback: first line or first 80 chars
    first_line = text.split("\n")[0].strip()
    if len(first_line) > _MAX_NAME_LEN:
        first_line = first_line[:_MAX_NAME_LEN].rsplit(" ", 1)[0]

    return first_line or "Untitled Project"
