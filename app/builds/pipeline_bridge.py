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

v2.2 (2026-03-10): Added resolve_build_target() — auto-detects which project
    the user is talking about from conversation context and sets
    build_target_id on the build project.
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
    """
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
        # v2.3: Check if the ProjectSession target has changed.
        # If so, the old build project is stale — archive it and create fresh.
        try:
            from app.shared_context.project_session import get_project_session
            _session = get_project_session(str(chat_project_id))
            if (_session.is_set and _session.project_id
                    and existing.build_target_id
                    and existing.build_target_id != _session.project_id):
                logger.info(
                    "[pipeline_bridge] Stale build project '%s' target=%s, session=%s — archiving",
                    existing.id, existing.build_target_id, _session.project_id,
                )
                existing.status = BuildStatus.completed
                db.commit()
                existing = None  # Force creation of new build project below
        except Exception as e:
            logger.debug("[pipeline_bridge] Session stale check failed: %s", e)

    if existing:
        if brief and not existing.original_brief:
            existing.original_brief = brief
            db.commit()
            db.refresh(existing)
        logger.info(
            "[pipeline_bridge] Found existing build project '%s' (%s) for chat_project=%s",
            existing.name, existing.id, chat_project_id,
        )
        return existing

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


# ═══════════════════════════════════════════════════════════════════
# v2.2: Build target resolution
# ═══════════════════════════════════════════════════════════════════

def resolve_build_target(
    db: Session,
    build_project: BuildProject,
    message: Optional[str] = None,
    chat_project_id: Optional[int] = None,
) -> Optional[str]:
    """Resolve the build target from conversation context and persist it.

    Analyses the current message and recent conversation history to
    determine which project (astra-backend, astra-frontend, driver-copilot)
    the user is talking about. Sets build_target_id on the build project.

    Returns the resolved build_target_id, or None if ambiguous.
    """
    # v2.3: Check ProjectSession FIRST — the scoping flow already resolved the target.
    # This takes priority even if build_target_id is already set, because
    # the session reflects the user's CURRENT intent (which may differ from
    # a stale build project from a previous run).
    try:
        from app.shared_context.project_session import get_project_session
        from app.pipeline_v2.target_registry import get_profile
        session = get_project_session(str(chat_project_id))
        if session.is_set and session.project_id:
            profile = get_profile(session.project_id)
            if profile:
                if build_project.build_target_id != profile.project_id:
                    build_project.build_target_id = profile.project_id
                    build_project.target_path = profile.project_root
                    db.commit()
                    db.refresh(build_project)
                    logger.info(
                        "[pipeline_bridge] Build target UPDATED from ProjectSession: %s (%s) → %s",
                        profile.project_id, profile.project_name, profile.project_root,
                    )
                    print(
                        f"[PIPELINE_BRIDGE] Build target from session: {profile.project_id} "
                        f"({profile.project_name}) → {profile.project_root}"
                    )
                else:
                    logger.info(
                        "[pipeline_bridge] Build target confirmed by session: %s",
                        profile.project_id,
                    )
                return profile.project_id
    except Exception as e:
        logger.debug("[pipeline_bridge] ProjectSession check failed: %s", e)

    # Fallback: if already set and no session override, keep existing
    if build_project.build_target_id:
        logger.info(
            "[pipeline_bridge] Build target already set (no session override): %s",
            build_project.build_target_id,
        )
        return build_project.build_target_id

    try:
        from app.pipeline_v2.target_registry import resolve_project_from_message

        # Build conversation history for context
        conversation_history = []
        if chat_project_id:
            try:
                from app.memory import service as memory_service
                history = memory_service.get_message_history(db, chat_project_id, limit=15)
                messages = history.messages if hasattr(history, 'messages') else history
                for msg in messages:
                    role = msg.role if hasattr(msg, 'role') else msg.get('role', '')
                    content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
                    conversation_history.append({"role": role, "content": content})
            except Exception as e:
                logger.debug("[pipeline_bridge] Could not load conversation history: %s", e)

        # Resolve from message + history
        profile = resolve_project_from_message(
            message or build_project.original_brief or "",
            conversation_history=conversation_history,
        )

        if profile:
            build_project.build_target_id = profile.project_id
            # Also set target_path from the profile
            build_project.target_path = profile.project_root
            db.commit()
            db.refresh(build_project)
            logger.info(
                "[pipeline_bridge] Resolved build target: %s (%s) for project '%s'",
                profile.project_id, profile.project_name, build_project.name,
            )
            print(
                f"[PIPELINE_BRIDGE] Build target resolved: {profile.project_id} "
                f"({profile.project_name}) → {profile.project_root}"
            )
            return profile.project_id
        else:
            logger.info("[pipeline_bridge] Could not resolve build target from context")
            return None

    except ImportError:
        logger.debug("[pipeline_bridge] target_registry not available")
        return None
    except Exception as e:
        logger.warning("[pipeline_bridge] Build target resolution failed: %s", e)
        return None


def get_build_target_profile(build_project: BuildProject):
    """Load the BuildTargetProfile for a build project.

    Returns the profile object, or None if not set / not found.
    """
    if not build_project.build_target_id:
        return None
    try:
        from app.pipeline_v2.target_registry import get_profile
        return get_profile(build_project.build_target_id)
    except ImportError:
        return None


# ═══════════════════════════════════════════════════════════════════
# Stage notifications
# ═══════════════════════════════════════════════════════════════════

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


def notify_narrative(
    db: Session,
    build_project_id: str,
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
) -> Optional[BuildProject]:
    """Push a rich narrative entry to the build project stage log."""
    from datetime import datetime, timezone

    narrative = {
        "title": title,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": duration_ms or None,
        "model_used": model_used or None,
        "input_summary": input_summary or None,
        "output_summary": output_summary or None,
        "sections": sections or [],
        "files_touched": files_touched or [],
        "warnings": warnings or [],
        "tokens_used": tokens_used or None,
    }

    try:
        return build_service.append_narrative(
            db, build_project_id, stage, narrative,
        )
    except Exception as e:
        logger.warning("[pipeline_bridge] Failed to append narrative: %s", e)
        return None


def save_weaver_extraction(
    db: Session,
    build_project_id: str,
    extraction: dict,
) -> Optional[BuildProject]:
    """Persist the trimmed Weaver extraction to the build project."""
    try:
        project = build_service.get_project(db, build_project_id)
        if not project:
            return None
        project.weaver_extraction = extraction
        db.commit()
        db.refresh(project)
        logger.info(
            "[pipeline_bridge] Saved Weaver extraction for %s (%d keys)",
            build_project_id, len(extraction),
        )
        return project
    except Exception as e:
        logger.warning("[pipeline_bridge] Failed to save Weaver extraction: %s", e)
        return None


def compile_report(
    db: Session,
    build_project_id: str,
) -> Optional[str]:
    """Compile the full build report. Called when pipeline completes."""
    try:
        return build_service.compile_build_report(db, build_project_id)
    except Exception as e:
        logger.warning("[pipeline_bridge] Failed to compile report: %s", e)
        return None


def _extract_project_name(brief: str) -> str:
    """Extract a short project name from the brief text."""
    if not brief:
        return "Untitled Project"

    text = brief.strip()

    patterns = [
        r"(?:I want|I need|we need)\s+to\s+(?:build|create|make|design|implement)\s+(?:out\s+)?(?:a|an|the)\s+(.{5,60}?)(?:\.|,|\n|$)",
        r"(?:build|create|make|design|implement)\s+(?:out\s+)?(?:a|an|the)\s+(.{5,60}?)(?:\.|,|\n|$)",
        r"(?:build|create|make|design|implement)\s+(?:out\s+)?(.{5,60}?)(?:\.|,|\n|$)",
        r"(?:I want|I need|we need)\s+(?:a|an|the)\s+(.{5,60}?)(?:\.|,|\n|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            name = re.sub(r"\s+(that|which|with|for|using|in|on)\s*$", "", name, flags=re.IGNORECASE)
            if len(name) > _MAX_NAME_LEN:
                name = name[:_MAX_NAME_LEN].rsplit(" ", 1)[0]
            return name.strip()

    first_line = text.split("\n")[0].strip()
    if len(first_line) > _MAX_NAME_LEN:
        first_line = first_line[:_MAX_NAME_LEN].rsplit(" ", 1)[0]

    return first_line or "Untitled Project"
