# FILE: app/project_registry/staleness.py
# Purpose: Proactive staleness detection for registered projects.
# Called-by: app.project_registry.chat_injection
# Depends-on: app.project_registry, app.project_registry.models, app.project_registry.service
# Last-renovated: 2026-06-11
"""
Proactive staleness detection for registered projects.

Called during chat routing to check if the project being discussed
has a stale architecture scan. If so, returns a suggestion message
and confirmation gate data for the frontend.

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from . import service as registry_service
from .models import RegisteredProject

logger = logging.getLogger(__name__)


def check_and_suggest_rescan(
    db: Session,
    message: str,
) -> Optional[dict]:
    """
    Check if the message references a project with a stale scan.

    Returns a dict suitable for injecting a confirmation gate into
    the SSE stream, or None if no suggestion is needed.

    Return format:
        {
            "project": RegisteredProject,
            "slug": str,
            "message": str,  # Human-readable suggestion
            "intent": "SCAN_PROJECT_ARCHITECTURE",
            "confirmation_data": {
                "intent": "scan_project",
                "extracted_query": "<slug>",
            },
        }
    """
    # Try to identify which project the message is about
    project = registry_service.resolve_project_from_message(db, message)
    if not project:
        return None

    # Check staleness
    result = registry_service.check_staleness(project)
    if not result["is_stale"]:
        return None

    logger.info(
        "[staleness] Project %s scan is stale (%.1f hours old)",
        project.name,
        result.get("hours_since_scan") or -1,
    )

    return {
        "project": project,
        "slug": project.slug,
        "message": result["message"],
        "intent": "SCAN_PROJECT_ARCHITECTURE",
        "confirmation_data": {
            "intent": "scan_project",
            "extracted_query": project.slug,
        },
    }