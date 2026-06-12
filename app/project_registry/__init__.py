# FILE: app/project_registry/__init__.py
# Purpose: ASTRA Project Registry — multi-project architecture awareness.
# Called-by: app.db, app.llm.routing._cd_project_registry, app.project_registry.api_router, app.project_registry.scanner (+2 more)
# Depends-on: app.project_registry.models, app.project_registry.service
# Last-renovated: 2026-06-11
"""
ASTRA Project Registry — multi-project architecture awareness.

Tracks external projects (apps ASTRA builds/manages) and their
architecture scan state. Enables scoped RAG queries, staleness
detection, and proactive rescan suggestions.

Usage:
    from app.project_registry import service as registry
    from app.project_registry.models import RegisteredProject

    # Register a new project
    project = registry.register_project(
        db, name="Driver CoPilot",
        root_paths=[r"D:\Astra Android Folder\AndroidDriverCopilot"],
        project_type="android",
        primary_language="kotlin",
    )

    # Check staleness
    result = registry.check_staleness(project)
    if result["is_stale"]:
        print(result["message"])

v1.0 (2026-03): Initial implementation.
"""

from . import service
from .models import RegisteredProject

__all__ = ["service", "RegisteredProject"]