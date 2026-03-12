# FILE: app/llm/routing/_cd_project_registry.py
"""
Dispatch table extension for Project Registry commands.

Adds SCAN_PROJECT_ARCHITECTURE and LIST_REGISTERED_PROJECTS
handlers to the command dispatch system.

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.llm.translation_routing import CanonicalIntent

logger = logging.getLogger(__name__)


def get_project_registry_entries(registry: Any) -> List[Dict[str, Any]]:
    """Return dispatch table entries for project registry commands."""
    pr_available = getattr(registry, "_PROJECT_REGISTRY_AVAILABLE", False)

    return [
        {
            "intent": CanonicalIntent.SCAN_PROJECT_ARCHITECTURE,
            "available": pr_available,
            "handler": _get_scan_handler(registry),
            "stage": "project_architecture_scan",
            "provider_override": lambda: ("local", "project_scanner"),
            "handler_kwargs_fn": _project_scan_kwargs,
        },
        {
            "intent": CanonicalIntent.LIST_REGISTERED_PROJECTS,
            "available": pr_available,
            "handler": _make_list_projects_handler(),
            "stage": "list_registered_projects",
            "provider_override": lambda: ("local", "project_registry"),
            "no_standard_args": True,
        },
    ]


def _get_scan_handler(registry: Any):
    """Get the project scan stream handler from registry."""
    handler = getattr(registry, "generate_project_scan_stream", None)
    return handler


def _project_scan_kwargs(req: Any, translation_result: Any) -> dict:
    """Extract project reference from context for scanning."""
    try:
        from app.project_registry import service as reg_svc
        from app.db import get_db_session

        extracted = getattr(translation_result, "extracted_context", {}) or {}
        slug = extracted.get("extracted_query", "")

        db = get_db_session()
        try:
            project = None
            if slug:
                project = reg_svc.get_project_by_slug(db, slug)
            if not project:
                project = reg_svc.resolve_project_from_message(db, req.message)
            return {
                "project": project,
                "chat_project_id": req.project_id,
            }
        finally:
            db.close()
    except Exception as e:
        logger.warning("[project_registry] Failed to resolve project: %s", e)
        return {"project": None, "chat_project_id": req.project_id}


def _make_list_projects_handler():
    """Create a handler that lists registered projects as SSE stream."""

    async def _list_stream():
        from app.db import get_db_session
        from app.project_registry import service as reg_svc

        db = get_db_session()
        try:
            projects = reg_svc.list_projects(db)
            if not projects:
                msg = "No projects registered yet. Use a register project command to add one."
                yield "data: " + json.dumps({"type": "token", "content": msg}) + "\n\n"
            else:
                header = f"**Registered Projects** ({len(projects)})\n\n"
                yield "data: " + json.dumps({"type": "token", "content": header}) + "\n\n"

                for p in projects:
                    staleness = reg_svc.check_staleness(p)
                    scan_str = p.last_scan_at.strftime("%Y-%m-%d %H:%M") if p.last_scan_at else "never"
                    tag = " STALE" if staleness["is_stale"] else ""
                    roots = reg_svc.get_root_paths(p)
                    line = (
                        f"- **{p.name}** ({p.project_type or 'unknown'}, "
                        f"{p.primary_language or '?'}){tag}\n"
                        f"  Last scan: {scan_str} - {p.last_scan_file_count or 0} files\n"
                        f"  Roots: {', '.join(roots)}\n\n"
                    )
                    yield "data: " + json.dumps({"type": "token", "content": line}) + "\n\n"
        except Exception as e:
            yield "data: " + json.dumps({"type": "error", "error": str(e)}) + "\n\n"
        finally:
            db.close()

        yield "data: " + json.dumps({
            "type": "done",
            "provider": "local",
            "model": "project_registry",
            "total_length": 0,
        }) + "\n\n"

    return _list_stream