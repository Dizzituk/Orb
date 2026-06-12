# FILE: app/project_registry/service.py
# Purpose: Service layer for the ASTRA Project Registry.
# Called-by: app.llm.routing._cd_project_registry, app.project_registry, app.project_registry.api_router, app.project_registry.scanner (+2 more)
# Depends-on: app.project_registry.models
# Last-renovated: 2026-06-11
"""
Service layer for the ASTRA Project Registry.

CRUD operations for registered projects, scan staleness checks,
and helpers for scoping architecture queries to specific projects.

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy.orm import Session

from .models import RegisteredProject

logger = logging.getLogger(__name__)

# ── Staleness thresholds ────────────────────────────────────────────────

# Default: scan is stale after 2 days
DEFAULT_STALE_HOURS = 48

# If there have been builds since the last scan, lower the threshold
POST_BUILD_STALE_HOURS = 12


# ── Slug generation ─────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    """Generate a URL-safe slug from a project name."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug or "project"


# ── CRUD ────────────────────────────────────────────────────────────────

def register_project(
    db: Session,
    name: str,
    root_paths: List[str],
    project_type: Optional[str] = None,
    primary_language: Optional[str] = None,
    description: Optional[str] = None,
) -> RegisteredProject:
    """Register a new external project for architecture tracking."""
    slug = _slugify(name)

    # Check for existing
    existing = db.query(RegisteredProject).filter(
        RegisteredProject.slug == slug,
    ).first()
    if existing:
        # Re-activate if soft-deleted
        if not existing.is_active:
            existing.is_active = True
            existing.root_paths_json = json.dumps(root_paths)
            existing.project_type = project_type
            existing.primary_language = primary_language
            existing.description = description
            db.commit()
            db.refresh(existing)
            logger.info("[project_registry] Re-activated project: %s", name)
            return existing
        raise ValueError(f"Project '{name}' already registered (slug={slug})")

    project = RegisteredProject(
        name=name,
        slug=slug,
        root_paths_json=json.dumps(root_paths),
        project_type=project_type,
        primary_language=primary_language,
        description=description,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    logger.info("[project_registry] Registered project: %s (%s)", name, slug)
    return project


def list_projects(db: Session, active_only: bool = True) -> List[RegisteredProject]:
    """List all registered projects."""
    query = db.query(RegisteredProject)
    if active_only:
        query = query.filter(RegisteredProject.is_active == True)
    return query.order_by(RegisteredProject.name).all()


def get_project_by_slug(db: Session, slug: str) -> Optional[RegisteredProject]:
    """Look up a project by its slug."""
    return db.query(RegisteredProject).filter(
        RegisteredProject.slug == slug,
        RegisteredProject.is_active == True,
    ).first()


def get_project_by_name(db: Session, name: str) -> Optional[RegisteredProject]:
    """Fuzzy match a project by name (case-insensitive contains)."""
    return db.query(RegisteredProject).filter(
        RegisteredProject.is_active == True,
        RegisteredProject.name.ilike(f"%{name}%"),
    ).first()


def get_project_by_id(db: Session, project_id: int) -> Optional[RegisteredProject]:
    """Look up a project by ID."""
    return db.query(RegisteredProject).filter(
        RegisteredProject.id == project_id,
        RegisteredProject.is_active == True,
    ).first()


def update_scan_metadata(
    db: Session,
    project_id: int,
    scan_id: int,
    file_count: int,
    arch_map_path: Optional[str] = None,
) -> None:
    """Update a project's scan metadata after a successful scan."""
    project = db.query(RegisteredProject).get(project_id)
    if not project:
        return
    project.last_scan_at = datetime.utcnow()
    project.last_scan_id = scan_id
    project.last_scan_file_count = file_count
    if arch_map_path:
        project.has_arch_map = True
        project.arch_map_path = arch_map_path
    db.commit()
    logger.info(
        "[project_registry] Updated scan metadata for %s: scan_id=%d, files=%d",
        project.name, scan_id, file_count,
    )


def deregister_project(db: Session, slug: str) -> bool:
    """Soft-delete a project."""
    project = get_project_by_slug(db, slug)
    if not project:
        return False
    project.is_active = False
    db.commit()
    logger.info("[project_registry] Deregistered project: %s", slug)
    return True


def get_root_paths(project: RegisteredProject) -> List[str]:
    """Parse the root paths JSON from a project."""
    try:
        return json.loads(project.root_paths_json)
    except (json.JSONDecodeError, TypeError):
        return []


# ── Staleness detection ─────────────────────────────────────────────────

def check_staleness(
    project: RegisteredProject,
    stale_hours: int = DEFAULT_STALE_HOURS,
) -> dict:
    """
    Check if a project's architecture scan is stale.

    Returns a dict with:
        - is_stale: bool
        - hours_since_scan: float or None
        - last_scan_at: datetime or None
        - message: human-readable staleness description
    """
    if not project.last_scan_at:
        return {
            "is_stale": True,
            "hours_since_scan": None,
            "last_scan_at": None,
            "message": f"{project.name} has never been scanned.",
        }

    now = datetime.utcnow()
    delta = now - project.last_scan_at
    hours = delta.total_seconds() / 3600

    if hours >= stale_hours:
        days = delta.days
        if days >= 1:
            age_str = f"{days} day{'s' if days != 1 else ''}"
        else:
            age_str = f"{int(hours)} hours"
        return {
            "is_stale": True,
            "hours_since_scan": round(hours, 1),
            "last_scan_at": project.last_scan_at,
            "message": (
                f"The architecture scan for {project.name} is {age_str} old. "
                f"Want me to refresh it?"
            ),
        }

    return {
        "is_stale": False,
        "hours_since_scan": round(hours, 1),
        "last_scan_at": project.last_scan_at,
        "message": "",
    }


def find_stale_projects(
    db: Session,
    stale_hours: int = DEFAULT_STALE_HOURS,
) -> List[dict]:
    """
    Find all projects with stale or missing scans.

    Returns list of dicts with project info and staleness details.
    Used by the chat routing layer to proactively suggest rescans.
    """
    projects = list_projects(db, active_only=True)
    stale = []
    for project in projects:
        result = check_staleness(project, stale_hours)
        if result["is_stale"]:
            stale.append({
                "project": project,
                "slug": project.slug,
                "name": project.name,
                **result,
            })
    return stale


def resolve_project_from_message(
    db: Session,
    message: str,
) -> Optional[RegisteredProject]:
    """
    Try to identify which registered project a message is about.

    Uses fuzzy name matching against the message text.
    Returns the best match or None.
    """
    projects = list_projects(db, active_only=True)
    if not projects:
        return None

    msg_lower = message.lower()

    # Exact name match first
    for project in projects:
        if project.name.lower() in msg_lower:
            return project

    # Slug match
    for project in projects:
        if project.slug in msg_lower:
            return project

    # Word-level fuzzy: check if any significant word from project name appears
    for project in projects:
        words = [w for w in project.name.lower().split() if len(w) >= 4]
        if words and all(w in msg_lower for w in words):
            return project

    return None