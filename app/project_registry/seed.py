# FILE: app/project_registry/seed.py
# Purpose: Seed the project registry with known projects.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.db, app.project_registry, app.project_registry.service
# Last-renovated: 2026-06-11
"""
Seed the project registry with known projects.

Run once to register ASTRA's known external projects.
Safe to run multiple times — skips existing projects.

Usage:
    python -m app.project_registry.seed
"""
from __future__ import annotations

import logging

from app.db import init_db, get_db_session
from . import service as registry_service

logger = logging.getLogger(__name__)


SEED_PROJECTS = [
    {
        "name": "Driver CoPilot",
        "root_paths": [r"D:\Astra Android Folder\AndroidDriverCopilot"],
        "project_type": "android",
        "primary_language": "kotlin",
        "description": (
            "Android delivery driver app (Kotlin/Jetpack Compose, MVVM, Room v8, ML Kit OCR). "
            "Accessibility Service for Yodel UI, voice interface with Gemini API, "
            "OBD2 Bluetooth, geofencing, OCR receipt capture."
        ),
    },
]


def seed_projects():
    """Register all seed projects. Skips existing ones."""
    init_db()
    db = get_db_session()
    try:
        for proj_data in SEED_PROJECTS:
            name = proj_data["name"]
            existing = registry_service.get_project_by_name(db, name)
            if existing:
                print(f"[seed] Already registered: {name} (slug={existing.slug})")
                continue

            project = registry_service.register_project(
                db=db,
                name=proj_data["name"],
                root_paths=proj_data["root_paths"],
                project_type=proj_data.get("project_type"),
                primary_language=proj_data.get("primary_language"),
                description=proj_data.get("description"),
            )
            print(f"[seed] Registered: {project.name} (slug={project.slug})")
    finally:
        db.close()

    print("[seed] Done.")


if __name__ == "__main__":
    seed_projects()