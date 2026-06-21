# FILE: app/pipeline_v2/target_registry.py
# Purpose: Build Target Registry.
# Called-by: app.builds.pipeline_bridge, app.cloud.build_and_deploy, app.debug.debug_chat, app.debug.orchestrator.decomposer (+15 more)
# Depends-on: app.pipeline_v2.build_targets
# Last-renovated: 2026-06-11
"""
Build Target Registry.

Central registry of all known build target profiles. The pipeline
looks up the active profile by ID to configure every stage.

v1.0 (2026-03-10): Initial — 3 profiles.
v1.1 (2026-03-10): Expanded signal keywords, JAVA_HOME for Gradle.
v1.2 (2026-03-11): Added AstraBridge profile, "work on yourself" keywords,
    Astra Bridge signal keywords.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from app.pipeline_v2.build_targets import BuildTargetProfile

logger = logging.getLogger(__name__)

# ── Re-export surface (SPLIT BATCH 9, 2026-06-21) ───────────────────────────
# Split into single-responsibility leaves; this stays the import front door (20
# importers) and re-exports the full public surface so none of them need editing.
#   target_profiles.py        — the 4 profile literals + _REGISTRY + ALL_PROFILES
#   project_resolution.py     — message -> target scoring
#   file_target_resolution.py — path -> target + the ambient job-target hint
# Registry accessors (get_profile/...) + the TargetGroup cluster stay here. The
# shared _REGISTRY is imported (not copied), so register_profile mutates the same
# dict the resolver leaves read.
from app.pipeline_v2.target_profiles import (  # noqa: F401
    ASTRA_BACKEND,
    ASTRA_FRONTEND,
    DRIVER_COPILOT,
    ASTRA_BRIDGE,
    _REGISTRY,
    ALL_PROFILES,
)
from app.pipeline_v2.project_resolution import (  # noqa: F401
    detect_all_projects_from_message,
    resolve_project_from_message,
    _check_explicit_build_target_directive,
    _score_projects_from_text,
    _SIGNAL_WEIGHTS,
    _PROJECT_SIGNALS,
    _SELF_WORK_SIGNALS,
)
from app.pipeline_v2.file_target_resolution import (  # noqa: F401
    set_job_target_hint,
    get_job_target_hint,
    resolve_target_for_files,
    _resolve_single_file,
)


# ── Registry accessors (read / mutate the shared _REGISTRY in place) ─────────
def get_profile(project_id: str) -> Optional[BuildTargetProfile]:
    """Look up a build target profile by ID."""
    return _REGISTRY.get(project_id)


def get_default_profile() -> BuildTargetProfile:
    """Return the default profile (ASTRA backend)."""
    target = os.getenv("ASTRA_V2_BUILD_TARGET", "astra-backend")
    return _REGISTRY.get(target, ASTRA_BACKEND)


def list_profiles() -> List[BuildTargetProfile]:
    """Return all registered profiles."""
    return list(_REGISTRY.values())


def register_profile(profile: BuildTargetProfile) -> None:
    """Register a new build target profile."""
    _REGISTRY[profile.project_id] = profile
    logger.info("[registry] Registered profile: %s (%s)", profile.project_id, profile.project_name)


# ═══════════════════════════════════════════════════════════════════
# Target Groups (v1.2, 2026-04-11)
# ═══════════════════════════════════════════════════════════════════
# A TargetGroup names a bundle of targets that frequently get worked on
# together as a coordinated system. When detect_all_projects_from_message
# returns multiple hits matching a known group, the build project inherits
# the group as its scope, signalling to the segmenter that this is a
# first-class multi-target job (not an accident).

from dataclasses import dataclass as _dc, field as _field


@_dc
class TargetGroup:
    group_id: str
    group_name: str
    target_ids: list           # member target IDs
    description: str = ""


_TARGET_GROUPS = {
    "astra-system": TargetGroup(
        group_id="astra-system",
        group_name="ASTRA System (backend + bridge + frontend)",
        target_ids=["astra-backend", "astra-bridge", "astra-frontend"],
        description="Coordinated upgrades spanning the ASTRA backend, the "
                    "Android bridge app, and the Electron desktop frontend.",
    ),
    "astra-mobile": TargetGroup(
        group_id="astra-mobile",
        group_name="ASTRA Backend + Bridge",
        target_ids=["astra-backend", "astra-bridge"],
        description="Backend + Android bridge coordinated work — the most "
                    "common multi-target pattern (e.g. hands-free upgrades, "
                    "offline sync, missed-reply recovery).",
    ),
    "delivery-system": TargetGroup(
        group_id="delivery-system",
        group_name="Backend + Driver CoPilot",
        target_ids=["astra-backend", "driver-copilot"],
        description="Backend + delivery driver Android app.",
    ),
}


def get_target_group(group_id: str):
    return _TARGET_GROUPS.get(group_id)


def list_target_groups() -> list:
    return list(_TARGET_GROUPS.values())


def find_group_for_targets(target_ids: set) -> "TargetGroup | None":
    """Find the smallest registered group that contains all given targets.

    Returns the most specific match. If multiple groups contain the targets,
    prefers the group with the fewest extra members (closest fit).
    """
    matches = []
    for g in _TARGET_GROUPS.values():
        if target_ids.issubset(set(g.target_ids)):
            matches.append((len(g.target_ids), g))
    if not matches:
        return None
    matches.sort(key=lambda m: m[0])
    return matches[0][1]
