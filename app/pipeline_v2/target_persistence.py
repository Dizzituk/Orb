# FILE: app/pipeline_v2/target_persistence.py
# Purpose: JSON persistence for DYNAMIC BuildTargetProfiles (project-scoping targets).
# Called-by: app.pipeline_v2.target_registry
# Depends-on: app.pipeline_v2.build_targets, app.pipeline_v2.target_profiles
# Last-renovated: 2026-07-04
"""
JSON persistence for dynamic BuildTargetProfiles.

Profiles created at runtime by the project scoping flow
(app/llm/project_scoping_stream.py -> register_profile) used to live only in
the in-memory _REGISTRY dict and vanished on backend restart, silently
re-routing greenfield jobs to the astra-backend default target. This leaf
gives them a durable home.

Store: data/build_targets.json (repo-root anchored, same convention as
data/routing_policy.json). One flat dict keyed by project_id, values are
dataclasses.asdict(BuildTargetProfile). Override the location with the
ASTRA_BUILD_TARGETS_STORE env var (used by tests).

Rules:
- The 4 built-in profiles (astra-backend / astra-frontend / driver-copilot /
  astra-bridge) are NEVER persisted — they are code, not state.
- Nothing here raises at import time or on a corrupt/missing store: readers
  log and return empty, writers log and return False.
- Writes are atomic (tmp file + os.replace).

v1.0 (2026-07-04): JOB 1 of the greenfield build-target routing fixpack.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

from app.pipeline_v2.build_targets import BuildTargetProfile
from app.pipeline_v2.target_profiles import (
    ASTRA_BACKEND,
    ASTRA_BRIDGE,
    ASTRA_FRONTEND,
    DRIVER_COPILOT,
)

logger = logging.getLogger(__name__)

# IDs that must never enter the JSON store. Derived from the built-in profile
# literals so a rename there propagates here automatically.
BUILTIN_PROFILE_IDS = frozenset({
    ASTRA_BACKEND.project_id,
    ASTRA_FRONTEND.project_id,
    DRIVER_COPILOT.project_id,
    ASTRA_BRIDGE.project_id,
})

_STORE_ENV = "ASTRA_BUILD_TARGETS_STORE"


def _store_path() -> Path:
    """Resolve the JSON store path (env override wins, else data/ at repo root)."""
    override = (os.getenv(_STORE_ENV) or "").strip()
    if override:
        return Path(override)
    # app/pipeline_v2/target_persistence.py -> <repo root>/data/build_targets.json
    return Path(__file__).resolve().parent.parent.parent / "data" / "build_targets.json"


def _read_store() -> Dict[str, dict]:
    """Read the raw store. Missing or corrupt file -> {} (logged, never raises)."""
    path = _store_path()
    try:
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            logger.warning("[target_persistence] Store %s is not a dict — ignoring", path)
            return {}
        return raw
    except Exception as e:
        logger.warning("[target_persistence] Could not read store %s: %s", path, e)
        return {}


def _write_store(store: Dict[str, dict]) -> None:
    """Atomically write the store (tmp + os.replace)."""
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def save_dynamic_profile(profile: BuildTargetProfile) -> bool:
    """Persist a dynamic profile to the JSON store.

    Built-in profile IDs are refused (store left untouched). Returns True only
    when the profile was actually written. Never raises.
    """
    try:
        project_id = getattr(profile, "project_id", None)
        if not project_id:
            logger.warning("[target_persistence] Refusing to persist profile without project_id")
            return False
        if project_id in BUILTIN_PROFILE_IDS:
            logger.debug(
                "[target_persistence] Skipping persist of built-in profile: %s", project_id,
            )
            return False
        store = _read_store()
        store[project_id] = dataclasses.asdict(profile)
        _write_store(store)
        logger.info(
            "[target_persistence] Saved dynamic profile '%s' -> %s", project_id, _store_path(),
        )
        return True
    except Exception as e:
        logger.warning("[target_persistence] Failed to persist profile: %s", e)
        return False


def load_dynamic_profiles() -> List[BuildTargetProfile]:
    """Load all persisted dynamic profiles.

    Tolerant of a missing/corrupt store and of field drift: unknown keys from
    older/newer schema versions are dropped, entries that still fail to
    construct are skipped with a warning. Built-in IDs found in the store are
    ignored (built-ins win). Never raises.
    """
    profiles: List[BuildTargetProfile] = []
    known_fields = set(BuildTargetProfile.__dataclass_fields__.keys())
    for project_id, raw in _read_store().items():
        if project_id in BUILTIN_PROFILE_IDS:
            logger.warning(
                "[target_persistence] Ignoring built-in ID '%s' found in store", project_id,
            )
            continue
        if not isinstance(raw, dict):
            logger.warning("[target_persistence] Skipping non-dict entry '%s'", project_id)
            continue
        fields = {k: v for k, v in raw.items() if k in known_fields}
        fields["project_id"] = project_id  # key is authoritative
        try:
            profiles.append(BuildTargetProfile(**fields))
        except Exception as e:
            logger.warning(
                "[target_persistence] Skipping unloadable profile '%s': %s", project_id, e,
            )
    return profiles


def remove_dynamic_profile(project_id: str) -> bool:
    """Remove a profile from the JSON store. True if it was present. Never raises."""
    try:
        store = _read_store()
        if project_id not in store:
            return False
        del store[project_id]
        _write_store(store)
        logger.info("[target_persistence] Removed dynamic profile '%s' from store", project_id)
        return True
    except Exception as e:
        logger.warning("[target_persistence] Failed to remove profile '%s': %s", project_id, e)
        return False
