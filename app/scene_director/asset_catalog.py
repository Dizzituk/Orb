# FILE: app/scene_director/asset_catalog.py
# Purpose: Loads and serves the scene asset catalogue (data/scene_assets/catalog.json) —
#          the only prefab_ids the director is allowed to emit.
# Called-by: app.scene_director.director/router, Unity + Room tab via GET /scene/catalog
# Depends-on: stdlib only
# Last-renovated: 2026-06-12
"""Asset catalogue for the scene director.

The catalogue is the vocabulary boundary between the director (LLM) and the
renderer (Unity PrefabCatalog). Entries here are STARTER GUESSES marked
"verified": false until Taz reconciles them against the populated Unity
PrefabCatalog (the ground truth) — see docs/astra_room/unity_drop/SETUP.md.

Cached in-process; reloaded automatically when the file's mtime changes, so
catalogue edits take effect without a backend restart.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "scene_assets" / "catalog.json"

_cache: Optional[Dict[str, Any]] = None
_cache_mtime: Optional[float] = None


def load_catalog() -> Dict[str, Any]:
    """Return the parsed catalogue dict ({"version":…, "entries":[…]}).

    mtime-cached. A missing or unparseable file returns an empty catalogue
    (and logs) rather than raising — the director then falls back to the
    deterministic scene, which is the designed worst case.
    """
    global _cache, _cache_mtime
    try:
        mtime = CATALOG_PATH.stat().st_mtime
    except OSError:
        logger.warning("[scene] catalogue missing at %s", CATALOG_PATH)
        return {"version": 0, "entries": []}
    if _cache is not None and _cache_mtime == mtime:
        return _cache
    try:
        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("[scene] catalogue unreadable (%s) — serving empty", exc)
        return {"version": 0, "entries": []}
    if isinstance(data, list):  # tolerate a bare entry list
        data = {"version": 0, "entries": data}
    data.setdefault("entries", [])
    _cache, _cache_mtime = data, mtime
    return data


def get(prefab_id: str) -> Optional[Dict[str, Any]]:
    """Return the catalogue entry for prefab_id, or None."""
    for entry in load_catalog()["entries"]:
        if entry.get("prefab_id") == prefab_id:
            return entry
    return None


def list_by_kind(kind: str) -> List[Dict[str, Any]]:
    """All entries of one kind: 'environment' | 'actor' | 'skybox'."""
    return [e for e in load_catalog()["entries"] if e.get("kind") == kind]


def known_ids(kind: Optional[str] = None) -> set[str]:
    """Set of valid prefab_ids, optionally restricted to one kind."""
    entries = list_by_kind(kind) if kind else load_catalog()["entries"]
    return {e["prefab_id"] for e in entries if e.get("prefab_id")}


def as_prompt_block() -> str:
    """Compact text listing of the catalogue for the director prompt."""
    lines: List[str] = []
    for kind, label in (("environment", "ENVIRONMENT PROPS"),
                        ("actor", "ACTORS"),
                        ("skybox", "SKYBOX PRESETS")):
        entries = list_by_kind(kind)
        if not entries:
            continue
        lines.append(f"{label}:")
        for e in entries:
            tags = ",".join(e.get("tags") or [])
            extra = ""
            if e.get("footprint_m"):
                fp = e["footprint_m"]
                extra = f" footprint {fp[0]}x{fp[2]}m"
            lines.append(f"  - {e['prefab_id']}: {e.get('display', '')} [{tags}]{extra}")
    return "\n".join(lines)
