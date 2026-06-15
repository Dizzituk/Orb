# FILE: app/scene_director/asset_catalog.py
# Purpose: Loads and serves the scene asset catalogue (data/scene_assets/catalog.json) —
#          the only prefab_ids the director may emit. v2 (2026-06-13): category/era/tag
#          queries + a grouped, capped, era-filtered as_prompt_block so the director sees
#          its full toy box without an unbounded prompt.
# Called-by: app.scene_director.director/critic/router/research, Unity + Room tab via GET /scene/catalog
# Depends-on: stdlib only
# Last-renovated: 2026-06-13
"""Asset catalogue for the scene director.

The catalogue is the vocabulary boundary between the director (LLM) and the
renderer (Unity PrefabCatalog). v1 entries were 10 starter guesses; v3
introspection (AstraRoomSetup.IntrospectCatalogue) indexes the real pack
contents by category with footprints. mtime-cached — edits take effect with no
restart.

ENTRY SHAPE (v2, all keys optional except prefab_id; old v1 entries still work):
  prefab_id, display, kind (environment|actor|skybox — backward-compat),
  category (terrain|vegetation|structure|vehicle|prop|character|skybox),
  tags[], footprint_m [x,y,z], drivable (bool), era, source_pack, unity_path, verified.
`kind` is derived from `category` (structure/prop/vegetation/terrain/vehicle →
environment; character → actor; skybox → skybox) so list_by_kind/known_ids keep working.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "scene_assets" / "catalog.json"

# Director-prompt grouping order + how many to show per category before summarising.
_PROMPT_GROUPS = [
    ("terrain", "TERRAIN & GROUND"),
    ("structure", "BUILDINGS & STRUCTURES"),
    ("vegetation", "VEGETATION (trees/plants/rocks — good for scatter)"),
    ("prop", "STREET FURNITURE & PROPS (lights/benches/signs/fences...)"),
    ("vehicle", "VEHICLES (drivable — use as vehicle-actors)"),
    ("character", "CHARACTERS (person-actors)"),
    ("skybox", "SKYBOX PRESETS"),
]
_DEFAULT_MAX_PER_GROUP = 36

_cache: Optional[Dict[str, Any]] = None
_cache_mtime: Optional[float] = None


def load_catalog() -> Dict[str, Any]:
    """Return the parsed catalogue dict ({"version":…, "entries":[…]}).
    mtime-cached. A missing/unparseable file returns an empty catalogue (logs,
    never raises) — the director then falls back to the deterministic scene."""
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
    if isinstance(data, list):
        data = {"version": 0, "entries": data}
    data.setdefault("entries", [])
    _cache, _cache_mtime = data, mtime
    return data


def _entries() -> List[Dict[str, Any]]:
    return load_catalog().get("entries", [])


def get(prefab_id: str) -> Optional[Dict[str, Any]]:
    for entry in _entries():
        if entry.get("prefab_id") == prefab_id:
            return entry
    return None


def _category_of(entry: Dict[str, Any]) -> str:
    """category if present, else inferred from kind (backward-compat)."""
    cat = entry.get("category")
    if cat:
        return cat
    kind = entry.get("kind")
    if kind == "actor":
        return "character"
    if kind == "skybox":
        return "skybox"
    return "structure"  # generic environment fallback


# ── kind queries (v1 API — unchanged) ────────────────────────────────────────

def list_by_kind(kind: str) -> List[Dict[str, Any]]:
    """All entries of one kind: 'environment' | 'actor' | 'skybox'."""
    return [e for e in _entries() if e.get("kind") == kind]


def known_ids(kind: Optional[str] = None) -> set[str]:
    entries = list_by_kind(kind) if kind else _entries()
    return {e["prefab_id"] for e in entries if e.get("prefab_id")}


# ── category / era / tag queries (v2) ────────────────────────────────────────

def list_by_category(category: str) -> List[Dict[str, Any]]:
    return [e for e in _entries() if _category_of(e) == category]


def known_ids_by_category(category: str) -> set[str]:
    return {e["prefab_id"] for e in list_by_category(category) if e.get("prefab_id")}


def categories() -> set[str]:
    return {_category_of(e) for e in _entries()}


def list_by_era(era: Optional[str]) -> List[Dict[str, Any]]:
    """Entries available for an era. None/'any'/'modern' returns everything that
    is modern or era-less (the common case). A specific era returns matches only."""
    if not era or era.lower() in ("any", "all"):
        return _entries()
    era = era.lower()
    out = [e for e in _entries() if str(e.get("era", "modern")).lower() == era]
    return out


def eras() -> set[str]:
    return {str(e.get("era", "modern")).lower() for e in _entries()}


def footprint_of(prefab_id: str) -> Optional[List[float]]:
    e = get(prefab_id)
    fp = e.get("footprint_m") if e else None
    return fp if isinstance(fp, list) and len(fp) >= 3 else None


def is_drivable(prefab_id: str) -> bool:
    e = get(prefab_id)
    return bool(e and e.get("drivable"))


def tagged(tag: str) -> List[Dict[str, Any]]:
    tag = tag.lower()
    return [e for e in _entries() if tag in [str(t).lower() for t in (e.get("tags") or [])]]


# ── director prompt block (v2: grouped by category, capped, era-filtered) ────

def as_prompt_block(era: Optional[str] = None, max_per_group: int = _DEFAULT_MAX_PER_GROUP) -> str:
    """Compact catalogue listing for the director prompt, grouped by category.
    Filtered to `era` (modern/era-less always included). Each group is capped at
    max_per_group with a '+N more' note so the prompt stays bounded; the director
    only ever uses ids it is shown (others are stripped by sanitise_scene)."""
    pool = list_by_era(era)
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for e in pool:
        by_cat.setdefault(_category_of(e), []).append(e)

    lines: List[str] = []
    for cat, label in _PROMPT_GROUPS:
        entries = by_cat.get(cat) or []
        if not entries:
            continue
        lines.append(f"{label}:")
        shown = entries[:max_per_group]
        for e in shown:
            tags = ",".join(e.get("tags") or [])
            extra = ""
            fp = e.get("footprint_m")
            if isinstance(fp, list) and len(fp) >= 3:
                extra = f" ~{round(fp[0], 1)}x{round(fp[2], 1)}m"
            drive = " [drivable]" if e.get("drivable") else ""
            lines.append(f"  - {e['prefab_id']}: {e.get('display', '')} [{tags}]{extra}{drive}")
        if len(entries) > len(shown):
            sample = ", ".join(x["prefab_id"] for x in entries[len(shown):len(shown) + 4])
            lines.append(f"  ...and {len(entries) - len(shown)} more {cat} (e.g. {sample}) — all valid ids")
    return "\n".join(lines)
