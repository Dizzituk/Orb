# FILE: app/intelligent_memory/hot_cache.py
"""
Layer 1 — Hot Cache.

In-memory Python dict holding high-frequency data that is needed
in almost every interaction. Loaded on startup, updated in real-time,
serialised to JSON on disk for persistence across restarts.

Contents:
  - User profile (name, location, background, goals)
  - Active projects (current phase, status, blockers)
  - Recent decisions (last 10 architectural/strategic choices)
  - Pipeline config (current models, feature flags, costs)
  - Preference snapshot (top 20 highest-confidence preferences)
  - Active action items (open tasks from conversations)

Target size: under 50KB. Items demoted to Knowledge Graph if exceeded.

Preloading: On conversation start, the full Hot Cache is injected as
system context. Zero retrieval latency.

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-MEM-001.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Persistence path
HOT_CACHE_PATH = os.getenv(
    "ASTRA_HOT_CACHE_PATH",
    os.path.join("D:", os.sep, "Orb", "data", "hot_cache.json"),
)
MAX_CACHE_BYTES = 50 * 1024  # 50KB target


# ═══════════════════════════════════════════════════════════════════
# Schema — defines every section of the hot cache
# ═══════════════════════════════════════════════════════════════════

DEFAULT_SCHEMA: Dict[str, Any] = {
    "user_profile": {
        "name": "",
        "location": "",
        "background": [],
        "goals": [],
        "communication_style": "",
    },
    "active_projects": {},
    "recent_decisions": [],
    "pipeline_config": {
        "builder_model": "",
        "verifier_model": "",
        "feature_flags": {},
        "current_costs": {},
    },
    "preference_snapshot": [],
    "action_items": [],
    "metadata": {
        "version": 1,
        "last_updated": "",
        "size_bytes": 0,
        "item_count": 0,
    },
}


# ═══════════════════════════════════════════════════════════════════
# Hot Cache class
# ═══════════════════════════════════════════════════════════════════

class HotCache:
    """In-memory fast-access layer for high-frequency data.

    Thread-safe for reads (single-writer assumption in ASTRA).
    All mutations go through set/update methods which auto-persist.
    """

    def __init__(self, cache_path: str = HOT_CACHE_PATH):
        self._data: Dict[str, Any] = {}
        self._cache_path = cache_path
        self._dirty = False
        self._load()

    # ── Core access ──────────────────────────────────────────────

    def get(self, section: str, key: str = "", default: Any = None) -> Any:
        """Read from the hot cache.

        Args:
            section: Top-level section (e.g. "user_profile").
            key: Optional sub-key within the section.
            default: Default if not found.

        Returns:
            The value, or default.
        """
        section_data = self._data.get(section, {})
        if not key:
            return section_data if section_data else default
        if isinstance(section_data, dict):
            return section_data.get(key, default)
        return default

    def set(self, section: str, key: str, value: Any) -> None:
        """Write to the hot cache.

        Args:
            section: Top-level section.
            key: Sub-key within the section.
            value: Value to store.
        """
        if section not in self._data:
            self._data[section] = {}
        if isinstance(self._data[section], dict):
            self._data[section][key] = value
        self._dirty = True
        self._update_metadata()
        self._persist()

    def set_section(self, section: str, data: Any) -> None:
        """Replace an entire section."""
        self._data[section] = data
        self._dirty = True
        self._update_metadata()
        self._persist()

    def delete(self, section: str, key: str = "") -> bool:
        """Remove a key or entire section."""
        if key and section in self._data and isinstance(self._data[section], dict):
            if key in self._data[section]:
                del self._data[section][key]
                self._dirty = True
                self._persist()
                return True
        elif not key and section in self._data:
            del self._data[section]
            self._dirty = True
            self._persist()
            return True
        return False

    # ── Convenience methods ──────────────────────────────────────

    def get_user_profile(self) -> Dict[str, Any]:
        """Get the user profile section."""
        return self.get("user_profile", default={})

    def get_active_projects(self) -> Dict[str, Any]:
        """Get all active projects."""
        return self.get("active_projects", default={})

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific project's state."""
        projects = self.get_active_projects()
        return projects.get(project_id)

    def update_project(self, project_id: str, updates: Dict[str, Any]) -> None:
        """Update a project's state (merge, not replace)."""
        projects = self.get("active_projects", default={})
        if project_id not in projects:
            projects[project_id] = {}
        projects[project_id].update(updates)
        projects[project_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.set_section("active_projects", projects)

    def add_decision(self, decision: Dict[str, Any]) -> None:
        """Add a decision to the recent decisions list (max 10)."""
        decisions = self.get("recent_decisions", default=[])
        if not isinstance(decisions, list):
            decisions = []
        decision["recorded_at"] = datetime.now(timezone.utc).isoformat()
        decisions.insert(0, decision)
        decisions = decisions[:10]  # Keep last 10
        self.set_section("recent_decisions", decisions)

    def add_action_item(self, item: Dict[str, Any]) -> None:
        """Add an action item."""
        items = self.get("action_items", default=[])
        if not isinstance(items, list):
            items = []
        item["created_at"] = datetime.now(timezone.utc).isoformat()
        item["status"] = item.get("status", "open")
        items.append(item)
        self.set_section("action_items", items)

    def complete_action_item(self, description_fragment: str) -> bool:
        """Mark an action item as complete by partial description match."""
        items = self.get("action_items", default=[])
        if not isinstance(items, list):
            return False
        frag_lower = description_fragment.lower()
        for item in items:
            if frag_lower in item.get("description", "").lower():
                item["status"] = "complete"
                item["completed_at"] = datetime.now(timezone.utc).isoformat()
                self.set_section("action_items", items)
                return True
        return False

    def update_preferences(self, preferences: List[Dict[str, Any]]) -> None:
        """Replace the preference snapshot (top N from confidence system)."""
        self.set_section("preference_snapshot", preferences[:20])

    def update_pipeline_config(self, config: Dict[str, Any]) -> None:
        """Update pipeline configuration snapshot."""
        existing = self.get("pipeline_config", default={})
        if isinstance(existing, dict):
            existing.update(config)
        else:
            existing = config
        self.set_section("pipeline_config", existing)

    # ── Bulk context for LLM injection ───────────────────────────

    def to_context_block(self) -> str:
        """Format the entire hot cache as a context block for LLM injection.

        Returns a compact text representation suitable for system prompt.
        """
        lines = ["=== ASTRA HOT CACHE (preloaded context) ===", ""]

        # User profile
        profile = self.get_user_profile()
        if profile and any(profile.values()):
            lines.append("USER:")
            if profile.get("name"):
                lines.append(f"  Name: {profile['name']}")
            if profile.get("location"):
                lines.append(f"  Location: {profile['location']}")
            if profile.get("background"):
                lines.append(f"  Background: {', '.join(profile['background'][:5])}")
            if profile.get("goals"):
                lines.append(f"  Goals: {', '.join(profile['goals'][:3])}")
            lines.append("")

        # Active projects
        projects = self.get_active_projects()
        if projects:
            lines.append("ACTIVE PROJECTS:")
            for pid, pdata in projects.items():
                status = pdata.get("status", "unknown")
                phase = pdata.get("phase", "")
                lines.append(f"  {pid}: {status}" + (f" ({phase})" if phase else ""))
                if pdata.get("blockers"):
                    lines.append(f"    Blockers: {', '.join(pdata['blockers'][:3])}")
            lines.append("")

        # Recent decisions
        decisions = self.get("recent_decisions", default=[])
        if decisions:
            lines.append("RECENT DECISIONS:")
            for d in decisions[:5]:
                lines.append(f"  - {d.get('summary', d.get('description', ''))[:80]}")
            lines.append("")

        # Action items (open only)
        items = self.get("action_items", default=[])
        open_items = [i for i in items if i.get("status") == "open"] if isinstance(items, list) else []
        if open_items:
            lines.append("OPEN ACTION ITEMS:")
            for item in open_items[:5]:
                lines.append(f"  - {item.get('description', '')[:80]}")
            lines.append("")

        return "\n".join(lines)

    # ── Query support ────────────────────────────────────────────

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Simple keyword search across all hot cache content.

        Returns list of {section, key, content, score} dicts.
        """
        query_lower = query.lower()
        keywords = [w for w in query_lower.split() if len(w) >= 3]
        if not keywords:
            return []

        results = []
        for section, section_data in self._data.items():
            if section == "metadata":
                continue
            text = json.dumps(section_data, default=str).lower()
            score = sum(1 for kw in keywords if kw in text) / len(keywords)
            if score > 0:
                results.append({
                    "section": section,
                    "content": text[:200],
                    "score": score,
                    "source": "hot_cache",
                })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]

    # ── Persistence ──────────────────────────────────────────────

    def _load(self) -> None:
        """Load from disk, or initialise with defaults."""
        path = Path(self._cache_path)
        if path.exists():
            try:
                raw = path.read_text(encoding="utf-8")
                self._data = json.loads(raw)
                logger.info(
                    "[hot_cache] Loaded from %s (%d bytes)",
                    self._cache_path, len(raw),
                )
                return
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("[hot_cache] Failed to load: %s, using defaults", e)

        self._data = json.loads(json.dumps(DEFAULT_SCHEMA))
        self._persist()
        logger.info("[hot_cache] Initialised with defaults")

    def _persist(self) -> None:
        """Write current state to disk."""
        try:
            path = Path(self._cache_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = json.dumps(self._data, indent=2, default=str)
            path.write_text(raw, encoding="utf-8")
            self._dirty = False
        except OSError as e:
            logger.error("[hot_cache] Persist failed: %s", e)

    def _update_metadata(self) -> None:
        """Update metadata section."""
        raw = json.dumps(self._data, default=str)
        self._data["metadata"] = {
            "version": self._data.get("metadata", {}).get("version", 0) + 1,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "size_bytes": len(raw),
            "item_count": self._count_items(),
        }

    def _count_items(self) -> int:
        """Count total items across all sections."""
        count = 0
        for key, val in self._data.items():
            if key == "metadata":
                continue
            if isinstance(val, dict):
                count += len(val)
            elif isinstance(val, list):
                count += len(val)
            else:
                count += 1
        return count

    @property
    def size_bytes(self) -> int:
        """Current serialised size."""
        return len(json.dumps(self._data, default=str))

    @property
    def is_over_budget(self) -> bool:
        """Check if cache exceeds target size."""
        return self.size_bytes > MAX_CACHE_BYTES

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            "size_bytes": self.size_bytes,
            "max_bytes": MAX_CACHE_BYTES,
            "utilisation_pct": round(self.size_bytes / MAX_CACHE_BYTES * 100, 1),
            "item_count": self._count_items(),
            "sections": list(self._data.keys()),
            "path": self._cache_path,
        }


# ═══════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════

_instance: Optional[HotCache] = None


def get_hot_cache() -> HotCache:
    """Get or create the HotCache singleton."""
    global _instance
    if _instance is None:
        _instance = HotCache()
    return _instance
