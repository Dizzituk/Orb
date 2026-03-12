# FILE: app/intelligent_memory/asset_index.py
"""
Asset Index — tracks every piece of content ASTRA has created,
downloaded, processed, or been given.

Prevents redundant work and enables intelligent reuse. When a new
task arrives, the asset index is queried before generation begins.

Asset types: code, document, spec, dataset, media, template, config.

Each asset is also registered as an Entity in the Knowledge Graph
with PRODUCED_BY and RELATED_TO relationships.

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-MEM-001.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ASSET_INDEX_PATH = os.getenv(
    "ASTRA_ASSET_INDEX_PATH",
    os.path.join("D:", os.sep, "Orb", "data", "asset_index.json"),
)


class AssetType:
    CODE = "code"
    DOCUMENT = "document"
    SPEC = "spec"
    DATASET = "dataset"
    MEDIA = "media"
    TEMPLATE = "template"
    CONFIG = "config"
    PRESENTATION = "presentation"
    REPORT = "report"


class AssetEntry:
    """A single indexed asset."""

    def __init__(
        self,
        asset_id: str,
        asset_type: str,
        name: str,
        path: str = "",
        description: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: str = "",
        project_id: str = "",
        content_hash: str = "",
    ):
        self.asset_id = asset_id
        self.asset_type = asset_type
        self.name = name
        self.path = path
        self.description = description
        self.tags = tags or []
        self.metadata = metadata or {}
        self.created_by = created_by
        self.project_id = project_id
        self.content_hash = content_hash
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.last_used_at = self.created_at
        self.use_count = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "asset_type": self.asset_type,
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_by": self.created_by,
            "project_id": self.project_id,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "use_count": self.use_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssetEntry":
        entry = cls(
            asset_id=data["asset_id"],
            asset_type=data.get("asset_type", ""),
            name=data.get("name", ""),
            path=data.get("path", ""),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            created_by=data.get("created_by", ""),
            project_id=data.get("project_id", ""),
            content_hash=data.get("content_hash", ""),
        )
        entry.created_at = data.get("created_at", entry.created_at)
        entry.last_used_at = data.get("last_used_at", entry.last_used_at)
        entry.use_count = data.get("use_count", 0)
        return entry


class AssetIndex:
    """Tracks all assets ASTRA has created or been given.

    Enables reuse detection: before generating new content, query
    the index for existing assets that match or overlap.
    """

    def __init__(self, index_path: str = ASSET_INDEX_PATH):
        self._index_path = index_path
        self._assets: Dict[str, AssetEntry] = {}
        self._load()

    # ── Registration ─────────────────────────────────────────────

    def register(self, entry: AssetEntry) -> str:
        """Register a new asset. Returns the asset_id."""
        self._assets[entry.asset_id] = entry
        self._persist()

        # Register in knowledge graph
        self._register_in_graph(entry)

        logger.info("[asset_index] Registered: %s (%s)", entry.name, entry.asset_type)
        return entry.asset_id

    def register_file(
        self,
        path: str,
        asset_type: str = AssetType.CODE,
        name: str = "",
        description: str = "",
        project_id: str = "",
        tags: Optional[List[str]] = None,
        created_by: str = "pipeline",
    ) -> str:
        """Convenience: register a file-based asset."""
        p = Path(path)
        asset_id = f"asset:{p.stem}:{_hash_short(path)}"
        content_hash = ""
        if p.exists():
            content_hash = hashlib.sha256(
                p.read_bytes()
            ).hexdigest()[:16]

        entry = AssetEntry(
            asset_id=asset_id,
            asset_type=asset_type,
            name=name or p.name,
            path=str(p),
            description=description,
            tags=tags or [],
            created_by=created_by,
            project_id=project_id,
            content_hash=content_hash,
        )
        return self.register(entry)

    # ── Lookup ───────────────────────────────────────────────────

    def get(self, asset_id: str) -> Optional[AssetEntry]:
        """Get an asset by ID."""
        return self._assets.get(asset_id)

    def find(
        self,
        asset_type: str = "",
        project_id: str = "",
        tags: Optional[List[str]] = None,
        name_contains: str = "",
        limit: int = 20,
    ) -> List[AssetEntry]:
        """Find assets by criteria."""
        results = []
        for entry in self._assets.values():
            if asset_type and entry.asset_type != asset_type:
                continue
            if project_id and entry.project_id != project_id:
                continue
            if tags and not set(tags).intersection(set(entry.tags)):
                continue
            if name_contains and name_contains.lower() not in entry.name.lower():
                continue
            results.append(entry)

        results.sort(key=lambda e: e.last_used_at, reverse=True)
        return results[:limit]

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Keyword search for retrieval router compatibility."""
        query_lower = query.lower()
        keywords = [w for w in query_lower.split() if len(w) >= 3]
        if not keywords:
            return []

        scored = []
        for entry in self._assets.values():
            searchable = " ".join([
                entry.name, entry.description,
                " ".join(entry.tags), entry.asset_type,
            ]).lower()
            hits = sum(1 for kw in keywords if kw in searchable)
            if hits > 0:
                scored.append({
                    "asset_id": entry.asset_id,
                    "name": entry.name,
                    "asset_type": entry.asset_type,
                    "description": entry.description[:100],
                    "score": hits / len(keywords),
                    "source": "asset_index",
                })

        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:limit]

    def check_reuse(
        self,
        description: str,
        asset_type: str = "",
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Check if existing assets can be reused for a task.

        Returns scored matches: exact, adaptable, or partial.
        """
        candidates = self.search(description, limit=10)

        for c in candidates:
            score = c["score"]
            if score >= 0.8:
                c["reuse_level"] = "exact"
            elif score >= 0.5:
                c["reuse_level"] = "adaptable"
            else:
                c["reuse_level"] = "partial"

        return [c for c in candidates if c["score"] >= 0.3]

    def record_use(self, asset_id: str) -> None:
        """Record that an asset was used/referenced."""
        entry = self._assets.get(asset_id)
        if entry:
            entry.use_count += 1
            entry.last_used_at = datetime.now(timezone.utc).isoformat()
            self._persist()

    # ── Stats ────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        type_counts: Dict[str, int] = {}
        for entry in self._assets.values():
            type_counts[entry.asset_type] = type_counts.get(entry.asset_type, 0) + 1
        return {
            "total_assets": len(self._assets),
            "asset_types": type_counts,
            "path": self._index_path,
        }

    # ── Graph integration ────────────────────────────────────────

    def _register_in_graph(self, entry: AssetEntry) -> None:
        """Register the asset as an entity in the knowledge graph."""
        try:
            from app.intelligent_memory.graph import (
                get_knowledge_graph, Entity, EntityType,
                Relationship, RelationType,
            )
            graph = get_knowledge_graph()

            entity = Entity(
                entity_id=entry.asset_id,
                entity_type=EntityType.ASSET,
                name=entry.name,
                attributes={
                    "asset_type": entry.asset_type,
                    "path": entry.path,
                    "description": entry.description,
                    "content_hash": entry.content_hash,
                },
                tags=entry.tags,
            )
            graph.add_entity(entity)

            if entry.project_id:
                graph.add_relationship(Relationship(
                    source_id=entry.asset_id,
                    target_id=f"project:{entry.project_id}",
                    relation_type=RelationType.PART_OF,
                ))
        except Exception as e:
            logger.debug("[asset_index] Graph registration failed: %s", e)

    # ── Persistence ──────────────────────────────────────────────

    def _load(self) -> None:
        path = Path(self._index_path)
        if not path.exists():
            logger.info("[asset_index] No existing index, starting empty")
            return
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            for item in data.get("assets", []):
                entry = AssetEntry.from_dict(item)
                self._assets[entry.asset_id] = entry
            logger.info("[asset_index] Loaded %d assets", len(self._assets))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[asset_index] Failed to load: %s", e)

    def _persist(self) -> None:
        try:
            path = Path(self._index_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            output = {
                "assets": [e.to_dict() for e in self._assets.values()],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            path.write_text(
                json.dumps(output, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error("[asset_index] Persist failed: %s", e)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _hash_short(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


_instance: Optional[AssetIndex] = None

def get_asset_index() -> AssetIndex:
    global _instance
    if _instance is None:
        _instance = AssetIndex()
    return _instance
