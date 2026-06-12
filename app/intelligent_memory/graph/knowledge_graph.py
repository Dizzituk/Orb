# FILE: app/intelligent_memory/graph/knowledge_graph.py
# Purpose: Layer 2 — Knowledge Graph.
# Called-by: app.intelligent_memory.graph
# Depends-on: app.intelligent_memory.graph.graph_models
# Last-renovated: 2026-06-11
"""
Layer 2 — Knowledge Graph.

NetworkX-based in-process graph for structured entity relationships.
Finds logically connected information regardless of text similarity.

Supports queries that vector search cannot:
  - "What assets do we have for AI economics?" → traverse topic→asset
  - "What decisions were superseded?" → follow SUPERSEDES chains
  - "What's blocking Driver CoPilot?" → reverse BLOCKED_BY traversal
  - "What depends on BVL?" → reverse DEPENDS_ON

Serialised to JSON on disk. Zero infrastructure dependency.

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-MEM-001.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Persistence path
GRAPH_PATH = os.getenv(
    "ASTRA_GRAPH_PATH",
    os.path.join("D:", os.sep, "Orb", "data", "knowledge_graph.json"),
)

# Lazy import NetworkX — installed via requirements.txt
_nx = None

def _get_nx():
    global _nx
    if _nx is None:
        import networkx as nx
        _nx = nx
    return _nx


from app.intelligent_memory.graph.graph_models import (
    Entity,
    EntityType,
    GraphQueryResult,
    Relationship,
    RelationType,
)


class KnowledgeGraph:
    """In-process knowledge graph backed by NetworkX.

    Nodes are Entity objects. Edges are Relationship objects.
    The graph is directed (relationships have direction).
    """

    def __init__(self, graph_path: str = GRAPH_PATH):
        self._graph_path = graph_path
        nx = _get_nx()
        self._graph: Any = nx.DiGraph()
        self._load()

    # ── Entity CRUD ──────────────────────────────────────────────

    def add_entity(self, entity: Entity) -> None:
        """Add or update an entity node."""
        self._graph.add_node(
            entity.entity_id,
            **entity.to_dict(),
        )
        self._persist()

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        if entity_id not in self._graph:
            return None
        data = dict(self._graph.nodes[entity_id])
        try:
            return Entity.from_dict(data)
        except (KeyError, ValueError):
            return None

    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity and all its relationships."""
        if entity_id not in self._graph:
            return False
        self._graph.remove_node(entity_id)
        self._persist()
        return True

    def update_entity(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """Update attributes on an existing entity."""
        if entity_id not in self._graph:
            return False
        node_data = self._graph.nodes[entity_id]
        attrs = node_data.get("attributes", {})
        attrs.update(updates)
        node_data["attributes"] = attrs
        node_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._persist()
        return True

    def find_entities(
        self,
        entity_type: Optional[EntityType] = None,
        tags: Optional[List[str]] = None,
        name_contains: str = "",
    ) -> List[Entity]:
        """Find entities by type, tags, or name substring."""
        results = []
        for node_id, data in self._graph.nodes(data=True):
            if entity_type and data.get("entity_type") != entity_type.value:
                continue
            if tags:
                node_tags = set(data.get("tags", []))
                if not node_tags.intersection(set(tags)):
                    continue
            if name_contains:
                name = data.get("name", "")
                if name_contains.lower() not in name.lower():
                    continue
            try:
                results.append(Entity.from_dict(data))
            except (KeyError, ValueError):
                continue
        return results

    # ── Relationship CRUD ────────────────────────────────────────

    def add_relationship(self, rel: Relationship) -> None:
        """Add a relationship edge between two entities.

        Creates target/source nodes if they don't exist (as stubs).
        """
        for eid in (rel.source_id, rel.target_id):
            if eid not in self._graph:
                self._graph.add_node(eid, entity_id=eid, name=eid,
                                      entity_type="unknown", attributes={},
                                      tags=[], created_at="", updated_at="")

        self._graph.add_edge(
            rel.source_id,
            rel.target_id,
            **rel.to_dict(),
        )
        self._persist()

    def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        relation_type: Optional[RelationType] = None,
    ) -> List[Relationship]:
        """Get relationships for an entity.

        Args:
            entity_id: The entity to query.
            direction: "outgoing", "incoming", or "both".
            relation_type: Filter by relationship type.
        """
        if entity_id not in self._graph:
            return []

        edges = []
        if direction in ("outgoing", "both"):
            for _, target, data in self._graph.out_edges(entity_id, data=True):
                edges.append(data)
        if direction in ("incoming", "both"):
            for source, _, data in self._graph.in_edges(entity_id, data=True):
                edges.append(data)

        results = []
        for edge_data in edges:
            try:
                rel = Relationship.from_dict(edge_data)
                if relation_type and rel.relation_type != relation_type:
                    continue
                results.append(rel)
            except (KeyError, ValueError):
                continue
        return results

    def remove_relationship(
        self,
        source_id: str,
        target_id: str,
    ) -> bool:
        """Remove a specific relationship."""
        if self._graph.has_edge(source_id, target_id):
            self._graph.remove_edge(source_id, target_id)
            self._persist()
            return True
        return False

    # ── Graph queries ────────────────────────────────────────────

    def traverse(
        self,
        start_id: str,
        relation_types: Optional[List[RelationType]] = None,
        max_depth: int = 2,
        direction: str = "outgoing",
    ) -> GraphQueryResult:
        """Traverse the graph from a starting entity.

        Follows edges up to max_depth hops, optionally filtering
        by relationship type. Returns all discovered entities and
        relationships.
        """
        if start_id not in self._graph:
            return GraphQueryResult(query=f"traverse:{start_id}")

        visited: Set[str] = set()
        entities: List[Entity] = []
        relationships: List[Relationship] = []
        frontier = {start_id}

        for depth in range(max_depth):
            next_frontier: Set[str] = set()
            for node_id in frontier:
                if node_id in visited:
                    continue
                visited.add(node_id)

                entity = self.get_entity(node_id)
                if entity:
                    entities.append(entity)

                rels = self.get_relationships(
                    node_id, direction=direction,
                    relation_type=None,
                )
                for rel in rels:
                    if relation_types and rel.relation_type not in relation_types:
                        continue
                    relationships.append(rel)
                    next_id = (
                        rel.target_id if direction != "incoming"
                        else rel.source_id
                    )
                    if next_id not in visited:
                        next_frontier.add(next_id)

            frontier = next_frontier
            if not frontier:
                break

        return GraphQueryResult(
            entities=entities,
            relationships=relationships,
            query=f"traverse:{start_id}",
            traversal_depth=max_depth,
        )

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Keyword search across all entity names and attributes.

        Returns results compatible with the retrieval router format.
        """
        query_lower = query.lower()
        keywords = [w for w in query_lower.split() if len(w) >= 3]
        if not keywords:
            return []

        scored = []
        for node_id, data in self._graph.nodes(data=True):
            searchable = json.dumps(data, default=str).lower()
            score = sum(1 for kw in keywords if kw in searchable)
            if score > 0:
                scored.append({
                    "entity_id": node_id,
                    "name": data.get("name", ""),
                    "entity_type": data.get("entity_type", ""),
                    "score": score / len(keywords),
                    "source": "knowledge_graph",
                })

        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:limit]

    def find_path(
        self,
        source_id: str,
        target_id: str,
    ) -> Optional[List[str]]:
        """Find shortest path between two entities."""
        nx = _get_nx()
        try:
            path = nx.shortest_path(
                self._graph, source_id, target_id,
            )
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    # ── Stats ────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return graph statistics."""
        type_counts: Dict[str, int] = {}
        for _, data in self._graph.nodes(data=True):
            etype = data.get("entity_type", "unknown")
            type_counts[etype] = type_counts.get(etype, 0) + 1

        return {
            "total_entities": self._graph.number_of_nodes(),
            "total_relationships": self._graph.number_of_edges(),
            "entity_types": type_counts,
            "path": self._graph_path,
        }

    # ── Persistence ──────────────────────────────────────────────

    def _load(self) -> None:
        """Load graph from JSON on disk."""
        path = Path(self._graph_path)
        if not path.exists():
            logger.info("[knowledge_graph] No existing graph, starting empty")
            return

        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)

            for node_data in data.get("nodes", []):
                nid = node_data.get("entity_id", "")
                if nid:
                    self._graph.add_node(nid, **node_data)

            for edge_data in data.get("edges", []):
                src = edge_data.get("source_id", "")
                tgt = edge_data.get("target_id", "")
                if src and tgt:
                    self._graph.add_edge(src, tgt, **edge_data)

            logger.info(
                "[knowledge_graph] Loaded: %d entities, %d relationships",
                self._graph.number_of_nodes(),
                self._graph.number_of_edges(),
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[knowledge_graph] Failed to load: %s", e)

    def _persist(self) -> None:
        """Save graph to JSON on disk."""
        try:
            nodes = []
            for node_id, data in self._graph.nodes(data=True):
                nodes.append(dict(data))

            edges = []
            for src, tgt, data in self._graph.edges(data=True):
                edges.append(dict(data))

            output = {"nodes": nodes, "edges": edges}
            path = Path(self._graph_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(output, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error("[knowledge_graph] Persist failed: %s", e)


# ═══════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════

_instance: Optional[KnowledgeGraph] = None


def get_knowledge_graph() -> KnowledgeGraph:
    """Get or create the KnowledgeGraph singleton."""
    global _instance
    if _instance is None:
        _instance = KnowledgeGraph()
    return _instance
