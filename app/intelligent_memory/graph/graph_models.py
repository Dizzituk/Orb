# FILE: app/intelligent_memory/graph/graph_models.py
# Purpose: Knowledge Graph entity and relationship type definitions.
# Called-by: app.intelligent_memory.graph, app.intelligent_memory.graph.knowledge_graph
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Knowledge Graph entity and relationship type definitions.

These are the structured types that live as nodes and edges in the
NetworkX graph. Each entity has a type, unique ID, and attribute dict.
Each relationship has a type, source, target, and metadata.

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-MEM-001.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════
# Entity types (graph nodes)
# ═══════════════════════════════════════════════════════════════════

class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""
    PROJECT = "project"
    FILE = "file"
    DECISION = "decision"
    PATTERN = "pattern"
    ASSET = "asset"
    TOPIC = "topic"
    PERSON = "person"
    CONVERSATION = "conversation"
    TECHNOLOGY = "technology"
    SPEC = "spec"
    PIPELINE_STAGE = "pipeline_stage"


# ═══════════════════════════════════════════════════════════════════
# Relationship types (graph edges)
# ═══════════════════════════════════════════════════════════════════

class RelationType(str, Enum):
    """Types of relationships between entities."""
    DEPENDS_ON = "depends_on"
    PRODUCED_BY = "produced_by"
    RELATED_TO = "related_to"
    SUPERSEDES = "supersedes"
    BLOCKED_BY = "blocked_by"
    PART_OF = "part_of"
    USES = "uses"
    DISCOVERED_IN = "discovered_in"
    CREATED_DURING = "created_during"
    IMPLEMENTS = "implements"
    REFERENCES = "references"


# ═══════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Entity:
    """A node in the knowledge graph."""
    entity_id: str              # Unique ID (e.g. "project:driver-copilot")
    entity_type: EntityType
    name: str                   # Human-readable name
    attributes: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "name": self.name,
            "attributes": self.attributes,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(
            entity_id=data["entity_id"],
            entity_type=EntityType(data["entity_type"]),
            name=data["name"],
            attributes=data.get("attributes", {}),
            tags=data.get("tags", []),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class Relationship:
    """An edge in the knowledge graph."""
    source_id: str              # Entity ID of source node
    target_id: str              # Entity ID of target node
    relation_type: RelationType
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "metadata": self.metadata,
            "weight": self.weight,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            metadata=data.get("metadata", {}),
            weight=data.get("weight", 1.0),
            created_at=data.get("created_at", ""),
        )


@dataclass
class GraphQueryResult:
    """Result from a knowledge graph query."""
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    query: str = ""
    traversal_depth: int = 0
    source: str = "knowledge_graph"
