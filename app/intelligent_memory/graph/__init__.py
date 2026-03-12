# FILE: app/intelligent_memory/graph/__init__.py
"""
Knowledge Graph subsystem.

NetworkX-based in-process graph for structured entity relationships.
Serialised to JSON on disk. Migrates to Neo4j if needed later.
"""
from app.intelligent_memory.graph.graph_models import (
    EntityType,
    RelationType,
    Entity,
    Relationship,
)
from app.intelligent_memory.graph.knowledge_graph import (
    KnowledgeGraph,
    get_knowledge_graph,
)

__all__ = [
    "EntityType",
    "RelationType",
    "Entity",
    "Relationship",
    "KnowledgeGraph",
    "get_knowledge_graph",
]
