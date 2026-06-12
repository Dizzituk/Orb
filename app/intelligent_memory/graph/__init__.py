# FILE: app/intelligent_memory/graph/__init__.py
# Purpose: Knowledge Graph subsystem.
# Called-by: app.intelligent_memory.asset_index, app.intelligent_memory.extraction, app.intelligent_memory.retrieval_router, app.optimize.pattern_learner
# Depends-on: app.intelligent_memory.graph.graph_models, app.intelligent_memory.graph.knowledge_graph
# Last-renovated: 2026-06-11
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
