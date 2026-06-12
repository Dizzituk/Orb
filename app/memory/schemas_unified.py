# FILE: app/memory/schemas_unified.py
# Purpose: Shared data classes for the unified memory system.
# Called-by: app.memory.domains.architecture, app.memory.domains.confidence, app.memory.domains.context, app.memory.domains.knowledge (+3 more)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Shared data classes for the unified memory system.

These are used by MemoryRouter, all domain stores, and UnifiedRAGQuery
to pass data in a normalised format regardless of which backing table
the data comes from.

Not Pydantic models — plain dataclasses for internal use.
The HTTP layer (api_router.py) has its own Pydantic schemas.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryResult:
    """
    A single result from any memory domain.

    Returned by MemoryRouter.query() and domain store query methods.
    The source_table field tells you where the data actually lives.
    """
    id: int
    domain: str
    content: str
    project_id: str
    relevance: float = 0.0
    file_path: Optional[str] = None
    source_table: str = ""          # 'arch_code_chunks', 'rag_entries', etc.
    status: str = "active"
    metadata: dict = field(default_factory=dict)
    # metadata examples:
    #   architecture: symbol_name, symbol_type, qualified_name, package_role
    #   knowledge: ingest_source
    #   decision: rationale, alternatives
    #   context: ttl, extracted_from


@dataclass
class StoreRequest:
    """
    Request to store a new memory entry.

    Passed to MemoryRouter.store() which delegates to the appropriate
    domain store.
    """
    domain: str
    content: str
    metadata: dict = field(default_factory=dict)
    project_id: str = "astra-core"
    file_path: Optional[str] = None


@dataclass
class QueryRequest:
    """
    Request to query memory.

    Passed to MemoryRouter.query(). If domains is None, searches all
    registered domains.
    """
    text: str
    project_id: str = "astra-core"
    domains: Optional[list[str]] = None
    limit: int = 10
    min_relevance: float = 0.0


@dataclass
class DomainStats:
    """
    Statistics for a single memory domain.

    Returned by domain store count/stats methods.
    """
    domain: str
    total_entries: int = 0
    active_entries: int = 0
    quarantined_entries: int = 0
    embedded_entries: int = 0
