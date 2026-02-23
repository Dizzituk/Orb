# FILE: app/memory/router.py
"""
MemoryRouter — single entry point for all memory operations.

This is the programmatic interface that every part of ASTRA imports
when it needs to read or write memory. It delegates to domain-specific
stores that each handle their own backing table.

NOT the HTTP API router — that's app/memory/api_router.py.

Usage:
    from app.memory.router import memory_router

    results = memory_router.query("weaver streaming", project_id="astra-core")
    entry_id = memory_router.store("knowledge", "Weaver handles chat streaming", {})
"""

import logging
from typing import Optional, Protocol, runtime_checkable

from app.memory.schemas_unified import (
    MemoryResult,
    StoreRequest,
    QueryRequest,
    DomainStats,
)

logger = logging.getLogger(__name__)


# =========================================================================
# Domain Store Protocol
# =========================================================================

@runtime_checkable
class DomainStore(Protocol):
    """
    Protocol that all domain stores must implement.

    Each domain (architecture, knowledge, preference, confidence, context)
    provides a store class that implements these methods.
    """

    @property
    def domain_name(self) -> str:
        """The domain this store handles (e.g. 'architecture')."""
        ...

    def query(
        self,
        text: str,
        project_id: str = "astra-core",
        limit: int = 10,
        min_relevance: float = 0.0,
    ) -> list[MemoryResult]:
        """Search this domain for relevant entries."""
        ...

    def store(self, request: StoreRequest) -> int:
        """Store a new entry. Returns the entry ID."""
        ...

    def quarantine(self, file_path: str, project_id: str = "astra-core") -> int:
        """Quarantine entries for a file path. Returns count affected."""
        ...

    def count(self, project_id: str = "astra-core") -> int:
        """Count active entries for a project."""
        ...

    def get_stats(self, project_id: str = "astra-core") -> DomainStats:
        """Get statistics for this domain."""
        ...


# =========================================================================
# MemoryRouter
# =========================================================================

class MemoryRouter:
    """
    Single entry point for all memory operations.

    Domain stores register themselves. Query/store calls are delegated
    to the appropriate store based on the domain parameter.
    """

    def __init__(self):
        self._stores: dict[str, DomainStore] = {}

    def register(self, store: DomainStore) -> None:
        """Register a domain store."""
        name = store.domain_name
        if name in self._stores:
            logger.warning(f"[memory_router] Overwriting store for domain: {name}")
        self._stores[name] = store
        logger.info(f"[memory_router] Registered domain store: {name}")

    @property
    def registered_domains(self) -> list[str]:
        """List all registered domain names."""
        return list(self._stores.keys())

    # -----------------------------------------------------------------
    # Query
    # -----------------------------------------------------------------

    def query(
        self,
        text: str,
        project_id: str = "astra-core",
        domains: Optional[list[str]] = None,
        limit: int = 10,
        min_relevance: float = 0.0,
    ) -> list[MemoryResult]:
        """
        Query across one or more memory domains.

        If domains is None, queries all registered domains.
        Results are merged and sorted by relevance (highest first).
        """
        target_domains = domains or list(self._stores.keys())
        all_results: list[MemoryResult] = []

        for domain in target_domains:
            store = self._stores.get(domain)
            if not store:
                logger.warning(f"[memory_router] No store for domain: {domain}")
                continue
            try:
                results = store.query(
                    text=text,
                    project_id=project_id,
                    limit=limit,
                    min_relevance=min_relevance,
                )
                all_results.extend(results)
            except Exception as e:
                logger.error(f"[memory_router] Query failed for domain {domain}: {e}")

        # Sort by relevance descending, take top N
        all_results.sort(key=lambda r: r.relevance, reverse=True)
        return all_results[:limit]

    # -----------------------------------------------------------------
    # Store
    # -----------------------------------------------------------------

    def store(
        self,
        domain: str,
        content: str,
        metadata: Optional[dict] = None,
        project_id: str = "astra-core",
        file_path: Optional[str] = None,
    ) -> int:
        """
        Store a new entry in the specified domain.

        Returns the entry ID from the backing store.
        """
        store = self._stores.get(domain)
        if not store:
            raise ValueError(f"No store registered for domain: {domain}")

        request = StoreRequest(
            domain=domain,
            content=content,
            metadata=metadata or {},
            project_id=project_id,
            file_path=file_path,
        )
        return store.store(request)

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def quarantine(
        self,
        file_path: str,
        project_id: str = "astra-core",
    ) -> int:
        """
        Quarantine entries across all domains for a file path.

        Returns total count of entries quarantined.
        """
        total = 0
        for name, store in self._stores.items():
            try:
                count = store.quarantine(file_path, project_id)
                total += count
            except Exception as e:
                logger.error(f"[memory_router] Quarantine failed for {name}: {e}")
        return total

    def purge_quarantined(self, project_id: str = "astra-core") -> int:
        """
        Purge all quarantined entries across all domains.

        Returns total count of entries purged.
        """
        total = 0
        for name, store in self._stores.items():
            try:
                if hasattr(store, "purge_quarantined"):
                    count = store.purge_quarantined(project_id)
                    total += count
            except Exception as e:
                logger.error(f"[memory_router] Purge failed for {name}: {e}")
        return total

    def wipe_domain(self, domain: str, project_id: str = "astra-core") -> int:
        """
        Delete all entries in a domain for a project.

        Returns count of entries deleted.
        """
        store = self._stores.get(domain)
        if not store:
            raise ValueError(f"No store registered for domain: {domain}")

        if hasattr(store, "wipe"):
            return store.wipe(project_id)

        raise NotImplementedError(f"Domain {domain} does not support wipe")

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------

    def get_stats(self) -> dict[str, DomainStats]:
        """Get statistics for all registered domains."""
        stats = {}
        for name, store in self._stores.items():
            try:
                stats[name] = store.get_stats()
            except Exception as e:
                logger.error(f"[memory_router] Stats failed for {name}: {e}")
                stats[name] = DomainStats(domain=name)
        return stats


# =========================================================================
# Module-level singleton
# =========================================================================

memory_router = MemoryRouter()
"""
Module-level singleton instance.

Import this wherever you need memory access:
    from app.memory.router import memory_router
"""
