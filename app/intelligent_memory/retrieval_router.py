# FILE: app/intelligent_memory/retrieval_router.py
"""
Retrieval Router — three-layer cascading query engine.

Every query hits layers in order:
  1. Hot Cache (instant, in-memory)
  2. Knowledge Graph (fast structured lookup)
  3. Deep Store (vector search, only if needed)

The router decides whether to continue to the next layer based on
result quality. If Hot Cache returns strong matches, the graph and
vector store are never touched.

Also supports:
  - Domain-aware pre-filtering before vector search
  - Result deduplication across layers
  - Retrieval metrics for the Optimize tab

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-MEM-001.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Thresholds for cascade control
HOT_CACHE_SUFFICIENT_SCORE = 0.7  # If hot cache hit >= this, skip deeper layers
GRAPH_SUFFICIENT_SCORE = 0.6      # If graph hit >= this, skip vector search
MIN_RESULTS_TO_STOP = 3           # If we have >= this many good results, stop


@dataclass
class RetrievalResult:
    """A single result from any layer."""
    content: str
    score: float = 0.0
    source: str = ""            # "hot_cache", "knowledge_graph", "deep_store", "asset_index"
    entity_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResponse:
    """Complete response from the retrieval router."""
    results: List[RetrievalResult] = field(default_factory=list)
    query: str = ""
    layers_searched: List[str] = field(default_factory=list)
    total_latency_ms: int = 0
    hot_cache_hits: int = 0
    graph_hits: int = 0
    deep_store_hits: int = 0
    asset_hits: int = 0

    @property
    def top_result(self) -> Optional[RetrievalResult]:
        return self.results[0] if self.results else None

    def to_context_block(self, max_results: int = 5) -> str:
        """Format results as a context block for LLM injection."""
        if not self.results:
            return ""
        lines = ["=== RETRIEVED CONTEXT ===", ""]
        for r in self.results[:max_results]:
            lines.append(f"[{r.source}] (score={r.score:.2f})")
            lines.append(f"  {r.content[:300]}")
            lines.append("")
        return "\n".join(lines)


class RetrievalRouter:
    """Three-layer cascading retrieval engine.

    Queries Hot Cache → Knowledge Graph → Deep Store in order.
    Stops early if sufficient results are found.
    """

    def __init__(self):
        self._hot_cache = None
        self._knowledge_graph = None
        self._asset_index = None
        self._metrics: Dict[str, Any] = {
            "total_queries": 0,
            "hot_cache_only": 0,
            "graph_stopped": 0,
            "full_cascade": 0,
            "avg_latency_ms": 0,
        }

    def _ensure_components(self) -> None:
        """Lazy-load components on first use."""
        if self._hot_cache is None:
            from app.intelligent_memory.hot_cache import get_hot_cache
            self._hot_cache = get_hot_cache()
        if self._knowledge_graph is None:
            from app.intelligent_memory.graph import get_knowledge_graph
            self._knowledge_graph = get_knowledge_graph()
        if self._asset_index is None:
            from app.intelligent_memory.asset_index import get_asset_index
            self._asset_index = get_asset_index()

    def query(
        self,
        text: str,
        limit: int = 10,
        include_assets: bool = True,
        force_deep: bool = False,
        domains: Optional[List[str]] = None,
    ) -> RetrievalResponse:
        """Run a cascading query across all memory layers.

        Args:
            text: The query text.
            limit: Max results to return.
            include_assets: Whether to search the asset index.
            force_deep: Force vector search even if hot/graph suffice.
            domains: Domain filter for deep store.

        Returns:
            RetrievalResponse with merged, deduplicated results.
        """
        self._ensure_components()
        t_start = time.time()
        response = RetrievalResponse(query=text)
        all_results: List[RetrievalResult] = []

        # ── Layer 1: Hot Cache ──
        hot_results = self._search_hot_cache(text, limit)
        all_results.extend(hot_results)
        response.hot_cache_hits = len(hot_results)
        response.layers_searched.append("hot_cache")

        best_score = max((r.score for r in hot_results), default=0)
        if (
            not force_deep
            and best_score >= HOT_CACHE_SUFFICIENT_SCORE
            and len(hot_results) >= MIN_RESULTS_TO_STOP
        ):
            self._metrics["hot_cache_only"] += 1
            return self._finalise(response, all_results, limit, t_start)

        # ── Layer 2: Knowledge Graph ──
        graph_results = self._search_graph(text, limit)
        all_results.extend(graph_results)
        response.graph_hits = len(graph_results)
        response.layers_searched.append("knowledge_graph")

        # Asset index (part of graph layer conceptually)
        if include_assets:
            asset_results = self._search_assets(text, limit)
            all_results.extend(asset_results)
            response.asset_hits = len(asset_results)

        best_score = max((r.score for r in all_results), default=0)
        total_good = sum(1 for r in all_results if r.score >= 0.4)
        if (
            not force_deep
            and best_score >= GRAPH_SUFFICIENT_SCORE
            and total_good >= MIN_RESULTS_TO_STOP
        ):
            self._metrics["graph_stopped"] += 1
            return self._finalise(response, all_results, limit, t_start)

        # ── Layer 3: Deep Store (existing RAG) ──
        deep_results = self._search_deep_store(text, limit, domains)
        all_results.extend(deep_results)
        response.deep_store_hits = len(deep_results)
        response.layers_searched.append("deep_store")
        self._metrics["full_cascade"] += 1

        return self._finalise(response, all_results, limit, t_start)

    # ── Layer implementations ────────────────────────────────────

    def _search_hot_cache(self, text: str, limit: int) -> List[RetrievalResult]:
        """Search the hot cache."""
        try:
            raw = self._hot_cache.search(text, limit=limit)
            return [
                RetrievalResult(
                    content=r.get("content", ""),
                    score=r.get("score", 0),
                    source="hot_cache",
                    metadata={"section": r.get("section", "")},
                )
                for r in raw
            ]
        except Exception as e:
            logger.warning("[retrieval] Hot cache search failed: %s", e)
            return []

    def _search_graph(self, text: str, limit: int) -> List[RetrievalResult]:
        """Search the knowledge graph."""
        try:
            raw = self._knowledge_graph.search(text, limit=limit)
            return [
                RetrievalResult(
                    content=f"{r.get('name', '')} ({r.get('entity_type', '')})",
                    score=r.get("score", 0),
                    source="knowledge_graph",
                    entity_id=r.get("entity_id", ""),
                    metadata=r,
                )
                for r in raw
            ]
        except Exception as e:
            logger.warning("[retrieval] Graph search failed: %s", e)
            return []

    def _search_assets(self, text: str, limit: int) -> List[RetrievalResult]:
        """Search the asset index."""
        try:
            raw = self._asset_index.search(text, limit=limit)
            return [
                RetrievalResult(
                    content=f"{r.get('name', '')}: {r.get('description', '')}",
                    score=r.get("score", 0),
                    source="asset_index",
                    entity_id=r.get("asset_id", ""),
                    metadata=r,
                )
                for r in raw
            ]
        except Exception as e:
            logger.warning("[retrieval] Asset search failed: %s", e)
            return []

    def _search_deep_store(
        self,
        text: str,
        limit: int,
        domains: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """Search the existing RAG/vector store (Deep Store).

        Bridges into the existing memory_router for backward compatibility.
        """
        try:
            from app.memory.router import memory_router
            raw = memory_router.query(
                text=text,
                domains=domains,
                limit=limit,
            )
            return [
                RetrievalResult(
                    content=r.content[:300],
                    score=r.relevance,
                    source="deep_store",
                    metadata={
                        "domain": r.domain,
                        "source_table": r.source_table,
                    },
                )
                for r in raw
            ]
        except Exception as e:
            logger.warning("[retrieval] Deep store search failed: %s", e)
            return []

    # ── Helpers ──────────────────────────────────────────────────

    def _finalise(
        self,
        response: RetrievalResponse,
        results: List[RetrievalResult],
        limit: int,
        t_start: float,
    ) -> RetrievalResponse:
        """Deduplicate, sort, trim, and attach metrics."""
        # Deduplicate by content prefix (first 100 chars)
        seen = set()
        unique = []
        for r in results:
            key = r.content[:100].lower()
            if key not in seen:
                seen.add(key)
                unique.append(r)

        unique.sort(key=lambda r: r.score, reverse=True)
        response.results = unique[:limit]
        response.total_latency_ms = int((time.time() - t_start) * 1000)

        # Update metrics
        self._metrics["total_queries"] += 1
        total = self._metrics["total_queries"]
        prev_avg = self._metrics["avg_latency_ms"]
        self._metrics["avg_latency_ms"] = (
            prev_avg + (response.total_latency_ms - prev_avg) / total
        )

        return response

    def get_metrics(self) -> Dict[str, Any]:
        """Return retrieval metrics for the Optimize tab."""
        total = self._metrics["total_queries"] or 1
        return {
            **self._metrics,
            "hot_cache_only_pct": round(
                self._metrics["hot_cache_only"] / total * 100, 1
            ),
            "graph_stopped_pct": round(
                self._metrics["graph_stopped"] / total * 100, 1
            ),
            "full_cascade_pct": round(
                self._metrics["full_cascade"] / total * 100, 1
            ),
        }


# ═══════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════

_instance: Optional[RetrievalRouter] = None


def get_retrieval_router() -> RetrievalRouter:
    """Get or create the RetrievalRouter singleton."""
    global _instance
    if _instance is None:
        _instance = RetrievalRouter()
    return _instance
