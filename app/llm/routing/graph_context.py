# FILE: app/llm/routing/graph_context.py
# Purpose: Bounded knowledge-graph walk -> cross-domain context block (Job 4 task 2).
# Called-by: app.llm.routing.memory_injection
# Depends-on: app.intelligent_memory.graph, app.astra_memory.topic_tagger
# Last-renovated: 2026-06-15
"""
Cross-domain graph context for the memory front door (MEMORY_JOBS_SPEC Job 4, task 2).

The embedding/tag channels recall facts that are SIMILAR to the query. They
cannot represent a RELATIONSHIP the user drew between two different domains —
"investments fund the Portugal move", "the legal claim pays for it". Job 3's
consolidation membrane writes those as explicit, strength-weighted edges in the
knowledge graph (consolidation_graph.py); this module reads them back at
retrieval time.

When the topics in play — the query's topic tags, plus the tags carried by the
records retrieval just surfaced — land on a graph node that sits on a strong
edge, we pull the connected node: one or two hops, strength-thresholded, hard
small cap. The connection is surfaced as a plain context line the chat model
can weave into its reply. Bounded by design: a full walk would re-introduce
exactly the context pollution the depth gate exists to prevent.

This runs INSIDE the front door, behind the D0-D4 gate (memory_injection only
calls it for D1+). Kill-switch ASTRA_GRAPH_WALK=off. Never raises into the
caller — every failure degrades to an empty block.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Only traverse edges at least this strong. Edges start at weight 1.0 (a single
# vetted cross-domain connection) and strengthen +1 each time the link recurs
# (consolidation_graph). Default 1.0 surfaces vetted one-offs too; raise
# ASTRA_GRAPH_WALK_MIN_STRENGTH so only recurring (load-bearing) links surface
# and one-offs fade.
_MIN_STRENGTH_DEFAULT = 1.0
_MAX_HOPS_DEFAULT = 2
_MAX_LINKS_DEFAULT = 3  # small cap — three connections, not a subgraph dump

# Consolidation writes topic nodes as "topic:<domain>" (consolidation_graph._node_for).
_NODE_PREFIX = "topic:"


def _enabled() -> bool:
    return os.getenv("ASTRA_GRAPH_WALK", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _f(env: str, default: float) -> float:
    try:
        return float(os.getenv(env, str(default)))
    except (TypeError, ValueError):
        return default


def _i(env: str, default: int) -> int:
    try:
        return int(os.getenv(env, str(default)))
    except (TypeError, ValueError):
        return default


def _anchor_domains(query_tags, records, max_record_scan: int = 5) -> List[str]:
    """Topic domains in play: the query's tags + the domains of the top few
    retrieved records (re-derived from title/one-liner with the shared tagger,
    since ExpandedRecord doesn't carry tags). De-duped, with 'general' dropped."""
    domains: List[str] = []
    seen: Set[str] = set()

    def _add(tag: Optional[str]) -> None:
        if not tag or tag == "general" or tag in seen:
            return
        seen.add(tag)
        domains.append(tag)

    for t in (query_tags or []):
        _add(t)

    if records:
        try:
            from app.astra_memory.topic_tagger import extract_tags
            for rec in list(records)[:max_record_scan]:
                text = f"{getattr(rec, 'title', '')} {getattr(rec, 'content', '')}"
                for t in extract_tags(text):
                    _add(t)
        except Exception as exc:
            logger.debug("[graph_context] record tagging skipped: %s", exc)

    return domains


def _node_name(graph, node_id: str) -> str:
    """Human name for a node; fall back to the slug after 'topic:'."""
    try:
        ent = graph.get_entity(node_id)
        if ent and ent.name:
            return ent.name
    except Exception:
        pass
    return node_id.split(":", 1)[-1].replace("_", " ")


def _format_link(graph, rel) -> str:
    """One plain, directional sentence for an edge, e.g.
    'investments funds the Portugal move'. Uses the stored relation phrase and
    respects the edge's original direction (source -> target) regardless of
    which end we reached it from."""
    try:
        relation = str((rel.metadata or {}).get("relation", "")).strip()
    except Exception:
        relation = ""
    if not relation:
        relation = "is connected to"
    src_name = _node_name(graph, rel.source_id)
    tgt_name = _node_name(graph, rel.target_id)
    if not src_name or not tgt_name:
        return ""
    return f"{src_name} {relation} {tgt_name}"


def build_graph_context(query_tags, records) -> str:
    """Walk the knowledge graph from the topics in play and surface strong
    cross-domain links as a context block. Returns "" when nothing qualifies."""
    if not _enabled():
        return ""
    domains = _anchor_domains(query_tags, records)
    if not domains:
        return ""

    min_strength = _f("ASTRA_GRAPH_WALK_MIN_STRENGTH", _MIN_STRENGTH_DEFAULT)
    max_hops = max(1, _i("ASTRA_GRAPH_WALK_MAX_HOPS", _MAX_HOPS_DEFAULT))
    max_links = max(1, _i("ASTRA_GRAPH_WALK_MAX_LINKS", _MAX_LINKS_DEFAULT))

    try:
        from app.intelligent_memory.graph import get_knowledge_graph
        graph = get_knowledge_graph()
    except Exception as exc:
        logger.debug("[graph_context] graph unavailable: %s", exc)
        return ""

    anchor_ids: Set[str] = {f"{_NODE_PREFIX}{d}" for d in domains}
    visited: Set[str] = set()
    frontier: Set[str] = set(anchor_ids)
    # de-duped by unordered endpoint pair -> (best weight, formatted line)
    links: Dict[Tuple[str, str], Tuple[float, str]] = {}

    try:
        for _hop in range(max_hops):
            next_frontier: Set[str] = set()
            for node_id in frontier:
                if node_id in visited:
                    continue
                visited.add(node_id)
                if graph.get_entity(node_id) is None:
                    continue
                for rel in graph.get_relationships(node_id, direction="both"):
                    if (rel.weight or 0.0) < min_strength:
                        continue
                    other_id = (
                        rel.target_id if rel.source_id == node_id else rel.source_id
                    )
                    # A link back into a topic already in play says nothing new —
                    # keep walking it (for the 2nd hop) but don't surface it.
                    if other_id not in anchor_ids:
                        line = _format_link(graph, rel)
                        if line:
                            key = tuple(sorted((rel.source_id, rel.target_id)))
                            prev = links.get(key)
                            if prev is None or (rel.weight or 0.0) > prev[0]:
                                links[key] = (rel.weight or 0.0, line)
                    if other_id not in visited:
                        next_frontier.add(other_id)
            frontier = next_frontier
            if not frontier:
                break
    except Exception as exc:
        logger.debug("[graph_context] walk failed: %s", exc)
        return ""

    if not links:
        return ""

    top = sorted(links.values(), key=lambda wl: wl[0], reverse=True)[:max_links]
    lines = [wl[1] for wl in top]
    logger.debug(
        "[graph_context] surfaced %d cross-domain link(s) from %d topic(s)",
        len(lines), len(domains),
    )
    return (
        "[CONNECTED MEMORY — links you have previously drawn between topics; "
        "mention only if it genuinely helps]\n"
        + "\n".join(f"- {ln}" for ln in lines)
    )


__all__ = ["build_graph_context"]
