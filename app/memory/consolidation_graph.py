# FILE: app/memory/consolidation_graph.py
# Purpose: Consolidate a session's cross-domain links into the knowledge graph.
# Called-by: app.memory.knowledge_extractor
# Depends-on: app.astra_memory.topic_tagger, app.intelligent_memory.graph, app.llm._streaming_utils_3
# Last-renovated: 2026-06-15
"""
Cross-domain link consolidation for the knowledge graph (MEMORY_JOBS_SPEC Job 3, task 3).

When a session is archived the membrane (knowledge_extractor) hands the rolling
summary here. We capture the RELATIONSHIPS the user drew between domains —
"investments fund the Portugal land", "the legal claim pays for the move" — as
explicit graph edges, so Job 4 retrieval can walk them. Similarity already lives
in the embedding/vector channel; this module is only for relationships
(node -- relation -- node -- strength), which vectors cannot represent.

Deterministic-first, then a cheap LLM for judgement:
  1. topic_tagger narrows the summary to the domains actually present.
  2. Only when >=2 distinct domains co-occur do we spend a flash-lite call, and
     it only confirms/labels the genuine connections (funding, dependency,
     causation, blocking) -- not mere co-mention in the same session.
  3. Edges strengthen on recurrence (weight bump, capped). Load-bearing links
     rise; one-off mentions stay weak.

Runs on the cheap batch tier at archive -- never latency-sensitive. Never raises
into the caller: every failure is logged and swallowed so consolidating one
session can never break the lifecycle scheduler.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)


# Recurrence strengthening: each repeat of a link bumps its weight, capped so a
# single heavily-repeated link can't dominate traversal. One-offs stay at base.
_BASE_WEIGHT = 1.0
_STRENGTH_INC = 1.0
_STRENGTH_MAX = 10.0


def _enabled() -> bool:
    """Kill-switch: ASTRA_GRAPH_CONSOLIDATION=off disables graph writes."""
    return os.getenv("ASTRA_GRAPH_CONSOLIDATION", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _get_link_model() -> Tuple[str, str]:
    """The cheap batch tier — same env-resolved class the summariser uses.

    No hardcoded fallback (Lane B 2026-07-01): unresolved config returns ""s
    and the guarded call site skips the LLM step."""
    from app.memory.model_env import summary_model_from_env
    return summary_model_from_env()


_LINK_SYSTEM_PROMPT = """\
You map the RELATIONSHIPS a user has drawn between different areas of their life \
or work, for a personal AI's knowledge graph.

You are given the topic domains present in a conversation plus its summary. \
Output ONLY the genuine, explicit connections the user made BETWEEN two \
different domains -- where one thing funds, pays for, depends on, blocks, \
causes, or is part of another.

CRITICAL RULES:
- Output ONLY a JSON array. No markdown fences, no preamble.
- Each item: {"source": "...", "relation": "...", "target": "..."}
- "source" and "target" are short names of a domain or thing
  (e.g. "investments", "Portugal move", "legal claim").
- "relation" is a short phrase for HOW they connect
  (e.g. "funds", "pays for", "depends on", "blocks", "part of").
- ONLY real connections the user stated or clearly implied. Two topics merely
  being discussed in the same session is NOT a connection -- skip those.
- 0 to 4 links. An empty array [] is the correct answer when no domains were
  actually connected.
- Never connect a domain to itself.

EXAMPLE OUTPUT:
[{"source": "investments", "relation": "funds", "target": "Portugal move"}]
"""


@dataclass
class DomainLink:
    """A single cross-domain connection extracted from a conversation."""
    source: str
    relation: str
    target: str


# =========================================================================
# Parsing
# =========================================================================

def _parse_links(raw: str) -> List[DomainLink]:
    """Parse the LLM response into DomainLinks. Tolerant of fences / preamble."""
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return []

    if not isinstance(data, list):
        return []

    out: List[DomainLink] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        relation = str(item.get("relation", "")).strip()
        if source and target and source.lower() != target.lower():
            out.append(DomainLink(source=source, relation=relation, target=target))
    return out[:5]


# =========================================================================
# Node mapping
# =========================================================================

def _node_for(text: str) -> Tuple[str, str]:
    """Map a link endpoint to a stable graph node id + display name.

    Prefer a known topic domain (so "Portugal move" and "the move to Portugal"
    collapse onto the same `topic:portugal` node); otherwise slug the phrase.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ("", "")
    try:
        from app.astra_memory.topic_tagger import extract_tags
        domains = [t for t in extract_tags(cleaned) if t != "general"]
    except Exception:
        domains = []
    if domains:
        domain = domains[0]
        return (f"topic:{domain}", domain)
    # Strip leading articles / possessives so phrasing variants of the same
    # free-text endpoint collapse onto one node ("the move to portugal" and
    # "move to portugal"), giving recurrence a chance to strengthen the edge.
    normalised = re.sub(
        r"\b(?:the|a|an|my|our|your|his|her|their|its)\b", " ", cleaned.lower()
    )
    slug = re.sub(r"[^a-z0-9]+", "_", normalised).strip("_")[:48]
    if not slug:
        return ("", "")
    return (f"topic:{slug}", cleaned[:60])


# =========================================================================
# Graph write (create or strengthen)
# =========================================================================

def _write_link(graph, link: DomainLink, session_id: int) -> bool:
    """Create the edge, or strengthen it if the same pair recurs. Returns
    True if an edge was written or strengthened."""
    from app.intelligent_memory.graph import (
        Entity, EntityType, Relationship, RelationType,
    )

    src_id, src_name = _node_for(link.source)
    tgt_id, tgt_name = _node_for(link.target)
    if not src_id or not tgt_id or src_id == tgt_id:
        return False

    # Ensure both endpoint nodes exist (don't clobber a richer existing node).
    for nid, nname in ((src_id, src_name), (tgt_id, tgt_name)):
        if graph.get_entity(nid) is None:
            graph.add_entity(Entity(
                entity_id=nid,
                entity_type=EntityType.TOPIC,
                name=nname,
                tags=["conversation"],
            ))

    relation_phrase = (link.relation or "related to").strip()[:60]

    # Match an existing edge between this unordered pair in EITHER direction.
    # Consolidation links are semantically symmetric (RELATED_TO) and the store is
    # a DIRECTED graph keyed on (source, target) — so a link drawn the reverse way
    # must STRENGTHEN the same edge, not spawn a duplicate (the bug where
    # ai<->investments was stored as two separate occ=1 edges, so MIN_STRENGTH
    # could never distinguish a load-bearing link from a one-off).
    existing = None
    for rel in graph.get_relationships(src_id, direction="both"):
        other = rel.target_id if rel.source_id == src_id else rel.source_id
        if other == tgt_id:
            existing = rel
            break

    if existing is not None:
        new_weight = min(existing.weight + _STRENGTH_INC, _STRENGTH_MAX)
        meta = dict(existing.metadata or {})
        meta["occurrences"] = int(meta.get("occurrences", 1)) + 1
        sessions = list(meta.get("sessions", []))
        if session_id not in sessions:
            sessions = (sessions + [session_id])[-10:]
        meta["sessions"] = sessions
        meta.setdefault("relation", relation_phrase)
        # Re-add on the EXISTING edge's orientation so the directed graph updates
        # that edge in place; the new link's orientation would create the reverse
        # duplicate we are trying to merge.
        graph.add_relationship(Relationship(
            source_id=existing.source_id,
            target_id=existing.target_id,
            relation_type=RelationType.RELATED_TO,
            weight=new_weight,
            metadata=meta,
        ))
        logger.debug(
            "[consolidation_graph] strengthened %s -> %s (w=%.1f, n=%d)",
            src_id, tgt_id, new_weight, meta["occurrences"],
        )
    else:
        graph.add_relationship(Relationship(
            source_id=src_id,
            target_id=tgt_id,
            relation_type=RelationType.RELATED_TO,
            weight=_BASE_WEIGHT,
            metadata={
                "relation": relation_phrase,
                "occurrences": 1,
                "sessions": [session_id],
                "origin": "conversation",
            },
        ))
        logger.debug("[consolidation_graph] created %s -> %s", src_id, tgt_id)
    return True


# =========================================================================
# Public entry point
# =========================================================================

async def detect_and_write_links(summary_text: str, session_id: int) -> int:
    """Detect cross-domain links in a session and write them to the graph.

    Returns the number of edges created or strengthened. Best-effort: never
    raises. Skips the LLM entirely unless >=2 distinct domains are present.
    """
    if not _enabled():
        return 0
    if not summary_text or not summary_text.strip():
        return 0

    # Deterministic gate: nothing cross-domain unless >=2 domains co-occur.
    try:
        from app.astra_memory.topic_tagger import extract_tags
        domains = sorted({t for t in extract_tags(summary_text) if t != "general"})
    except Exception as exc:
        logger.debug("[consolidation_graph] tag gate failed: %s", exc)
        return 0
    if len(domains) < 2:
        logger.debug(
            "[consolidation_graph] session %s: <2 domains (%s) — no cross-domain link",
            session_id, domains,
        )
        return 0

    # LLM judgement (cheap batch tier).
    provider, model = _get_link_model()
    if not provider or not model:
        logger.warning("[consolidation_graph] link model unconfigured — skipping")
        return 0
    try:
        from app.llm._streaming_utils_3 import call_llm_text
        raw = await call_llm_text(
            provider=provider,
            model=model,
            system_prompt=_LINK_SYSTEM_PROMPT,
            user_prompt=(
                "Topic domains present: " + ", ".join(domains) + "\n\n"
                "Conversation summary:\n" + summary_text[:6000]
            ),
            max_tokens=600,
            timeout_seconds=30,
        )
    except Exception as exc:
        logger.warning("[consolidation_graph] link LLM failed for session %s: %s", session_id, exc)
        return 0

    links = _parse_links(raw)
    if not links:
        logger.info("[consolidation_graph] session %s: no cross-domain links found", session_id)
        return 0

    try:
        from app.intelligent_memory.graph import get_knowledge_graph
        graph = get_knowledge_graph()
    except Exception as exc:
        logger.warning("[consolidation_graph] graph unavailable: %s", exc)
        return 0

    written = 0
    for link in links:
        try:
            if _write_link(graph, link, session_id):
                written += 1
        except Exception as exc:
            logger.warning("[consolidation_graph] write failed for %s: %s", link, exc)

    logger.info(
        "[consolidation_graph] session %s: %d cross-domain edge(s) written/strengthened",
        session_id, written,
    )
    return written
