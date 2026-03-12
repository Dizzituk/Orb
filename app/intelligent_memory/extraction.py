# FILE: app/intelligent_memory/extraction.py
"""
Knowledge Extraction Pipeline — processes conversations into
structured memory entries across all three layers.

Extraction categories:
  - New facts → Hot Cache
  - Decisions → Knowledge Graph
  - Action items → Hot Cache + Knowledge Graph
  - Generated assets → Asset Index
  - Pattern discoveries → Knowledge Graph
  - Preference updates → Hot Cache
  - Project state changes → Hot Cache + Knowledge Graph

Runs at conversation end or periodically during long conversations.
Uses a lightweight LLM call for classification (candidate for local
inference migration).

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-MEM-001.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExtractionItem:
    """A single extracted item from a conversation."""
    category: str       # fact, decision, action_item, asset, pattern, preference, project_state
    key: str            # Short identifier
    value: str          # The extracted information
    confidence: float = 0.7
    source_message_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Summary of an extraction run."""
    items: List[ExtractionItem] = field(default_factory=list)
    hot_cache_updates: int = 0
    graph_updates: int = 0
    asset_registrations: int = 0
    errors: List[str] = field(default_factory=list)
    duration_ms: int = 0


EXTRACTION_SYSTEM = """You are a knowledge extractor. Given a conversation transcript,
identify facts worth storing in long-term memory.

OUTPUT FORMAT: Respond with ONLY a JSON array. No markdown fences, no preamble.

Each item must have:
  "category": one of [fact, decision, action_item, asset, pattern, preference, project_state]
  "key": short identifier (e.g. "current_project_phase", "file_size_preference")
  "value": the extracted information
  "confidence": 0.0-1.0 how confident you are this is correct

RULES:
- Only extract information that is clearly stated or strongly implied
- Prefer specific facts over vague observations
- Decisions should capture WHAT was decided and WHY
- Action items should be actionable with clear ownership
- Patterns should describe what worked or failed and why
- Skip small talk, greetings, and meta-conversation
"""


async def extract_from_transcript(
    transcript: str,
    project_id: str = "astra-core",
) -> ExtractionResult:
    """Extract structured knowledge from a conversation transcript.

    Args:
        transcript: The conversation text to process.
        project_id: Current project context.

    Returns:
        ExtractionResult with items routed to appropriate layers.
    """
    import time
    t_start = time.time()
    result = ExtractionResult()

    # Generate extractions via LLM
    items = await _call_extraction_llm(transcript)
    result.items = items

    # Route each item to the appropriate memory layer
    for item in items:
        try:
            _route_item(item, project_id)
            if item.category in ("fact", "preference", "action_item", "project_state"):
                result.hot_cache_updates += 1
            if item.category in ("decision", "pattern", "project_state"):
                result.graph_updates += 1
            if item.category == "asset":
                result.asset_registrations += 1
        except Exception as e:
            result.errors.append(f"{item.key}: {e}")
            logger.warning("[extraction] Failed to route %s: %s", item.key, e)

    result.duration_ms = int((time.time() - t_start) * 1000)
    logger.info(
        "[extraction] Extracted %d items: %d hot, %d graph, %d assets (%dms)",
        len(items), result.hot_cache_updates,
        result.graph_updates, result.asset_registrations,
        result.duration_ms,
    )
    return result


def extract_from_transcript_sync(
    transcript: str,
    project_id: str = "astra-core",
) -> ExtractionResult:
    """Synchronous wrapper for extraction (for non-async contexts)."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    extract_from_transcript(transcript, project_id),
                )
                return future.result(timeout=30)
        return loop.run_until_complete(
            extract_from_transcript(transcript, project_id)
        )
    except Exception:
        return asyncio.run(
            extract_from_transcript(transcript, project_id)
        )


# ═══════════════════════════════════════════════════════════════════
# LLM call
# ═══════════════════════════════════════════════════════════════════

async def _call_extraction_llm(transcript: str) -> List[ExtractionItem]:
    """Call the extraction model to classify conversation content."""
    provider = os.getenv("SUMMARY_PROVIDER", "google")
    model = os.getenv("SUMMARY_MODEL", "gemini-2.5-flash-lite")

    user_prompt = (
        f"Extract structured knowledge from this conversation:\n\n"
        f"{transcript[:10000]}"
    )

    try:
        from app.pipeline_v2.llm_caller import call_llm
        raw = await call_llm(
            provider=provider,
            model=model,
            system_prompt=EXTRACTION_SYSTEM,
            user_prompt=user_prompt,
            max_tokens=4000,
            temperature=0.0,
        )
    except Exception as e:
        logger.error("[extraction] LLM call failed: %s", e)
        return []

    return _parse_extraction_response(raw)


def _parse_extraction_response(raw: str) -> List[ExtractionItem]:
    """Parse the LLM's JSON response into ExtractionItems."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(data, list):
        return []

    items = []
    for item_data in data:
        if not isinstance(item_data, dict):
            continue
        try:
            items.append(ExtractionItem(
                category=item_data.get("category", "fact"),
                key=item_data.get("key", ""),
                value=item_data.get("value", ""),
                confidence=float(item_data.get("confidence", 0.7)),
                metadata=item_data.get("metadata", {}),
            ))
        except (ValueError, TypeError):
            continue

    return items


# ═══════════════════════════════════════════════════════════════════
# Routing
# ═══════════════════════════════════════════════════════════════════

def _route_item(item: ExtractionItem, project_id: str) -> None:
    """Route an extracted item to the appropriate memory layer."""

    if item.category == "fact":
        _route_to_hot_cache(item, "user_profile")

    elif item.category == "preference":
        _route_to_hot_cache(item, "preference_snapshot")

    elif item.category == "decision":
        _route_to_hot_cache_decision(item)
        _route_to_graph_decision(item, project_id)

    elif item.category == "action_item":
        _route_to_hot_cache_action(item)

    elif item.category == "pattern":
        _route_to_graph_pattern(item, project_id)

    elif item.category == "project_state":
        _route_to_hot_cache_project(item, project_id)
        _route_to_graph_project(item, project_id)

    elif item.category == "asset":
        _route_to_asset_index(item, project_id)


def _route_to_hot_cache(item: ExtractionItem, section: str) -> None:
    from app.intelligent_memory.hot_cache import get_hot_cache
    cache = get_hot_cache()
    cache.set(section, item.key, item.value)


def _route_to_hot_cache_decision(item: ExtractionItem) -> None:
    from app.intelligent_memory.hot_cache import get_hot_cache
    cache = get_hot_cache()
    cache.add_decision({
        "summary": item.value,
        "key": item.key,
        "confidence": item.confidence,
    })


def _route_to_hot_cache_action(item: ExtractionItem) -> None:
    from app.intelligent_memory.hot_cache import get_hot_cache
    cache = get_hot_cache()
    cache.add_action_item({
        "description": item.value,
        "key": item.key,
    })


def _route_to_hot_cache_project(item: ExtractionItem, project_id: str) -> None:
    from app.intelligent_memory.hot_cache import get_hot_cache
    cache = get_hot_cache()
    cache.update_project(project_id, {item.key: item.value})


def _route_to_graph_decision(item: ExtractionItem, project_id: str) -> None:
    from app.intelligent_memory.graph import (
        get_knowledge_graph, Entity, EntityType,
        Relationship, RelationType,
    )
    graph = get_knowledge_graph()
    entity_id = f"decision:{item.key}"
    graph.add_entity(Entity(
        entity_id=entity_id,
        entity_type=EntityType.DECISION,
        name=item.value[:100],
        attributes={"full_text": item.value, "confidence": item.confidence},
        tags=["decision"],
    ))
    graph.add_relationship(Relationship(
        source_id=entity_id,
        target_id=f"project:{project_id}",
        relation_type=RelationType.PART_OF,
    ))


def _route_to_graph_pattern(item: ExtractionItem, project_id: str) -> None:
    from app.intelligent_memory.graph import (
        get_knowledge_graph, Entity, EntityType,
        Relationship, RelationType,
    )
    graph = get_knowledge_graph()
    entity_id = f"pattern:{item.key}"
    graph.add_entity(Entity(
        entity_id=entity_id,
        entity_type=EntityType.PATTERN,
        name=item.value[:100],
        attributes={"full_text": item.value, "confidence": item.confidence},
        tags=["pattern"],
    ))
    graph.add_relationship(Relationship(
        source_id=entity_id,
        target_id=f"project:{project_id}",
        relation_type=RelationType.DISCOVERED_IN,
    ))


def _route_to_graph_project(item: ExtractionItem, project_id: str) -> None:
    from app.intelligent_memory.graph import get_knowledge_graph
    graph = get_knowledge_graph()
    graph.update_entity(f"project:{project_id}", {item.key: item.value})


def _route_to_asset_index(item: ExtractionItem, project_id: str) -> None:
    from app.intelligent_memory.asset_index import get_asset_index, AssetEntry
    idx = get_asset_index()
    idx.register(AssetEntry(
        asset_id=f"asset:{item.key}",
        asset_type=item.metadata.get("asset_type", "document"),
        name=item.key,
        description=item.value,
        project_id=project_id,
        tags=["extracted"],
    ))
