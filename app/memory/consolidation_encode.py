# FILE: app/memory/consolidation_encode.py
# Purpose: Encode finalised memory items for retrieval (tag + embed).
# Called-by: app.memory.knowledge_extractor
# Depends-on: app.astra_memory.topic_tagger, app.astra_memory.retrieval, app.embeddings.service
# Last-renovated: 2026-06-15
"""
Encode-for-retrieval for consolidated memory items (MEMORY_JOBS_SPEC Job 3, task 4).

After the membrane finalises a durable fact this makes it findable by Job 4
retrieval, in two channels:

  - TAG: the value is tagged with the shared topic_tagger vocabulary and
    upserted as a HotIndex row (record_type 'memory') -- the store the
    astra_memory retrieval layer reads. Keyed by the fact's PreferenceRecord id,
    so re-consolidating a session upserts in place and never duplicates.
  - EMBED: the finished value is vectorised (ENCODE ONLY -- the embedder never
    reads or decides; it only encodes the text the membrane already chose) into
    the embeddings table under (source_type='memory', source_id=pref_id). The
    semantic channel in semantic_candidates maps that back onto the HotIndex row.

Correct model-role split: deterministic code + the membrane's cheap LLM decide
WHAT is durable; the embeddings model only encodes the finished item.

Best-effort: embedding needs the Gemini key and is skipped (not fatal) when it
is unavailable. Never raises into the caller.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# record_type used in the HotIndex; source_type used in the embeddings table.
# Kept identical and registered in semantic_candidates._SOURCE_TO_RECORD_TYPE so
# the vector channel can resurface these items.
MEMORY_RECORD_TYPE = "memory"
MEMORY_SOURCE_TYPE = "memory"


def encode_memory_item(
    db,
    project_id,
    pref_id: int,
    key: str,
    value,
    reembed: bool = True,
) -> bool:
    """Tag + embed one finalised memory item. Returns True if it was tagged
    (the HotIndex row written); embedding is best-effort on top of that.

    reembed: re-generate the vector (value was created or changed). When False
    the embedding is only generated if one does not already exist, so an item
    that is merely reinforced doesn't pay for a redundant embedding call but is
    still guaranteed to be present in the vector channel at least once.
    """
    text = ("" if value is None else str(value)).strip()
    if not text or not pref_id:
        return False
    record_id = str(pref_id)

    # 1. TAG -> HotIndex (the store Job 4 retrieval reads). No API cost.
    try:
        from app.astra_memory.topic_tagger import extract_tags, extract_entities
        from app.astra_memory.retrieval import upsert_hot_index, RetrievalCost

        tag_source = f"{key} {text}"
        tags = extract_tags(tag_source)
        entities = extract_entities(tag_source)
        title = (key or "memory").replace("_", " ").strip()[:80] or "memory"

        upsert_hot_index(
            db=db,
            record_type=MEMORY_RECORD_TYPE,
            record_id=record_id,
            title=title,
            one_liner=text[:300],
            tags=tags,
            entities=entities,
            retrieval_priority=0.75,  # consolidated facts are high-value
            retrieval_cost=RetrievalCost.TINY,
        )
    except Exception as exc:
        logger.warning("[consolidation_encode] hot-index failed for pref %s: %s", pref_id, exc)
        return False

    # 2. EMBED (encode only) -> embeddings table. Best-effort.
    if project_id is not None:
        try:
            from app.embeddings.service import (
                generate_embedding,
                store_embedding,
                delete_embeddings_for_source,
                embedding_exists,
            )

            need = reembed or not embedding_exists(
                db, project_id, MEMORY_SOURCE_TYPE, pref_id,
            )
            if need:
                vec = generate_embedding(text, task_type="RETRIEVAL_DOCUMENT")
                if vec:
                    delete_embeddings_for_source(
                        db, project_id, MEMORY_SOURCE_TYPE, pref_id,
                    )
                    store_embedding(
                        db, project_id, MEMORY_SOURCE_TYPE, pref_id, text, vec, 0,
                    )
                else:
                    logger.debug(
                        "[consolidation_encode] no embedding for pref %s "
                        "(embedding key unavailable?) — tagged only",
                        pref_id,
                    )
        except Exception as exc:
            logger.debug("[consolidation_encode] embed skipped for pref %s: %s", pref_id, exc)

    return True
