# FILE: app/intelligent_memory/memory_startup.py
"""
Intelligent Memory System startup.

Called at application boot to initialise all three layers:
  1. Hot Cache — load from disk or create with defaults
  2. Knowledge Graph — load from disk or create empty
  3. Asset Index — load from disk or create empty
  4. Retrieval Router — create with lazy component loading

Also seeds the Hot Cache with data from existing memory systems
(HotIndex table, preferences, project state) for migration.

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-MEM-001.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def init_intelligent_memory() -> Dict[str, Any]:
    """Initialise the intelligent memory system.

    Call once at application boot, after the existing memory system
    has been initialised (so we can seed from it).

    Returns:
        Summary dict with init results.
    """
    summary: Dict[str, Any] = {}

    # Layer 1: Hot Cache
    try:
        from app.intelligent_memory.hot_cache import get_hot_cache
        cache = get_hot_cache()
        stats = cache.get_stats()
        summary["hot_cache"] = {
            "status": "loaded",
            "size_bytes": stats["size_bytes"],
            "items": stats["item_count"],
        }
        logger.info("[imem_startup] Hot Cache: %d items, %d bytes",
                     stats["item_count"], stats["size_bytes"])
    except Exception as e:
        summary["hot_cache"] = {"status": "error", "error": str(e)}
        logger.error("[imem_startup] Hot Cache failed: %s", e)

    # Layer 2: Knowledge Graph
    try:
        from app.intelligent_memory.graph import get_knowledge_graph
        graph = get_knowledge_graph()
        stats = graph.get_stats()
        summary["knowledge_graph"] = {
            "status": "loaded",
            "entities": stats["total_entities"],
            "relationships": stats["total_relationships"],
        }
        logger.info("[imem_startup] Knowledge Graph: %d entities, %d rels",
                     stats["total_entities"], stats["total_relationships"])
    except Exception as e:
        summary["knowledge_graph"] = {"status": "error", "error": str(e)}
        logger.error("[imem_startup] Knowledge Graph failed: %s", e)

    # Asset Index
    try:
        from app.intelligent_memory.asset_index import get_asset_index
        idx = get_asset_index()
        stats = idx.get_stats()
        summary["asset_index"] = {
            "status": "loaded",
            "total_assets": stats["total_assets"],
        }
        logger.info("[imem_startup] Asset Index: %d assets",
                     stats["total_assets"])
    except Exception as e:
        summary["asset_index"] = {"status": "error", "error": str(e)}
        logger.error("[imem_startup] Asset Index failed: %s", e)

    # Retrieval Router (lazy, just instantiate)
    try:
        from app.intelligent_memory.retrieval_router import get_retrieval_router
        get_retrieval_router()
        summary["retrieval_router"] = {"status": "ready"}
    except Exception as e:
        summary["retrieval_router"] = {"status": "error", "error": str(e)}
        logger.error("[imem_startup] Retrieval Router failed: %s", e)

    # Seed from existing memory if hot cache is empty
    try:
        cache = get_hot_cache()
        if cache._count_items() <= len(cache._data.get("metadata", {})):
            _seed_from_existing(cache)
            summary["seed"] = "seeded from existing memory"
        else:
            summary["seed"] = "skipped (cache has data)"
    except Exception as e:
        summary["seed"] = {"error": str(e)}

    logger.info("[imem_startup] Intelligent Memory initialised: %s", summary)
    return summary


def _seed_from_existing(cache) -> None:
    """Seed the hot cache from existing ASTRA memory systems.

    Pulls data from:
      - HotIndex table (astra_hot_index)
      - Preference records (astra_preferences)
      - Pipeline config from env vars
    """
    import os

    logger.info("[imem_startup] Seeding hot cache from existing memory...")

    # Seed user profile from known facts
    cache.set("user_profile", "name", "Taz")
    cache.set("user_profile", "location", "Cornwall, UK")
    cache.set("user_profile", "background", [
        "delivery driver (Yodel)",
        "10 years personal training",
        "17 years catering management",
        "bodyboarder and surfer",
    ])
    cache.set("user_profile", "goals", [
        "5-year exit from delivery driving",
        "Portugal D7 visa relocation",
        "ASTRA as autonomous development platform",
    ])

    # Seed active projects
    cache.set_section("active_projects", {
        "astra-backend": {
            "name": "ASTRA Backend",
            "status": "active",
            "phase": "v2.2 pipeline",
            "root": "D:/Orb",
        },
        "astra-frontend": {
            "name": "ASTRA Desktop",
            "status": "active",
            "root": "D:/orb-desktop",
        },
        "driver-copilot": {
            "name": "Driver CoPilot",
            "status": "active",
            "phase": "Phase 1",
            "root": "D:/Astra Android Folder/AndroidDriverCopilot",
        },
    })

    # Seed pipeline config
    cache.update_pipeline_config({
        "builder_model": os.getenv("ASTRA_V2_BUILDER_MODEL", "gpt-5.4"),
        "verifier_model": os.getenv("ASTRA_V2_VERIFIER_MODEL", "gemini-2.5-flash"),
        "feature_flags": {
            "v2_pipeline": os.getenv("ASTRA_V2_PIPELINE", "false"),
            "bvl_enabled": os.getenv("ASTRA_BVL_ENABLED", "true"),
            "enhanced_checkout": os.getenv("ASTRA_V2_ENHANCED_CHECKOUT", "true"),
        },
    })

    # Seed preferences from existing preference table
    try:
        from app.db import get_db_session
        from app.astra_memory.preference_models import PreferenceRecord
        db = get_db_session()
        prefs = db.query(PreferenceRecord).filter(
            PreferenceRecord.status == "active"
        ).order_by(PreferenceRecord.confidence.desc()).limit(20).all()
        pref_list = [
            {"key": p.preference_key, "value": p.value_text, "confidence": p.confidence}
            for p in prefs
        ]
        cache.update_preferences(pref_list)
        db.close()
        logger.info("[imem_startup] Seeded %d preferences", len(pref_list))
    except Exception as e:
        logger.debug("[imem_startup] Preference seeding skipped: %s", e)

    logger.info("[imem_startup] Hot cache seeded")
