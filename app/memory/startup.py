# FILE: app/memory/startup.py
"""
Memory system initialisation.

Called at application boot to:
1. Register all domain stores with the MemoryRouter singleton
2. Run schema migrations (idempotent)
3. Seed initial data (idempotent)

Usage:
    from app.memory.startup import init_memory_system
    init_memory_system()

    # Or selectively:
    from app.memory.startup import register_domain_stores
    register_domain_stores()
"""

import logging

from app.memory.router import memory_router

logger = logging.getLogger(__name__)


def register_domain_stores() -> list[str]:
    """
    Register all domain stores with the MemoryRouter singleton.

    Returns list of registered domain names.
    """
    registered = []

    # Architecture store
    try:
        from app.memory.domains.architecture import ArchitectureStore
        memory_router.register(ArchitectureStore())
        registered.append("architecture")
    except Exception as e:
        logger.error("[startup] Failed to register architecture store: %s", e)

    # Preference store
    try:
        from app.memory.domains.preferences import PreferenceStore
        memory_router.register(PreferenceStore())
        registered.append("preference")
    except Exception as e:
        logger.error("[startup] Failed to register preference store: %s", e)

    # Conversation context store
    try:
        from app.memory.domains.context import ContextStore
        memory_router.register(ContextStore())
        registered.append("context")
    except Exception as e:
        logger.error("[startup] Failed to register context store: %s", e)

    # Confidence score store
    try:
        from app.memory.domains.confidence import ConfidenceStore
        memory_router.register(ConfidenceStore())
        registered.append("confidence")
    except Exception as e:
        logger.error("[startup] Failed to register confidence store: %s", e)

    # Domain knowledge store
    try:
        from app.memory.domains.knowledge import KnowledgeStore
        memory_router.register(KnowledgeStore())
        registered.append("knowledge")
    except Exception as e:
        logger.error("[startup] Failed to register knowledge store: %s", e)

    logger.info(
        "[startup] Registered %d domain stores: %s",
        len(registered), registered,
    )
    return registered


def run_seeds() -> dict:
    """
    Run all idempotent seed operations.

    Returns summary dict of seed results.
    """
    results = {}

    # Architecture tier seeds (T1–T5)
    try:
        from app.memory.seed_tiers import seed_all_tiers
        results["architecture_tiers"] = seed_all_tiers()
    except Exception as e:
        logger.error("[startup] Architecture tier seeding failed: %s", e)
        results["architecture_tiers"] = {"error": str(e)}

    # Design decisions (T4)
    try:
        from app.memory.domains.decisions import seed_decisions
        results["decisions"] = seed_decisions()
    except Exception as e:
        logger.error("[startup] Decision seeding failed: %s", e)
        results["decisions"] = {"error": str(e)}

    # Preference seeds (Job 4)
    try:
        from app.memory.domains.preference_seeds import seed_preferences
        results["preferences"] = seed_preferences()
    except Exception as e:
        logger.error("[startup] Preference seeding failed: %s", e)
        results["preferences"] = {"error": str(e)}

    logger.info("[startup] Seed results: %s", results)
    return results


def init_memory_system() -> dict:
    """
    Full memory system initialisation.

    Call once at application boot.

    Returns:
        Summary dict with registration and seed results.
    """
    summary = {}

    # Step 1: Schema migrations
    try:
        from app.memory.migrations import run_memory_migrations
        summary["migrations"] = run_memory_migrations()
    except Exception as e:
        logger.error("[startup] Migrations failed: %s", e)
        summary["migrations"] = {"error": str(e)}

    # Step 2: Register domain stores
    summary["registered_domains"] = register_domain_stores()

    # Step 3: Seed data
    summary["seeds"] = run_seeds()

    # Step 4: Start conversation session lifecycle scheduler (v10.0)
    try:
        from app.memory.session_lifecycle import start_lifecycle_scheduler
        start_lifecycle_scheduler(interval_minutes=15)
        summary["lifecycle_scheduler"] = "started"
    except Exception as e:
        logger.error("[startup] Lifecycle scheduler failed: %s", e)
        summary["lifecycle_scheduler"] = {"error": str(e)}

    logger.info("[startup] Memory system initialised: %s", summary)
    return summary
