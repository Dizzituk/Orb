# FILE: app/memory/domains/preference_seeds.py
"""
Seed initial user preferences.

Populates astra_preferences with known hard-rule preferences.
Idempotent — skips preferences that already exist.

Usage:
    from app.memory.domains.preference_seeds import seed_preferences
    result = seed_preferences()
    print(result)
"""

import logging
from typing import Optional

from app.db import get_db_session
from app.astra_memory.preference_models import PreferenceRecord
from app.astra_memory.preference_service import create_hard_rule

logger = logging.getLogger(__name__)


# =========================================================================
# Seed definitions
# =========================================================================

SEED_PREFERENCES = [
    {
        "key": "file_size_target_kb",
        "value": 20,
        "namespace": "development",
        "applies_to": "all",
        "description": "Target file size in KB. Logic files should aim for this.",
    },
    {
        "key": "file_size_max_kb",
        "value": 30,
        "namespace": "development",
        "applies_to": "all",
        "description": "Absolute maximum file size in KB. Never exceed.",
    },
    {
        "key": "modularity_first",
        "value": True,
        "namespace": "development",
        "applies_to": "all",
        "description": "Always prefer modular, composable code over monoliths.",
    },
    {
        "key": "no_git_commands",
        "value": True,
        "namespace": "development",
        "applies_to": "all",
        "description": "ASTRA must never execute git commands. User handles all git.",
    },
    {
        "key": "evidence_first",
        "value": True,
        "namespace": "development",
        "applies_to": "specgate",
        "description": "Gather codebase evidence before generating specs. Non-negotiable.",
    },
    {
        "key": "best_way_not_quickest",
        "value": True,
        "namespace": "development",
        "applies_to": "all",
        "description": "Optimise for quality and correctness, never for speed.",
    },
    {
        "key": "dont_guess_anything",
        "value": True,
        "namespace": "development",
        "applies_to": "all",
        "description": "Never guess file paths, function names, or API shapes. Verify.",
    },
    {
        "key": "sandbox_execution_only",
        "value": True,
        "namespace": "development",
        "applies_to": "implementer",
        "description": "All generated code executes in sandbox, never on host.",
    },
    {
        "key": "small_files_over_bundling",
        "value": True,
        "namespace": "development",
        "applies_to": "all",
        "description": "Prefer many small files over convenience bundling.",
    },
    {
        "key": "user_handles_git",
        "value": True,
        "namespace": "development",
        "applies_to": "all",
        "description": "User manages all git operations. ASTRA never runs git.",
    },
]


# =========================================================================
# Seed function
# =========================================================================

def seed_preferences() -> dict:
    """
    Seed initial hard-rule preferences. Idempotent.

    Returns:
        Dict with 'inserted' count and 'skipped' count.
    """
    db = get_db_session()
    inserted = 0
    skipped = 0

    try:
        for seed in SEED_PREFERENCES:
            # Check if already exists
            existing = db.query(PreferenceRecord).filter(
                PreferenceRecord.preference_key == seed["key"],
            ).first()

            if existing:
                logger.debug(
                    "[preference_seeds] Skipping '%s' — already exists",
                    seed["key"],
                )
                skipped += 1
                continue

            create_hard_rule(
                db=db,
                preference_key=seed["key"],
                preference_value=seed["value"],
                applies_to=seed.get("applies_to"),
                context_pointer="seed:initial_hard_rules",
            )

            # Update namespace to match domain (create_hard_rule uses
            # 'hard_rules' namespace by default, we want domain-specific)
            pref = db.query(PreferenceRecord).filter(
                PreferenceRecord.preference_key == seed["key"],
            ).first()
            if pref and pref.namespace != seed["namespace"]:
                pref.namespace = seed["namespace"]
                db.commit()

            inserted += 1
            logger.info(
                "[preference_seeds] Seeded: %s = %s (%s)",
                seed["key"], seed["value"], seed["namespace"],
            )

        result = {"inserted": inserted, "skipped": skipped}
        logger.info("[preference_seeds] Seed complete: %s", result)
        return result

    finally:
        db.close()


# =========================================================================
# CLI entry point
# =========================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    result = seed_preferences()
    print(f"Seed result: {result}")
    sys.exit(0)
