# FILE: app/memory/seed_tiers.py
# Purpose: Master seeder for architecture memory tiers.
# Called-by: app.memory.startup
# Depends-on: app.memory.domains.decisions, app.memory.domains.dependency_scanner, app.memory.domains.package_summariser, app.memory.domains.tier_content
# Last-renovated: 2026-06-11
"""
Master seeder for architecture memory tiers.

Runs all tier seeding functions in order. Idempotent — safe to
call multiple times. Each tier checks for existing entries before
inserting.

Usage:
    from app.memory.seed_tiers import seed_all_tiers
    result = seed_all_tiers()
    print(result)

Or from command line:
    python -m app.memory.seed_tiers
"""

import logging
import sys

logger = logging.getLogger(__name__)


def seed_all_tiers(root_dir: str = r"D:\Orb") -> dict:
    """
    Seed all 5 architecture memory tiers.

    Args:
        root_dir: Project root for dependency scanning and package summaries.

    Returns:
        Summary dict with counts per tier.
    """
    results = {}

    # T1 + T2: Curated content
    from app.memory.domains.tier_content import seed_tiers_1_and_2
    t1t2 = seed_tiers_1_and_2()
    results["T1_system_overview"] = t1t2["tier1_inserted"]
    results["T2_subsystem_contracts"] = t1t2["tier2_inserted"]
    print(f"[seed] T1: {t1t2['tier1_inserted']} inserted, T2: {t1t2['tier2_inserted']} inserted")

    # T3: Dependency graph
    from app.memory.domains.dependency_scanner import store_dependency_graph
    try:
        graph_id = store_dependency_graph(root_dir=root_dir)
        results["T3_dependency_graph"] = 1
        print(f"[seed] T3: dependency graph stored (id={graph_id})")
    except Exception as e:
        results["T3_dependency_graph"] = 0
        results["T3_error"] = str(e)
        print(f"[seed] T3: FAILED — {e}")
        logger.error(f"[seed_tiers] T3 failed: {e}")

    # T4: Design decisions
    from app.memory.domains.decisions import seed_decisions
    t4_count = seed_decisions()
    results["T4_design_decisions"] = t4_count
    print(f"[seed] T4: {t4_count} decisions seeded")

    # T5: Package summaries
    arch_dir = f"{root_dir}\\.architecture"
    from app.memory.domains.package_summariser import store_package_summaries
    try:
        t5_count = store_package_summaries(arch_dir=arch_dir)
        results["T5_package_summaries"] = t5_count
        print(f"[seed] T5: {t5_count} package summaries stored")
    except Exception as e:
        results["T5_package_summaries"] = 0
        results["T5_error"] = str(e)
        print(f"[seed] T5: FAILED — {e}")
        logger.error(f"[seed_tiers] T5 failed: {e}")

    return results


if __name__ == "__main__":
    # Allow running from command line
    import os
    sys.path.insert(0, r"D:\Orb")
    os.chdir(r"D:\Orb")
    from dotenv import load_dotenv
    load_dotenv(r"D:\Orb\.env")

    result = seed_all_tiers()
    print(f"\n{'='*50}")
    print("SEED COMPLETE")
    print(f"{'='*50}")
    for k, v in result.items():
        print(f"  {k}: {v}")
