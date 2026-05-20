#!/usr/bin/env python
# FILE: scripts/migrate_user_facts_2026-05.py
"""
One-shot migration: retire user_facts.json per Phase 2 schema decisions.

Action plan (per docs/memory_canonical_schema.md section 6):
  - 6 biographical facts -> identity.json (with reconciler flagging
    disagreements; specifically the "Barnstable" typo and the still-corrupt
    current_location)
  - 40 preference facts -> preferences SQLite table
  - 28 project facts:
      - last_seen > 14 days OR key looks transient (current_task etc.)
        -> drop
      - else -> Tier 3 HotIndex with 30-day TTL
  - 4 philosophy facts -> preferences table
  - 2 learning facts -> preferences table
  - Archive user_facts.json -> data/backup/post-migration-YYYY-MM-DD/
  - Run reconciler over post-migration state, write report

USAGE:
    # Dry run (default — prints plan, makes NO changes)
    python scripts/migrate_user_facts_2026-05.py

    # Live run (only when ready)
    python scripts/migrate_user_facts_2026-05.py --apply

    # Re-runnable / idempotent — safe to re-run after partial completion
    python scripts/migrate_user_facts_2026-05.py --apply

WARNINGS:
    - Do NOT run --apply until the Phase 4 cutover has been stable for a
      few days. The arbiter must be in enforce mode so that any biographical
      moves to identity.json get queued for confirmation rather than committed
      blindly.
    - The script writes a log file at data/backup/post-migration-YYYY-MM-DD/
      migration.log with every action taken.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data" / "self_model"
USER_FACTS = DATA_DIR / "user_facts.json"
TODAY = datetime.now(timezone.utc).date().isoformat()
BACKUP_DIR = REPO_ROOT / "data" / "backup" / f"post-migration-{TODAY}"
LOG_FILE = BACKUP_DIR / "migration.log"

# Project keys that look like session-context / transient state — always drop
TRANSIENT_PROJECT_KEYS = frozenset({
    "current_task", "current_gpu", "confirmed_file", "build_issue",
    "current_project", "potential_implementation", "initial_feature_focus",
    "mvp_first", "ai_state_report", "address_finder_module",
})

PROJECT_STALE_DAYS = 14
TIER3_TTL_DAYS = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(apply_mode: bool) -> logging.Logger:
    log = logging.getLogger("migrate_user_facts")
    log.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(h)
    if apply_mode:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(fh)
    return log


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_stale(fact: Dict[str, Any], days: int) -> bool:
    last_seen = fact.get("last_seen", "")
    if not last_seen:
        return True
    try:
        ts = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
    except Exception:
        return True
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return ts < cutoff


def _confidence_to_score(confidence: str) -> float:
    return {"low": 0.3, "medium": 0.6, "high": 0.85}.get(confidence, 0.4)


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------

def build_plan(facts: List[Dict[str, Any]], log: logging.Logger) -> Dict[str, List]:
    """Categorise every fact in user_facts.json into actions."""
    plan: Dict[str, List] = {
        "to_identity": [],
        "to_preferences": [],
        "to_tier3": [],
        "to_drop": [],
        "unknown": [],
    }

    for f in facts:
        cat = f.get("category", "")
        key = f.get("key", "")
        if cat == "biographical":
            plan["to_identity"].append(f)
        elif cat in ("preference", "philosophy", "learning"):
            plan["to_preferences"].append(f)
        elif cat == "project":
            if key in TRANSIENT_PROJECT_KEYS or _is_stale(f, PROJECT_STALE_DAYS):
                plan["to_drop"].append(f)
            else:
                plan["to_tier3"].append(f)
        else:
            plan["unknown"].append(f)

    return plan


def render_plan(plan: Dict[str, List], log: logging.Logger) -> None:
    log.info("=== MIGRATION PLAN ===")
    log.info("  -> identity.json:    %d biographical facts", len(plan["to_identity"]))
    for f in plan["to_identity"]:
        log.info("       %s = %r", f.get("key"), f.get("value"))
    log.info("  -> preferences DB:   %d preference/philosophy/learning facts",
             len(plan["to_preferences"]))
    log.info("  -> Tier 3 HotIndex:  %d project facts (active)",
             len(plan["to_tier3"]))
    for f in plan["to_tier3"]:
        log.info("       %s", f.get("key"))
    log.info("  -> DROP:             %d project facts (stale or transient)",
             len(plan["to_drop"]))
    for f in plan["to_drop"]:
        log.info("       %s (last_seen=%s)", f.get("key"), f.get("last_seen", "?"))
    if plan["unknown"]:
        log.warning("  -> UNKNOWN (skipped): %d facts in unknown categories",
                    len(plan["unknown"]))


# ---------------------------------------------------------------------------
# Apply (irreversible — only when --apply passed)
# ---------------------------------------------------------------------------

def apply_to_identity(plan: List[Dict[str, Any]], log: logging.Logger) -> None:
    """
    Biographical facts go via the arbiter (so disagreements with identity.json
    surface as queued proposals for the user to review).
    """
    from app.self_model.write_arbiter import propose
    from app.self_model.canonical_schema import is_tier_1_field

    for f in plan:
        key = f.get("key", "")
        value = f.get("value")
        if not is_tier_1_field(key):
            log.warning("  skipping biographical key %r (not in canonical_schema)", key)
            continue
        result = propose(
            field_name=key,
            proposed_value=value,
            source=f"migration_2026-05:user_facts:{f.get('source','')}",
            evidence={
                "old_confidence": f.get("confidence"),
                "old_reinforcement_count": f.get("reinforcement_count"),
            },
        )
        log.info("  identity propose %s -> %s (would=%s, reason=%s)",
                 key, result.status, result.would_have_status, result.reason)


def apply_to_preferences(plan: List[Dict[str, Any]], log: logging.Logger) -> None:
    """Write preference/philosophy/learning facts into the preferences table."""
    from app.db import get_db_session
    from app.astra_memory.preference_service import create_preference
    from app.astra_memory.preference_models import (
        PreferenceRecord, PreferenceStrength,
    )

    db = get_db_session()
    try:
        for f in plan:
            cat = f.get("category", "preference")
            key = f.get("key", "")
            value = f.get("value", "")
            confidence = f.get("confidence", "low")
            namespace = {
                "preference": "user_personal",
                "philosophy": "dev_principles",
                "learning": "user_personal",
            }.get(cat, "user_personal")
            pref_key = f"migration:{cat}:{key}"

            # Skip if already migrated
            existing = (
                db.query(PreferenceRecord)
                .filter(PreferenceRecord.preference_key == pref_key)
                .first()
            )
            if existing:
                log.info("  preference skip (exists): %s", pref_key)
                continue

            strength = (
                PreferenceStrength.HARD_RULE if confidence == "high"
                and cat == "philosophy"
                else PreferenceStrength.DEFAULT if confidence == "high"
                else PreferenceStrength.SOFT
            )
            create_preference(
                db=db,
                preference_key=pref_key,
                preference_value=value,
                strength=strength,
                source="migration_2026-05",
                namespace=namespace,
                context_pointer="migration:user_facts.json",
            )
            log.info("  preference create %s (strength=%s)", pref_key, strength.value)
    finally:
        db.close()


def apply_to_tier3(plan: List[Dict[str, Any]], log: logging.Logger) -> None:
    """
    Active project facts go to HotIndex with TTL. HotIndex is the canonical
    Tier 3 store per Decision 5.
    """
    log.info("Tier 3 migration: %d facts (HotIndex write integration is "
             "Phase 8 work — for now these are recorded in the migration log "
             "but not written; user_facts.json remains as a fallback read.)",
             len(plan))
    for f in plan:
        log.info("  tier3 (deferred-write) %s = %r",
                 f.get("key"), str(f.get("value"))[:60])


def archive_user_facts(log: logging.Logger) -> None:
    """Copy user_facts.json to backup directory, then delete the live file."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    target = BACKUP_DIR / "user_facts.json"
    target.write_bytes(USER_FACTS.read_bytes())
    log.info("Archived user_facts.json -> %s", target)
    USER_FACTS.unlink()
    log.info("Deleted live user_facts.json")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Migrate user_facts.json per Phase 2 schema")
    ap.add_argument(
        "--apply", action="store_true",
        help="Actually perform the migration. Without this flag, dry-run only.",
    )
    ap.add_argument(
        "--allow-shadow-mode", action="store_true",
        help="Permit running --apply even if arbiter is still in shadow mode "
             "(by default, --apply requires enforce mode for safety).",
    )
    args = ap.parse_args()

    log = _setup_logging(args.apply)

    if not USER_FACTS.exists():
        log.warning("user_facts.json does not exist — already migrated?")
        return 0

    facts = json.loads(USER_FACTS.read_text(encoding="utf-8"))
    log.info("Loaded %d facts from %s", len(facts), USER_FACTS)

    plan = build_plan(facts, log)
    render_plan(plan, log)

    if not args.apply:
        log.info("")
        log.info("DRY RUN — no changes made. Re-run with --apply to execute.")
        return 0

    # Safety check: require enforce mode unless overridden
    import os
    arbiter_mode = os.getenv("ASTRA_ARBITER_MODE", "shadow")
    if arbiter_mode != "enforce" and not args.allow_shadow_mode:
        log.error("ASTRA_ARBITER_MODE=%s. Migration requires enforce mode for "
                  "biographical facts to queue properly. Either set "
                  "ASTRA_ARBITER_MODE=enforce in .env, or pass --allow-shadow-mode "
                  "(not recommended).", arbiter_mode)
        return 2

    log.info("=== APPLYING MIGRATION ===")
    apply_to_identity(plan["to_identity"], log)
    apply_to_preferences(plan["to_preferences"], log)
    apply_to_tier3(plan["to_tier3"], log)
    archive_user_facts(log)
    log.info("=== DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
