#!/usr/bin/env python
# FILE: scripts/backfill_topic_tags_2026-05.py
"""
Backfill: re-tag every HotIndex record using the new topic_tagger.

The previous tagger only produced six code-related tags
(code/python/testing/architecture/documentation/debugging), so existing
records are tagged badly even though their content is fine. Plus the
`entities` column has been written as the string 'null' for every record
since the schema added it.

This script:
  - reads every astra_hot_index record
  - rebuilds tags and entities from the record's title + one_liner
  - updates the row in place
  - preserves all other fields (priority, cost, timestamps, etc.)

USAGE:
    # Dry run (default — prints counts, makes NO changes)
    python scripts/backfill_topic_tags_2026-05.py

    # Live run
    python scripts/backfill_topic_tags_2026-05.py --apply

    # Limit (handy for testing on a subset)
    python scripts/backfill_topic_tags_2026-05.py --limit 100 --apply

Idempotent — safe to re-run.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.db import get_db_session
from app.astra_memory.preference_models import HotIndex
from app.astra_memory.topic_tagger import extract_tags, extract_entities


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-tag HotIndex records.")
    ap.add_argument("--apply", action="store_true",
                    help="Actually write changes (default: dry-run).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N records (for testing).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s")
    log = logging.getLogger("backfill_tags")

    db = get_db_session()
    try:
        q = db.query(HotIndex)
        if args.limit:
            q = q.limit(args.limit)
        records = q.all()
        log.info("Loaded %d HotIndex records", len(records))

        tag_changes = 0
        entity_changes = 0
        new_tags_counter: Counter = Counter()
        new_entities_counter: Counter = Counter()

        for r in records:
            # Combine title + one_liner + bullets for tagger
            text_parts = [r.title or "", r.one_liner or ""]
            if r.bullets_5:
                if isinstance(r.bullets_5, list):
                    text_parts.extend(r.bullets_5)
                elif isinstance(r.bullets_5, str):
                    text_parts.append(r.bullets_5)
            combined = "\n".join(p for p in text_parts if p)

            new_tags = extract_tags(combined)
            new_entities = extract_entities(combined)

            old_tags = r.tags or []
            old_entities = r.entities if isinstance(r.entities, list) else []

            if sorted(new_tags) != sorted(old_tags):
                tag_changes += 1
                for t in new_tags:
                    new_tags_counter[t] += 1
            if sorted(new_entities) != sorted(old_entities):
                entity_changes += 1
                for e in new_entities:
                    new_entities_counter[e] += 1

            if args.apply:
                r.tags = new_tags
                r.entities = new_entities  # [] when empty (NOT None / 'null')

        log.info("")
        log.info("=== SUMMARY ===")
        log.info("  records scanned:    %d", len(records))
        log.info("  tag-set changes:    %d", tag_changes)
        log.info("  entity-set changes: %d", entity_changes)
        log.info("")
        log.info("  top 15 NEW tag distributions (post-backfill):")
        for tag, n in new_tags_counter.most_common(15):
            log.info("    %4d  %s", n, tag)
        log.info("")
        log.info("  top 15 NEW entity distributions (post-backfill):")
        for ent, n in new_entities_counter.most_common(15):
            log.info("    %4d  %s", n, ent)

        if args.apply:
            log.info("")
            log.info("Committing changes...")
            db.commit()
            log.info("Done.")
        else:
            log.info("")
            log.info("DRY RUN \u2014 no changes written. Re-run with --apply to commit.")

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
