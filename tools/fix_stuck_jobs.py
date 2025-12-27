# FILE: tools/fix_stuck_jobs.py
"""Mark specific jobs as FAILED in Orb's SQLite DB.

Use this when a previous bug left jobs stuck in RUNNING.
Safe: only updates rows where state == 'running' and id is in STUCK.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = r"D:\Orb\data\orb_memory.db"

# Replace these IDs as needed
STUCK = [
    "1305f2f8-617b-435d-aa61-e785f89ef8da",
    "939c8a6e-445b-4e69-9f2d-948adf6e3b29",
]


def main() -> int:
    if not os.path.exists(DB_PATH):
        raise SystemExit(f"DB not found: {DB_PATH}")

    # timezone-aware UTC timestamp (avoids utcnow() deprecation in Python 3.13+)
    now = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        q = """
        UPDATE jobs
        SET state=?,
            error_type=?,
            error_message=?,
            completed_at=?,
            duration_seconds=?
        WHERE id=? AND state='running'
        """

        updated = 0
        for jid in STUCK:
            cur.execute(
                q,
                (
                    "failed",
                    "internal_error",
                    "Marked failed: stuck RUNNING from earlier DB serialization bug.",
                    now,
                    0.0,
                    jid,
                ),
            )
            updated += int(cur.rowcount or 0)

        conn.commit()
    finally:
        conn.close()

    print(f"Updated rows: {updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
