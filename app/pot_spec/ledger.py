# FILE: app/pot_spec/ledger.py
"""Append-only ledger for deterministic replay.

Each event is a single JSON object written as one line (ndjson).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any


def _utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def append_event(job_artifact_root: str, job_id: str, event: dict[str, Any]) -> str:
    """Append event to jobs/<job_id>/ledger/events.ndjson.

    Returns the absolute path written.
    """
    ledger_dir = os.path.join(job_artifact_root, job_id, "ledger")
    os.makedirs(ledger_dir, exist_ok=True)
    path = os.path.join(ledger_dir, "events.ndjson")

    record = dict(event)
    record.setdefault("ts", _utc_iso())

    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")

    return path
