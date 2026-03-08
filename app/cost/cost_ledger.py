# FILE: app/overwatcher/cost_ledger.py
"""
Persistent cost ledger for ASTRA.

Append-only JSONL file that records every LLM call's cost data.
Survives restarts. Used for daily/monthly aggregation and the
cost dashboard endpoint.

File location: data/cost_ledger.jsonl
Rotation: When file exceeds 100MB, rotated to cost_ledger.{date}.jsonl

Usage:
    from app.cost.cost_ledger import append_record, read_records_since

    append_record(record_dict)
    records = read_records_since(start_datetime)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# =========================================================================
# Configuration
# =========================================================================

LEDGER_DIR = os.getenv("ORB_COST_LEDGER_DIR", "data")
LEDGER_FILENAME = "cost_ledger.jsonl"
MAX_LEDGER_SIZE_BYTES = 100 * 1024 * 1024  # 100MB rotation threshold

_write_lock = threading.Lock()


def _ledger_path() -> Path:
    """Get the path to the cost ledger file."""
    return Path(LEDGER_DIR) / LEDGER_FILENAME


def _ensure_dir() -> None:
    """Ensure the ledger directory exists."""
    Path(LEDGER_DIR).mkdir(parents=True, exist_ok=True)


# =========================================================================
# Write
# =========================================================================

def append_record(record: Dict[str, Any]) -> bool:
    """
    Append a cost record to the ledger.

    Thread-safe. Returns True on success, False on failure.
    Never raises — cost tracking must not break the pipeline.
    """
    try:
        _ensure_dir()
        path = _ledger_path()

        # Check rotation
        if path.exists() and path.stat().st_size > MAX_LEDGER_SIZE_BYTES:
            _rotate_ledger(path)

        line = json.dumps(record, default=str) + "\n"

        with _write_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)

        return True

    except Exception as e:
        logger.error("[cost_ledger] Failed to write record: %s", e)
        return False


def _rotate_ledger(path: Path) -> None:
    """Rotate the ledger file by adding a date suffix."""
    try:
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rotated = path.parent / f"cost_ledger.{date_str}.jsonl"
        shutil.move(str(path), str(rotated))
        logger.info("[cost_ledger] Rotated to %s", rotated.name)
    except Exception as e:
        logger.error("[cost_ledger] Rotation failed: %s", e)


# =========================================================================
# Read
# =========================================================================

def read_records_since(
    since: datetime,
    max_records: int = 50_000,
) -> List[Dict[str, Any]]:
    """
    Read cost records from the ledger since a given datetime.

    Reads from the end of the file for efficiency.
    Returns records in chronological order.
    """
    path = _ledger_path()
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    since_iso = since.isoformat()

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    ts = record.get("timestamp", "")
                    if ts >= since_iso:
                        records.append(record)
                        if len(records) >= max_records:
                            break
                except json.JSONDecodeError:
                    continue

    except Exception as e:
        logger.error("[cost_ledger] Failed to read records: %s", e)

    return records


def read_all_today() -> List[Dict[str, Any]]:
    """Read all records from today (UTC)."""
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    return read_records_since(today_start)


def read_all_this_month() -> List[Dict[str, Any]]:
    """Read all records from this calendar month (UTC)."""
    now = datetime.now(timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return read_records_since(month_start)


__all__ = [
    "append_record",
    "read_records_since",
    "read_all_today",
    "read_all_this_month",
]
