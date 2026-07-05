# FILE: app/cost/cost_report.py
# Purpose: Cost rollup reporting — spend by stage, model, job and day from the cost ledger.
# Called-by: app.endpoints.cost_dashboard
# Depends-on: app.cost.cost_ledger
# Last-renovated: 2026-07-04 (Derek phase 1)
"""
Cost rollup reporting over data/cost_ledger.jsonl.

One boring function: rollup(by=..., days=...) -> per-key aggregates.
Unpriced records (cost_usd null — model missing from the pricing table)
are counted separately per bucket so under-reporting is visible, never
silently folded into $0.00.

Usage:
    from app.cost.cost_report import rollup
    rollup(by="stage", days=7)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

VALID_KEYS = ("stage", "model", "job", "day", "provider")


def _bucket_key(record: Dict[str, Any], by: str) -> str:
    """Derive the rollup bucket for one ledger record."""
    if by == "day":
        return str(record.get("timestamp", ""))[:10] or "unknown"
    if by == "job":
        return str(record.get("job_id") or "") or "(no job)"
    return str(record.get(by) or "unknown")


def rollup(
    by: str = "stage",
    days: Optional[int] = 7,
    job_id: Optional[str] = None,
    max_records: int = 200_000,
) -> Dict[str, Any]:
    """
    Aggregate ledger records into buckets.

    by:      "stage" | "model" | "job" | "day" | "provider"
    days:    look-back window (None = whole ledger, capped by max_records)
    job_id:  optional filter to a single job before bucketing

    Returns {"by", "days", "generated_at", "totals": {...}, "buckets": {key: {...}}}
    where each bucket carries records / prompt_tokens / completion_tokens /
    cost_usd (sum of priced records) / unpriced_records.
    """
    if by not in VALID_KEYS:
        raise ValueError(f"rollup by must be one of {VALID_KEYS}, got {by!r}")

    from app.cost.cost_ledger import read_records_since

    since = (
        datetime.now(timezone.utc) - timedelta(days=days)
        if days is not None
        else datetime(1970, 1, 1, tzinfo=timezone.utc)
    )
    records = read_records_since(since, max_records=max_records)

    buckets: Dict[str, Dict[str, Any]] = {}
    totals = {
        "records": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
        "unpriced_records": 0,
    }

    for r in records:
        if job_id and str(r.get("job_id") or "") != job_id:
            continue
        key = _bucket_key(r, by)
        b = buckets.setdefault(key, {
            "records": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
            "unpriced_records": 0,
        })
        cost = r.get("cost_usd")
        b["records"] += 1
        b["prompt_tokens"] += int(r.get("prompt_tokens") or 0)
        b["completion_tokens"] += int(r.get("completion_tokens") or 0)
        totals["records"] += 1
        totals["prompt_tokens"] += int(r.get("prompt_tokens") or 0)
        totals["completion_tokens"] += int(r.get("completion_tokens") or 0)
        if cost is None:
            b["unpriced_records"] += 1
            totals["unpriced_records"] += 1
        else:
            b["cost_usd"] = round(b["cost_usd"] + float(cost), 6)
            totals["cost_usd"] = round(totals["cost_usd"] + float(cost), 6)

    ordered = dict(
        sorted(buckets.items(), key=lambda kv: kv[1]["cost_usd"], reverse=True)
    )
    return {
        "by": by,
        "days": days,
        "job_id": job_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "totals": totals,
        "buckets": ordered,
    }


__all__ = ["rollup", "VALID_KEYS"]
