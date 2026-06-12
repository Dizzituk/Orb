# FILE: app/finance/services/work_upload_logger.py
# Purpose: Deterministic logger for delivery screenshots uploaded from the phone.
# Called-by: app.debug.gemini_finance_tools
# Depends-on: app.bridge.uploads_store, app.finance.services.work_folder_logger, app.tools.finance_tools
# Last-renovated: 2026-06-11
"""
Deterministic logger for delivery screenshots uploaded from the phone.

Sibling of work_folder_logger: same printer-in-code loop, different source.
Where the folder logger walks a Work folder on disk, this one walks the most
recent image uploads in the bridge upload ledger -- the shots the user just
sent from the Astra Bridge app -- and writes one work-ledger row per date.

The shift date and finish time come from the ORIGINAL screenshot filename
(Screenshot_YYYYMMDD-HHMMSS), which the upload pipeline now preserves. We do
NOT infer the date from upload time (a shot sent days later would log to the
wrong day) and we do NOT ask a chat model to interpret anything: vision reads
one integer (the parcel count), deterministic code writes the row. A shot
whose name carries no parseable date is flagged, never guessed.

Re-runnable: dates already complete in the ledger are skipped, so saying
"log it" twice will not double-write.

Entry point: log_uploaded_work_screenshots(), wired as a chat tool so the
model triggers the pass with one call after the user uploads screenshot(s).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


async def log_uploaded_work_screenshots(
    *,
    count: int = 1,
    start_time: str = "10:00",
    miles: "Optional[float]" = 120.0,
    rate_per_parcel: "Optional[float]" = None,
    overwrite: bool = False,
) -> dict:
    """Log the most recent `count` uploaded delivery screenshots.

    Dates/finish times are parsed from each upload's preserved original
    filename; the parcel count is read with one fast vision call; the row is
    written via finish_work_day_handler. Already-logged dates are skipped.
    Returns a summary dict (logged / skipped / failed / undated).
    """
    from app.bridge.uploads_store import recent_records
    from app.finance.services.work_folder_logger import (
        _parse_filename, _logged_dates, _read_parcels,
    )

    recs = recent_records(limit=max(1, int(count or 1)), image_only=True)
    if not recs:
        return {"ok": False, "error": "No recent image uploads found to log."}

    # date -> (finish_time, path); latest finish time wins per date.
    by_date: dict = {}
    undated: list = []
    for r in recs:
        name = r.original_filename or Path(r.path).name
        parsed = _parse_filename(name)
        if parsed is None:
            undated.append(name or r.id)
            continue
        d, finish = parsed
        prev = by_date.get(d)
        if prev is None or finish > prev[0]:
            by_date[d] = (finish, Path(r.path))

    if not by_date:
        return {"ok": False, "error": (
            "Couldn't read a shift date (Screenshot_YYYYMMDD-HHMMSS) from the "
            "uploaded file name(s). Re-upload so the original screenshot name "
            "is kept, or log via the Work folder."
        ), "undated": undated}

    done = set() if overwrite else _logged_dates()

    from app.tools.finance_tools import finish_work_day_handler

    end_odo = float(miles) if miles is not None else None
    logged, skipped, failed = [], [], []

    for work_date in sorted(by_date):
        finish_time, img = by_date[work_date]
        if work_date in done:
            skipped.append(work_date.isoformat())
            continue

        parcels, err = _read_parcels(img)            # the one model call
        if parcels is None:
            failed.append({"date": work_date.isoformat(), "reason": err})
            continue

        args = {
            "work_date": work_date.isoformat(),
            "start_time": start_time,
            "end_time": finish_time,
            "parcels": parcels,
            "screenshot_path": str(img),
        }
        if miles is not None:
            args["start_odometer"] = 0.0
            args["end_odometer"] = end_odo
        if rate_per_parcel is not None:
            args["rate_per_parcel"] = rate_per_parcel

        res = await finish_work_day_handler(args, context=None)
        if res.get("ok"):
            done.add(work_date)
            logged.append({
                "date": work_date.isoformat(),
                "finish": finish_time,
                "parcels": parcels,
                "earnings": res.get("gross_earnings"),
            })
        else:
            failed.append({"date": work_date.isoformat(),
                           "reason": res.get("error", "write failed")})

    return {
        "ok": True,
        "source": "uploads",
        "dates_seen": len(by_date),
        "logged": logged,
        "skipped": skipped,
        "failed": failed,
        "undated": undated,
    }