# FILE: app/finance/services/work_folder_logger.py
# Purpose: Deterministic folder logger for end-of-day delivery screenshots.
# Called-by: app.debug.gemini_finance_tools, app.finance.services.work_upload_logger
# Depends-on: app.db, app.drive.file_utils, app.finance.models_workday, app.llm.gemini_vision (+1 more)
# Last-renovated: 2026-06-11
"""
Deterministic folder logger for end-of-day delivery screenshots.

The printer, in code. Given a folder of shift screenshots, it walks them ONE
date at a time -- parse the date and finish time from the filename, read the
parcel count off that one image, write that day's row to the work ledger,
then move on. It never holds the whole batch in context and never asks a chat
model to decide what to do per image.

Why this exists:
    A chat model told to "log all the screenshots" batches twenty-odd
    read_image calls, holds every result, then narrates a summary instead of
    writing any rows -- and on a retry re-reads the whole folder because
    nothing was written to resume from. This loop removes the model from the
    per-image decision: the only model call is a bounded, fast vision read of
    a single image for the parcel number.

    Each written row IS the checkpoint. Dates already complete in the ledger
    are skipped, so a re-run resumes rather than starting over or double-
    writing.

No Chinese whispers:
    There is no model-to-model handoff of meaning here. Vision returns a
    number; deterministic code parses it; deterministic code writes it.
    Nothing re-interprets a description. Vision stays on Gemini, fast/flash
    tier -- quick and cheap, and reading one integer needs nothing heavier.

Filenames are standard Android screenshots: Screenshot_YYYYMMDD-HHMMSS.png
-> date 2026-04-09, finish time 19:32 (the capture time is the finish time,
the user's rule). When a date has more than one screenshot, the LATEST one
wins -- it is the genuine end-of-day shot, with the final parcel total.

Entry point: log_work_folder(), wired as the `log_work_folder` tool so the
chat model triggers the whole pass with a single call.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Screenshot_20260409-193222.png  ->  ('20260409', '193222')
_SCREENSHOT_RE = re.compile(r"(\d{8})[-_](\d{6})")

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif"}

# The one field that genuinely needs OCR. Everything else is in the filename.
_PARCEL_QUESTION = (
    "This is the end-of-day summary screen from a parcel delivery driver's "
    "app. How many parcels were delivered/completed today? Reply with ONLY "
    "the number (e.g. 120). If you genuinely cannot find a delivered/"
    "completed count, reply exactly NONE."
)


def _parse_filename(name: str) -> "Optional[tuple[date, str]]":
    """'Screenshot_20260409-193222.png' -> (date(2026,4,9), '19:32'). None if unmatched."""
    m = _SCREENSHOT_RE.search(name)
    if not m:
        return None
    ymd, hms = m.group(1), m.group(2)
    try:
        d = datetime.strptime(ymd, "%Y%m%d").date()
    except ValueError:
        return None
    hh, mm = hms[0:2], hms[2:4]
    if not (hh.isdigit() and mm.isdigit() and int(hh) < 24 and int(mm) < 60):
        return None
    return d, f"{hh}:{mm}"


def _parse_parcels(answer: str) -> "Optional[int]":
    """First integer in the vision answer (1-3 digits). None for NONE / absent."""
    if not answer or answer.strip().upper().startswith("NONE"):
        return None
    m = re.search(r"\d{1,3}", answer.replace(",", ""))
    return int(m.group(0)) if m else None


def _pick_latest_per_date(images):
    """date -> (finish_time, path) keeping the LATEST screenshot for each date.
    Two shots on one day means the later one is the real end-of-day total."""
    chosen = {}
    for img in images:
        parsed = _parse_filename(img.name)
        if parsed is None:
            continue
        d, finish = parsed
        prev = chosen.get(d)
        if prev is None or finish > prev[0]:
            chosen[d] = (finish, img)
    return chosen


def _resolve_work_folder(folder: "Optional[str]") -> "Optional[Path]":
    """Explicit path wins; otherwise <Documents>/Work."""
    if folder:
        p = Path(folder)
        return p if p.is_dir() else None
    try:
        from app.drive.file_utils import get_category_paths
        docs = get_category_paths().get("documents")
        if docs:
            work = Path(docs) / "Work"
            if work.is_dir():
                return work
    except Exception as exc:
        logger.warning("[work_folder_logger] folder resolve failed: %s", exc)
    return None


def _logged_dates():
    """Dates already complete in the ledger -- the skip set / resume checkpoint."""
    from app.db import SessionLocal
    from app.finance.models_workday import WorkDay
    db = SessionLocal()
    try:
        rows = (db.query(WorkDay.work_date)
                  .filter(WorkDay.status == "complete").all())
        return {r[0] for r in rows if r[0] is not None}
    finally:
        db.close()


def _read_parcels(image_path: Path):
    """Fast Gemini vision read of the parcel count. Returns (parcels, error)."""
    try:
        from app.llm.gemini_vision import ask_about_image
        res = ask_about_image(
            image_source=str(image_path),
            user_question=_PARCEL_QUESTION,
            tier="fast",
        )
    except Exception as exc:
        return None, f"vision call failed: {exc}"
    if res.get("error"):
        return None, f"vision error: {res['error']}"
    parcels = _parse_parcels(res.get("answer", ""))
    if parcels is None:
        return None, "no parcel count read from image"
    if parcels <= 0:
        return None, "read 0 parcels (likely a misread) -- left for manual check"
    return parcels, None


async def log_work_folder(
    *,
    folder: "Optional[str]" = None,
    start_time: str = "10:00",
    miles: "Optional[float]" = 120.0,
    rate_per_parcel: "Optional[float]" = None,
    overwrite: bool = False,
) -> dict:
    """Walk a folder of end-of-day screenshots one date at a time, writing a
    work-ledger row per new day. Re-runnable; already-logged dates are skipped.
    Returns a summary dict (counts + per-day lines + any failures)."""
    work_dir = _resolve_work_folder(folder)
    if work_dir is None:
        return {"ok": False, "error": (
            "Couldn't find the work screenshots folder -- expected a Work "
            "folder under Documents, or pass an explicit folder path."
        )}

    images = [p for p in work_dir.iterdir()
              if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
    by_date = _pick_latest_per_date(images)
    if not by_date:
        return {"ok": False, "error": (
            f"No dated screenshots (Screenshot_YYYYMMDD-HHMMSS) found in {work_dir}."
        )}

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

        res = await finish_work_day_handler(args, context=None)   # write = checkpoint
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
        "folder": str(work_dir),
        "dates_seen": len(by_date),
        "logged": logged,
        "skipped": skipped,
        "failed": failed,
    }

