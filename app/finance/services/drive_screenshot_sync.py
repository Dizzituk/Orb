# FILE: app/finance/services/drive_screenshot_sync.py
"""
Google Drive folder sync for Yodel earnings screenshots.

Watches a configured Drive folder for new image files,
downloads them, runs OCR, and creates daily work log entries.
Tracks processed files to avoid duplicates.

Flow:
1. User links a Drive folder (from phone camera uploads)
2. System lists images in that folder
3. For each unprocessed image: download → OCR → create work log
4. Mark as processed in local DB
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from app.db import Base

logger = logging.getLogger(__name__)

ALLOWED_MIMES = {
    "image/png",
    "image/jpeg",
    "image/webp",
}


# ── Tracking model ────────────────────────────────────────

class ProcessedScreenshot(Base):
    """Tracks which Drive files have already been processed."""
    __tablename__ = "finance_processed_screenshots"

    id = Column(Integer, primary_key=True)
    drive_file_id = Column(String(200), unique=True, nullable=False, index=True)
    drive_filename = Column(String(500))
    drive_folder_id = Column(String(200))
    processed_at = Column(DateTime, default=datetime.utcnow)
    ocr_success = Column(Boolean, default=False)
    work_log_id = Column(Integer, nullable=True)
    error_message = Column(String(1000), nullable=True)


# ── Config model ──────────────────────────────────────────

class ScreenshotFolderConfig(Base):
    """Stores the linked Drive folder for screenshot sync."""
    __tablename__ = "finance_screenshot_folder_config"

    id = Column(Integer, primary_key=True)
    folder_id = Column(String(200), nullable=False)
    folder_name = Column(String(500))
    linked_at = Column(DateTime, default=datetime.utcnow)
    auto_sync = Column(Boolean, default=True)
    last_sync = Column(DateTime, nullable=True)
    sync_count = Column(Integer, default=0)


# ── Core sync logic ───────────────────────────────────────

async def get_linked_folder(db: Session) -> Optional[dict]:
    """Get the currently linked Drive folder config."""
    config = db.query(ScreenshotFolderConfig).first()
    if not config:
        return None
    return {
        "folder_id": config.folder_id,
        "folder_name": config.folder_name,
        "linked_at": str(config.linked_at),
        "auto_sync": config.auto_sync,
        "last_sync": str(config.last_sync) if config.last_sync else None,
        "sync_count": config.sync_count,
    }


async def link_folder(
    db: Session, folder_id: str, folder_name: str
) -> dict:
    """Link a Drive folder for screenshot sync."""
    # Remove any existing config
    db.query(ScreenshotFolderConfig).delete()
    db.commit()

    config = ScreenshotFolderConfig(
        folder_id=folder_id,
        folder_name=folder_name,
    )
    db.add(config)
    db.commit()

    logger.info("[screenshot_sync] Linked folder: %s (%s)", folder_name, folder_id)
    return {
        "linked": True,
        "folder_id": folder_id,
        "folder_name": folder_name,
    }


async def unlink_folder(db: Session) -> dict:
    """Remove the linked Drive folder."""
    count = db.query(ScreenshotFolderConfig).delete()
    db.commit()
    return {"unlinked": True, "removed": count}


async def sync_screenshots(db: Session) -> dict:
    """Scan linked Drive folder for new screenshots and process them.

    Returns summary of what was found and processed.
    """
    from app.finance.services.drive_browser_service import (
        list_folder_contents,
        download_file_bytes,
    )
    from app.finance.services.screenshot_ocr_service import (
        save_screenshot,
        extract_via_llm,
    )

    config = db.query(ScreenshotFolderConfig).first()
    if not config:
        return {"error": "No folder linked. Connect a Drive folder first."}

    # List images in the folder
    try:
        contents = list_folder_contents(
            config.folder_id,
            file_types=["png", "jpg", "jpeg", "webp"],
        )
    except Exception as e:
        logger.error("[screenshot_sync] Failed to list folder: %s", e)
        return {"error": f"Could not access Drive folder: {e}"}

    files = contents.get("files", [])
    if not files:
        return {
            "folder": config.folder_name,
            "found": 0,
            "new": 0,
            "processed": 0,
            "errors": 0,
            "results": [],
        }

    # Filter out already-processed files
    processed_ids = {
        r[0] for r in
        db.query(ProcessedScreenshot.drive_file_id).all()
    }
    new_files = [f for f in files if f["id"] not in processed_ids]

    results = []
    error_count = 0

    for file_info in new_files:
        file_id = file_info["id"]
        filename = file_info["name"]
        mime = file_info.get("mimeType", "image/png")

        record = ProcessedScreenshot(
            drive_file_id=file_id,
            drive_filename=filename,
            drive_folder_id=config.folder_id,
        )

        try:
            # Download from Drive
            file_bytes = download_file_bytes(file_id)
            if not file_bytes:
                record.error_message = "Empty file downloaded"
                record.ocr_success = False
                db.add(record)
                db.commit()
                error_count += 1
                continue

            # OCR: deterministic Tesseract first (also saves locally), API fallback
            from app.finance.services.screenshot_ocr_deterministic import extract_from_image
            from app.finance.services.screenshot_ocr_service import ScreenshotOCRResult

            local_path = save_screenshot(file_bytes, filename)
            det_result = extract_from_image(local_path)

            if det_result.is_valid and det_result.confidence >= 70:
                # Deterministic OCR succeeded — map to expected format
                # Override OCR date with filename date (most reliable)
                import re as _re
                fname_date_m = _re.search(r'(\d{4})(\d{2})(\d{2})', filename)
                if fname_date_m:
                    from datetime import date as _date
                    try:
                        fname_date = _date(
                            int(fname_date_m.group(1)),
                            int(fname_date_m.group(2)),
                            int(fname_date_m.group(3)),
                        )
                        det_result.work_date = fname_date
                    except ValueError:
                        pass

                ocr_result = ScreenshotOCRResult(
                    success=True,
                    work_date=det_result.work_date,
                    tour_id=det_result.tour_id,
                    user_id=det_result.user_id,
                    delivery_count=det_result.delivery_count,
                    collections=det_result.collections,
                    stops=det_result.attempted_stops,
                    attempted=det_result.attempted_stops,
                    done=det_result.deliveries or det_result.delivery_count,
                    failed_deliveries=det_result.not_attempted,
                    gross_earnings=det_result.gross_earnings,
                    route_area=None,
                    message=f"Deterministic OCR ({det_result.confidence}% confidence)",
                )
                logger.info(
                    "[screenshot_sync] Deterministic OCR: %s (%s%% confidence, %d fields)",
                    filename, det_result.confidence, len(det_result.fields_extracted),
                )
            else:
                # Fall back to API-based vision OCR
                logger.info(
                    "[screenshot_sync] Deterministic OCR insufficient (%s%% confidence), "
                    "falling back to API for %s",
                    det_result.confidence, filename,
                )
                ocr_result = await extract_via_llm(file_bytes, mime)

            if ocr_result.success:
                # Create work log entry
                work_log_id = await _create_work_log_from_ocr(db, ocr_result)
                record.ocr_success = True
                record.work_log_id = work_log_id
                results.append({
                    "filename": filename,
                    "success": True,
                    "work_date": str(ocr_result.work_date),
                    "deliveries": ocr_result.delivery_count,
                    "earnings": ocr_result.gross_earnings,
                    "work_log_id": work_log_id,
                })
            else:
                record.ocr_success = False
                record.error_message = ocr_result.message
                error_count += 1
                results.append({
                    "filename": filename,
                    "success": False,
                    "message": ocr_result.message,
                })

        except Exception as e:
            logger.error("[screenshot_sync] Failed processing %s: %s", filename, e)
            record.ocr_success = False
            record.error_message = str(e)[:1000]
            error_count += 1
            results.append({
                "filename": filename,
                "success": False,
                "message": str(e),
            })

        db.add(record)
        db.commit()

    # Update sync timestamp
    config.last_sync = datetime.utcnow()
    config.sync_count += len(new_files)
    db.commit()

    return {
        "folder": config.folder_name,
        "found": len(files),
        "already_processed": len(processed_ids),
        "new": len(new_files),
        "processed": len([r for r in results if r.get("success")]),
        "errors": error_count,
        "results": results,
    }


async def _create_work_log_from_ocr(db: Session, ocr) -> int:
    """Create a DailyWorkLog entry from OCR extraction."""
    from app.finance.models import DailyWorkLog

    # Check for duplicate date
    existing = db.query(DailyWorkLog).filter(
        DailyWorkLog.work_date == ocr.work_date
    ).first()

    if existing:
        logger.info(
            "[screenshot_sync] Work log already exists for %s, updating",
            ocr.work_date,
        )
        existing.tour_id = ocr.tour_id or existing.tour_id
        existing.delivery_count = ocr.delivery_count or existing.delivery_count
        existing.collections = ocr.collections or existing.collections
        existing.stops = ocr.stops or existing.stops
        existing.attempted = ocr.attempted or existing.attempted
        existing.done = ocr.done or existing.done
        existing.failed_deliveries = ocr.failed_deliveries or existing.failed_deliveries
        existing.gross_earnings = ocr.gross_earnings or existing.gross_earnings
        existing.route_area = ocr.route_area or existing.route_area
        # source: drive_sync
        db.commit()
        return existing.id

    total_parcels = ocr.delivery_count + ocr.collections
    rate = (ocr.gross_earnings / total_parcels) if total_parcels > 0 else 0.0

    # Compute HMRC tax year (6 Apr - 5 Apr)
    wd = ocr.work_date
    if wd.month > 4 or (wd.month == 4 and wd.day >= 6):
        tax_year = f"{wd.year}/{str(wd.year + 1)[-2:]}"
    else:
        tax_year = f"{wd.year - 1}/{str(wd.year)[-2:]}"

    log = DailyWorkLog(
        work_date=ocr.work_date,
        tour_id=ocr.tour_id,
        user_id=ocr.user_id,
        delivery_count=ocr.delivery_count,
        collections=ocr.collections,
        stops=ocr.stops,
        attempted=ocr.attempted,
        done=ocr.done,
        failed_deliveries=ocr.failed_deliveries,
        gross_earnings=ocr.gross_earnings,
        route_area=ocr.route_area,
        rate_per_parcel=round(rate, 2),
        total_parcels=total_parcels,
        tax_year=tax_year,
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    logger.info(
        "[screenshot_sync] Created work log #%d for %s (%d deliveries, £%.2f)",
        log.id, ocr.work_date, ocr.delivery_count, ocr.gross_earnings,
    )
    return log.id


async def get_sync_history(db: Session, limit: int = 20) -> list:
    """Get recent processed screenshots for the sync log."""
    records = (
        db.query(ProcessedScreenshot)
        .order_by(ProcessedScreenshot.processed_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "filename": r.drive_filename,
            "processed_at": str(r.processed_at),
            "success": r.ocr_success,
            "work_log_id": r.work_log_id,
            "error": r.error_message,
        }
        for r in records
    ]







