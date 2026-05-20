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

            # Unified OCR pipeline (Tesseract → Gemini Flash → OpenAI)
            from app.finance.services.ocr_pipeline import extract_from_bytes
            from app.finance.services.daily_log_ingest import save_ocr_result_as_log

            ocr_result = await extract_from_bytes(
                file_bytes, mime_type=mime, filename=filename,
            )

            # Drive filename often carries the real date (YYYYMMDD_HHMMSS_...).
            # If the OCR missed the date, or the filename date is newer/cleaner,
            # prefer the filename date. This was the old behaviour too.
            import re as _re
            fname_date_m = _re.search(r'(\d{4})(\d{2})(\d{2})', filename)
            if fname_date_m and ocr_result.success:
                from datetime import date as _date
                try:
                    fname_date = _date(
                        int(fname_date_m.group(1)),
                        int(fname_date_m.group(2)),
                        int(fname_date_m.group(3)),
                    )
                    if not ocr_result.work_date:
                        ocr_result.work_date = fname_date
                except ValueError:
                    pass

            if ocr_result.success and ocr_result.work_date:
                log = save_ocr_result_as_log(db, ocr_result)
                record.ocr_success = True
                record.work_log_id = log.id if log else None
                results.append({
                    "filename": filename,
                    "success": True,
                    "work_date": str(ocr_result.work_date),
                    "deliveries": ocr_result.delivery_count,
                    "earnings": ocr_result.gross_earnings,
                    "work_log_id": log.id if log else None,
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







