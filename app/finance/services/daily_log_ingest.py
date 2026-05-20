# FILE: app/finance/services/daily_log_ingest.py
"""
Convert OCR extraction results into DailyWorkLog rows.

Used by every screenshot ingestion surface (upload endpoint, Drive
sync, chat tool). Previously each surface rolled its own version and
got at least one thing wrong (tax_year format, field names, etc).

Pure orchestration \u2014 no network I/O, no OCR logic. Keeps
ocr_pipeline.py focused on "image bytes \u2192 structured data" and this
module focused on "structured data \u2192 DB row".
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from app.finance.models import DailyWorkLog
from app.finance.schemas import ScreenshotOCRResult
from app.finance.utils.tax_year import tax_year_for

logger = logging.getLogger(__name__)


def save_ocr_result_as_log(
    db: Session, ocr: ScreenshotOCRResult, screenshot_path: Optional[str] = None,
) -> Optional[DailyWorkLog]:
    """Upsert a DailyWorkLog from a successful OCR extraction.

    Returns the created/updated log, or None if the OCR result doesn't
    have enough data to persist (missing work_date is the dealbreaker).

    If a log already exists for that date, fields from the new OCR
    overwrite only where the new value is non-empty. This lets a user
    re-upload a cleaner screenshot without losing earlier data.
    """
    if not ocr or not ocr.success or not ocr.work_date:
        logger.info(
            "[daily_log_ingest] Skipping: success=%s, date=%s",
            getattr(ocr, "success", None),
            getattr(ocr, "work_date", None),
        )
        return None

    delivery_count = int(ocr.delivery_count or 0)
    collections = int(ocr.collections or 0)
    total_parcels = delivery_count + collections
    earnings = float(ocr.gross_earnings or 0.0)
    rate = round(earnings / total_parcels, 2) if total_parcels > 0 else 0.0
    tax_year = tax_year_for(ocr.work_date)

    existing = (
        db.query(DailyWorkLog)
        .filter(DailyWorkLog.work_date == ocr.work_date)
        .first()
    )

    if existing:
        # Only overwrite where we have a better value
        if ocr.tour_id:
            existing.tour_id = ocr.tour_id
        if ocr.user_id:
            existing.user_id = ocr.user_id
        if delivery_count:
            existing.delivery_count = delivery_count
        if collections:
            existing.collections = collections
        if ocr.stops:
            existing.stops = int(ocr.stops)
        if ocr.attempted:
            existing.attempted = int(ocr.attempted)
        if ocr.done:
            existing.done = int(ocr.done)
        if ocr.failed_deliveries:
            existing.failed_deliveries = int(ocr.failed_deliveries)
        if earnings:
            existing.gross_earnings = earnings
        if total_parcels:
            existing.total_parcels = total_parcels
            if rate:
                existing.rate_per_parcel = rate
                existing.per_delivery = (
                    round(earnings / delivery_count, 2) if delivery_count > 0 else 0.0
                )
        if ocr.route_area:
            existing.route_area = ocr.route_area
        if screenshot_path:
            existing.screenshot_path = screenshot_path
        existing.ocr_processed = True
        db.commit()
        db.refresh(existing)
        logger.info(
            "[daily_log_ingest] Updated existing log #%d for %s",
            existing.id, ocr.work_date,
        )
        return existing

    log = DailyWorkLog(
        work_date=ocr.work_date,
        tour_id=ocr.tour_id,
        user_id=ocr.user_id,
        delivery_count=delivery_count,
        collections=collections,
        stops=int(ocr.stops or 0),
        attempted=int(ocr.attempted or 0),
        done=int(ocr.done or 0),
        failed_deliveries=int(ocr.failed_deliveries or 0),
        gross_earnings=earnings,
        route_area=ocr.route_area,
        rate_per_parcel=rate,
        total_parcels=total_parcels,
        per_delivery=round(earnings / delivery_count, 2) if delivery_count > 0 else 0.0,
        screenshot_path=screenshot_path,
        ocr_processed=True,
        tax_year=tax_year,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    logger.info(
        "[daily_log_ingest] Created log #%d for %s (%d deliveries, \u00a3%.2f)",
        log.id, ocr.work_date, log.delivery_count, log.gross_earnings,
    )
    return log
