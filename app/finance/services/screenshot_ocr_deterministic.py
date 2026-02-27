# FILE: app/finance/services/screenshot_ocr_deterministic.py
"""
Deterministic OCR for Yodel delivery screenshots.

Uses Tesseract (local, free, no API) + regex pattern matching.
Multi-pass preprocessing handles both clean and damaged screenshots.
Falls back to None if minimum fields can't be extracted (caller
uses API vision as fallback).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

REQUIRED_FIELDS = {"tour_date", "delivery_count"}
DESIRED_FIELDS = {
    "tour_date", "user_id", "tour_id", "delivery_count",
    "attempted_stops", "start_time", "end_time", "duration",
}


@dataclass
class YodelOCRResult:
    """Extracted data from a Yodel Tour Summary screenshot."""
    work_date: Optional[date] = None
    user_id: Optional[str] = None
    tour_id: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_hours: Optional[float] = None
    delivery_count: int = 0
    attempted_stops: int = 0
    completed_stops: int = 0
    collections: int = 0
    deliveries: int = 0
    stores: int = 0
    lockers: int = 0
    not_attempted: int = 0
    failed_deliveries: int = 0
    gross_earnings: float = 0.0
    raw_text: str = ""
    confidence: float = 0.0
    method: str = "tesseract"
    fields_extracted: list[str] = field(default_factory=list)
    fields_missing: list[str] = field(default_factory=list)
    is_valid: bool = False


# ── Multi-pass OCR extraction ─────────────────────────────

def extract_from_image(image_path: str | Path) -> YodelOCRResult:
    """Extract Yodel tour data using multi-pass Tesseract OCR.
    
    Runs multiple preprocessing strategies and merges the best fields
    from each pass. This handles both clean and damaged screenshots.
    """
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    try:
        img = Image.open(image_path)
    except Exception as e:
        logger.error("[ocr_deterministic] Failed to open image: %s", e)
        result = YodelOCRResult()
        result.fields_missing = list(DESIRED_FIELDS)
        return result

    gray = img.convert("L")

    # Define preprocessing passes
    passes = []

    # Pass 1: High contrast + sharpen (best for clean screenshots)
    enhancer = ImageEnhance.Contrast(gray)
    hc = enhancer.enhance(2.0)
    passes.append(("contrast_sharp", hc.filter(ImageFilter.SHARPEN), ""))

    # Pass 2: Threshold binarise + PSM 6 (best for damaged/cracked)
    thresh = gray.point(lambda x: 255 if x > 150 else 0)
    passes.append(("threshold_psm6", thresh, "--oem 3 --psm 6"))

    # Pass 3: Aggressive threshold (recover numbers from noise)
    thresh2 = gray.point(lambda x: 255 if x > 180 else 0)
    passes.append(("threshold_aggressive", thresh2, "--oem 3 --psm 4"))

    # Run each pass and collect text
    all_texts = []
    for name, processed_img, config in passes:
        try:
            text = pytesseract.image_to_string(processed_img, lang="eng", config=config)
            all_texts.append((name, text))
            logger.debug("[ocr] Pass %s: %d chars", name, len(text))
        except Exception as e:
            logger.warning("[ocr] Pass %s failed: %s", name, e)

    if not all_texts:
        result = YodelOCRResult()
        result.fields_missing = list(DESIRED_FIELDS)
        return result

    # Parse each pass separately, then merge best fields
    best_result = YodelOCRResult()
    best_result.raw_text = "\n---\n".join(f"[{n}]\n{t}" for n, t in all_texts)
    best_result.method = "tesseract_multipass"

    for name, text in all_texts:
        candidate = _parse_all_fields(text)
        _merge_best(best_result, candidate)

    _validate(best_result)
    return best_result


def extract_from_text(raw_text: str) -> YodelOCRResult:
    """Extract from pre-extracted text (no image processing needed)."""
    result = _parse_all_fields(raw_text)
    result.raw_text = raw_text
    result.method = "text_parse"
    _validate(result)
    return result


# ── Field parsing ─────────────────────────────────────────

def _parse_all_fields(text: str) -> YodelOCRResult:
    """Parse all known Yodel fields from OCR text."""
    r = YodelOCRResult()

    # Tour Date: "8th November", "18th November", etc.
    # Fuzzy month matching: "Noveyhber" -> November (handle OCR noise)
    date_m = re.search(
        r"(?:tour\s*date|date)\s*[:\.\-]?\s*(\d{1,2})(?:st|nd|rd|th)?\s+(\w{3,12})",
        text, re.IGNORECASE,
    )
    if date_m:
        day = int(date_m.group(1))
        month_raw = date_m.group(2).lower()
        month = _fuzzy_month(month_raw)
        if month:
            # Check for explicit year after month
            year_m = re.search(
                re.escape(date_m.group(0)) + r"\s+(\d{4})", text
            )
            year = int(year_m.group(1)) if year_m else _infer_year(month, day)
            try:
                r.work_date = date(year, month, day)
                r.fields_extracted.append("tour_date")
            except ValueError:
                pass

    # User ID: DA9735 (also handle OCR noise like DAI735)
    uid_m = re.search(
        r"user\s*id\s*[:\.\-]?\s*([A-Z0-9]{2,8})",
        text, re.IGNORECASE,
    )
    if uid_m:
        raw_uid = uid_m.group(1).upper()
        # Common OCR fix: I->1 in user IDs like DA9735
        r.user_id = raw_uid.replace("I", "1") if raw_uid[:2] == "DA" else raw_uid
        # Ensure it matches known pattern (letters + digits)
        if re.match(r"[A-Z]{1,3}\d{3,6}", r.user_id):
            r.fields_extracted.append("user_id")
        else:
            r.user_id = raw_uid  # keep original if fix didn't help
            r.fields_extracted.append("user_id")

    # Tour ID: T009 (handle T@09, TOO9, etc.)
    tid_m = re.search(
        r"tour\s*id\s*[:\.\-]?\s*([A-Z0-9@]{1,6})",
        text, re.IGNORECASE,
    )
    if tid_m:
        raw_tid = tid_m.group(1).upper()
        # OCR fixes: @ -> 0, O -> 0 in tour IDs
        fixed = raw_tid.replace("@", "0").replace("OO", "00").replace("O", "0")
        # Ensure format: letter + digits (T009)
        if re.match(r"[A-Z]\d+", fixed):
            r.tour_id = fixed
        else:
            r.tour_id = raw_tid
        r.fields_extracted.append("tour_id")

    # Start time
    start_m = re.search(r"start\s*[:\.\-]?\s*(\d{1,2})[:\.](\d{2})", text, re.IGNORECASE)
    if start_m:
        r.start_time = f"{start_m.group(1)}:{start_m.group(2)}"
        r.fields_extracted.append("start_time")

    # End time
    end_m = re.search(r"end\s*[:\.\-]?\s*(\d{1,2})[:\.](\d{2})", text, re.IGNORECASE)
    if end_m:
        r.end_time = f"{end_m.group(1)}:{end_m.group(2)}"
        r.fields_extracted.append("end_time")

    # Duration: 6h 55m, 7h 54m, etc.
    dur_m = re.search(r"duration\s*[:\.\-]?\s*(\d{1,2})\s*h\s*(\d{1,2})\s*m", text, re.IGNORECASE)
    if dur_m:
        r.duration_hours = round(int(dur_m.group(1)) + int(dur_m.group(2)) / 60.0, 2)
        r.fields_extracted.append("duration")

    # Delivered Parcels: look for number after label
    dp_m = re.search(r"delivered\s*parcels?\s*[:\.\-\|]?\s*(\d+)", text, re.IGNORECASE)
    if dp_m:
        r.delivery_count = int(dp_m.group(1))
        r.fields_extracted.append("delivery_count")

    # Attempted/Completed Stops
    stops_m = re.search(
        r"(?:attempted|completed)\s*/?\\?\s*(?:completed)?\s*stops?\s*[:\.\-]?\s*(\d+)",
        text, re.IGNORECASE,
    )
    if stops_m:
        r.attempted_stops = int(stops_m.group(1))
        r.completed_stops = r.attempted_stops
        r.fields_extracted.append("attempted_stops")

    # Collections
    coll_m = re.search(r"collections?\s*[:\.\-]?\s*(\d+)", text, re.IGNORECASE)
    if coll_m:
        r.collections = int(coll_m.group(1))
        r.fields_extracted.append("collections")

    # Deliveries (distinct from "Delivered Parcels")
    del_m = re.search(r"\bdeliveries\s*[:\.\-]?\s*(\d+)", text, re.IGNORECASE)
    if del_m:
        r.deliveries = int(del_m.group(1))
        r.fields_extracted.append("deliveries")

    # Stores
    store_m = re.search(r"\bstores?\s*[:\.\-]?\s*(\d+)", text, re.IGNORECASE)
    if store_m:
        r.stores = int(store_m.group(1))
        r.fields_extracted.append("stores")

    # Lockers
    lock_m = re.search(r"\blockers?\s*[:\.\-]?\s*(\d+)", text, re.IGNORECASE)
    if lock_m:
        r.lockers = int(lock_m.group(1))
        r.fields_extracted.append("lockers")

    # Not Attempted
    na_m = re.search(r"not\s*att?e?mpted\s*[:\.\-]?\s*(\d+)", text, re.IGNORECASE)
    if na_m:
        r.not_attempted = int(na_m.group(1))
        r.failed_deliveries = r.not_attempted
        r.fields_extracted.append("not_attempted")

    r.fields_missing = [f for f in DESIRED_FIELDS if f not in r.fields_extracted]
    return r


def _merge_best(target: YodelOCRResult, candidate: YodelOCRResult) -> None:
    """Merge fields from candidate into target, preferring non-empty values."""
    if candidate.work_date and not target.work_date:
        target.work_date = candidate.work_date
        if "tour_date" not in target.fields_extracted:
            target.fields_extracted.append("tour_date")

    if candidate.user_id and not target.user_id:
        target.user_id = candidate.user_id
        if "user_id" not in target.fields_extracted:
            target.fields_extracted.append("user_id")

    if candidate.tour_id and not target.tour_id:
        target.tour_id = candidate.tour_id
        if "tour_id" not in target.fields_extracted:
            target.fields_extracted.append("tour_id")

    if candidate.start_time and not target.start_time:
        target.start_time = candidate.start_time
        if "start_time" not in target.fields_extracted:
            target.fields_extracted.append("start_time")

    if candidate.end_time and not target.end_time:
        target.end_time = candidate.end_time
        if "end_time" not in target.fields_extracted:
            target.fields_extracted.append("end_time")

    if candidate.duration_hours and not target.duration_hours:
        target.duration_hours = candidate.duration_hours
        if "duration" not in target.fields_extracted:
            target.fields_extracted.append("duration")

    # Numeric fields: take the larger (non-zero) value
    if candidate.delivery_count > target.delivery_count:
        target.delivery_count = candidate.delivery_count
        if "delivery_count" not in target.fields_extracted:
            target.fields_extracted.append("delivery_count")

    if candidate.attempted_stops > target.attempted_stops:
        target.attempted_stops = candidate.attempted_stops
        target.completed_stops = candidate.attempted_stops
        if "attempted_stops" not in target.fields_extracted:
            target.fields_extracted.append("attempted_stops")

    if candidate.collections > target.collections:
        target.collections = candidate.collections
    if candidate.deliveries > target.deliveries:
        target.deliveries = candidate.deliveries
    if candidate.stores > target.stores:
        target.stores = candidate.stores
    if candidate.lockers > target.lockers:
        target.lockers = candidate.lockers
    if candidate.not_attempted > target.not_attempted:
        target.not_attempted = candidate.not_attempted
        target.failed_deliveries = candidate.not_attempted

    # Recompute missing
    target.fields_missing = [f for f in DESIRED_FIELDS if f not in target.fields_extracted]


def _validate(result: YodelOCRResult) -> None:
    """Validate extracted data and set confidence score."""
    checks_passed = 0
    checks_total = 0

    checks_total += 1
    if result.work_date and result.work_date <= date.today():
        checks_passed += 1

    checks_total += 1
    if result.delivery_count > 0:
        checks_passed += 1

    if result.deliveries > 0 and result.delivery_count > 0:
        checks_total += 1
        if result.deliveries <= result.delivery_count:
            checks_passed += 1

    if result.attempted_stops > 0:
        checks_total += 1
        if 1 <= result.attempted_stops <= 300:
            checks_passed += 1

    if result.duration_hours:
        checks_total += 1
        if 0.5 <= result.duration_hours <= 16:
            checks_passed += 1

    field_score = len(result.fields_extracted) / max(len(DESIRED_FIELDS), 1)
    validation_score = checks_passed / max(checks_total, 1)
    result.confidence = round((field_score * 0.6 + validation_score * 0.4) * 100, 1)

    extracted_set = set(result.fields_extracted)
    result.is_valid = REQUIRED_FIELDS.issubset(extracted_set)


# ── Helpers ───────────────────────────────────────────────

def _fuzzy_month(raw: str) -> Optional[int]:
    """Match month name with fuzzy tolerance for OCR noise.
    
    Handles: 'noveyhber' -> november, 'novemher' -> november, etc.
    """
    raw = raw.lower().strip()

    # Exact match first
    if raw in MONTH_NAMES:
        return MONTH_NAMES[raw]

    # Fuzzy: find the month name with smallest edit distance
    best_month = None
    best_dist = 999

    for name, num in MONTH_NAMES.items():
        if len(name) < 4:
            continue  # skip abbreviations for fuzzy match
        dist = _edit_distance(raw, name)
        # Allow up to 3 character differences for long month names
        max_dist = 3 if len(name) >= 7 else 2
        if dist < best_dist and dist <= max_dist:
            best_dist = dist
            best_month = num

    return best_month


def _edit_distance(a: str, b: str) -> int:
    """Simple Levenshtein distance."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost))
        prev = curr

    return prev[len(b)]


def _infer_year(month: int, day: int) -> int:
    """Infer year when not in screenshot. If future, use last year."""
    today = date.today()
    try:
        candidate = date(today.year, month, day)
        return today.year - 1 if candidate > today else today.year
    except ValueError:
        return today.year

