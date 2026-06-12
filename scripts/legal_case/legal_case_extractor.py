# FILE: scripts/legal_case/legal_case_extractor.py
# Purpose: Legal case screenshot extractor.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.llm.gemini_vision, scripts.legal_case._bootstrap, scripts.legal_case._extraction_prompt
# Last-renovated: 2026-06-11
"""
Legal case screenshot extractor.

Walks the Work Screen shots directory, sends every PNG/JPG through
Gemini vision with a structured-extraction prompt, and writes a single
JSON log of the results.

Parallelism: ThreadPoolExecutor (default 8 workers). Each Gemini call
is network-bound, so threads are the right primitive — no need for
agents, sub-processes, or asyncio.

Idempotent: on re-run, skips images whose filenames already appear in
the output JSON unless --force is passed.

Usage:
    python -m scripts.legal_case.legal_case_extractor            # full run
    python -m scripts.legal_case.legal_case_extractor --sample 5 # first 5 only
    python -m scripts.legal_case.legal_case_extractor --force    # re-do everything

Output: C:\\Users\\dizzi\\OneDrive\\Documents\\Work Legal\\screenshots_ocr.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.legal_case._bootstrap import bootstrap_astra
from scripts.legal_case._extraction_prompt import EXTRACTION_PROMPT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SCREENSHOTS_ROOT = Path(r"C:\Users\dizzi\OneDrive\Documents\Work Legal\Work Screen shots")
OUTPUT_FILE = Path(r"C:\Users\dizzi\OneDrive\Documents\Work Legal\screenshots_ocr.json")
VALID_EXT = {".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"}

# Filename pattern for "Screenshot_YYYYMMDD-HHMMSS.ext".
_FILENAME_DATE_RE = re.compile(r"(\d{4})(\d{2})(\d{2})[-_](\d{2})(\d{2})(\d{2})")


def _parse_filename_datetime(name: str) -> Dict[str, Optional[str]]:
    """Extract date/time from the filename as a best-effort fallback."""
    m = _FILENAME_DATE_RE.search(name)
    if not m:
        return {"filename_date": None, "filename_time": None}
    y, mo, d, h, mi, s = m.groups()
    return {
        "filename_date": f"{y}-{mo}-{d}",
        "filename_time": f"{h}:{mi}:{s}",
    }


def _parse_model_json(raw: str) -> Dict[str, Any]:
    """Pull the JSON object out of the model's response, tolerating fences."""
    if not raw:
        return {"_parse_error": "empty response"}
    s = raw.strip()
    # Strip ```json ... ``` fences if present.
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    # Grab the outermost {...}.
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"_parse_error": "no JSON object found", "_raw": raw[:500]}
    try:
        return json.loads(s[start:end + 1])
    except json.JSONDecodeError as e:
        return {"_parse_error": f"JSONDecodeError: {e}", "_raw": raw[:500]}


def _process_one(image_path: Path) -> Dict[str, Any]:
    """Send one image through Gemini and return a normalised record."""
    from app.llm.gemini_vision import ask_about_image

    rel = image_path.relative_to(SCREENSHOTS_ROOT).as_posix()
    record: Dict[str, Any] = {
        "filename": image_path.name,
        "relative_path": rel,
        "full_path": str(image_path),
        **_parse_filename_datetime(image_path.name),
    }
    t0 = time.monotonic()
    try:
        result = ask_about_image(image_path, EXTRACTION_PROMPT, tier="default")
        raw_answer = result.get("answer", "") or ""
        parsed = _parse_model_json(raw_answer)
        record["extraction"] = parsed
        record["provider"] = result.get("provider")
        record["model"] = result.get("model")
        if "_parse_error" in parsed:
            record["status"] = "parse_error"
        elif parsed.get("type") == "not_work":
            record["status"] = "not_work"
        else:
            record["status"] = "ok"
    except Exception as e:
        logger.warning("[extract] %s failed: %s", image_path.name, e)
        record["status"] = "error"
        record["error"] = str(e)
    record["elapsed_seconds"] = round(time.monotonic() - t0, 2)
    return record


def _gather_images(root: Path) -> List[Path]:
    """Every PNG/JPG under the screenshots tree, sorted by filename."""
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in VALID_EXT:
            out.append(p)
    out.sort(key=lambda p: p.name)
    return out


def _load_existing(output_file: Path) -> Dict[str, Dict[str, Any]]:
    """Load prior OCR results keyed by filename, if any."""
    if not output_file.exists():
        return {}
    try:
        data = json.loads(output_file.read_text(encoding="utf-8"))
        items = data.get("items", [])
        return {rec["filename"]: rec for rec in items if "filename" in rec}
    except Exception as e:
        logger.warning("[extract] Existing output unreadable (%s); starting fresh", e)
        return {}


def run(sample: Optional[int] = None, force: bool = False, workers: int = 8) -> Dict[str, Any]:
    """Main entry point. Returns the final result dict also written to disk."""
    bootstrap_astra()

    images = _gather_images(SCREENSHOTS_ROOT)
    logger.info("[extract] Found %d images under %s", len(images), SCREENSHOTS_ROOT)

    existing = {} if force else _load_existing(OUTPUT_FILE)
    todo = [p for p in images if p.name not in existing]
    if sample:
        todo = todo[:sample]
    logger.info("[extract] Processing %d (skipping %d already done)", len(todo), len(images) - len(todo) if not force else 0)

    results: Dict[str, Dict[str, Any]] = dict(existing)
    t_start = time.monotonic()
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, p): p for p in todo}
        for fut in as_completed(futures):
            rec = fut.result()
            results[rec["filename"]] = rec
            done += 1
            if done % 5 == 0 or done == len(todo):
                logger.info("[extract]   %d/%d done (%.1fs elapsed)", done, len(todo), time.monotonic() - t_start)

    # Persist full, sorted-by-filename.
    items = [results[k] for k in sorted(results.keys())]
    by_status: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    for rec in items:
        by_status[rec.get("status", "unknown")] = by_status.get(rec.get("status", "unknown"), 0) + 1
        ext = rec.get("extraction") or {}
        t = ext.get("type")
        if t:
            by_type[t] = by_type.get(t, 0) + 1

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "screenshots_root": str(SCREENSHOTS_ROOT),
        "total_images": len(items),
        "status_counts": by_status,
        "type_counts": by_type,
        "items": items,
    }
    OUTPUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("[extract] Wrote %s (%d items)", OUTPUT_FILE, len(items))
    logger.info("[extract] Status: %s", by_status)
    logger.info("[extract] Types:  %s", by_type)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="Only process the first N unprocessed images")
    ap.add_argument("--force", action="store_true", help="Re-process images even if already in the output JSON")
    ap.add_argument("--workers", type=int, default=8, help="Thread pool size (default 8)")
    args = ap.parse_args()
    run(sample=args.sample, force=args.force, workers=args.workers)


if __name__ == "__main__":
    main()
