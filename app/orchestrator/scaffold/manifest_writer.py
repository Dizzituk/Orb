# FILE: app/orchestrator/scaffold/manifest_writer.py
"""
Manifest Writer for the Scaffold Engine.

Writes scaffold files and fill manifest sidecars to the segment's
working directory. Also writes a scaffold_summary.json for the
segment trace.

Pure file I/O — no LLM calls, no external dependencies beyond stdlib.

v1.0 (2026-03-01): Initial implementation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import List, Optional

from app.orchestrator.scaffold.models import (
    ScaffoldFile,
    ScaffoldResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PATH ENCODING
# =============================================================================


def encode_scaffold_filename(file_path: str) -> str:
    """Encode a file path into a flat scaffold filename.

    Forward slashes and backslashes become double underscores.
    Example: "app/education/api.py" → "app__education__api.py"
    """
    norm = file_path.replace("\\", "/")
    return norm.replace("/", "__")


def decode_scaffold_filename(encoded: str) -> str:
    """Decode a scaffold filename back to a file path.

    Example: "app__education__api.py" → "app/education/api.py"
    """
    # Split on extension first to preserve it
    base, ext = os.path.splitext(encoded)
    return base.replace("__", "/") + ext


# =============================================================================
# MAIN WRITER
# =============================================================================


def write_scaffold_result(
    result: ScaffoldResult,
    job_dir: str,
    segment_id: str,
) -> str:
    """Write all scaffold files and manifests to the segment directory.

    Creates:
        segments/{segment_id}/scaffold/
          ├── {encoded_path}              (scaffold source file)
          ├── {encoded_path}.fills.json   (fill manifest)
          └── scaffold_summary.json       (segment summary)

    Args:
        result: The ScaffoldResult containing all scaffold files.
        job_dir: Path to the job directory (e.g. jobs/jobs/sg-XXXXX).
        segment_id: The segment ID (e.g. "seg-01-models").

    Returns:
        Path to the scaffold directory.
    """
    scaffold_dir = os.path.join(job_dir, "segments", segment_id, "scaffold")
    os.makedirs(scaffold_dir, exist_ok=True)

    files_written = 0

    for scaffold_file in result.files:
        _write_single_scaffold(scaffold_file, scaffold_dir)
        files_written += 1

    # Write summary
    summary = _build_summary(result, files_written)
    summary_path = os.path.join(scaffold_dir, "scaffold_summary.json")
    _write_json(summary_path, summary)

    logger.info(
        "[manifest_writer] Wrote %d scaffold(s) + %d manifest(s) + summary to %s",
        files_written, files_written, scaffold_dir,
    )

    return scaffold_dir


def _write_single_scaffold(
    scaffold_file: ScaffoldFile,
    scaffold_dir: str,
) -> None:
    """Write a single scaffold file and its fill manifest.

    Also computes and stores the scaffold hash.
    """
    encoded_name = encode_scaffold_filename(scaffold_file.file_path)

    # Compute hash before writing
    scaffold_file.compute_scaffold_hash()

    # Compute locked region hashes
    _compute_locked_hashes(scaffold_file)

    # Write scaffold source file
    scaffold_path = os.path.join(scaffold_dir, encoded_name)
    _write_text(scaffold_path, scaffold_file.content)

    # Write fill manifest
    manifest_path = os.path.join(scaffold_dir, encoded_name + ".fills.json")
    _write_json(manifest_path, scaffold_file.manifest.to_dict())

    logger.debug(
        "[manifest_writer] Wrote %s (%d lines, %d fills, hash=%s...)",
        encoded_name,
        scaffold_file.line_count,
        scaffold_file.fill_count,
        scaffold_file.manifest.scaffold_hash[:12],
    )


def _compute_locked_hashes(scaffold_file: ScaffoldFile) -> None:
    """Compute content hashes for all locked regions.

    This enables the validator to detect if the Implementer
    modified locked content during the fill process.
    """
    lines = scaffold_file.content.splitlines()

    for region in scaffold_file.manifest.locked_regions:
        # Line numbers are 1-indexed
        start_idx = region.line_start - 1
        end_idx = region.line_end  # Inclusive, so no -1 for slice end
        region_lines = lines[start_idx:end_idx]
        region_text = "\n".join(region_lines)
        region.content_hash = hashlib.sha256(
            region_text.encode("utf-8")
        ).hexdigest()


# =============================================================================
# SUMMARY BUILDER
# =============================================================================


def _build_summary(result: ScaffoldResult, files_written: int) -> dict:
    """Build the scaffold_summary.json content."""
    file_details = []
    for sf in result.files:
        file_details.append({
            "file_path": sf.file_path,
            "language": sf.language.value,
            "role": sf.role.value,
            "line_count": sf.line_count,
            "fill_count": sf.fill_count,
            "is_complete": sf.is_complete,
            "scaffold_hash": sf.manifest.scaffold_hash[:16],
        })

    return {
        "segment_id": result.segment_id,
        "generated_at": _iso_now(),
        "files_scaffolded": files_written,
        "files_complete": result.complete_files,
        "files_with_fills": result.file_count - result.complete_files,
        "total_fills": result.total_fills,
        "total_lines": result.total_lines,
        "skipped_files": result.skipped_files,
        "warnings": result.warnings,
        "generation_time_ms": round(result.generation_time_ms, 1),
        "files": file_details,
    }


# =============================================================================
# SCAFFOLD LOADER (for the Implementer)
# =============================================================================


def load_scaffold_for_file(
    job_dir: str,
    segment_id: str,
    file_path: str,
) -> Optional[ScaffoldFile]:
    """Load a scaffold file and its manifest from the segment directory.

    Called by the modified Implementer to check if a scaffold exists
    for a given file.

    Returns None if no scaffold exists for this file.
    """
    scaffold_dir = os.path.join(job_dir, "segments", segment_id, "scaffold")
    if not os.path.isdir(scaffold_dir):
        return None

    encoded_name = encode_scaffold_filename(file_path)
    scaffold_path = os.path.join(scaffold_dir, encoded_name)
    manifest_path = os.path.join(scaffold_dir, encoded_name + ".fills.json")

    if not os.path.isfile(scaffold_path):
        return None

    try:
        with open(scaffold_path, "r", encoding="utf-8") as f:
            content = f.read()

        manifest_dict = {}
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_dict = json.load(f)

        from app.orchestrator.scaffold.models import (
            FillManifest,
            FillMarker,
            FillDifficulty,
            FileLanguage,
            FileRole,
            LockedRegion,
        )

        # Reconstruct FillManifest from dict
        fills = []
        for fd in manifest_dict.get("fills", []):
            fills.append(FillMarker(
                id=fd["id"],
                location=fd.get("location", ""),
                line_start=fd.get("line_start", 0),
                line_end=fd.get("line_end", 0),
                context=fd.get("context", ""),
                max_lines=fd.get("max_lines", 10),
                inputs_available=fd.get("inputs_available", []),
                return_type=fd.get("return_type", ""),
                difficulty=FillDifficulty(fd.get("difficulty", "standard")),
            ))

        locked = []
        for rd in manifest_dict.get("locked_regions", []):
            locked.append(LockedRegion(
                line_start=rd.get("line_start", 0),
                line_end=rd.get("line_end", 0),
                region_type=rd.get("type", ""),
                content_hash=rd.get("content_hash", ""),
            ))

        manifest = FillManifest(
            file_path=file_path,
            scaffold_hash=manifest_dict.get("scaffold_hash", ""),
            fills=fills,
            locked_regions=locked,
        )

        # Detect language/role from path
        from app.orchestrator.scaffold.arch_parser import detect_language, detect_role
        lang = detect_language(file_path)
        role = detect_role(file_path)

        sf = ScaffoldFile(
            file_path=file_path,
            content=content,
            manifest=manifest,
            language=lang,
            role=role,
        )

        logger.debug(
            "[manifest_writer] Loaded scaffold for %s (%d lines, %d fills)",
            file_path, sf.line_count, sf.fill_count,
        )
        return sf

    except Exception as e:
        logger.warning(
            "[manifest_writer] Failed to load scaffold for %s: %s",
            file_path, e,
        )
        return None


def load_scaffold_result(
    job_dir: str,
    segment_id: str,
) -> Optional[ScaffoldResult]:
    """Load the complete ScaffoldResult for a segment.

    Reads scaffold_summary.json and loads all referenced scaffolds.
    """
    scaffold_dir = os.path.join(job_dir, "segments", segment_id, "scaffold")
    summary_path = os.path.join(scaffold_dir, "scaffold_summary.json")

    if not os.path.isfile(summary_path):
        return None

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        result = ScaffoldResult(
            segment_id=segment_id,
            skipped_files=summary.get("skipped_files", []),
            warnings=summary.get("warnings", []),
            generation_time_ms=summary.get("generation_time_ms", 0.0),
        )

        for file_detail in summary.get("files", []):
            file_path = file_detail.get("file_path", "")
            if file_path:
                sf = load_scaffold_for_file(job_dir, segment_id, file_path)
                if sf:
                    result.files.append(sf)

        return result

    except Exception as e:
        logger.warning(
            "[manifest_writer] Failed to load scaffold result for %s/%s: %s",
            job_dir, segment_id, e,
        )
        return None


# =============================================================================
# FILE I/O HELPERS
# =============================================================================


def _write_text(path: str, content: str) -> None:
    """Write text content to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def _write_json(path: str, data: dict) -> None:
    """Write JSON data to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _iso_now() -> str:
    """ISO timestamp for the current moment."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
