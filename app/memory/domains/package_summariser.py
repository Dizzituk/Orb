# FILE: app/memory/domains/package_summariser.py
"""
Tier 5 — Package Summaries.

Reads the latest SIGNATURES and INDEX files from .architecture/
and generates one summary per package directory. Each summary
describes the files, key classes/functions, and role of the package.

Finds the latest SIGNATURES file by glob + modification date,
not by hardcoded timestamp.

Stored in rag_entries with domain='architecture' and tier='T5'.
"""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.db import get_db_session
from app.memory.rag_entries_model import RAGEntry

logger = logging.getLogger(__name__)

DOMAIN = "architecture"
PROJECT = "astra-core"
TIER = "T5"


# =========================================================================
# Find latest architecture files
# =========================================================================

def _find_latest_file(arch_dir: str, pattern: str) -> Optional[str]:
    """
    Find the latest file matching a glob pattern in the architecture dir.

    Sorts by modification time, returns the newest.
    """
    matches = glob(os.path.join(arch_dir, pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


# =========================================================================
# Package summary generation
# =========================================================================

def generate_package_summaries(
    arch_dir: str = r"D:\Orb\.architecture",
) -> list[dict]:
    """
    Generate one summary per package from the latest SIGNATURES file.

    Returns list of dicts: {package, file_count, key_symbols, summary_text}
    """
    sig_path = _find_latest_file(arch_dir, "SIGNATURES_*.json")
    if not sig_path:
        logger.warning("[package_summariser] No SIGNATURES file found")
        return []

    logger.info(f"[package_summariser] Reading {sig_path}")
    with open(sig_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_file = data.get("by_file", {})
    if not by_file:
        logger.warning("[package_summariser] No by_file data in SIGNATURES")
        return []

    # Group signatures by package directory
    packages: dict[str, dict] = defaultdict(lambda: {
        "files": [],
        "classes": [],
        "functions": [],
        "total_chunks": 0,
    })

    for filepath, chunks in by_file.items():
        # Normalise path and extract package
        rel = _to_relative(filepath)
        if not rel:
            continue

        pkg = _get_package(rel)
        if not pkg:
            continue

        pkg_data = packages[pkg]
        pkg_data["files"].append(rel)
        pkg_data["total_chunks"] += len(chunks)

        for chunk in chunks:
            kind = chunk.get("kind", "")
            name = chunk.get("name", "")
            sig = chunk.get("signature", "")

            if kind == "class":
                bases = chunk.get("bases", [])
                base_str = f"({', '.join(bases)})" if bases else ""
                pkg_data["classes"].append(f"{name}{base_str}")
            elif kind in ("function", "async_function"):
                pkg_data["functions"].append(f"{name}({sig})" if sig else name)

    # Generate summaries
    summaries = []
    for pkg, data in sorted(packages.items()):
        summary = _build_summary(pkg, data)
        summaries.append(summary)

    logger.info(f"[package_summariser] Generated {len(summaries)} package summaries")
    return summaries


def _to_relative(filepath: str) -> Optional[str]:
    """Convert absolute path to relative from project root."""
    normalised = filepath.replace("\\", "/")

    for prefix in ["D:/Orb/", "D:/orb-desktop/"]:
        if normalised.startswith(prefix):
            return normalised[len(prefix):]

    # Try case-insensitive
    lower = normalised.lower()
    for prefix in ["d:/orb/", "d:/orb-desktop/"]:
        if lower.startswith(prefix):
            return normalised[len(prefix):]

    return None


def _get_package(rel_path: str) -> Optional[str]:
    """
    Extract package directory from a relative file path.

    app/memory/router.py → app/memory
    app/llm/pipeline/critique.py → app/llm/pipeline
    main.py → (root)
    """
    parts = rel_path.rsplit("/", 1)
    if len(parts) == 1:
        return "(root)"
    return parts[0]


def _build_summary(package: str, data: dict) -> dict:
    """Build a structured summary for one package."""
    file_count = len(data["files"])
    class_count = len(data["classes"])
    func_count = len(data["functions"])

    # Build text summary
    lines = [
        f"PACKAGE: {package}",
        f"Files: {file_count}, Classes: {class_count}, Functions: {func_count}, "
        f"Total symbols: {data['total_chunks']}",
    ]

    if data["classes"]:
        top_classes = data["classes"][:10]
        lines.append(f"Key classes: {', '.join(top_classes)}")

    if data["functions"]:
        top_funcs = data["functions"][:10]
        # Truncate long signatures
        short_funcs = [f[:60] for f in top_funcs]
        lines.append(f"Key functions: {', '.join(short_funcs)}")

    if data["files"]:
        # Show file names only (not full paths)
        filenames = sorted(set(
            f.rsplit("/", 1)[-1] for f in data["files"]
        ))
        if len(filenames) > 15:
            shown = filenames[:15]
            lines.append(f"Files: {', '.join(shown)} (+{len(filenames) - 15} more)")
        else:
            lines.append(f"Files: {', '.join(filenames)}")

    return {
        "package": package,
        "file_count": file_count,
        "class_count": class_count,
        "func_count": func_count,
        "summary_text": "\n".join(lines),
    }


# =========================================================================
# Storage
# =========================================================================

def store_package_summaries(
    arch_dir: str = r"D:\Orb\.architecture",
) -> int:
    """
    Generate and store package summaries in rag_entries.

    Replaces any existing T5 entries (full refresh).
    Returns count of summaries stored.
    """
    summaries = generate_package_summaries(arch_dir)
    if not summaries:
        return 0

    db = get_db_session()
    try:
        # Remove existing T5 entries
        existing = db.query(RAGEntry).filter(
            RAGEntry.domain == DOMAIN,
            RAGEntry.project_id == PROJECT,
            RAGEntry.chunk_text.like(f"[{TIER}:%"),
        ).all()
        for e in existing:
            db.delete(e)

        # Insert new summaries
        count = 0
        for s in summaries:
            entry = RAGEntry(
                project_id=PROJECT,
                domain=DOMAIN,
                chunk_text=f"[{TIER}:{s['package']}] {s['summary_text']}",
                status="ACTIVE",
                ingest_source="package_scan",
                indexed_at=datetime.utcnow(),
            )
            db.add(entry)
            count += 1

        db.commit()
        logger.info(f"[package_summariser] Stored {count} package summaries")
        return count
    finally:
        db.close()
