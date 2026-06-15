# FILE: app/rag/_answerer_structured_context.py
# Purpose: Structured codebase context for RAG queries.
# Called-by: app.rag.answerer
# Depends-on: app.db, app.memory.rag_entries_model
# Last-renovated: 2026-06-11
"""
Structured codebase context for RAG queries.

Loads pre-computed data from .architecture/ files and memory to give
the LLM a high-level codebase overview without needing to search chunks.

Data sources:
    - INDEX_*.json      → file counts, sizes, language breakdown
    - IMPORT_GRAPH.json → dependency stats, most-coupled modules
    - Package summaries → subsystem descriptions (from memory T5 seeds)

This context is always injected (budget: ~2,000 tokens) alongside
the chunk-based code context and memory context.

v11.0 (2026-02-23): Job 11A — structured codebase intelligence.
"""

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ARCH_DIR = Path(os.getenv("ORB_ROOT", r"D:\Orb")) / ".architecture"


def _find_latest_file(directory: Path, prefix: str) -> Optional[Path]:
    """Find the latest file matching prefix_YYYY-MM-DD_*.json pattern."""
    candidates = sorted(directory.glob(f"{prefix}_*.json"), reverse=True)
    if candidates:
        return candidates[0]
    # Fallback to non-dated version
    fallback = directory / f"{prefix}.json"
    return fallback if fallback.exists() else None


def _load_index_stats() -> dict:
    """
    Load file statistics from the latest INDEX JSON.

    Returns dict with: total_files, total_lines, total_bytes,
    language_breakdown, biggest_files (top 10).
    """
    idx_path = _find_latest_file(_ARCH_DIR, "INDEX")
    if not idx_path or not idx_path.exists():
        logger.debug("[structured_context] No INDEX file found")
        return {}

    try:
        with open(idx_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Two on-disk INDEX formats are supported (2026-06-14 fix):
        #   RAG format → {"scanned_files": [{path, lines, bytes, language}]}
        #               (written as INDEX_<ts>.json by a full scan)
        #   Legacy     → {"files": [{path, size_bytes, ext, ...}]}
        #               (written as INDEX.json by the same scan)
        # Previously only "scanned_files"/"bytes"/"language" were read, so when
        # only the legacy INDEX.json was present the overview was always empty.
        files = data.get("scanned_files") or data.get("files") or []
        if not files:
            return {}

        def _bytes(f: dict) -> int:
            return f.get("bytes") or f.get("size_bytes") or 0

        def _lines(f: dict) -> int:
            return f.get("lines", 0) or 0

        def _lang(f: dict) -> str:
            lang = (f.get("language") or "").strip()
            if lang:
                return lang
            # Legacy index has no language — fall back to the file extension.
            return (f.get("ext") or "").lstrip(".").lower()

        total_lines = sum(_lines(f) for f in files)
        total_bytes = sum(_bytes(f) for f in files)

        # Language breakdown
        lang_counter = Counter()
        for f in files:
            lang = _lang(f)
            if lang:
                lang_counter[lang] += 1

        # Top 10 biggest files by bytes
        by_size = sorted(files, key=_bytes, reverse=True)
        biggest = []
        for f in by_size[:10]:
            path = f.get("path", "")
            # Make path relative
            rel = path.replace("D:\\Orb\\", "").replace("D:/Orb/", "")
            size_kb = _bytes(f) / 1024
            biggest.append(f"{rel} ({size_kb:.1f}KB, {_lines(f)} lines)")

        return {
            "total_files": len(files),
            "total_lines": total_lines,
            "total_bytes": total_bytes,
            "language_breakdown": dict(lang_counter.most_common(10)),
            "biggest_files": biggest,
            "source": idx_path.name,
        }
    except Exception as e:
        logger.warning("[structured_context] Failed to load INDEX: %s", e)
        return {}


def _load_import_graph_stats() -> dict:
    """
    Load dependency statistics from IMPORT_GRAPH.json.

    Returns dict with: total_modules, total_edges,
    most_depended_on (top 10), most_dependencies (top 10).
    """
    graph_path = _ARCH_DIR / "IMPORT_GRAPH.json"
    if not graph_path.exists():
        logger.debug("[structured_context] No IMPORT_GRAPH.json found")
        return {}

    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        stats = data.get("stats", {})
        if not stats:
            return {}

        # Top 10 most depended-on
        most_depended = []
        for entry in stats.get("most_depended_on", [])[:10]:
            mod = entry.get("module", "")
            deps = entry.get("dependents", 0)
            most_depended.append(f"{mod} ({deps} dependents)")

        # Top 10 most dependencies
        most_deps = []
        for entry in stats.get("most_dependencies", [])[:10]:
            mod = entry.get("module", "")
            count = entry.get("count", 0)
            most_deps.append(f"{mod} ({count} imports)")

        return {
            "total_modules": stats.get("total_modules", 0),
            "total_edges": stats.get("total_edges", 0),
            "unique_targets": stats.get("unique_targets", 0),
            "most_depended_on": most_depended,
            "most_dependencies": most_deps,
        }
    except Exception as e:
        logger.warning("[structured_context] Failed to load IMPORT_GRAPH: %s", e)
        return {}


def _load_package_summaries() -> list[str]:
    """
    Load package summaries from memory (T5 seeds).

    Returns list of summary text strings.
    """
    try:
        from app.db import SessionLocal
        from app.memory.rag_entries_model import RAGEntry

        db = SessionLocal()
        try:
            entries = (
                db.query(RAGEntry)
                .filter(
                    RAGEntry.domain == "architecture",
                    RAGEntry.chunk_text.like("[T5:%"),
                )
                .all()
            )
            summaries = []
            for e in entries:
                content = getattr(e, "chunk_text", "")
                if content:
                    summaries.append(content[:200])  # Cap per summary
            return summaries
        finally:
            db.close()
    except Exception as e:
        logger.warning("[structured_context] Failed to load package summaries: %s", e)
        return []


def build_structured_context(token_budget: int = 2000) -> str:
    """
    Build a structured codebase overview for LLM context injection.

    Combines INDEX stats, IMPORT_GRAPH stats, and package summaries
    into a formatted section. Always included in RAG queries.

    Args:
        token_budget: Maximum tokens (~4 chars each).

    Returns:
        Formatted string. Empty if no data available.
    """
    char_budget = token_budget * 4
    sections = []
    chars_used = 0

    # --- Index stats ---
    idx = _load_index_stats()
    if idx:
        total_mb = idx.get("total_bytes", 0) / (1024 * 1024)
        lines = [
            "[CODEBASE OVERVIEW]",
            f"Total files: {idx['total_files']}",
            f"Total lines: {idx.get('total_lines', 0):,}",
            f"Total size: {total_mb:.1f}MB",
        ]

        langs = idx.get("language_breakdown", {})
        if langs:
            lang_str = ", ".join(f"{k}: {v}" for k, v in list(langs.items())[:6])
            lines.append(f"Languages: {lang_str}")

        biggest = idx.get("biggest_files", [])
        if biggest:
            lines.append("Biggest files:")
            for f in biggest[:7]:
                lines.append(f"  - {f}")

        lines.append(f"[/CODEBASE OVERVIEW]")
        section = "\n".join(lines)

        if chars_used + len(section) < char_budget:
            sections.append(section)
            chars_used += len(section)

    # --- Import graph stats ---
    graph = _load_import_graph_stats()
    if graph and chars_used < char_budget:
        lines = [
            "[DEPENDENCY GRAPH]",
            f"Total modules: {graph.get('total_modules', 0)}",
            f"Total import edges: {graph.get('total_edges', 0)}",
            f"Unique import targets: {graph.get('unique_targets', 0)}",
        ]

        depended = graph.get("most_depended_on", [])
        if depended:
            lines.append("Most depended-on (hub modules):")
            for d in depended[:7]:
                lines.append(f"  - {d}")

        most_deps = graph.get("most_dependencies", [])
        if most_deps:
            lines.append("Most dependencies (complex modules):")
            for d in most_deps[:7]:
                lines.append(f"  - {d}")

        lines.append("[/DEPENDENCY GRAPH]")
        section = "\n".join(lines)

        if chars_used + len(section) < char_budget:
            sections.append(section)
            chars_used += len(section)

    # --- Package summaries (condensed) ---
    summaries = _load_package_summaries()
    if summaries and chars_used < char_budget:
        lines = ["[PACKAGE SUMMARIES]"]
        for s in summaries:
            line = f"  - {s}"
            if chars_used + len(line) > char_budget:
                break
            lines.append(line)
            chars_used += len(line)
        lines.append(f"[/PACKAGE SUMMARIES]")
        sections.append("\n".join(lines))

    if not sections:
        return ""

    result = "\n\n".join(sections)
    logger.info(
        "[structured_context] Built overview: %d sections, ~%d tokens",
        len(sections), chars_used // 4,
    )
    return result
