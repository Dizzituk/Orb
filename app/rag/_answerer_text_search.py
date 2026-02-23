# FILE: app/rag/_answerer_text_search.py
"""
On-demand full-text search across the codebase for RAG queries.

Used when the user asks "how many times does X appear", "find all
references to Y", "which files contain Z", etc.

Searches the architecture_file_content table (full source code stored
by the architecture scanner). Falls back to CODEBASE.md if the DB
table is empty.

Results are formatted as a summary section for LLM context injection.

v11.0 (2026-02-23): Job 11B — on-demand codebase text search.
"""

import logging
import re
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

# Budget for search results in the context
_DEFAULT_TOKEN_BUDGET = 1500


# =========================================================================
# Search term extraction
# =========================================================================

# Patterns that indicate a counting/search question
_COUNT_PATTERNS = [
    re.compile(r"how\s+many\s+(?:times?\s+)?(?:does|do|is|are)\s+(.+?)(?:\s+(?:appear|occur|exist|used|mentioned|referenced|found|show\s+up|come\s+up))", re.IGNORECASE),
    re.compile(r"how\s+many\s+(.+?)(?:\s+(?:are\s+there|exist|do\s+we\s+have))", re.IGNORECASE),
    re.compile(r"(?:find|search\s+for|count|locate)\s+(?:all\s+)?(?:occurrences?\s+of\s+|references?\s+to\s+|instances?\s+of\s+)?['\"]?(.+?)['\"]?\s*(?:in\s+the\s+codebase)?$", re.IGNORECASE),
    re.compile(r"which\s+files?\s+(?:contain|have|use|import|reference|mention)\s+['\"]?(.+?)['\"]?", re.IGNORECASE),
    re.compile(r"where\s+(?:does|do|is|are)\s+['\"]?(.+?)['\"]?\s+(?:used|referenced|imported|called|mentioned|defined)", re.IGNORECASE),
]


def extract_search_term(question: str) -> Optional[str]:
    """
    Extract the search term from a codebase search question.

    Returns None if the question doesn't look like a search/count query.
    """
    for pattern in _COUNT_PATTERNS:
        match = pattern.search(question)
        if match:
            term = match.group(1).strip().strip("'\"")
            # Sanity check — term shouldn't be too long or too short
            if 1 <= len(term) <= 60:
                return term
    return None


# =========================================================================
# Database search
# =========================================================================

def _search_file_contents(term: str) -> list[dict]:
    """
    Search architecture_file_content table for a term.

    Returns list of dicts: {file_path, count, sample_lines}
    """
    try:
        from app.db import SessionLocal
        from app.memory.architecture_models import (
            ArchitectureFileContent,
            ArchitectureFileIndex,
        )

        db = SessionLocal()
        try:
            # Join content with file index to get paths
            rows = (
                db.query(
                    ArchitectureFileIndex.path,
                    ArchitectureFileContent.content_text,
                )
                .join(
                    ArchitectureFileContent,
                    ArchitectureFileContent.file_index_id == ArchitectureFileIndex.id,
                )
                .all()
            )

            if not rows:
                logger.info("[text_search] No file contents in DB — table empty")
                return []

            results = []
            term_lower = term.lower()

            for file_path, content in rows:
                if not content:
                    continue

                # Count occurrences (case-insensitive)
                count = content.lower().count(term_lower)
                if count == 0:
                    continue

                # Extract sample lines containing the term
                sample_lines = []
                for i, line in enumerate(content.split("\n"), 1):
                    if term_lower in line.lower():
                        clean = line.strip()[:120]
                        sample_lines.append(f"L{i}: {clean}")
                        if len(sample_lines) >= 3:
                            break

                # Make path relative
                rel_path = file_path
                for prefix in ("D:\\Orb\\", "D:/Orb/"):
                    rel_path = rel_path.replace(prefix, "")

                results.append({
                    "file_path": rel_path,
                    "count": count,
                    "sample_lines": sample_lines,
                })

            # Sort by count descending
            results.sort(key=lambda x: x["count"], reverse=True)
            return results

        finally:
            db.close()

    except Exception as e:
        logger.warning("[text_search] DB search failed: %s", e)
        return []


# =========================================================================
# Format results
# =========================================================================

def _format_search_results(
    term: str,
    results: list[dict],
    token_budget: int,
) -> str:
    """Format search results for LLM context injection."""
    if not results:
        return (
            f"[TEXT SEARCH: '{term}']\n"
            f"No occurrences found in the codebase.\n"
            f"Note: This searches the architecture_file_content table. "
            f"If empty, run 'CREATE ARCHITECTURE MAP' first.\n"
            f"[/TEXT SEARCH]"
        )

    char_budget = token_budget * 4
    total_occurrences = sum(r["count"] for r in results)
    total_files = len(results)

    lines = [
        f"[TEXT SEARCH: '{term}']",
        f"Total occurrences: {total_occurrences} across {total_files} files",
        "",
    ]
    chars_used = sum(len(l) for l in lines)

    # Top files with samples
    lines.append("Top files by occurrence count:")
    for r in results[:15]:
        file_line = f"  {r['file_path']}: {r['count']} occurrences"
        if chars_used + len(file_line) > char_budget:
            break
        lines.append(file_line)
        chars_used += len(file_line)

        # Add sample lines if budget allows
        for sample in r["sample_lines"][:2]:
            sample_line = f"    {sample}"
            if chars_used + len(sample_line) > char_budget:
                break
            lines.append(sample_line)
            chars_used += len(sample_line)

    if total_files > 15:
        lines.append(f"  ... and {total_files - 15} more files")

    lines.append(f"[/TEXT SEARCH]")
    return "\n".join(lines)


# =========================================================================
# Public API
# =========================================================================

def search_codebase_text(
    question: str,
    token_budget: int = _DEFAULT_TOKEN_BUDGET,
) -> str:
    """
    Search codebase for a term extracted from the user's question.

    Returns formatted results for LLM context injection.
    Returns empty string if the question doesn't look like a search query.

    Args:
        question:     The user's codebase question.
        token_budget: Maximum tokens for the results section.

    Returns:
        Formatted search results, or empty string.
    """
    term = extract_search_term(question)
    if not term:
        return ""

    logger.info("[text_search] Searching for: '%s'", term)
    results = _search_file_contents(term)

    formatted = _format_search_results(term, results, token_budget)
    logger.info(
        "[text_search] '%s': %d occurrences in %d files",
        term,
        sum(r["count"] for r in results),
        len(results),
    )
    return formatted
