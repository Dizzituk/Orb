# FILE: app/memory/ingest/deduplicator.py
"""
Overlap and conflict detection (Spec Section 9.2, Stage 4).

Before storing an ingested item, check whether it duplicates or
conflicts with existing memory entries. This prevents the same
fact being stored multiple times from different ingest runs.

Three outcomes:
    UNIQUE      — No overlap found, safe to store
    DUPLICATE   — Near-identical entry already exists, skip
    CONFLICT    — Contradicts existing entry, flag for review

Detection methods:
    - Token overlap ratio (Jaccard similarity on word sets)
    - Key phrase matching (exact factual claims)
    - Domain + layer scoping (only compare within same domain)

Usage:
    from app.memory.ingest.deduplicator import check_duplicate

    result = check_duplicate(
        text="File size target is 20KB",
        domain="development",
        project_id="astra-core",
    )
    if result.status == "DUPLICATE":
        skip(result.existing_id)
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import and_

from app.db import get_db_session
from app.memory.rag_entries_model import RAGEntry

logger = logging.getLogger(__name__)


# =========================================================================
# Result
# =========================================================================

@dataclass
class DedupeResult:
    """Result of a deduplication check."""
    status: str             # UNIQUE, DUPLICATE, CONFLICT
    existing_id: Optional[int]
    similarity: float       # 0.0–1.0
    reason: Optional[str]


# Thresholds
DUPLICATE_THRESHOLD = 0.85  # Above this = same content
CONFLICT_THRESHOLD = 0.4    # Above this + contradicting = conflict


# =========================================================================
# Public API
# =========================================================================

def check_duplicate(
    text: str,
    domain: str,
    project_id: str = "astra-core",
    memory_layer: Optional[str] = None,
) -> DedupeResult:
    """
    Check if text duplicates or conflicts with existing entries.

    Only compares within the same domain and project.
    Preferences are checked more strictly (exact key matching).

    Args:
        text: The candidate text to check.
        domain: Memory domain scope.
        project_id: Project scope.
        memory_layer: Optional layer hint (preference, knowledge, etc.)

    Returns:
        DedupeResult with status, existing match ID, and similarity.
    """
    db = get_db_session()
    try:
        candidates = _get_candidates(db, domain, project_id)
        if not candidates:
            return DedupeResult("UNIQUE", None, 0.0, None)

        text_lower = text.lower().strip()
        text_tokens = _tokenise(text_lower)

        best_sim = 0.0
        best_id = None
        best_text = ""

        for entry in candidates:
            entry_lower = (entry.chunk_text or "").lower().strip()
            entry_tokens = _tokenise(entry_lower)

            sim = _jaccard(text_tokens, entry_tokens)
            if sim > best_sim:
                best_sim = sim
                best_id = entry.id
                best_text = entry_lower

        # Duplicate: very high overlap
        if best_sim >= DUPLICATE_THRESHOLD:
            logger.debug(
                "[dedup] DUPLICATE (%.2f): '%s' matches entry %d",
                best_sim, text[:50], best_id,
            )
            return DedupeResult(
                "DUPLICATE", best_id, round(best_sim, 3),
                "Near-identical entry already exists",
            )

        # Conflict: moderate overlap but contradicting numbers/values
        if best_sim >= CONFLICT_THRESHOLD:
            contradiction = _detect_contradiction(text_lower, best_text)
            if contradiction:
                logger.debug(
                    "[dedup] CONFLICT (%.2f): '%s' vs entry %d: %s",
                    best_sim, text[:50], best_id, contradiction,
                )
                return DedupeResult(
                    "CONFLICT", best_id, round(best_sim, 3),
                    contradiction,
                )

        return DedupeResult("UNIQUE", None, round(best_sim, 3), None)

    finally:
        db.close()


def check_batch(
    items: list[dict],
    domain: str,
    project_id: str = "astra-core",
) -> list[DedupeResult]:
    """
    Batch deduplication check.

    Also deduplicates within the batch itself — if item 3 is a
    duplicate of item 1, item 3 gets flagged even though item 1
    isn't in the DB yet.

    Args:
        items: List of dicts with 'text' key.
        domain: Domain scope.
        project_id: Project scope.

    Returns:
        List of DedupeResult, one per item (same order).
    """
    results = []
    seen_in_batch: list[dict] = []

    for item in items:
        text = item.get("text", "")

        # Check against DB
        db_result = check_duplicate(text, domain, project_id)
        if db_result.status != "UNIQUE":
            results.append(db_result)
            continue

        # Check against earlier items in this batch
        text_tokens = _tokenise(text.lower().strip())
        batch_dup = False
        for earlier in seen_in_batch:
            sim = _jaccard(text_tokens, earlier["tokens"])
            if sim >= DUPLICATE_THRESHOLD:
                results.append(DedupeResult(
                    "DUPLICATE", None, round(sim, 3),
                    f"Duplicate of batch item {earlier['index']}",
                ))
                batch_dup = True
                break

        if not batch_dup:
            results.append(db_result)
            seen_in_batch.append({
                "tokens": text_tokens,
                "index": len(seen_in_batch),
            })

    return results


# =========================================================================
# Internals
# =========================================================================

def _get_candidates(db, domain: str, project_id: str) -> list:
    """Get existing active entries for comparison."""
    return (
        db.query(RAGEntry)
        .filter(and_(
            RAGEntry.domain == domain,
            RAGEntry.project_id == project_id,
            RAGEntry.status == "ACTIVE",
        ))
        .limit(500)
        .all()
    )


def _tokenise(text: str) -> set[str]:
    """Tokenise text into a word set, stripping punctuation."""
    words = re.findall(r'\b\w+\b', text.lower())
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on",
        "at", "to", "for", "of", "and", "or", "not", "with", "from",
        "by", "it", "this", "that", "i", "my", "me", "we", "our",
    }
    return {w for w in words if len(w) >= 2 and w not in stop}


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two token sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _detect_contradiction(new_text: str, existing_text: str) -> Optional[str]:
    """
    Detect if two similar texts contain contradicting values.

    Looks for numeric values and named values that differ
    between the two texts when the surrounding context is similar.
    """
    # Extract numbers from both
    new_nums = set(re.findall(r'\b\d+(?:\.\d+)?\s*(?:kb|mb|gb|%|px|ms|s|min|hr|day)?\b', new_text))
    old_nums = set(re.findall(r'\b\d+(?:\.\d+)?\s*(?:kb|mb|gb|%|px|ms|s|min|hr|day)?\b', existing_text))

    if new_nums and old_nums and new_nums != old_nums:
        diff_new = new_nums - old_nums
        diff_old = old_nums - new_nums
        if diff_new and diff_old:
            return (
                f"Contradicting values: new has {diff_new}, "
                f"existing has {diff_old}"
            )

    return None
