# FILE: app/orchestrator/ledger_compactor.py
"""
Evidence Ledger Compactor — consolidate verbose ledger entries into summaries.

BUILD_ID: 2026-02-28-v1.0-ledger-compactor

After a pass completes, individual file_read and decision entries are
consolidated into compact summaries. This prevents the ledger from growing
unbounded across multi-pass app builds.

Flow:
    1. Group entries by segment_id
    2. For each segment: summarise file_reads, decisions, constraints
    3. Replace originals with compact summary entries
    4. Preserve flags and corrections (these are always relevant)

Called from seg_job_post.py after all segments complete.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LEDGER_COMPACTOR_BUILD_ID = "2026-02-28-v1.0-ledger-compactor"


def compact_ledger(
    ledger: Any,
    job_dir: str,
    pass_number: int = 1,
    emit: Any = None,
) -> int:
    """Compact the ledger after a pass completes.

    Replaces verbose entries with segment-level summaries.
    Returns count of entries removed (net reduction).

    Args:
        ledger: EvidenceLedger instance
        job_dir: Path to job directory (for save)
        pass_number: Which pass just completed (for summary labelling)
        emit: Optional progress callback
    """
    try:
        from app.orchestrator.evidence_ledger import (
            LedgerEntry, ledger_append, save_ledger,
        )
    except ImportError:
        logger.warning("[ledger_compactor] Cannot import evidence_ledger")
        return 0

    if not ledger or not ledger.entries:
        return 0

    # Group entries by segment
    by_segment: Dict[str, List] = {}
    for entry in ledger.entries:
        seg = entry.segment_id or "_global"
        by_segment.setdefault(seg, []).append(entry)

    original_count = len(ledger.entries)
    preserved: List = []
    summaries_added = 0

    for seg_id, entries in by_segment.items():
        # Always preserve these types — they're compact and always relevant
        keep_types = {"flag", "correction", "constraint"}
        kept = [e for e in entries if e.type in keep_types]
        compactable = [e for e in entries if e.type not in keep_types]

        if len(compactable) < 3:
            # Not worth compacting — keep as-is
            preserved.extend(entries)
            continue

        # Build compact summary
        summary_parts = []

        # Summarise file reads
        file_reads = [e for e in compactable if e.type == "file_read"]
        if file_reads:
            paths = [e.path or e.key or "?" for e in file_reads]
            summary_parts.append(f"Files read ({len(paths)}): {', '.join(paths[:10])}")
            if len(paths) > 10:
                summary_parts[-1] += f" (+{len(paths)-10} more)"

        # Summarise decisions
        decisions = [e for e in compactable if e.type == "decision"]
        if decisions:
            decision_lines = []
            for d in decisions:
                if d.key and d.value:
                    decision_lines.append(f"{d.key}={d.value}")
                elif d.summary:
                    decision_lines.append(d.summary)
            summary_parts.append(
                f"Decisions ({len(decisions)}): {'; '.join(decision_lines[:15])}"
            )
            if len(decision_lines) > 15:
                summary_parts[-1] += f" (+{len(decision_lines)-15} more)"

        # Summarise codebase facts
        facts = [e for e in compactable if e.type == "codebase_fact"]
        if facts:
            fact_summaries = [f.summary for f in facts if f.summary]
            summary_parts.append(
                f"Facts ({len(facts)}): {'; '.join(fact_summaries[:8])}"
            )

        # Summarise verifications
        verifications = [e for e in compactable if e.type == "verification"]
        if verifications:
            passed = sum(1 for v in verifications if "pass" in (v.value or "").lower())
            summary_parts.append(
                f"Verifications: {passed}/{len(verifications)} passed"
            )

        if summary_parts:
            # Create the compact summary entry
            summary_text = (
                f"Pass {pass_number} summary for {seg_id}: "
                + " | ".join(summary_parts)
            )
            ledger_append(
                ledger,
                entry_type="codebase_fact",
                stage="compactor",
                summary=summary_text,
                segment_id=seg_id if seg_id != "_global" else None,
            )
            summaries_added += 1

        # Keep the preserved entries + compact summary, drop originals
        preserved.extend(kept)

    # Replace entries with compacted set
    # The new summary entries were already appended to ledger.entries by ledger_append,
    # so we need to include those too
    new_entries = ledger.entries[original_count:]  # The summaries we just added
    ledger.entries = preserved + new_entries

    try:
        save_ledger(ledger, job_dir)
    except Exception as exc:
        logger.warning("[ledger_compactor] Failed to save compacted ledger: %s", exc)

    reduction = original_count - len(ledger.entries)
    if emit and reduction > 0:
        emit(
            f"  📦 Ledger compacted: {original_count} → {len(ledger.entries)} entries "
            f"(-{reduction}, +{summaries_added} summaries)"
        )
    logger.info(
        "[ledger_compactor] Compacted: %d → %d entries (-%d, +%d summaries)",
        original_count, len(ledger.entries), reduction, summaries_added,
    )

    return reduction
