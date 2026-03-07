# FILE: app/agentic_pipeline/batch_sizer.py
"""
Batch Sizer — Pre-flight calculation for agentic loop batching.

Determines whether all segments fit in one context window pass or
need splitting into batches (the '3D printer' approach). Each batch
gets its own agentic loop run; earlier batches' locked exports become
immutable facts for later batches.

Budget: 250K tokens input max. At ~5K tokens per segment, ~47 segments
per batch. Most jobs fit in one pass.

v1.0 (2026-03-05): Initial implementation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Token budget constants
INPUT_TOKEN_BUDGET = 250_000
HEADROOM_TOKENS = 750_000  # Available for multi-turn loop

# Base overhead: arch map + skeleton contract + manifest + experience patterns
# Real measurements from sg-bc6118fe (13 segments):
#   Arch map: ~17.7K tokens (70.9KB)
#   Skeleton contract: ~7.5K tokens (29.3KB) — scales with segment count
#   Manifest: ~5K tokens (19.5KB) — scales with segment count
#   Experience patterns: ~750 tokens
BASE_OVERHEAD_TOKENS = 25_000  # arch map (~18K) + experience (~1K) + system prompt (~6K)
SKELETON_TOKENS_PER_SEGMENT = 600  # ~7.5K / 13 segments from real data
MANIFEST_TOKENS_PER_SEGMENT = 400  # ~5K / 13 segments from real data

# Per-segment costs (from real job analysis)
TEMPLATE_TOKENS_PER_SEGMENT = 250  # deterministic template is compact
SEGMENT_SPEC_TOKENS = 400  # per-segment spec from manifest
MODIFY_FILE_TOKENS_PER_FILE = 5_000  # existing file content (~8KB avg, truncated to 12K)
PREFLIGHT_TOKENS_PER_FILE = 50  # one-line EXISTS/CREATE facts


@dataclass
class SegmentInfo:
    """Metadata for batch sizing calculations."""
    segment_id: str
    file_scope: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    estimated_tokens: int = 0
    modify_file_count: int = 0
    create_file_count: int = 0


@dataclass
class Batch:
    """A batch of segments to process in one agentic loop pass."""
    batch_index: int
    segment_ids: List[str] = field(default_factory=list)
    estimated_input_tokens: int = 0
    locked_exports_from: List[int] = field(default_factory=list)

    @property
    def segment_count(self) -> int:
        return len(self.segment_ids)


@dataclass
class BatchPlan:
    """Complete batching plan for a job."""
    batches: List[Batch] = field(default_factory=list)
    total_segments: int = 0
    fits_single_pass: bool = True
    reason: str = ""

    @property
    def batch_count(self) -> int:
        return len(self.batches)


def estimate_segment_tokens(
    segment: SegmentInfo,
) -> int:
    """Estimate input token cost for a single segment.

    Includes: skeleton share + manifest share + template + spec +
    MODIFY file content + preflight facts.
    """
    tokens = SKELETON_TOKENS_PER_SEGMENT  # per-segment share of skeleton contract
    tokens += MANIFEST_TOKENS_PER_SEGMENT  # per-segment share of manifest
    tokens += TEMPLATE_TOKENS_PER_SEGMENT  # deterministic template
    tokens += SEGMENT_SPEC_TOKENS  # per-segment spec
    tokens += segment.modify_file_count * MODIFY_FILE_TOKENS_PER_FILE
    tokens += len(segment.file_scope) * PREFLIGHT_TOKENS_PER_FILE
    return tokens


def _build_dependency_graph(
    segments: List[SegmentInfo],
) -> Dict[str, set]:
    """Build segment_id -> set of dependency segment_ids."""
    all_ids = {s.segment_id for s in segments}
    graph: Dict[str, set] = {}
    for seg in segments:
        deps = {d for d in seg.dependencies if d in all_ids}
        graph[seg.segment_id] = deps
    return graph


def _find_batch_boundary(
    segments: List[SegmentInfo],
    dep_graph: Dict[str, set],
    token_budget: int,
) -> int:
    """Find how many segments fit in the budget, respecting dependencies.

    Returns the count of segments (from the front of the list) that
    fit within the token budget. Will not split a dependency group —
    if segment B depends on segment A, both go in the same batch.

    Returns at least 1 (even if over budget, we process at minimum
    one segment per batch to guarantee progress).
    """
    running_tokens = BASE_OVERHEAD_TOKENS
    included: set = set()

    for i, seg in enumerate(segments):
        seg_tokens = seg.estimated_tokens or estimate_segment_tokens(seg)

        # Check if adding this segment (and any unseen deps) fits
        needed = seg_tokens
        pending_deps = seg.dependencies
        for dep_id in pending_deps:
            if dep_id not in included:
                dep_seg = next(
                    (s for s in segments if s.segment_id == dep_id), None
                )
                if dep_seg:
                    needed += dep_seg.estimated_tokens or estimate_segment_tokens(dep_seg)

        if running_tokens + needed > token_budget and included:
            return len(included)

        running_tokens += seg_tokens
        included.add(seg.segment_id)

    return len(included)


def compute_batch_plan(
    segments: List[SegmentInfo],
    preflight_facts: Optional[Dict[str, Any]] = None,
) -> BatchPlan:
    """Compute the batching plan for a job.

    Args:
        segments: List of SegmentInfo with file_scope and dependencies.
        preflight_facts: Optional pre-flight data to refine MODIFY counts.

    Returns:
        BatchPlan with one or more batches.
    """
    if not segments:
        return BatchPlan(reason="no segments")

    # Enrich segments with token estimates
    for seg in segments:
        if preflight_facts:
            modify_count = sum(
                1 for f in seg.file_scope
                if preflight_facts.get(f, {}).get("exists", False)
            )
            seg.modify_file_count = modify_count
            seg.create_file_count = len(seg.file_scope) - modify_count
        else:
            # Conservative: assume half are MODIFY
            seg.modify_file_count = len(seg.file_scope) // 2
            seg.create_file_count = len(seg.file_scope) - seg.modify_file_count

        seg.estimated_tokens = estimate_segment_tokens(seg)

    total_tokens = BASE_OVERHEAD_TOKENS + sum(s.estimated_tokens for s in segments)

    # Single-pass check
    if total_tokens <= INPUT_TOKEN_BUDGET:
        batch = Batch(
            batch_index=0,
            segment_ids=[s.segment_id for s in segments],
            estimated_input_tokens=total_tokens,
        )
        plan = BatchPlan(
            batches=[batch],
            total_segments=len(segments),
            fits_single_pass=True,
            reason=f"all {len(segments)} segments fit ({total_tokens:,} tokens)",
        )
        logger.info(
            "[batch_sizer] Single-pass: %d segments, ~%d tokens",
            len(segments), total_tokens,
        )
        return plan

    # Multi-batch: split at dependency boundaries
    dep_graph = _build_dependency_graph(segments)
    remaining = list(segments)
    batches: List[Batch] = []
    batch_idx = 0

    while remaining:
        boundary = _find_batch_boundary(
            remaining, dep_graph, INPUT_TOKEN_BUDGET,
        )
        boundary = max(1, boundary)  # At least one segment per batch

        batch_segments = remaining[:boundary]
        remaining = remaining[boundary:]

        batch_tokens = BASE_OVERHEAD_TOKENS + sum(
            s.estimated_tokens for s in batch_segments
        )
        # Add locked exports overhead from prior batches
        if batches:
            batch_tokens += len(batches) * 2000  # ~2K per prior batch exports

        batch = Batch(
            batch_index=batch_idx,
            segment_ids=[s.segment_id for s in batch_segments],
            estimated_input_tokens=batch_tokens,
            locked_exports_from=list(range(batch_idx)),
        )
        batches.append(batch)
        batch_idx += 1

    plan = BatchPlan(
        batches=batches,
        total_segments=len(segments),
        fits_single_pass=False,
        reason=(
            f"{len(segments)} segments split into {len(batches)} batches "
            f"(exceeded {INPUT_TOKEN_BUDGET:,} token budget)"
        ),
    )

    logger.info(
        "[batch_sizer] Multi-batch: %d segments -> %d batches",
        len(segments), len(batches),
    )
    for b in batches:
        logger.info(
            "[batch_sizer]   Batch %d: %d segments, ~%d tokens",
            b.batch_index, b.segment_count, b.estimated_input_tokens,
        )

    return plan
