# FILE: app/optimize/proposer.py
"""
Phase C: Proposer.

Generates ranked optimisation proposals from decomposition and
profiling data. Each proposal maps to one of the 8 categories
with predicted impact, risk level, and estimated token cost.

Proposals are ranked by impact-to-risk ratio. The user reviews
and approves before execution.

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-OPT-001.
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Callable, Dict, List, Optional

from app.optimize.config import (
    CATEGORY_RISK,
    FILE_SIZE_OVERSIZED_BYTES,
    FILE_SIZE_TARGET_KB,
    HIGH_COUPLING_THRESHOLD,
    MIN_PROPOSAL_CONFIDENCE,
)
from app.optimize.models import (
    Bottleneck,
    ChunkManifest,
    ChunkMetrics,
    DeadCodeCandidate,
    ProfileResult,
    Proposal,
    ProposalCategory,
    ProposalStatus,
    RiskLevel,
)

logger = logging.getLogger(__name__)


async def propose(
    manifest: ChunkManifest,
    profile_result: ProfileResult,
    emit: Optional[Callable[[str], None]] = None,
) -> List[Proposal]:
    """Run Phase C: Generate optimisation proposals.

    Args:
        manifest: ChunkManifest from Decompose.
        profile_result: ProfileResult from Profile.
        emit: Progress callback.

    Returns:
        Ranked list of Proposal objects.
    """
    emit = emit or (lambda msg: None)
    t_start = time.time()

    emit("💡 Propose: Generating optimisation proposals...")

    proposals: List[Proposal] = []

    # Category 1: Dead code removal
    dead_proposals = _propose_dead_code(manifest.dead_code)
    proposals.extend(dead_proposals)
    emit(f"   Dead code: {len(dead_proposals)} proposals")

    # Category 7: File split (oversized files)
    split_proposals = _propose_file_splits(manifest, profile_result)
    proposals.extend(split_proposals)
    emit(f"   File splits: {len(split_proposals)} proposals")

    # Category 8: Dependency cleanup
    dep_proposals = _propose_dependency_cleanup(manifest, profile_result)
    proposals.extend(dep_proposals)
    emit(f"   Dependency cleanup: {len(dep_proposals)} proposals")

    # Category 4: Logic simplification (high complexity)
    logic_proposals = _propose_logic_simplification(profile_result)
    proposals.extend(logic_proposals)
    emit(f"   Logic simplification: {len(logic_proposals)} proposals")

    # Category 6: Model call reduction (detect via code patterns)
    model_proposals = _propose_model_call_reduction(manifest)
    proposals.extend(model_proposals)
    emit(f"   Model call reduction: {len(model_proposals)} proposals")

    # Rank by impact-to-risk ratio
    proposals.sort(key=lambda p: p.impact_risk_ratio, reverse=True)

    # Assign IDs
    for i, p in enumerate(proposals):
        p.proposal_id = _generate_id(p, i)
        p.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")

    duration = time.time() - t_start
    emit(f"✅ Propose complete ({duration:.1f}s): {len(proposals)} proposals generated")

    return proposals


# ═══════════════════════════════════════════════════════════════════
# Category-specific proposers
# ═══════════════════════════════════════════════════════════════════

def _propose_dead_code(
    candidates: List[DeadCodeCandidate],
) -> List[Proposal]:
    """Generate proposals for dead code removal."""
    proposals = []
    for dc in candidates:
        if dc.confidence < MIN_PROPOSAL_CONFIDENCE:
            continue
        proposals.append(Proposal(
            category=ProposalCategory.DEAD_CODE,
            title=f"Remove dead {dc.item_type}: {dc.name}",
            description=(
                f"{dc.reason}. File: {dc.path}. "
                f"Confidence: {dc.confidence:.0%}."
            ),
            target_chunks=[dc.path],
            predicted_improvement="Reduced codebase size, cleaner imports",
            risk=RiskLevel.LOW,
            estimated_token_cost=0.01,
            confidence=dc.confidence,
            impact_score=0.3,
        ))
    return proposals


def _propose_file_splits(
    manifest: ChunkManifest,
    profile: ProfileResult,
) -> List[Proposal]:
    """Generate proposals for splitting oversized files."""
    proposals = []
    metrics_map = {m.path: m for m in profile.chunk_metrics}

    for chunk in manifest.chunks:
        if not chunk.is_oversized:
            continue

        metrics = metrics_map.get(chunk.path)
        size_kb = chunk.size_bytes // 1024
        complexity = metrics.cyclomatic_complexity if metrics else 0

        impact = min(1.0, size_kb / 100)  # Bigger = higher impact

        proposals.append(Proposal(
            category=ProposalCategory.FILE_SPLIT_MERGE,
            title=f"Split oversized file: {chunk.name} ({size_kb}KB)",
            description=(
                f"{chunk.path} is {size_kb}KB (target: {FILE_SIZE_TARGET_KB}KB). "
                f"Lines: {chunk.lines}. Complexity: {complexity}. "
                f"Split into focused submodules."
            ),
            target_chunks=[chunk.path],
            predicted_improvement=f"Reduce from {size_kb}KB to <{FILE_SIZE_TARGET_KB}KB per file",
            risk=RiskLevel.LOW,
            estimated_token_cost=0.05,
            confidence=0.85,
            impact_score=impact,
        ))

    return proposals


def _propose_dependency_cleanup(
    manifest: ChunkManifest,
    profile: ProfileResult,
) -> List[Proposal]:
    """Generate proposals for dependency cleanup."""
    proposals = []
    metrics_map = {m.path: m for m in profile.chunk_metrics}

    for chunk in manifest.chunks:
        metrics = metrics_map.get(chunk.path)
        if not metrics:
            continue

        total_coupling = metrics.coupling_in + metrics.coupling_out
        if total_coupling > HIGH_COUPLING_THRESHOLD:
            proposals.append(Proposal(
                category=ProposalCategory.DEPENDENCY_CLEANUP,
                title=f"Reduce coupling: {chunk.name}",
                description=(
                    f"{chunk.path} has {metrics.coupling_in} incoming and "
                    f"{metrics.coupling_out} outgoing dependencies "
                    f"(total: {total_coupling}). Consider introducing "
                    f"an interface or reducing direct imports."
                ),
                target_chunks=[chunk.path],
                predicted_improvement=f"Reduce coupling from {total_coupling} to <{HIGH_COUPLING_THRESHOLD}",
                risk=RiskLevel.MEDIUM,
                estimated_token_cost=0.08,
                confidence=0.6,
                impact_score=min(1.0, total_coupling / 50),
            ))

    return proposals


def _propose_logic_simplification(
    profile: ProfileResult,
) -> List[Proposal]:
    """Generate proposals for high-complexity files."""
    proposals = []

    for metrics in profile.chunk_metrics:
        if metrics.cyclomatic_complexity < 40:
            continue

        proposals.append(Proposal(
            category=ProposalCategory.LOGIC_SIMPLIFICATION,
            title=f"Simplify logic: {Path_stem(metrics.path)}",
            description=(
                f"{metrics.path} has cyclomatic complexity {metrics.cyclomatic_complexity}. "
                f"Consider extracting helper functions, replacing nested conditionals "
                f"with lookup tables, or decomposing into focused submodules."
            ),
            target_chunks=[metrics.path],
            predicted_improvement=f"Reduce complexity from {metrics.cyclomatic_complexity} to <30",
            risk=RiskLevel.MEDIUM,
            estimated_token_cost=0.10,
            confidence=0.65,
            impact_score=min(1.0, metrics.cyclomatic_complexity / 100),
        ))

    return proposals


def _propose_model_call_reduction(
    manifest: ChunkManifest,
) -> List[Proposal]:
    """Detect files likely making LLM calls that could be deterministic.

    Scans for patterns: call_llm, providers.registry, openai, anthropic.
    Files in routing/ and translation/ are prime candidates.
    """
    proposals = []
    llm_patterns = [
        "call_llm", "providers/registry", "classifier",
        "routing/chat_routing", "routing/core",
    ]

    for chunk in manifest.chunks:
        path_lower = chunk.path.lower()
        if not any(p in path_lower for p in llm_patterns):
            continue

        # Only propose for files with significant coupling
        if chunk.dependents < 3:
            continue

        proposals.append(Proposal(
            category=ProposalCategory.MODEL_CALL_REDUCTION,
            title=f"Evaluate LLM call reduction: {chunk.name}",
            description=(
                f"{chunk.path} appears to involve LLM routing/classification. "
                f"Profile actual call frequency and check if common cases "
                f"could be handled by deterministic rules."
            ),
            target_chunks=[chunk.path],
            predicted_improvement="Reduced latency and token cost for common paths",
            risk=RiskLevel.MEDIUM,
            estimated_token_cost=0.15,
            confidence=0.5,
            impact_score=0.7,
        ))

    return proposals


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _generate_id(proposal: Proposal, index: int) -> str:
    """Generate a unique proposal ID."""
    raw = f"{proposal.category.value}:{proposal.target_chunks}:{index}"
    return f"opt-{hashlib.sha256(raw.encode()).hexdigest()[:8]}"


def Path_stem(path: str) -> str:
    """Get the filename stem from a path string."""
    parts = path.replace("\\", "/").split("/")
    name = parts[-1] if parts else path
    return name.rsplit(".", 1)[0] if "." in name else name
