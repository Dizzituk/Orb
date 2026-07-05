# FILE: app/experience/retrieval.py
# Purpose: Two-stage retrieval for the Experience Database.
# Called-by: app.llm.critical_pipeline.stream_handler
# Depends-on: app.embeddings.models, app.embeddings.service, app.experience.experience_store, app.experience.models
# Last-renovated: 2026-06-11
"""
Two-stage retrieval for the Experience Database.

Stage 1: Indexed filter (SQL query on category, stage, job_type, etc.)
Stage 2: Semantic ranking (embedding similarity on pre-filtered candidates)

Section 4 of the Unified Memory System v3.0 spec.

Usage:
    results = retrieve_for_stage(
        db=db,
        stage="critical_pipeline",
        context="Generating architecture for refactor of architecture_executor.py",
        job_type="refactor",
        language="python",
        max_results=8,
    )
    
    injection_text = format_injection(results, max_tokens=2000)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from .experience_store import query_patterns, record_injection
from .models import ExperiencePattern

logger = logging.getLogger(__name__)


# =============================================================================
# TOKEN BUDGETS PER STAGE (Section 4.3)
# =============================================================================

STAGE_TOKEN_BUDGETS = {
    "critical_pipeline": 3000,
    "critique": 1500,
    "cohesion_check": 2000,
    "implementer": 1500,
    "overwatcher": 1500,
    "phase_checkout": 800,
    "final_checkout": 2000,
}

DEFAULT_TOKEN_BUDGET = 1500


# =============================================================================
# RETRIEVE FOR STAGE
# =============================================================================

def retrieve_for_stage(
    db: Session,
    *,
    stage: str,
    context: str = "",
    job_type: Optional[str] = None,
    language: Optional[str] = None,
    error_signature: Optional[str] = None,
    max_results: int = 8,
    min_confidence: float = 0.3,
) -> List[ExperiencePattern]:
    """
    Two-stage retrieval for a pipeline stage.

    Stage 1: Indexed filter using SQL (fast).
    Stage 2: Semantic ranking using embedding similarity (if embeddings available).

    If semantic ranking is not available (no embeddings), falls back to
    confidence-sorted results from Stage 1.

    Args:
        db: SQLAlchemy session
        stage: Pipeline stage requesting patterns
        context: Semantic context string for ranking
        job_type: Job type filter (also includes universal patterns)
        language: Language filter
        error_signature: Exact-match for boot fix patterns
        max_results: Max patterns to return
        min_confidence: Minimum confidence threshold
    """
    # === Stage 1: Indexed filter ===
    # Build stage filter based on what the calling stage needs
    stage_filters = _get_stage_query_config(stage)

    candidates = query_patterns(
        db,
        stages=stage_filters.get("stages"),
        categories=stage_filters.get("categories"),
        job_type=job_type,
        language=language,
        error_signature=error_signature,
        min_confidence=min_confidence,
        limit=50,  # Pre-filter cap
    )

    if not candidates:
        return []

    logger.debug(
        f"[retrieval] Stage 1 for {stage}: {len(candidates)} candidates "
        f"(job_type={job_type}, language={language})"
    )

    # === Stage 2: Semantic ranking ===
    if context and len(candidates) > max_results:
        ranked = _semantic_rank(db, candidates, context, max_results)
    else:
        # No semantic ranking needed — just take top N by confidence
        ranked = candidates[:max_results]

    # Record that these patterns were retrieved (for tracking)
    for pattern in ranked:
        try:
            record_injection(db, pattern.id)
        except Exception:
            pass

    logger.info(
        f"[retrieval] {stage}: returning {len(ranked)} patterns "
        f"(from {len(candidates)} candidates)"
    )

    return ranked


# =============================================================================
# STAGE QUERY CONFIGURATIONS (Section 4.2)
# =============================================================================

def _get_stage_query_config(stage: str) -> Dict[str, Any]:
    """
    Get the query configuration for a specific pipeline stage.

    Defines which stages and categories each pipeline stage queries.
    Section 4.2 of the spec.
    """
    configs = {
        "critical_pipeline": {
            "stages": ["critical_pipeline", "critique", "cohesion_check"],
            "categories": [
                "architecture_decision",
                "error_fix",
                "implementation_strategy",
            ],
        },
        "critique": {
            "stages": ["critique", "critical_pipeline"],
            "categories": [
                "false_positive",
                "architecture_decision",
                "error_fix",
            ],
        },
        "cohesion_check": {
            "stages": ["cohesion_check", "critical_pipeline", "implementer"],
            "categories": [
                "error_fix",
                "architecture_decision",
            ],
        },
        "implementer": {
            "stages": ["implementer", "phase_checkout"],
            "categories": [
                "error_fix",
                "implementation_strategy",
                "boot_resolution",
            ],
        },
        "overwatcher": {
            "stages": ["implementer", "overwatcher", "phase_checkout"],
            "categories": [
                "error_fix",
                "implementation_strategy",
                "boot_resolution",
            ],
        },
        "phase_checkout": {
            "stages": ["phase_checkout"],
            "categories": [
                "boot_resolution",
                "error_fix",
            ],
        },
        "final_checkout": {
            "stages": [
                "final_checkout", "phase_checkout",
                "implementer", "critical_pipeline",
            ],
            "categories": [
                "error_fix",
                "performance_pattern",
                "boot_resolution",
            ],
        },
    }

    return configs.get(stage, {
        "stages": [stage],
        "categories": None,  # No category filter
    })


# =============================================================================
# SEMANTIC RANKING (Stage 2)
# =============================================================================

def _semantic_rank(
    db: Session,
    candidates: List[ExperiencePattern],
    context: str,
    max_results: int,
) -> List[ExperiencePattern]:
    """
    Rank candidates by embedding similarity to the context string.

    Falls back to confidence-based ordering if embeddings aren't available.
    """
    try:
        from app.embeddings.service import generate_embedding, cosine_similarity

        # Generate embedding for the context
        context_embedding = generate_embedding(context, task_type="RETRIEVAL_QUERY")
        if not context_embedding:
            return candidates[:max_results]

        # Get embeddings for candidates
        scored = []
        for pattern in candidates:
            if pattern.embedded:
                # Try to get the stored embedding
                pattern_embedding = _get_pattern_embedding(db, pattern.id)
                if pattern_embedding:
                    sim = cosine_similarity(context_embedding, pattern_embedding)
                    # Combine semantic similarity with confidence for final score
                    # 60% semantic, 40% confidence
                    combined = (sim * 0.6) + (pattern.confidence * 0.4)
                    scored.append((pattern, combined))
                    continue

            # No embedding available — use confidence only
            scored.append((pattern, pattern.confidence))

        # Sort by combined score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:max_results]]

    except ImportError:
        logger.debug("[retrieval] Embedding service not available, using confidence ranking")
        return candidates[:max_results]
    except Exception as e:
        logger.warning(f"[retrieval] Semantic ranking failed: {e}")
        return candidates[:max_results]


def _get_pattern_embedding(
    db: Session,
    pattern_id: int,
) -> Optional[List[float]]:
    """Get the stored embedding vector for a pattern.

    LANE E: only vectors in the ACTIVE query model-space are returned — the
    context embedding above rides the router's query path, so scoring a
    different-model row would be mixed-model cosine. Legacy rows become
    visible again once reembed_batch restamps them.
    """
    try:
        from app.embeddings.models import Embedding
        from app.embeddings.provider_router import text_query_spec
        from app.embeddings.service import _model_space_criterion

        emb = db.query(Embedding).filter(
            Embedding.source_type == "experience_pattern",
            Embedding.source_id == pattern_id,
            _model_space_criterion(text_query_spec().label),
        ).first()

        if emb and emb.embedding:
            import json
            return json.loads(emb.embedding) if isinstance(emb.embedding, str) else emb.embedding
    except Exception:
        pass
    return None


# =============================================================================
# INJECTION FORMATTING (Section 4.3)
# =============================================================================

def format_injection(
    patterns: List[ExperiencePattern],
    max_tokens: Optional[int] = None,
    stage: Optional[str] = None,
) -> str:
    """
    Format patterns for injection into an LLM prompt.

    Patterns are formatted based on confidence level:
    - ≥0.8: CONSTRAINT (hard rule)
    - 0.5–0.8: HINT (suggestion)
    - 0.3–0.5: BACKGROUND (low priority)
    - <0.3: SKIP

    Fills the token budget in priority order:
    CONSTRAINT → CONTEXT → HINT → BACKGROUND

    Returns:
        Formatted string ready for prompt injection, or empty string.
    """
    if not patterns:
        return ""

    budget = max_tokens or STAGE_TOKEN_BUDGETS.get(stage or "", DEFAULT_TOKEN_BUDGET)

    # Sort by injection priority (CONSTRAINT first)
    constraints = []
    hints = []
    backgrounds = []

    for p in patterns:
        tier = p.injection_tier
        if tier == "CONSTRAINT":
            constraints.append(p)
        elif tier == "HINT":
            hints.append(p)
        elif tier == "BACKGROUND":
            backgrounds.append(p)
        # SKIP tier is ignored

    # Build injection blocks in priority order
    blocks = []
    tokens_used = 0

    if constraints:
        blocks.append("## EXPERIENCE CONSTRAINTS (MUST follow)")
        tokens_used += 10
        for p in constraints:
            block = _format_single_pattern(p, "CONSTRAINT")
            block_tokens = len(block.split()) * 1.3  # Rough token estimate
            if tokens_used + block_tokens > budget:
                break
            blocks.append(block)
            tokens_used += block_tokens

    if hints:
        blocks.append("\n## EXPERIENCE HINTS (SHOULD consider)")
        tokens_used += 10
        for p in hints:
            block = _format_single_pattern(p, "HINT")
            block_tokens = len(block.split()) * 1.3
            if tokens_used + block_tokens > budget:
                break
            blocks.append(block)
            tokens_used += block_tokens

    if backgrounds and tokens_used < budget * 0.8:
        blocks.append("\n## BACKGROUND CONTEXT (for reference)")
        tokens_used += 10
        for p in backgrounds:
            block = _format_single_pattern(p, "BACKGROUND")
            block_tokens = len(block.split()) * 1.3
            if tokens_used + block_tokens > budget:
                break
            blocks.append(block)
            tokens_used += block_tokens

    if not blocks:
        return ""

    return "\n".join(blocks)


def _format_single_pattern(pattern: ExperiencePattern, tier: str) -> str:
    """Format a single pattern for injection."""
    lines = []

    if tier == "CONSTRAINT":
        lines.append(f"- **MUST:** {pattern.description}")
        if pattern.resolution:
            lines.append(f"  Resolution: {pattern.resolution}")
    elif tier == "HINT":
        lines.append(f"- **CONSIDER:** {pattern.description}")
        if pattern.resolution:
            lines.append(f"  Suggestion: {pattern.resolution}")
    else:  # BACKGROUND
        lines.append(f"- Note: {pattern.description}")

    if pattern.root_cause:
        lines.append(f"  (Root cause: {pattern.root_cause[:100]})")

    return "\n".join(lines)
