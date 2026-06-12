# FILE: app/memory/complexity_router.py
# Purpose: Complexity-aware model routing (Spec Section 10, Job 6B).
# Called-by: app.memory.integration
# Depends-on: app.memory.complexity
# Last-renovated: 2026-06-11
"""
Complexity-aware model routing (Spec Section 10, Job 6B).

Bridges the complexity classifier (complexity.py) and escalation
pathway (escalation.py) with the existing LLM job routing system
(app/llm/routing/job_routing.py).

The existing 8-route system classifies by CONTENT TYPE (text, code,
image, video). This module adds a DIFFICULTY dimension that adjusts
model selection within a content type:

    Content Type    | ping_pong  | lookup     | reasoning | deep
    ────────────────|────────────|────────────|───────────|──────────
    CHAT_LIGHT      | local      | local      | sonnet    | sonnet
    TEXT_HEAVY       | local_rag  | local_rag  | sonnet    | opus
    CODE_MEDIUM      | sonnet     | sonnet     | sonnet    | opus
    ORCHESTRATOR     | opus       | opus       | opus      | opus

The complexity router is called AFTER the job classifier but BEFORE
the provider is locked. It can upgrade the model tier but never
downgrade below what the job type requires.

Usage:
    from app.memory.complexity_router import route_with_complexity

    result = route_with_complexity(
        query="redesign the RAG memory architecture",
        job_type="ORCHESTRATOR",
    )
    # result["provider"] = "anthropic"
    # result["model_tier"] = "opus"
    # result["complexity"] = ComplexityResult(...)
"""

import logging
from typing import Optional

from app.memory.complexity import classify_complexity, ComplexityResult

logger = logging.getLogger(__name__)


# =========================================================================
# Minimum tier floors per job type
# =========================================================================

# These are the MINIMUM model tiers for each job type.
# Complexity can push UP but never below these floors.

JOB_TYPE_FLOORS = {
    "chat.light": "local",
    "text.heavy": "local_rag",
    "code.medium": "sonnet",
    "orchestrator": "opus",
    "image.simple": "specialist",
    "image.complex": "specialist",
    "video.heavy": "specialist",
    "opus.critic": "opus",
}

# Tier ordering for comparison
TIER_ORDER = {
    "local": 0,
    "local_rag": 1,
    "sonnet": 2,
    "opus": 3,
    "specialist": 4,
}


# =========================================================================
# Public API
# =========================================================================

def route_with_complexity(
    query: str,
    job_type: Optional[str] = None,
    intent: Optional[str] = None,
    attachments: Optional[list] = None,
    context_depth: int = 0,
    confidence_score: Optional[float] = None,
) -> dict:
    """
    Route a query considering both content type and complexity.

    Args:
        query: The user's input text.
        job_type: Existing job type from the 8-route classifier.
        intent: Resolved intent from translation layer.
        attachments: List of attachment paths/dicts.
        context_depth: Active conversation context entries.
        confidence_score: Translation layer confidence.

    Returns:
        Dict with:
            model_tier: Final model tier after complexity adjustment
            job_type: Original job type (unchanged)
            complexity: ComplexityResult from classifier
            was_upgraded: True if complexity pushed tier above floor
            floor_tier: The minimum tier from job type
    """
    # Step 1: Classify complexity
    complexity = classify_complexity(
        query=query,
        intent=intent,
        attachments=attachments,
        context_depth=context_depth,
        confidence_score=confidence_score,
    )

    # Step 2: Get the floor for this job type
    floor_tier = JOB_TYPE_FLOORS.get(job_type, "sonnet")
    floor_rank = TIER_ORDER.get(floor_tier, 2)

    # Step 3: Get the complexity-suggested tier
    complexity_tier = complexity.model_target
    complexity_rank = TIER_ORDER.get(complexity_tier, 2)

    # Step 4: Take the higher of floor and complexity
    if complexity_rank > floor_rank:
        final_tier = complexity_tier
        was_upgraded = True
    else:
        final_tier = floor_tier
        was_upgraded = False

    logger.debug(
        "[complexity_router] query='%s' job=%s floor=%s "
        "complexity=%s(%s) final=%s upgraded=%s",
        query[:40], job_type, floor_tier,
        complexity.tier, complexity_tier,
        final_tier, was_upgraded,
    )

    return {
        "model_tier": final_tier,
        "job_type": job_type,
        "complexity": complexity,
        "was_upgraded": was_upgraded,
        "floor_tier": floor_tier,
    }


def should_use_rag(
    complexity: ComplexityResult,
    job_type: Optional[str] = None,
) -> bool:
    """
    Determine if RAG memory injection is needed for this query.

    RAG is used for:
        - lookup tier (always)
        - reasoning tier (always)
        - deep tier (always)
        - Any code/architecture job type

    RAG is skipped for:
        - ping_pong (trivial, no context needed)
        - multimodal (specialist handles own context)
    """
    if complexity.needs_rag:
        return True

    # Code and architecture jobs always benefit from RAG
    if job_type in ("code.medium", "orchestrator", "text.heavy"):
        return True

    return False


def get_rag_depth_for_tier(tier: str) -> int:
    """
    Get the RAG retrieval depth for a complexity tier.

    Higher tiers get deeper memory retrieval:
        ping_pong → 0 results (no RAG)
        lookup    → 5 results
        reasoning → 10 results
        deep      → 20 results
    """
    depths = {
        "ping_pong": 0,
        "lookup": 5,
        "reasoning": 10,
        "deep": 20,
        "multimodal": 5,
    }
    return depths.get(tier, 10)
