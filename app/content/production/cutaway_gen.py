# FILE: app/content/production/cutaway_gen.py
"""
Cutaway Concept Generator — AI-powered (Spec Section 7.1).

Generates visual concepts for cutaway segments:
- Animated explainers
- AI-generated realistic footage (Sora)
- Data visualisations
- Text overlays
- Reference footage suggestions

All concepts require user approval before generation
(supervised mode) unless autonomous threshold is met.
"""
import json
import logging
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session

from app.content.models import ContentPiece, ContentApproval

logger = logging.getLogger(__name__)

CUTAWAY_TYPES = {
    "animated_explainer": {
        "description": "Cartoon-style animation simplifying a concept",
        "cost_tier": "medium",
        "generation_tool": "motion_graphics_template",
    },
    "ai_video": {
        "description": "Photorealistic AI-generated footage (Sora)",
        "cost_tier": "high",
        "generation_tool": "sora_api",
    },
    "data_visualisation": {
        "description": "Charts, stats, comparisons as motion graphics",
        "cost_tier": "low",
        "generation_tool": "deterministic_render",
    },
    "text_overlay": {
        "description": "Key quotes or definitions displayed on screen",
        "cost_tier": "free",
        "generation_tool": "deterministic_render",
    },
    "reference_footage": {
        "description": "Stock or Creative Commons footage",
        "cost_tier": "low",
        "generation_tool": "stock_search",
    },
}

# ─── PROMPT ───

CUTAWAY_SYSTEM = """You are a video production assistant creating cutaway concepts.
For a given content piece, suggest visual cutaways to intercut with the anchor footage.

Rules:
- Each cutaway should illustrate or reinforce a specific point
- Vary the types (don't suggest all text overlays or all AI video)
- Keep descriptions specific enough to produce (not vague)
- Consider cost: text_overlay and data_visualisation are free/cheap, ai_video is expensive
- Suggest placement: which part of the argument each cutaway supports

For each cutaway, provide:
- type: animated_explainer | ai_video | data_visualisation | text_overlay | reference_footage
- description: What the visual shows (specific, actionable)
- placement_context: Which argument/point this supports
- duration_seconds: Suggested duration (2-8 seconds typically)
- priority: high | medium | low

Respond with JSON: {"cutaways": [...]}"""

CUTAWAY_USER = """Content piece: {title}
Category: {category}
Key arguments: {key_arguments}
Key excerpts: {key_excerpts}

Generate cutaway concepts for this content piece. Suggest 4-8 cutaways."""


async def generate_cutaway_concepts(
    db: Session,
    piece_id: str,
) -> List[Dict[str, Any]]:
    """
    Generate cutaway concepts for a content piece.
    Returns list of concept dicts to be stored on the piece.
    """
    piece = db.query(ContentPiece).get(piece_id)
    if not piece:
        raise ValueError(f"Content piece {piece_id} not found")

    # Build prompt context
    key_arguments = "\n".join(
        f"- {exc}" for exc in (piece.key_excerpts or [])[:5]
    ) or "No excerpts available"

    prompt = CUTAWAY_USER.format(
        title=piece.title,
        category=piece.content_category,
        key_arguments=key_arguments,
        key_excerpts=json.dumps(piece.key_excerpts or []),
    )

    # Use the shared LLM call helper from scout
    from app.content.scout import _llm_call_json

    result = await _llm_call_json(
        CUTAWAY_SYSTEM, prompt,
        model="gemini-2.5-flash",
        max_tokens=3000,
    )

    if not result or "cutaways" not in result:
        logger.warning(f"[cutaway_gen] No concepts generated for {piece_id}")
        return []

    # Enrich with metadata
    concepts = []
    for i, cut in enumerate(result["cutaways"]):
        cut_type = cut.get("type", "text_overlay")
        type_info = CUTAWAY_TYPES.get(cut_type, CUTAWAY_TYPES["text_overlay"])

        concepts.append({
            "index": i,
            "type": cut_type,
            "description": cut.get("description", ""),
            "placement_context": cut.get("placement_context", ""),
            "duration_seconds": cut.get("duration_seconds", 4),
            "priority": cut.get("priority", "medium"),
            "cost_tier": type_info["cost_tier"],
            "generation_tool": type_info["generation_tool"],
            "status": "proposed",  # proposed → approved → generated → placed
            "asset_path": None,
        })

    # Store on piece
    piece.cutaway_concepts = concepts
    db.commit()

    logger.info(
        f"[cutaway_gen] Generated {len(concepts)} concepts for "
        f"'{piece.title}'"
    )
    return concepts


def approve_cutaway(
    db: Session,
    piece_id: str,
    cutaway_index: int,
    decision: str = "approved",
    modifications: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Approve or reject a specific cutaway concept.
    Logs the decision for preference learning.
    """
    piece = db.query(ContentPiece).get(piece_id)
    if not piece:
        raise ValueError(f"Content piece {piece_id} not found")

    concepts = piece.cutaway_concepts or []
    if cutaway_index >= len(concepts):
        raise ValueError(f"Cutaway index {cutaway_index} out of range")

    concept = concepts[cutaway_index]

    if decision == "approved":
        concept["status"] = "approved"
        if modifications:
            concept.update(modifications)
    elif decision == "rejected":
        concept["status"] = "rejected"
    elif decision == "modified":
        concept["status"] = "approved"
        concept.update(modifications or {})

    concepts[cutaway_index] = concept
    piece.cutaway_concepts = concepts

    # Log approval for preference learning
    from app.content.service import log_approval
    log_approval(
        db, piece_id,
        approval_type="cutaway",
        decision=decision,
        proposed={"cutaway_index": cutaway_index, **concept},
        modifications=modifications,
    )

    db.commit()

    logger.info(
        f"[cutaway_gen] Cutaway {cutaway_index} {decision} for "
        f"'{piece.title}'"
    )
    return concept


def get_approved_cutaways(
    db: Session,
    piece_id: str,
) -> List[Dict[str, Any]]:
    """Get all approved cutaway concepts for a piece."""
    piece = db.query(ContentPiece).get(piece_id)
    if not piece:
        return []

    return [
        c for c in (piece.cutaway_concepts or [])
        if c.get("status") == "approved"
    ]


def check_autonomous_cutaway_eligibility(
    db: Session,
) -> Dict[str, Any]:
    """
    Check if cutaway generation can go autonomous.
    Requires >90% approval rate with 20+ decisions.
    """
    from app.content.service import get_approval_stats
    stats = get_approval_stats(db, approval_type="cutaway")
    return {
        "eligible": stats.get("autonomous_ready", False),
        "approval_rate": stats.get("approval_rate", 0.0),
        "total_decisions": stats.get("total", 0),
        "threshold": 0.9,
        "min_decisions": 20,
    }
