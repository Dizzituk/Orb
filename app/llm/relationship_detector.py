# FILE: app/llm/relationship_detector.py
"""
Relationship Detector for Orb Routing Pipeline.

Version: 1.0.0 - Critical Pipeline Spec Implementation

Implements Spec §3 (Relationship Detection):
- Pairwise relationship detection between modalities
- Uses lightweight summaries for efficient classification
- Determines if content is "related", "unrelated", or "unclear"

PAIRWISE RELATIONSHIPS:
- REL_IMAGE_TEXT: Are images related to text documents?
- REL_IMAGE_CODE: Are images related to code files?
- REL_VIDEO_TEXT: Is video related to text documents?
- REL_VIDEO_CODE: Is video related to code files?
- REL_CODE_TEXT: Is code related to text documents?
- REL_IMAGE_VIDEO: Are images related to video?

RELATIONSHIP VALUES:
- "related": Content is clearly connected, should be processed together
- "unrelated": Content is clearly separate, can be processed independently
- "unclear": Cannot determine, treat as potentially related

SUMMARY BUILDING (Spec §3.1):
- text_summary: From TEXT_FILE + text parts of MIXED_FILE
- code_summary: From CODE_FILE comments/structure
- image_summary: From quick OCR/describe calls
- video_preteaser: Short description of video content

Usage:
    from app.llm.relationship_detector import detect_relationships, RelationshipResult
from app.llm._relationship_detector_utils import RELATIONSHIP_DETECTION_ENABLED, RELATIONSHIP_PROMPT, ROUTER_DEBUG, build_code_summary, build_image_summary, build_text_summary, build_video_preteaser, detect_relationships_heuristic
    
    result = await detect_relationships(
        classification_result=classification,
        user_text=message,
        llm_call=cheap_llm_call,  # Use cheap model like gpt-4.1-mini
    )
"""

import os
import re
import json
import logging
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Enable relationship detection (can be disabled for simple cases)

# Maximum summary length for efficiency
MAX_SUMMARY_CHARS = 600

# Router debug mode


# =============================================================================
# RELATIONSHIP ENUM
# =============================================================================

class RelationshipType(str, Enum):
    """Pairwise relationship types."""
    RELATED = "related"
    UNRELATED = "unrelated"
    UNCLEAR = "unclear"


# =============================================================================
# RELATIONSHIP RESULT
# =============================================================================

@dataclass
class RelationshipResult:
    """
    Result of relationship detection.
    
    Contains all pairwise relationships between modalities.
    """
    # Pairwise relationships (Spec §3)
    REL_IMAGE_TEXT: RelationshipType = RelationshipType.UNCLEAR
    REL_IMAGE_CODE: RelationshipType = RelationshipType.UNCLEAR
    REL_VIDEO_TEXT: RelationshipType = RelationshipType.UNCLEAR
    REL_VIDEO_CODE: RelationshipType = RelationshipType.UNCLEAR
    REL_CODE_TEXT: RelationshipType = RelationshipType.UNCLEAR
    REL_IMAGE_VIDEO: RelationshipType = RelationshipType.UNCLEAR
    
    # Summaries used for detection
    text_summary: str = ""
    code_summary: str = ""
    image_summary: str = ""
    video_preteaser: str = ""
    
    # Metadata
    detection_model: Optional[str] = None
    detection_method: str = "heuristic"  # "heuristic" or "llm"
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "REL_IMAGE_TEXT": self.REL_IMAGE_TEXT.value,
            "REL_IMAGE_CODE": self.REL_IMAGE_CODE.value,
            "REL_VIDEO_TEXT": self.REL_VIDEO_TEXT.value,
            "REL_VIDEO_CODE": self.REL_VIDEO_CODE.value,
            "REL_CODE_TEXT": self.REL_CODE_TEXT.value,
            "REL_IMAGE_VIDEO": self.REL_IMAGE_VIDEO.value,
            "detection_method": self.detection_method,
            "confidence": self.confidence,
        }
    
    def get_relationship(self, rel_name: str) -> RelationshipType:
        """Get relationship by name."""
        return getattr(self, rel_name, RelationshipType.UNCLEAR)
    
    def is_related(self, rel_name: str) -> bool:
        """Check if a relationship is 'related' or 'unclear' (treat as related)."""
        rel = self.get_relationship(rel_name)
        return rel in (RelationshipType.RELATED, RelationshipType.UNCLEAR)
    
    def is_definitely_unrelated(self, rel_name: str) -> bool:
        """Check if a relationship is definitely 'unrelated'."""
        return self.get_relationship(rel_name) == RelationshipType.UNRELATED


# =============================================================================
# SUMMARY BUILDERS (Spec §3.1)
# =============================================================================


# =============================================================================
# HEURISTIC RELATIONSHIP DETECTION
# =============================================================================


# =============================================================================
# LLM-BASED RELATIONSHIP DETECTION
# =============================================================================


async def detect_relationships_llm(
    user_text: str,
    file_map: str,
    text_summary: str,
    code_summary: str,
    image_summary: str,
    video_preteaser: str,
    llm_call: Callable[[str], Awaitable[str]],
    model: str = "gpt-4.1-mini",
) -> RelationshipResult:
    """
    Detect relationships using LLM call.
    
    Args:
        user_text: User's message
        file_map: File map string
        *_summary: Content summaries
        llm_call: Async callable for LLM call
        model: Model used (for logging)
    
    Returns:
        RelationshipResult with LLM-based relationships
    """
    result = RelationshipResult(
        detection_method="llm",
        detection_model=model,
        text_summary=text_summary,
        code_summary=code_summary,
        image_summary=image_summary,
        video_preteaser=video_preteaser,
    )
    
    prompt = RELATIONSHIP_PROMPT.format(
        user_text=user_text[:1000],
        file_map=file_map[:1000],
        text_summary=text_summary or "(none)",
        code_summary=code_summary or "(none)",
        image_summary=image_summary or "(none)",
        video_preteaser=video_preteaser or "(none)",
    )
    
    try:
        response = await llm_call(prompt)
        
        # Parse JSON from response
        json_match = re.search(r'\{[^{}]+\}', response)
        if json_match:
            data = json.loads(json_match.group())
            
            # Map to RelationshipType
            for key in ["REL_IMAGE_TEXT", "REL_IMAGE_CODE", "REL_VIDEO_TEXT",
                       "REL_VIDEO_CODE", "REL_CODE_TEXT", "REL_IMAGE_VIDEO"]:
                value = data.get(key, "unclear").lower()
                if value == "related":
                    setattr(result, key, RelationshipType.RELATED)
                elif value == "unrelated":
                    setattr(result, key, RelationshipType.UNRELATED)
                else:
                    setattr(result, key, RelationshipType.UNCLEAR)
            
            result.confidence = 0.85
            logger.debug(f"[relationship] LLM detection complete: {result.to_dict()}")
        else:
            logger.warning("[relationship] Could not parse LLM response, falling back to heuristic")
            result.detection_method = "heuristic_fallback"
            result.confidence = 0.5
            
    except Exception as e:
        logger.warning(f"[relationship] LLM detection failed: {e}, falling back to heuristic")
        result.detection_method = "heuristic_fallback"
        result.confidence = 0.5
    
    return result


# =============================================================================
# MAIN DETECTION FUNCTION
# =============================================================================

async def detect_relationships(
    classification_result: Any,
    user_text: str,
    file_map: str = "",
    llm_call: Optional[Callable[[str], Awaitable[str]]] = None,
    image_descriptions: Optional[Dict[str, str]] = None,
    video_descriptions: Optional[Dict[str, str]] = None,
    use_llm: bool = True,
) -> RelationshipResult:
    """
    Detect pairwise relationships between modalities.
    
    This is the main entry point for relationship detection.
    
    Args:
        classification_result: ClassificationResult from file_classifier
        user_text: User's message
        file_map: File map string (from build_file_map)
        llm_call: Optional async callable for LLM-based detection
        image_descriptions: Optional dict of file_id -> image description
        video_descriptions: Optional dict of file_id -> video description
        use_llm: Whether to use LLM detection (requires llm_call)
    
    Returns:
        RelationshipResult with all pairwise relationships
    """
    # Extract from classification result
    text_files = getattr(classification_result, "text_files", [])
    code_files = getattr(classification_result, "code_files", [])
    image_files = getattr(classification_result, "image_files", [])
    video_files = getattr(classification_result, "video_files", [])
    mixed_files = getattr(classification_result, "mixed_files", [])
    
    has_text = getattr(classification_result, "has_text", len(text_files) > 0)
    has_code = getattr(classification_result, "has_code", len(code_files) > 0)
    has_image = getattr(classification_result, "has_image", len(image_files) > 0 or len(mixed_files) > 0)
    has_video = getattr(classification_result, "has_video", len(video_files) > 0)
    
    # Build summaries (Spec §3.1)
    text_summary = build_text_summary(text_files, mixed_files)
    code_summary = build_code_summary(code_files)
    image_summary = build_image_summary(image_files, mixed_files, image_descriptions)
    video_preteaser = build_video_preteaser(video_files, video_descriptions)
    
    # Decide detection method
    if use_llm and llm_call and RELATIONSHIP_DETECTION_ENABLED:
        # Only use LLM if we have multiple modalities
        modality_count = sum([has_text, has_code, has_image, has_video])
        if modality_count >= 2:
            logger.debug("[relationship] Using LLM detection (multiple modalities)")
            return await detect_relationships_llm(
                user_text=user_text,
                file_map=file_map,
                text_summary=text_summary,
                code_summary=code_summary,
                image_summary=image_summary,
                video_preteaser=video_preteaser,
                llm_call=llm_call,
            )
    
    # Use heuristic detection
    logger.debug("[relationship] Using heuristic detection")
    return detect_relationships_heuristic(
        user_text=user_text,
        has_text=has_text,
        has_code=has_code,
        has_image=has_image,
        has_video=has_video,
        text_summary=text_summary,
        code_summary=code_summary,
        image_summary=image_summary,
        video_preteaser=video_preteaser,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "RelationshipType",
    
    # Result
    "RelationshipResult",
    
    # Summary builders
    "build_text_summary",
    "build_code_summary",
    "build_image_summary",
    "build_video_preteaser",
    
    # Detection
    "detect_relationships",
    "detect_relationships_heuristic",
    "detect_relationships_llm",
    
    # Configuration
    "RELATIONSHIP_DETECTION_ENABLED",
    "MAX_SUMMARY_CHARS",
]