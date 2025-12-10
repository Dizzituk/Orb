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
RELATIONSHIP_DETECTION_ENABLED = os.getenv("ORB_RELATIONSHIP_DETECTION", "1") == "1"

# Maximum summary length for efficiency
MAX_SUMMARY_CHARS = 600

# Router debug mode
ROUTER_DEBUG = os.getenv("ORB_ROUTER_DEBUG", "0") == "1"


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

def build_text_summary(
    text_files: List[Any],
    mixed_files: List[Any],
    max_chars: int = MAX_SUMMARY_CHARS,
) -> str:
    """
    Build text summary from TEXT_FILE and text parts of MIXED_FILE.
    
    Args:
        text_files: List of ClassifiedFile with file_type == TEXT_FILE
        mixed_files: List of ClassifiedFile with file_type == MIXED_FILE
        max_chars: Maximum summary length
    
    Returns:
        Summary string
    """
    parts = []
    
    # Add text file info
    for tf in text_files[:5]:  # Limit to 5 files
        name = getattr(tf, "original_name", str(tf))
        text = getattr(tf, "extracted_text", "") or ""
        if text:
            parts.append(f"[{name}]: {text[:200]}")
        else:
            parts.append(f"[{name}]")
    
    # Add mixed file text parts
    for mf in mixed_files[:3]:
        name = getattr(mf, "original_name", str(mf))
        text = getattr(mf, "extracted_text", "") or ""
        if text:
            parts.append(f"[{name}]: {text[:150]}")
        else:
            parts.append(f"[{name}] (mixed: text+images)")
    
    summary = "\n".join(parts)
    return summary[:max_chars] if len(summary) > max_chars else summary


def build_code_summary(
    code_files: List[Any],
    max_chars: int = MAX_SUMMARY_CHARS,
) -> str:
    """
    Build code summary from CODE_FILE.
    
    Focuses on:
    - File names (reveal purpose)
    - Module/class/function names
    - Comments and docstrings
    
    Args:
        code_files: List of ClassifiedFile with file_type == CODE_FILE
        max_chars: Maximum summary length
    
    Returns:
        Summary string
    """
    parts = []
    
    for cf in code_files[:5]:  # Limit to 5 files
        name = getattr(cf, "original_name", str(cf))
        code = getattr(cf, "extracted_text", "") or ""
        
        if code:
            # Extract key elements from code
            summary_parts = [f"[{name}]:"]
            
            # Look for class definitions
            classes = re.findall(r'class\s+(\w+)', code[:2000])
            if classes:
                summary_parts.append(f"classes: {', '.join(classes[:5])}")
            
            # Look for function definitions
            functions = re.findall(r'(?:def|function|async function)\s+(\w+)', code[:2000])
            if functions:
                summary_parts.append(f"functions: {', '.join(functions[:8])}")
            
            # Look for imports (reveal dependencies)
            imports = re.findall(r'(?:import|from)\s+([^\s;]+)', code[:1000])
            if imports:
                summary_parts.append(f"imports: {', '.join(set(imports[:5]))}")
            
            parts.append(" ".join(summary_parts))
        else:
            parts.append(f"[{name}]")
    
    summary = "\n".join(parts)
    return summary[:max_chars] if len(summary) > max_chars else summary


def build_image_summary(
    image_files: List[Any],
    mixed_files: List[Any],
    image_descriptions: Optional[Dict[str, str]] = None,
    max_chars: int = MAX_SUMMARY_CHARS,
) -> str:
    """
    Build image summary from IMAGE_FILE and images in MIXED_FILE.
    
    Args:
        image_files: List of ClassifiedFile with file_type == IMAGE_FILE
        mixed_files: List of ClassifiedFile with file_type == MIXED_FILE
        image_descriptions: Optional dict of file_id -> description
        max_chars: Maximum summary length
    
    Returns:
        Summary string
    """
    parts = []
    descriptions = image_descriptions or {}
    
    for img in image_files[:5]:
        file_id = getattr(img, "file_id", "")
        name = getattr(img, "original_name", str(img))
        
        if file_id in descriptions:
            parts.append(f"[{name}]: {descriptions[file_id][:150]}")
        else:
            parts.append(f"[{name}]")
    
    # Note mixed files with images
    for mf in mixed_files[:3]:
        name = getattr(mf, "original_name", str(mf))
        count = getattr(mf, "embedded_image_count", 0)
        parts.append(f"[{name}]: {count} embedded images")
    
    summary = "\n".join(parts)
    return summary[:max_chars] if len(summary) > max_chars else summary


def build_video_preteaser(
    video_files: List[Any],
    video_descriptions: Optional[Dict[str, str]] = None,
    max_chars: int = MAX_SUMMARY_CHARS,
) -> str:
    """
    Build video preteaser from VIDEO_FILE.
    
    This is a SHORT description (not full transcript).
    
    Args:
        video_files: List of ClassifiedFile with file_type == VIDEO_FILE
        video_descriptions: Optional dict of file_id -> short description
        max_chars: Maximum summary length
    
    Returns:
        Preteaser string
    """
    parts = []
    descriptions = video_descriptions or {}
    
    for vid in video_files[:3]:
        file_id = getattr(vid, "file_id", "")
        name = getattr(vid, "original_name", str(vid))
        size = getattr(vid, "size_bytes", 0)
        size_mb = size / (1024 * 1024) if size else 0
        
        if file_id in descriptions:
            parts.append(f"[{name}] ({size_mb:.1f}MB): {descriptions[file_id][:200]}")
        else:
            parts.append(f"[{name}] ({size_mb:.1f}MB)")
    
    summary = "\n".join(parts)
    return summary[:max_chars] if len(summary) > max_chars else summary


# =============================================================================
# HEURISTIC RELATIONSHIP DETECTION
# =============================================================================

def detect_relationships_heuristic(
    user_text: str,
    has_text: bool,
    has_code: bool,
    has_image: bool,
    has_video: bool,
    text_summary: str = "",
    code_summary: str = "",
    image_summary: str = "",
    video_preteaser: str = "",
) -> RelationshipResult:
    """
    Detect relationships using heuristics (no LLM call).
    
    This is faster but less accurate than LLM detection.
    
    Heuristics:
    - If user mentions "this image" + code context → IMAGE_CODE = related
    - If user mentions debugging/error + video → VIDEO_CODE = related
    - If only one modality present → relationships with absent modalities = unrelated
    - Default to "unclear" to be safe
    
    Args:
        user_text: User's message
        has_*: Modality presence flags
        *_summary: Content summaries
    
    Returns:
        RelationshipResult with heuristic-based relationships
    """
    result = RelationshipResult(detection_method="heuristic", confidence=0.6)
    
    user_lower = user_text.lower()
    
    # If modality not present, mark relationships with it as "unrelated"
    if not has_image:
        result.REL_IMAGE_TEXT = RelationshipType.UNRELATED
        result.REL_IMAGE_CODE = RelationshipType.UNRELATED
        result.REL_IMAGE_VIDEO = RelationshipType.UNRELATED
    
    if not has_video:
        result.REL_VIDEO_TEXT = RelationshipType.UNRELATED
        result.REL_VIDEO_CODE = RelationshipType.UNRELATED
        result.REL_IMAGE_VIDEO = RelationshipType.UNRELATED
    
    if not has_code:
        result.REL_IMAGE_CODE = RelationshipType.UNRELATED
        result.REL_VIDEO_CODE = RelationshipType.UNRELATED
        result.REL_CODE_TEXT = RelationshipType.UNRELATED
    
    if not has_text:
        result.REL_IMAGE_TEXT = RelationshipType.UNRELATED
        result.REL_VIDEO_TEXT = RelationshipType.UNRELATED
        result.REL_CODE_TEXT = RelationshipType.UNRELATED
    
    # Now detect positive relationships from user text
    
    # VIDEO + CODE relationship
    if has_video and has_code:
        video_code_keywords = [
            "debug", "error", "bug", "fix", "issue", "problem",
            "screen", "recording", "demo", "showing", "reproduce",
            "log", "console", "output", "trace", "stack",
        ]
        if any(kw in user_lower for kw in video_code_keywords):
            result.REL_VIDEO_CODE = RelationshipType.RELATED
            result.confidence = 0.8
            logger.debug("[relationship] VIDEO+CODE detected as related (debug keywords)")
    
    # IMAGE + CODE relationship
    if has_image and has_code:
        image_code_keywords = [
            "screenshot", "error", "ui", "interface", "design",
            "diagram", "flow", "architecture", "output", "result",
        ]
        if any(kw in user_lower for kw in image_code_keywords):
            result.REL_IMAGE_CODE = RelationshipType.RELATED
            result.confidence = 0.8
            logger.debug("[relationship] IMAGE+CODE detected as related (ui/error keywords)")
    
    # IMAGE + TEXT relationship
    if has_image and has_text:
        image_text_keywords = [
            "document", "pdf", "page", "figure", "illustration",
            "diagram", "chart", "table", "screenshot",
        ]
        if any(kw in user_lower for kw in image_text_keywords):
            result.REL_IMAGE_TEXT = RelationshipType.RELATED
            result.confidence = 0.75
            logger.debug("[relationship] IMAGE+TEXT detected as related (document keywords)")
    
    # VIDEO + TEXT relationship
    if has_video and has_text:
        video_text_keywords = [
            "tutorial", "guide", "documentation", "explain",
            "demo", "walkthrough", "manual",
        ]
        if any(kw in user_lower for kw in video_text_keywords):
            result.REL_VIDEO_TEXT = RelationshipType.RELATED
            result.confidence = 0.75
            logger.debug("[relationship] VIDEO+TEXT detected as related (tutorial keywords)")
    
    # CODE + TEXT relationship
    if has_code and has_text:
        code_text_keywords = [
            "spec", "requirement", "documentation", "readme",
            "api", "contract", "design", "architecture",
        ]
        if any(kw in user_lower for kw in code_text_keywords):
            result.REL_CODE_TEXT = RelationshipType.RELATED
            result.confidence = 0.8
            logger.debug("[relationship] CODE+TEXT detected as related (spec keywords)")
    
    # Store summaries
    result.text_summary = text_summary
    result.code_summary = code_summary
    result.image_summary = image_summary
    result.video_preteaser = video_preteaser
    
    return result


# =============================================================================
# LLM-BASED RELATIONSHIP DETECTION
# =============================================================================

RELATIONSHIP_PROMPT = """Analyze the relationship between different file types in this request.

USER REQUEST:
{user_text}

FILE MAP:
{file_map}

CONTENT SUMMARIES:
Text documents: {text_summary}
Code files: {code_summary}
Images: {image_summary}
Videos: {video_preteaser}

For each pair of modalities present, determine if they are:
- "related": The content is clearly connected and should be processed together
- "unrelated": The content is clearly separate and independent
- "unclear": Cannot determine, default to treating as potentially related

Respond ONLY with JSON (no other text):
{{
    "REL_IMAGE_TEXT": "related|unrelated|unclear",
    "REL_IMAGE_CODE": "related|unrelated|unclear",
    "REL_VIDEO_TEXT": "related|unrelated|unclear",
    "REL_VIDEO_CODE": "related|unrelated|unclear",
    "REL_CODE_TEXT": "related|unrelated|unclear",
    "REL_IMAGE_VIDEO": "related|unrelated|unclear"
}}"""


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