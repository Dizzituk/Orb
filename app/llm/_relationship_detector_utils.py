import logging
import os
import re
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
MAX_SUMMARY_CHARS = 600


RELATIONSHIP_DETECTION_ENABLED = os.getenv("ORB_RELATIONSHIP_DETECTION", "1") == "1"

ROUTER_DEBUG = os.getenv("ORB_ROUTER_DEBUG", "0") == "1"

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
