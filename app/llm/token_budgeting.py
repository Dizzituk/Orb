# FILE: app/llm/token_budgeting.py
"""
Token Budget Management for Orb Routing Pipeline.

Version: 1.0.0 - Critical Pipeline Spec Implementation

Implements Spec §7 (Token Budgeting and Truncation Strategy):
- Relative budget allocation per modality
- Truncation rules with priority order
- Separate budgets for critical vs non-critical jobs

BASE ALLOCATION (non-critical, Spec §7.1):
- 20% - User text + task metadata + file map
- 40% - Code (snippets + summary)
- 20% - Video transcript/summary
- 15% - Text docs (summaries/excerpts)
- 5%  - Image descriptions/OCR text

CRITICAL ALLOCATION (Spec §8.1):
- 20% - User text + file map + instructions
- 45% - Code (higher weight)
- 20% - Video summary/transcript
- 10% - Text docs
- 5%  - Images

TRUNCATION PRIORITY (Spec §7.2):
1. Always preserve: user query, task description, file map, minimal summaries
2. First truncate: raw code snippets (keep code_summary)
3. Then truncate: video transcript (keep video_summary)
4. Then truncate: text excerpts (keep text_summary)
5. Finally shorten: image descriptions (keep minimal labels)

NEVER DROP:
- User query
- Task id/description
- At least a minimal summary per modality

Usage:
    from app.llm.token_budgeting import (
        TokenBudget,
        allocate_budget,
        apply_truncation,
    )
    
    budget = allocate_budget(
        max_context_tokens=200000,
        is_critical=False,
        has_code=True,
        has_video=True,
    )
    
    truncated = apply_truncation(budget, content_dict)
"""

import os
import logging
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default max context tokens per model
DEFAULT_MAX_CONTEXT = {
    "gpt": 128000,
    "sonnet": 200000,
    "opus": 200000,
    "gemini2": 128000,
    "gemini3": 200000,
}

# Minimum tokens to preserve per section
MIN_USER_TOKENS = 500
MIN_FILE_MAP_TOKENS = 200
MIN_SUMMARY_TOKENS = 100

# Router debug mode
ROUTER_DEBUG = os.getenv("ORB_ROUTER_DEBUG", "0") == "1"


# =============================================================================
# BUDGET ALLOCATION PROFILES
# =============================================================================

@dataclass
class BudgetProfile:
    """Budget allocation profile (percentages)."""
    user_and_meta: float = 0.20
    code: float = 0.40
    video: float = 0.20
    text_docs: float = 0.15
    images: float = 0.05
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "user_and_meta": self.user_and_meta,
            "code": self.code,
            "video": self.video,
            "text_docs": self.text_docs,
            "images": self.images,
        }


# Pre-defined profiles
PROFILE_DEFAULT = BudgetProfile(
    user_and_meta=0.20,
    code=0.40,
    video=0.20,
    text_docs=0.15,
    images=0.05,
)

PROFILE_CRITICAL = BudgetProfile(
    user_and_meta=0.20,
    code=0.45,
    video=0.20,
    text_docs=0.10,
    images=0.05,
)

PROFILE_VIDEO_HEAVY = BudgetProfile(
    user_and_meta=0.15,
    code=0.20,
    video=0.45,
    text_docs=0.15,
    images=0.05,
)

PROFILE_CODE_ONLY = BudgetProfile(
    user_and_meta=0.20,
    code=0.70,
    video=0.0,
    text_docs=0.10,
    images=0.0,
)

PROFILE_TEXT_ONLY = BudgetProfile(
    user_and_meta=0.25,
    code=0.0,
    video=0.0,
    text_docs=0.70,
    images=0.05,
)


# =============================================================================
# TOKEN BUDGET
# =============================================================================

@dataclass
class TokenBudget:
    """
    Token budget allocation for a task.
    
    Contains both the allocated budgets and the actual usage.
    """
    # Total budget
    max_context_tokens: int
    
    # Profile used
    profile_name: str
    profile: BudgetProfile
    
    # Allocated tokens per section
    allocated_user_and_meta: int = 0
    allocated_code: int = 0
    allocated_video: int = 0
    allocated_text_docs: int = 0
    allocated_images: int = 0
    
    # Actual usage (filled after truncation)
    used_user_and_meta: int = 0
    used_code: int = 0
    used_video: int = 0
    used_text_docs: int = 0
    used_images: int = 0
    
    # Truncation info
    truncations_applied: List[str] = field(default_factory=list)
    over_budget: bool = False
    
    @property
    def total_allocated(self) -> int:
        return (
            self.allocated_user_and_meta +
            self.allocated_code +
            self.allocated_video +
            self.allocated_text_docs +
            self.allocated_images
        )
    
    @property
    def total_used(self) -> int:
        return (
            self.used_user_and_meta +
            self.used_code +
            self.used_video +
            self.used_text_docs +
            self.used_images
        )
    
    @property
    def remaining(self) -> int:
        return self.max_context_tokens - self.total_used
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_context_tokens": self.max_context_tokens,
            "used_tokens_estimate": self.total_used,
            "profile": self.profile_name,
            "allocation": {
                "user_and_meta": self.allocated_user_and_meta,
                "code": self.allocated_code,
                "video": self.allocated_video,
                "text_docs": self.allocated_text_docs,
                "images": self.allocated_images,
            },
            "usage": {
                "user_and_meta": self.used_user_and_meta,
                "code": self.used_code,
                "video": self.used_video,
                "text_docs": self.used_text_docs,
                "images": self.used_images,
            },
            "truncations": self.truncations_applied,
            "over_budget": self.over_budget,
        }


# =============================================================================
# BUDGET ALLOCATION
# =============================================================================

def select_profile(
    is_critical: bool = False,
    has_code: bool = False,
    has_video: bool = False,
    has_text: bool = False,
    has_image: bool = False,
) -> Tuple[str, BudgetProfile]:
    """
    Select appropriate budget profile based on content.
    
    Args:
        is_critical: Is this a critical pipeline task?
        has_*: Modality presence flags
    
    Returns:
        (profile_name, BudgetProfile)
    """
    # Critical overrides other profiles
    if is_critical:
        return ("critical", PROFILE_CRITICAL)
    
    # Count present modalities
    modality_count = sum([has_code, has_video, has_text, has_image])
    
    # Single modality special cases
    if modality_count == 1:
        if has_code:
            return ("code_only", PROFILE_CODE_ONLY)
        if has_text:
            return ("text_only", PROFILE_TEXT_ONLY)
    
    # Video-heavy (video + anything)
    if has_video and has_code:
        return ("video_heavy", PROFILE_VIDEO_HEAVY)
    
    # Default profile
    return ("default", PROFILE_DEFAULT)


def allocate_budget(
    max_context_tokens: int,
    is_critical: bool = False,
    has_code: bool = False,
    has_video: bool = False,
    has_text: bool = False,
    has_image: bool = False,
    profile_override: Optional[BudgetProfile] = None,
) -> TokenBudget:
    """
    Allocate token budget for a task.
    
    Args:
        max_context_tokens: Maximum context tokens for target model
        is_critical: Is this a critical pipeline task?
        has_*: Modality presence flags
        profile_override: Optional custom profile
    
    Returns:
        TokenBudget with allocations
    """
    # Select or use override profile
    if profile_override:
        profile_name = "custom"
        profile = profile_override
    else:
        profile_name, profile = select_profile(
            is_critical=is_critical,
            has_code=has_code,
            has_video=has_video,
            has_text=has_text,
            has_image=has_image,
        )
    
    budget = TokenBudget(
        max_context_tokens=max_context_tokens,
        profile_name=profile_name,
        profile=profile,
    )
    
    # Allocate based on profile percentages
    budget.allocated_user_and_meta = int(max_context_tokens * profile.user_and_meta)
    budget.allocated_code = int(max_context_tokens * profile.code) if has_code else 0
    budget.allocated_video = int(max_context_tokens * profile.video) if has_video else 0
    budget.allocated_text_docs = int(max_context_tokens * profile.text_docs) if has_text else 0
    budget.allocated_images = int(max_context_tokens * profile.images) if has_image else 0
    
    # Redistribute unused allocations
    unused = max_context_tokens - budget.total_allocated
    if unused > 0:
        # Give unused to code first, then text
        if has_code:
            budget.allocated_code += unused
        elif has_text:
            budget.allocated_text_docs += unused
        else:
            budget.allocated_user_and_meta += unused
    
    logger.debug(
        f"[token_budget] Allocated {max_context_tokens} tokens with profile '{profile_name}': "
        f"user={budget.allocated_user_and_meta}, code={budget.allocated_code}, "
        f"video={budget.allocated_video}, text={budget.allocated_text_docs}, "
        f"images={budget.allocated_images}"
    )
    
    return budget


# =============================================================================
# TOKEN ESTIMATION
# =============================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    
    Uses rough heuristic: ~4 characters per token for English.
    """
    if not text:
        return 0
    return len(text) // 4


def estimate_content_tokens(content: Dict[str, Any]) -> Dict[str, int]:
    """
    Estimate tokens for all content sections.
    
    Args:
        content: Dict with keys:
            - user_text: str
            - file_map: str
            - code_content: str
            - video_content: str
            - text_content: str
            - image_content: str
    
    Returns:
        Dict with token estimates per section
    """
    return {
        "user_and_meta": estimate_tokens(
            content.get("user_text", "") + content.get("file_map", "")
        ),
        "code": estimate_tokens(content.get("code_content", "")),
        "video": estimate_tokens(content.get("video_content", "")),
        "text_docs": estimate_tokens(content.get("text_content", "")),
        "images": estimate_tokens(content.get("image_content", "")),
    }


# =============================================================================
# TRUNCATION
# =============================================================================

class TruncationLevel(str, Enum):
    """Truncation severity levels."""
    NONE = "none"
    LIGHT = "light"       # Trim excess, keep structure
    MODERATE = "moderate" # Summarize, remove details
    HEAVY = "heavy"       # Keep only essential summaries
    MINIMAL = "minimal"   # Bare minimum to preserve context


def truncate_text(
    text: str,
    max_tokens: int,
    keep_start: bool = True,
    keep_end: bool = False,
    min_tokens: int = MIN_SUMMARY_TOKENS,
) -> Tuple[str, int]:
    """
    Truncate text to fit token budget.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed
        keep_start: Keep beginning of text
        keep_end: Keep end of text
        min_tokens: Minimum tokens to preserve
    
    Returns:
        (truncated_text, actual_tokens)
    """
    if not text:
        return "", 0
    
    current_tokens = estimate_tokens(text)
    
    if current_tokens <= max_tokens:
        return text, current_tokens
    
    # Calculate target chars
    max_chars = max(max_tokens * 4, min_tokens * 4)
    
    if keep_start and keep_end:
        # Keep both ends, cut middle
        half_chars = max_chars // 2
        truncated = text[:half_chars] + "\n...[truncated]...\n" + text[-half_chars:]
    elif keep_end:
        # Keep end
        truncated = "...[truncated]...\n" + text[-max_chars:]
    else:
        # Keep start (default)
        truncated = text[:max_chars] + "\n...[truncated]..."
    
    return truncated, estimate_tokens(truncated)


def apply_truncation(
    budget: TokenBudget,
    content: Dict[str, Any],
    preserve_user: bool = True,
) -> Dict[str, str]:
    """
    Apply truncation to content to fit within budget.
    
    Truncation priority order (Spec §7.2):
    1. Preserve: user query, file map, minimal summaries
    2. Truncate: raw code snippets first
    3. Then: video transcript
    4. Then: text excerpts
    5. Finally: image descriptions
    
    Args:
        budget: TokenBudget with allocations
        content: Dict with content strings
        preserve_user: Always preserve user content
    
    Returns:
        Dict with truncated content
    """
    result = {}
    
    # Estimate current usage
    estimates = estimate_content_tokens(content)
    
    # 1. User and meta (always preserve)
    user_text = content.get("user_text", "")
    file_map = content.get("file_map", "")
    
    if preserve_user:
        result["user_text"] = user_text
        result["file_map"] = file_map
        budget.used_user_and_meta = estimate_tokens(user_text + file_map)
    else:
        combined = user_text + "\n" + file_map
        truncated, tokens = truncate_text(combined, budget.allocated_user_and_meta)
        result["user_text"] = truncated
        result["file_map"] = ""  # Included in user_text
        budget.used_user_and_meta = tokens
    
    # 2. Code (truncate first if over)
    code_content = content.get("code_content", "")
    code_summary = content.get("code_summary", "")
    
    if estimates["code"] > budget.allocated_code:
        # Over budget: keep summary, truncate code
        if code_summary:
            truncated, tokens = truncate_text(
                code_summary + "\n\n" + code_content,
                budget.allocated_code,
            )
            budget.truncations_applied.append("code_content")
        else:
            truncated, tokens = truncate_text(code_content, budget.allocated_code)
            budget.truncations_applied.append("code_content")
        result["code_content"] = truncated
        budget.used_code = tokens
    else:
        result["code_content"] = code_content
        budget.used_code = estimates["code"]
    
    # 3. Video (truncate second)
    video_content = content.get("video_content", "")
    video_summary = content.get("video_summary", "")
    
    if estimates["video"] > budget.allocated_video:
        # Over budget: prefer summary
        if video_summary and estimate_tokens(video_summary) <= budget.allocated_video:
            result["video_content"] = video_summary
            budget.used_video = estimate_tokens(video_summary)
            budget.truncations_applied.append("video_transcript")
        else:
            truncated, tokens = truncate_text(
                video_summary or video_content,
                budget.allocated_video,
            )
            result["video_content"] = truncated
            budget.used_video = tokens
            budget.truncations_applied.append("video_content")
    else:
        result["video_content"] = video_content
        budget.used_video = estimates["video"]
    
    # 4. Text docs (truncate third)
    text_content = content.get("text_content", "")
    text_summary = content.get("text_summary", "")
    
    if estimates["text_docs"] > budget.allocated_text_docs:
        if text_summary and estimate_tokens(text_summary) <= budget.allocated_text_docs:
            result["text_content"] = text_summary
            budget.used_text_docs = estimate_tokens(text_summary)
            budget.truncations_applied.append("text_excerpts")
        else:
            truncated, tokens = truncate_text(
                text_summary or text_content,
                budget.allocated_text_docs,
            )
            result["text_content"] = truncated
            budget.used_text_docs = tokens
            budget.truncations_applied.append("text_content")
    else:
        result["text_content"] = text_content
        budget.used_text_docs = estimates["text_docs"]
    
    # 5. Images (truncate last)
    image_content = content.get("image_content", "")
    
    if estimates["images"] > budget.allocated_images:
        truncated, tokens = truncate_text(image_content, budget.allocated_images)
        result["image_content"] = truncated
        budget.used_images = tokens
        budget.truncations_applied.append("image_descriptions")
    else:
        result["image_content"] = image_content
        budget.used_images = estimates["images"]
    
    # Check if still over budget
    budget.over_budget = budget.total_used > budget.max_context_tokens
    
    if budget.over_budget:
        logger.warning(
            f"[token_budget] Still over budget after truncation: "
            f"{budget.total_used}/{budget.max_context_tokens}"
        )
    else:
        logger.debug(
            f"[token_budget] Final usage: {budget.total_used}/{budget.max_context_tokens} "
            f"({len(budget.truncations_applied)} truncations)"
        )
    
    return result


# =============================================================================
# MODEL CONTEXT LIMITS
# =============================================================================

def get_model_max_context(model_id: str) -> int:
    """
    Get maximum context tokens for a model.
    
    Args:
        model_id: Model identifier string
    
    Returns:
        Maximum context tokens
    """
    model_lower = model_id.lower()
    
    # OpenAI models
    if "gpt-4.1" in model_lower or "gpt-4o" in model_lower:
        return 128000
    if "gpt-4" in model_lower:
        return 128000
    if "gpt-3.5" in model_lower:
        return 16000
    
    # Anthropic models
    if "opus" in model_lower:
        return 200000
    if "sonnet" in model_lower:
        return 200000
    if "haiku" in model_lower:
        return 200000
    
    # Google models
    if "gemini-3" in model_lower:
        return 200000
    if "gemini-2.5" in model_lower:
        return 200000
    if "gemini-2" in model_lower:
        return 128000
    if "gemini" in model_lower:
        return 128000
    
    # Default
    return 100000


def create_budget_for_model(
    model_id: str,
    is_critical: bool = False,
    has_code: bool = False,
    has_video: bool = False,
    has_text: bool = False,
    has_image: bool = False,
) -> TokenBudget:
    """
    Create a token budget for a specific model.
    
    Convenience function that combines model limit lookup with budget allocation.
    
    Args:
        model_id: Target model identifier
        is_critical: Is this a critical pipeline task?
        has_*: Modality presence flags
    
    Returns:
        TokenBudget configured for the model
    """
    max_context = get_model_max_context(model_id)
    
    return allocate_budget(
        max_context_tokens=max_context,
        is_critical=is_critical,
        has_code=has_code,
        has_video=has_video,
        has_text=has_text,
        has_image=has_image,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    "BudgetProfile",
    "TokenBudget",
    "TruncationLevel",
    
    # Profiles
    "PROFILE_DEFAULT",
    "PROFILE_CRITICAL",
    "PROFILE_VIDEO_HEAVY",
    "PROFILE_CODE_ONLY",
    "PROFILE_TEXT_ONLY",
    
    # Functions
    "select_profile",
    "allocate_budget",
    "estimate_tokens",
    "estimate_content_tokens",
    "truncate_text",
    "apply_truncation",
    "get_model_max_context",
    "create_budget_for_model",
    
    # Constants
    "DEFAULT_MAX_CONTEXT",
    "MIN_USER_TOKENS",
    "MIN_FILE_MAP_TOKENS",
    "MIN_SUMMARY_TOKENS",
]