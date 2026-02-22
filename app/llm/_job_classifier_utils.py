from __future__ import annotations
from .schemas import JobType, Provider, RoutingConfig, RoutingDecision
from typing import Tuple


VIDEO_SIZE_THRESHOLD = 10 * 1024 * 1024

def _has_complex_vision_keywords(message_lower: str) -> bool:
    """Check for complex vision analysis keywords."""
    return any(kw in message_lower for kw in RoutingConfig.IMAGE_COMPLEX_KEYWORDS)

def _has_video_deep_analysis_keywords(message_lower: str) -> bool:
    """Check for deep semantic video analysis keywords."""
    from .job_classifier import VIDEO_DEEP_ANALYSIS_KEYWORDS
    return any(kw in message_lower for kw in VIDEO_DEEP_ANALYSIS_KEYWORDS)

def get_provider_for_job(job_type: JobType) -> Tuple[Provider, str]:
    """Get provider and model for a job type."""
    return RoutingConfig.get_routing(job_type)

def get_routing_for_job_type(job_type_str: str) -> RoutingDecision:
    """Get routing for a job type string."""
    from .job_classifier import _make_decision
    try:
        jt = JobType(job_type_str)
    except ValueError:
        jt = JobType.CHAT_LIGHT
    return _make_decision(jt, f"Direct lookup: {job_type_str}")

def is_vision_job(job_type: JobType) -> bool:
    """Check if job type requires vision capabilities."""
    return job_type in {
        JobType.IMAGE_SIMPLE,
        JobType.IMAGE_COMPLEX,
        JobType.VIDEO_HEAVY,
        JobType.OPUS_CRITIC,
        JobType.DOCUMENT_PDF_VISION,
    }

def is_claude_forbidden(job_type: JobType) -> bool:
    """Check if this job type MUST NOT go to Claude."""
    forbidden = {
        JobType.IMAGE_SIMPLE,
        JobType.IMAGE_COMPLEX,
        JobType.VIDEO_HEAVY,
        JobType.OPUS_CRITIC,
        JobType.DOCUMENT_PDF_VISION,
        JobType.SIMPLE_VISION,
        JobType.HEAVY_MULTIMODAL_CRITIQUE,
        JobType.IMAGE_ANALYSIS,
        JobType.SCREENSHOT_ANALYSIS,
        JobType.VIDEO_ANALYSIS,
        JobType.OCR,
        JobType.CV_PARSING,
    }
    return job_type in forbidden

def is_claude_allowed(job_type: JobType) -> bool:
    """Check if this job type is explicitly allowed to go to Claude."""
    allowed = {
        JobType.CODE_MEDIUM,
        JobType.ORCHESTRATOR,
        JobType.SMALL_CODE,
        JobType.BIG_ARCHITECTURE,
        JobType.COMPLEX_CODE_CHANGE,
        JobType.CODEGEN_FULL_FILE,
        JobType.ARCHITECTURE_DESIGN,
        JobType.CODE_REVIEW,
        JobType.SPEC_REVIEW,
        JobType.REFACTOR,
        JobType.IMPLEMENTATION_PLAN,
        JobType.HIGH_STAKES_INFRA,
        JobType.SECURITY_SENSITIVE_CHANGE,
        JobType.PRIVACY_SENSITIVE_CHANGE,
        JobType.PUBLIC_APP_PACKAGING,
        JobType.ARCHITECTURE,
        JobType.DEEP_PLANNING,
        JobType.SECURITY_REVIEW,
        JobType.COMPLEX_CODE,
        JobType.REFACTORING,
        JobType.MIGRATION,
        JobType.BUG_ANALYSIS,
        JobType.SIMPLE_CODE_CHANGE,
        JobType.SMALL_BUGFIX,
        JobType.BUG_FIX,
    }
    return job_type in allowed
