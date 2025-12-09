# FILE: app/llm/job_classifier.py
"""
Job classification for LLM routing.

Version: 0.12.17 - Multi-File & Mixed-Media Video/Image Routing

8-ROUTE CLASSIFICATION:
1. CHAT_LIGHT → OpenAI gpt-4.1-mini (casual chat)
2. TEXT_HEAVY → OpenAI gpt-4.1 (heavy text work, text-only PDFs)
3. CODE_MEDIUM → Anthropic Sonnet (scoped code, 1-3 files)
4. ORCHESTRATOR → Anthropic Opus (architecture, multi-file)
5. IMAGE_SIMPLE → Gemini Flash (single screenshot, simple request)
6. IMAGE_COMPLEX → Gemini 2.5 Pro (multi-image 2+, complex vision)
7. VIDEO_HEAVY → Gemini 3.0 Pro (video >10MB, multi-video 2+, mixed media, deep analysis)
8. OPUS_CRITIC → Gemini 3.0 Pro (explicit Opus review only)

VIDEO/IMAGE ROUTING RULES (v0.12.17):
- Single small video + simple prompt → IMAGE_SIMPLE (Flash)
- Single large video (>10MB) → VIDEO_HEAVY (3.0 Pro)
- Multiple videos (2+) → VIDEO_HEAVY (3.0 Pro) [NEW]
- Video + image(s) mixed → VIDEO_HEAVY (3.0 Pro) [NEW]
- Small video + deep analysis keywords → VIDEO_HEAVY (3.0 Pro)
- Single image + simple prompt → IMAGE_SIMPLE (Flash)
- Multiple images (2+) → IMAGE_COMPLEX (2.5 Pro) [ADJUSTED THRESHOLD]
- Single image + complex keywords → IMAGE_COMPLEX (2.5 Pro)

HARD RULES:
- Images/video NEVER go to Claude (enforced here)
- PDFs NEVER go to Claude (enforced here)
- PDFs with image_count == 0 → TEXT_HEAVY (GPT)
- PDFs with image_count > 0 → IMAGE_COMPLEX (Gemini)
- opus.critic is EXPLICIT ONLY (no fuzzy matching)

v0.12.17:
- Added multi-video (2+) → VIDEO_HEAVY routing
- Added mixed media (video + image) → VIDEO_HEAVY routing
- Adjusted image threshold: 2+ images → IMAGE_COMPLEX (was >2)
- Enhanced debug logging for video/image counts and decisions

v0.13.1:
- Added deep semantic video analysis override for small videos
- Keywords: "find best shots", "extract narrative", "segment scenes", etc.
- Added is_claude_allowed() for attachment safety rule
"""

import os
import logging
from typing import Optional, List, Dict, Any, Tuple

from .schemas import (
    JobType, Provider, RoutingDecision, RoutingConfig, AttachmentInfo
)

logger = logging.getLogger(__name__)

# ============================================================================
# ROUTER DEBUG MODE
# ============================================================================
# Set ORB_ROUTER_DEBUG=1 in .env to enable detailed routing diagnostics
ROUTER_DEBUG = os.getenv("ORB_ROUTER_DEBUG", "0") == "1"

def _debug_log(msg: str):
    """Print debug message if ROUTER_DEBUG is enabled."""
    if ROUTER_DEBUG:
        print(f"[router-debug] {msg}")

# Size threshold for video escalation (10MB)
VIDEO_SIZE_THRESHOLD = 10 * 1024 * 1024

# Deep semantic video analysis keywords (triggers VIDEO_HEAVY even for small videos)
VIDEO_DEEP_ANALYSIS_KEYWORDS: set = {
    "find best shots",
    "extract narrative",
    "segment scenes",
    "identify key scenes",
    "select highlight moments",
    "analyse storyline",
    "analyze storyline",
    "structure this video into chapters",
    "chapter this video",
    "find key moments",
    "extract highlights",
    "scene detection",
    "narrative structure",
    "story arc",
    "identify chapters",
    "semantic analysis",
    "deep video analysis",
    "detailed video analysis",
}


def classify_job(
    message: str,
    attachments: Optional[List[AttachmentInfo]] = None,
    requested_job_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> RoutingDecision:
    """
    Classify a job and return routing decision.
    
    Priority order:
    1. Explicit opus.critic (job_type or metadata)
    2. Explicit document.pdf_text / document.pdf_vision override
    3. User override detection ("force Opus", "use GPT")
    4. Video attachments → VIDEO_HEAVY or IMAGE_SIMPLE
    5. Image attachments → IMAGE_SIMPLE or IMAGE_COMPLEX
    6. PDF attachments → TEXT_HEAVY or IMAGE_COMPLEX (based on image_count)
    7. Code detection → ORCHESTRATOR or CODE_MEDIUM
    8. Text complexity → TEXT_HEAVY or CHAT_LIGHT
    9. Default → CHAT_LIGHT
    
    Args:
        message: User message text
        attachments: List of AttachmentInfo objects
        requested_job_type: Explicitly requested job type string
        metadata: Optional metadata dict (for opus.critic trigger)
    
    Returns:
        RoutingDecision with job_type, provider, model, reason
    """
    attachments = attachments or []
    metadata = metadata or {}
    message_lower = message.lower()
    
    # =========================================================================
    # DEBUG LOGGING - Full visibility into classification inputs
    # =========================================================================
    if ROUTER_DEBUG:
        _debug_log("=" * 70)
        _debug_log("CLASSIFICATION START")
        _debug_log("=" * 70)
        _debug_log(f"Message (first 200 chars): {repr(message[:200])}")
        _debug_log(f"Message length: {len(message)} chars")
        _debug_log(f"Requested job_type: {requested_job_type}")
        _debug_log(f"Metadata: {metadata}")
        _debug_log(f"Total attachments: {len(attachments)}")
        
        if attachments:
            for i, att in enumerate(attachments):
                _debug_log(f"  Attachment {i+1}:")
                _debug_log(f"    filename: {att.filename}")
                _debug_log(f"    mime_type: {getattr(att, 'mime_type', 'N/A')}")
                _debug_log(f"    size_bytes: {getattr(att, 'size_bytes', 'N/A')}")
                _debug_log(f"    is_code: {att.is_code}")
                _debug_log(f"    is_image: {att.is_image}")
                _debug_log(f"    is_video: {att.is_video}")
                _debug_log(f"    is_pdf: {att.is_pdf}")
    
    # =========================================================================
    # 1. EXPLICIT OPUS.CRITIC CHECK (opt-in only, no fuzzy matching)
    # =========================================================================
    
    if requested_job_type == "opus.critic":
        _debug_log("Section 1: OPUS.CRITIC - Explicit request")
        return _make_decision(
            JobType.OPUS_CRITIC,
            "Explicit opus.critic job type requested"
        )
    
    # Metadata trigger: source_model == claude-opus + intent == critic
    if (metadata.get("source_model", "").startswith("claude-opus") and 
        metadata.get("intent") == "critic"):
        return _make_decision(
            JobType.OPUS_CRITIC,
            "Metadata indicates Opus output critique request"
        )
    
    # =========================================================================
    # 2. EXPLICIT PDF OVERRIDES
    # =========================================================================
    
    if requested_job_type == "document.pdf_text":
        return _make_decision(
            JobType.DOCUMENT_PDF_TEXT,
            "Explicit document.pdf_text override",
            user_override=True
        )
    
    if requested_job_type == "document.pdf_vision":
        return _make_decision(
            JobType.DOCUMENT_PDF_VISION,
            "Explicit document.pdf_vision override",
            user_override=True
        )
    
    # =========================================================================
    # 3. USER OVERRIDE DETECTION
    # =========================================================================
    
    override_result = _detect_user_override(message_lower)
    if override_result:
        return override_result
    
    # =========================================================================
    # 4. VIDEO ATTACHMENTS → GEMINI ONLY
    # =========================================================================
    
    video_attachments = [a for a in attachments if a.is_video]
    image_attachments = [a for a in attachments if a.is_image]
    
    if video_attachments:
        video_count = len(video_attachments)
        image_count = len(image_attachments)
        total_video_size = sum(a.size_bytes for a in video_attachments)
        
        if ROUTER_DEBUG:
            _debug_log(f"Section 4: VIDEO DETECTION - {video_count} video(s), {image_count} image(s)")
            _debug_log(f"  Total video size: {total_video_size / 1024 / 1024:.1f}MB")
        
        # Check for deep semantic analysis keywords (overrides size-based logic)
        needs_deep_analysis = _has_video_deep_analysis_keywords(message_lower)
        
        # RULE: Mixed media (video + image) → Always VIDEO_HEAVY (Gemini 3.0 Pro)
        if video_count > 0 and image_count > 0:
            if ROUTER_DEBUG:
                _debug_log(f"  → MIXED MEDIA: {video_count} video(s) + {image_count} image(s) → VIDEO_HEAVY")
            return _make_decision(
                JobType.VIDEO_HEAVY,
                f"Mixed media: {video_count} video(s) + {image_count} image(s) → Gemini 3.0 Pro"
            )
        
        # RULE: Multiple videos (2+) → Always VIDEO_HEAVY (Gemini 3.0 Pro)
        if video_count >= 2:
            if ROUTER_DEBUG:
                _debug_log(f"  → MULTIPLE VIDEOS: {video_count} videos → VIDEO_HEAVY")
            return _make_decision(
                JobType.VIDEO_HEAVY,
                f"Multiple videos ({video_count}) → Gemini 3.0 Pro"
            )
        
        # RULE: Large video (>10MB) → VIDEO_HEAVY (Gemini 3.0 Pro)
        if total_video_size > VIDEO_SIZE_THRESHOLD:
            if ROUTER_DEBUG:
                _debug_log(f"  → LARGE VIDEO: {total_video_size / 1024 / 1024:.1f}MB → VIDEO_HEAVY")
            return _make_decision(
                JobType.VIDEO_HEAVY,
                f"Large video ({total_video_size / 1024 / 1024:.1f}MB) → Gemini 3.0 Pro"
            )
        
        # RULE: Small video + deep analysis keywords → VIDEO_HEAVY (Gemini 3.0 Pro)
        elif needs_deep_analysis:
            if ROUTER_DEBUG:
                _debug_log(f"  → DEEP ANALYSIS: Small video with semantic analysis → VIDEO_HEAVY")
            return _make_decision(
                JobType.VIDEO_HEAVY,
                f"Small video ({total_video_size / 1024 / 1024:.1f}MB) with deep analysis → Gemini 3.0 Pro"
            )
        
        # RULE: Single small video, simple request → IMAGE_SIMPLE (Gemini Flash)
        else:
            if ROUTER_DEBUG:
                _debug_log(f"  → SIMPLE VIDEO: Single small video → IMAGE_SIMPLE (Flash)")
            return _make_decision(
                JobType.IMAGE_SIMPLE,
                f"Small video ({total_video_size / 1024 / 1024:.1f}MB), simple request → Gemini Flash"
            )
    
    # =========================================================================
    # 5. IMAGE ATTACHMENTS → GEMINI ONLY (NEVER CLAUDE)
    # =========================================================================
    
    # Note: image_attachments already computed above for mixed media check
    if image_attachments:
        image_count = len(image_attachments)
        total_image_size = sum(a.size_bytes for a in image_attachments)
        
        if ROUTER_DEBUG:
            _debug_log(f"Section 5: IMAGE DETECTION - {image_count} image(s)")
            _debug_log(f"  Total image size: {total_image_size / 1024 / 1024:.1f}MB")
        
        # RULE: Multiple images (2+) OR complex vision keywords → IMAGE_COMPLEX (Gemini 2.5 Pro)
        if image_count >= 2 or _has_complex_vision_keywords(message_lower):
            if ROUTER_DEBUG:
                if image_count >= 2:
                    _debug_log(f"  → MULTIPLE IMAGES: {image_count} images → IMAGE_COMPLEX")
                else:
                    _debug_log(f"  → COMPLEX VISION: Complex keywords detected → IMAGE_COMPLEX")
            
            return _make_decision(
                JobType.IMAGE_COMPLEX,
                f"{image_count} image(s), complex analysis → Gemini 2.5 Pro"
            )
        
        # RULE: Single image, simple request → IMAGE_SIMPLE (Gemini Flash)
        else:
            if ROUTER_DEBUG:
                _debug_log(f"  → SIMPLE IMAGE: Single image, simple request → IMAGE_SIMPLE")
            return _make_decision(
                JobType.IMAGE_SIMPLE,
                f"Single image, simple analysis → Gemini Flash"
            )
    
    # =========================================================================
    # 6. DOCUMENT FILES (.md, .txt, .csv) → Smart routing based on intent
    # =========================================================================
    
    doc_attachments = [a for a in attachments 
                       if a.filename.lower().endswith(('.md', '.txt', '.csv', '.json', '.yaml', '.yml'))]
    
    if ROUTER_DEBUG:
        _debug_log(f"Section 6: DOCUMENT FILES - Found {len(doc_attachments)} document(s)")
    
    if doc_attachments:
        # Architecture/design keywords - these take PRIORITY
        architecture_keywords = [
            "architecture design", "high-level architecture", "system architecture",
            "design a revised architecture", "architect", "architectural",
            "design a system", "design a new system", "refactor the entire",
            "comprehensive architectural analysis", "deep system dive",
            "structural improvements", "system design", "component design"
        ]
        
        # Simple doc processing keywords
        simple_doc_keywords = [
            "summari", "explain", "describe", "list", "outline",
            "convert", "format", "show me", "what is", "what does",
            "tell me about", "read this", "preview", "quick look"
        ]
        
        # Check for architecture/design intent FIRST (higher priority)
        has_architecture_request = any(kw in message_lower for kw in architecture_keywords)
        if ROUTER_DEBUG:
            matched_arch = [kw for kw in architecture_keywords if kw in message_lower]
            _debug_log(f"  Architecture check: {has_architecture_request} (matched: {matched_arch})")
        
        if has_architecture_request:
            _debug_log(f"  → Returning ORCHESTRATOR (architecture design)")
            return _make_decision(
                JobType.ORCHESTRATOR,
                f"Document with architecture design request: {len(doc_attachments)} file(s)"
            )
        
        # Then check for simple processing
        has_simple_request = any(kw in message_lower for kw in simple_doc_keywords)
        if ROUTER_DEBUG:
            matched_simple = [kw for kw in simple_doc_keywords if kw in message_lower]
            _debug_log(f"  Simple doc check: {has_simple_request} (matched: {matched_simple})")
        
        if has_simple_request:
            _debug_log(f"  → Returning CHAT_LIGHT (simple doc processing)")
            return _make_decision(
                JobType.CHAT_LIGHT,
                f"Document task: {len(doc_attachments)} file(s), simple processing"
            )
        
        if ROUTER_DEBUG:
            _debug_log(f"  No keywords matched, falling through to next section")
        # Default for documents: let it fall through to other checks
        # (will likely end up as CHAT_LIGHT in the default section)
    
    # =========================================================================
    # 7. PDF ATTACHMENTS → DETERMINISTIC ROUTING (NEVER CLAUDE)
    # =========================================================================
    
    pdf_attachments = [a for a in attachments if a.is_pdf]
    if pdf_attachments:
        return _classify_pdf(pdf_attachments, message_lower)
    
    # =========================================================================
    # 8. CODE DETECTION → ANTHROPIC (Sonnet or Opus)
    # =========================================================================
    
    code_attachments = [a for a in attachments if a.is_code]
    if ROUTER_DEBUG:
        _debug_log(f"Section 8: CODE DETECTION - Found {len(code_attachments)} code file(s)")
    
    if code_attachments or _has_code_keywords(message_lower):
        # Bugfix keywords - prefer CODE_MEDIUM for these
        bugfix_keywords = [
            "bug", "bugfix", "fix", "error", "issue", "traceback",
            "stack trace", "identify bug", "find the issue", "debug",
            "inspect", "find any bug", "code bugfix", "please inspect"
        ]
        
        has_bugfix_request = any(kw in message_lower for kw in bugfix_keywords)
        
        if ROUTER_DEBUG:
            matched_bugfix = [kw for kw in bugfix_keywords if kw in message_lower]
            _debug_log(f"  Bugfix check: {has_bugfix_request} (matched: {matched_bugfix})")
            _debug_log(f"  Code files: {len(code_attachments)}")
        
        # Single code file + bugfix keywords → CODE_MEDIUM (Sonnet)
        if len(code_attachments) == 1 and has_bugfix_request:
            _debug_log(f"  → Returning CODE_MEDIUM (single file + bugfix)")
            return _make_decision(
                JobType.CODE_MEDIUM,
                "Single code file with bugfix request"
            )
        
        # Multi-file OR architecture keywords → ORCHESTRATOR (Opus)
        if len(code_attachments) > 3 or _has_architecture_keywords(message_lower):
            _debug_log(f"  → Returning ORCHESTRATOR (multi-file or architecture)")
            return _make_decision(
                JobType.ORCHESTRATOR,
                "Multi-file or architecture-level code task"
            )
        
        # Scoped code keywords → CODE_MEDIUM (Sonnet)
        elif _has_scoped_code_keywords(message_lower):
            _debug_log(f"  → Returning CODE_MEDIUM (scoped code task)")
            return _make_decision(
                JobType.CODE_MEDIUM,
                "Scoped code task (1-3 files)"
            )
        
        # Default for code: CODE_MEDIUM (Sonnet) instead of ORCHESTRATOR
        # Only escalate to Opus when truly needed (multi-file, architecture)
        else:
            _debug_log(f"  → Returning CODE_MEDIUM (default for code)")
            return _make_decision(
                JobType.CODE_MEDIUM,
                "Code task (defaulting to Sonnet for safety)"
            )
    
    # =========================================================================
    # 9. TEXT COMPLEXITY DETECTION
    # =========================================================================
    
    if _has_heavy_text_keywords(message_lower):
        return _make_decision(
            JobType.TEXT_HEAVY,
            "Heavy text work detected"
        )
    
    # =========================================================================
    # 10. LEGACY JOB TYPE NORMALIZATION
    # =========================================================================
    
    if requested_job_type:
        try:
            legacy_type = JobType(requested_job_type)
            primary_type = RoutingConfig.normalize_job_type(legacy_type)
            return _make_decision(
                primary_type,
                f"Normalized from legacy type: {requested_job_type}"
            )
        except ValueError:
            logger.warning(f"Unknown job type requested: {requested_job_type}")
    
    # =========================================================================
    # 11. DEFAULT → CHAT_LIGHT
    # =========================================================================
    
    return _make_decision(
        JobType.CHAT_LIGHT,
        "Default classification (casual chat)"
    )


def _classify_pdf(
    pdf_attachments: List[AttachmentInfo],
    message_lower: str
) -> RoutingDecision:
    """
    Deterministic PDF routing based on image count.
    
    Rules:
    - image_count == 0 → TEXT_HEAVY (GPT)
    - image_count > 0 → IMAGE_COMPLEX (Gemini)
    - NEVER split a PDF across models
    - NEVER route to Claude
    """
    total_image_count = 0
    total_text_chars = 0
    
    for pdf in pdf_attachments:
        if pdf.pdf_image_count is not None:
            total_image_count += pdf.pdf_image_count
        if pdf.pdf_text_chars is not None:
            total_text_chars += pdf.pdf_text_chars
    
    # If we don't have image count data, assume it needs vision (safer)
    if all(pdf.pdf_image_count is None for pdf in pdf_attachments):
        logger.warning("PDF image count not analyzed - defaulting to vision route")
        return _make_decision(
            JobType.IMAGE_COMPLEX,
            "PDF not analyzed for images - using vision route (safe default)",
            pdf_image_count=None,
            pdf_text_chars=total_text_chars
        )
    
    # Deterministic routing based on image count
    if total_image_count == 0:
        return _make_decision(
            JobType.TEXT_HEAVY,
            f"Text-only PDF ({total_text_chars} chars, 0 images) → GPT",
            pdf_image_count=0,
            pdf_text_chars=total_text_chars
        )
    else:
        return _make_decision(
            JobType.IMAGE_COMPLEX,
            f"PDF with {total_image_count} image(s) → Gemini vision",
            pdf_image_count=total_image_count,
            pdf_text_chars=total_text_chars
        )


def _make_decision(
    job_type: JobType,
    reason: str,
    user_override: bool = False,
    pdf_image_count: Optional[int] = None,
    pdf_text_chars: Optional[int] = None,
) -> RoutingDecision:
    """Create a RoutingDecision with provider/model lookup."""
    provider, model = RoutingConfig.get_routing(job_type)
    
    if ROUTER_DEBUG:
        _debug_log("=" * 70)
        _debug_log("CLASSIFICATION COMPLETE")
        _debug_log(f"  Job Type: {job_type.value}")
        _debug_log(f"  Provider: {provider.value}")
        _debug_log(f"  Model: {model}")
        _debug_log(f"  Reason: {reason}")
        if user_override:
            _debug_log(f"  User Override: YES")
        if pdf_image_count is not None:
            _debug_log(f"  PDF Image Count: {pdf_image_count}")
        if pdf_text_chars is not None:
            _debug_log(f"  PDF Text Chars: {pdf_text_chars}")
        _debug_log("=" * 70)
    
    return RoutingDecision(
        job_type=job_type,
        provider=provider,
        model=model,
        reason=reason,
        user_override=user_override,
        pdf_image_count=pdf_image_count,
        pdf_text_chars=pdf_text_chars,
    )


def _detect_user_override(message_lower: str) -> Optional[RoutingDecision]:
    """Detect explicit user overrides in message."""
    
    if any(p in message_lower for p in ["force opus", "use opus", "send to opus"]):
        return _make_decision(
            JobType.ORCHESTRATOR,
            "User requested Opus explicitly",
            user_override=True
        )
    
    if any(p in message_lower for p in ["force sonnet", "use sonnet", "send to sonnet"]):
        return _make_decision(
            JobType.CODE_MEDIUM,
            "User requested Sonnet explicitly",
            user_override=True
        )
    
    if any(p in message_lower for p in ["force gpt", "use gpt", "send to gpt", "use openai"]):
        return _make_decision(
            JobType.TEXT_HEAVY,
            "User requested GPT explicitly",
            user_override=True
        )
    
    if any(p in message_lower for p in ["force gemini", "use gemini", "send to gemini"]):
        return _make_decision(
            JobType.IMAGE_SIMPLE,
            "User requested Gemini explicitly",
            user_override=True
        )
    
    return None


def _has_code_keywords(message_lower: str) -> bool:
    """Check if message contains code-related keywords."""
    code_indicators = [
        "code", "function", "class", "method", "variable",
        "bug", "error", "exception", "debug", "fix",
        "implement", "refactor", "test", "unit test",
        "api", "endpoint", "route", "handler",
        "import", "module", "package", "library",
        "def ", "async ", "await ", "return ",
        "```", "python", "javascript", "typescript",
    ]
    return any(kw in message_lower for kw in code_indicators)


def _has_scoped_code_keywords(message_lower: str) -> bool:
    """Check for scoped/small code task keywords."""
    return any(kw in message_lower for kw in RoutingConfig.CODE_MEDIUM_KEYWORDS)


def _has_architecture_keywords(message_lower: str) -> bool:
    """Check for architecture/complex code keywords."""
    return any(kw in message_lower for kw in RoutingConfig.ORCHESTRATOR_KEYWORDS)


def _has_heavy_text_keywords(message_lower: str) -> bool:
    """Check for heavy text work keywords."""
    return any(kw in message_lower for kw in RoutingConfig.TEXT_HEAVY_KEYWORDS)


def _has_complex_vision_keywords(message_lower: str) -> bool:
    """Check for complex vision analysis keywords."""
    return any(kw in message_lower for kw in RoutingConfig.IMAGE_COMPLEX_KEYWORDS)


def _has_video_deep_analysis_keywords(message_lower: str) -> bool:
    """
    Check for deep semantic video analysis keywords.
    
    These keywords trigger VIDEO_HEAVY even for small videos (<10MB).
    """
    return any(kw in message_lower for kw in VIDEO_DEEP_ANALYSIS_KEYWORDS)


# =============================================================================
# ATTACHMENT PREPARATION
# =============================================================================

def prepare_attachments(
    raw_attachments: Optional[List[Dict[str, Any]]]
) -> List[AttachmentInfo]:
    """Convert raw attachment dicts to AttachmentInfo objects."""
    if not raw_attachments:
        return []
    
    result = []
    for att in raw_attachments:
        info = AttachmentInfo(
            filename=att.get("filename", att.get("original_name", "unknown")),
            mime_type=att.get("mime_type", att.get("content_type")),
            size_bytes=att.get("size_bytes", att.get("size", 0)),
            pdf_image_count=att.get("pdf_image_count"),
            pdf_text_chars=att.get("pdf_text_chars"),
            pdf_page_count=att.get("pdf_page_count"),
        )
        result.append(info)
    
    return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def classify_and_route(
    message: str,
    attachments: Optional[List[Dict[str, Any]]] = None,
    job_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> RoutingDecision:
    """Main entry point: prepare attachments and classify."""
    attachment_infos = prepare_attachments(attachments)
    return classify_job(message, attachment_infos, job_type, metadata)


def get_provider_for_job(job_type: JobType) -> Tuple[Provider, str]:
    """Get provider and model for a job type."""
    return RoutingConfig.get_routing(job_type)


def get_routing_for_job_type(job_type_str: str) -> RoutingDecision:
    """Get routing for a job type string (backward compat)."""
    try:
        jt = JobType(job_type_str)
    except ValueError:
        jt = JobType.CHAT_LIGHT
    return _make_decision(jt, f"Direct lookup: {job_type_str}")


def get_model_config() -> Dict[str, str]:
    """Get current model configuration (backward compat)."""
    return {
        "openai": os.getenv("OPENAI_MODEL_LIGHT_CHAT", "gpt-4.1-mini"),
        "openai_heavy": os.getenv("OPENAI_MODEL_HEAVY_TEXT", "gpt-4.1"),
        "anthropic_sonnet": os.getenv("ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-5-20250929"),
        "anthropic_opus": os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-5-20250514"),
        "gemini_fast": os.getenv("GEMINI_VISION_MODEL_FAST", "gemini-2.0-flash"),
        "gemini_complex": os.getenv("GEMINI_VISION_MODEL_COMPLEX", "gemini-2.5-pro"),
        "gemini_video": os.getenv("GEMINI_VIDEO_HEAVY_MODEL", "gemini-3.0-pro-preview"),
        "gemini_critic": os.getenv("GEMINI_OPUS_CRITIC_MODEL", "gemini-3.0-pro-preview"),
    }


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
    """
    Check if this job type is explicitly allowed to go to Claude.
    
    Only CODE_MEDIUM and ORCHESTRATOR (and their legacy aliases) are allowed.
    Used for attachment safety rule enforcement.
    """
    allowed = {
        JobType.CODE_MEDIUM,
        JobType.ORCHESTRATOR,
        # Legacy aliases that map to these
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