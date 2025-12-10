# FILE: app/llm/job_classifier.py
"""
Job classification for LLM routing.

Version: 0.15.0 - Critical Pipeline Spec Integration

CHANGES FROM v0.14.2:
- Integrated file_classifier.py for MIXED_FILE detection
- Stable [FILE_X] naming via build_file_map()
- Enhanced modality flags with HAS_MIXED
- PDF routing now uses file_classifier's MIXED_FILE detection
- Backward compatible: compute_modality_flags() returns same structure + new fields

8-ROUTE CLASSIFICATION:
1. CHAT_LIGHT → OpenAI gpt-4.1-mini (casual chat)
2. TEXT_HEAVY → OpenAI gpt-4.1 (heavy text work, text-only PDFs)
3. CODE_MEDIUM → Anthropic Sonnet (scoped code, 1-3 files)
4. ORCHESTRATOR → Anthropic Opus (architecture, multi-file)
5. IMAGE_SIMPLE → Gemini Flash (LEGACY ONLY - never auto-selected)
6. IMAGE_COMPLEX → Gemini 2.5 Pro (ALL images, MIXED_FILE with images)
7. VIDEO_HEAVY → Gemini 3.0 Pro (ALL videos)
8. OPUS_CRITIC → Gemini 3.0 Pro (explicit Opus review only)
9. VIDEO_CODE_DEBUG → 2-step pipeline: Gemini3 transcribe → Sonnet code

MIXED_FILE ROUTING (v0.15.0):
- PDFs with embedded images → MIXED_FILE → IMAGE_COMPLEX
- DOCX with embedded images → MIXED_FILE → IMAGE_COMPLEX
- PPTX (always has slides) → MIXED_FILE → IMAGE_COMPLEX
- Text-only PDFs → TEXT_FILE → TEXT_HEAVY

HARD RULES:
- Images/video NEVER go to Claude (enforced here)
- PDFs NEVER go to Claude (enforced here)
- MIXED_FILE (docs with images) → Gemini (vision required)
- opus.critic is EXPLICIT ONLY (no fuzzy matching)
- Gemini Flash is NEVER auto-selected
"""

import os
import logging
from typing import Optional, List, Dict, Any, Tuple, Union

from .schemas import (
    JobType, Provider, RoutingDecision, RoutingConfig, AttachmentInfo
)

# v0.15.0: Import file_classifier for MIXED_FILE detection
try:
    from .file_classifier import (
        classify_attachments,
        classify_from_attachment_info,
        build_file_map,
        FileType,
        ClassificationResult,
        has_vision_content,
    )
    FILE_CLASSIFIER_AVAILABLE = True
except ImportError:
    FILE_CLASSIFIER_AVAILABLE = False
    # Stub for backward compatibility
    class FileType:
        TEXT_FILE = "TEXT_FILE"
        CODE_FILE = "CODE_FILE"
        IMAGE_FILE = "IMAGE_FILE"
        VIDEO_FILE = "VIDEO_FILE"
        MIXED_FILE = "MIXED_FILE"

logger = logging.getLogger(__name__)

# ============================================================================
# ROUTER DEBUG MODE
# ============================================================================
ROUTER_DEBUG = os.getenv("ORB_ROUTER_DEBUG", "0") == "1"

def _debug_log(msg: str):
    """Print debug message if ROUTER_DEBUG is enabled."""
    if ROUTER_DEBUG:
        print(f"[router-debug] {msg}")

# Size threshold for video escalation (10MB)
VIDEO_SIZE_THRESHOLD = 10 * 1024 * 1024

# Deep semantic video analysis keywords
VIDEO_DEEP_ANALYSIS_KEYWORDS: set = {
    "find best shots", "extract narrative", "segment scenes",
    "identify key scenes", "select highlight moments",
    "analyse storyline", "analyze storyline",
    "structure this video into chapters", "chapter this video",
    "find key moments", "extract highlights", "scene detection",
    "narrative structure", "story arc", "identify chapters",
    "semantic analysis", "deep video analysis", "detailed video analysis",
}


# =============================================================================
# MODALITY FLAG HELPER (v0.15.0 - Enhanced with file_classifier)
# =============================================================================

def compute_modality_flags(
    attachments: List[AttachmentInfo],
    base_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute modality flags from attachments.
    
    v0.15.0: Now uses file_classifier for MIXED_FILE detection.
    Backward compatible - returns same structure plus new fields.
    
    Args:
        attachments: List of AttachmentInfo objects
        base_path: Optional base path for file access (for MIXED_FILE detection)
    
    Returns:
        Dict with:
            # Original fields (backward compat)
            has_video: bool
            has_image: bool
            has_code: bool
            has_text: bool
            has_pdf: bool
            video_count: int
            image_count: int
            code_count: int
            text_count: int
            video_attachments: List[AttachmentInfo]
            image_attachments: List[AttachmentInfo]
            code_attachments: List[AttachmentInfo]
            
            # New fields (v0.15.0)
            has_mixed: bool  # Documents with embedded images
            mixed_count: int
            classification_result: Optional[ClassificationResult]
            file_map: Optional[str]  # Stable [FILE_X] naming
    """
    # Basic categorization (backward compat)
    video_attachments = [a for a in attachments if a.is_video]
    image_attachments = [a for a in attachments if a.is_image]
    code_attachments = [a for a in attachments if a.is_code]
    text_attachments = [a for a in attachments if a.is_document and not a.is_pdf]
    pdf_attachments = [a for a in attachments if a.is_pdf]
    
    # v0.15.0: Use file_classifier for enhanced detection
    classification_result = None
    file_map = None
    has_mixed = False
    mixed_count = 0
    
    if FILE_CLASSIFIER_AVAILABLE and attachments:
        try:
            classification_result = classify_from_attachment_info(attachments)
            file_map = build_file_map(classification_result)
            
            # Extract MIXED_FILE info
            has_mixed = classification_result.HAS_MIXED
            mixed_count = len(classification_result.mixed_files)
            
            if ROUTER_DEBUG:
                _debug_log(f"file_classifier results:")
                _debug_log(f"  HAS_TEXT: {classification_result.HAS_TEXT}")
                _debug_log(f"  HAS_CODE: {classification_result.HAS_CODE}")
                _debug_log(f"  HAS_IMAGE: {classification_result.HAS_IMAGE}")
                _debug_log(f"  HAS_VIDEO: {classification_result.HAS_VIDEO}")
                _debug_log(f"  HAS_MIXED: {classification_result.HAS_MIXED}")
                _debug_log(f"  mixed_files: {[f.file_id for f in classification_result.mixed_files]}")
                
        except Exception as e:
            logger.warning(f"file_classifier failed, using fallback: {e}")
            if ROUTER_DEBUG:
                _debug_log(f"file_classifier error: {e}")
    
    # Fallback MIXED_FILE detection for PDFs (if file_classifier unavailable)
    if not FILE_CLASSIFIER_AVAILABLE:
        for pdf in pdf_attachments:
            if pdf.pdf_image_count and pdf.pdf_image_count > 0:
                has_mixed = True
                mixed_count += 1
    
    return {
        # Original fields (backward compat)
        "has_video": len(video_attachments) > 0,
        "has_image": len(image_attachments) > 0,
        "has_code": len(code_attachments) > 0,
        "has_text": len(text_attachments) > 0 or len(pdf_attachments) > 0,
        "has_pdf": len(pdf_attachments) > 0,
        "video_count": len(video_attachments),
        "image_count": len(image_attachments),
        "code_count": len(code_attachments),
        "text_count": len(text_attachments) + len(pdf_attachments),
        "video_attachments": video_attachments,
        "image_attachments": image_attachments,
        "code_attachments": code_attachments,
        
        # New fields (v0.15.0)
        "has_mixed": has_mixed,
        "mixed_count": mixed_count,
        "classification_result": classification_result,
        "file_map": file_map,
    }


def classify_job(
    message: str,
    attachments: Optional[List[AttachmentInfo]] = None,
    requested_job_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> RoutingDecision:
    """
    Classify a job and return routing decision.
    
    v0.15.0: Enhanced with MIXED_FILE detection via file_classifier.
    
    Priority order:
    1. Explicit opus.critic (job_type or metadata)
    2. Explicit document.pdf_text / document.pdf_vision override
    3. User override detection ("force Opus", "use GPT")
    4. Video+Code → VIDEO_CODE_DEBUG pipeline
    5. Video attachments → VIDEO_HEAVY
    6. Code attachments → CODE_MEDIUM/ORCHESTRATOR (code wins over images)
    7. Image attachments → IMAGE_COMPLEX
    8. MIXED_FILE (docs with images) → IMAGE_COMPLEX (v0.15.0)
    9. PDF attachments → TEXT_HEAVY or IMAGE_COMPLEX
    10. Text complexity → TEXT_HEAVY or CHAT_LIGHT
    11. Default → CHAT_LIGHT
    
    Args:
        message: User message text
        attachments: List of AttachmentInfo objects
        requested_job_type: Explicitly requested job type string
        metadata: Optional metadata dict
    
    Returns:
        RoutingDecision with job_type, provider, model, reason
    """
    attachments = attachments or []
    metadata = metadata or {}
    message_lower = message.lower()
    
    # =========================================================================
    # DEBUG LOGGING
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
                _debug_log(f"    pdf_image_count: {getattr(att, 'pdf_image_count', 'N/A')}")
    
    # =========================================================================
    # 1. EXPLICIT OPUS.CRITIC CHECK
    # =========================================================================
    
    if requested_job_type == "opus.critic":
        _debug_log("Section 1: OPUS.CRITIC - Explicit request")
        return _make_decision(
            JobType.OPUS_CRITIC,
            "Explicit opus.critic job type requested"
        )
    
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
    # 3.5 COMPUTE MODALITY FLAGS (v0.15.0: with file_classifier)
    # =========================================================================
    
    modality_flags = compute_modality_flags(attachments)
    
    if ROUTER_DEBUG:
        _debug_log(f"Section 3.5: MODALITY FLAGS")
        _debug_log(f"  has_video: {modality_flags['has_video']} ({modality_flags['video_count']})")
        _debug_log(f"  has_image: {modality_flags['has_image']} ({modality_flags['image_count']})")
        _debug_log(f"  has_code: {modality_flags['has_code']} ({modality_flags['code_count']})")
        _debug_log(f"  has_text: {modality_flags['has_text']}")
        _debug_log(f"  has_mixed: {modality_flags['has_mixed']} ({modality_flags['mixed_count']})")
        if modality_flags.get('file_map'):
            _debug_log(f"  file_map generated: {len(modality_flags['file_map'])} chars")
    
    # =========================================================================
    # 4. VIDEO+CODE DEBUG PIPELINE (v0.14.1)
    # =========================================================================
    
    if modality_flags["has_video"] and modality_flags["has_code"]:
        if ROUTER_DEBUG:
            _debug_log(f"  → VIDEO+CODE detected: {modality_flags['video_count']} video(s) + {modality_flags['code_count']} code file(s)")
            _debug_log(f"  → Returning VIDEO_CODE_DEBUG (Gemini3 → Sonnet pipeline)")
        
        return _make_decision(
            JobType.VIDEO_CODE_DEBUG,
            f"Video+Code debug pipeline: {modality_flags['video_count']} video(s) + {modality_flags['code_count']} code file(s) → Gemini3 transcription → Sonnet coding",
            file_map=modality_flags.get('file_map')
        )
    
    # =========================================================================
    # 5. VIDEO ATTACHMENTS → GEMINI 3 PRO
    # =========================================================================
    
    video_attachments = modality_flags["video_attachments"]
    image_attachments = modality_flags["image_attachments"]
    
    if video_attachments:
        video_count = len(video_attachments)
        total_video_size = sum(a.size_bytes for a in video_attachments)
        
        if ROUTER_DEBUG:
            _debug_log(f"Section 5: VIDEO DETECTION - {video_count} video(s)")
            _debug_log(f"  Total video size: {total_video_size / 1024 / 1024:.1f}MB")
            _debug_log(f"  → ALL VIDEO → VIDEO_HEAVY (Gemini 3.0 Pro)")
        
        return _make_decision(
            JobType.VIDEO_HEAVY,
            f"Video analysis: {video_count} video(s), {total_video_size / 1024 / 1024:.1f}MB → Gemini 3.0 Pro",
            file_map=modality_flags.get('file_map')
        )
    
    # =========================================================================
    # 6. CODE ATTACHMENTS → ANTHROPIC (code wins over images)
    # =========================================================================
    
    code_attachments = modality_flags["code_attachments"]
    
    if code_attachments or _has_code_keywords(message_lower):
        if ROUTER_DEBUG:
            _debug_log(f"Section 6: CODE DETECTION - Found {len(code_attachments)} code file(s)")
            if modality_flags["has_image"]:
                _debug_log(f"  Note: {modality_flags['image_count']} image(s) also present - code takes priority")
            if modality_flags["has_mixed"]:
                _debug_log(f"  Note: {modality_flags['mixed_count']} mixed file(s) also present - code takes priority")
        
        bugfix_keywords = [
            "bug", "bugfix", "fix", "error", "issue", "traceback",
            "stack trace", "identify bug", "find the issue", "debug",
            "inspect", "find any bug", "code bugfix", "please inspect"
        ]
        
        has_bugfix_request = any(kw in message_lower for kw in bugfix_keywords)
        
        if ROUTER_DEBUG:
            matched_bugfix = [kw for kw in bugfix_keywords if kw in message_lower]
            _debug_log(f"  Bugfix check: {has_bugfix_request} (matched: {matched_bugfix})")
        
        if len(code_attachments) == 1 and has_bugfix_request:
            _debug_log(f"  → Returning CODE_MEDIUM (single file + bugfix)")
            return _make_decision(
                JobType.CODE_MEDIUM,
                "Single code file with bugfix request → Sonnet",
                file_map=modality_flags.get('file_map')
            )
        
        if len(code_attachments) > 3 or _has_architecture_keywords(message_lower):
            _debug_log(f"  → Returning ORCHESTRATOR (multi-file or architecture)")
            return _make_decision(
                JobType.ORCHESTRATOR,
                "Multi-file or architecture-level code task → Opus",
                file_map=modality_flags.get('file_map')
            )
        
        elif _has_scoped_code_keywords(message_lower):
            _debug_log(f"  → Returning CODE_MEDIUM (scoped code task)")
            return _make_decision(
                JobType.CODE_MEDIUM,
                "Scoped code task (1-3 files) → Sonnet",
                file_map=modality_flags.get('file_map')
            )
        
        else:
            _debug_log(f"  → Returning CODE_MEDIUM (default for code)")
            return _make_decision(
                JobType.CODE_MEDIUM,
                "Code task (defaulting to Sonnet)",
                file_map=modality_flags.get('file_map')
            )
    
    # =========================================================================
    # 7. IMAGE ATTACHMENTS → GEMINI 2.5 PRO
    # =========================================================================
    
    if image_attachments:
        image_count = len(image_attachments)
        total_image_size = sum(a.size_bytes for a in image_attachments)
        
        if ROUTER_DEBUG:
            _debug_log(f"Section 7: IMAGE DETECTION - {image_count} image(s)")
            _debug_log(f"  Total image size: {total_image_size / 1024 / 1024:.1f}MB")
            _debug_log(f"  → ALL IMAGES → IMAGE_COMPLEX (Gemini 2.5 Pro)")
        
        return _make_decision(
            JobType.IMAGE_COMPLEX,
            f"Image analysis: {image_count} image(s) → Gemini 2.5 Pro",
            file_map=modality_flags.get('file_map')
        )
    
    # =========================================================================
    # 7.5. MIXED_FILE (Docs with images) → GEMINI 2.5 PRO (v0.15.0)
    # =========================================================================
    
    if modality_flags["has_mixed"]:
        mixed_count = modality_flags["mixed_count"]
        
        if ROUTER_DEBUG:
            _debug_log(f"Section 7.5: MIXED_FILE DETECTION - {mixed_count} mixed file(s)")
            _debug_log(f"  → MIXED_FILE → IMAGE_COMPLEX (Gemini 2.5 Pro for vision)")
        
        return _make_decision(
            JobType.IMAGE_COMPLEX,
            f"Documents with embedded images: {mixed_count} mixed file(s) → Gemini 2.5 Pro (vision required)",
            file_map=modality_flags.get('file_map')
        )
    
    # =========================================================================
    # 8. DOCUMENT FILES (.md, .txt, .csv) → Smart routing
    # =========================================================================
    
    doc_attachments = [a for a in attachments 
                       if a.filename.lower().endswith(('.md', '.txt', '.csv', '.json', '.yaml', '.yml'))]
    
    if ROUTER_DEBUG:
        _debug_log(f"Section 8: DOCUMENT FILES - Found {len(doc_attachments)} document(s)")
    
    if doc_attachments:
        architecture_keywords = [
            "architecture design", "high-level architecture", "system architecture",
            "design a revised architecture", "architect", "architectural",
            "design a system", "design a new system", "refactor the entire",
            "comprehensive architectural analysis", "deep system dive",
            "structural improvements", "system design", "component design"
        ]
        
        simple_doc_keywords = [
            "summari", "explain", "describe", "list", "outline",
            "convert", "format", "show me", "what is", "what does",
            "tell me about", "read this", "preview", "quick look"
        ]
        
        has_architecture_request = any(kw in message_lower for kw in architecture_keywords)
        
        if has_architecture_request:
            _debug_log(f"  → Returning ORCHESTRATOR (architecture design)")
            return _make_decision(
                JobType.ORCHESTRATOR,
                f"Document with architecture design request: {len(doc_attachments)} file(s)",
                file_map=modality_flags.get('file_map')
            )
        
        has_simple_request = any(kw in message_lower for kw in simple_doc_keywords)
        
        if has_simple_request:
            _debug_log(f"  → Returning CHAT_LIGHT (simple doc processing)")
            return _make_decision(
                JobType.CHAT_LIGHT,
                f"Document task: {len(doc_attachments)} file(s), simple processing",
                file_map=modality_flags.get('file_map')
            )
    
    # =========================================================================
    # 9. PDF ATTACHMENTS → DETERMINISTIC ROUTING
    # =========================================================================
    
    pdf_attachments = [a for a in attachments if a.is_pdf]
    if pdf_attachments:
        return _classify_pdf(pdf_attachments, message_lower, modality_flags.get('file_map'))
    
    # =========================================================================
    # 10. TEXT COMPLEXITY DETECTION
    # =========================================================================
    
    if _has_heavy_text_keywords(message_lower):
        return _make_decision(
            JobType.TEXT_HEAVY,
            "Heavy text work detected"
        )
    
    # =========================================================================
    # 11. LEGACY JOB TYPE NORMALIZATION
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
    # 12. DEFAULT → CHAT_LIGHT
    # =========================================================================
    
    return _make_decision(
        JobType.CHAT_LIGHT,
        "Default classification (casual chat)"
    )


def _classify_pdf(
    pdf_attachments: List[AttachmentInfo],
    message_lower: str,
    file_map: Optional[str] = None,
) -> RoutingDecision:
    """
    Deterministic PDF routing based on image count.
    
    v0.15.0: Uses pdf_image_count for MIXED_FILE detection.
    
    Rules:
    - image_count == 0 → TEXT_HEAVY (GPT)
    - image_count > 0 → IMAGE_COMPLEX (Gemini) [MIXED_FILE]
    - NEVER route to Claude
    """
    total_image_count = 0
    total_text_chars = 0
    
    for pdf in pdf_attachments:
        if pdf.pdf_image_count is not None:
            total_image_count += pdf.pdf_image_count
        if pdf.pdf_text_chars is not None:
            total_text_chars += pdf.pdf_text_chars
    
    if ROUTER_DEBUG:
        _debug_log(f"Section 9: PDF ROUTING")
        _debug_log(f"  Total PDFs: {len(pdf_attachments)}")
        _debug_log(f"  Total image count: {total_image_count}")
        _debug_log(f"  Total text chars: {total_text_chars}")
    
    # If no image count data, check text as signal
    if all(pdf.pdf_image_count is None for pdf in pdf_attachments):
        if total_text_chars > 0:
            _debug_log(f"  → TEXT_HEAVY (text-only, no image analysis)")
            return _make_decision(
                JobType.TEXT_HEAVY,
                f"Text-based PDF ({total_text_chars} chars, image count not analyzed) → GPT",
                pdf_image_count=None,
                pdf_text_chars=total_text_chars,
                file_map=file_map
            )
        else:
            _debug_log(f"  → IMAGE_COMPLEX (safe default, no text)")
            return _make_decision(
                JobType.IMAGE_COMPLEX,
                "PDF not analyzed (no text, no image count) - using vision route (safe default)",
                pdf_image_count=None,
                pdf_text_chars=total_text_chars,
                file_map=file_map
            )
    
    # Deterministic routing based on image count
    if total_image_count == 0:
        _debug_log(f"  → TEXT_HEAVY (0 images)")
        return _make_decision(
            JobType.TEXT_HEAVY,
            f"Text-only PDF ({total_text_chars} chars, 0 images) → GPT",
            pdf_image_count=0,
            pdf_text_chars=total_text_chars,
            file_map=file_map
        )
    else:
        _debug_log(f"  → IMAGE_COMPLEX (MIXED_FILE: {total_image_count} images)")
        return _make_decision(
            JobType.IMAGE_COMPLEX,
            f"PDF with {total_image_count} image(s) [MIXED_FILE] → Gemini vision",
            pdf_image_count=total_image_count,
            pdf_text_chars=total_text_chars,
            file_map=file_map
        )


def _make_decision(
    job_type: JobType,
    reason: str,
    user_override: bool = False,
    pdf_image_count: Optional[int] = None,
    pdf_text_chars: Optional[int] = None,
    file_map: Optional[str] = None,
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
        if file_map:
            _debug_log(f"  File Map: {len(file_map)} chars")
        _debug_log("=" * 70)
    
    decision = RoutingDecision(
        job_type=job_type,
        provider=provider,
        model=model,
        reason=reason,
        user_override=user_override,
        pdf_image_count=pdf_image_count,
        pdf_text_chars=pdf_text_chars,
    )
    
    # v0.15.0: Attach file_map to decision metadata (for prompt injection)
    if file_map:
        if not hasattr(decision, 'metadata'):
            # Store in reason as workaround if RoutingDecision doesn't have metadata field
            decision.reason = f"{reason}\n\n{file_map}"
    
    return decision


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
            JobType.IMAGE_COMPLEX,
            "User requested Gemini explicitly → Gemini 2.5 Pro",
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
    """Check for deep semantic video analysis keywords."""
    return any(kw in message_lower for kw in VIDEO_DEEP_ANALYSIS_KEYWORDS)


# =============================================================================
# ATTACHMENT PREPARATION
# =============================================================================

def prepare_attachments(
    raw_attachments: Optional[Union[List[Dict[str, Any]], List[AttachmentInfo]]]
) -> List[AttachmentInfo]:
    """Convert raw attachment dicts to AttachmentInfo objects."""
    if not raw_attachments:
        return []
    
    result = []
    for att in raw_attachments:
        if isinstance(att, AttachmentInfo):
            result.append(att)
        else:
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
    attachments: Optional[Union[List[Dict[str, Any]], List[AttachmentInfo]]] = None,
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
    """Get routing for a job type string."""
    try:
        jt = JobType(job_type_str)
    except ValueError:
        jt = JobType.CHAT_LIGHT
    return _make_decision(jt, f"Direct lookup: {job_type_str}")


def get_model_config() -> Dict[str, str]:
    """Get current model configuration."""
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


# =============================================================================
# v0.15.0: FILE MAP HELPERS
# =============================================================================

def get_file_map_for_attachments(attachments: List[AttachmentInfo]) -> Optional[str]:
    """
    Generate stable file map for attachments.
    
    Returns:
        File map string or None if file_classifier unavailable
    """
    if not FILE_CLASSIFIER_AVAILABLE or not attachments:
        return None
    
    try:
        result = classify_from_attachment_info(attachments)
        return build_file_map(result)
    except Exception as e:
        logger.warning(f"Failed to generate file map: {e}")
        return None