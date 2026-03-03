# FILE: app/llm/_job_classifier_classify.py
"""
Core classification logic for job routing.

Extracted from job_classifier.py — contains the main classify_job()
function and its direct helpers (_classify_pdf, _make_decision,
_detect_user_override, keyword checkers).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from .schemas import (
    JobType, Provider, RoutingDecision, RoutingConfig, AttachmentInfo
)

logger = logging.getLogger(__name__)

ROUTER_DEBUG = os.getenv("ORB_ROUTER_DEBUG", "0") == "1"


def _debug_log(msg: str):
    """Print debug message if ROUTER_DEBUG is enabled."""
    if ROUTER_DEBUG:
        # Replace unicode arrows that break cp1252 console encoding
        safe_msg = msg.replace("\u2192", "->").replace("\u2190", "<-")
        print(f"[router-debug] {safe_msg}")


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

    if file_map:
        if not hasattr(decision, 'metadata'):
            decision.reason = f"{reason}\n\n{file_map}"

    return decision


def _detect_user_override(message_lower: str) -> Optional[RoutingDecision]:
    """Detect explicit user overrides in message (legacy style)."""
    if any(p in message_lower for p in ["force opus", "use opus", "send to opus"]):
        return _make_decision(JobType.ORCHESTRATOR, "User requested Opus explicitly", user_override=True)
    if any(p in message_lower for p in ["force sonnet", "use sonnet", "send to sonnet"]):
        return _make_decision(JobType.CODE_MEDIUM, "User requested Sonnet explicitly", user_override=True)
    if any(p in message_lower for p in ["force gpt", "use gpt", "send to gpt", "use openai"]):
        return _make_decision(JobType.TEXT_HEAVY, "User requested GPT explicitly", user_override=True)
    if any(p in message_lower for p in ["force gemini", "use gemini", "send to gemini"]):
        return _make_decision(JobType.IMAGE_COMPLEX, "User requested Gemini explicitly → Gemini 2.5 Pro", user_override=True)
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


def _has_codebase_intent(message_lower: str) -> bool:
    """Check if message expresses intent to explore/analyse the codebase.

    v9.0: Used to detect compound intent (video + codebase) so the classifier
    routes to a model with both multimodal input AND tool access, rather than
    the vision-only pipeline.

    Matches phrases like:
    - "go through the code base"
    - "look at the code"
    - "check the implementation"
    - "debug this in the codebase"
    """
    # Phrase-level patterns (high confidence)
    codebase_phrases = [
        "code base", "codebase", "code-base",
        "go through the code", "look at the code", "check the code",
        "look through the code", "search the code", "find in the code",
        "look at the implementation", "check the implementation",
        "read the file", "read the source", "look at the file",
        "go into the code", "check the file", "search the file",
        "explore the code", "navigate the code",
        "ground this in", "ground it in",
        "what's in the code", "what is in the code",
        "find the relevant", "look at the relevant",
        "check the backend", "check the frontend",
        "look at the backend", "look at the frontend",
        "go through the project", "look at the project",
        "check the source", "examine the code",
        "check what we have", "what we already have",
        "what's already in place", "already in place",
        "come up with a plan",  # planning intent combined with video = code work
    ]
    # Two or more of these single-word signals in the same message
    code_signal_words = [
        "debug", "implement", "refactor", "fix", "pipeline",
        "component", "module", "router", "endpoint", "schema",
        "frontend", "backend", "api",
    ]
    if any(phrase in message_lower for phrase in codebase_phrases):
        return True
    # Require 2+ signal words to avoid false positives on casual mentions
    signal_count = sum(1 for w in code_signal_words if w in message_lower)
    return signal_count >= 2


def _classify_pdf(
    pdf_attachments: List[AttachmentInfo],
    message_lower: str,
    file_map: Optional[str] = None,
) -> RoutingDecision:
    """Deterministic PDF routing based on image count."""
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

    if all(pdf.pdf_image_count is None for pdf in pdf_attachments):
        if total_text_chars > 0:
            _debug_log(f"  → TEXT_HEAVY (text-only, no image analysis)")
            return _make_decision(
                JobType.TEXT_HEAVY,
                f"Text-based PDF ({total_text_chars} chars, image count not analyzed) → GPT",
                pdf_image_count=None, pdf_text_chars=total_text_chars, file_map=file_map,
            )
        else:
            _debug_log(f"  → IMAGE_COMPLEX (safe default, no text)")
            return _make_decision(
                JobType.IMAGE_COMPLEX,
                "PDF not analyzed (no text, no image count) - using vision route (safe default)",
                pdf_image_count=None, pdf_text_chars=total_text_chars, file_map=file_map,
            )

    if total_image_count == 0:
        _debug_log(f"  → TEXT_HEAVY (0 images)")
        return _make_decision(
            JobType.TEXT_HEAVY,
            f"Text-only PDF ({total_text_chars} chars, 0 images) → GPT",
            pdf_image_count=0, pdf_text_chars=total_text_chars, file_map=file_map,
        )
    else:
        _debug_log(f"  → IMAGE_COMPLEX (MIXED_FILE: {total_image_count} images)")
        return _make_decision(
            JobType.IMAGE_COMPLEX,
            f"PDF with {total_image_count} image(s) [MIXED_FILE] → Gemini vision",
            pdf_image_count=total_image_count, pdf_text_chars=total_text_chars, file_map=file_map,
        )


def classify_job(
    message: str,
    attachments: Optional[List[AttachmentInfo]] = None,
    requested_job_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> RoutingDecision:
    """Classify a job and return routing decision.

    Priority order:
    1. Explicit opus.critic
    2. Explicit document.pdf_text / document.pdf_vision override
    3. User override detection
    4. Video+Code → VIDEO_CODE_DEBUG pipeline
    5. Video attachments → VIDEO_HEAVY
    6. Code attachments → CODE_MEDIUM/ORCHESTRATOR
    7. Image attachments → IMAGE_COMPLEX
    8. MIXED_FILE → IMAGE_COMPLEX
    9. PDF attachments → TEXT_HEAVY or IMAGE_COMPLEX
    10. Text complexity → TEXT_HEAVY or CHAT_LIGHT
    11. Default → CHAT_LIGHT
    """
    # Lazy import to avoid circular dependency
    from .job_classifier import compute_modality_flags, FILE_CLASSIFIER_AVAILABLE

    attachments = attachments or []
    metadata = metadata or {}
    message_lower = message.lower()

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

    # 1. EXPLICIT OPUS.CRITIC CHECK
    if requested_job_type == "opus.critic":
        _debug_log("Section 1: OPUS.CRITIC - Explicit request")
        return _make_decision(JobType.OPUS_CRITIC, "Explicit opus.critic job type requested")

    if (metadata.get("source_model", "").startswith("claude-opus") and
        metadata.get("intent") == "critic"):
        return _make_decision(JobType.OPUS_CRITIC, "Metadata indicates Opus output critique request")

    # 2. EXPLICIT PDF OVERRIDES
    if requested_job_type == "document.pdf_text":
        return _make_decision(JobType.DOCUMENT_PDF_TEXT, "Explicit document.pdf_text override", user_override=True)
    if requested_job_type == "document.pdf_vision":
        return _make_decision(JobType.DOCUMENT_PDF_VISION, "Explicit document.pdf_vision override", user_override=True)

    # 3. USER OVERRIDE DETECTION (Legacy)
    override_result = _detect_user_override(message_lower)
    if override_result:
        return override_result

    # 3.5 COMPUTE MODALITY FLAGS
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

    # 4. VIDEO+CODE DEBUG PIPELINE (code file attachments)
    if modality_flags["has_video"] and modality_flags["has_code"]:
        if ROUTER_DEBUG:
            _debug_log(f"  → VIDEO+CODE detected (file attachments)")
            _debug_log(f"  → Returning VIDEO_CODE_DEBUG")
        return _make_decision(
            JobType.VIDEO_CODE_DEBUG,
            f"Video+Code debug pipeline: {modality_flags['video_count']} video(s) + {modality_flags['code_count']} code file(s)",
            file_map=modality_flags.get('file_map'),
        )

    # 4.5 VIDEO + CODEBASE INTENT (v9.0)
    # If the message mentions codebase/code exploration alongside a video,
    # route to Gemini 3.1 Pro customtools — one model, video in, tools out.
    # This must come BEFORE the video short-circuit in Section 5.
    if modality_flags["has_video"] and _has_codebase_intent(message_lower):
        video_count = modality_flags["video_count"]
        total_video_size = sum(a.size_bytes for a in modality_flags["video_attachments"])
        if ROUTER_DEBUG:
            _debug_log(f"Section 4.5: VIDEO + CODEBASE INTENT")
            _debug_log(f"  -> {video_count} video(s) + codebase intent in message")
            _debug_log(f"  -> Returning VIDEO_CODE_TOOLS (Gemini 3.1 Pro customtools)")
        return _make_decision(
            JobType.VIDEO_CODE_TOOLS,
            f"Video+Codebase intent: {video_count} video(s), {total_video_size / 1024 / 1024:.1f}MB + code intent -> Gemini 3.1 Pro customtools",
            file_map=modality_flags.get('file_map'),
        )

    # 5. VIDEO ATTACHMENTS → GEMINI 3 PRO (pure vision, no code intent)
    video_attachments = modality_flags["video_attachments"]
    image_attachments = modality_flags["image_attachments"]

    if video_attachments:
        video_count = len(video_attachments)
        total_video_size = sum(a.size_bytes for a in video_attachments)
        if ROUTER_DEBUG:
            _debug_log(f"Section 5: VIDEO DETECTION - {video_count} video(s)")
            _debug_log(f"  → ALL VIDEO → VIDEO_HEAVY (Gemini 3.0 Pro)")
        return _make_decision(
            JobType.VIDEO_HEAVY,
            f"Video analysis: {video_count} video(s), {total_video_size / 1024 / 1024:.1f}MB → Gemini 3.0 Pro",
            file_map=modality_flags.get('file_map'),
        )

    # 6. CODE ATTACHMENTS → ANTHROPIC
    code_attachments = modality_flags["code_attachments"]
    if code_attachments or _has_code_keywords(message_lower):
        if ROUTER_DEBUG:
            _debug_log(f"Section 6: CODE DETECTION - Found {len(code_attachments)} code file(s)")

        bugfix_keywords = [
            "bug", "bugfix", "fix", "error", "issue", "traceback",
            "stack trace", "identify bug", "find the issue", "debug",
            "inspect", "find any bug", "code bugfix", "please inspect"
        ]
        has_bugfix_request = any(kw in message_lower for kw in bugfix_keywords)

        if len(code_attachments) == 1 and has_bugfix_request:
            return _make_decision(JobType.CODE_MEDIUM, "Single code file with bugfix request → Sonnet",
                                  file_map=modality_flags.get('file_map'))
        if len(code_attachments) > 3 or _has_architecture_keywords(message_lower):
            return _make_decision(JobType.ORCHESTRATOR, "Multi-file or architecture-level code task → Opus",
                                  file_map=modality_flags.get('file_map'))
        elif _has_scoped_code_keywords(message_lower):
            return _make_decision(JobType.CODE_MEDIUM, "Scoped code task (1-3 files) → Sonnet",
                                  file_map=modality_flags.get('file_map'))
        else:
            return _make_decision(JobType.CODE_MEDIUM, "Code task (defaulting to Sonnet)",
                                  file_map=modality_flags.get('file_map'))

    # 7. IMAGE ATTACHMENTS → GEMINI 2.5 PRO
    if image_attachments:
        image_count = len(image_attachments)
        total_image_size = sum(a.size_bytes for a in image_attachments)
        return _make_decision(
            JobType.IMAGE_COMPLEX,
            f"Image analysis: {image_count} image(s) → Gemini 2.5 Pro",
            file_map=modality_flags.get('file_map'),
        )

    # 7.5. MIXED_FILE → GEMINI 2.5 PRO
    if modality_flags["has_mixed"]:
        mixed_count = modality_flags["mixed_count"]
        return _make_decision(
            JobType.IMAGE_COMPLEX,
            f"Documents with embedded images: {mixed_count} mixed file(s) → Gemini 2.5 Pro (vision required)",
            file_map=modality_flags.get('file_map'),
        )

    # 8. DOCUMENT FILES → Smart routing
    doc_attachments = [a for a in attachments
                       if a.filename.lower().endswith(('.md', '.txt', '.csv', '.json', '.yaml', '.yml'))]
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
        if any(kw in message_lower for kw in architecture_keywords):
            return _make_decision(JobType.ORCHESTRATOR,
                                  f"Document with architecture design request: {len(doc_attachments)} file(s)",
                                  file_map=modality_flags.get('file_map'))
        if any(kw in message_lower for kw in simple_doc_keywords):
            return _make_decision(JobType.CHAT_LIGHT,
                                  f"Document task: {len(doc_attachments)} file(s), simple processing",
                                  file_map=modality_flags.get('file_map'))

    # 9. PDF ATTACHMENTS → DETERMINISTIC ROUTING
    pdf_attachments = [a for a in attachments if a.is_pdf]
    if pdf_attachments:
        return _classify_pdf(pdf_attachments, message_lower, modality_flags.get('file_map'))

    # 10. TEXT COMPLEXITY DETECTION
    if _has_heavy_text_keywords(message_lower):
        return _make_decision(JobType.TEXT_HEAVY, "Heavy text work detected")

    # 11. LEGACY JOB TYPE NORMALIZATION
    if requested_job_type:
        try:
            legacy_type = JobType(requested_job_type)
            primary_type = RoutingConfig.normalize_job_type(legacy_type)
            return _make_decision(primary_type, f"Normalized from legacy type: {requested_job_type}")
        except ValueError:
            logger.warning(f"Unknown job type requested: {requested_job_type}")

    # 12. DEFAULT → CHAT_LIGHT
    return _make_decision(JobType.CHAT_LIGHT, "Default classification (casual chat)")
