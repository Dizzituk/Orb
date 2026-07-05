# FILE: app/llm/job_classifier.py
# Purpose: Job classification for LLM routing.
# Called-by: app.endpoints.chat_attachments, app.llm._job_classifier_classify, app.llm._job_classifier_utils, app.llm.pipeline.high_stakes (+5 more)
# Depends-on: app.llm._job_classifier_classify, app.llm._job_classifier_utils, app.llm.file_classifier, app.llm.schemas
# Last-renovated: 2026-06-11
"""
Job classification for LLM routing.

Version: 0.15.1 - Simplified OVERRIDE → Frontier Model Routing

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
from app.llm._job_classifier_utils import (
    VIDEO_SIZE_THRESHOLD, _has_complex_vision_keywords,
    _has_video_deep_analysis_keywords, get_provider_for_job,
    get_routing_for_job_type, is_claude_allowed, is_claude_forbidden,
    is_vision_job,
)

# Classification logic (extracted for modularity)
from app.llm._job_classifier_classify import (
    classify_job,
    _make_decision,
    _classify_pdf,
    _detect_user_override,
    _has_code_keywords,
    _has_scoped_code_keywords,
    _has_architecture_keywords,
    _has_heavy_text_keywords,
    _debug_log,
    ROUTER_DEBUG,
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
    class FileType:
        TEXT_FILE = "TEXT_FILE"
        CODE_FILE = "CODE_FILE"
        IMAGE_FILE = "IMAGE_FILE"
        VIDEO_FILE = "VIDEO_FILE"
        MIXED_FILE = "MIXED_FILE"

logger = logging.getLogger(__name__)


# ============================================================================
# v0.15.1: FRONTIER MODEL CONFIGURATION
# ============================================================================

# LANE D (2026-07-02): literal fallbacks removed — the three *_FRONTIER_MODEL_ID
# vars are seeded in .env with the old effective values. Module-level constants
# (from-imported by routing/core and others), so env changes here are
# RESTART-GATED — flagged in the /settings/models audit.
# Ensure .env is loaded even outside the app boot path (tests/scripts) — the
# same pattern as seed_tiers/gemini_vision/model_env. override=False.
try:  # pragma: no cover - environment plumbing
    from dotenv import load_dotenv as _laned_load_dotenv
    _laned_load_dotenv()
except Exception:
    pass
from app.llm.frontier_models import get_provider_default_model as _provider_default

GEMINI_FRONTIER_MODEL_ID = os.getenv("GEMINI_FRONTIER_MODEL_ID") or _provider_default("google", strict=False)
ANTHROPIC_FRONTIER_MODEL_ID = os.getenv("ANTHROPIC_FRONTIER_MODEL_ID") or _provider_default("anthropic", strict=False)
OPENAI_FRONTIER_MODEL_ID = os.getenv("OPENAI_FRONTIER_MODEL_ID") or _provider_default("openai", strict=False)


# ============================================================================
# v0.15.1: SIMPLIFIED OVERRIDE DETECTION
# ============================================================================

def detect_frontier_override(message: str) -> Optional[Tuple[str, str, str]]:
    """
    Detect OVERRIDE command at start of a line and return frontier model routing.

    Returns:
        Tuple of (provider, model_id, cleaned_message) if OVERRIDE found
        None if no OVERRIDE detected
    """
    if not message:
        return None

    lines = message.split('\n')
    override_line_idx = None
    override_payload = ""

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper().startswith("OVERRIDE"):
            override_line_idx = idx
            if len(stripped) > 8:
                override_payload = stripped[8:].strip()
            else:
                override_payload = ""
            break

    if override_line_idx is None:
        return None

    payload_lower = override_payload.lower()

    if ROUTER_DEBUG:
        _debug_log(f"OVERRIDE DETECTED at line {override_line_idx}")
        _debug_log(f"  Payload: '{override_payload}'")

    if any(kw in payload_lower for kw in ["claude", "anthropic", "opus"]):
        force_provider = "anthropic"
        force_model_id = ANTHROPIC_FRONTIER_MODEL_ID
    elif any(kw in payload_lower for kw in ["chatgpt", "openai", "gpt"]):
        force_provider = "openai"
        force_model_id = OPENAI_FRONTIER_MODEL_ID
    elif any(kw in payload_lower for kw in ["gemini", "google"]):
        force_provider = "google"
        force_model_id = GEMINI_FRONTIER_MODEL_ID
    else:
        force_provider = "google"
        force_model_id = GEMINI_FRONTIER_MODEL_ID

    if ROUTER_DEBUG:
        _debug_log(f"  → {force_provider} frontier: {force_model_id}")

    cleaned_lines = lines[:override_line_idx] + lines[override_line_idx + 1:]
    cleaned_message = '\n'.join(cleaned_lines).strip()

    if not cleaned_message:
        cleaned_message = "Analyze and describe the content in detail."
        if ROUTER_DEBUG:
            _debug_log("  → Empty message after OVERRIDE removal, using default prompt")

    return (force_provider, force_model_id, cleaned_message)


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
# MODALITY FLAG HELPER
# =============================================================================

def compute_modality_flags(
    attachments: List[AttachmentInfo],
    base_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute modality flags from attachments."""
    video_attachments = [a for a in attachments if a.is_video]
    image_attachments = [a for a in attachments if a.is_image]
    code_attachments = [a for a in attachments if a.is_code]
    text_attachments = [a for a in attachments if a.is_document and not a.is_pdf]
    pdf_attachments = [a for a in attachments if a.is_pdf]

    classification_result = None
    file_map = None
    has_mixed = False
    mixed_count = 0

    if FILE_CLASSIFIER_AVAILABLE and attachments:
        try:
            classification_result = classify_from_attachment_info(attachments)
            file_map = build_file_map(classification_result)
            has_mixed = classification_result.HAS_MIXED
            mixed_count = len(classification_result.mixed_files)
            if ROUTER_DEBUG:
                _debug_log(f"file_classifier results:")
                _debug_log(f"  HAS_MIXED: {classification_result.HAS_MIXED}")
                _debug_log(f"  mixed_files: {[f.file_id for f in classification_result.mixed_files]}")
        except Exception as e:
            logger.warning(f"file_classifier failed, using fallback: {e}")

    if not FILE_CLASSIFIER_AVAILABLE:
        for pdf in pdf_attachments:
            if pdf.pdf_image_count and pdf.pdf_image_count > 0:
                has_mixed = True
                mixed_count += 1

    return {
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
        "has_mixed": has_mixed,
        "mixed_count": mixed_count,
        "classification_result": classification_result,
        "file_map": file_map,
    }


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
        if isinstance(att, str):
            att = {"filename": att}
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


def get_model_config() -> Dict[str, str]:
    """Get current model configuration (env-only; LANE D — no literal fallbacks)."""
    return {
        "openai": os.getenv("OPENAI_MODEL_LIGHT_CHAT") or _provider_default("openai", strict=False),
        "openai_heavy": os.getenv("OPENAI_MODEL_HEAVY_TEXT") or _provider_default("openai", strict=False),
        "anthropic_sonnet": os.getenv("ANTHROPIC_SONNET_MODEL") or _provider_default("anthropic", strict=False),
        "anthropic_opus": os.getenv("ANTHROPIC_OPUS_MODEL") or _provider_default("anthropic", strict=False),
        "gemini_fast": os.getenv("GEMINI_VISION_MODEL_FAST") or _provider_default("google", strict=False),
        "gemini_complex": os.getenv("GEMINI_VISION_MODEL_COMPLEX") or _provider_default("google", strict=False),
        "gemini_video": os.getenv("GEMINI_VIDEO_HEAVY_MODEL") or _provider_default("google", strict=False),
        "gemini_critic": os.getenv("GEMINI_OPUS_CRITIC_MODEL") or _provider_default("google", strict=False),
        "gemini_frontier": GEMINI_FRONTIER_MODEL_ID,
        "anthropic_frontier": ANTHROPIC_FRONTIER_MODEL_ID,
        "openai_frontier": OPENAI_FRONTIER_MODEL_ID,
    }


# =============================================================================
# FILE MAP HELPERS
# =============================================================================

def get_file_map_for_attachments(attachments: List[AttachmentInfo]) -> Optional[str]:
    """Generate stable file map for attachments."""
    if not FILE_CLASSIFIER_AVAILABLE or not attachments:
        return None
    try:
        result = classify_from_attachment_info(attachments)
        return build_file_map(result)
    except Exception as e:
        logger.warning(f"Failed to generate file map: {e}")
        return None
