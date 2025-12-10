# FILE: app/llm/__init__.py
"""
LLM module exports.

v0.15.0: Added Critical Pipeline Spec module exports.
v0.12.7: Added video analysis and is_video_mime_type exports.
v0.12.4: Added new vision functions for direct image Q&A.
"""

# ============== ROUTER EXPORTS ==============

from app.llm.router import call_llm, quick_chat, request_code, review_work
from app.llm.router import (
    analyze_with_vision,
    web_search_query,
    list_job_types,
    get_routing_info,
    is_policy_routing_enabled,
    enable_policy_routing,
)

# ============== SCHEMA EXPORTS ==============

from app.llm.schemas import LLMTask, LLMResult, JobType, Provider

# ============== VISION EXPORTS ==============

from app.llm.gemini_vision import (
    analyze_image,
    is_image_mime_type,
    ask_about_image,
    check_vision_available,
    get_vision_model_for_complexity,
    analyze_video,
)

# ============== FILE ANALYZER EXPORTS ==============

from app.llm.file_analyzer import (
    extract_text_content,
    detect_document_type,
    parse_cv_with_llm,
    generate_document_summary,
    is_video_mime_type,
    is_audio_mime_type,
)

# ============== CLIENT EXPORTS ==============

from app.llm.clients import (
    check_provider_availability,
    list_available_providers,
    get_embeddings,
)

# ============== v0.15.0: CRITICAL PIPELINE SPEC MODULES ==============

# File Classifier (Spec §1 & §2)
try:
    from app.llm.file_classifier import (
        classify_attachments,
        classify_from_attachment_info,
        build_file_map,
        FileType,
        ClassifiedFile,
        ClassificationResult,
        has_vision_content,
        has_any_media,
        get_file_by_id,
        get_files_by_type,
    )
    _file_classifier_exports = [
        "classify_attachments",
        "classify_from_attachment_info",
        "build_file_map",
        "FileType",
        "ClassifiedFile",
        "ClassificationResult",
        "has_vision_content",
        "has_any_media",
        "get_file_by_id",
        "get_files_by_type",
    ]
except ImportError:
    _file_classifier_exports = []

# Audit Logger (Spec §12)
try:
    from app.llm.audit_logger import (
        get_audit_logger,
        RoutingTrace,
        AuditEventType,
        TaskAuditLog,
        ModalityFlags,
        RelationshipFlags,
    )
    _audit_logger_exports = [
        "get_audit_logger",
        "RoutingTrace",
        "AuditEventType",
        "TaskAuditLog",
        "ModalityFlags",
        "RelationshipFlags",
    ]
except ImportError:
    _audit_logger_exports = []

# Relationship Detector (Spec §3)
try:
    from app.llm.relationship_detector import (
        detect_relationships,
        detect_relationships_heuristic,
        RelationshipType,
        RelationshipResult,
    )
    _relationship_detector_exports = [
        "detect_relationships",
        "detect_relationships_heuristic",
        "RelationshipType",
        "RelationshipResult",
    ]
except ImportError:
    _relationship_detector_exports = []

# Preprocessor (Spec §5 & §6)
try:
    from app.llm.preprocessor import (
        preprocess_task,
        build_task_context,
        build_critical_context,
        VideoPreprocessResult,
        ImagePreprocessResult,
        CodePreprocessResult,
        TextPreprocessResult,
        TaskPreprocessResult,
    )
    _preprocessor_exports = [
        "preprocess_task",
        "build_task_context",
        "build_critical_context",
        "VideoPreprocessResult",
        "ImagePreprocessResult",
        "CodePreprocessResult",
        "TextPreprocessResult",
        "TaskPreprocessResult",
    ]
except ImportError:
    _preprocessor_exports = []

# Token Budgeting (Spec §7)
try:
    from app.llm.token_budgeting import (
        allocate_budget,
        apply_truncation,
        create_budget_for_model,
        select_profile,
        TokenBudget,
        PROFILE_DEFAULT,
        PROFILE_CRITICAL,
        PROFILE_VIDEO_HEAVY,
        PROFILE_CODE_ONLY,
        PROFILE_TEXT_ONLY,
    )
    _token_budgeting_exports = [
        "allocate_budget",
        "apply_truncation",
        "create_budget_for_model",
        "select_profile",
        "TokenBudget",
        "PROFILE_DEFAULT",
        "PROFILE_CRITICAL",
        "PROFILE_VIDEO_HEAVY",
        "PROFILE_CODE_ONLY",
        "PROFILE_TEXT_ONLY",
    ]
except ImportError:
    _token_budgeting_exports = []

# Task Extractor (Spec §4)
try:
    from app.llm.task_extractor import (
        extract_tasks,
        extract_tasks_heuristic,
        detect_multi_task_heuristic,
        Task,
        TaskExtractionResult,
        Modality as TaskModality,
    )
    _task_extractor_exports = [
        "extract_tasks",
        "extract_tasks_heuristic",
        "detect_multi_task_heuristic",
        "Task",
        "TaskExtractionResult",
        "TaskModality",
    ]
except ImportError:
    _task_extractor_exports = []

# Fallbacks (Spec §11)
try:
    from app.llm.fallbacks import (
        FallbackHandler,
        handle_video_failure,
        handle_vision_failure,
        handle_overwatcher_failure,
        handle_critique_failure,
        get_fallback_chain,
        get_next_fallback,
        FailureType,
        FallbackAction,
        FallbackEvent,
        FallbackResult,
        FALLBACK_CHAINS,
    )
    _fallbacks_exports = [
        "FallbackHandler",
        "handle_video_failure",
        "handle_vision_failure",
        "handle_overwatcher_failure",
        "handle_critique_failure",
        "get_fallback_chain",
        "get_next_fallback",
        "FailureType",
        "FallbackAction",
        "FallbackEvent",
        "FallbackResult",
        "FALLBACK_CHAINS",
    ]
except ImportError:
    _fallbacks_exports = []

# ============== OPTIONAL POLICY EXPORTS ==============

try:
    from app.llm.policy import (
        load_routing_policy,
        get_policy_for_job,
        make_routing_decision,
        RoutingPolicy,
        RoutingDecision,
        JobPolicy,
        PolicyError,
        UnknownJobTypeError,
        DataValidationError,
        ProviderCapabilityError,
    )
    _policy_exports = [
        "load_routing_policy",
        "get_policy_for_job",
        "make_routing_decision",
        "RoutingPolicy",
        "RoutingDecision",
        "JobPolicy",
        "PolicyError",
        "UnknownJobTypeError",
        "DataValidationError",
        "ProviderCapabilityError",
    ]
except ImportError:
    _policy_exports = []

# ============== ALL EXPORTS ==============

__all__ = [
    # Router
    "call_llm",
    "quick_chat",
    "request_code",
    "review_work",
    "analyze_with_vision",
    "web_search_query",
    "list_job_types",
    "get_routing_info",
    "is_policy_routing_enabled",
    "enable_policy_routing",
    # Schemas
    "LLMTask",
    "LLMResult",
    "JobType",
    "Provider",
    # Vision
    "analyze_image",
    "is_image_mime_type",
    "ask_about_image",
    "check_vision_available",
    "get_vision_model_for_complexity",
    "analyze_video",
    # File analyzer
    "extract_text_content",
    "detect_document_type",
    "parse_cv_with_llm",
    "generate_document_summary",
    "is_video_mime_type",
    "is_audio_mime_type",
    # Clients
    "check_provider_availability",
    "list_available_providers",
    "get_embeddings",
] + (
    _file_classifier_exports +
    _audit_logger_exports +
    _relationship_detector_exports +
    _preprocessor_exports +
    _token_budgeting_exports +
    _task_extractor_exports +
    _fallbacks_exports +
    _policy_exports
)