# FILE: app/astra_memory/__init__.py
"""
ASTRA Memory System (Job 5) - v2.0

Three layers of memory:
1. Job-local: Per-job state, execution timeline, Overwatcher verdicts
2. Global brain: Cross-job preferences, patterns, lessons
3. Overwatcher state: Risk scores, intervention history, cross-job patterns

v2.0 Additions:
- Confidence system with Record and Preference confidence types
- Evidence ledger (append-only) for preference learning
- Hot/cold storage with summary pyramids
- Intent depth gating (D0-D4) for fast recall
- 2-stage retrieval (cheap candidate selection → depth-gated expansion)

Tables:
- astra_jobs: Job registry with spec/arch links
- astra_job_files: Files touched (Atlas link)
- astra_job_events: Ledger index
- astra_job_chunks: Execution chunks
- astra_overwatch_summary: Per-job Overwatcher stats
- astra_global_prefs: Cross-job preferences (legacy)
- astra_overwatch_patterns: Cross-job patterns
- astra_preferences: User preferences with confidence (v2.0)
- astra_preference_evidence: Append-only evidence ledger (v2.0)
- astra_hot_index: Fast retrieval index (v2.0)
- astra_summary_pyramids: Multi-level summaries (v2.0)
- astra_memory_confidence: Confidence for non-preference records (v2.0)
"""

# =============================================================================
# ORIGINAL MODELS
# =============================================================================

from app.astra_memory.models import (
    AstraJob,
    JobFile,
    JobEvent,
    JobChunk,
    OverwatchSummary,
    GlobalPref,
    OverwatchPattern,
)

# =============================================================================
# v2.0 PREFERENCE MODELS
# =============================================================================

from app.astra_memory.preference_models import (
    # Enums
    ConfidenceType,
    PreferenceStrength,
    RecordStatus,
    SignalType,
    RetrievalCost,
    IntentDepth,
    # Models
    PreferenceRecord,
    PreferenceEvidence,
    HotIndex,
    SummaryPyramid,
    MemoryRecordConfidence,
)

# =============================================================================
# ORIGINAL SERVICE FUNCTIONS
# =============================================================================

from app.astra_memory.service import (
    # Job lifecycle
    create_job,
    update_job_status,
    link_spec_to_job,
    link_arch_to_job,
    # File tracking
    record_file_touch,
    get_files_for_job,
    get_jobs_for_file,
    # Event projection
    project_event_to_db,
    get_events_for_job,
    # Chunk tracking
    create_chunk,
    update_chunk_status,
    # Overwatcher
    get_or_create_overwatch_summary,
    record_overwatch_intervention,
    record_overwatch_pattern,
    get_patterns_for_file,
    # Global prefs (legacy)
    set_global_pref,
    get_global_pref,
    get_prefs_for_component,
    # Queries
    get_job,
    get_jobs_by_status,
    get_escalated_jobs,
)

# =============================================================================
# v2.0 CONFIDENCE SCORING
# =============================================================================

from app.astra_memory.confidence_scoring import (
    # Core scoring
    compute_decay,
    compute_weighted_sum,
    weighted_sum_to_confidence,
    compute_confidence_score,
    # Preference confidence
    recompute_preference_confidence,
    batch_recompute_confidence,
    # Evidence recording
    append_preference_evidence,
    record_contradiction,
    # Record confidence
    get_or_create_record_confidence,
    update_record_confidence,
    # Namespace
    check_namespace_mutation_allowed,
)

# =============================================================================
# v2.0 PREFERENCE SERVICE
# =============================================================================

from app.astra_memory.preference_service import (
    # Creation
    create_preference,
    create_hard_rule,
    # Updates
    update_preference_value,
    reinforce_preference,
    # Queries
    get_preference,
    get_preference_value,
    get_preferences_for_component,
    get_hard_rules,
    get_disputed_preferences,
    # Behavior rules
    resolve_preference_for_default,
    should_apply_preference_silently,
    # Learning
    learn_from_behavior,
    # Resolution
    resolve_disputed_preference,
    expire_preference,
)

# =============================================================================
# v2.0 RETRIEVAL
# =============================================================================

from app.astra_memory.retrieval import (
    # Intent classification
    classify_intent_depth,
    EXPLICIT_DEPTH_TOKENS,
    # Retrieval types
    RetrievalCandidate,
    ExpandedRecord,
    RetrievalResult,
    # Core retrieval
    stage1_candidate_selection,
    stage2_expand_candidates,
    retrieve_for_query,
    # Preference retrieval
    get_applicable_preferences,
    get_highest_confidence_preference,
    should_apply_preference,
    # Hot index management
    upsert_hot_index,
    upsert_summary_pyramid,
)

# =============================================================================
# v2.0 CONFIGURATION
# =============================================================================

from app.astra_memory.confidence_config import (
    EvidenceWeights,
    ConfidenceThresholds,
    DecayConfig,
    RetrievalDepthConfig,
    NamespaceConfig,
    ConfidenceSystemConfig,
    DEFAULT_CONFIG,
    get_config,
    get_evidence_weight,
)

# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    # === Original Models ===
    "AstraJob",
    "JobFile",
    "JobEvent",
    "JobChunk",
    "OverwatchSummary",
    "GlobalPref",
    "OverwatchPattern",
    
    # === v2.0 Enums ===
    "ConfidenceType",
    "PreferenceStrength",
    "RecordStatus",
    "SignalType",
    "RetrievalCost",
    "IntentDepth",
    
    # === v2.0 Models ===
    "PreferenceRecord",
    "PreferenceEvidence",
    "HotIndex",
    "SummaryPyramid",
    "MemoryRecordConfidence",
    
    # === Original Service Functions ===
    "create_job",
    "update_job_status",
    "link_spec_to_job",
    "link_arch_to_job",
    "record_file_touch",
    "get_files_for_job",
    "get_jobs_for_file",
    "project_event_to_db",
    "get_events_for_job",
    "create_chunk",
    "update_chunk_status",
    "get_or_create_overwatch_summary",
    "record_overwatch_intervention",
    "record_overwatch_pattern",
    "get_patterns_for_file",
    "set_global_pref",
    "get_global_pref",
    "get_prefs_for_component",
    "get_job",
    "get_jobs_by_status",
    "get_escalated_jobs",
    
    # === v2.0 Confidence Scoring ===
    "compute_decay",
    "compute_weighted_sum",
    "weighted_sum_to_confidence",
    "compute_confidence_score",
    "recompute_preference_confidence",
    "batch_recompute_confidence",
    "append_preference_evidence",
    "record_contradiction",
    "get_or_create_record_confidence",
    "update_record_confidence",
    "check_namespace_mutation_allowed",
    
    # === v2.0 Preference Service ===
    "create_preference",
    "create_hard_rule",
    "update_preference_value",
    "reinforce_preference",
    "get_preference",
    "get_preference_value",
    "get_preferences_for_component",
    "get_hard_rules",
    "get_disputed_preferences",
    "resolve_preference_for_default",
    "should_apply_preference_silently",
    "learn_from_behavior",
    "resolve_disputed_preference",
    "expire_preference",
    
    # === v2.0 Retrieval ===
    "classify_intent_depth",
    "EXPLICIT_DEPTH_TOKENS",
    "RetrievalCandidate",
    "ExpandedRecord",
    "RetrievalResult",
    "stage1_candidate_selection",
    "stage2_expand_candidates",
    "retrieve_for_query",
    "get_applicable_preferences",
    "get_highest_confidence_preference",
    "should_apply_preference",
    "upsert_hot_index",
    "upsert_summary_pyramid",
    
    # === v2.0 Configuration ===
    "EvidenceWeights",
    "ConfidenceThresholds",
    "DecayConfig",
    "RetrievalDepthConfig",
    "NamespaceConfig",
    "ConfidenceSystemConfig",
    "DEFAULT_CONFIG",
    "get_config",
    "get_evidence_weight",
]
