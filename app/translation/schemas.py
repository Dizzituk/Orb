# FILE: app/translation/schemas.py
"""
Pydantic models for the ASTRA Translation Layer.
Defines canonical intents, modes, gate results, and feedback structures.

v1.1 (2026-01): Added Spec Gate flow intents (WEAVER_BUILD_SPEC, SEND_TO_SPEC_GATE)
"""
from __future__ import annotations
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone


def utcnow():
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


class TranslationMode(str, Enum):
    """First classification step: what mode is this message?"""
    CHAT = "chat"                    # Normal conversation - NO backend actions
    COMMAND_CAPABLE = "command"      # User in command context (wake phrase or UI)
    FEEDBACK = "feedback"            # User correcting behavior


# FILE: app/translation/schemas.py

class CanonicalIntent(str, Enum):
    """Closed set of allowed intents. No free-form execution."""
    # Chat (no action)
    CHAT_ONLY = "CHAT_ONLY"
    
    # Architecture commands
    ARCHITECTURE_MAP_WITH_FILES = "ARCHITECTURE_MAP_WITH_FILES"
    ARCHITECTURE_MAP_STRUCTURE_ONLY = "ARCHITECTURE_MAP_STRUCTURE_ONLY"
    ARCHITECTURE_UPDATE_ATLAS_ONLY = "ARCHITECTURE_UPDATE_ATLAS_ONLY"
    
    # Sandbox control
    START_SANDBOX_ZOMBIE_SELF = "START_SANDBOX_ZOMBIE_SELF"
    SCAN_SANDBOX_STRUCTURE = "SCAN_SANDBOX_STRUCTURE"
    
    # =========================================================================
    # SPEC GATE FLOW (v1.1)
    # =========================================================================
    
    # Weaver: Build spec from ramble/conversation
    WEAVER_BUILD_SPEC = "WEAVER_BUILD_SPEC"
    
    # Send refined spec to Spec Gate for validation
    SEND_TO_SPEC_GATE = "SEND_TO_SPEC_GATE"
    
    # =========================================================================
    # HIGH-STAKES PIPELINE CONTROL (require confirmation)
    # =========================================================================
    
    # Pipeline control (high-stakes)
    RUN_CRITICAL_PIPELINE_FOR_JOB = "RUN_CRITICAL_PIPELINE_FOR_JOB"
    
    # Overwatcher (applies validated spec + pipeline result)
    OVERWATCHER_EXECUTE_CHANGES = "OVERWATCHER_EXECUTE_CHANGES"
    
    # Feedback (no action, just logging)
    USER_BEHAVIOR_FEEDBACK = "USER_BEHAVIOR_FEEDBACK"
    
    # =========================================================================
    # RAG ARCHITECTURE QUERY (v1.2)
    # =========================================================================
    
    # Search codebase and answer questions
    RAG_CODEBASE_QUERY = "RAG_CODEBASE_QUERY"
    
    # =========================================================================
    # EMBEDDING MANAGEMENT (v1.3)
    # =========================================================================
    
    # Check embedding status
    EMBEDDING_STATUS = "EMBEDDING_STATUS"
    
    # Trigger embedding generation
    GENERATE_EMBEDDINGS = "GENERATE_EMBEDDINGS"
    
    # =========================================================================
    # FILESYSTEM QUERY (v1.4)
    # =========================================================================
    
    # List/find files in scan index (no shell commands)
    FILESYSTEM_QUERY = "FILESYSTEM_QUERY"


class LatencyTier(str, Enum):
    """Which cost tier resolved this intent?"""
    TIER_0_RULES = "tier_0"      # Pure regex/string, no LLM
    TIER_1_CLASSIFIER = "tier_1" # GPT-5 mini classifier
    TIER_2_DEEP = "tier_2"       # Never for routing (only for actual work)


class GateResult(BaseModel):
    """Result of passing through a gate."""
    passed: bool
    gate_name: str
    reason: Optional[str] = None
    blocked_by: Optional[str] = None


class DirectiveGateResult(GateResult):
    """Specific result for directive vs story gate."""
    detected_pattern: Optional[str] = None  # e.g., "past_tense", "question", "future_planning"
    original_text_snippet: Optional[str] = None


class ContextGateResult(GateResult):
    """Result for context gate - checks required parameters."""
    missing_context: List[str] = Field(default_factory=list)
    provided_context: Dict[str, Any] = Field(default_factory=dict)


class ConfirmationGateResult(GateResult):
    """Result for high-stakes confirmation gate."""
    requires_confirmation: bool = False
    confirmation_prompt: Optional[str] = None
    awaiting_confirmation: bool = False


class TranslationResult(BaseModel):
    """Complete result of translating a user message."""
    # Input
    original_text: str
    timestamp: datetime = Field(default_factory=utcnow)
    
    # Mode classification
    mode: TranslationMode
    wake_phrase_detected: Optional[str] = None
    
    # Intent resolution
    resolved_intent: CanonicalIntent
    intent_confidence: float = 1.0  # 0.0-1.0, 1.0 for rule-based
    latency_tier: LatencyTier
    
    # Gate results
    directive_gate: Optional[DirectiveGateResult] = None
    context_gate: Optional[ContextGateResult] = None
    confirmation_gate: Optional[ConfirmationGateResult] = None
    
    # Final decision
    should_execute: bool = False
    execution_blocked_reason: Optional[str] = None
    
    # For commands that need context
    extracted_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Cache hit info
    from_phrase_cache: bool = False
    cache_pattern_matched: Optional[str] = None


class FeedbackEvent(BaseModel):
    """Structured feedback for misfire learning."""
    timestamp: datetime = Field(default_factory=utcnow)
    original_text: str
    resolved_intent: Optional[CanonicalIntent] = None
    expected_intent: CanonicalIntent
    feedback_type: str  # "false_positive" or "false_negative"
    user_correction: Optional[str] = None
    translation_result: Optional[TranslationResult] = None


class PhraseCacheEntry(BaseModel):
    """Entry in the phrase→intent cache."""
    pattern: str                    # Normalized pattern
    intent: CanonicalIntent
    confidence: float = 1.0
    hit_count: int = 0
    last_hit: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)
    source: str = "feedback"        # "feedback", "high_confidence_classification", "manual"
    promoted_to_tier0: bool = False # Has this been promoted to rule-based?


class IntentDefinition(BaseModel):
    """Definition of a canonical intent and its behavior."""
    intent: CanonicalIntent
    trigger_phrases: List[str]          # Exact matches (case-sensitive where noted)
    trigger_patterns: List[str] = Field(default_factory=list)  # Regex patterns
    requires_context: List[str] = Field(default_factory=list)  # Required context fields
    requires_confirmation: bool = False
    confirmation_prompt: Optional[str] = None
    description: str
    behavior: str


class ClassifierRequest(BaseModel):
    """Request to Tier 1 classifier."""
    text: str
    candidate_intents: List[CanonicalIntent]
    context: Dict[str, Any] = Field(default_factory=dict)


class ClassifierResponse(BaseModel):
    """Response from Tier 1 classifier."""
    intent: CanonicalIntent
    confidence: float
    reasoning: Optional[str] = None
