# FILE: app/astra_memory/confidence_config.py
"""
ASTRA Memory Confidence System - Configuration

Centralized config for all confidence scoring parameters.
All knobs referenced in the spec are defined here.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class EvidenceWeights:
    """
    Weight values for different signal types in preference learning.
    
    These determine how much each type of evidence contributes to
    preference confidence scoring.
    """
    # Explicit signals (high confidence)
    explicit_instruction: float = 3.0  # "save to memory", "always do X"
    explicit_approval: float = 2.0     # User confirms a suggestion
    
    # Implicit signals (lower confidence)
    implicit_repeated: float = 0.5     # Same behavior observed multiple times
    one_off_choice: float = 0.2        # Single instance of a choice
    
    # Negative signals
    contradiction: float = -3.0        # User contradicts previous preference


@dataclass(frozen=True)
class ConfidenceThresholds:
    """
    Thresholds for confidence-based behavior.
    """
    # Below this: treat as suggestion only, don't enforce
    suggestion_threshold: float = 0.65
    
    # At or above this: apply silently as default
    apply_threshold: float = 0.85
    
    # Minimum evidence count for implicit-only preferences
    min_evidence_count: int = 2


@dataclass(frozen=True)
class DecayConfig:
    """
    Time-based decay parameters for confidence scoring.
    """
    # Half-life in days for weak/soft signals
    # After this many days, signal weight is halved
    half_life_days: float = 30.0
    
    # Saturation speed for confidence mapping
    # Higher k = faster saturation toward 1.0
    k: float = 0.5


@dataclass(frozen=True)
class RetrievalDepthConfig:
    """
    Configuration for intent depth gating (D0-D4).
    """
    # Maximum items to expand at each depth
    d0_max_items: int = 0      # Chat: no memory
    d1_max_items: int = 5      # Brief: small, fast
    d2_max_items: int = 15     # Normal: targeted
    d3_max_items: int = 50     # Deep: heavy
    d4_max_items: int = 200    # Forensic: everything
    
    # Token caps per depth (approximate)
    d0_token_cap: int = 0
    d1_token_cap: int = 500
    d2_token_cap: int = 2000
    d3_token_cap: int = 8000
    d4_token_cap: int = 32000
    
    # Summary pyramid level selection per depth
    # L0=1 sentence, L1=5 bullets, L2=1-2 paragraphs, L3=full
    d1_summary_level: int = 0  # L0/L1
    d2_summary_level: int = 1  # L1/L2
    d3_summary_level: int = 2  # L2/L3


@dataclass(frozen=True)
class NamespaceConfig:
    """
    Namespace separation rules.
    """
    # Namespaces that cannot cross-contaminate
    protected_namespaces: tuple = (
        "user_personal",      # Long-term personal preferences
        "safety_critical",    # Safety rules
        "hard_rules",         # Immutable preferences
    )
    
    # Namespaces that can be updated by repo scans
    repo_mutable_namespaces: tuple = (
        "repo_derived",       # Facts from repo analysis
        "atlas_nodes",        # Architecture entities
        "code_patterns",      # Detected code patterns
    )


@dataclass
class ConfidenceSystemConfig:
    """
    Master configuration for the confidence system.
    """
    evidence_weights: EvidenceWeights = field(default_factory=EvidenceWeights)
    thresholds: ConfidenceThresholds = field(default_factory=ConfidenceThresholds)
    decay: DecayConfig = field(default_factory=DecayConfig)
    retrieval: RetrievalDepthConfig = field(default_factory=RetrievalDepthConfig)
    namespaces: NamespaceConfig = field(default_factory=NamespaceConfig)


# Global default config instance
DEFAULT_CONFIG = ConfidenceSystemConfig()


def get_config() -> ConfidenceSystemConfig:
    """Get the current confidence system configuration."""
    return DEFAULT_CONFIG


# Convenience accessors
def get_evidence_weight(signal_type: str) -> float:
    """Get weight for a signal type."""
    cfg = get_config().evidence_weights
    weights = {
        "explicit": cfg.explicit_instruction,
        "explicit_instruction": cfg.explicit_instruction,
        "explicit_approval": cfg.explicit_approval,
        "approval": cfg.explicit_approval,
        "implicit": cfg.implicit_repeated,
        "implicit_repeated": cfg.implicit_repeated,
        "one_off": cfg.one_off_choice,
        "one_off_choice": cfg.one_off_choice,
        "contradiction": cfg.contradiction,
    }
    return weights.get(signal_type, 0.0)


__all__ = [
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
