# FILE: app/memory/domains/__init__.py
# Purpose: Memory domain stores package.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.memory.domains.architecture, app.memory.domains.confidence, app.memory.domains.confidence_learning, app.memory.domains.confidence_resolver (+7 more)
# Last-renovated: 2026-06-11
"""
Memory domain stores package.

Each module implements a DomainStore for one memory domain.
Stores are registered with the MemoryRouter at startup via
app/memory/startup.py.

Five domains (Spec Section 4):
    architecture — architecture.py (System Architecture Memory, 5-tier)
    preference   — preferences.py (User Preference Memory)
    context      — context.py (Conversation Context, ephemeral TTL)
    confidence   — confidence.py (Phrase-to-Intent Confidence Scores)
    knowledge    — knowledge.py (Per-Domain Factual Knowledge)

Confidence subsystem (Spec Section 5):
    confidence_resolver.py  — 4-band response behaviour
    confidence_learning.py  — Learning loop hooks (confirm/correct/timeout)

Supporting modules:
    decisions.py          — Tier 4 design decisions log
    dependency_scanner.py — Tier 3 dependency graph generator
    package_summariser.py — Tier 5 package summary generator
    tier_content.py       — Tier 1+2 curated content seeds

    preference_registry.py — Domain namespace validation
    preference_seeds.py    — Initial hard-rule preference seeds
    preference_capture.py  — Three capture channels (explicit, correction, inferred)
"""

# ── Domain stores ────────────────────────────────────────────────────
from app.memory.domains.architecture import ArchitectureStore
from app.memory.domains.preferences import PreferenceStore
from app.memory.domains.context import ContextStore
from app.memory.domains.confidence import ConfidenceStore
from app.memory.domains.knowledge import KnowledgeStore

# ── Confidence subsystem ─────────────────────────────────────────────
from app.memory.domains.confidence_resolver import (
    resolve_action,
    resolve_no_match,
    get_band_for_confidence,
    ConfidenceAction,
)
from app.memory.domains.confidence_learning import (
    on_confirmation,
    on_correction,
    on_timeout,
    run_decay,
    get_learning_stats,
)

# ── Preference utilities ─────────────────────────────────────────────
from app.memory.domains.preference_registry import (
    is_valid_domain,
    get_domain_info,
    list_domains,
    list_domain_names,
    validate_domain,
)
from app.memory.domains.preference_seeds import seed_preferences
from app.memory.domains.preference_capture import (
    capture_explicit,
    capture_correction,
    capture_inferred,
    detect_preference_intent,
)

# ── Decision store ───────────────────────────────────────────────────
from app.memory.domains.decisions import DecisionStore, seed_decisions

__all__ = [
    # Domain stores (all 5)
    "ArchitectureStore",
    "PreferenceStore",
    "ContextStore",
    "ConfidenceStore",
    "KnowledgeStore",
    # Confidence subsystem
    "resolve_action",
    "resolve_no_match",
    "get_band_for_confidence",
    "ConfidenceAction",
    "on_confirmation",
    "on_correction",
    "on_timeout",
    "run_decay",
    "get_learning_stats",
    # Decision store
    "DecisionStore",
    # Preference registry
    "is_valid_domain",
    "get_domain_info",
    "list_domains",
    "list_domain_names",
    "validate_domain",
    # Preference seeds
    "seed_preferences",
    # Preference capture channels
    "capture_explicit",
    "capture_correction",
    "capture_inferred",
    "detect_preference_intent",
    # Decision seeds
    "seed_decisions",
]
