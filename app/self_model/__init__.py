# FILE: app/self_model/__init__.py
# Purpose: Self-Model & Evolution Layer
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.self_model.capability_map, app.self_model.journal, app.self_model.observer, app.self_model.suggestions (+1 more)
# Last-renovated: 2026-06-11
"""
Self-Model & Evolution Layer

Top-level awareness layer for ASTRA. Provides:
- Capability map: what ASTRA can and cannot do, refreshed from architecture scans
- User model: transparent knowledge about the user, surfaced conversationally
- Pattern observer: detects recurring friction, failures, and opportunities
- Suggestion engine: proposes improvements in plain language, never acts alone
- Evolution journal: records what ASTRA has learned and suggested over time

The Self-Model never executes changes autonomously. It observes, learns,
connects, and suggests. Every action requires explicit user approval.

Part of ASTRA's partnership model: observe → suggest → wait → act only if approved.
"""

from app.self_model.capability_map import CapabilityMap, get_capability_map, refresh_capabilities
from app.self_model.user_model import UserModel, get_user_model
from app.self_model.observer import PatternObserver, get_observer
from app.self_model.suggestions import Suggestion, SuggestionEngine, get_suggestion_engine
from app.self_model.journal import EvolutionJournal, get_journal

__all__ = [
    "CapabilityMap", "get_capability_map", "refresh_capabilities",
    "UserModel", "get_user_model",
    "PatternObserver", "get_observer",
    "Suggestion", "SuggestionEngine", "get_suggestion_engine",
    "EvolutionJournal", "get_journal",
]
