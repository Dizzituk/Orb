# FILE: app/grounding/__init__.py
# Purpose: Grounded Intelligence System — Package Init.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Grounded Intelligence System — Package Init.

Prevents ASTRA from generating confident responses to factual
or claims-dependent queries without first retrieving and anchoring
to current, verifiable sources. For contested topics, ensures
balanced multi-perspective presentation.

Modules:
- topic_classifier:       Categorises queries (personal | claims | contested)
- freshness_policy:       Domain-specific max data age rules
- grounding_gate:         Orchestrator that forces search before LLM responds
- source_attribution:     Natural citation formatting
- perspective_engine:     Debate type detection and perspective identification
- perspective_prompts:    Tailored system prompts per debate type
- perspective_formatter:  Multi-perspective evidence structuring
- chat_integration:       Wiring layer for chat_routing
"""
from __future__ import annotations

__all__ = [
    "topic_classifier",
    "freshness_policy",
    "grounding_gate",
    "source_attribution",
    "perspective_engine",
    "perspective_prompts",
    "perspective_formatter",
    "chat_integration",
]
