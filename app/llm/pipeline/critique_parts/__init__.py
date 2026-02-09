"""Critique pipeline submodules (Block 5 decomposition).

This package contains the decomposed components of the critique pipeline,
extracted from the monolithic critique.py for maintainability.

Each module handles a single responsibility:
- model_config: Provider/model configuration
- blocker_filtering: Approved blocker type enforcement
- grounding_validation: Spec-ref fabrication detection
- section_authority: LLM-suggestion section handling
- evidence_resolution: CRITICAL_CLAIMS validation
- scope_creep: Endpoint drift and excluded feature detection
- spec_compliance: Platform/stack/scope mismatch detection
"""
