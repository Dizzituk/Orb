# FILE: app/pot_spec/grounded/spec_generation.py
"""
SpecGate Grounded POT Spec Builder - Facade Module

This module serves as the backward-compatible facade for the refactored
SpecGate system. It re-exports all public APIs from the modular components.

Version History:
- v2.0 (2026-02-01): MAJOR REFACTOR - Split into focused modules
  - text_helpers.py: Path/keyword extraction utilities
  - multi_file_detection.py: Multi-file operation detection and building
  - weaver_parser.py: Weaver intent parsing
  - grounding_engine.py: Core grounding logic
  - question_generator.py: Question generation
  - step_derivation.py: Step/test derivation from domains
  - completeness_checker.py: Spec completeness checking
  - markdown_builder.py: POT spec markdown builder
  - spec_runner.py: Main async entry point

Original monolithic file was 144KB. Now split into 9 focused modules.

Usage:
    # All existing imports continue to work:
    from app.pot_spec.grounded.spec_generation import run_spec_gate_grounded
    from app.pot_spec.grounded.spec_generation import build_pot_spec_markdown
    
    # Or import from specific modules for clarity:
    from app.pot_spec.grounded.spec_runner import run_spec_gate_grounded
    from app.pot_spec.grounded.markdown_builder import build_pot_spec_markdown

Version: v2.0 (2026-02-01) - Modular refactor facade
Build ID: 2026-02-01-v2.0-intelligent-classification
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Build ID for verification
BUILD_ID = "2026-02-01-v2.0-intelligent-classification"
print(f"[SPEC_GENERATION_LOADED] BUILD_ID={BUILD_ID} (FACADE MODULE)")
logger.info(f"[spec_generation] Facade module loaded: BUILD_ID={BUILD_ID}")

# =============================================================================
# RE-EXPORTS FROM MODULAR COMPONENTS
# =============================================================================

# Main entry point
from .spec_runner import run_spec_gate_grounded

# Markdown builder
from .markdown_builder import build_pot_spec_markdown

# Text helpers
from .text_helpers import (
    _extract_paths_from_text,
    _extract_keywords,
)

# Multi-file detection
from .multi_file_detection import (
    MULTI_FILE_SCOPE_INDICATORS,
    UNICODE_QUOTES,
    _has_multi_file_scope,
    _normalize_quotes,
    _extract_search_and_replace_terms,
    _detect_multi_file_intent,
    _convert_discovery_to_raw_matches,
    _build_multi_file_operation,
)

# Weaver parser
from .weaver_parser import parse_weaver_intent

# Grounding engine
from .grounding_engine import ground_intent_with_evidence

# Question generator
from .question_generator import generate_grounded_questions

# Step derivation
from .step_derivation import (
    _derive_steps_from_domain,
    _derive_tests_from_domain,
)

# Completeness checker
from .completeness_checker import _is_spec_complete_enough

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Main entry point
    "run_spec_gate_grounded",
    
    # Markdown builder
    "build_pot_spec_markdown",
    
    # Text helpers
    "_extract_paths_from_text",
    "_extract_keywords",
    
    # Multi-file detection
    "MULTI_FILE_SCOPE_INDICATORS",
    "UNICODE_QUOTES",
    "_has_multi_file_scope",
    "_normalize_quotes",
    "_extract_search_and_replace_terms",
    "_detect_multi_file_intent",
    "_convert_discovery_to_raw_matches",
    "_build_multi_file_operation",
    
    # Weaver parser
    "parse_weaver_intent",
    
    # Grounding engine
    "ground_intent_with_evidence",
    
    # Question generator
    "generate_grounded_questions",
    
    # Step derivation
    "_derive_steps_from_domain",
    "_derive_tests_from_domain",
    
    # Completeness checker
    "_is_spec_complete_enough",
    
    # Build ID
    "BUILD_ID",
]


# =============================================================================
# MODULE VERIFICATION
# =============================================================================

def _verify_modules():
    """Verify all sub-modules are properly loaded."""
    modules = [
        ("text_helpers", _extract_paths_from_text),
        ("multi_file_detection", _detect_multi_file_intent),
        ("weaver_parser", parse_weaver_intent),
        ("grounding_engine", ground_intent_with_evidence),
        ("question_generator", generate_grounded_questions),
        ("step_derivation", _derive_steps_from_domain),
        ("completeness_checker", _is_spec_complete_enough),
        ("markdown_builder", build_pot_spec_markdown),
        ("spec_runner", run_spec_gate_grounded),
    ]
    
    for name, func in modules:
        if func is None:
            logger.error(f"[spec_generation] FAILED TO LOAD: {name}")
            return False
        logger.debug(f"[spec_generation] Loaded: {name}")
    
    return True


# Run verification on import
_modules_ok = _verify_modules()
if _modules_ok:
    logger.info("[spec_generation] All 9 sub-modules loaded successfully")
else:
    logger.error("[spec_generation] Some sub-modules failed to load!")
