# FILE: app/orchestrator/integration_check.py
"""
Cross-Segment Integration Check (Phase 3).

Verifies that segments produced by the orchestrator segment loop actually
work together. Runs AFTER all segments complete, BEFORE the final summary.

Two tiers:
    Tier 1 - Deterministic (no LLM): AST parsing, regex, filesystem checks.
        1. Import resolution: cross-segment imports resolve to real definitions
        2. Interface contracts: exposes/consumes match actual file contents
        3. File references: cross-segment path references are correct
        4. Duplicate definitions: no conflicting table/route/export names

    Tier 2 - Lightweight LLM (optional, advisory): single LLM call for
        semantic compatibility, naming consistency, integration completeness.
        Produces warnings only, never errors.

Design:
    - READ-ONLY: inspects output files, never modifies them
    - Host-direct filesystem access (same pattern as file_verifier.py)
    - Dataclass-based results with to_dict()/from_dict() (matching segment_schemas.py)
    - All logging uses [INTEGRATION_CHECK] prefix
    - Crash-safe: exceptions caught and reported, never crash the segment loop

Phase 3 of Pipeline Segmentation.

v1.0 (2026-02-08): Initial implementation
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from app.orchestrator._integration_check_utils_6 import INTEGRATION_CHECK_BUILD_ID, _DEFAULT_PROJECT_ROOTS, _PY_ROUTE_RE, _SQL_TABLE_RE, _TS_ROUTE_RE, _collect_segment_outputs, _get_project_roots, _verify_exposes
from app.orchestrator._integration_check_utils_7 import _check_duplicate_definitions, _check_interface_contracts, _check_typescript_cross_imports, _looks_like_project_import, _module_to_expected_path, _normalise_path, _run_llm_integration_review, _verify_consumes
from app.orchestrator._integration_check_utils_8 import IntegrationCheckResult, _build_file_to_segment_map, _check_file_references, _check_import_resolution, _check_python_cross_imports, run_integration_check

logger = logging.getLogger(__name__)
print(f"[INTEGRATION_CHECK_LOADED] BUILD_ID={INTEGRATION_CHECK_BUILD_ID}")

# --- Internal imports ---
from app.pot_spec.grounded.segment_schemas import (
    InterfaceContract,
    SegmentManifest,
    SegmentSpec,
    SegmentStatus,
)
from app.orchestrator.segment_state import JobState, SegmentState
from app.orchestrator.ast_helpers import (
    extract_python_definitions,
    extract_typescript_exports,
    resolve_python_import,
    resolve_typescript_import,
    get_all_defined_names,
    get_all_imports,
)

# Type alias
ProgressCallback = Optional[Callable[[str], None]]


# =============================================================================
# RESULT MODELS
# =============================================================================


@dataclass
class IntegrationIssue:
    """A single cross-segment integration issue."""

    severity: str       # "error" | "warning" | "info"
    check_type: str     # "import_resolution" | "interface_contract" | "file_reference" | "duplicate_definition" | "llm_review"
    segment_a: str      # segment_id of the producer
    segment_b: str      # segment_id of the consumer (or "N/A" for duplicates)
    file_a: str         # file in segment A
    file_b: str         # file in segment B
    expected: str       # what the contract/import says
    actual: str         # what was actually found (or "missing")
    message: str        # human-readable description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "check_type": self.check_type,
            "segment_a": self.segment_a,
            "segment_b": self.segment_b,
            "file_a": self.file_a,
            "file_b": self.file_b,
            "expected": self.expected,
            "actual": self.actual,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntegrationIssue":
        return cls(
            severity=data.get("severity", "error"),
            check_type=data.get("check_type", "unknown"),
            segment_a=data.get("segment_a", ""),
            segment_b=data.get("segment_b", ""),
            file_a=data.get("file_a", ""),
            file_b=data.get("file_b", ""),
            expected=data.get("expected", ""),
            actual=data.get("actual", ""),
            message=data.get("message", ""),
        )

# =============================================================================
# HELPERS
# =============================================================================

# =============================================================================
# TIER 1: IMPORT RESOLUTION
# =============================================================================

# =============================================================================
# TIER 1: INTERFACE CONTRACT VERIFICATION
# =============================================================================

# =============================================================================
# TIER 1: FILE REFERENCE CONSISTENCY
# =============================================================================


# =============================================================================
# TIER 1: DUPLICATE DEFINITION DETECTION
# =============================================================================

# =============================================================================
# TIER 2: LIGHTWEIGHT LLM REVIEW
# =============================================================================

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


__all__ = [
    "IntegrationIssue",
    "IntegrationCheckResult",
    "run_integration_check",
]