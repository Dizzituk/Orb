# FILE: app/llm/critical_pipeline/job_classification.py
"""
Job type classification for Critical Pipeline routing.

Determines whether a job is MICRO_EXECUTION, SCAN_ONLY, or ARCHITECTURE
based on SpecGate's pre-classification or keyword-based fallback heuristics.
"""

import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)


# =============================================================================
# Job Kind Constants
# =============================================================================

class JobKind:
    """Job classification for pipeline routing."""
    MICRO_EXECUTION = "micro_execution"   # Simple read/write/answer tasks
    ARCHITECTURE = "architecture"          # Design/build/refactor tasks
    SCAN_ONLY = "scan_only"               # Read-only scan/search/enumerate jobs


# =============================================================================
# Classification
# =============================================================================

def classify_job_kind(spec_data: Dict[str, Any], message: str) -> str:
    """
    Classify job as MICRO_EXECUTION, SCAN_ONLY, or ARCHITECTURE.

    Checks spec_data.get("job_kind") FIRST — SpecGate's deterministic
    classification takes priority.  Only falls back to keyword matching
    if job_kind is missing.
    """
    # === Check for pre-classified job_kind from SpecGate ===
    spec_job_kind = spec_data.get("job_kind", "")
    spec_job_kind_confidence = spec_data.get("job_kind_confidence", 0.0)
    spec_job_kind_reason = spec_data.get("job_kind_reason", "")

    if spec_job_kind and spec_job_kind != "unknown":
        logger.info(
            "[critical_pipeline] USING SPEC JOB_KIND: %s (confidence=%.2f, reason='%s')",
            spec_job_kind, spec_job_kind_confidence, spec_job_kind_reason,
        )

        _KIND_MAP = {
            "micro_execution": JobKind.MICRO_EXECUTION,
            "scan_only": JobKind.SCAN_ONLY,
            "repo_change": JobKind.ARCHITECTURE,
            "architecture": JobKind.ARCHITECTURE,
        }
        kind = _KIND_MAP.get(spec_job_kind)
        if kind:
            return kind

        logger.warning(
            "[critical_pipeline] Unknown job_kind '%s' — escalating to architecture",
            spec_job_kind,
        )
        return JobKind.ARCHITECTURE

    # === FALLBACK: SpecGate didn't classify ===
    logger.warning(
        "[critical_pipeline] job_kind not set by SpecGate (found='%s') — using fallback classification",
        spec_job_kind,
    )
    return _classify_by_heuristics(spec_data, message)


# =============================================================================
# Fallback Heuristic Classification
# =============================================================================

def _classify_by_heuristics(spec_data: Dict[str, Any], message: str) -> str:
    """Keyword / signal based fallback when SpecGate doesn't provide job_kind."""

    # Combine all text for analysis
    summary = (spec_data.get("summary", "") or "").lower()
    objective = (spec_data.get("objective", "") or "").lower()
    title = (spec_data.get("title", "") or "").lower()
    goal = (spec_data.get("goal", "") or "").lower()
    msg_lower = (message or "").lower()
    all_text = f"{summary} {objective} {title} {goal} {msg_lower}"

    # --- Sandbox / file path signals (strongest signal) ---
    has_sandbox_input = bool(spec_data.get("sandbox_input_path") or spec_data.get("input_file_path"))
    has_sandbox_output = bool(
        spec_data.get("sandbox_output_path")
        or spec_data.get("output_file_path")
        or spec_data.get("planned_output_path")
    )
    has_sandbox_reply = bool(spec_data.get("sandbox_generated_reply"))
    sandbox_discovery_used = spec_data.get("sandbox_discovery_used", False)

    # Check constraints for file paths
    for constraint in spec_data.get("constraints_from_repo", []):
        if isinstance(constraint, str):
            if "planned output path" in constraint.lower() or "reply.txt" in constraint.lower():
                has_sandbox_output = True

    # Check what_exists for sandbox input
    for item in spec_data.get("what_exists", []):
        if isinstance(item, str) and "sandbox input" in item.lower():
            has_sandbox_input = True

    # --- MICRO FAST PATH ---
    if sandbox_discovery_used and has_sandbox_input:
        logger.info(
            "[critical_pipeline] FALLBACK MICRO: sandbox_discovery_used=True, "
            "input=%s, output=%s",
            has_sandbox_input, has_sandbox_output,
        )
        return JobKind.MICRO_EXECUTION

    if has_sandbox_input and has_sandbox_output and has_sandbox_reply:
        logger.info(
            "[critical_pipeline] FALLBACK MICRO: Full sandbox resolution (input+output+reply)"
        )
        return JobKind.MICRO_EXECUTION

    # --- Keyword scoring ---
    micro_score, arch_score, scan_score = _score_keywords(all_text)

    # Boost micro score if resolved file paths present
    if has_sandbox_input and has_sandbox_output:
        micro_score += 5
    elif has_sandbox_input:
        micro_score += 3
    elif has_sandbox_output:
        micro_score += 2

    # Step count heuristic
    steps = spec_data.get("proposed_steps", spec_data.get("steps", []))
    if isinstance(steps, list):
        if len(steps) <= 5:
            micro_score += 1
        elif len(steps) > 10:
            arch_score += 2

    # Boost scan score if SpecGate resolved scan parameters
    if spec_data.get("scan_roots"):
        scan_score += 5
    if spec_data.get("scan_terms"):
        scan_score += 3

    logger.info(
        "[critical_pipeline] FALLBACK classification: micro=%d, arch=%d, scan=%d, "
        "sandbox_discovery=%s, paths=%s/%s, scan_roots=%s",
        micro_score, arch_score, scan_score,
        sandbox_discovery_used, has_sandbox_input, has_sandbox_output,
        bool(spec_data.get("scan_roots")),
    )

    # Decision
    if scan_score > micro_score and scan_score > arch_score:
        return JobKind.SCAN_ONLY
    if micro_score > arch_score:
        return JobKind.MICRO_EXECUTION
    if arch_score > micro_score:
        return JobKind.ARCHITECTURE
    if has_sandbox_input or has_sandbox_output:
        return JobKind.MICRO_EXECUTION  # tie-breaker
    return JobKind.ARCHITECTURE  # safe default


def _score_keywords(all_text: str):
    """Return (micro_score, arch_score, scan_score) from keyword matching."""

    MICRO_INDICATORS = [
        "read the", "read file", "open file", "find the file",
        "answer the question", "answer question", "reply to",
        "write reply", "write answer", "print answer",
        "summarize", "summarise", "extract", "copy",
        "find document", "find the document",
        "what does", "what is", "tell me",
        "underneath", "below", "same folder",
        "sandbox", "desktop", "test folder",
        "read-only", "reply (read-only)",
    ]

    ARCH_INDICATORS = [
        "design", "architect", "build system", "create system",
        "implement feature", "add feature", "new module",
        "refactor", "restructure", "redesign",
        "api endpoint", "database schema", "migration",
        "integration", "pipeline", "service",
        "authentication", "authorization",
        "full implementation", "complete implementation",
        "specgate", "spec gate", "overwatcher",
    ]

    SCAN_INDICATORS = [
        "scan the", "scan all", "scan for", "scan folder", "scan folders",
        "scan drive", "scan d:", "scan c:", "scan directory",
        "find all occurrences", "find all references", "find all files",
        "find all folders", "search for", "search the",
        "search entire", "search across", "search all",
        "list all", "list references", "list files",
        "enumerate", "report full paths", "report all",
        "where is", "locate all", "show me all",
        "references to", "mentions of", "occurrences of",
    ]

    micro = sum(1 for ind in MICRO_INDICATORS if ind in all_text)
    arch = sum(1 for ind in ARCH_INDICATORS if ind in all_text)
    scan = sum(1 for ind in SCAN_INDICATORS if ind in all_text)
    return micro, arch, scan
