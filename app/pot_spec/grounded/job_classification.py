# FILE: app/pot_spec/grounded/job_classification.py
"""
Job Kind Classifier (v1.9, v1.10, v1.18, v1.19)

Deterministic job classification - NO LLM calls.
Classifies jobs as micro_execution, repo_change, architecture, scan_only, or unknown.

Version Notes:
-------------
v1.9 (2026-01): Initial deterministic classifier
v1.10 (2026-01): Added greenfield_build detection
v1.18 (2026-01): Added scan_only job type
v1.19 (2026-01-30): CRITICAL FIX - SCAN_ONLY with write targets
    - SCAN_ONLY classification now checks for write targets before returning
    - If sandbox_output_path exists OR sandbox_output_mode != CHAT_ONLY, job is NOT scan_only
    - Fixes bug where "find all files + create reply" was misclassified as SCAN_ONLY
    - Jobs with CREATE targets now correctly fall through to micro_execution
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from .domain_detection import detect_domains, DOMAIN_KEYWORDS
from .spec_models import GroundedPOTSpec

logger = logging.getLogger(__name__)


# =============================================================================
# EVIDENCE CONFIG BY JOB SIZE
# =============================================================================

EVIDENCE_CONFIG = {
    "tiny": {
        "include_arch_map": False,
        "include_codebase_report": False,
    },
    "normal": {
        "include_arch_map": True,
        "include_codebase_report": True,
        "arch_map_max_lines": 300,
        "codebase_report_max_lines": 200,
    },
    "critical": {
        "include_arch_map": True,
        "include_codebase_report": True,
        "arch_map_max_lines": 500,
        "codebase_report_max_lines": 300,
    },
}


# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================

def classify_job_kind(
    spec: GroundedPOTSpec,
    intent: Dict[str, Any],
) -> Tuple[str, float, str]:
    """
    v1.9: Classify job as micro_execution, repo_change, architecture, scan_only, or unknown.
    
    This is DETERMINISTIC - NO LLM calls. Critical Pipeline MUST obey this.
    
    MICRO_EXECUTION (seconds):
    - sandbox_discovery_used=True AND sandbox_input_path exists
    - Simple read/write/answer with ≤5 steps
    
    REPO_CHANGE (minutes):
    - Edit existing code files
    - Add new files to existing module
    - Bug fixes, updates
    
    ARCHITECTURE (2-5 minutes):
    - Design new subsystem/module
    - Multi-module refactoring
    - v1.10: Greenfield builds
    
    SCAN_ONLY (seconds-minutes):
    - Read-only filesystem scans
    - No writes, no sandbox_input_path needed
    
    UNKNOWN:
    - Too ambiguous → escalate to architecture
    
    Returns:
        (job_kind, confidence, reason)
    """
    raw_text = (intent.get("raw_text", "") or "").lower()
    goal_text = (spec.goal or "").lower()
    all_text = f"{raw_text} {goal_text}"
    
    detected_domains = detect_domains(all_text)
    
    # =========================================================================
    # v1.19 RULE 0a: SCAN_ONLY jobs (with write target check)
    # =========================================================================
    
    if "scan_only" in detected_domains:
        # v1.19 FIX: SCAN_ONLY must NOT have any write targets
        # If there's an output path or non-CHAT_ONLY mode, this is NOT a scan_only job
        has_write_target = False
        write_target_reason = ""
        
        if spec.sandbox_output_path:
            has_write_target = True
            write_target_reason = f"sandbox_output_path={spec.sandbox_output_path}"
        elif spec.sandbox_output_mode and spec.sandbox_output_mode.lower() not in ("", "chat_only"):
            has_write_target = True
            write_target_reason = f"sandbox_output_mode={spec.sandbox_output_mode}"
        elif spec.is_multi_target_read and spec.sandbox_output_mode and spec.sandbox_output_mode.lower() != "chat_only":
            has_write_target = True
            write_target_reason = f"multi_target_read with output_mode={spec.sandbox_output_mode}"
        
        if has_write_target:
            logger.info(
                "[job_classification] v1.19 scan_only domain detected BUT has write target (%s) - NOT a scan_only job",
                write_target_reason
            )
            # Fall through to other classification rules
        else:
            matched_keywords = [kw for kw in DOMAIN_KEYWORDS.get("scan_only", []) if kw in all_text][:3]
            reason = f"scan_only domain detected (READ_ONLY scan, no write targets): matched {matched_keywords}"
            logger.info("[job_classification] v1.19 classify_job_kind: SCAN_ONLY - %s", reason)
            return ("scan_only", 0.92, reason)
    
    # =========================================================================
    # v1.10 RULE 0: Greenfield builds are ALWAYS architecture jobs
    # =========================================================================
    
    if "greenfield_build" in detected_domains:
        matched_keywords = [kw for kw in DOMAIN_KEYWORDS.get("greenfield_build", []) if kw in all_text][:3]
        reason = f"greenfield_build domain detected (CREATE_NEW): matched {matched_keywords}"
        logger.info("[job_classification] v1.10 classify_job_kind: ARCHITECTURE (greenfield) - %s", reason)
        return ("architecture", 0.90, reason)
    
    # =========================================================================
    # RULE 1: Sandbox discovery with resolved paths → MICRO_EXECUTION
    # =========================================================================
    
    if spec.sandbox_discovery_used and spec.sandbox_input_path:
        reason = (
            f"sandbox_discovery_used=True with input={spec.sandbox_input_path}, "
            f"output={spec.sandbox_output_path or 'pending'}"
        )
        logger.info("[job_classification] v1.9 classify_job_kind: MICRO_EXECUTION - %s", reason)
        return ("micro_execution", 0.95, reason)
    
    # =========================================================================
    # RULE 2: Check step count
    # =========================================================================
    
    step_count = len(spec.proposed_steps)
    
    # =========================================================================
    # RULE 3: Keyword-based classification
    # =========================================================================
    
    micro_keywords = [
        "read the", "read file", "find the file", "find file",
        "answer the question", "answer question", "reply to",
        "write reply", "write answer", "print answer",
        "summarize", "summarise", "extract", "copy",
        "find document", "what does", "what is", "tell me",
        "sandbox", "desktop", "test folder", "message file",
    ]
    
    repo_change_keywords = [
        "edit", "update", "fix bug", "fix the", "modify",
        "change the", "add to", "append", "patch",
        "update file", "edit file", "change file",
    ]
    
    arch_keywords = [
        "design", "architect", "build system", "create system",
        "implement feature", "add feature", "new module",
        "refactor", "restructure", "redesign",
        "api endpoint", "database schema", "migration",
        "integration", "pipeline", "service layer",
        "authentication", "authorization",
        "full implementation", "complete implementation",
        "specgate", "spec gate", "overwatcher",
        "new subsystem", "multi-module", "architecture",
        "build me", "make me", "create a", "build a", "make a",
        "new app", "new project", "new game", "from scratch",
    ]
    
    micro_score = sum(1 for kw in micro_keywords if kw in all_text)
    repo_score = sum(1 for kw in repo_change_keywords if kw in all_text)
    arch_score = sum(1 for kw in arch_keywords if kw in all_text)
    
    if spec.sandbox_input_path or spec.sandbox_output_path:
        micro_score += 5
    if spec.sandbox_generated_reply:
        micro_score += 3
    
    if step_count > 10:
        arch_score += 3
    elif step_count > 5:
        arch_score += 1
        repo_score += 1
    elif step_count <= 3:
        micro_score += 2
    
    logger.info(
        "[job_classification] v1.9 classify_job_kind scores: micro=%d, repo=%d, arch=%d, steps=%d",
        micro_score, repo_score, arch_score, step_count
    )
    
    # =========================================================================
    # DECISION LOGIC
    # =========================================================================
    
    total_score = micro_score + repo_score + arch_score
    
    if total_score == 0:
        return ("unknown", 0.3, "No classification keywords matched - escalating to architecture")
    
    if micro_score >= repo_score and micro_score >= arch_score:
        confidence = min(0.9, 0.5 + (micro_score / 10))
        reason = f"micro keywords ({micro_score}) >= repo ({repo_score}) and arch ({arch_score})"
        return ("micro_execution", confidence, reason)
    
    if repo_score > micro_score and repo_score >= arch_score:
        confidence = min(0.85, 0.5 + (repo_score / 10))
        reason = f"repo_change keywords ({repo_score}) > micro ({micro_score}) and >= arch ({arch_score})"
        return ("repo_change", confidence, reason)
    
    if arch_score > micro_score and arch_score > repo_score:
        confidence = min(0.9, 0.5 + (arch_score / 10))
        reason = f"architecture keywords ({arch_score}) > micro ({micro_score}) and repo ({repo_score})"
        return ("architecture", confidence, reason)
    
    return ("unknown", 0.4, "Ambiguous classification - escalating to architecture")


def classify_job_size(weaver_output: str) -> str:
    """
    Classify job size for evidence loading decisions.
    
    Returns: "tiny", "normal", or "critical"
    """
    if not weaver_output:
        return "normal"
    
    text = weaver_output.lower()
    word_count = len(text.split())
    
    tiny_indicators = [
        "reply to", "read the message", "write a reply",
        "find the file", "simple test", "message file",
        "respond to", "answer the"
    ]
    if any(w in text for w in tiny_indicators) and word_count < 100:
        return "tiny"
    
    critical_words = [
        "refactor", "security", "encrypt", "migration",
        "schema", "all files", "entire codebase", "complete rewrite",
        "breaking change", "backwards compat"
    ]
    if any(w in text for w in critical_words) or word_count > 500:
        return "critical"
    
    return "normal"
