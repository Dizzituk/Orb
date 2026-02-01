# FILE: app/pot_spec/grounded/refactor_formatter.py
"""
SpecGate v2.0 - Refactor Output Formatter

Generates dual-format output from RefactorPlan:
1. Human-readable markdown for SPoT output (context window)
2. Machine-readable JSON for grounding_data (pipeline consumption)

v2.0 (2026-02-01): Initial implementation
    - format_human_readable() for markdown summary
    - format_machine_readable() for JSON grounding_data
    - format_confirmation_message() for user-facing confirmation

Design Principles:
- Human output: Clear, scannable, shows what will happen
- Machine output: Full structured data for downstream stages
- No approval checkboxes - AI made decisions, user sees flags
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# BUILD VERIFICATION
# =============================================================================
REFACTOR_FORMATTER_BUILD_ID = "2026-02-01-v2.0-initial"
print(f"[REFACTOR_FORMATTER_LOADED] BUILD_ID={REFACTOR_FORMATTER_BUILD_ID}")

# =============================================================================
# IMPORTS
# =============================================================================

from .refactor_schemas import (
    MatchBucket,
    ChangeDecision,
    RiskLevel,
    RefactorPlan,
    BucketSummary,
    RefactorFlag,
)


# =============================================================================
# HUMAN-READABLE OUTPUT (Markdown)
# =============================================================================

def format_human_readable(plan: RefactorPlan) -> str:
    """
    Generate human-readable markdown summary from RefactorPlan.
    
    This goes into the SPoT output and is what the user sees in the context window.
    Designed to be scannable with clear visual hierarchy.
    """
    lines = []
    
    # Header
    lines.append(f"## Refactor Analysis: {plan.search_term} → {plan.replace_term}")
    lines.append("")
    
    # Summary box
    lines.append("### Summary")
    lines.append(f"- **Total Files:** {plan.total_files}")
    lines.append(f"- **Total Occurrences:** {plan.total_occurrences}")
    lines.append(f"- **Will Change:** {plan.change_count} occurrences across {len(plan.files_to_change)} files")
    lines.append(f"- **Will Skip:** {plan.skip_count} occurrences across {len(plan.files_to_skip)} files")
    if plan.flag_count > 0:
        lines.append(f"- **Flagged for Review:** {plan.flag_count} occurrences across {len(plan.files_to_flag)} files")
    lines.append("")
    
    # Breakdown by category
    lines.append("### Breakdown by Category")
    lines.append("")
    
    # WILL CHANGE section
    change_buckets = [
        (bucket_name, summary) 
        for bucket_name, summary in plan.bucket_summaries.items()
        if summary.decision == ChangeDecision.CHANGE
    ]
    if change_buckets:
        lines.append("#### ✅ WILL CHANGE (Safe)")
        lines.append("")
        lines.append("| Category | Files | Occurrences | Risk |")
        lines.append("|----------|-------|-------------|------|")
        for bucket_name, summary in sorted(change_buckets, key=lambda x: x[1].total_count, reverse=True):
            display_name = _format_bucket_name(summary.bucket)
            risk_emoji = _risk_emoji(summary.risk_level)
            lines.append(f"| {display_name} | {len(summary.sample_files)} | {summary.total_count} | {risk_emoji} {summary.risk_level.value.upper()} |")
        lines.append("")
    
    # WILL SKIP section
    skip_buckets = [
        (bucket_name, summary)
        for bucket_name, summary in plan.bucket_summaries.items()
        if summary.decision == ChangeDecision.SKIP
    ]
    if skip_buckets:
        lines.append("#### ⏭️ WILL SKIP (Too Risky)")
        lines.append("")
        lines.append("| Category | Files | Occurrences | Reason |")
        lines.append("|----------|-------|-------------|--------|")
        for bucket_name, summary in sorted(skip_buckets, key=lambda x: x[1].total_count, reverse=True):
            display_name = _format_bucket_name(summary.bucket)
            reason = _truncate(summary.reasoning, 50)
            lines.append(f"| {display_name} | {len(summary.sample_files)} | {summary.total_count} | {reason} |")
        lines.append("")
    
    # FLAGGED section
    flag_buckets = [
        (bucket_name, summary)
        for bucket_name, summary in plan.bucket_summaries.items()
        if summary.decision == ChangeDecision.FLAG
    ]
    if flag_buckets:
        lines.append("#### 🚩 FLAGGED (Review Recommended)")
        lines.append("")
        lines.append("| Category | Files | Occurrences | Reason |")
        lines.append("|----------|-------|-------------|--------|")
        for bucket_name, summary in sorted(flag_buckets, key=lambda x: x[1].total_count, reverse=True):
            display_name = _format_bucket_name(summary.bucket)
            reason = _truncate(summary.reasoning, 50)
            lines.append(f"| {display_name} | {len(summary.sample_files)} | {summary.total_count} | {reason} |")
        lines.append("")
    
    # Flags section (informational alerts)
    if plan.flags:
        lines.append("### 🚩 Flags (Informational)")
        lines.append("")
        for flag in plan.flags:
            severity_icon = {"INFO": "ℹ️", "WARNING": "⚠️", "CAUTION": "🔶"}.get(flag.severity, "📌")
            lines.append(f"- {severity_icon} **{flag.flag_type}:** {flag.message}")
            if flag.recommendation:
                lines.append(f"  - *Recommendation:* {flag.recommendation}")
        lines.append("")
    
    # Execution phases
    if plan.execution_phases:
        lines.append("### 📋 Recommended Execution Order")
        lines.append("")
        for phase in plan.execution_phases:
            risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠"}.get(phase.get("risk", "medium"), "🔵")
            lines.append(f"**Phase {phase['phase']}:** {phase['name']} {risk_emoji}")
            lines.append(f"  - {phase['description']}")
            lines.append(f"  - {phase['file_count']} files, {phase['occurrence_count']} changes")
        lines.append("")
    
    # Exclusions
    if plan.exclusions:
        lines.append("### 🚫 Explicitly Excluded")
        lines.append("")
        for exc in plan.exclusions:
            lines.append(f"- **{_format_bucket_name_str(exc['category'])}** ({exc['count']} occurrences): {exc['reason']}")
        lines.append("")
    
    # Sample files being changed (top 20)
    if plan.files_to_change:
        lines.append("### 📁 Sample Files to Modify")
        lines.append("")
        sample_files = plan.files_to_change[:20]
        for f in sample_files:
            # Show relative path if possible
            display_path = _shorten_path(f)
            lines.append(f"- `{display_path}`")
        if len(plan.files_to_change) > 20:
            lines.append(f"- *... and {len(plan.files_to_change) - 20} more files*")
        lines.append("")
    
    # Classification metadata
    lines.append("---")
    lines.append(f"*Classification: {plan.classification_model} • {plan.classification_duration_ms}ms*")
    
    return "\n".join(lines)


def _format_bucket_name(bucket: MatchBucket) -> str:
    """Format bucket enum to human-readable name."""
    name_map = {
        MatchBucket.CODE_IDENTIFIER: "Code Identifiers",
        MatchBucket.IMPORT_PATH: "Import Paths",
        MatchBucket.MODULE_PACKAGE: "Module/Package Names",
        MatchBucket.ENV_VAR_KEY: "Environment Variables",
        MatchBucket.CONFIG_KEY: "Config Keys",
        MatchBucket.API_ROUTE: "API Routes",
        MatchBucket.FILE_FOLDER_NAME: "File/Folder Names",
        MatchBucket.DATABASE_ARTIFACT: "Database Files",
        MatchBucket.HISTORICAL_DATA: "Historical Data/Logs",
        MatchBucket.DOCUMENTATION: "Documentation",
        MatchBucket.UI_LABEL: "UI Labels",
        MatchBucket.TEST_ASSERTION: "Test Assertions",
        MatchBucket.UNKNOWN: "Uncategorized",
    }
    return name_map.get(bucket, bucket.value.replace("_", " ").title())


def _format_bucket_name_str(bucket_str: str) -> str:
    """Format bucket string to human-readable name."""
    try:
        bucket = MatchBucket(bucket_str)
        return _format_bucket_name(bucket)
    except ValueError:
        return bucket_str.replace("_", " ").title()


def _risk_emoji(risk: RiskLevel) -> str:
    """Get emoji for risk level."""
    return {
        RiskLevel.LOW: "🟢",
        RiskLevel.MEDIUM: "🟡",
        RiskLevel.HIGH: "🟠",
        RiskLevel.CRITICAL: "🔴",
    }.get(risk, "⚪")


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def _shorten_path(path: str) -> str:
    """Shorten path for display."""
    # Remove common prefixes
    for prefix in ["D:\\Orb\\", "D:\\orb-desktop\\", "D:\\", "C:\\"]:
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


# =============================================================================
# MACHINE-READABLE OUTPUT (JSON for grounding_data)
# =============================================================================

def format_machine_readable(plan: RefactorPlan) -> Dict[str, Any]:
    """
    Generate machine-readable JSON from RefactorPlan.
    
    This goes into grounding_data and is consumed by Critical Pipeline
    and Implementer stages. Contains full structured data.
    """
    return {
        # Core refactor metadata
        "refactor_summary": {
            "search_term": plan.search_term,
            "replace_term": plan.replace_term,
            "total_files": plan.total_files,
            "total_occurrences": plan.total_occurrences,
            "change_count": plan.change_count,
            "skip_count": plan.skip_count,
            "flag_count": plan.flag_count,
        },
        
        # All classified matches (full detail)
        "classified_matches": [m.to_dict() for m in plan.classified_matches],
        
        # Bucket summaries (aggregated view)
        "bucket_summaries": {k: v.to_dict() for k, v in plan.bucket_summaries.items()},
        
        # File lists by decision
        "files_to_change": plan.files_to_change,
        "files_to_skip": plan.files_to_skip,
        "files_to_flag": plan.files_to_flag,
        
        # Flags for information
        "flags": [f.to_dict() for f in plan.flags],
        
        # Execution plan
        "execution_phases": plan.execution_phases,
        
        # Exclusions
        "exclusions": plan.exclusions,
        
        # Classification metadata
        "classification": {
            "model": plan.classification_model,
            "duration_ms": plan.classification_duration_ms,
            "timestamp": plan.classification_timestamp.isoformat(),
        },
        
        # Validation
        "is_valid": plan.is_valid,
        "validation_errors": plan.validation_errors,
    }


# =============================================================================
# CONFIRMATION MESSAGE
# =============================================================================

def format_confirmation_message(plan: RefactorPlan) -> str:
    """
    Generate a confirmation message for the user.
    
    This is shown when SpecGate asks the user to confirm the refactor.
    Unlike the old system, this shows intelligent analysis, not raw data.
    """
    lines = []
    
    # Header
    lines.append(f"## 🔄 Refactor Plan: `{plan.search_term}` → `{plan.replace_term}`")
    lines.append("")
    
    # Quick summary
    lines.append(f"**Scope:** {plan.total_files} files, {plan.total_occurrences} occurrences")
    lines.append("")
    
    # What will happen
    lines.append("### What Will Happen")
    lines.append("")
    
    if plan.change_count > 0:
        lines.append(f"✅ **{plan.change_count} occurrences** will be changed across {len(plan.files_to_change)} files")
        # Show bucket breakdown
        change_buckets = [(k, v) for k, v in plan.bucket_summaries.items() if v.decision == ChangeDecision.CHANGE]
        for bucket_name, summary in change_buckets[:5]:
            lines.append(f"   - {_format_bucket_name(summary.bucket)}: {summary.total_count}")
    
    if plan.skip_count > 0:
        lines.append("")
        lines.append(f"⏭️ **{plan.skip_count} occurrences** will be SKIPPED (left unchanged)")
        skip_buckets = [(k, v) for k, v in plan.bucket_summaries.items() if v.decision == ChangeDecision.SKIP]
        for bucket_name, summary in skip_buckets[:3]:
            lines.append(f"   - {_format_bucket_name(summary.bucket)}: {summary.total_count} ({summary.reasoning})")
    
    if plan.flag_count > 0:
        lines.append("")
        lines.append(f"🚩 **{plan.flag_count} occurrences** flagged for your awareness")
    
    # Show important flags
    if plan.flags:
        lines.append("")
        lines.append("### ⚠️ Important Notes")
        for flag in plan.flags:
            if flag.severity in ("WARNING", "CAUTION"):
                lines.append(f"- {flag.message}")
                if flag.recommendation:
                    lines.append(f"  *{flag.recommendation}*")
    
    # Confirmation prompt
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**Proceed with this refactor?** (yes/no)")
    lines.append("")
    lines.append("*You can stop the workflow at any stage by not continuing.*")
    
    return "\n".join(lines)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "format_human_readable",
    "format_machine_readable",
    "format_confirmation_message",
]
