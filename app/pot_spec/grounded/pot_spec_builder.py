# FILE: app/pot_spec/grounded/pot_spec_builder.py
"""
SpecGate v3.0 - POT Spec Builder (Planner-First)

Deterministic markdown builder for POT (Proof Of Task) specifications.
Takes a RefactorPlanV3 + constraints and outputs complete, grounded specs.

Design Principles:
- NO LLM calls - pure deterministic formatting
- NO "MIGRATION REQUIRED" unless allows_migration=True
- Outputs all sections: Goal, Scope, Mapping, CHANGE/PRESERVE/FLAG tables, Acceptance, Assumptions
- Testable: same input always produces same output

v3.0 (2026-02-01): Initial implementation for planner-first architecture

Used by:
- spec_runner.py for final POT spec generation
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Build ID for verification
POT_SPEC_BUILDER_BUILD_ID = "2026-02-01-v3.0-planner-first"
print(f"[POT_SPEC_BUILDER_LOADED] BUILD_ID={POT_SPEC_BUILDER_BUILD_ID}")

from .refactor_schemas import (
    ChangeDecision,
    RiskLevel,
    RiskClass,
    ReasonCode,
    ClassifiedMatchV3,
    RefactorPlanV3,
    BlockingIssue,
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Maximum matches to show in each table (avoid massive markdown)
MAX_CHANGE_ROWS = 50
MAX_PRESERVE_ROWS = 20
MAX_FLAG_ROWS = 20


# =============================================================================
# CORE BUILDER
# =============================================================================

def build_pot_spec_markdown(
    plan: RefactorPlanV3,
    constraints: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build deterministic POT spec markdown from a RefactorPlanV3.
    
    Args:
        plan: The complete refactor plan with all classified matches
        constraints: Optional constraints from Weaver (text_only, no_renames, etc.)
    
    Returns:
        Complete POT spec markdown string
    
    HARD RULES:
    - Never output "MIGRATION REQUIRED" unless allows_migration=True
    - All tables are deterministic (sorted by file, then line)
    - No LLM calls - pure formatting
    """
    constraints = constraints or {}
    
    # v3.0 TRACE: Confirm deterministic builder is being used (not LLM Contract v1)
    logger.info(
        "[pot_spec_builder] v3.0 BUILDING POT SPEC: search=%s, replace=%s, files=%d, matches=%d",
        plan.search_term, plan.replace_term, plan.total_files, len(plan.classified_matches or [])
    )
    print(f"[pot_spec_builder] v3.0 BUILDING POT SPEC (deterministic, no LLM): {plan.search_term} -> {plan.replace_term}")
    
    # Validate no migration wording unless allowed
    _validate_no_migration_unless_allowed(plan, constraints)
    
    sections = []
    
    # Header
    sections.append(_build_header(plan))
    
    # Goal section
    sections.append(_build_goal_section(plan, constraints))
    
    # Scope section
    sections.append(_build_scope_section(plan, constraints))
    
    # Rename mapping section
    sections.append(_build_mapping_section(plan))
    
    # CHANGE table
    change_section = _build_change_table(plan)
    if change_section:
        sections.append(change_section)
    
    # PRESERVE table
    preserve_section = _build_preserve_table(plan)
    if preserve_section:
        sections.append(preserve_section)
    
    # FLAG table
    flag_section = _build_flag_table(plan, constraints)
    if flag_section:
        sections.append(flag_section)
    
    # Acceptance checks
    sections.append(_build_acceptance_section(plan))
    
    # Assumptions section
    sections.append(_build_assumptions_section(plan, constraints))
    
    # Risk summary (if any high/critical)
    risk_section = _build_risk_summary(plan)
    if risk_section:
        sections.append(risk_section)
    
    return "\n\n".join(s for s in sections if s)


def _validate_no_migration_unless_allowed(
    plan: RefactorPlanV3,
    constraints: Dict[str, Any],
) -> None:
    """
    Enforce: No migration wording unless job includes migration.
    
    If constraints include "no file/folder renames" or "text-only":
    - Any path literal → FLAG + PRESERVE (never "migration required")
    """
    allows_migration = constraints.get("allows_migration", False)
    text_only = plan.text_only or constraints.get("text_only", False)
    no_renames = plan.no_renames or constraints.get("no_renames", False)
    
    if allows_migration:
        return  # Migration wording is allowed
    
    # Check for any migration-implying reason codes and fix them
    for match in plan.classified_matches:
        if text_only or no_renames:
            # Force path literals to FLAG, not migration
            if match.reason_code == ReasonCode.PATH_LITERAL_WORKS_NOW:
                match.decision = ChangeDecision.FLAG
                match.reason_text = "Path literal - works now, migration out of scope"
                # NEVER say "migration required"


# =============================================================================
# SECTION BUILDERS
# =============================================================================

def _build_header(plan: RefactorPlanV3) -> str:
    """Build the spec header."""
    timestamp = plan.classification_timestamp.strftime("%Y-%m-%d %H:%M UTC")
    return f"""# SPoT Spec — Text-only UI rebrand: {plan.search_term} → {plan.replace_term}

**Generated**: {timestamp}  
**Model**: {plan.classification_model or 'N/A'}  
**Status**: {'⚠️ BLOCKED' if plan.has_unresolved_unknowns() else '✅ READY'}"""


def _build_goal_section(plan: RefactorPlanV3, constraints: Dict[str, Any]) -> str:
    """Build the Goal section."""
    goal_text = constraints.get("goal", "")
    if not goal_text:
        goal_text = f"Replace user-facing branding text \"{plan.search_term}\" with \"{plan.replace_term}\" in the codebase."
    
    return f"""## Goal

{goal_text}"""


def _build_scope_section(plan: RefactorPlanV3, constraints: Dict[str, Any]) -> str:
    """Build the Scope section."""
    # Get roots from plan
    roots = constraints.get("roots", [])
    if not roots and plan.files_to_change:
        # Infer roots from file paths
        seen_roots = set()
        for f in plan.files_to_change[:5]:
            parts = f.replace("\\", "/").split("/")
            if len(parts) >= 2:
                root = "/".join(parts[:2])
                seen_roots.add(root)
        roots = list(seen_roots)[:3]
    
    roots_str = ", ".join(roots) if roots else "(auto-detected)"
    
    # Build exclusions list
    exclusions = []
    if plan.text_only:
        exclusions.append("No file/folder renames")
    if plan.no_renames:
        exclusions.append("No image/icon/logo changes")
    if not plan.allows_migration:
        exclusions.append("No data migrations")
    
    exclusions_str = ", ".join(exclusions) if exclusions else "None specified"
    
    return f"""## Scope

- **Root(s)**: {roots_str}
- **Operation**: File content edits only
- **Exclusions**: {exclusions_str}
- **Files affected**: {plan.total_files}
- **Total occurrences**: {plan.total_occurrences}"""


def _build_mapping_section(plan: RefactorPlanV3) -> str:
    """Build the Rename Mapping section with case variants."""
    search = plan.search_term
    replace = plan.replace_term
    
    # Generate case variants
    variants = []
    
    # UPPER
    if search.upper() != search:
        variants.append(f"- `\\b{search.upper()}\\b` → `{replace.upper()}`")
    else:
        variants.append(f"- `\\b{search}\\b` → `{replace}`")
    
    # Title case
    if search.title() != search.upper():
        variants.append(f"- `\\b{search.title()}\\b` → `{replace.title()}`")
    
    # lowercase
    if search.lower() != search.title():
        variants.append(f"- `\\b{search.lower()}\\b` → `{replace.lower()}`")
    
    variants_str = "\n".join(variants)
    
    return f"""## Rename Mapping (Case-Preserving, Word-Boundary)

{variants_str}"""


def _build_change_table(plan: RefactorPlanV3) -> str:
    """Build the CHANGE table for matches that will be modified."""
    change_matches = [m for m in plan.classified_matches if m.decision == ChangeDecision.CHANGE]
    
    if not change_matches:
        return ""
    
    # Sort by file then line
    change_matches.sort(key=lambda m: (m.file_path.lower(), m.line_number))
    
    # Truncate if too many
    truncated = len(change_matches) > MAX_CHANGE_ROWS
    display_matches = change_matches[:MAX_CHANGE_ROWS]
    
    rows = []
    for m in display_matches:
        file_name = os.path.basename(m.file_path)
        # Escape pipe characters in content
        snippet = m.line_content[:60].replace("|", "\\|").strip()
        if len(m.line_content) > 60:
            snippet += "..."
        reason = m.reason_text[:50] if m.reason_text else _default_reason(m.reason_code)
        rows.append(f"| {file_name} | {m.line_number} | `{snippet}` | {reason} |")
    
    rows_str = "\n".join(rows)
    truncate_note = f"\n\n*...and {len(change_matches) - MAX_CHANGE_ROWS} more*" if truncated else ""
    
    return f"""## ✅ CHANGE ({len(change_matches)} matches)

Files that will be modified:

| File | Line | Match | Reasoning |
|------|------|-------|-----------|
{rows_str}{truncate_note}"""


def _build_preserve_table(plan: RefactorPlanV3) -> str:
    """Build the PRESERVE (SKIP) table for matches left unchanged."""
    skip_matches = [m for m in plan.classified_matches if m.decision == ChangeDecision.SKIP]
    
    if not skip_matches:
        return ""
    
    # Sort by file then line
    skip_matches.sort(key=lambda m: (m.file_path.lower(), m.line_number))
    
    # Truncate if too many
    truncated = len(skip_matches) > MAX_PRESERVE_ROWS
    display_matches = skip_matches[:MAX_PRESERVE_ROWS]
    
    rows = []
    for m in display_matches:
        file_name = os.path.basename(m.file_path)
        snippet = m.match_text[:40].replace("|", "\\|") if m.match_text else m.line_content[:40].replace("|", "\\|")
        reason = m.reason_text[:60] if m.reason_text else _default_reason(m.reason_code)
        rows.append(f"| {file_name} | {m.line_number} | `{snippet}` | SKIP | {reason} |")
    
    rows_str = "\n".join(rows)
    truncate_note = f"\n\n*...and {len(skip_matches) - MAX_PRESERVE_ROWS} more*" if truncated else ""
    
    return f"""## 🚫 PRESERVE ({len(skip_matches)} matches)

Files left unchanged (operational invariants):

| File | Line | Match | Decision | Reasoning |
|------|------|-------|----------|-----------|
{rows_str}{truncate_note}"""


def _build_flag_table(plan: RefactorPlanV3, constraints: Dict[str, Any]) -> str:
    """Build the FLAG table for informational notices."""
    flag_matches = [m for m in plan.classified_matches if m.decision == ChangeDecision.FLAG]
    
    if not flag_matches:
        return ""
    
    # Sort by file then line
    flag_matches.sort(key=lambda m: (m.file_path.lower(), m.line_number))
    
    # Truncate if too many
    truncated = len(flag_matches) > MAX_FLAG_ROWS
    display_matches = flag_matches[:MAX_FLAG_ROWS]
    
    rows = []
    for m in display_matches:
        file_name = os.path.basename(m.file_path)
        snippet = m.match_text[:40].replace("|", "\\|") if m.match_text else m.line_content[:40].replace("|", "\\|")
        # CRITICAL: Never say "migration required" unless allowed
        impact = m.impact_note or m.reason_text or "Review recommended"
        if not constraints.get("allows_migration", False):
            impact = impact.replace("migration required", "out of scope")
            impact = impact.replace("Migration required", "Out of scope")
        rows.append(f"| {file_name} | {m.line_number} | `{snippet}` | {impact[:60]} |")
    
    rows_str = "\n".join(rows)
    truncate_note = f"\n\n*...and {len(flag_matches) - MAX_FLAG_ROWS} more*" if truncated else ""
    
    # Use correct header - NOT "Migration Required"
    return f"""## ⚠️ FLAG ({len(flag_matches)} matches)

Informational notices (proceed, but user should know):

| File | Line | Match | Impact |
|------|------|-------|--------|
{rows_str}{truncate_note}"""


def _build_acceptance_section(plan: RefactorPlanV3) -> str:
    """Build the Acceptance Checks section."""
    checks = []
    
    # Standard checks
    checks.append("- [ ] Application boots successfully")
    checks.append(f"- [ ] UI shows \"{plan.replace_term}\" in title/header/menus")
    
    # Add check for preserved items if any
    if plan.skip_count > 0:
        checks.append("- [ ] Auth/session still works (storage keys unchanged)")
    
    # Add check for flagged items if any
    if plan.flag_count > 0:
        checks.append("- [ ] Flagged items reviewed and acceptable")
    
    checks_str = "\n".join(checks)
    
    return f"""## Acceptance Checks

{checks_str}"""


def _build_assumptions_section(plan: RefactorPlanV3, constraints: Dict[str, Any]) -> str:
    """Build the Assumptions Applied section."""
    assumptions = []
    
    # Standard assumptions based on what was preserved
    has_token_preserves = any(
        m.reason_code == ReasonCode.TOKEN_PREFIX_VALIDATION or
        m.reason_code == ReasonCode.STORAGE_KEY_IN_USE
        for m in plan.classified_matches if m.decision == ChangeDecision.SKIP
    )
    if has_token_preserves:
        assumptions.append(f"- **Token prefixes** (`{plan.search_term.lower()}_...`): Preserved by default")
    
    has_env_preserves = any(
        m.reason_code == ReasonCode.ENV_VAR_EXTERNAL_DEPENDENCY
        for m in plan.classified_matches if m.decision == ChangeDecision.SKIP
    )
    if has_env_preserves:
        assumptions.append(f"- **Environment variables** (`{plan.search_term.upper()}_*`): Preserved by default")
    
    has_path_flags = any(
        m.reason_code == ReasonCode.PATH_LITERAL_WORKS_NOW
        for m in plan.classified_matches if m.decision == ChangeDecision.FLAG
    )
    if has_path_flags:
        assumptions.append("- **Path literals**: Flagged but preserved (separate migration if needed)")
    
    has_db_preserves = any(
        m.reason_code == ReasonCode.DATABASE_ARTIFACT or
        m.reason_code == ReasonCode.HISTORICAL_DATA_NO_VALUE
        for m in plan.classified_matches if m.decision == ChangeDecision.SKIP
    )
    if has_db_preserves:
        assumptions.append("- **Database artifacts**: Excluded (historical data, no value in changing)")
    
    # Add constraints-based assumptions
    if plan.text_only:
        assumptions.append("- **Text-only mode**: No file/folder renames")
    if plan.questions_none:
        assumptions.append("- **Non-interactive mode**: All decisions made automatically")
    
    if not assumptions:
        assumptions.append("- No special assumptions applied")
    
    assumptions_str = "\n".join(assumptions)
    
    return f"""## Assumptions Applied (No Questions Asked)

{assumptions_str}"""


def _build_risk_summary(plan: RefactorPlanV3) -> str:
    """Build risk summary if there are blocking issues or high risk."""
    if plan.computed_risk_class in (RiskClass.LOW, RiskClass.MEDIUM):
        if not plan.expansion_failures and not plan.risk_factors:
            return ""  # No risk summary needed
    
    sections = []
    
    # Risk class
    risk_emoji = {
        RiskClass.LOW: "🟢",
        RiskClass.MEDIUM: "🟡",
        RiskClass.HIGH: "🟠",
        RiskClass.CRITICAL: "🔴",
    }
    sections.append(f"**Risk Class**: {risk_emoji.get(plan.computed_risk_class, '⚪')} {plan.computed_risk_class.value.upper()}")
    
    # Risk factors
    if plan.risk_factors:
        factors = "\n".join(f"- {f}" for f in plan.risk_factors)
        sections.append(f"\n**Risk Factors**:\n{factors}")
    
    # Blocking issues
    if plan.expansion_failures:
        issues = "\n".join(f"- {str(i)}" for i in plan.expansion_failures[:5])
        sections.append(f"\n**Blocking Issues** ({len(plan.expansion_failures)}):\n{issues}")
    
    content = "\n".join(sections)
    
    return f"""## Risk Summary

{content}"""


def _default_reason(reason_code: ReasonCode) -> str:
    """Get default human-readable reason text for a reason code."""
    defaults = {
        ReasonCode.UI_TEXT_VISIBLE_TO_USER: "User-visible UI text",
        ReasonCode.DOCUMENTATION_STRING: "Documentation string",
        ReasonCode.COMMENT_TEXT: "Comment text",
        ReasonCode.TEST_ASSERTION_VALUE: "Test assertion value",
        ReasonCode.INTERNAL_IDENTIFIER: "Internal identifier",
        ReasonCode.STORAGE_KEY_IN_USE: "Storage key in use",
        ReasonCode.ENV_VAR_EXTERNAL_DEPENDENCY: "Env var with external dependency",
        ReasonCode.TOKEN_PREFIX_VALIDATION: "Token prefix used in validation",
        ReasonCode.DATABASE_ARTIFACT: "Database artifact",
        ReasonCode.HISTORICAL_DATA_NO_VALUE: "Historical data - no value in changing",
        ReasonCode.API_KEY_OR_SECRET: "API key or secret",
        ReasonCode.PATH_LITERAL_WORKS_NOW: "Path literal - works now",
        ReasonCode.IMPORT_PATH_CASCADE: "Import path - cascade updates needed",
        ReasonCode.API_ROUTE_EXTERNAL_CONSUMERS: "API route - external consumers",
        ReasonCode.PACKAGE_NAME_BREAKING: "Package name - breaking change",
        ReasonCode.UNKNOWN_CONTEXT: "Context unclear",
        ReasonCode.EXPANSION_FAILED: "Evidence expansion failed",
    }
    return defaults.get(reason_code, "Classified by policy")


# =============================================================================
# RISK REGISTER BUILDER
# =============================================================================

def build_risk_register(plan: RefactorPlanV3) -> str:
    """
    Build supplementary risk register markdown.
    
    Separate from the main POT spec - provides detailed risk information.
    """
    sections = []
    
    sections.append("# Risk Register")
    sections.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    
    # Preserved items
    preserved = [m for m in plan.classified_matches if m.decision == ChangeDecision.SKIP]
    if preserved:
        sections.append("\n## Preserved Items\n")
        for m in preserved[:30]:
            sections.append(f"- `{m.file_path}:{m.line_number}` - {m.reason_text or _default_reason(m.reason_code)}")
    
    # Flagged items
    flagged = [m for m in plan.classified_matches if m.decision == ChangeDecision.FLAG]
    if flagged:
        sections.append("\n## Flagged Items\n")
        for m in flagged[:30]:
            sections.append(f"- `{m.file_path}:{m.line_number}` - {m.impact_note or m.reason_text or 'Review recommended'}")
    
    # Expansion failures
    if plan.expansion_failures:
        sections.append("\n## Expansion Failures\n")
        for issue in plan.expansion_failures:
            sections.append(f"- `{issue.file_path}:{issue.line_number or '?'}` - {issue.reason}")
    
    return "\n".join(sections)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "build_pot_spec_markdown",
    "build_risk_register",
    "POT_SPEC_BUILDER_BUILD_ID",
]
