# FILE: app/pot_spec/governance_rules.py
# Purpose: Pipeline Governance Rules — Non-Negotiable Build Constraints.
# Called-by: app.llm.critical_pipeline.prompt_builder, app.pot_spec.grounded.simple_create
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Pipeline Governance Rules — Non-Negotiable Build Constraints.

These rules are HARD CONSTRAINTS that every pipeline stage must respect.
They are not suggestions for the LLM to consider — they are deterministic
checks that block non-compliant specs and implementation plans.

Rules:
------
1. DETERMINISTIC-FIRST
   Always prefer deterministic code (templates, patterns, scaffolding)
   over LLM-generated code. LLM fills gaps only.

2. NEW-FILES-FIRST
   Default to creating new files. Only edit existing files when there
   is genuinely no other way. Small patches > monolithic edits.

3. FILE SIZE DISCIPLINE
   Target: 20KB per file. Hard max: 30KB.
   If a file would exceed this, split into cooperating modules.
   Single responsibility per file. No god files.

These rules are injected into:
- SpecGate prompts (so specs are built compliant from the start)
- Segmentation (so segments respect file boundaries)
- Critical pipeline prompts (so implementation follows the rules)
- Overwatcher checks (so violations are caught post-build)

v2.1 (2026-02): Initial implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# =========================================================================
# Constants
# =========================================================================

FILE_SIZE_TARGET_KB = 20
FILE_SIZE_HARD_MAX_KB = 30
FILE_SIZE_TARGET_BYTES = FILE_SIZE_TARGET_KB * 1024
FILE_SIZE_HARD_MAX_BYTES = FILE_SIZE_HARD_MAX_KB * 1024

# Maximum lines added to an existing file before we mandate a new module
MAX_LINES_ADDED_TO_EXISTING = 50

# Minimum percentage of code that should come from templates/patterns
# before LLM generation is acceptable
DETERMINISTIC_TARGET_PERCENT = 70


# =========================================================================
# Governance check results
# =========================================================================

@dataclass
class GovernanceViolation:
    """A single rule violation detected in a spec or implementation plan."""
    rule: str                   # "deterministic_first" | "new_files_first" | "file_size"
    severity: str               # "block" | "warn"
    message: str                # Human-readable description
    file_path: Optional[str] = None  # Which file is affected
    suggestion: str = ""        # How to fix it


@dataclass
class GovernanceResult:
    """Result of governance checks on a spec or plan."""
    compliant: bool
    violations: List[GovernanceViolation] = field(default_factory=list)
    warnings: List[GovernanceViolation] = field(default_factory=list)

    @property
    def has_blockers(self) -> bool:
        return any(v.severity == "block" for v in self.violations)

    def summary(self) -> str:
        if self.compliant:
            return "All governance rules passed."
        parts = []
        for v in self.violations:
            parts.append(f"[{v.severity.upper()}] {v.rule}: {v.message}")
            if v.suggestion:
                parts.append(f"  → {v.suggestion}")
        return "\n".join(parts)


# =========================================================================
# Spec-level governance prompt injection
# =========================================================================

SPEC_GATE_GOVERNANCE_PROMPT = """
## PIPELINE GOVERNANCE RULES (Non-Negotiable)

You MUST design specs that comply with ALL of the following rules.
These are hard constraints, not suggestions.

### Rule 1: DETERMINISTIC-FIRST
- Always look for template/pattern/scaffolding routes BEFORE planning LLM generation.
- If a known pattern exists (API wrapper, CRUD, OAuth2 flow, FastAPI router, config file,
  database model, test scaffolding), use deterministic code generation.
- LLM should only fill genuinely creative logic — business rules inside function bodies,
  novel algorithms, domain-specific processing.
- Mark deterministic sections as TEMPLATE and LLM sections as LLM_FILL in the spec.
- Target: 70%+ of code should be deterministic scaffolding, 30% or less LLM-generated.

### Rule 2: NEW-FILES-FIRST
- DEFAULT to creating new files. Every new piece of functionality gets its own module.
- Only edit an existing file when:
  (a) You are fixing a bug IN that file, OR
  (b) You are adding an import/registration line (< 5 lines), OR
  (c) There is genuinely no way to achieve the goal without editing (rare).
- If you need to add > 50 lines to an existing file, CREATE A NEW MODULE instead
  and import it from the existing file.
- Rationale: Small, focused files are easier to debug, test, and iterate on.

### Rule 3: FILE SIZE DISCIPLINE
- Target file size: 20KB (approximately 500 lines).
- Hard maximum: 30KB. NO EXCEPTIONS.
- If a module would exceed 30KB, split it into cooperating submodules.
- Each file should have a SINGLE RESPONSIBILITY.
- Prefer composable modules over monolithic files.
- Name split files clearly: feature_models.py, feature_logic.py, feature_api.py.

When generating steps, outputs, and acceptance criteria, ensure they
reflect these rules. A spec that proposes a 400-line edit to an existing
file or a new 800-line monolith is NON-COMPLIANT and must be restructured.
"""


# =========================================================================
# Critical pipeline governance prompt injection
# =========================================================================

CRITICAL_PIPELINE_GOVERNANCE_PROMPT = """
## BUILD RULES (Non-Negotiable)

1. DETERMINISTIC-FIRST: Use templates/patterns for boilerplate. Only use LLM
   generation for genuinely creative logic inside function bodies.

2. NEW-FILES-FIRST: Create new files by default. Only edit existing files for
   bug fixes or tiny registration lines (< 5 lines). If adding > 50 lines
   to an existing file, create a new module and import instead.

3. FILE SIZE: Target 20KB per file. Hard max 30KB. Split into submodules
   if exceeded. Single responsibility per file.
"""


# =========================================================================
# Governance check functions
# =========================================================================

def check_spec_governance(
    outputs: list,
    steps: list,
    existing_files_touched: list[str] = None,
    new_files_created: list[str] = None,
    estimated_sizes: dict[str, int] = None,
) -> GovernanceResult:
    """
    Check a spec against governance rules.

    Args:
        outputs: List of output artifacts from the spec.
        steps: List of implementation steps.
        existing_files_touched: Files that will be modified.
        new_files_created: New files that will be created.
        estimated_sizes: Estimated file sizes in bytes {path: size}.

    Returns:
        GovernanceResult with violations if non-compliant.
    """
    violations = []
    warnings = []

    existing_files_touched = existing_files_touched or []
    new_files_created = new_files_created or []
    estimated_sizes = estimated_sizes or {}

    # Rule 2: Check excessive edits to existing files
    if len(existing_files_touched) > 0 and len(new_files_created) == 0:
        # Editing existing files but not creating any new ones — suspicious
        warnings.append(GovernanceViolation(
            rule="new_files_first",
            severity="warn",
            message=f"Plan edits {len(existing_files_touched)} existing file(s) but creates no new files.",
            suggestion="Consider extracting new functionality into separate modules.",
        ))

    # Rule 2: Flag large edits to existing files
    for step in steps:
        step_lower = (step if isinstance(step, str) else str(step)).lower()
        # Detect steps that describe adding lots of code to existing files
        if any(phrase in step_lower for phrase in [
            "add to existing", "modify existing", "extend existing",
            "append to", "insert into", "update the existing",
        ]):
            for f in existing_files_touched:
                size = estimated_sizes.get(f, 0)
                if size > FILE_SIZE_HARD_MAX_BYTES:
                    violations.append(GovernanceViolation(
                        rule="new_files_first",
                        severity="block",
                        message=f"Editing {f} would exceed {FILE_SIZE_HARD_MAX_KB}KB limit.",
                        file_path=f,
                        suggestion=f"Extract new logic into a separate module and import from {f}.",
                    ))

    # Rule 3: Check estimated file sizes
    for path, size in estimated_sizes.items():
        if size > FILE_SIZE_HARD_MAX_BYTES:
            violations.append(GovernanceViolation(
                rule="file_size",
                severity="block",
                message=f"{path} estimated at {size // 1024}KB, exceeds {FILE_SIZE_HARD_MAX_KB}KB hard max.",
                file_path=path,
                suggestion=f"Split {path} into smaller modules with single responsibilities.",
            ))
        elif size > FILE_SIZE_TARGET_BYTES:
            warnings.append(GovernanceViolation(
                rule="file_size",
                severity="warn",
                message=f"{path} estimated at {size // 1024}KB, exceeds {FILE_SIZE_TARGET_KB}KB target.",
                file_path=path,
                suggestion=f"Consider splitting {path} if it grows further.",
            ))

    all_violations = violations + [w for w in warnings if w.severity == "block"]
    compliant = not any(v.severity == "block" for v in all_violations)

    return GovernanceResult(
        compliant=compliant,
        violations=violations,
        warnings=warnings,
    )


def check_file_size_compliance(file_path: str, content: str) -> GovernanceResult:
    """
    Check a single file's content against size rules.
    Used by Overwatcher to validate build output.
    """
    size = len(content.encode("utf-8"))
    violations = []
    warnings = []

    if size > FILE_SIZE_HARD_MAX_BYTES:
        violations.append(GovernanceViolation(
            rule="file_size",
            severity="block",
            message=f"{file_path} is {size // 1024}KB, exceeds {FILE_SIZE_HARD_MAX_KB}KB hard max.",
            file_path=file_path,
            suggestion="Split into smaller modules.",
        ))
    elif size > FILE_SIZE_TARGET_BYTES:
        warnings.append(GovernanceViolation(
            rule="file_size",
            severity="warn",
            message=f"{file_path} is {size // 1024}KB, exceeds {FILE_SIZE_TARGET_KB}KB target.",
            file_path=file_path,
            suggestion="Consider splitting if adding more logic.",
        ))

    return GovernanceResult(
        compliant=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


def check_edit_vs_create(
    file_path: str,
    lines_added: int,
    current_file_size: int = 0,
) -> GovernanceResult:
    """
    Check whether an edit to an existing file should be a new module instead.
    Used by the critical pipeline before applying changes.
    """
    violations = []
    warnings = []

    projected_size = current_file_size + (lines_added * 80)  # ~80 bytes per line estimate

    if lines_added > MAX_LINES_ADDED_TO_EXISTING:
        violations.append(GovernanceViolation(
            rule="new_files_first",
            severity="block",
            message=f"Adding {lines_added} lines to {file_path} exceeds {MAX_LINES_ADDED_TO_EXISTING} line limit for existing file edits.",
            file_path=file_path,
            suggestion=f"Create a new module for this logic and add a single import line to {file_path}.",
        ))

    if projected_size > FILE_SIZE_HARD_MAX_BYTES:
        violations.append(GovernanceViolation(
            rule="file_size",
            severity="block",
            message=f"Edit would push {file_path} to ~{projected_size // 1024}KB, exceeding {FILE_SIZE_HARD_MAX_KB}KB.",
            file_path=file_path,
            suggestion="Extract into a new module.",
        ))

    return GovernanceResult(
        compliant=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


__all__ = [
    # Constants
    "FILE_SIZE_TARGET_KB",
    "FILE_SIZE_HARD_MAX_KB",
    "FILE_SIZE_TARGET_BYTES",
    "FILE_SIZE_HARD_MAX_BYTES",
    "MAX_LINES_ADDED_TO_EXISTING",
    "DETERMINISTIC_TARGET_PERCENT",
    # Prompt injections
    "SPEC_GATE_GOVERNANCE_PROMPT",
    "CRITICAL_PIPELINE_GOVERNANCE_PROMPT",
    # Data classes
    "GovernanceViolation",
    "GovernanceResult",
    # Check functions
    "check_spec_governance",
    "check_file_size_compliance",
    "check_edit_vs_create",
]
