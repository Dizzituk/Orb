# FILE: app/orchestrator/compiler_models.py
"""
Data models for the Implementation Compiler.

Contains FileBrief, FileFunction, FileImport, CompilerProfile,
and CompilationResult — shared between the compiler and validator.

v1.0 (2026-02-20): Split from implementation_compiler.py
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class CompilerProfile(str, Enum):
    """Job type determines how briefs are structured."""
    REFACTOR = "refactor"
    GREENFIELD = "greenfield"
    MODIFY = "modify"


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class FileImport:
    """A single import statement needed by a file."""
    statement: str          # Full import line e.g. "from ._constants import FOO"
    source_segment: str     # Which segment provides this symbol (empty for stdlib)
    symbols: List[str]      # Individual symbol names imported


@dataclass
class FileFunction:
    """A function/class/constant to be placed in a file."""
    name: str
    kind: str               # "function", "class", "constant"
    signature: str           # Full signature line
    body: str                # Complete source code (for refactors)
    line_count: int          # Lines in body
    is_async: bool = False
    docstring: str = ""


@dataclass
class FileBrief:
    """Complete implementation brief for a single target file."""

    # Identity
    file_path: str                                  # Target file path
    operation: str                                  # "CREATE" or "MODIFY"
    segment_id: str                                 # Owning segment

    # Content — what goes in this file
    functions: List[FileFunction] = field(default_factory=list)
    imports: List[FileImport] = field(default_factory=list)

    # Contracts — what this file must satisfy
    exports: List[str] = field(default_factory=list)        # Symbols this file defines
    consumed_by: Dict[str, List[str]] = field(default_factory=dict)  # {sibling_file: [symbols]}
    consumes_from: Dict[str, List[str]] = field(default_factory=dict)  # {sibling_file: [symbols]}

    # Context
    instruction: str = ""                           # Profile-specific directive
    feedback: List[str] = field(default_factory=list)  # Previous failure feedback
    design_notes: str = ""                          # From architecture document

    # v1.1: Deterministic frozen import block (Job 2)
    frozen_import_section: str = ""                  # Pre-rendered frozen import prompt section

    # v1.2: Code scaffold (Job 6)
    scaffold_section: str = ""                        # Pre-built code skeleton with [LLM_FILL] markers

    # Metadata
    estimated_lines: int = 0
    profile: str = ""

    def to_markdown(self) -> str:
        """Render this brief as a structured markdown document."""
        parts: List[str] = []
        parts.append(f"# Implementation Brief: `{self.file_path}`")
        parts.append(f"**Operation:** {self.operation}")
        parts.append(f"**Segment:** {self.segment_id}")
        parts.append(f"**Profile:** {self.profile}")
        parts.append("")

        # Instruction block — FIRST for salience
        if self.instruction:
            parts.append("## Directive")
            parts.append("")
            parts.append(self.instruction)
            parts.append("")

        # v6.1 FIX 24b: Mandatory symbol checklist — SECOND for attention.
        # Placed before function bodies so the LLM sees the complete list
        # of required symbols before it starts writing. This combats the
        # LLM dropping symbols from the tail of long briefs.
        if self.functions and self.profile == "refactor":
            func_names = [f.name for f in self.functions]
            parts.append("## MANDATORY SYMBOL CHECKLIST")
            parts.append("")
            parts.append(
                f"This file MUST contain ALL {len(func_names)} symbols listed below. "
                f"Do NOT skip any. After writing the file, verify every symbol is present."
            )
            parts.append("")
            for i, name in enumerate(func_names, 1):
                parts.append(f"{i}. `{name}`")
            parts.append("")

        # Feedback from previous failures — THIRD for urgency
        if self.feedback:
            parts.append("## Previous Failure Feedback (MUST address)")
            parts.append("")
            for fb in self.feedback:
                parts.append(f"- {fb}")
            parts.append("")

        # Source code to implement — THIRD (primary content for refactors)
        if self.functions:
            parts.append("## Functions to Implement")
            parts.append("")
            for func in self.functions:
                _kind_label = func.kind.title()
                parts.append(f"### `{func.name}` ({_kind_label}, {func.line_count} lines)")
                parts.append("")
                if func.body:
                    parts.append("```python")
                    parts.append(func.body)
                    parts.append("```")
                elif func.signature:
                    parts.append(f"Signature: `{func.signature}`")
                parts.append("")

        # Imports — v1.1: frozen imports take priority when available
        if self.frozen_import_section:
            parts.append(self.frozen_import_section)
        elif self.imports:
            parts.append("## Required Imports")
            parts.append("")
            parts.append("```python")
            for imp in self.imports:
                parts.append(imp.statement)
            parts.append("```")
            parts.append("")

        # Interface contract
        if self.exports or self.consumed_by:
            parts.append("## Interface Contract")
            parts.append("")
            if self.exports:
                parts.append("**This file exports:**")
                for exp in self.exports:
                    parts.append(f"- `{exp}`")
                parts.append("")
            if self.consumed_by:
                parts.append("**Consumed by sibling files:**")
                for sibling, symbols in self.consumed_by.items():
                    parts.append(f"- `{sibling}`: {', '.join(f'`{s}`' for s in symbols)}")
                parts.append("")
            if self.consumes_from:
                parts.append("**Imports from sibling files:**")
                for sibling, symbols in self.consumes_from.items():
                    parts.append(f"- `{sibling}`: {', '.join(f'`{s}`' for s in symbols)}")
                parts.append("")

        # v1.2: Code scaffold (Job 6) — before design notes for salience
        if self.scaffold_section:
            parts.append(self.scaffold_section)

        # Design notes from architecture (secondary context)
        if self.design_notes:
            parts.append("## Design Notes")
            parts.append("")
            parts.append(self.design_notes)
            parts.append("")

        return "\n".join(parts)


@dataclass
class CompilationResult:
    """Output of the implementation compiler."""
    briefs: List[FileBrief]
    profile: CompilerProfile
    total_functions: int = 0
    total_estimated_lines: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.value,
            "total_briefs": len(self.briefs),
            "total_functions": self.total_functions,
            "total_estimated_lines": self.total_estimated_lines,
            "warnings": self.warnings,
            "briefs": [
                {
                    "file_path": b.file_path,
                    "operation": b.operation,
                    "function_count": len(b.functions),
                    "export_count": len(b.exports),
                    "estimated_lines": b.estimated_lines,
                }
                for b in self.briefs
            ],
        }


# =============================================================================
# PROFILE-SPECIFIC INSTRUCTIONS
# =============================================================================


def build_instruction(
    profile: CompilerProfile,
    file_path: str,
    functions: List[FileFunction],
    operation: str = "CREATE",
) -> str:
    """Build the profile-specific directive for a file's brief.

    v2.0: operation param ensures CREATE files get CREATE directives
    even when the global profile is MODIFY.
    """

    if profile == CompilerProfile.REFACTOR:
        func_names = ", ".join(f"`{f.name}`" for f in functions[:5])
        has_bodies = any(f.body and len(f.body) > 20 for f in functions)

        if has_bodies:
            return (
                "**REFACTOR MODE — TRANSPLANT VERBATIM**\n\n"
                f"This file contains {len(functions)} function(s): {func_names}.\n"
                "The exact source code is provided below. **Copy each function verbatim** — "
                "preserve all logic, variable names, docstrings, and error handling. "
                "Only update import paths to reflect the new package structure.\n\n"
                "**DO NOT** rewrite, simplify, optimise, or reimagine these functions. "
                "The goal is a faithful extraction — the code must behave identically "
                "to the monolith version."
            )
        else:
            return (
                "**REFACTOR MODE — IMPLEMENT TO SIGNATURE**\n\n"
                f"This file contains {len(functions)} function(s): {func_names}.\n"
                "Source code bodies are not available. Implement each function "
                "matching the exact signatures provided. Ensure types and return "
                "values match the interface contract."
            )

    elif profile == CompilerProfile.GREENFIELD:
        return (
            "**GREENFIELD MODE — IMPLEMENT TO CONTRACT**\n\n"
            f"This file contains {len(functions)} function(s) to implement.\n"
            "No existing source code — implement fresh to satisfy the interface "
            "contract and acceptance criteria. Ensure all exported symbols are "
            "defined and all consumed symbols are imported."
        )

    elif profile == CompilerProfile.MODIFY:
        if operation == "CREATE":
            return (
                "**CREATE MODE — NEW FILE**\n\n"
                "This file does not exist yet. Implement it from scratch "
                "following the design notes and interface contract below. "
                "Ensure all exported symbols are defined."
            )
        return (
            "**MODIFY MODE — UPDATE EXISTING FILE**\n\n"
            "This file already exists. Apply the specified changes while "
            "preserving existing functionality. Ensure backward compatibility "
            "for all exported symbols."
        )

    return ""


# =============================================================================
# PROFILE DETECTION
# =============================================================================


def detect_profile(
    architecture_text: str,
    enrichment: dict | None,
    source_file_evidence: dict | None,
) -> CompilerProfile:
    """
    Determine the compiler profile based on available evidence.

    - REFACTOR: enrichment has source_extract with function bodies
    - MODIFY: source files exist but no enrichment extraction
    - GREENFIELD: no source files, building from scratch
    """
    if enrichment and enrichment.get("source_extract"):
        source_extract = enrichment["source_extract"]
        if any(len(body) > 20 for body in source_extract.values()):
            return CompilerProfile.REFACTOR

    if source_file_evidence:
        return CompilerProfile.MODIFY

    return CompilerProfile.GREENFIELD
