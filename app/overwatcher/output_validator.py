# FILE: app/overwatcher/output_validator.py
"""
Output Validator for Overwatcher (Job 3)

Spec §4.4: Enforcing "no-code / no-commands"
- Post-check validator that rejects outputs containing prohibited patterns
- On rejection: reprompt with "contract violated, re-emit in allowed schema only"
- Contract violations do not count as ErrorSignature strikes

Detects:
- Fenced code blocks (```)
- Shell prompts (PS>, $, >, cmd, etc.)
- Language-specific code patterns
- Diff/patch patterns
- Step-by-step command sequences
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class ViolationType(str, Enum):
    """Type of contract violation detected."""
    CODE_FENCE = "code_fence"
    SHELL_PROMPT = "shell_prompt"
    PYTHON_CODE = "python_code"
    DIFF_PATCH = "diff_patch"
    COMMAND_SEQUENCE = "command_sequence"
    JAVASCRIPT_CODE = "javascript_code"
    SQL_CODE = "sql_code"
    FILE_CONTENT = "file_content"
    INLINE_CODE = "inline_code"


@dataclass
class Violation:
    """A single contract violation."""
    violation_type: ViolationType
    pattern_matched: str
    line_number: Optional[int] = None
    excerpt: str = ""  # Short excerpt showing the violation
    
    def to_dict(self) -> dict:
        return {
            "type": self.violation_type.value,
            "pattern": self.pattern_matched,
            "line": self.line_number,
            "excerpt": self.excerpt[:100] if self.excerpt else "",
        }


@dataclass
class ValidationResult:
    """Result of output validation."""
    is_valid: bool
    violations: List[Violation] = field(default_factory=list)
    reprompt_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "valid": self.is_valid,
            "violations": [v.to_dict() for v in self.violations],
            "reprompt": self.reprompt_message,
        }


# =============================================================================
# Pattern Definitions
# =============================================================================

# Code fence patterns (most reliable indicator)
CODE_FENCE_PATTERNS = [
    (r"```[\w]*\s*\n", "triple backtick code fence"),
    (r"~~~[\w]*\s*\n", "tilde code fence"),
]

# Shell prompt patterns
SHELL_PROMPT_PATTERNS = [
    (r"^\s*PS\s*[A-Za-z]?:?[>\s]", "PowerShell prompt"),
    (r"^\s*\$\s+\w", "Unix shell prompt"),
    (r"^\s*>\s*\w", "Generic shell prompt"),
    (r"^\s*C:\\[>\s]", "Windows cmd prompt"),
    (r"^\s*bash\s*[#\$]", "Bash prompt"),
    (r"^\s*zsh\s*[#%]", "Zsh prompt"),
    (r"^\s*root@", "Root prompt"),
    (r"^\s*user@", "User prompt"),
    (r"^\s*cmd[>\s]", "CMD prompt"),
]

# Python code patterns (indented = likely in code context)
PYTHON_CODE_PATTERNS = [
    (r"^\s{4,}def\s+\w+\s*\(", "Python function definition"),
    (r"^\s{4,}class\s+\w+\s*[\(:]", "Python class definition"),
    (r"^\s{4,}import\s+\w+", "Python import (indented)"),
    (r"^\s{4,}from\s+\w+\s+import", "Python from-import (indented)"),
    (r"^\s{4,}async\s+def\s+\w+", "Python async function"),
    (r"^\s{4,}@\w+", "Python decorator"),
    (r"^\s{4,}if\s+__name__\s*==", "Python main guard"),
    (r"^\s{4,}try:\s*$", "Python try block"),
    (r"^\s{4,}except\s+\w+", "Python except clause"),
    (r"^\s{4,}with\s+\w+.*:\s*$", "Python with statement"),
    (r"^\s{4,}for\s+\w+\s+in\s+", "Python for loop"),
    (r"^\s{4,}while\s+.*:\s*$", "Python while loop"),
    (r"^\s{4,}return\s+", "Python return statement"),
    (r"^\s{4,}yield\s+", "Python yield statement"),
    (r"^\s{4,}raise\s+\w+", "Python raise statement"),
]

# Diff/patch patterns
DIFF_PATTERNS = [
    (r"^\s*[-+]{3}\s+[ab]/", "Unified diff header"),
    (r"^\s*@@\s*-\d+", "Diff hunk header"),
    (r"^\s*\+\s*def\s+\w+", "Diff adding function"),
    (r"^\s*\+\s*class\s+\w+", "Diff adding class"),
    (r"^\s*\+\s*import\s+", "Diff adding import"),
    (r"^\s*-\s*def\s+\w+", "Diff removing function"),
    (r"^\s*-\s*class\s+\w+", "Diff removing class"),
    (r"^\s*diff\s+--git", "Git diff header"),
    (r"^\s*index\s+[a-f0-9]+\.\.[a-f0-9]+", "Git index line"),
]

# JavaScript/TypeScript patterns
JS_CODE_PATTERNS = [
    (r"^\s{4,}function\s+\w+\s*\(", "JS function definition"),
    (r"^\s{4,}const\s+\w+\s*=\s*\(", "JS arrow function"),
    (r"^\s{4,}let\s+\w+\s*=", "JS let declaration"),
    (r"^\s{4,}var\s+\w+\s*=", "JS var declaration"),
    (r"^\s{4,}export\s+(default\s+)?", "JS export"),
    (r"^\s{4,}import\s+.*\s+from\s+['\"]", "JS import"),
    (r"^\s{4,}async\s+function", "JS async function"),
    (r"^\s{4,}=>\s*\{", "JS arrow function body"),
]

# SQL patterns
SQL_CODE_PATTERNS = [
    (r"(?i)^\s{4,}SELECT\s+", "SQL SELECT"),
    (r"(?i)^\s{4,}INSERT\s+INTO", "SQL INSERT"),
    (r"(?i)^\s{4,}UPDATE\s+\w+\s+SET", "SQL UPDATE"),
    (r"(?i)^\s{4,}DELETE\s+FROM", "SQL DELETE"),
    (r"(?i)^\s{4,}CREATE\s+TABLE", "SQL CREATE TABLE"),
    (r"(?i)^\s{4,}ALTER\s+TABLE", "SQL ALTER TABLE"),
    (r"(?i)^\s{4,}DROP\s+TABLE", "SQL DROP TABLE"),
]

# File content indicators
FILE_CONTENT_PATTERNS = [
    (r"^#\s*FILE:\s*\S+\.\w+", "File header comment"),
    (r"^#!\s*/usr/bin/", "Shebang line"),
    (r"^#!\s*/bin/", "Shebang line"),
    (r"^<\?xml", "XML declaration"),
    (r"^<!DOCTYPE", "HTML doctype"),
    (r"^\s*<html", "HTML tag"),
]

# Command sequence patterns (step-by-step instructions that are effectively implementation)
COMMAND_SEQUENCE_PATTERNS = [
    (r"(?i)step\s*\d+[:.]\s*(run|execute|type|enter|copy|paste)", "Step-by-step commands"),
    (r"(?i)^\d+\.\s*(run|execute|type|enter)\s+", "Numbered command instruction"),
    (r"(?i)first,?\s*(run|execute|type)\s+", "Sequential command"),
    (r"(?i)then,?\s*(run|execute|type)\s+", "Sequential command"),
    (r"(?i)finally,?\s*(run|execute|type)\s+", "Sequential command"),
    (r"(?i)now\s+(run|execute|type)\s+", "Command instruction"),
]


# =============================================================================
# Validation Functions
# =============================================================================

def _check_patterns(
    text: str,
    patterns: List[Tuple[str, str]],
    violation_type: ViolationType,
) -> List[Violation]:
    """Check text against a list of patterns."""
    violations = []
    lines = text.split("\n")
    
    for line_num, line in enumerate(lines, 1):
        for pattern, description in patterns:
            if re.search(pattern, line, re.MULTILINE):
                violations.append(Violation(
                    violation_type=violation_type,
                    pattern_matched=description,
                    line_number=line_num,
                    excerpt=line.strip()[:80],
                ))
                break  # One violation per line is enough
    
    return violations


def _check_code_fences(text: str) -> List[Violation]:
    """Check for code fence blocks."""
    violations = []
    
    # Find all code fence blocks
    fence_pattern = r"```(\w*)\n([\s\S]*?)```"
    matches = re.finditer(fence_pattern, text)
    
    for match in matches:
        lang = match.group(1) or "unknown"
        content_preview = match.group(2)[:50].strip()
        
        # Calculate line number
        line_num = text[:match.start()].count("\n") + 1
        
        violations.append(Violation(
            violation_type=ViolationType.CODE_FENCE,
            pattern_matched=f"Code fence ({lang})",
            line_number=line_num,
            excerpt=f"```{lang}\\n{content_preview}...",
        ))
    
    return violations


def _check_inline_code_density(text: str) -> List[Violation]:
    """Check for excessive inline code that looks like implementation."""
    violations = []
    
    # Count backtick-wrapped segments
    inline_code_pattern = r"`[^`]+`"
    matches = re.findall(inline_code_pattern, text)
    
    # If there are many inline code segments with code-like content, flag it
    code_like_count = 0
    for match in matches:
        content = match.strip("`")
        # Check if it looks like actual code vs just a reference
        if any([
            "(" in content and ")" in content,  # Function call
            "=" in content,  # Assignment
            "." in content and len(content) > 20,  # Method chain
            content.startswith("def "),
            content.startswith("class "),
            "{" in content and "}" in content,  # Dict/object literal
        ]):
            code_like_count += 1
    
    # Threshold: more than 5 code-like inline segments is suspicious
    if code_like_count > 5:
        violations.append(Violation(
            violation_type=ViolationType.INLINE_CODE,
            pattern_matched=f"Excessive inline code ({code_like_count} code-like segments)",
            excerpt=f"Found {code_like_count} inline code segments that appear to be implementation",
        ))
    
    return violations


def validate_overwatcher_output(text: str) -> ValidationResult:
    """
    Validate Overwatcher output for contract compliance.
    
    Returns ValidationResult with is_valid=False if any violations found.
    """
    if not text:
        return ValidationResult(is_valid=True)
    
    all_violations: List[Violation] = []
    
    # Check code fences (highest priority)
    all_violations.extend(_check_code_fences(text))
    
    # Check shell prompts
    all_violations.extend(_check_patterns(
        text, SHELL_PROMPT_PATTERNS, ViolationType.SHELL_PROMPT
    ))
    
    # Check Python code
    all_violations.extend(_check_patterns(
        text, PYTHON_CODE_PATTERNS, ViolationType.PYTHON_CODE
    ))
    
    # Check diffs
    all_violations.extend(_check_patterns(
        text, DIFF_PATTERNS, ViolationType.DIFF_PATCH
    ))
    
    # Check JavaScript
    all_violations.extend(_check_patterns(
        text, JS_CODE_PATTERNS, ViolationType.JAVASCRIPT_CODE
    ))
    
    # Check SQL
    all_violations.extend(_check_patterns(
        text, SQL_CODE_PATTERNS, ViolationType.SQL_CODE
    ))
    
    # Check file content markers
    all_violations.extend(_check_patterns(
        text, FILE_CONTENT_PATTERNS, ViolationType.FILE_CONTENT
    ))
    
    # Check command sequences
    all_violations.extend(_check_patterns(
        text, COMMAND_SEQUENCE_PATTERNS, ViolationType.COMMAND_SEQUENCE
    ))
    
    # Check inline code density
    all_violations.extend(_check_inline_code_density(text))
    
    if all_violations:
        # Build reprompt message
        violation_summary = ", ".join(set(v.pattern_matched for v in all_violations[:5]))
        reprompt = (
            "CONTRACT VIOLATION: Your output contains prohibited content. "
            f"Detected: {violation_summary}. "
            "You MUST NOT output code, patches, diffs, file contents, or shell commands. "
            "Re-emit your response using ONLY the allowed schema: "
            "decision, diagnosis, fix_actions (high-level descriptions only), "
            "constraints, and verification steps."
        )
        
        return ValidationResult(
            is_valid=False,
            violations=all_violations,
            reprompt_message=reprompt,
        )
    
    return ValidationResult(is_valid=True)


def validate_fix_action(description: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a single fix action description.
    
    Returns (is_valid, violation_message)
    """
    result = validate_overwatcher_output(description)
    if not result.is_valid:
        return False, f"Fix action contains code: {result.violations[0].pattern_matched}"
    return True, None


def validate_diagnosis(diagnosis: str) -> Tuple[bool, Optional[str]]:
    """
    Validate diagnosis text.
    
    Returns (is_valid, violation_message)
    """
    result = validate_overwatcher_output(diagnosis)
    if not result.is_valid:
        return False, f"Diagnosis contains code: {result.violations[0].pattern_matched}"
    return True, None


# =============================================================================
# Reprompt Helper
# =============================================================================

REPROMPT_SYSTEM_ADDENDUM = """
CRITICAL: Your previous response violated the output contract.

You MUST NOT include:
- Code blocks (```)
- Shell/PowerShell commands or prompts
- Function/class definitions
- Diffs or patches
- File contents
- Step-by-step command sequences

You MUST output ONLY:
{
  "decision": "PASS|FAIL|NEEDS_INFO",
  "diagnosis": "Brief root cause (no code)",
  "fix_actions": [{"order": N, "target_file": "path", "action_type": "type", "description": "What to do (NOT how)", "rationale": "Why"}],
  "constraints": ["What must NOT change"],
  "verification": [{"command": "test command", "expected_outcome": "description"}]
}

Describe WHAT needs to be done, not HOW to implement it.
"""


def build_reprompt_messages(
    original_messages: List[dict],
    invalid_response: str,
    validation_result: ValidationResult,
) -> List[dict]:
    """
    Build messages for reprompting after a contract violation.
    
    Adds violation context and contract reminder.
    """
    # Start with original messages
    messages = list(original_messages)
    
    # Add the invalid response as assistant message
    messages.append({
        "role": "assistant",
        "content": invalid_response,
    })
    
    # Add correction instruction as user message
    violation_types = set(v.violation_type.value for v in validation_result.violations[:5])
    correction_msg = (
        f"Your response violated the Overwatcher contract. "
        f"Violations detected: {', '.join(violation_types)}. "
        f"Please re-emit your analysis using ONLY the allowed JSON schema. "
        f"NO CODE, NO COMMANDS, NO DIFFS. "
        f"Describe actions at a high level only."
    )
    
    messages.append({
        "role": "user",
        "content": correction_msg,
    })
    
    return messages


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Types
    "ViolationType",
    "Violation",
    "ValidationResult",
    # Functions
    "validate_overwatcher_output",
    "validate_fix_action",
    "validate_diagnosis",
    "build_reprompt_messages",
    # Constants
    "REPROMPT_SYSTEM_ADDENDUM",
]
