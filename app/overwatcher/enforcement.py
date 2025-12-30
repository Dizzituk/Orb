# FILE: app/overwatcher/enforcement.py
"""Global Enforcement Layer: Overwatcher Output Validation.

This module enforces the NON-NEGOTIABLE rules for Overwatcher outputs:

1. NO CODE - Overwatcher must never output:
   - Code in any language
   - Diffs/patches
   - Full files
   - Shell/PowerShell commands
   - Step-by-step command sequences

2. STRUCTURED OUTPUT ONLY - Overwatcher must output:
   - DECISION (pass/fail/needs_info)
   - DIAGNOSIS (what's wrong)
   - FIX_ACTIONS (high-level, delegated)
   - CONSTRAINTS (boundaries)
   - VERIFICATION (what evidence to return)

3. TOKEN CAPS - Enforce output limits:
   - Default: 600-1500 tokens
   - Maximum: 2000 tokens
   - Break-glass: 4000 tokens (requires explicit flag)

4. REJECTION - Invalid outputs are rejected and logged.

This layer is called AFTER every Overwatcher response, BEFORE
any action is taken. It is the final safety gate.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Token limits
DEFAULT_OUTPUT_TOKENS = 1500
MAX_OUTPUT_TOKENS = 2000
BREAK_GLASS_TOKENS = 4000

# Approximate tokens per character (conservative)
CHARS_PER_TOKEN = 4


# =============================================================================
# Code Detection Patterns
# =============================================================================

# These patterns indicate CODE that Overwatcher must not produce

CODE_FENCE_PATTERN = re.compile(
    r"```(?:python|javascript|typescript|bash|powershell|shell|sh|ps1|cmd|sql|json|yaml|yml|xml|html|css|java|go|rust|c|cpp|csharp|ruby|php|swift|kotlin)?\s*\n",
    re.IGNORECASE
)

# Function/class definitions
FUNCTION_DEF_PATTERN = re.compile(
    r"^\s*(def|async def|class|function|const|let|var|public|private|protected)\s+\w+",
    re.MULTILINE
)

# Import statements
IMPORT_PATTERN = re.compile(
    r"^\s*(import|from|require|using|include|#include)\s+",
    re.MULTILINE
)

# Shell commands (PowerShell, bash)
SHELL_COMMAND_PATTERN = re.compile(
    r"^\s*(\$|>|PS>|C:\\>|#\s*!|chmod|mkdir|cd|ls|dir|cat|echo|pip|npm|yarn|git|python|pytest|ruff|mypy|Copy-Item|Remove-Item|Get-|Set-|New-|Invoke-)",
    re.MULTILINE | re.IGNORECASE
)

# Diff/patch markers
DIFF_PATTERN = re.compile(
    r"^\s*[\+\-]{3}\s+[ab]?/?|^\s*@@\s*[\-\+]?\d+|^\s*diff\s+--git",
    re.MULTILINE
)

# File path with content (like writing a file)
FILE_WRITE_PATTERN = re.compile(
    r"(# FILE:|// FILE:|<!-- FILE:|```\s*\w+\s*\n#\s*\w+\.)",
    re.IGNORECASE
)

# Indented code blocks (4+ spaces followed by code-like content)
INDENTED_CODE_PATTERN = re.compile(
    r"^(\s{4,}|\t+)(if|for|while|def|class|return|import|try|except|with|async)\s",
    re.MULTILINE
)

# JSON/dict literals that look like data structures
DATA_STRUCTURE_PATTERN = re.compile(
    r"^\s*[\{\[][\s\S]{50,}[\}\]]\s*$",
    re.MULTILINE
)


# =============================================================================
# Violation Types
# =============================================================================

class ViolationType(str, Enum):
    """Types of enforcement violations."""
    CODE_FENCE = "code_fence"
    FUNCTION_DEF = "function_definition"
    IMPORT_STATEMENT = "import_statement"
    SHELL_COMMAND = "shell_command"
    DIFF_PATCH = "diff_patch"
    FILE_WRITE = "file_write"
    INDENTED_CODE = "indented_code"
    DATA_STRUCTURE = "data_structure"
    TOKEN_LIMIT = "token_limit"
    MISSING_FIELD = "missing_field"
    INVALID_STRUCTURE = "invalid_structure"


@dataclass
class Violation:
    """A single enforcement violation."""
    type: ViolationType
    message: str
    excerpt: str = ""  # Offending content (truncated)
    line_number: int = 0
    severity: str = "error"  # error, warning
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "message": self.message,
            "excerpt": self.excerpt[:100] if self.excerpt else "",
            "line_number": self.line_number,
            "severity": self.severity,
        }


@dataclass
class EnforcementResult:
    """Result of enforcement check."""
    valid: bool
    violations: List[Violation] = field(default_factory=list)
    warnings: List[Violation] = field(default_factory=list)
    token_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": [w.to_dict() for w in self.warnings],
            "token_count": self.token_count,
        }


# =============================================================================
# Code Detection
# =============================================================================

def detect_code_violations(text: str) -> List[Violation]:
    """Detect code patterns in text.
    
    Returns list of violations found.
    """
    violations = []
    
    # Check for code fences
    matches = CODE_FENCE_PATTERN.finditer(text)
    for match in matches:
        violations.append(Violation(
            type=ViolationType.CODE_FENCE,
            message="Code fence detected - Overwatcher must not output code blocks",
            excerpt=match.group(0),
            line_number=text[:match.start()].count('\n') + 1,
        ))
    
    # Check for function/class definitions
    matches = FUNCTION_DEF_PATTERN.finditer(text)
    for match in matches:
        violations.append(Violation(
            type=ViolationType.FUNCTION_DEF,
            message="Function/class definition detected",
            excerpt=match.group(0),
            line_number=text[:match.start()].count('\n') + 1,
        ))
    
    # Check for import statements
    matches = IMPORT_PATTERN.finditer(text)
    for match in matches:
        violations.append(Violation(
            type=ViolationType.IMPORT_STATEMENT,
            message="Import statement detected",
            excerpt=match.group(0),
            line_number=text[:match.start()].count('\n') + 1,
        ))
    
    # Check for shell commands
    matches = SHELL_COMMAND_PATTERN.finditer(text)
    for match in matches:
        violations.append(Violation(
            type=ViolationType.SHELL_COMMAND,
            message="Shell command detected - Overwatcher must not output commands",
            excerpt=match.group(0),
            line_number=text[:match.start()].count('\n') + 1,
        ))
    
    # Check for diffs/patches
    matches = DIFF_PATTERN.finditer(text)
    for match in matches:
        violations.append(Violation(
            type=ViolationType.DIFF_PATCH,
            message="Diff/patch content detected",
            excerpt=match.group(0),
            line_number=text[:match.start()].count('\n') + 1,
        ))
    
    # Check for file write patterns
    matches = FILE_WRITE_PATTERN.finditer(text)
    for match in matches:
        violations.append(Violation(
            type=ViolationType.FILE_WRITE,
            message="File write pattern detected",
            excerpt=match.group(0),
            line_number=text[:match.start()].count('\n') + 1,
        ))
    
    # Check for indented code
    matches = INDENTED_CODE_PATTERN.finditer(text)
    for match in matches:
        violations.append(Violation(
            type=ViolationType.INDENTED_CODE,
            message="Indented code block detected",
            excerpt=match.group(0),
            line_number=text[:match.start()].count('\n') + 1,
        ))
    
    return violations


def estimate_tokens(text: str) -> int:
    """Estimate token count from text."""
    return len(text) // CHARS_PER_TOKEN


# =============================================================================
# Structure Validation
# =============================================================================

REQUIRED_FIELDS = {"decision", "diagnosis"}
OPTIONAL_FIELDS = {"fix_actions", "constraints", "verification", "blockers", "confidence"}


def validate_structure(output: Dict[str, Any]) -> List[Violation]:
    """Validate that output has required structure.
    
    Returns list of violations.
    """
    violations = []
    
    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in output or output[field] is None:
            violations.append(Violation(
                type=ViolationType.MISSING_FIELD,
                message=f"Missing required field: {field}",
                severity="error",
            ))
    
    # Validate decision value
    if "decision" in output:
        valid_decisions = {"pass", "fail", "needs_info", "PASS", "FAIL", "NEEDS_INFO"}
        if output["decision"] not in valid_decisions:
            violations.append(Violation(
                type=ViolationType.INVALID_STRUCTURE,
                message=f"Invalid decision value: {output['decision']}",
                severity="error",
            ))
    
    # Check fix_actions structure (should be list of high-level actions)
    if "fix_actions" in output and output["fix_actions"]:
        fix_actions = output["fix_actions"]
        if isinstance(fix_actions, list):
            for i, action in enumerate(fix_actions):
                if isinstance(action, dict):
                    # Check for code in action descriptions
                    desc = action.get("description", "")
                    if desc:
                        code_violations = detect_code_violations(desc)
                        for v in code_violations:
                            v.message = f"Code in fix_action[{i}]: {v.message}"
                            violations.append(v)
    
    return violations


# =============================================================================
# Main Enforcement Function
# =============================================================================

def enforce_overwatcher_output(
    raw_output: str,
    parsed_output: Optional[Dict[str, Any]] = None,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    break_glass: bool = False,
) -> EnforcementResult:
    """Enforce all Overwatcher output rules.
    
    This is the MAIN enforcement function. Call it after every
    Overwatcher response, before taking any action.
    
    Args:
        raw_output: Raw text output from Overwatcher
        parsed_output: Parsed JSON/dict output (if available)
        max_tokens: Maximum allowed tokens
        break_glass: If True, allow up to BREAK_GLASS_TOKENS
    
    Returns:
        EnforcementResult with validity and violations
    """
    violations = []
    warnings = []
    
    # Estimate tokens
    token_count = estimate_tokens(raw_output)
    
    # Check token limit
    effective_limit = BREAK_GLASS_TOKENS if break_glass else max_tokens
    if token_count > effective_limit:
        violations.append(Violation(
            type=ViolationType.TOKEN_LIMIT,
            message=f"Output exceeds token limit: {token_count} > {effective_limit}",
            severity="error",
        ))
    elif token_count > DEFAULT_OUTPUT_TOKENS:
        warnings.append(Violation(
            type=ViolationType.TOKEN_LIMIT,
            message=f"Output above recommended limit: {token_count} > {DEFAULT_OUTPUT_TOKENS}",
            severity="warning",
        ))
    
    # Detect code violations
    code_violations = detect_code_violations(raw_output)
    violations.extend(code_violations)
    
    # Validate structure if parsed output available
    if parsed_output is not None:
        structure_violations = validate_structure(parsed_output)
        violations.extend(structure_violations)
    
    # Determine validity
    errors = [v for v in violations if v.severity == "error"]
    valid = len(errors) == 0
    
    # Log violations
    if not valid:
        logger.warning(
            f"[enforcement] Overwatcher output REJECTED: {len(errors)} violations"
        )
        for v in errors:
            logger.warning(f"  - {v.type.value}: {v.message}")
    
    return EnforcementResult(
        valid=valid,
        violations=violations,
        warnings=warnings,
        token_count=token_count,
    )


def enforce_and_reject(
    raw_output: str,
    parsed_output: Optional[Dict[str, Any]] = None,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    break_glass: bool = False,
) -> Tuple[bool, Optional[Dict[str, Any]], EnforcementResult]:
    """Enforce rules and return sanitized output or None.
    
    Args:
        raw_output: Raw Overwatcher output
        parsed_output: Parsed output dict
        max_tokens: Token limit
        break_glass: Allow extended tokens
    
    Returns:
        (valid, output_or_none, enforcement_result)
        
        If valid=True, output_or_none is the parsed output.
        If valid=False, output_or_none is None.
    """
    result = enforce_overwatcher_output(
        raw_output=raw_output,
        parsed_output=parsed_output,
        max_tokens=max_tokens,
        break_glass=break_glass,
    )
    
    if result.valid:
        return True, parsed_output, result
    else:
        return False, None, result


# =============================================================================
# Allowlist for Implementer
# =============================================================================

# The Implementer (Claude Sonnet) IS allowed to produce these
# This is the inverse of Overwatcher restrictions

IMPLEMENTER_ALLOWED = {
    "code": True,
    "diffs": True,
    "full_files": True,
    "shell_commands": True,
    "powershell": True,
    "file_writes": True,
}


def is_implementer_output_valid(output: str) -> bool:
    """Check if output is valid for Implementer role.
    
    Implementer CAN produce code, commands, files.
    This is a permissive check.
    """
    # Implementer has almost no restrictions
    # Just check for obviously malicious patterns
    
    dangerous_patterns = [
        r"rm\s+-rf\s+/",           # rm -rf /
        r"del\s+/s\s+/q\s+c:\\",   # del /s /q c:\
        r"format\s+c:",            # format c:
        r":(){:|:&};:",            # fork bomb
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, output, re.IGNORECASE):
            logger.error(f"[enforcement] Dangerous pattern in Implementer output: {pattern}")
            return False
    
    return True


__all__ = [
    # Config
    "DEFAULT_OUTPUT_TOKENS",
    "MAX_OUTPUT_TOKENS",
    "BREAK_GLASS_TOKENS",
    # Types
    "ViolationType",
    "Violation",
    "EnforcementResult",
    # Detection
    "detect_code_violations",
    "estimate_tokens",
    "validate_structure",
    # Main enforcement
    "enforce_overwatcher_output",
    "enforce_and_reject",
    # Implementer
    "is_implementer_output_valid",
]
