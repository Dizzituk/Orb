from __future__ import annotations
import json
import logging
import os
import re
from app.overwatcher._sandbox_build_validator_utils_2 import ERROR_TYPE_PATTERNS, FILE_PATH_PATTERNS
from typing import List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
PROJECT_VITE_REACT = "vite_react"
PROJECT_PYTHON_BACKEND = "python_backend"


MAX_BUILD_FIX_ATTEMPTS = int(os.getenv("OVERWATCHER_MAX_BUILD_FIX_ATTEMPTS", "3"))

MAX_BUILD_OUTPUT_CHARS = 5000

ALLOWED_FIX_COMMANDS = [
    "npm install",
    "npm ci",
    "npx tsc",
    "npx vite build",
    "python -m py_compile",
    "pip install",
]

def detect_project_from_path(file_path: str) -> Optional[str]:
    """Determine which sandbox project a file belongs to based on its path.

    Args:
        file_path: Sandbox file path (e.g. C:\\Orb\\orb-desktop\\src\\App.tsx)

    Returns:
        PROJECT_VITE_REACT, PROJECT_PYTHON_BACKEND, or None
    """
    # Normalize path separators for comparison
    normalized = file_path.replace("/", "\\").lower()

    if "orb-desktop" in normalized or "orb\\orb-desktop" in normalized:
        return PROJECT_VITE_REACT
    elif "orb\\orb\\" in normalized or normalized.endswith("orb\\orb"):
        return PROJECT_PYTHON_BACKEND
    # Also match paths like C:\Orb\Orb\app\...
    elif re.search(r"c:\\orb\\orb\\", normalized):
        return PROJECT_PYTHON_BACKEND
    return None

def parse_build_error_output(
    stdout: str,
    stderr: str,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Parse build error output to extract structured error information.

    Args:
        stdout: Build command stdout
        stderr: Build command stderr

    Returns:
        Tuple of (error_summary, error_type, affected_files)
    """
    combined = f"{stderr}\n{stdout}"
    if not combined.strip():
        return None, None, []

    # Extract error type
    error_type = None
    for pattern, etype in ERROR_TYPE_PATTERNS:
        if pattern.search(combined):
            error_type = etype
            break

    # Extract affected files
    affected_files: List[str] = []
    seen_paths: set = set()
    for pattern in FILE_PATH_PATTERNS:
        for match in pattern.finditer(combined):
            fpath = match.group(1) if match.lastindex else match.group(0)
            # Normalize and deduplicate
            fpath_normalized = fpath.replace("/", "\\").strip()
            if fpath_normalized not in seen_paths and len(fpath_normalized) > 3:
                # Filter out common false positives
                if not fpath_normalized.startswith("http") and "node_modules" not in fpath_normalized:
                    seen_paths.add(fpath_normalized)
                    affected_files.append(fpath_normalized)

    # Extract error summary (first meaningful error line)
    error_summary = None
    for line in combined.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Look for lines that contain error indicators
        if any(
            kw in line.lower()
            for kw in ["error", "failed", "syntaxerror", "typeerror", "cannot"]
        ):
            error_summary = line[:300]  # Truncate long lines
            break

    return error_summary, error_type, affected_files

def _truncate_output(text: str, max_chars: int = MAX_BUILD_OUTPUT_CHARS) -> str:
    """Truncate text, keeping head and tail for diagnostic value."""
    if not text or len(text) <= max_chars:
        return text or ""
    half = max_chars // 2
    return (
        text[:half]
        + f"\n\n... [{len(text) - max_chars} chars truncated] ...\n\n"
        + text[-half:]
    )

def _parse_diagnostic_response(raw_text: str) -> DiagnosticResult:
    """Parse the LLM's diagnostic response into structured data.

    Handles: raw JSON, JSON in code fences, partial/malformed JSON.
    """
    if not raw_text:
        return DiagnosticResult(
            diagnosis="Empty response from diagnostic LLM",
            root_cause="llm_error",
        )

    text = raw_text.strip()

    # Try to extract JSON from code fence
    fence_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parse
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object with brace matching
        start = text.find("{")
        if start >= 0:
            depth = 0
            in_string = False
            escape = False
            end = -1
            for i, char in enumerate(text[start:], start):
                if escape:
                    escape = False
                    continue
                if char == "\\":
                    escape = True
                    continue
                if char == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

    if not data or not isinstance(data, dict):
        logger.warning(
            "[build_validator] Could not parse diagnostic response: %s",
            raw_text[:300],
        )
        return DiagnosticResult(
            diagnosis=f"Could not parse diagnostic response: {raw_text[:200]}",
            root_cause="parse_error",
            raw_response=raw_text,
        )

    # Build fix actions
    fixes: List[BuildFixAction] = []
    for fix_data in data.get("fixes", []):
        fix_type = fix_data.get("fix_type", "")

        # Validate fix type
        if fix_type not in ("rewrite_file", "run_command", "revert_file"):
            logger.warning(
                "[build_validator] Unknown fix_type '%s' — skipping", fix_type
            )
            continue

        # Validate run_command safety
        if fix_type == "run_command":
            command = fix_data.get("command", "")
            if not _is_safe_command(command):
                logger.warning(
                    "[build_validator] Unsafe command rejected: '%s'", command
                )
                continue

        fixes.append(
            BuildFixAction(
                fix_type=fix_type,
                file_path=fix_data.get("file_path"),
                content=fix_data.get("content"),
                command=fix_data.get("command"),
                diagnosis=data.get("diagnosis", ""),
                rationale=fix_data.get("rationale", ""),
            )
        )

    return DiagnosticResult(
        diagnosis=data.get("diagnosis", "No diagnosis provided"),
        root_cause=data.get("root_cause", "unknown"),
        fixes=fixes,
        confidence=float(data.get("confidence", 0.0)),
        raw_response=raw_text,
    )

def _is_safe_command(command: str) -> bool:
    """Check if a command is in the allowed list (safety constraint).

    Only permits known safe operations like npm install, pip install, etc.
    """
    if not command:
        return False
    cmd_lower = command.strip().lower()
    return any(cmd_lower.startswith(prefix.lower()) for prefix in ALLOWED_FIX_COMMANDS)
