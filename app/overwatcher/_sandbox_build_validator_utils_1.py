from __future__ import annotations
import os
import re


BUILD_VALIDATOR_BUILD_ID = "2026-02-04-v1.1-path-inference"

BUILD_VALIDATION_ENABLED = os.getenv(
    "OVERWATCHER_BUILD_VALIDATION_ENABLED", "1"
).lower() in {"1", "true", "yes"}

MAX_DIAGNOSTIC_PROMPT_CHARS = 15000

PROJECT_UNKNOWN = "unknown"

FILE_PATH_PATTERNS = [
    # Vite/Node: absolute Windows paths
    re.compile(r"([A-Z]:\\[^\s:]+\.\w+)", re.IGNORECASE),
    # Vite/Node: relative paths with line numbers
    re.compile(r"([\w./\\-]+\.\w+):(\d+):(\d+)"),
    # Python: File "path", line N
    re.compile(r'File "([^"]+)"', re.IGNORECASE),
]

ERROR_TYPE_PATTERNS = [
    (re.compile(r"SyntaxError", re.IGNORECASE), "SyntaxError"),
    (re.compile(r"TypeError", re.IGNORECASE), "TypeError"),
    (re.compile(r"ReferenceError", re.IGNORECASE), "ReferenceError"),
    (re.compile(r"ModuleNotFoundError", re.IGNORECASE), "ModuleNotFoundError"),
    (re.compile(r"ImportError", re.IGNORECASE), "ImportError"),
    (re.compile(r"Cannot find module", re.IGNORECASE), "ModuleNotFound"),
    (re.compile(r"Failed to load PostCSS config", re.IGNORECASE), "PostCSSConfigError"),
    (re.compile(r"is not valid JSON", re.IGNORECASE), "JSONParseError"),
    (re.compile(r"Unexpected token", re.IGNORECASE), "JSONParseError"),
    (re.compile(r"ERR_MODULE_NOT_FOUND", re.IGNORECASE), "ModuleNotFound"),
    (re.compile(r"TS\d{4}", re.IGNORECASE), "TypeScriptError"),
    (re.compile(r"ENOENT", re.IGNORECASE), "FileNotFound"),
    (re.compile(r"Cannot resolve", re.IGNORECASE), "ResolutionError"),
]

DIAGNOSTIC_SYSTEM_PROMPT = """You are a build error diagnostic expert for a Vite + React + Electron frontend and a Python FastAPI backend.

You are given:
1. The original spec (what was intended)
2. The POT execution results (what files were changed)
3. The build error output (what went wrong)

YOUR TASK: Diagnose the root cause and generate a fix.

RESPOND WITH ONLY A VALID JSON OBJECT matching this schema:
{{
  "diagnosis": "One sentence describing what went wrong",
  "root_cause": "encoding|syntax|import|config|dependency|type_error|other",
  "confidence": 0.0-1.0,
  "fixes": [
    {{
      "fix_type": "rewrite_file|run_command|revert_file",
      "file_path": "<use same absolute path from error output>",
      "content": "Full corrected file content (for rewrite_file only)",
      "command": "npm install (for run_command only)",
      "rationale": "Why this fix addresses the root cause"
    }}
  ]
}}

RULES:
1. fix_type must be one of: rewrite_file, run_command, revert_file
2. For rewrite_file: provide the COMPLETE corrected file content (not a diff)
3. For run_command: ONLY these commands are allowed: npm install, npm ci, npx tsc, npx vite build, python -m py_compile, pip install
4. For revert_file: provide the file_path to revert (content from POT executor backup)
5. All file paths must use the SAME absolute paths shown in the error output and modified files list (do NOT change drive letters or path prefixes)
6. Do NOT suggest commands that delete files, modify system config, or access the network beyond npm
7. Focus on the MINIMAL fix needed — do not rewrite files that weren't part of the error
8. If the error mentions missing node_modules, suggest run_command with "npm install"
9. Output ONLY JSON — no markdown, no explanations outside the JSON
10. If the error is a UTF-8 BOM corruption (unexpected token at start of JSON), rewrite the affected file with clean UTF-8 content (no BOM)
"""

DIAGNOSTIC_USER_PROMPT = """## Build Error Diagnostic

### Spec Intent
{spec_summary}

### Files Modified by POT Execution
{modified_files_summary}

### Build Error Output
```
{build_error_output}
```

### Build Details
- Project Type: {project_type}
- Build Command: {build_command}
- Exit Code: {exit_code}
- Error Type: {error_type}
- Affected Files: {affected_files}

### Fix Attempt
This is fix attempt {attempt} of {max_attempts}.
{previous_fix_summary}

Diagnose the root cause and provide the minimal fix."""
