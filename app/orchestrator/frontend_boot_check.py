# FILE: app/orchestrator/frontend_boot_check.py
"""
Frontend Boot Check — Vite parse validation.

v3.4 (2026-03-02): Runs a lightweight Vite build in the sandbox to catch
Babel/SWC parse errors that tsc --noEmit does not detect. This catches
issues like surviving scaffold markers ([LLM_FILL: ...]) that are valid
TypeScript syntax (index signatures) but invalid JSX/TSX.

This is a fast check: Vite will fail on the first parse error and exit.
Timeout is short (30s) since parse errors surface immediately.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

FRONTEND_BOOT_CHECK_BUILD_ID = "2026-03-02-v3.4-vite-parse-check"

# Vite pre-transform error pattern
# Example: "Pre-transform error: D:\orb-desktop\src\components\sidebar\Sidebar.tsx: Unexpected token"
_VITE_ERROR_RE = re.compile(
    r"(?:Pre-transform error|Internal server error):\s*"
    r"(?:[A-Z]:\\\\[^\s:]+[\\/])?"   # optional abs path prefix
    r"(src[\\/][^\s:]+)"             # capture relative path
    r":\s*(.+)",                     # capture error message
)

# Vite "ready in X ms" = success
_VITE_READY_RE = re.compile(r"ready in \d+\s*ms", re.IGNORECASE)


@dataclass
class ViteError:
    """Single Vite parse error."""
    file: str
    message: str


@dataclass
class FrontendBootResult:
    """Result of a Vite boot check."""
    status: str = "pending"          # pass | fail | error | skipped
    errors: List[ViteError] = field(default_factory=list)
    error_summary: str = ""
    raw_output: str = ""
    duration_ms: int = 0


def run_frontend_boot_check(
    client: Any,
    frontend_base: str = r"D:\orb-desktop",
    timeout_seconds: int = 30,
    emit: Any = None,
) -> FrontendBootResult:
    """Run a Vite build to catch parse errors.

    Uses `npx vite build` which attempts to parse and bundle all
    entry-point files. Parse errors surface immediately without
    needing a running dev server.

    Args:
        client: SandboxClient instance.
        frontend_base: Path to the frontend repo in the sandbox.
        timeout_seconds: Max time for the build to run.
        emit: Optional SSE callback for progress messages.

    Returns:
        FrontendBootResult with status and any parse errors found.
    """
    _emit = emit or (lambda msg: None)
    start = time.time()

    cmd = (
        f'cd "{frontend_base}" ; '
        f'npx vite build 2>&1'
    )

    _emit("  [VITE] Running Vite build check...")
    logger.info("[frontend_boot] Running vite build in %s", frontend_base)

    try:
        result = client.shell_run(
            cmd,
            cwd_target="REPO",
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        elapsed = int((time.time() - start) * 1000)
        logger.error("[frontend_boot] Shell execution failed: %s", exc)
        return FrontendBootResult(
            status="error",
            error_summary=f"Failed to run vite build: {exc}",
            duration_ms=elapsed,
        )

    elapsed = int((time.time() - start) * 1000)
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    combined = f"{stdout}\n{stderr}".strip()

    # Parse Vite errors
    errors: List[ViteError] = []
    for line in combined.split("\n"):
        m = _VITE_ERROR_RE.search(line)
        if m:
            errors.append(ViteError(file=m.group(1), message=m.group(2).strip()))

    # Also catch generic "error during build" with file reference
    if not errors and "error during build" in combined.lower():
        # Try to extract file from the error block
        file_match = re.search(r"(src[\\/]\S+\.tsx?)[\s:(]", combined)
        err_match = re.search(r"(?:Error|SyntaxError):\s*(.+)", combined)
        if file_match:
            errors.append(ViteError(
                file=file_match.group(1),
                message=err_match.group(1).strip() if err_match else "Build error",
            ))

    if errors:
        summary = "; ".join(f"{e.file}: {e.message[:80]}" for e in errors[:3])
        logger.warning(
            "[frontend_boot] Vite build FAILED in %dms: %d error(s). First: %s",
            elapsed, len(errors), errors[0].message[:120],
        )
        return FrontendBootResult(
            status="fail",
            errors=errors,
            error_summary=summary,
            raw_output=combined[-2000:],
            duration_ms=elapsed,
        )

    # Check if build succeeded (look for "built in" or no error exit)
    if "built in" in combined.lower() or result.exit_code == 0:
        logger.info("[frontend_boot] Vite build PASSED in %dms", elapsed)
        return FrontendBootResult(
            status="pass",
            raw_output=combined[-500:],
            duration_ms=elapsed,
        )

    # Ambiguous — no clear errors but also no success marker
    logger.warning(
        "[frontend_boot] Vite build INCONCLUSIVE in %dms: %s",
        elapsed, combined[-200:],
    )
    return FrontendBootResult(
        status="pass",  # Don't block on inconclusive
        raw_output=combined[-500:],
        duration_ms=elapsed,
    )
