# FILE: app/pipeline_v2/gradle_daemon.py
"""
Gradle Daemon Pre-Warm & Async Build Manager.

Manages Gradle daemon lifecycle for Android builds so the pipeline
doesn't block on cold compilation. Two capabilities:

1. Pre-warm: Start the Gradle daemon at pipeline start so it's ready
   when the first compilation happens.
2. Async build: Fire-and-forget build with polling, so GPT can
   continue working while Gradle compiles.

ENVIRONMENT RULES:
- Gradle runs on the HOST, never in the sandbox.
- Android projects live under D:/Astra Android Folder on the host.
- ASTRA own repos (D:/Orb, D:/orb-desktop) are never Gradle targets.

v1.0 (2026-04-06): Initial implementation.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile

logger = logging.getLogger(__name__)

# Android Studio bundled JDK
_JAVA_HOME = r"C:\Program Files\Android\Android Studio\jbr"


@dataclass
class BuildResult:
    """Result of an async Gradle build."""
    job_key: str
    status: str  # "running", "success", "failed", "timeout"
    returncode: int = -1
    stdout: str = ""
    stderr: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    elapsed_sec: float = 0.0


# In-memory registry of running/completed builds
_builds: Dict[str, BuildResult] = {}
_build_tasks: Dict[str, asyncio.Task] = {}


def needs_gradle(profile: Optional["BuildTargetProfile"]) -> bool:
    """Check if this profile requires Gradle (i.e. is an Android/Kotlin build)."""
    if profile is None:
        return False
    return profile.build_system == "gradle"


async def pre_warm_daemon(profile: "BuildTargetProfile") -> bool:
    """Start the Gradle daemon in the background so it's warm for builds.

    This is a fire-and-forget operation. The daemon starts downloading
    deps and initialising in the background while the pipeline does
    its deterministic scaffold work. By the time GPT needs to compile,
    the daemon should be ready.

    Returns True if the daemon was started, False if not needed or failed.
    """
    if not needs_gradle(profile):
        return False

    project_root = profile.project_root.replace("/", "\\")
    gradlew = os.path.join(project_root, "gradlew.bat")

    if not os.path.isfile(gradlew):
        logger.warning(
            "[gradle_daemon] No gradlew.bat found at %s — skipping pre-warm",
            gradlew,
        )
        return False

    cmd = (
        f'$env:JAVA_HOME = "{_JAVA_HOME}" ; '
        f'cd "{project_root}" ; '
        f'.\\gradlew.bat --daemon --status 2>&1'
    )

    logger.info(
        "[gradle_daemon] Pre-warming Gradle daemon for %s (%s)",
        profile.project_name, project_root,
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            "powershell", "-Command", cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Don't await — let it run in background
        # The daemon persists after this process exits
        asyncio.create_task(_daemon_watcher(proc, profile.project_name))
        return True
    except Exception as e:
        logger.error("[gradle_daemon] Failed to start daemon: %s", e)
        return False


async def _daemon_watcher(proc: asyncio.subprocess.Process, name: str) -> None:
    """Watch the daemon start process and log when it completes."""
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        rc = proc.returncode or 0
        out = stdout.decode("utf-8", errors="replace") if stdout else ""
        if rc == 0:
            logger.info("[gradle_daemon] Daemon ready for %s", name)
        else:
            logger.warning(
                "[gradle_daemon] Daemon start returned %d for %s: %s",
                rc, name, out[:200],
            )
    except asyncio.TimeoutError:
        logger.warning("[gradle_daemon] Daemon start timed out for %s", name)
    except Exception as e:
        logger.error("[gradle_daemon] Daemon watcher error: %s", e)


async def start_async_build(
    profile: "BuildTargetProfile",
    build_cmd: Optional[str] = None,
    job_key: Optional[str] = None,
    timeout_sec: int = 600,
) -> str:
    """Start a Gradle build in the background. Returns a job_key for polling.

    The build runs as a subprocess on the host. The caller can continue
    doing other work and poll with check_build() to see if it's done.

    Args:
        profile: The build target profile.
        build_cmd: Override command. Defaults to profile.syntax_check_cmd.
        job_key: Custom key. Auto-generated if None.
        timeout_sec: Max time before the build is killed.

    Returns:
        The job_key to use with check_build() and get_build_result().
    """
    cmd = build_cmd or profile.syntax_check_cmd
    if not cmd:
        cmd = profile.build_cmd

    if job_key is None:
        job_key = f"gradle_{profile.project_id}_{int(time.time())}"

    result = BuildResult(
        job_key=job_key,
        status="running",
        started_at=time.time(),
    )
    _builds[job_key] = result

    logger.info(
        "[gradle_daemon] Starting async build: %s (timeout=%ds)",
        job_key, timeout_sec,
    )

    task = asyncio.create_task(
        _run_build(job_key, cmd, timeout_sec)
    )
    _build_tasks[job_key] = task

    return job_key


async def _run_build(job_key: str, cmd: str, timeout_sec: int) -> None:
    """Execute the build subprocess and update the result when done."""
    result = _builds.get(job_key)
    if not result:
        return

    try:
        proc = await asyncio.create_subprocess_exec(
            "powershell", "-Command", cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_sec
        )

        result.returncode = proc.returncode or 0
        result.stdout = (
            stdout.decode("utf-8", errors="replace") if stdout else ""
        )
        result.stderr = (
            stderr.decode("utf-8", errors="replace") if stderr else ""
        )
        result.status = "success" if result.returncode == 0 else "failed"
        result.finished_at = time.time()
        result.elapsed_sec = result.finished_at - result.started_at

        logger.info(
            "[gradle_daemon] Build %s %s in %.1fs (rc=%d)",
            job_key, result.status, result.elapsed_sec, result.returncode,
        )

    except asyncio.TimeoutError:
        result.status = "timeout"
        result.finished_at = time.time()
        result.elapsed_sec = result.finished_at - result.started_at
        logger.warning(
            "[gradle_daemon] Build %s timed out after %ds",
            job_key, timeout_sec,
        )
    except Exception as e:
        result.status = "failed"
        result.stderr = str(e)
        result.finished_at = time.time()
        result.elapsed_sec = result.finished_at - result.started_at
        logger.error("[gradle_daemon] Build %s crashed: %s", job_key, e)


def check_build(job_key: str) -> Optional[BuildResult]:
    """Check the status of an async build. Returns None if key not found."""
    return _builds.get(job_key)


def is_build_running(job_key: str) -> bool:
    """Check if a build is still running."""
    result = _builds.get(job_key)
    return result is not None and result.status == "running"


def get_build_result(job_key: str) -> Optional[BuildResult]:
    """Get the final result of a completed build. Returns None if still running."""
    result = _builds.get(job_key)
    if result and result.status != "running":
        return result
    return None


def clear_build(job_key: str) -> None:
    """Remove a build from the registry."""
    _builds.pop(job_key, None)
    _build_tasks.pop(job_key, None)


def clear_all_builds() -> None:
    """Clear all build records."""
    _builds.clear()
    _build_tasks.clear()