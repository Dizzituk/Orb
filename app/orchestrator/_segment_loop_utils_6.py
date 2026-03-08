from __future__ import annotations
import logging
import os
from app.orchestrator.segment_state import JobState
from app.pot_spec.grounded.segment_schemas import SegmentSpec, SegmentStatus
from datetime import datetime, timezone
from typing import Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


SEGMENT_LOOP_BUILD_ID = "2026-02-20-v6.0-implementation-compiler-wired"

def _find_latest_arch(seg_dir: str) -> Optional[str]:
    """
    Find the latest architecture version file in a segment's arch directory.

    Scans for arch_v{N}.md files and returns the path to the highest version.
    Used by both execution and cohesion checking to ensure consistent version
    resolution across the entire pipeline.

    v5.8: Replaces hardcoded v1/v2 checks and static v3/v2/v1 fallback lists.
    """
    arch_dir = os.path.join(seg_dir, "arch")
    if not os.path.isdir(arch_dir):
        return None

    max_version = 0
    max_path = None
    for fname in os.listdir(arch_dir):
        if fname.startswith("arch_v") and fname.endswith(".md"):
            try:
                v = int(fname.replace("arch_v", "").replace(".md", ""))
                if v > max_version:
                    max_version = v
                    max_path = os.path.join(arch_dir, fname)
            except ValueError:
                pass
    return max_path

def _clear_stale_arch_versions(seg_dir: str) -> int:
    """
    Remove stale autofix arch versions when a fresh regen produces arch_v1.md.

    When the Critical Pipeline regenerates an architecture (e.g. after cohesion
    regen feedback), it writes to arch_v1.md. Any existing v2, v3, etc. from
    previous cohesion autofixes are now stale and must be removed so that:
      1. The cohesion checker reads the fresh regen (not old autofix patches)
      2. The executor loads the correct version
      3. Version numbers don't drift upward across runs

    v5.8: Fixes the recurring import-logging cohesion loop where regen wrote
    a correct v1 but stale v2/v3 (without the fix) kept being loaded instead.

    Returns:
        Number of stale files removed.
    """
    arch_dir = os.path.join(seg_dir, "arch")
    if not os.path.isdir(arch_dir):
        return 0

    removed = 0
    for fname in os.listdir(arch_dir):
        if fname.startswith("arch_v") and fname.endswith(".md") and fname != "arch_v1.md":
            try:
                stale_path = os.path.join(arch_dir, fname)
                os.remove(stale_path)
                removed += 1
                logger.info("[SEGMENT_LOOP] v5.8 Removed stale arch: %s", stale_path)
            except OSError as e:
                logger.warning("[SEGMENT_LOOP] v5.8 Could not remove stale arch %s: %s", fname, e)
    return removed

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def is_segment_blocked(segment: SegmentSpec, state: JobState) -> bool:
    """
    Check if a segment should be BLOCKED (dependency FAILED or BLOCKED).

    Distinct from "can't execute yet" (dependency PENDING/IN_PROGRESS).
    """
    for dep_id in segment.dependencies:
        dep_state = state.segments.get(dep_id)
        if dep_state is None:
            continue
        if dep_state.status in (SegmentStatus.FAILED.value, SegmentStatus.BLOCKED.value):
            return True
    return False

def collect_segment_outputs(segment_id: str, job_dir_path: str) -> List[str]:
    """
    After implementation, collect what files were actually created/modified
    by this segment.

    Checks the segment's output directory for any files. Also checks the
    state for output_files recorded by the implementer.
    """
    output_dir = os.path.join(job_dir_path, "segments", segment_id, "output")
    output_files: List[str] = []

    if os.path.isdir(output_dir):
        for root, _dirs, files in os.walk(output_dir):
            for f in files:
                output_files.append(os.path.join(root, f))

    logger.info(
        "[SEGMENT_LOOP] Collected %d output file(s) for %s",
        len(output_files), segment_id,
    )
    return output_files

def _load_source_file_evidence(
    manifest: "SegmentManifest",
    project_roots: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    v3.1: Pre-load existing source files from the SANDBOX.

    Scans ALL segments' file_scope entries across the manifest, reads files
    from the SANDBOX (the only source of truth for the repo), and returns
    their content. This ensures every segment has access to the real source
    code — preventing the LLM from fabricating function signatures.

    Args:
        manifest: The full segment manifest
        project_roots: Project root directories to resolve relative paths.
                       Defaults to ["D:\\Orb", "D:\\orb-desktop"].

    Returns:
        Dict of {relative_path: file_content} for files found in sandbox.
        Content is capped at 250K chars per file.
    """
    if project_roots is None:
        project_roots = ["D:\\Orb", "D:\\orb-desktop"]

    MAX_SOURCE_CHARS = 250_000
    source_files: Dict[str, str] = {}
    seen_paths: set = set()

    # Get sandbox client once for all reads
    try:
        from app.sandbox.client import SandboxClient
        client = SandboxClient()
    except Exception as e:
        logger.warning("[segment_loop] v3.1 Cannot create sandbox client: %s", e)
        return source_files

    for seg in manifest.segments:
        for rel_path in seg.file_scope:
            normalised = rel_path.replace("/", os.sep).replace("\\", os.sep).lower()
            if normalised in seen_paths:
                continue
            seen_paths.add(normalised)

            # Try each project root in the SANDBOX
            for root in project_roots:
                abs_path = os.path.join(root, rel_path.replace("/", os.sep).replace("\\", os.sep))
                try:
                    cmd = f'Get-Content -Path "{abs_path}" -Raw -Encoding UTF8'
                    result = client.shell_run(
                        cmd=["powershell", "-NoProfile", "-Command", cmd],
                    )
                    stdout = result.get("stdout", "") if isinstance(result, dict) else getattr(result, "stdout", "")
                    if stdout and stdout.strip():
                        content = stdout[:MAX_SOURCE_CHARS]
                        source_files[rel_path] = content
                        logger.info(
                            "[segment_loop] v3.1 Source file pre-loaded from sandbox: %s (%d chars)",
                            rel_path, len(content),
                        )
                        break
                except Exception as exc:
                    logger.debug(
                        "[segment_loop] v3.1 Not found in sandbox at %s: %s",
                        abs_path, exc,
                    )
                    continue

    if source_files:
        print(
            f"[segment_loop] 📖 Pre-loaded {len(source_files)} source file(s) "
            f"from sandbox: {', '.join(source_files.keys())}"
        )

    return source_files

def _read_file_via_sandbox(abs_path: str) -> Optional[str]:
    """
    v3.1 Read a file from the SANDBOX filesystem via the sandbox bridge.

    During implementation, earlier segments write files to the sandbox —
    not the host. We MUST read from the sandbox to see those files.
    The sandbox is the ONLY source of truth. No host fallback.
    """
    try:
        from app.sandbox.client import SandboxClient
        client = SandboxClient()
        read_cmd = f'Get-Content -Path "{abs_path}" -Raw -Encoding UTF8'
        result = client.shell_run(
            cmd=["powershell", "-NoProfile", "-Command", read_cmd]
        )
        stdout = ""
        if isinstance(result, dict):
            stdout = result.get("stdout", "")
        else:
            stdout = getattr(result, "stdout", "")
        if stdout and stdout.strip():
            logger.debug(
                "[build_sibling_interfaces] v3.0 Read %d chars from sandbox: %s",
                len(stdout), abs_path,
            )
            return stdout
    except Exception as e:
        logger.debug(
            "[build_sibling_interfaces] v3.0 Sandbox read failed for %s: %s",
            abs_path, e,
        )

    logger.debug("[build_sibling_interfaces] v3.1 Empty/missing in sandbox: %s", abs_path)
    return None


def _build_sibling_interfaces(
    segment: "SegmentSpec",
    state: "JobState",
    job_dir_path: str,
) -> str:
    """
    v3.0 Deterministic sibling interface extraction via SANDBOX.

    For each completed upstream segment, reads the actual implemented files
    from the SANDBOX (not host) and extracts their public interface using AST.
    Returns formatted evidence text injected into the architecture/implementation
    prompt.

    v3.0: Reads via sandbox bridge so newly-created files from earlier
    segments are visible. Falls back to host for pre-existing files.
    """
    try:
        from app.pipeline_v2.checks.deterministic_checker import (
            extract_segment_interface,
            format_segment_interfaces,
        )
    except ImportError:
        logger.debug("[build_sibling_interfaces] deterministic_checker not available")
        return ""

    interfaces = []
    for dep_id in (segment.dependencies or []):
        dep_state = state.segments.get(dep_id)
        if dep_state is None or dep_state.status != SegmentStatus.COMPLETE.value:
            continue
        for fpath in (dep_state.output_files or []):
            if not fpath.endswith(".py"):
                continue
            abs_path = fpath
            if not os.path.isabs(fpath):
                abs_path = os.path.join("D:/Orb", fpath)
            file_content = _read_file_via_sandbox(abs_path)
            if file_content is None:
                logger.debug(
                    "[build_sibling_interfaces] Could not read %s from sandbox or host",
                    fpath,
                )
                continue
            try:
                iface = extract_segment_interface(fpath, file_content)
                interfaces.append(iface)
            except Exception as e:
                logger.debug(
                    "[build_sibling_interfaces] Interface extraction failed for %s: %s",
                    fpath, e,
                )

    if not interfaces:
        return ""

    formatted = format_segment_interfaces(interfaces)
    logger.info(
        "[build_sibling_interfaces] v3.0 Extracted %d sibling interfaces for %s (via sandbox)",
        len(interfaces), segment.segment_id,
    )
    return formatted
