# FILE: app/pipeline_v2/scaffold_engine.py
"""
ASTRA v2.2 Scaffold Engine — deterministic file generator.

Reads the SpecGate manifest and produces skeleton files:
- Directory structure
- File headers with imports
- Class/function signatures with docstrings
- Interface definitions and type stubs
- Route registration stubs
- TODO markers where the Agentic Builder needs to fill logic

No LLM calls. No tokens spent. Pure deterministic output.
80-90% of each file is laid down here.

v1.0 (2026-03-07): Initial implementation for ASTRA v2.1.
v2.0 (2026-03-10): Multi-project targeting — Kotlin scaffold support,
    profile-aware path resolution.
v2.3 (2026-04-18): HARD NON-DESTRUCTIVE GUARD — before writing any
    CREATE-marked file, check if it already exists on the host
    filesystem. If it does, demote is_new → False and queue as MODIFY
    for the agentic builder. Prevents the scaffold engine from
    clobbering pre-existing host files when SpecGate's grounding
    data is incomplete (e.g. sandbox offline during inspection).
    This honours the 'never destroy host files without explicit
    confirmation' policy at the pipeline level, not just the agent
    level.
v2.4 (2026-04-18): BASENAME-AWARE PATH REDIRECTION — when the spec
    emits a bare filename (e.g. "DriverCopilotDatabase.kt") or a path
    that doesn't match the real on-disk location, walk the target
    project's file tree once to build a basename index. If the spec's
    basename matches exactly one existing file, redirect the write to
    that path instead of the default resolver's guess. Prevents the
    scaffold engine from creating bogus duplicate files in package-
    root positions when the real file already lives in a subpackage.
    Falls back to the default path on zero matches or ambiguous
    (multiple) matches. Combined with the v2.3 guard, an existing file
    gets redirected here and then preserved — never overwritten.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from app.pipeline_v2.models import ScaffoldFile, ScaffoldResult

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile

logger = logging.getLogger(__name__)


async def run_scaffold_engine(
    manifest: Dict[str, Any],
    spec: Dict[str, Any],
    job_dir: str,
    on_progress: Optional[Callable[[str], None]] = None,
    profile: Optional["BuildTargetProfile"] = None,
) -> ScaffoldResult:
    """Run the Scaffold Engine to produce deterministic file skeletons.

    Args:
        manifest: Segment manifest from SpecGate.
        spec: The verified spec content.
        job_dir: Job directory for saving artifacts.
        on_progress: Progress callback for UI updates.
        profile: Build target profile (determines language, paths).

    Returns:
        ScaffoldResult with all skeleton files.
    """
    t_start = time.time()
    emit = on_progress or (lambda msg: None)
    result = ScaffoldResult()

    lang = profile.language if profile else "python"
    emit(f"🏗️ Scaffold Engine: Generating {lang} file skeletons...")

    segments = manifest.get("segments", [])
    skeleton_contract = _load_skeleton_contract(job_dir)
    # Collect all files across segments
    all_files: List[Dict[str, Any]] = []
    for seg in segments:
        seg_id = seg.get("segment_id", "")
        file_scope = seg.get("file_scope", [])
        requirements = seg.get("requirements", [])
        grounding = seg.get("grounding_data", {})
        # v2.1 (2026-04-12): Phase 1 Job 15 — capture per-segment target_id
        # so scaffold writes route to the correct repo for multi-target jobs.
        seg_target_id = seg.get("target_id")

        for fp in file_scope:
            is_new = _is_create_file(fp, grounding)
            all_files.append({
                "path": fp,
                "segment_id": seg_id,
                "target_id": seg_target_id,
                "is_new": is_new,
                "requirements": requirements,
                "grounding": grounding,
            })

    emit(f"   Files to scaffold: {len(all_files)}")

    # Generate skeleton for each file
    from app.pipeline_v2.sandbox_tools import write_file as sandbox_write

    # v2.3 (2026-04-18): track how many CREATE writes were demoted to MODIFY
    # because the file already existed on disk. Surfaced in the final summary.
    demoted_existing = 0

    # v2.4 (2026-04-18): per-run cache of project basename indices, keyed by
    # target_id. Built lazily on first use per target so we only walk each
    # project tree once per scaffold run. For jobs with a single target this
    # is a single walk; multi-target jobs get one walk per touched target.
    basename_index_cache: Dict[str, Dict[str, List[str]]] = {}
    redirected_count = 0

    for file_info in all_files:
        fp = file_info["path"]
        is_new = file_info["is_new"]

        # v2.1 (2026-04-12): Phase 1 Job 15 — resolve per-file profile from
        # the segment's target_id. Falls back to the passed-in profile if
        # the segment has no target (single-target or legacy jobs).
        file_profile = profile
        _tid = file_info.get("target_id")
        if _tid:
            try:
                from app.pipeline_v2.target_registry import get_profile
                _resolved = get_profile(_tid)
                if _resolved is not None:
                    file_profile = _resolved
            except Exception as _pe:
                logger.debug("[scaffold_engine] profile lookup failed for target_id=%s: %s", _tid, _pe)

        # v2.4 (2026-04-18): BASENAME-AWARE PATH REDIRECTION.
        # Before anything else, check if the spec path's basename already
        # exists somewhere in the target project tree. If there's a single
        # unambiguous match, redirect `fp` to that existing path. This
        # prevents the scaffold from creating bogus duplicates like
        # drivercopilot/DriverCopilotDatabase.kt at the package root when
        # the real file lives at drivercopilot/data/DriverCopilotDatabase.kt.
        # Zero matches or multiple matches → no redirect, use fp as-is.
        if is_new and file_profile is not None:
            redirected_fp = _maybe_redirect_to_existing_path(
                fp, file_profile, basename_index_cache,
            )
            if redirected_fp is not None and redirected_fp != fp:
                logger.info(
                    "[scaffold_engine] v2.4 PATH-REDIRECT: spec said '%s' → "
                    "existing path '%s' (basename match in target tree)",
                    fp, redirected_fp,
                )
                emit(f"   📍 [REDIRECT] {fp} → {redirected_fp} (matched existing file)")
                fp = redirected_fp
                redirected_count += 1

        # v2.3 (2026-04-18): HARD NON-DESTRUCTIVE GUARD.
        # If grounding said this file is new but it already exists on disk,
        # demote to MODIFY rather than overwriting it. This protects host
        # files when SpecGate's grounding data is incomplete (for example,
        # when the sandbox is offline during inspection and verified_files
        # didn't populate). The agentic builder owns MODIFY edits via its
        # read_file / write_file tool loop — that's the right place for
        # changes to existing code, not blind scaffold overwrites.
        if is_new and _exists_on_host(fp, file_profile):
            is_new = False
            demoted_existing += 1
            abs_path = _resolve_for_log(fp, file_profile)
            logger.info(
                "[scaffold_engine] v2.3 DEMOTED to MODIFY — file exists on host: %s",
                abs_path,
            )
            emit(f"   🛡️ [SKIP-EXISTS] {fp} — already on disk, queued for agentic builder")

        if is_new:
            skeleton = _generate_skeleton(
                fp,
                file_info["requirements"],
                skeleton_contract,
                spec,
                file_profile,
            )
            scaffold_file = ScaffoldFile(
                path=fp,
                content=skeleton,
                is_new=True,
                char_count=len(skeleton),
            )
            result.files.append(scaffold_file)

            # Write to sandbox using per-file profile (multi-target aware)
            ok = await sandbox_write(fp, skeleton, profile=file_profile)
            status = "✅" if ok else "❌"
            _tgt = file_profile.project_id if file_profile else "?"
            emit(f"   {status} [CREATE] {fp} -> {_tgt} ({len(skeleton):,} chars)")
        else:
            # MODIFY files: don't write a skeleton, just record them
            scaffold_file = ScaffoldFile(
                path=fp,
                content="",
                is_new=False,
                char_count=0,
            )
            result.files.append(scaffold_file)
            # Only emit the explicit MODIFY marker if this wasn't already
            # reported as a SKIP-EXISTS above (avoid duplicate log lines).
            if file_info["is_new"]:
                pass  # already emitted SKIP-EXISTS above
            else:
                emit(f"   📝 [MODIFY] {fp}")

    result.total_files = len(all_files)
    result.duration_seconds = time.time() - t_start

    # v2.1: Copy Gradle wrapper for Android greenfield projects
    if profile and profile.language == "kotlin":
        try:
            from app.pipeline_v2.scaffolds.android_config_scaffolds import copy_gradle_wrapper
            if copy_gradle_wrapper(profile.project_root.replace('/', os.sep)):
                emit("   Copied Gradle wrapper (gradlew, jar, properties)")
        except Exception as _gw_err:
            emit(f"   Gradle wrapper copy failed: {_gw_err}")
    create_count = sum(1 for f in result.files if f.is_new)
    modify_count = sum(1 for f in result.files if not f.is_new)
    summary_bits = []
    if demoted_existing:
        summary_bits.append(f"{demoted_existing} existing file(s) preserved")
    if redirected_count:
        summary_bits.append(f"{redirected_count} path(s) redirected to existing locations")
    summary_tail = (", " + ", ".join(summary_bits)) if summary_bits else ""
    emit(f"\n🏗️ Scaffold complete: {create_count} skeletons written, "
         f"{modify_count} MODIFY files queued{summary_tail} "
         f"({result.duration_seconds:.1f}s)")

    return result


# ══════════════════════════════════════════════════════════════════
# v2.3 Non-destructive guard helpers
# ══════════════════════════════════════════════════════════════════

def _exists_on_host(
    file_path: str,
    profile: Optional["BuildTargetProfile"] = None,
) -> bool:
    """Return True if `file_path` (as scaffold would write it) already
    exists on the host filesystem.

    Resolves relative paths against the profile's project_root via
    sandbox_tools._resolve_path, then calls os.path.exists. Works for
    all target types because host-mode writes and sandbox-mode writes
    both land on host-visible disk (the sandbox mounts the host repo).

    Fails safe: any resolution or IO error returns False, so the
    scaffold engine will attempt the write normally (preserving
    existing behaviour rather than mysteriously skipping files).
    """
    try:
        from app.pipeline_v2.sandbox_tools import _resolve_path
        abs_path = _resolve_path(file_path, profile)
        return os.path.exists(abs_path.replace("/", os.sep))
    except Exception as e:
        logger.debug(
            "[scaffold_engine] _exists_on_host check failed for %s: %s — defaulting to False",
            file_path, e,
        )
        return False


def _resolve_for_log(
    file_path: str,
    profile: Optional["BuildTargetProfile"] = None,
) -> str:
    """Resolve a relative path to its absolute host path for log messages.

    Separate from _exists_on_host so log formatting stays resilient even
    if resolution throws.
    """
    try:
        from app.pipeline_v2.sandbox_tools import _resolve_path
        return _resolve_path(file_path, profile)
    except Exception:
        return file_path


# ══════════════════════════════════════════════════════════════════
# v2.4 Basename-aware path redirection
# ══════════════════════════════════════════════════════════════════

# Directories to skip when walking the project tree. These contain either
# generated artifacts, VCS metadata, or vendored dependencies that we never
# want to treat as scaffold targets.
_WALK_SKIP_DIRS = frozenset({
    ".git", ".gradle", ".idea", ".vscode", "build", ".build",
    "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache",
    "venv", ".venv", "env", ".env", "dist", "out", "target",
    "bin", "obj", ".next", ".nuxt", ".cache", "coverage",
})


def _build_project_basename_index(project_root: str) -> Dict[str, List[str]]:
    """Walk the project tree once and build a {basename: [relative_path, ...]}
    index.

    Paths are returned relative to project_root, forward-slash normalised.
    Well-known noise directories (build output, VCS, vendored deps) are
    skipped. Multiple files with the same basename are all kept in the
    list — the caller decides how to handle ambiguity.

    Returns an empty dict if the root doesn't exist or can't be read.
    """
    index: Dict[str, List[str]] = {}
    root_os = project_root.replace("/", os.sep)
    if not os.path.isdir(root_os):
        logger.debug(
            "[scaffold_engine] v2.4 basename index: root not a directory: %s",
            root_os,
        )
        return index

    try:
        for dirpath, dirnames, filenames in os.walk(root_os):
            # In-place prune to skip noise dirs. os.walk respects dirnames edits.
            dirnames[:] = [d for d in dirnames if d not in _WALK_SKIP_DIRS]
            for fname in filenames:
                abs_file = os.path.join(dirpath, fname)
                rel = os.path.relpath(abs_file, root_os).replace(os.sep, "/")
                index.setdefault(fname.lower(), []).append(rel)
    except Exception as e:
        logger.warning(
            "[scaffold_engine] v2.4 basename index walk failed for %s: %s",
            root_os, e,
        )
        return {}

    logger.debug(
        "[scaffold_engine] v2.4 basename index built for %s: %d unique basenames, %d total files",
        root_os, len(index), sum(len(v) for v in index.values()),
    )
    return index


def _maybe_redirect_to_existing_path(
    spec_path: str,
    profile: "BuildTargetProfile",
    basename_index_cache: Dict[str, Dict[str, List[str]]],
) -> Optional[str]:
    """If `spec_path` is an unrooted/ambiguous path and its basename matches
    exactly one existing file in the target project tree, return the
    existing relative path. Otherwise return None.

    Rules:
      - Absolute paths (drive letter prefix) are left alone — the caller
        clearly wanted that exact location.
      - Paths that already contain a directory separator are only
        redirected when the resolver's default would place them somewhere
        clearly wrong (checked by: does the spec path itself exist on
        disk relative to project root? if yes, no redirect needed; if no,
        try basename lookup).
      - Bare basenames (no slashes) always get basename lookup.
      - Multiple basename matches → None (ambiguous, caller keeps spec path).
      - Zero matches → None (truly new file, caller keeps spec path).
    """
    try:
        norm = (spec_path or "").replace("\\", "/").strip()
        if not norm:
            return None
        # Absolute path — trust the caller, don't redirect.
        if len(norm) > 1 and norm[1] == ":":
            return None

        basename = norm.rsplit("/", 1)[-1]
        if not basename or "." not in basename:
            # No usable basename (e.g. directory-ish path)
            return None

        project_root = (profile.project_root or "").replace("\\", "/").rstrip("/")
        if not project_root:
            return None

        # If the spec path already resolves to something that exists on disk,
        # no need to redirect — the path is already correct.
        try:
            candidate_abs = profile.resolve_path(norm).replace("/", os.sep)
            if os.path.exists(candidate_abs):
                return None
        except Exception:
            pass  # fall through to basename lookup

        # Build / fetch the basename index for this target, cached per run.
        cache_key = profile.project_id or project_root
        index = basename_index_cache.get(cache_key)
        if index is None:
            index = _build_project_basename_index(project_root)
            basename_index_cache[cache_key] = index

        matches = index.get(basename.lower(), [])
        if len(matches) == 0:
            return None  # truly new file
        if len(matches) > 1:
            logger.info(
                "[scaffold_engine] v2.4 basename '%s' ambiguous in %s — "
                "%d matches (%s) — no redirect applied",
                basename, profile.project_id, len(matches),
                ", ".join(matches[:4]) + ("…" if len(matches) > 4 else ""),
            )
            return None

        # Exactly one match — redirect.
        return matches[0]
    except Exception as e:
        logger.debug(
            "[scaffold_engine] v2.4 redirect helper failed for '%s': %s — no redirect",
            spec_path, e,
        )
        return None


# ═══════════════════════════════════════════════════════════════════
# Skeleton generation per file type
# ═══════════════════════════════════════════════════════════════════

def _generate_skeleton(
    file_path: str,
    requirements: List[str],
    skeleton_contract: Dict[str, Any],
    spec: Dict[str, Any],
    profile: Optional["BuildTargetProfile"] = None,
) -> str:
    """Generate a deterministic skeleton for a new file."""
    norm = file_path.replace("\\", "/")
    ext = os.path.splitext(norm)[1].lower()

    # v2.1: Check Android config files first (build.gradle.kts, AndroidManifest.xml, etc.)
    # These need full, compilable content — not stubs.
    if profile and profile.language == "kotlin":
        from app.pipeline_v2.scaffolds.android_config_scaffolds import generate_android_config
        config_content = generate_android_config(norm, requirements, profile)
        if config_content is not None:
            return config_content

    if ext == ".kt":
        from app.pipeline_v2.scaffolds.kotlin_scaffolds import generate_kotlin_skeleton
        return generate_kotlin_skeleton(norm, requirements, profile)
    elif ext == ".py":
        return _skeleton_python(norm, requirements, skeleton_contract)
    elif ext in (".tsx", ".ts", ".jsx", ".js"):
        return _skeleton_typescript(norm, requirements, skeleton_contract)
    elif ext == ".css":
        return _skeleton_css(norm, requirements)
    elif ext == ".xml" and "res/" in norm:
        return _skeleton_android_xml(norm, requirements, profile)
    elif ext == ".xml":
        # Non-res XML (like AndroidManifest) — try Android config scaffolds
        if profile and profile.language == "kotlin":
            from app.pipeline_v2.scaffolds.android_config_scaffolds import generate_android_config
            config_content = generate_android_config(norm, requirements, profile)
            if config_content is not None:
                return config_content
        return f'<?xml version="1.0" encoding="utf-8"?>\n<!-- {norm} — scaffold -->\n'
    else:
        return f"# Scaffold stub for {norm}\n# TODO: Implement\n"


# ═══════════════════════════════════════════════════════════════════
# Android XML skeleton (for res/xml/ configs etc.)
# ═══════════════════════════════════════════════════════════════════

def _skeleton_android_xml(
    path: str,
    requirements: List[str],
    profile: Optional["BuildTargetProfile"] = None,
) -> str:
    """Generate Android XML skeleton (resource files)."""
    basename = path.rsplit("/", 1)[-1]

    if "accessibility" in basename.lower():
        return (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<accessibility-service\n'
            '    xmlns:android="http://schemas.android.com/apk/res/android"\n'
            '    android:accessibilityEventTypes="typeAllMask"\n'
            '    android:accessibilityFeedbackType="feedbackGeneric"\n'
            '    android:canRetrieveWindowContent="true"\n'
            '    android:notificationTimeout="100"\n'
            '    android:description="@string/accessibility_service_description"\n'
            '    />\n'
            '<!-- TODO: Configure accessibility service from spec -->\n'
        )

    return (
        f'<?xml version="1.0" encoding="utf-8"?>\n'
        f'<!-- {basename} — auto-generated scaffold -->\n'
        f'<!-- TODO: Implement from spec -->\n'
        f'<resources>\n'
        f'</resources>\n'
    )


# ═══════════════════════════════════════════════════════════════════
# Python skeleton (unchanged from v2.1)
# ═══════════════════════════════════════════════════════════════════

def _skeleton_python(
    path: str,
    requirements: List[str],
    skeleton_contract: Dict[str, Any],
) -> str:
    """Generate Python skeleton with imports, classes, functions."""
    basename = path.rsplit("/", 1)[-1].replace(".py", "")
    module_path = path.replace("/", ".").replace(".py", "")

    lines = [
        f'"""',
        f'{basename} — auto-generated scaffold.',
        f'',
        f'Module: {module_path}',
        f'',
    ]

    if requirements:
        lines.append("Requirements:")
        for r in requirements:
            lines.append(f"  - {r}")

    lines.extend([
        f'"""',
        f'from __future__ import annotations',
        f'',
        f'import logging',
        f'from typing import Any, Dict, List, Optional',
        f'',
        f'logger = logging.getLogger(__name__)',
        f'',
    ])

    contract = skeleton_contract.get(path, {})
    exports = contract.get("exports", [])
    for export in exports:
        name = export.get("name", "unknown")
        kind = export.get("kind", "function")
        signature = export.get("signature", "")

        if kind == "class":
            lines.extend([
                f'class {name}:',
                f'    """TODO: Implement {name}."""',
                f'    pass',
                f'',
            ])
        elif kind == "function":
            sig = signature or f"def {name}() -> None"
            lines.extend([
                f'{sig}:',
                f'    """TODO: Implement {name}."""',
                f'    raise NotImplementedError("{name}")',
                f'',
            ])

    if not exports:
        lines.extend([
            f'# TODO: Implement {basename}',
            f'# Requirements from spec will guide the Agentic Builder',
            f'',
        ])

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# TypeScript skeleton (unchanged from v2.1)
# ═══════════════════════════════════════════════════════════════════

def _skeleton_typescript(
    path: str,
    requirements: List[str],
    skeleton_contract: Dict[str, Any],
) -> str:
    """Generate TypeScript/React skeleton."""
    basename = path.rsplit("/", 1)[-1]
    is_component = basename.endswith(".tsx") and basename[0].isupper()
    is_hook = basename.startswith("use")
    component_name = basename.replace(".tsx", "").replace(".ts", "")

    lines = [
        f'/**',
        f' * {component_name} — auto-generated scaffold.',
        f' *',
    ]
    if requirements:
        for r in requirements:
            lines.append(f' * - {r}')
    lines.extend([
        f' */',
        f'',
    ])

    if is_component:
        lines.extend([
            f"import React from 'react';",
            f'',
            f'interface {component_name}Props {{',
            f'  // TODO: Define props',
            f'}}',
            f'',
            f'export function {component_name}(props: {component_name}Props) {{',
            f'  // TODO: Implement {component_name}',
            f'  return (',
            f'    <div className="{_to_kebab(component_name)}">',
            f'      <p>{component_name} — scaffold placeholder</p>',
            f'    </div>',
            f'  );',
            f'}}',
            f'',
        ])
    elif is_hook:
        lines.extend([
            f"import {{ useState, useEffect, useCallback }} from 'react';",
            f'',
            f'export function {component_name}() {{',
            f'  // TODO: Implement {component_name}',
            f'  return {{}};',
            f'}}',
            f'',
        ])
    else:
        lines.extend([
            f'// TODO: Implement {component_name}',
            f'',
            f'export {{}};',
            f'',
        ])

    return "\n".join(lines)


def _skeleton_css(path: str, requirements: List[str]) -> str:
    """Generate CSS skeleton with class stubs."""
    basename = path.rsplit("/", 1)[-1].replace(".css", "")

    lines = [
        f'/* {basename} — auto-generated scaffold */',
        f'',
        f'.{_to_kebab(basename)} {{',
        f'  /* TODO: Implement styles */',
        f'}}',
        f'',
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _is_create_file(file_path: str, grounding: Dict) -> bool:
    """Determine if a file is CREATE (new) or MODIFY (existing)."""
    create_targets = grounding.get("create_targets", [])
    for ct in create_targets:
        if isinstance(ct, dict):
            ct_path = ct.get("path", "").replace("\\", "/")
        else:
            ct_path = str(ct).replace("\\", "/")
        if ct_path.lower() == file_path.replace("\\", "/").lower():
            return True

    verified = grounding.get("verified_files", [])
    for vf in verified:
        if isinstance(vf, dict):
            vf_path = vf.get("path", "").replace("\\", "/")
        else:
            vf_path = str(vf).replace("\\", "/")
        if vf_path.lower() == file_path.replace("\\", "/").lower():
            return False

    new_files = grounding.get("new_files", [])
    for nf in new_files:
        nf_path = (nf.get("path", "") if isinstance(nf, dict) else str(nf)).replace("\\", "/")
        if nf_path.lower() == file_path.replace("\\", "/").lower():
            return True

    return True


def _load_skeleton_contract(job_dir: str) -> Dict[str, Any]:
    """Load skeleton contract from job directory."""
    import json
    contract_path = os.path.join(job_dir, "segments", "skeleton_contract.json")
    if os.path.exists(contract_path):
        try:
            with open(contract_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("[scaffold] Could not load skeleton contract: %s", e)
    return {}


def _to_kebab(name: str) -> str:
    """Convert PascalCase or camelCase to kebab-case."""
    result = []
    for i, c in enumerate(name):
        if c.isupper() and i > 0:
            result.append("-")
        result.append(c.lower())
    return "".join(result)
