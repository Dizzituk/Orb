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
    emit(f"\n🏗️ Scaffold complete: {create_count} skeletons written, "
         f"{modify_count} MODIFY files queued ({result.duration_seconds:.1f}s)")

    return result


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
