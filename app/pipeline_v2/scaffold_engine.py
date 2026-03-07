# FILE: app/pipeline_v2/scaffold_engine.py
"""
ASTRA v2.1 Scaffold Engine — deterministic file generator.

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
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from app.pipeline_v2.models import ScaffoldFile, ScaffoldResult

logger = logging.getLogger(__name__)


async def run_scaffold_engine(
    manifest: Dict[str, Any],
    spec: Dict[str, Any],
    job_dir: str,
    on_progress: Optional[Callable[[str], None]] = None,
) -> ScaffoldResult:
    """Run the Scaffold Engine to produce deterministic file skeletons.

    Args:
        manifest: Segment manifest from SpecGate.
        spec: The verified spec content.
        job_dir: Job directory for saving artifacts.
        on_progress: Progress callback for UI updates.

    Returns:
        ScaffoldResult with all skeleton files.
    """
    t_start = time.time()
    emit = on_progress or (lambda msg: None)
    result = ScaffoldResult()

    emit("🏗️ Scaffold Engine: Generating deterministic file skeletons...")

    segments = manifest.get("segments", [])
    skeleton_contract = _load_skeleton_contract(job_dir)

    # Collect all files across segments
    all_files: List[Dict[str, Any]] = []
    for seg in segments:
        seg_id = seg.get("segment_id", "")
        file_scope = seg.get("file_scope", [])
        requirements = seg.get("requirements", [])
        grounding = seg.get("grounding_data", {})

        for fp in file_scope:
            is_new = _is_create_file(fp, grounding)
            all_files.append({
                "path": fp,
                "segment_id": seg_id,
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

        if is_new:
            skeleton = _generate_skeleton(
                fp,
                file_info["requirements"],
                skeleton_contract,
                spec,
            )
            scaffold_file = ScaffoldFile(
                path=fp,
                content=skeleton,
                is_new=True,
                char_count=len(skeleton),
            )
            result.files.append(scaffold_file)

            # Write to sandbox
            ok = await sandbox_write(fp, skeleton)
            status = "✅" if ok else "❌"
            emit(f"   {status} [CREATE] {fp} ({len(skeleton):,} chars)")
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

    create_count = sum(1 for f in result.files if f.is_new)
    modify_count = sum(1 for f in result.files if not f.is_new)
    emit(f"\n🏗️ Scaffold complete: {create_count} skeletons written, "
         f"{modify_count} MODIFY files queued ({result.duration_seconds:.1f}s)")

    return result


# ---------------------------------------------------------------------------
# Skeleton generation per file type
# ---------------------------------------------------------------------------

def _generate_skeleton(
    file_path: str,
    requirements: List[str],
    skeleton_contract: Dict[str, Any],
    spec: Dict[str, Any],
) -> str:
    """Generate a deterministic skeleton for a new file."""
    norm = file_path.replace("\\", "/")
    ext = os.path.splitext(norm)[1].lower()

    if ext == ".py":
        return _skeleton_python(norm, requirements, skeleton_contract)
    elif ext in (".tsx", ".ts", ".jsx", ".js"):
        return _skeleton_typescript(norm, requirements, skeleton_contract)
    elif ext == ".css":
        return _skeleton_css(norm, requirements)
    else:
        return f"# Scaffold stub for {norm}\n# TODO: Implement\n"


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

    # Add interface stubs from skeleton contract
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

    # If no exports defined, add a placeholder
    if not exports:
        lines.extend([
            f'# TODO: Implement {basename}',
            f'# Requirements from spec will guide the Agentic Builder',
            f'',
        ])

    return "\n".join(lines)


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

    # If it's in verified_files, it exists → MODIFY
    verified = grounding.get("verified_files", [])
    for vf in verified:
        if isinstance(vf, dict):
            vf_path = vf.get("path", "").replace("\\", "/")
        else:
            vf_path = str(vf).replace("\\", "/")
        if vf_path.lower() == file_path.replace("\\", "/").lower():
            return False

    # Default: if grounding says it's new
    new_files = grounding.get("new_files", [])
    for nf in new_files:
        nf_path = (nf.get("path", "") if isinstance(nf, dict) else str(nf)).replace("\\", "/")
        if nf_path.lower() == file_path.replace("\\", "/").lower():
            return True

    return True  # Default to CREATE if unclear


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
