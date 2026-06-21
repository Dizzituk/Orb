# FILE: app/pipeline_v2/scaffold_templates.py
# Purpose: Scaffold Engine deterministic skeleton generation per file type (template data).
# Called-by: app.pipeline_v2.scaffold_engine (shim)
# Depends-on: app.pipeline_v2.scaffolds.android_config_scaffolds (lazy), app.pipeline_v2.scaffolds.kotlin_scaffolds (lazy)
# Last-renovated: 2026-06-21
"""
Scaffold Engine deterministic skeleton generators.

Split out of scaffold_engine.py (BATCH 4) verbatim. The _generate_skeleton
dispatcher plus per-language skeleton templates (python/typescript/css/android-xml).
Pure content generation — never resolves host paths.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile


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


def _to_kebab(name: str) -> str:
    """Convert PascalCase or camelCase to kebab-case."""
    result = []
    for i, c in enumerate(name):
        if c.isupper() and i > 0:
            result.append("-")
        result.append(c.lower())
    return "".join(result)
