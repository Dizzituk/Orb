# FILE: app/llm/critical_pipeline/artifact_binding.py
"""
Artifact binding extraction and prompt building.

Extracts artifact output bindings from spec data, resolves path
templates, and builds the binding prompt section for architecture
generation.
"""

import logging
import os
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


# =============================================================================
# Path Template Resolution
# =============================================================================

def _resolve_path_template(template: str, context: Dict[str, Any]) -> str:
    """Resolve {placeholders} in path templates."""
    result = template
    for key, val in context.items():
        result = result.replace(f"{{{key}}}", str(val))
    return result


# =============================================================================
# Artifact Binding Extraction
# =============================================================================

def extract_artifact_bindings(
    spec_data: Dict[str, Any],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Extract artifact output bindings from spec data.

    Looks in multiple locations within the spec for output definitions:
    spec_data["outputs"], spec_data["artifacts"], spec_data["file_outputs"],
    and spec_data["metadata"]["artifacts"].

    Each binding includes: name, path, content_type, scope_constraints.
    """
    bindings: List[Dict[str, Any]] = []

    # Source 1: spec.outputs
    outputs = spec_data.get("outputs", [])
    if isinstance(outputs, list):
        for out in outputs:
            if isinstance(out, dict):
                binding = _extract_single_binding(out, context)
                if binding:
                    bindings.append(binding)

    # Source 2: spec.artifacts
    artifacts = spec_data.get("artifacts", [])
    if isinstance(artifacts, list):
        for art in artifacts:
            if isinstance(art, dict):
                binding = _extract_single_binding(art, context)
                if binding:
                    bindings.append(binding)

    # Source 3: spec.file_outputs
    file_outputs = spec_data.get("file_outputs", [])
    if isinstance(file_outputs, list):
        for fo in file_outputs:
            if isinstance(fo, dict):
                binding = _extract_single_binding(fo, context)
                if binding:
                    bindings.append(binding)

    # Source 4: spec.metadata.artifacts
    meta_artifacts = spec_data.get("metadata", {}).get("artifacts", [])
    if isinstance(meta_artifacts, list):
        for ma in meta_artifacts:
            if isinstance(ma, dict):
                binding = _extract_single_binding(ma, context)
                if binding:
                    # Avoid duplicates by path
                    if not any(b["path"] == binding["path"] for b in bindings):
                        bindings.append(binding)

    # Source 5: content_verbatim + location (simple single-file binding)
    content_verbatim = (
        spec_data.get("content_verbatim")
        or spec_data.get("context", {}).get("content_verbatim")
        or spec_data.get("metadata", {}).get("content_verbatim")
    )
    location = (
        spec_data.get("location")
        or spec_data.get("context", {}).get("location")
        or spec_data.get("metadata", {}).get("location")
    )

    if content_verbatim and location:
        resolved_location = _resolve_path_template(location, context)
        if not any(b["path"] == resolved_location for b in bindings):
            bindings.append({
                "name": os.path.basename(resolved_location),
                "path": resolved_location,
                "content_type": _infer_content_type(resolved_location),
                "content_verbatim": content_verbatim,
                "scope_constraints": (
                    spec_data.get("scope_constraints")
                    or spec_data.get("context", {}).get("scope_constraints")
                    or spec_data.get("metadata", {}).get("scope_constraints")
                    or []
                ),
            })

    if bindings:
        logger.info(
            "[artifact_binding] Extracted %d binding(s): %s",
            len(bindings),
            [b["path"] for b in bindings],
        )

    return bindings


def _extract_single_binding(
    item: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Extract a single artifact binding from a dict."""
    name = item.get("name", item.get("filename", ""))
    path_template = item.get("path", item.get("location", item.get("target", "")))

    if not path_template:
        return None

    resolved_path = _resolve_path_template(path_template, context)

    return {
        "name": name or os.path.basename(resolved_path),
        "path": resolved_path,
        "content_type": item.get(
            "content_type",
            item.get("type", _infer_content_type(resolved_path)),
        ),
        "content_verbatim": item.get("content_verbatim", ""),
        "scope_constraints": item.get("scope_constraints", []),
    }


# =============================================================================
# Content Type Inference
# =============================================================================

_EXT_TO_TYPE = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript-react",
    ".jsx": "javascript-react",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".txt": "text",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
    ".sh": "shell",
    ".bat": "batch",
    ".ps1": "powershell",
    ".toml": "toml",
    ".ini": "ini",
    ".env": "environment",
}


def _infer_content_type(filename: str) -> str:
    """Infer content type from file extension."""
    _, ext = os.path.splitext(filename.lower())
    return _EXT_TO_TYPE.get(ext, "text")


# =============================================================================
# Prompt Building
# =============================================================================

def build_artifact_binding_prompt(bindings: List[Dict[str, Any]]) -> str:
    """
    Build the artifact binding prompt section for architecture generation.

    Returns a string to append to the system prompt.
    """
    if not bindings:
        return ""

    sections = ["\n## Artifact Output Bindings\n"]
    sections.append(
        "The following concrete file paths have been resolved for this job. "
        "Your architecture MUST specify these exact paths in the File Inventory.\n"
    )

    for i, b in enumerate(bindings, 1):
        sections.append(f"### Binding {i}: `{b['name']}`")
        sections.append(f"- **Path:** `{b['path']}`")
        sections.append(f"- **Type:** {b['content_type']}")

        if b.get("content_verbatim"):
            sections.append(
                f'- **Content (VERBATIM):** "{b["content_verbatim"][:200]}"'
            )

        if b.get("scope_constraints"):
            sections.append("- **Scope Constraints:**")
            for sc in b["scope_constraints"]:
                sections.append(f"  - {sc}")

        sections.append("")

    sections.append(
        "Include ALL bindings above in your File Inventory section "
        "with their EXACT paths.\n"
    )

    return "\n".join(sections)
