# FILE: app/orchestrator/extraction_binding.py
"""
Extraction Binding — Inject enrichment source extractions into architecture content.

v5.26 (2026-02-17): New module.

When the implementer generates code for a segment, it needs the EXACT source
code of the functions it's extracting from the monolith. The enrichment stage
(Stage 4B) already AST-parses the monolith, assigns functions to segments, and
saves per-segment `enrichment.json` with full function bodies in `source_extract`.

This module:
1. Loads the enrichment data for a segment
2. Formats the source extractions as a markdown section
3. Injects that section into the architecture content before it reaches
   the Overwatcher/Implementer

The result: the implementer gets the exact functions to transplant, with clear
instructions not to rewrite them. No searching through 2471-line monoliths.

For facade segments, builds an export map from all completed siblings'
enrichment data so the facade knows exactly what to import and re-export.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BUILD_ID = "2026-02-17-v1.0-extraction-binding"
print(f"[EXTRACTION_BINDING_LOADED] BUILD_ID={BUILD_ID}")


def load_segment_enrichment(
    job_dir_path: str,
    segment_id: str,
) -> Optional[Dict[str, Any]]:
    """Load cached enrichment.json for a segment.

    Returns the full enrichment dict or None if not available.
    """
    path = os.path.join(job_dir_path, "segments", segment_id, "enrichment.json")
    if not os.path.isfile(path):
        logger.debug("[extraction_binding] No enrichment for %s", segment_id)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("[extraction_binding] Failed to load %s: %s", path, e)
        return None


def build_extraction_block(
    enrichment: Dict[str, Any],
    segment_id: str,
) -> str:
    """Build a markdown block containing the exact source code to transplant.

    This block is injected into the architecture content so the implementer
    has the precise function bodies, imports, and constants for this segment.
    """
    source_extract = enrichment.get("source_extract", {})
    if not source_extract:
        logger.info("[extraction_binding] No source_extract for %s", segment_id)
        return ""

    parts: List[str] = []

    parts.append("## EXTRACTION BINDING — Source Code to Transplant (v5.26)")
    parts.append("")
    parts.append(
        "The following functions were AST-extracted from the monolith source file. "
        "These are the EXACT implementations to place in this segment's target file. "
        "**Copy them verbatim** — preserve all logic, variable names, docstrings, "
        "and error handling. Only update import paths to reflect the new package "
        "structure (e.g. `from app.orchestrator.segment_loop._constants import ...`)."
    )
    parts.append("")
    parts.append(
        "**DO NOT** rewrite, simplify, optimise, or reimagine these functions. "
        "The goal is a faithful extraction — the code must behave identically "
        "to the monolith version."
    )
    parts.append("")

    # Emit each function body
    for func_name, func_body in source_extract.items():
        parts.append(f"### `{func_name}`")
        parts.append("")
        parts.append("```python")
        parts.append(func_body)
        parts.append("```")
        parts.append("")

    # Also include import guidance from enrichment
    enrichment_imports = enrichment.get("imports", [])
    if enrichment_imports:
        parts.append("### Required Imports (from monolith)")
        parts.append("")
        parts.append(
            "These import lines were used by the functions above in the monolith. "
            "Adapt paths to the new package structure but preserve the imported symbols."
        )
        parts.append("")
        parts.append("```python")
        for imp in enrichment_imports:
            parts.append(imp)
        parts.append("```")
        parts.append("")

    # Constants assigned to this segment
    enrichment_constants = enrichment.get("constants", [])
    if enrichment_constants:
        parts.append("### Constants (assigned to this segment)")
        parts.append("")
        for const in enrichment_constants:
            if isinstance(const, dict):
                name = const.get("name", "")
                code = const.get("code", const.get("body", ""))
                if code:
                    parts.append(f"```python\n{code}\n```")
                elif name:
                    parts.append(f"- `{name}`")
            else:
                parts.append(f"- `{const}`")
        parts.append("")

    parts.append("---")
    parts.append("")

    return "\n".join(parts)


def build_facade_export_map(
    job_dir_path: str,
    manifest_segments: list,
    facade_segment_id: str,
) -> str:
    """Build a complete export map for the facade segment.

    Reads enrichment from ALL sibling segments and collects their exported
    symbols. Returns a markdown block that tells the facade exactly what
    to import from each submodule and re-export.
    """
    parts: List[str] = []

    parts.append("## FACADE EXPORT MAP — Complete Re-Export Specification (v5.26)")
    parts.append("")
    parts.append(
        "The following is a definitive map of every symbol that must be "
        "imported from submodules and re-exported by the facade `__init__.py`. "
        "This was built from the actual enrichment data of completed segments."
    )
    parts.append("")

    total_exports = 0

    for seg in manifest_segments:
        seg_id = seg.segment_id if hasattr(seg, 'segment_id') else seg.get('segment_id', '')
        if seg_id == facade_segment_id:
            continue

        enrichment = load_segment_enrichment(job_dir_path, seg_id)
        if not enrichment:
            continue

        # Get the file scope for this segment
        file_scope = seg.file_scope if hasattr(seg, 'file_scope') else seg.get('file_scope', [])

        # Collect function names and signatures
        functions = enrichment.get("functions", [])
        constants = enrichment.get("constants", [])
        classes = enrichment.get("classes", [])

        if not functions and not constants and not classes:
            continue

        # Determine the import path from the file scope
        import_module = ""
        if file_scope:
            # Convert file path to module path
            # e.g. app/orchestrator/segment_loop/_dependency.py -> app.orchestrator.segment_loop._dependency
            first_file = file_scope[0].replace("\\", "/")
            if first_file.endswith(".py"):
                first_file = first_file[:-3]
            import_module = first_file.replace("/", ".")

        parts.append(f"### From `{import_module}` ({seg_id})")
        parts.append("")

        if functions:
            parts.append("**Functions:**")
            for func in functions:
                name = func.get("name", "") if isinstance(func, dict) else str(func)
                sig = func.get("signature", "") if isinstance(func, dict) else ""
                if sig:
                    parts.append(f"- `{sig}`")
                elif name:
                    parts.append(f"- `{name}`")
                total_exports += 1
            parts.append("")

        if constants:
            parts.append("**Constants:**")
            for const in constants:
                name = const.get("name", "") if isinstance(const, dict) else str(const)
                parts.append(f"- `{name}`")
                total_exports += 1
            parts.append("")

        if classes:
            parts.append("**Classes:**")
            for cls in classes:
                name = cls.get("name", "") if isinstance(cls, dict) else str(cls)
                parts.append(f"- `{name}`")
                total_exports += 1
            parts.append("")

    if total_exports == 0:
        logger.info("[extraction_binding] No exports found for facade %s", facade_segment_id)
        return ""

    parts.append(f"**Total symbols to re-export: {total_exports}**")
    parts.append("")
    parts.append(
        "The facade `__init__.py` must import and re-export ALL of the above symbols "
        "so that existing code using `from app.orchestrator.segment_loop import X` "
        "continues to work without modification."
    )
    parts.append("")
    parts.append("---")
    parts.append("")

    logger.info(
        "[extraction_binding] Facade export map for %s: %d symbols from %d segments",
        facade_segment_id, total_exports, len(manifest_segments) - 1,
    )

    return "\n".join(parts)


def inject_extraction_into_architecture(
    architecture_content: str,
    extraction_block: str,
) -> str:
    """Inject the extraction block into architecture content.

    Appends after the File Inventory section (if found) so the implementer
    sees: architecture design → file inventory → exact source code to use.
    """
    if not extraction_block:
        return architecture_content

    import re

    # Try to insert after File Inventory section.
    # The File Inventory contains sub-headings (### New Files, ### Modified Files)
    # so we must match the ENTIRE section including sub-headings, not just
    # the ## File Inventory heading. We look for ## File Inventory and consume
    # everything up to the next ## (level-2) heading or the separator ---.
    # v1.1 FIX: Previous lazy match (.*?) stopped at ### sub-headings,
    # inserting the block between ## File Inventory and ### New Files,
    # which broke the file inventory parser.
    file_inv_pattern = re.compile(
        r'(^#{1,2}\s*.*[Ff]ile\s*[Ii]nventory.*?)(?=^#{1,2}\s[^#]|^---\s*$|\Z)',
        re.MULTILINE | re.DOTALL,
    )
    match = file_inv_pattern.search(architecture_content)

    if match:
        insert_pos = match.end()
        return (
            architecture_content[:insert_pos]
            + "\n\n"
            + extraction_block
            + architecture_content[insert_pos:]
        )

    # Fallback: insert before ## Detailed File Specifications if it exists
    detail_pattern = re.compile(
        r'^#{1,2}\s+Detailed\s+File',
        re.MULTILINE | re.IGNORECASE,
    )
    detail_match = detail_pattern.search(architecture_content)
    if detail_match:
        insert_pos = detail_match.start()
        return (
            architecture_content[:insert_pos]
            + extraction_block
            + "\n\n"
            + architecture_content[insert_pos:]
        )

    # No File Inventory found — append at end
    return architecture_content + "\n\n" + extraction_block
