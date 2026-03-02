# FILE: app/orchestrator/scaffold/integration.py
"""
Scaffold Engine Pipeline Integration.

Wires the scaffold engine into the ASTRA pipeline between architecture
approval and implementation. Controlled by feature flag ASTRA_SCAFFOLD_ENGINE.

Pipeline flow when enabled:
  Architecture Approved -> Scaffold Engine -> Implementer (fills only)

Pipeline flow when disabled (current default):
  Architecture Approved -> Implementer (full file generation)

v1.0 (2026-03-01): Batch 2 — feature-flagged integration.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from .arch_parser import parse_architecture
from .generator_python import generate_python_scaffold
from .generator_typescript import generate_typescript_scaffold
from .manifest_writer import write_scaffold_result
from .models import (
    FileLanguage,
    ParsedFileSpec,
    ScaffoldFile,
    ScaffoldResult,
)
from .validator import validate_scaffold_result

logger = logging.getLogger(__name__)

# Feature flag — disabled by default until validated
SCAFFOLD_ENABLED = os.environ.get("ASTRA_SCAFFOLD_ENGINE", "0") == "1"


def is_scaffold_enabled() -> bool:
    """Check if the scaffold engine is enabled."""
    return SCAFFOLD_ENABLED


def run_scaffold_engine(
    arch_text: str,
    segment_id: str,
    job_dir: str,
    file_scope: Optional[List[str]] = None,
    emit: Optional[Any] = None,
) -> Optional[ScaffoldResult]:
    """Run the scaffold engine on an approved architecture.

    This is the main entry point called from the segment loop after
    architecture approval and before implementation.

    Args:
        arch_text: Approved architecture markdown.
        segment_id: Segment being scaffolded.
        job_dir: Job directory for manifest output.
        file_scope: Expected file paths (for validation).
        emit: SSE callback for progress messages.

    Returns:
        ScaffoldResult with scaffolded files, or None if disabled/failed.
    """
    if not SCAFFOLD_ENABLED:
        return None

    _emit = emit or (lambda msg: None)
    start = time.time()

    _emit(f"  [SCAFFOLD] v1.0 Running scaffold engine for {segment_id}...")

    result = ScaffoldResult(segment_id=segment_id)

    try:
        # Step 1: Parse architecture into structured file specs
        _emit("    [1/5] Parsing architecture...")
        file_specs = parse_architecture(arch_text)

        if not file_specs:
            _emit("    No parseable file specs found — skipping scaffold")
            return None

        _emit(f"    Parsed {len(file_specs)} file spec(s)")

        # Step 2: Convention extraction (placeholder — requires reference file)
        _emit("    [2/5] Convention extraction skipped (no reference file in v1.0)")

        # Step 3: Generate scaffolds
        _emit("    [3/5] Generating scaffolds...")
        for spec in file_specs:
            if not spec.is_scaffold_eligible():
                result.skipped_files.append(spec.file_path)
                continue

            scaffold = _generate_scaffold_for_spec(spec)
            if scaffold:
                result.files.append(scaffold)
                _emit(
                    f"      {spec.file_path}: {scaffold.fill_count} fills, "
                    f"{scaffold.line_count} lines"
                )
            else:
                result.skipped_files.append(spec.file_path)
                result.warnings.append(
                    f"No generator for {spec.file_path} "
                    f"(lang={spec.language}, role={spec.role})"
                )

        # Step 4: Validate scaffolds
        _emit("    [4/5] Validating scaffolds...")
        validation = validate_scaffold_result(result)
        if validation.status == "fail":
            error_count = sum(
                1 for i in validation.issues if i.severity.value == "error"
            )
            _emit(f"    Validation FAILED: {error_count} error(s)")
            for issue in validation.issues:
                _emit(f"      [{issue.severity.value}] {issue.file_path}: {issue.message}")
                result.warnings.append(
                    f"[{issue.severity.value}] {issue.file_path}: {issue.message}"
                )
        else:
            _emit(f"    Validation passed ({len(validation.issues)} warning(s))")

        # Step 5: Write manifests
        _emit("    [5/5] Writing scaffold manifests...")
        for sf in result.files:
            sf.compute_scaffold_hash()

        manifest_path = write_scaffold_result(result, job_dir)
        if manifest_path:
            _emit(f"    Manifest written: {manifest_path}")

    except Exception as e:
        logger.error("[scaffold] Engine failed for %s: %s", segment_id, e)
        result.warnings.append(f"Scaffold engine error: {e}")
        _emit(f"    [ERROR] Scaffold engine failed: {e}")

    elapsed = (time.time() - start) * 1000
    result.generation_time_ms = elapsed

    _emit(
        f"  [SCAFFOLD] Complete: {result.file_count} files scaffolded, "
        f"{result.total_fills} fills, {result.complete_files} fully deterministic, "
        f"{len(result.skipped_files)} skipped ({elapsed:.0f}ms)"
    )

    return result


def get_scaffold_for_task(
    scaffold_result: Optional[ScaffoldResult],
    file_path: str,
) -> Optional[ScaffoldFile]:
    """Look up the scaffold for a specific file path.

    Called by the Implementer to check if a scaffold exists for the
    file it's about to generate. If so, it fills only the LLM_FILL
    markers instead of generating the entire file.

    Returns ScaffoldFile or None.
    """
    if scaffold_result is None:
        return None
    return scaffold_result.get_scaffold_for_path(file_path)


def build_fill_prompt(scaffold: ScaffoldFile) -> str:
    """Build a targeted prompt for the Implementer to fill scaffold markers.

    Instead of generating the entire file, the Implementer receives:
    1. The scaffold content with LLM_FILL markers
    2. Instructions to ONLY replace the fill markers
    3. The manifest describing what each fill needs

    Returns the prompt string.
    """
    fills_desc = []
    for fill in scaffold.manifest.fills:
        desc = (
            f"- {fill.id} at {fill.location}: {fill.context}"
        )
        if fill.inputs_available:
            desc += f" (inputs: {', '.join(fill.inputs_available)})"
        if fill.return_type:
            desc += f" -> {fill.return_type}"
        fills_desc.append(desc)

    prompt = (
        f"SCAFFOLD FILE: {scaffold.file_path}\n"
        f"FILL COUNT: {scaffold.fill_count}\n\n"
        f"FILLS TO IMPLEMENT:\n"
        + "\n".join(fills_desc)
        + "\n\n"
        f"SCAFFOLD CONTENT:\n"
        f"```\n{scaffold.content}\n```\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Replace each LLM_FILL/FILL_END block with the implementation.\n"
        f"2. Keep ALL other code EXACTLY as-is (imports, signatures, decorators).\n"
        f"3. Do NOT modify lines outside the fill markers.\n"
        f"4. Output the COMPLETE file with fills replaced.\n"
        f"5. No markdown fences in your output.\n"
    )
    return prompt


def _generate_scaffold_for_spec(spec: ParsedFileSpec) -> Optional[ScaffoldFile]:
    """Dispatch to the correct language generator."""
    if spec.language == FileLanguage.PYTHON:
        return generate_python_scaffold(spec)
    elif spec.language == FileLanguage.TYPESCRIPT:
        return generate_typescript_scaffold(spec)
    else:
        return None
