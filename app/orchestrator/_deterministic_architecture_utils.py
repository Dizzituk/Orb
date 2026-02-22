from __future__ import annotations
import logging
import os
from app.orchestrator.codebase_scanner_models import FileScanResult, SymbolKind
from app.orchestrator.refactor_segmenter_models import FileNode, RefactorBuildPlan, SegmentPlan
from typing import Dict
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


DETERMINISTIC_ARCH_BUILD_ID = "2026-02-21-v1.1-fix19-relative-import-nesting"

def _build_header(segment: SegmentPlan, plan: RefactorBuildPlan) -> str:
    """Build the document header."""
    return f"""# Architecture — {segment.segment_id}

> **Generated deterministically from codebase scan. No LLM involved.**
> Source: `{plan.source_file}` → Package: `{plan.target_package}`
> Segment {segment.segment_index + 1} of {plan.total_segments}
> Tiers: {segment.tiers_included}
> Dependencies: {segment.dependencies or 'none'}"""

def _build_context(plan: RefactorBuildPlan, source_scan: FileScanResult) -> str:
    """Build the refactor context section."""
    lines = [
        "## Refactor Context",
        "",
        f"Decomposing `{plan.source_file}` ({source_scan.line_count} lines, "
        f"{source_scan.function_count} functions, {source_scan.class_count} classes, "
        f"{source_scan.constant_count} constants) into a subpackage.",
        "",
        f"Public symbols to preserve: {len(plan.public_symbols)}",
    ]

    if plan.warnings:
        lines.append("")
        lines.append("### Warnings")
        for w in plan.warnings:
            lines.append(f"- {w}")

    if source_scan.health_issues:
        lines.append("")
        lines.append("### Source Health Issues")
        for issue in source_scan.health_issues:
            lines.append(f"- [{issue.severity.value}] {issue.description}")

    return "\n".join(lines)

def _infer_purpose(node: FileNode) -> str:
    """Infer a file's purpose from its contents."""
    if node.is_facade:
        return "Package facade — re-exports all public symbols"
    if node.is_data_only:
        return "Constants and configuration"

    kinds = set()
    for s in node.symbols:
        if s.kind in (SymbolKind.FUNCTION, SymbolKind.ASYNC_FUNCTION):
            kinds.add("functions")
        elif s.kind == SymbolKind.CLASS:
            kinds.add("classes")
        elif s.kind in (SymbolKind.CONSTANT, SymbolKind.DATA_STRUCTURE):
            kinds.add("constants")

    if not kinds:
        return "Empty — may need removal from inventory"
    return ", ".join(sorted(kinds)).capitalize()

def _name_in_source(name: str, node: FileNode, source_scan: FileScanResult) -> bool:
    """Check if a name appears in any of this node's symbol bodies."""
    import re
    for sym in node.symbols:
        src_sym = source_scan.symbols.get(sym.name)
        if src_sym and src_sym.source_code:
            if re.search(r'\b' + re.escape(name) + r'\b', src_sym.source_code):
                return True
    return False

def _sort_symbols(symbols: list) -> list:
    """Sort symbols: constants first, then classes, then functions."""
    order = {
        SymbolKind.CONSTANT: 0,
        SymbolKind.DATA_STRUCTURE: 1,
        SymbolKind.CLASS: 2,
        SymbolKind.FUNCTION: 3,
        SymbolKind.ASYNC_FUNCTION: 4,
    }
    return sorted(symbols, key=lambda s: (order.get(s.kind, 5), s.name))

def generate_all_architectures(
    plan: RefactorBuildPlan,
    source_scan: FileScanResult,
) -> Dict[str, str]:
    """
    Generate architecture documents for every segment in the plan.

    Returns: {segment_id: architecture_markdown}
    """
    architectures: Dict[str, str] = {}
    completed: Dict[str, str] = {}  # file_path → source (empty until tiers complete)

    for segment in plan.segments:
        arch = generate_segment_architecture(
            segment=segment,
            plan=plan,
            source_scan=source_scan,
            completed_tiers=completed,
        )
        architectures[segment.segment_id] = arch

    logger.info(
        "[deterministic_arch] Generated %d architectures for %s",
        len(architectures), plan.source_file,
    )
    return architectures

def save_architecture(
    arch_text: str,
    job_dir: str,
    segment_id: str,
    version: int = 1,
) -> str:
    """Save a generated architecture to the segment's arch directory."""
    arch_dir = os.path.join(job_dir, "segments", segment_id, "arch")
    os.makedirs(arch_dir, exist_ok=True)

    path = os.path.join(arch_dir, f"arch_v{version}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(arch_text)

    logger.info("[deterministic_arch] Saved: %s (%d chars)", path, len(arch_text))
    return path
