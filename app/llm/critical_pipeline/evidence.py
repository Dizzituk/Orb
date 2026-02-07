# FILE: app/llm/critical_pipeline/evidence.py
"""
Evidence gathering for Critical Pipeline.

Provides CriticalPipelineEvidence dataclass and functions to gather
architecture maps, codebase reports, and file-specific evidence.
Gives Critical Pipeline the same visibility as SpecGate (read-only).
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict

from app.llm.critical_pipeline.config import (
    FULL_EVIDENCE_AVAILABLE,
    REPORT_RESOLVER_AVAILABLE,
    format_evidence_for_prompt,
    gather_filesystem_evidence,
    get_latest_architecture_map,
    get_latest_codebase_report_full,
    read_report_content,
    sandbox_read_file,
    sandbox_list_directory,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Evidence Dataclass
# =============================================================================

@dataclass
class CriticalPipelineEvidence:
    """
    Evidence bundle for Critical Pipeline decision-making.

    Critical Pipeline needs full visibility to make informed "how to" decisions.
    This gives it the same evidence-gathering powers as SpecGate.
    """
    # Architecture understanding
    arch_map_content: Optional[str] = None
    arch_map_filename: Optional[str] = None
    arch_map_mtime: Optional[str] = None

    # Codebase understanding
    codebase_report_content: Optional[str] = None
    codebase_report_filename: Optional[str] = None
    codebase_report_mtime: Optional[str] = None

    # Job-specific file evidence (from spec or gathered)
    file_evidence: Optional[Any] = None  # EvidencePackage

    # Multi-target file contents
    multi_target_files: List[Dict[str, Any]] = field(default_factory=list)

    # Evidence loading status
    arch_map_loaded: bool = False
    codebase_report_loaded: bool = False
    file_evidence_loaded: bool = False

    # Errors encountered
    errors: List[str] = field(default_factory=list)

    def to_context_string(
        self,
        max_arch_chars: int = 15000,
        max_codebase_chars: int = 10000,
    ) -> str:
        """Format evidence as context string for LLM prompt."""
        sections = []

        if self.arch_map_content:
            arch_excerpt = self.arch_map_content[:max_arch_chars]
            if len(self.arch_map_content) > max_arch_chars:
                arch_excerpt += (
                    f"\n... [truncated, "
                    f"{len(self.arch_map_content) - max_arch_chars} more chars]"
                )
            sections.append(
                f"## Architecture Map\n"
                f"Source: {self.arch_map_filename} (mtime: {self.arch_map_mtime})\n\n"
                f"{arch_excerpt}\n"
            )

        if self.codebase_report_content:
            codebase_excerpt = self.codebase_report_content[:max_codebase_chars]
            if len(self.codebase_report_content) > max_codebase_chars:
                codebase_excerpt += (
                    f"\n... [truncated, "
                    f"{len(self.codebase_report_content) - max_codebase_chars} more chars]"
                )
            sections.append(
                f"## Codebase Report\n"
                f"Source: {self.codebase_report_filename} "
                f"(mtime: {self.codebase_report_mtime})\n\n"
                f"{codebase_excerpt}\n"
            )

        if self.multi_target_files:
            file_section = ["## Target Files (Content)"]
            for f in self.multi_target_files:
                file_section.append(f"\n### {f.get('name', 'Unknown')}")
                file_section.append(f"Path: {f.get('path', 'Unknown')}")
                content = f.get('content', '')
                if len(content) > 2000:
                    content = (
                        content[:2000]
                        + f"\n... [truncated, {len(f.get('content', '')) - 2000} more chars]"
                    )
                file_section.append(f"```\n{content}\n```")
            sections.append("\n".join(file_section))

        if self.file_evidence and FULL_EVIDENCE_AVAILABLE and format_evidence_for_prompt:
            try:
                sections.append(format_evidence_for_prompt(self.file_evidence))
            except Exception as e:
                logger.warning(
                    "[critical_pipeline] Failed to format file evidence: %s", e
                )

        if self.errors:
            sections.append(
                "## Evidence Gathering Errors\n"
                + "\n".join(f"- {e}" for e in self.errors)
            )

        return "\n\n".join(sections) if sections else "(No evidence gathered)"


# =============================================================================
# Evidence Gathering
# =============================================================================

def gather_critical_pipeline_evidence(
    spec_data: Dict[str, Any],
    message: str,
    include_arch_map: bool = True,
    include_codebase_report: bool = False,
    include_file_evidence: bool = True,
    arch_map_max_lines: int = 500,
    codebase_max_lines: int = 300,
) -> CriticalPipelineEvidence:
    """
    Gather comprehensive evidence for Critical Pipeline.

    This gives Critical Pipeline the same visibility as SpecGate:
    - Architecture map (system structure)
    - Codebase report (file contents and patterns)
    - File-specific evidence (for the job at hand)

    Critical Pipeline is READ-ONLY - no writes ever.
    """
    evidence = CriticalPipelineEvidence()

    logger.info(
        "[critical_pipeline] gather_critical_pipeline_evidence: "
        "arch=%s, codebase=%s, files=%s",
        include_arch_map, include_codebase_report, include_file_evidence,
    )

    # ----- Load Architecture Map -----
    if include_arch_map and REPORT_RESOLVER_AVAILABLE and get_latest_architecture_map:
        try:
            resolved = get_latest_architecture_map()
            if resolved and resolved.found:
                content, _truncated = read_report_content(
                    resolved, max_lines=arch_map_max_lines
                )
                if content:
                    evidence.arch_map_content = content
                    evidence.arch_map_filename = resolved.filename
                    evidence.arch_map_mtime = (
                        resolved.mtime.strftime("%Y-%m-%d %H:%M:%S")
                        if resolved.mtime else None
                    )
                    evidence.arch_map_loaded = True
                    logger.info(
                        "[critical_pipeline] Loaded architecture map: %s (%d chars)",
                        resolved.filename, len(content),
                    )
            else:
                evidence.errors.append("Architecture map not found")
        except Exception as e:
            logger.warning("[critical_pipeline] Failed to load architecture map: %s", e)
            evidence.errors.append(f"Architecture map load failed: {str(e)[:100]}")

    # ----- Load Codebase Report (optional - can be heavy) -----
    if (
        include_codebase_report
        and REPORT_RESOLVER_AVAILABLE
        and get_latest_codebase_report_full
    ):
        try:
            resolved = get_latest_codebase_report_full()
            if resolved and resolved.found:
                content, _truncated = read_report_content(
                    resolved, max_lines=codebase_max_lines
                )
                if content:
                    evidence.codebase_report_content = content
                    evidence.codebase_report_filename = resolved.filename
                    evidence.codebase_report_mtime = (
                        resolved.mtime.strftime("%Y-%m-%d %H:%M:%S")
                        if resolved.mtime else None
                    )
                    evidence.codebase_report_loaded = True
                    logger.info(
                        "[critical_pipeline] Loaded codebase report: %s (%d chars)",
                        resolved.filename, len(content),
                    )
            else:
                evidence.errors.append("Codebase report not found")
        except Exception as e:
            logger.warning("[critical_pipeline] Failed to load codebase report: %s", e)
            evidence.errors.append(f"Codebase report load failed: {str(e)[:100]}")

    # ----- Extract File Evidence from Spec (multi-target files) -----
    if include_file_evidence:
        _gather_file_evidence(evidence, spec_data, message)

    logger.info(
        "[critical_pipeline] Evidence gathering complete: "
        "arch=%s, codebase=%s, files=%d, errors=%d",
        evidence.arch_map_loaded,
        evidence.codebase_report_loaded,
        len(evidence.multi_target_files),
        len(evidence.errors),
    )

    return evidence


def _gather_file_evidence(
    evidence: CriticalPipelineEvidence,
    spec_data: Dict[str, Any],
    message: str,
) -> None:
    """
    Extract file evidence from spec data, checking multiple locations.

    Mutates *evidence* in place.
    """
    # Check MULTIPLE LOCATIONS for multi_target_files.
    # Data may be at root or nested in grounding_data depending on persistence path.
    multi_target_files, source_location = _find_multi_target_files(spec_data)

    if multi_target_files:
        logger.info(
            "[critical_pipeline] Using %d multi_target_files from %s",
            len(multi_target_files), source_location,
        )
        for mtf in multi_target_files:
            evidence.multi_target_files.append({
                "name": mtf.get("name", "Unknown"),
                "path": mtf.get("path", mtf.get("resolved_path", "Unknown")),
                "content": mtf.get("content", mtf.get("full_content", "")),
                "found": mtf.get("found", True),
            })
        evidence.file_evidence_loaded = True
        return

    # Check for single-file evidence
    if spec_data.get("sandbox_input_path") or spec_data.get("sandbox_input_excerpt"):
        input_path = spec_data.get("sandbox_input_path", "Unknown")
        input_excerpt = spec_data.get("sandbox_input_excerpt", "")
        if input_excerpt:
            evidence.multi_target_files.append({
                "name": os.path.basename(input_path) if input_path else "input",
                "path": input_path,
                "content": input_excerpt,
                "found": True,
            })
            evidence.file_evidence_loaded = True
            return

    # If we still don't have file evidence, try gathering it ourselves
    if FULL_EVIDENCE_AVAILABLE and gather_filesystem_evidence:
        try:
            combined_text = (
                f"{message}\n"
                f"{spec_data.get('goal', '')}\n"
                f"{spec_data.get('summary', '')}"
            )
            file_pkg = gather_filesystem_evidence(combined_text)
            if file_pkg and file_pkg.has_valid_targets():
                evidence.file_evidence = file_pkg
                evidence.file_evidence_loaded = True
                for fe in file_pkg.get_all_valid_targets():
                    evidence.multi_target_files.append({
                        "name": (
                            os.path.basename(fe.resolved_path)
                            if fe.resolved_path else fe.original_reference
                        ),
                        "path": fe.resolved_path or fe.original_reference,
                        "content": fe.full_content or fe.content_preview or "",
                        "found": fe.exists and fe.readable,
                    })
                logger.info(
                    "[critical_pipeline] Gathered file evidence: %s",
                    file_pkg.to_summary(),
                )
        except Exception as e:
            logger.warning("[critical_pipeline] Failed to gather file evidence: %s", e)
            evidence.errors.append(f"File evidence gathering failed: {str(e)[:100]}")


def _find_multi_target_files(spec_data: Dict[str, Any]):
    """
    Search multiple spec locations for multi_target_files.

    Returns (list_of_files, source_location_name).
    """
    # Location 1: Root level
    mtf = spec_data.get("multi_target_files", [])
    if mtf:
        return mtf, "root"

    # Location 2: In grounding_data (where SpecGate v1.40+ stores it)
    grounding_data = spec_data.get("grounding_data", {})
    mtf = grounding_data.get("multi_target_files", [])
    if mtf:
        logger.info(
            "[critical_pipeline] Found multi_target_files in grounding_data: %d entries",
            len(mtf),
        )
        return mtf, "grounding_data"

    # Location 3: In evidence_package
    evidence_pkg = spec_data.get("evidence_package", {})
    mtf = evidence_pkg.get("multi_target_files", [])
    if mtf:
        logger.info(
            "[critical_pipeline] Found multi_target_files in evidence_package: %d entries",
            len(mtf),
        )
        return mtf, "evidence_package"

    # Location 4: In sandbox_discovery_result
    sandbox_result = spec_data.get("sandbox_discovery_result", {})
    mtf = sandbox_result.get("multi_target_files", [])
    if mtf:
        logger.info(
            "[critical_pipeline] Found multi_target_files in sandbox_discovery_result: %d entries",
            len(mtf),
        )
        return mtf, "sandbox_discovery_result"

    return [], "none"


# =============================================================================
# File / Directory Read Helpers
# =============================================================================

def read_file_for_critical_pipeline(
    path: str,
    max_chars: int = 50000,
) -> Optional[str]:
    """
    Read a single file for Critical Pipeline.

    Uses sandbox client if available, falls back to direct read.
    """
    if FULL_EVIDENCE_AVAILABLE and sandbox_read_file:
        success, content = sandbox_read_file(path, max_chars=max_chars)
        if success:
            return content

    # Fallback to direct read
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read(max_chars)
    except Exception as e:
        logger.warning(
            "[critical_pipeline] read_file_for_critical_pipeline failed for %s: %s",
            path, e,
        )
        return None


def list_directory_for_critical_pipeline(path: str) -> List[Dict[str, Any]]:
    """
    List directory contents for Critical Pipeline.

    Uses sandbox client if available, falls back to os.listdir.
    """
    if FULL_EVIDENCE_AVAILABLE and sandbox_list_directory:
        success, files = sandbox_list_directory(path)
        if success:
            return files

    # Fallback to direct listing
    try:
        result = []
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            result.append({
                "path": item_path,
                "name": item,
                "is_dir": os.path.isdir(item_path),
                "size": (
                    os.path.getsize(item_path)
                    if os.path.isfile(item_path) else None
                ),
            })
        return result
    except Exception as e:
        logger.warning(
            "[critical_pipeline] list_directory_for_critical_pipeline failed for %s: %s",
            path, e,
        )
        return []
