# FILE: app/llm/critical_pipeline_stream.py
"""
Critical Pipeline streaming handler for ASTRA command flow.

v2.13 (2026-02-07): MULTI-ROOT PATH INSTRUCTIONS
- Architecture prompt now instructs LLM to use orb-desktop/ prefix for frontend files
- Backend paths relative to D:\Orb, frontend paths prefixed with orb-desktop/
- Fixes: LLM producing bare src/ paths that resolve to D:\Orb\src\ instead of D:\orb-desktop\src\

v2.12 (2026-02-06): PENDING_EVIDENCE MECHANICAL GUARD
- Hard gate after spec loading: if validation_status == pending_evidence, refuse to proceed
- Also blocks on: blocked, error, needs_clarification statuses
- Emits clear SSE message explaining what to do + done event with blocked=True
- Prevents architecture/execution from "filling in blanks" when CRITICAL ERs unfulfilled
- Logs + prints guard activation for debugging

v2.11 (2026-02-06): SPEC CONSTRAINT ENFORCEMENT
- Extract constraints from spec (key_requirements, design_preferences, constraints, grounding_data)
- Inject as INVIOLABLE rules in system prompt BEFORE evidence/spec content
- Constraints are positioned so the LLM is primed to treat them as hard boundaries
- DECISION blocks can never override explicit spec constraints
- Keyword matching for constraint-like language ("don't rewrite", "as-is", "never", etc.)
- Fixes: Architecture ignoring spec constraints (e.g., rewriting existing code, writing to disk)

v2.10 (2026-02-02): POT SPEC MARKDOWN INJECTION
- Retrieve db_spec.content_markdown alongside content_json
- Pass spec_markdown to run_high_stakes_with_critique() for grounded architecture
- Architecture LLM now receives FULL POT spec with real file paths, line numbers
- Implements "Ground and trust" philosophy - spec IS the instruction set

v2.9.2 (2026-01-30): MULTI-LOCATION EVIDENCE GATHERING FIX
- gather_critical_pipeline_evidence() now checks MULTIPLE LOCATIONS for multi_target_files
- Checks: root level, grounding_data, evidence_package, sandbox_discovery_result
- Matches the same multi-location logic as micro_quickcheck()
- Fixes '📚 File evidence loaded: 1 file(s)' when data is in grounding_data
- Logs source location when multi_target_files are found

v2.9.1 (2026-01-30): MICRO_QUICKCHECK MULTI-LOCATION
- micro_quickcheck() now checks multiple locations for multi_target_files
- Matches SpecGate v1.41 data persistence structure

v2.9 (2026-01-30): FULL EVIDENCE ACCESS
- Critical Pipeline now has the SAME evidence gathering powers as SpecGate
- Can load architecture maps, codebase reports, and read any file
- Added CriticalPipelineEvidence dataclass for evidence bundling
- Added gather_critical_pipeline_evidence() for comprehensive evidence loading
- Added read_file_for_critical_pipeline() for single file reads
- Added list_directory_for_critical_pipeline() for directory listing
- Evidence is now injected into LLM prompts for informed decision-making
- This allows Critical Pipeline to build architecture maps, not just read them

v2.8 (2026-01-30): MULTI-TARGET READ FALLBACK
- If multi_target_files has entries but is_multi_target_read flag is missing,
  automatically treat as multi-target (handles persistence chain issues)
- Fixes MICRO-CHECK-001 when flag doesn't persist but file data does

v2.7 (2026-01-30): MULTI-TARGET READ SUPPORT
- micro_quickcheck() Check 1 now handles is_multi_target_read=True
- For multi-target: validates multi_target_files has entries (not sandbox_input_path)
- For single file: validates sandbox_input_path exists (original behavior)
- Check 3 automatically skips input path validation for multi-target reads

v2.6 (2026-01-25): SCAN_ONLY SECURITY HARDENING (CRITICAL)
- Added HARD SECURITY GATE to scan_quickcheck() - blocks bare drive letters
- scan_roots MUST be within SAFE_DEFAULT_SCAN_ROOTS (D:\\Orb, D:\\orb-desktop)
- Bare drive letters (D:\\, C:\\) are ALWAYS rejected as scan targets
- This prevents scanning the host PC filesystem (ONLY sandbox allowed)
- Added validate_scan_roots import from spec_gate_grounded
- Added fallback validation function if import fails

v2.5 (2026-01-25): SCAN_ONLY Job Type Support
- Added JobKind.SCAN_ONLY for read-only filesystem scan/search/enumerate jobs
- Added ScanQuickcheckResult and scan_quickcheck() for scan validation
- Added _generate_scan_execution_plan() for scan job plans
- Added SCAN_ONLY path to generate_critical_pipeline_stream()
- SCAN_ONLY jobs: No sandbox_input_path/output_path required, CHAT_ONLY output
- Falls back to keyword classification if SpecGate doesn't set job_kind

v2.4 (2026-01-25): Mode-Aware Plan Generation - CRITICAL SAFETY FIX
- _generate_micro_execution_plan() now respects sandbox_output_mode
- CHAT_ONLY: NO "Write Output File" step, NO "Verify output file exists"
- REWRITE_IN_PLACE / APPEND_IN_PLACE: Write to same file
- SEPARATE_REPLY_FILE: Write to output path
- Plans now clearly indicate output mode and whether files will be modified

v2.3 (2026-01): Mode-Aware Quickcheck Validation
- micro_quickcheck() now reads sandbox_output_mode from spec
- CHAT_ONLY: Skips output path checks (MICRO-CHECK-002, 004, 006)
- REWRITE_IN_PLACE: Requires output_path == input_path
- APPEND_IN_PLACE: Requires output_path == input_path
- SEPARATE_REPLY_FILE: Requires output_path exists

v2.2 (2026-01): Quickcheck Validation for Micro Jobs
- Added MicroQuickcheckResult and micro_quickcheck() for deterministic validation
- Micro jobs now get fast tick-box checks before "Ready for Overwatcher"
- No LLM critique for micro jobs - pure deterministic validation

v2.1 (2026-01-04): Artifact Binding Support
- Extracts artifact bindings from spec for Overwatcher
- Includes content_verbatim, location, scope_constraints in architecture prompt
- Generates concrete file paths for implementation

v2.0: Real pipeline integration with Block 4-6.
"""

import json
import logging
import asyncio
import os
from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict
from uuid import uuid4

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# =============================================================================
# v2.8 BUILD VERIFICATION
# =============================================================================
CRITICAL_PIPELINE_BUILD_ID = "2026-02-06-v2.12-pending-evidence-mechanical-guard"
print(f"[CRITICAL_PIPELINE_LOADED] BUILD_ID={CRITICAL_PIPELINE_BUILD_ID}")
logger.info(f"[critical_pipeline] Module loaded: BUILD_ID={CRITICAL_PIPELINE_BUILD_ID}")

# =============================================================================
# Pipeline Imports (Block 4-6)
# =============================================================================

try:
    from app.llm.pipeline.high_stakes import (
        run_high_stakes_with_critique,
        store_architecture_artifact,
        get_environment_context,
        HIGH_STAKES_JOB_TYPES,
    )
    _PIPELINE_AVAILABLE = True
except ImportError as e:
    _PIPELINE_AVAILABLE = False
    logger.warning(f"[critical_pipeline] Pipeline modules not available: {e}")

try:
    from app.llm.pipeline.critique_schemas import CritiqueResult
except ImportError:
    CritiqueResult = None

# =============================================================================
# Schema Imports
# =============================================================================

try:
    from app.llm.schemas import LLMTask, JobType
    from app.jobs.schemas import (
        JobEnvelope,
        JobType as Phase4JobType,
        Importance,
        DataSensitivity,
        Modality,
        JobBudget,
        OutputContract,
    )
    _SCHEMAS_AVAILABLE = True
except ImportError as e:
    _SCHEMAS_AVAILABLE = False
    logger.warning(f"[critical_pipeline] Schema imports failed: {e}")

# =============================================================================
# Spec Service Imports
# =============================================================================

try:
    from app.specs.service import get_spec, get_latest_validated_spec, get_spec_schema
    _SPECS_SERVICE_AVAILABLE = True
except ImportError:
    _SPECS_SERVICE_AVAILABLE = False
    get_spec = None
    get_latest_validated_spec = None
    get_spec_schema = None

# =============================================================================
# Evidence Collector Import (for grounding Critical Pipeline)
# =============================================================================

try:
    from app.pot_spec.evidence_collector import load_evidence, EvidenceBundle
    _EVIDENCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[critical_pipeline] Evidence collector not available: {e}")
    _EVIDENCE_AVAILABLE = False
    load_evidence = None
    EvidenceBundle = None

# =============================================================================
# v2.9: Full Evidence Gathering (same as SpecGate)
# =============================================================================

try:
    from app.pot_spec.grounded.evidence_gathering import (
        gather_filesystem_evidence,
        gather_multi_target_evidence,
        gather_system_wide_scan_evidence,
        EvidencePackage,
        FileEvidence,
        format_evidence_for_prompt,
        sandbox_path_exists,
        sandbox_read_file,
        sandbox_list_directory,
    )
    _FULL_EVIDENCE_AVAILABLE = True
    logger.info("[critical_pipeline] v2.9 Full evidence gathering loaded successfully")
except ImportError as e:
    logger.warning(f"[critical_pipeline] v2.9 Full evidence gathering not available: {e}")
    _FULL_EVIDENCE_AVAILABLE = False
    gather_filesystem_evidence = None
    gather_multi_target_evidence = None
    gather_system_wide_scan_evidence = None
    EvidencePackage = None
    FileEvidence = None
    format_evidence_for_prompt = None
    sandbox_path_exists = None
    sandbox_read_file = None
    sandbox_list_directory = None

# v2.9: Architecture map and codebase report loaders
try:
    from app.llm.local_tools.latest_report_resolver import (
        get_latest_architecture_map,
        get_latest_codebase_report_full,
        read_report_content,
    )
    _REPORT_RESOLVER_AVAILABLE = True
    logger.info("[critical_pipeline] v2.9 Report resolver loaded successfully")
except ImportError as e:
    logger.warning(f"[critical_pipeline] v2.9 Report resolver not available: {e}")
    _REPORT_RESOLVER_AVAILABLE = False
    get_latest_architecture_map = None
    get_latest_codebase_report_full = None
    read_report_content = None

# =============================================================================
# v2.6: Scan Security Imports (HARD SECURITY GATE)
# =============================================================================

try:
    from app.pot_spec.spec_gate_grounded import (
        validate_scan_roots,
        SAFE_DEFAULT_SCAN_ROOTS,
    )
    _SCAN_SECURITY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[critical_pipeline] Scan security functions not available: {e}")
    _SCAN_SECURITY_AVAILABLE = False
    # Fallback definitions for safety
    SAFE_DEFAULT_SCAN_ROOTS = ["D:\\Orb", "D:\\orb-desktop"]
    
    def validate_scan_roots(scan_roots):
        """Fallback validation - only allow SAFE_DEFAULT_SCAN_ROOTS."""
        valid = []
        rejected = []
        for root in scan_roots:
            normalized = root.replace('/', '\\').rstrip('\\')
            # Reject bare drive letters
            if len(normalized) <= 3:
                rejected.append(root)
                continue
            # Check if within allowed roots
            is_allowed = False
            for allowed in SAFE_DEFAULT_SCAN_ROOTS:
                allowed_norm = allowed.replace('/', '\\').rstrip('\\').lower()
                root_norm = normalized.lower()
                if root_norm == allowed_norm or root_norm.startswith(allowed_norm + '\\'):
                    is_allowed = True
                    break
            if is_allowed:
                valid.append(normalized)
            else:
                rejected.append(root)
        if not valid:
            valid = SAFE_DEFAULT_SCAN_ROOTS.copy()
        return valid, rejected

# =============================================================================
# Memory Service Imports
# =============================================================================

try:
    from app.memory import service as memory_service, schemas as memory_schemas
except ImportError:
    memory_service = None
    memory_schemas = None

# =============================================================================
# Audit Logger Imports
# =============================================================================

try:
    from app.llm.audit_logger import RoutingTrace
except ImportError:
    RoutingTrace = None

# =============================================================================
# Stage Models (env-driven model resolution)
# =============================================================================

try:
    from app.llm.stage_models import get_critical_pipeline_config
    _STAGE_MODELS_AVAILABLE = True
except ImportError:
    _STAGE_MODELS_AVAILABLE = False

# =============================================================================
# Configuration
# =============================================================================

def _get_pipeline_model_config() -> dict:
    if _STAGE_MODELS_AVAILABLE:
        try:
            cfg = get_critical_pipeline_config()
            return {"provider": cfg.provider, "model": cfg.model}
        except Exception:
            pass
    return {
        "provider": os.getenv("CRITICAL_PIPELINE_PROVIDER", "anthropic"),
        "model": os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-5-20251101"),
    }


# =============================================================================
# v2.9: Comprehensive Evidence Gathering for Critical Pipeline
# =============================================================================

@dataclass
class CriticalPipelineEvidence:
    """
    v2.9: Evidence bundle for Critical Pipeline decision-making.
    
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
    
    def to_context_string(self, max_arch_chars: int = 15000, max_codebase_chars: int = 10000) -> str:
        """
        Format evidence as context string for LLM prompt.
        
        Args:
            max_arch_chars: Max chars for architecture map excerpt
            max_codebase_chars: Max chars for codebase report excerpt
        """
        sections = []
        
        if self.arch_map_content:
            arch_excerpt = self.arch_map_content[:max_arch_chars]
            if len(self.arch_map_content) > max_arch_chars:
                arch_excerpt += f"\n... [truncated, {len(self.arch_map_content) - max_arch_chars} more chars]"
            sections.append(f"""## Architecture Map
Source: {self.arch_map_filename} (mtime: {self.arch_map_mtime})

{arch_excerpt}
""")
        
        if self.codebase_report_content:
            codebase_excerpt = self.codebase_report_content[:max_codebase_chars]
            if len(self.codebase_report_content) > max_codebase_chars:
                codebase_excerpt += f"\n... [truncated, {len(self.codebase_report_content) - max_codebase_chars} more chars]"
            sections.append(f"""## Codebase Report
Source: {self.codebase_report_filename} (mtime: {self.codebase_report_mtime})

{codebase_excerpt}
""")
        
        if self.multi_target_files:
            file_section = ["## Target Files (Content)"]
            for f in self.multi_target_files:
                file_section.append(f"\n### {f.get('name', 'Unknown')}")
                file_section.append(f"Path: {f.get('path', 'Unknown')}")
                content = f.get('content', '')
                if len(content) > 2000:
                    content = content[:2000] + f"\n... [truncated, {len(f.get('content', '')) - 2000} more chars]"
                file_section.append(f"```\n{content}\n```")
            sections.append("\n".join(file_section))
        
        if self.file_evidence and _FULL_EVIDENCE_AVAILABLE and format_evidence_for_prompt:
            try:
                sections.append(format_evidence_for_prompt(self.file_evidence))
            except Exception as e:
                logger.warning(f"[critical_pipeline] v2.9 Failed to format file evidence: {e}")
        
        if self.errors:
            sections.append(f"## Evidence Gathering Errors\n" + "\n".join(f"- {e}" for e in self.errors))
        
        return "\n\n".join(sections) if sections else "(No evidence gathered)"


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
    v2.9: Gather comprehensive evidence for Critical Pipeline.
    
    This gives Critical Pipeline the same visibility as SpecGate:
    - Architecture map (system structure)
    - Codebase report (file contents and patterns)
    - File-specific evidence (for the job at hand)
    
    Critical Pipeline is READ-ONLY - no writes ever.
    
    Args:
        spec_data: The validated spec from SpecGate
        message: The user's original message
        include_arch_map: Load architecture map (default True)
        include_codebase_report: Load codebase report (default False - heavy)
        include_file_evidence: Gather file evidence from spec (default True)
        arch_map_max_lines: Max lines to load from arch map
        codebase_max_lines: Max lines to load from codebase report
        
    Returns:
        CriticalPipelineEvidence with all gathered evidence
    """
    evidence = CriticalPipelineEvidence()
    
    logger.info(
        "[critical_pipeline] v2.9 gather_critical_pipeline_evidence: "
        "arch=%s, codebase=%s, files=%s",
        include_arch_map, include_codebase_report, include_file_evidence
    )
    
    # =========================================================================
    # Load Architecture Map
    # =========================================================================
    
    if include_arch_map and _REPORT_RESOLVER_AVAILABLE and get_latest_architecture_map:
        try:
            resolved = get_latest_architecture_map()
            if resolved and resolved.found:
                content, truncated = read_report_content(resolved, max_lines=arch_map_max_lines)
                if content:
                    evidence.arch_map_content = content
                    evidence.arch_map_filename = resolved.filename
                    evidence.arch_map_mtime = resolved.mtime.strftime("%Y-%m-%d %H:%M:%S") if resolved.mtime else None
                    evidence.arch_map_loaded = True
                    logger.info(
                        "[critical_pipeline] v2.9 Loaded architecture map: %s (%d chars)",
                        resolved.filename, len(content)
                    )
            else:
                evidence.errors.append("Architecture map not found")
        except Exception as e:
            logger.warning(f"[critical_pipeline] v2.9 Failed to load architecture map: {e}")
            evidence.errors.append(f"Architecture map load failed: {str(e)[:100]}")
    
    # =========================================================================
    # Load Codebase Report (optional - can be heavy)
    # =========================================================================
    
    if include_codebase_report and _REPORT_RESOLVER_AVAILABLE and get_latest_codebase_report_full:
        try:
            resolved = get_latest_codebase_report_full()
            if resolved and resolved.found:
                content, truncated = read_report_content(resolved, max_lines=codebase_max_lines)
                if content:
                    evidence.codebase_report_content = content
                    evidence.codebase_report_filename = resolved.filename
                    evidence.codebase_report_mtime = resolved.mtime.strftime("%Y-%m-%d %H:%M:%S") if resolved.mtime else None
                    evidence.codebase_report_loaded = True
                    logger.info(
                        "[critical_pipeline] v2.9 Loaded codebase report: %s (%d chars)",
                        resolved.filename, len(content)
                    )
            else:
                evidence.errors.append("Codebase report not found")
        except Exception as e:
            logger.warning(f"[critical_pipeline] v2.9 Failed to load codebase report: {e}")
            evidence.errors.append(f"Codebase report load failed: {str(e)[:100]}")
    
    # =========================================================================
    # Extract File Evidence from Spec (multi-target files)
    # =========================================================================
    
    if include_file_evidence:
        # v2.9.2: Check MULTIPLE LOCATIONS for multi_target_files
        # Data may be at root or nested in grounding_data depending on persistence path
        
        # Location 1: Root level
        multi_target_files = spec_data.get("multi_target_files", [])
        source_location = "root"
        
        # Location 2: In grounding_data (where SpecGate v1.40+ stores it)
        if not multi_target_files:
            grounding_data = spec_data.get("grounding_data", {})
            multi_target_files = grounding_data.get("multi_target_files", [])
            if multi_target_files:
                source_location = "grounding_data"
                logger.info(
                    "[critical_pipeline] v2.9.2 Found multi_target_files in grounding_data: %d entries",
                    len(multi_target_files)
                )
        
        # Location 3: In evidence_package (alternative nesting)
        if not multi_target_files:
            evidence_pkg = spec_data.get("evidence_package", {})
            multi_target_files = evidence_pkg.get("multi_target_files", [])
            if multi_target_files:
                source_location = "evidence_package"
                logger.info(
                    "[critical_pipeline] v2.9.2 Found multi_target_files in evidence_package: %d entries",
                    len(multi_target_files)
                )
        
        # Location 4: In sandbox_discovery_result
        if not multi_target_files:
            sandbox_result = spec_data.get("sandbox_discovery_result", {})
            multi_target_files = sandbox_result.get("multi_target_files", [])
            if multi_target_files:
                source_location = "sandbox_discovery_result"
                logger.info(
                    "[critical_pipeline] v2.9.2 Found multi_target_files in sandbox_discovery_result: %d entries",
                    len(multi_target_files)
                )
        
        if multi_target_files:
            logger.info(
                "[critical_pipeline] v2.9.2 Using %d multi_target_files from %s",
                len(multi_target_files), source_location
            )
            
            for mtf in multi_target_files:
                file_entry = {
                    "name": mtf.get("name", "Unknown"),
                    "path": mtf.get("path", mtf.get("resolved_path", "Unknown")),
                    "content": mtf.get("content", mtf.get("full_content", "")),
                    "found": mtf.get("found", True),
                }
                evidence.multi_target_files.append(file_entry)
            
            evidence.file_evidence_loaded = True
        
        # Also check for single-file evidence
        elif spec_data.get("sandbox_input_path") or spec_data.get("sandbox_input_excerpt"):
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
        
        # If we still don't have file evidence, try gathering it ourselves
        if not evidence.file_evidence_loaded and _FULL_EVIDENCE_AVAILABLE and gather_filesystem_evidence:
            try:
                # Gather evidence from the message + spec context
                combined_text = f"{message}\n{spec_data.get('goal', '')}\n{spec_data.get('summary', '')}"
                
                file_pkg = gather_filesystem_evidence(combined_text)
                
                if file_pkg and file_pkg.has_valid_targets():
                    evidence.file_evidence = file_pkg
                    evidence.file_evidence_loaded = True
                    
                    # Also extract file contents for easy access
                    for fe in file_pkg.get_all_valid_targets():
                        evidence.multi_target_files.append({
                            "name": os.path.basename(fe.resolved_path) if fe.resolved_path else fe.original_reference,
                            "path": fe.resolved_path or fe.original_reference,
                            "content": fe.full_content or fe.content_preview or "",
                            "found": fe.exists and fe.readable,
                        })
                    
                    logger.info(
                        "[critical_pipeline] v2.9 Gathered file evidence: %s",
                        file_pkg.to_summary()
                    )
            except Exception as e:
                logger.warning(f"[critical_pipeline] v2.9 Failed to gather file evidence: {e}")
                evidence.errors.append(f"File evidence gathering failed: {str(e)[:100]}")
    
    logger.info(
        "[critical_pipeline] v2.9 Evidence gathering complete: "
        "arch=%s, codebase=%s, files=%d, errors=%d",
        evidence.arch_map_loaded,
        evidence.codebase_report_loaded,
        len(evidence.multi_target_files),
        len(evidence.errors)
    )
    
    return evidence


def read_file_for_critical_pipeline(path: str, max_chars: int = 50000) -> Optional[str]:
    """
    v2.9: Read a single file for Critical Pipeline.
    
    Uses sandbox client if available, falls back to direct read.
    This gives Critical Pipeline the ability to read any file it needs.
    
    Args:
        path: File path to read
        max_chars: Maximum characters to read
        
    Returns:
        File content or None if read failed
    """
    if _FULL_EVIDENCE_AVAILABLE and sandbox_read_file:
        success, content = sandbox_read_file(path, max_chars=max_chars)
        if success:
            return content
    
    # Fallback to direct read
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read(max_chars)
    except Exception as e:
        logger.warning(f"[critical_pipeline] v2.9 read_file_for_critical_pipeline failed for {path}: {e}")
        return None


def list_directory_for_critical_pipeline(path: str) -> List[Dict[str, Any]]:
    """
    v2.9: List directory contents for Critical Pipeline.
    
    Uses sandbox client if available, falls back to os.listdir.
    
    Args:
        path: Directory path to list
        
    Returns:
        List of file/directory info dicts
    """
    if _FULL_EVIDENCE_AVAILABLE and sandbox_list_directory:
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
                "size": os.path.getsize(item_path) if os.path.isfile(item_path) else None,
            })
        return result
    except Exception as e:
        logger.warning(f"[critical_pipeline] v2.9 list_directory_for_critical_pipeline failed for {path}: {e}")
        return []


# =============================================================================
# Job Type Classification (v2.2)
# =============================================================================

class JobKind:
    """Job classification for pipeline routing.
    
    v2.5 (2026-01-25): Added SCAN_ONLY for read-only filesystem scan jobs.
    """
    MICRO_EXECUTION = "micro_execution"  # Simple read/write/answer tasks
    ARCHITECTURE = "architecture"         # Design/build/refactor tasks
    SCAN_ONLY = "scan_only"              # v2.5: Read-only scan/search/enumerate jobs


def _classify_job_kind(spec_data: Dict[str, Any], message: str) -> str:
    """
    Classify job as MICRO_EXECUTION or ARCHITECTURE.
    
    v2.3 FIX: Now checks spec_data.get("job_kind") FIRST.
    SpecGate's deterministic classification takes priority.
    Only falls back to keyword matching if job_kind is missing.
    
    Returns: JobKind.MICRO_EXECUTION or JobKind.ARCHITECTURE
    """
    # ========================================================================
    # v2.3 FIX: Check for pre-classified job_kind from SpecGate FIRST
    # ========================================================================
    
    spec_job_kind = spec_data.get("job_kind", "")
    spec_job_kind_confidence = spec_data.get("job_kind_confidence", 0.0)
    spec_job_kind_reason = spec_data.get("job_kind_reason", "")
    
    if spec_job_kind and spec_job_kind != "unknown":
        # SpecGate already classified this - OBEY IT
        logger.info(
            "[critical_pipeline] v2.5 USING SPEC JOB_KIND: %s (confidence=%.2f, reason='%s')",
            spec_job_kind, spec_job_kind_confidence, spec_job_kind_reason
        )
        
        if spec_job_kind == "micro_execution":
            return JobKind.MICRO_EXECUTION
        elif spec_job_kind == "scan_only":
            # v2.5: SCAN_ONLY jobs are read-only scan/search/enumerate operations
            return JobKind.SCAN_ONLY
        elif spec_job_kind == "repo_change":
            # repo_change is faster than architecture but not as fast as micro
            # For now, treat as architecture (needs some design work)
            return JobKind.ARCHITECTURE
        elif spec_job_kind == "architecture":
            return JobKind.ARCHITECTURE
        else:
            # Unknown or unexpected value - escalate to architecture
            logger.warning(
                "[critical_pipeline] v2.5 Unknown job_kind '%s' - escalating to architecture",
                spec_job_kind
            )
            return JobKind.ARCHITECTURE
    
    # ========================================================================
    # FALLBACK: SpecGate didn't classify (or returned "unknown")
    # Use local classification logic
    # ========================================================================
    
    logger.warning(
        "[critical_pipeline] v2.3 job_kind not set by SpecGate (found='%s') - using fallback classification",
        spec_job_kind
    )
    
    # Combine all text for analysis
    summary = spec_data.get("summary", "").lower()
    objective = spec_data.get("objective", "").lower()
    title = spec_data.get("title", "").lower()
    goal = spec_data.get("goal", "").lower()
    msg_lower = message.lower()
    
    all_text = f"{summary} {objective} {title} {goal} {msg_lower}"
    
    # ========================================================================
    # Check for sandbox/file paths resolved by SpecGate (STRONGEST signal)
    # ========================================================================
    
    # Primary fields from GroundedPOTSpec
    has_sandbox_input = bool(spec_data.get("sandbox_input_path"))
    has_sandbox_output = bool(spec_data.get("sandbox_output_path"))
    has_sandbox_reply = bool(spec_data.get("sandbox_generated_reply"))
    sandbox_discovery_used = spec_data.get("sandbox_discovery_used", False)
    
    # Also check for nested fields or alternate names
    if not has_sandbox_input:
        has_sandbox_input = bool(spec_data.get("input_file_path"))
    if not has_sandbox_output:
        has_sandbox_output = bool(spec_data.get("output_file_path") or spec_data.get("planned_output_path"))
    
    # Check constraints for file paths (SpecGate adds these)
    constraints_from_repo = spec_data.get("constraints_from_repo", [])
    for constraint in constraints_from_repo:
        if isinstance(constraint, str):
            if "planned output path" in constraint.lower() or "reply.txt" in constraint.lower():
                has_sandbox_output = True
    
    # Check what_exists for sandbox input
    what_exists = spec_data.get("what_exists", [])
    for item in what_exists:
        if isinstance(item, str) and "sandbox input" in item.lower():
            has_sandbox_input = True
    
    # ========================================================================
    # MICRO FAST PATH: If sandbox discovery resolved files, it's micro
    # ========================================================================
    
    if sandbox_discovery_used and has_sandbox_input:
        logger.info(
            "[critical_pipeline] v2.3 FALLBACK MICRO: sandbox_discovery_used=True, input=%s, output=%s",
            has_sandbox_input, has_sandbox_output
        )
        return JobKind.MICRO_EXECUTION
    
    if has_sandbox_input and has_sandbox_output and has_sandbox_reply:
        logger.info(
            "[critical_pipeline] v2.3 FALLBACK MICRO: Full sandbox resolution found (input+output+reply)"
        )
        return JobKind.MICRO_EXECUTION
    
    # ========================================================================
    # Keyword-based classification (last resort fallback)
    # ========================================================================
    
    # MICRO indicators (simple file operations)
    micro_indicators = [
        "read the", "read file", "open file", "find the file",
        "answer the question", "answer question", "reply to",
        "write reply", "write answer", "print answer",
        "summarize", "summarise", "extract", "copy",
        "find document", "find the document",
        "what does", "what is", "tell me",
        "underneath", "below", "same folder",
        "sandbox", "desktop", "test folder",
        "read-only", "reply (read-only)",
    ]
    
    # ARCHITECTURE indicators (design/build work)
    arch_indicators = [
        "design", "architect", "build system", "create system",
        "implement feature", "add feature", "new module",
        "refactor", "restructure", "redesign",
        "api endpoint", "database schema", "migration",
        "integration", "pipeline", "service",
        "authentication", "authorization",
        "full implementation", "complete implementation",
        "specgate", "spec gate", "overwatcher",  # Orb system design
    ]
    
    # v2.5: SCAN_ONLY indicators (read-only scan/search/enumerate)
    scan_indicators = [
        "scan the", "scan all", "scan for", "scan folder", "scan folders",
        "scan drive", "scan d:", "scan c:", "scan directory",
        "find all occurrences", "find all references", "find all files",
        "find all folders", "search for", "search the",
        "search entire", "search across", "search all",
        "list all", "list references", "list files",
        "enumerate", "report full paths", "report all",
        "where is", "locate all", "show me all",
        "references to", "mentions of", "occurrences of",
    ]
    
    # Count matches
    micro_score = sum(1 for ind in micro_indicators if ind in all_text)
    arch_score = sum(1 for ind in arch_indicators if ind in all_text)
    scan_score = sum(1 for ind in scan_indicators if ind in all_text)
    
    # Boost micro score if we have resolved file paths
    if has_sandbox_input and has_sandbox_output:
        micro_score += 5  # Strong signal - SpecGate already resolved paths
    elif has_sandbox_input:
        micro_score += 3
    elif has_sandbox_output:
        micro_score += 2
    
    # Check step count from spec (micro jobs typically have ≤5 steps)
    steps = spec_data.get("proposed_steps", spec_data.get("steps", []))
    if isinstance(steps, list):
        if len(steps) <= 5:
            micro_score += 1
        elif len(steps) > 10:
            arch_score += 2
    
    # v2.5: Check for scan_roots in spec (from SpecGate scan discovery)
    has_scan_roots = bool(spec_data.get("scan_roots"))
    has_scan_terms = bool(spec_data.get("scan_terms"))
    
    # Boost scan score if SpecGate resolved scan parameters
    if has_scan_roots:
        scan_score += 5
    if has_scan_terms:
        scan_score += 3
    
    # Log classification
    logger.info(
        "[critical_pipeline] v2.5 FALLBACK classification: micro_score=%d, arch_score=%d, scan_score=%d, "
        "sandbox_discovery=%s, has_paths=%s/%s, has_scan_roots=%s",
        micro_score, arch_score, scan_score, sandbox_discovery_used, 
        has_sandbox_input, has_sandbox_output, has_scan_roots
    )
    
    # v2.5: Check for SCAN_ONLY first (highest priority for scan jobs)
    if scan_score > micro_score and scan_score > arch_score:
        logger.info("[critical_pipeline] v2.5 FALLBACK SCAN_ONLY: scan_score=%d wins", scan_score)
        return JobKind.SCAN_ONLY
    
    # Decision: prefer MICRO if scores are close and paths are resolved
    if micro_score > arch_score:
        return JobKind.MICRO_EXECUTION
    elif arch_score > micro_score:
        return JobKind.ARCHITECTURE
    elif has_sandbox_input or has_sandbox_output:
        # Tie-breaker: if any paths are resolved, it's micro
        return JobKind.MICRO_EXECUTION
    else:
        # Default to architecture for safety
        return JobKind.ARCHITECTURE


# =============================================================================
# Micro Quickcheck Validation (v2.2)
# =============================================================================

@dataclass
class MicroQuickcheckResult:
    """Result of micro-execution quickcheck validation.
    
    This is a fast, deterministic validation - NO LLM calls.
    Pure tick-box checks to verify spec/plan alignment.
    """
    passed: bool
    issues: List[Dict[str, str]] = field(default_factory=list)
    summary: str = ""


def micro_quickcheck(spec_data: Dict[str, Any], plan_text: str) -> MicroQuickcheckResult:
    """
    Fast deterministic validation for micro-execution jobs.
    NO LLM calls - pure tick-box checks.
    
    v2.8 (2026-01-30): MULTI-TARGET READ FALLBACK
    - If multi_target_files has entries but is_multi_target_read=False,
      automatically treat as multi-target (handles persistence chain issues)
    
    v2.7 (2026-01-30): MULTI-TARGET READ SUPPORT
    - Check 1 now handles is_multi_target_read=True
    - For multi-target: validates multi_target_files has entries
    - For single file: validates sandbox_input_path exists
    - Check 3 skips input path validation for multi-target reads
    
    v2.3 Checks (mode-aware):
    1. sandbox_input_path exists in spec (OR multi_target_files for multi-read)
    2. sandbox_output_path validation (mode-dependent):
       - CHAT_ONLY: skip (no output file required)
       - REWRITE_IN_PLACE: require output_path == input_path
       - APPEND_IN_PLACE: require output_path == input_path
       - SEPARATE_REPLY_FILE: require output_path exists
    3. Plan paths match spec paths (skip output check for CHAT_ONLY)
    4. Plan has only safe operations (no destructive commands)
    5. If plan says "write output" but no generated_reply exists → fail (skip for CHAT_ONLY)
    
    Returns:
        MicroQuickcheckResult with pass/fail and any issues found
    """
    issues: List[Dict[str, str]] = []
    
    # v2.9.1 DIAGNOSTIC: Log spec_data keys for debugging
    logger.info(
        "[micro_quickcheck] v2.9.1 DIAGNOSTIC spec_data keys: %s",
        list(spec_data.keys())[:20]  # First 20 keys
    )
    
    # Log specific fields we're looking for
    logger.info(
        "[micro_quickcheck] v2.9.1 DIAGNOSTIC key checks: "
        "is_multi_target_read=%s, multi_target_files=%s, grounding_data=%s, "
        "sandbox_input_path=%s",
        spec_data.get("is_multi_target_read"),
        bool(spec_data.get("multi_target_files")),
        bool(spec_data.get("grounding_data")),
        bool(spec_data.get("sandbox_input_path"))
    )
    
    # =========================================================================
    # v2.3: Extract output_mode for mode-aware validation
    # =========================================================================
    
    output_mode = (spec_data.get("sandbox_output_mode") or "").lower()
    logger.info("[micro_quickcheck] v2.7 output_mode=%s", output_mode)
    
    # =========================================================================
    # v2.8: Check for multi-target read mode (with FALLBACK detection)
    # v2.9.1: Also check grounding_data for multi_target_files
    # =========================================================================
    
    is_multi_target_read = spec_data.get("is_multi_target_read", False)
    
    # v2.9.1: Also check grounding_data for is_multi_target_read flag
    if not is_multi_target_read:
        grounding_data_check = spec_data.get("grounding_data", {})
        is_multi_target_read = grounding_data_check.get("is_multi_target_read", False)
        if is_multi_target_read:
            logger.info("[micro_quickcheck] v2.9.1 Found is_multi_target_read=True in grounding_data")
    
    # v2.9.1: Check multiple locations for multi_target_files
    multi_target_files = spec_data.get("multi_target_files", [])
    
    # If not at root, check grounding_data (where SpecGate v1.40 stores it)
    if not multi_target_files:
        grounding_data = spec_data.get("grounding_data", {})
        multi_target_files = grounding_data.get("multi_target_files", [])
        if multi_target_files:
            logger.info(
                "[micro_quickcheck] v2.9.1 Found multi_target_files in grounding_data: %d entries",
                len(multi_target_files)
            )
    
    # Also check evidence_package if present
    if not multi_target_files:
        evidence_pkg = spec_data.get("evidence_package", {})
        multi_target_files = evidence_pkg.get("multi_target_files", [])
        if multi_target_files:
            logger.info(
                "[micro_quickcheck] v2.9.1 Found multi_target_files in evidence_package: %d entries",
                len(multi_target_files)
            )
    
    # Also check sandbox_discovery_result
    if not multi_target_files:
        sandbox_result = spec_data.get("sandbox_discovery_result", {})
        multi_target_files = sandbox_result.get("multi_target_files", [])
        if multi_target_files:
            logger.info(
                "[micro_quickcheck] v2.9.1 Found multi_target_files in sandbox_discovery_result: %d entries",
                len(multi_target_files)
            )
    
    # v2.8 FALLBACK: If multi_target_files has entries but flag is missing, 
    # treat as multi-target anyway. This handles persistence chain issues.
    if not is_multi_target_read and multi_target_files and len(multi_target_files) > 0:
        logger.warning(
            "[micro_quickcheck] v2.8 FALLBACK: is_multi_target_read=False but multi_target_files has %d entries - treating as multi-target",
            len(multi_target_files)
        )
        is_multi_target_read = True
    
    logger.info(
        "[micro_quickcheck] v2.8 is_multi_target_read=%s, multi_target_files_count=%d",
        is_multi_target_read, len(multi_target_files) if multi_target_files else 0
    )
    
    # =========================================================================
    # Check 1: Input path resolved (v2.7: supports multi-target read)
    # =========================================================================
    
    input_path = ""  # Will be set for single-file mode
    
    if is_multi_target_read:
        # v2.7: Multi-target read - check for multi_target_files instead of single path
        if not multi_target_files or len(multi_target_files) == 0:
            issues.append({
                "id": "MICRO-CHECK-001",
                "description": "is_multi_target_read=True but multi_target_files is empty - no input sources",
                "severity": "blocking",
            })
        else:
            # Multi-target has inputs - CHECK PASSED
            logger.info(
                "[micro_quickcheck] v2.7 MULTI-TARGET READ: %d input files found - CHECK 1 PASSED",
                len(multi_target_files)
            )
    else:
        # Single input file mode - original behavior
        input_path = (
            spec_data.get("sandbox_input_path") or
            spec_data.get("input_file_path") or
            ""
        )
        
        if not input_path:
            issues.append({
                "id": "MICRO-CHECK-001",
                "description": "sandbox_input_path not resolved in spec - cannot verify input source",
                "severity": "blocking",
            })
    
    # =========================================================================
    # Check 2: Output path resolved (v2.3: mode-aware)
    # =========================================================================
    
    output_path = (
        spec_data.get("sandbox_output_path") or
        spec_data.get("output_file_path") or
        spec_data.get("planned_output_path") or
        ""
    )
    
    # v2.3: Mode-aware output path validation
    if output_mode == "chat_only":
        # CHAT_ONLY: No output file required - skip this check
        logger.info("[micro_quickcheck] CHAT_ONLY mode - skipping output path check")
    elif output_mode == "rewrite_in_place":
        if not output_path:
            issues.append({
                "id": "MICRO-CHECK-002",
                "description": "REWRITE_IN_PLACE requires sandbox_output_path",
                "severity": "blocking",
            })
        elif output_path != input_path:
            issues.append({
                "id": "MICRO-CHECK-002",
                "description": f"REWRITE_IN_PLACE requires output_path == input_path (got '{output_path}' vs '{input_path}')",
                "severity": "blocking",
            })
    elif output_mode == "append_in_place":
        if not output_path:
            issues.append({
                "id": "MICRO-CHECK-002",
                "description": "APPEND_IN_PLACE requires sandbox_output_path",
                "severity": "blocking",
            })
        elif output_path != input_path:
            issues.append({
                "id": "MICRO-CHECK-002",
                "description": f"APPEND_IN_PLACE requires output_path == input_path (got '{output_path}' vs '{input_path}')",
                "severity": "blocking",
            })
    elif output_mode == "separate_reply_file":
        if not output_path:
            issues.append({
                "id": "MICRO-CHECK-002",
                "description": "SEPARATE_REPLY_FILE requires sandbox_output_path",
                "severity": "blocking",
            })
    else:
        # Unknown or empty mode - use original behavior (require output_path)
        if not output_path:
            issues.append({
                "id": "MICRO-CHECK-002",
                "description": "sandbox_output_path not resolved in spec - cannot verify output destination",
                "severity": "blocking",
            })
    
    # =========================================================================
    # Check 3: Plan references correct paths
    # =========================================================================
    
    if input_path and input_path not in plan_text:
        # Check for path variations (forward/back slashes, case)
        input_normalized = input_path.replace("\\", "/").lower()
        plan_normalized = plan_text.replace("\\", "/").lower()
        
        if input_normalized not in plan_normalized:
            issues.append({
                "id": "MICRO-CHECK-003",
                "description": f"Plan does not reference spec input path: {input_path}",
                "severity": "blocking",
            })
    
    # v2.3: Skip output path check for CHAT_ONLY mode
    if output_mode != "chat_only" and output_path and output_path not in plan_text:
        # Check for path variations
        output_normalized = output_path.replace("\\", "/").lower()
        plan_normalized = plan_text.replace("\\", "/").lower()
        
        if output_normalized not in plan_normalized:
            issues.append({
                "id": "MICRO-CHECK-004",
                "description": f"Plan does not reference spec output path: {output_path}",
                "severity": "blocking",
            })
    
    # =========================================================================
    # Check 4: Unsafe operations
    # =========================================================================
    
    unsafe_patterns = [
        "rm -rf", "rmdir /s", "del /f /q",
        "format c:", "format d:",
        "DROP TABLE", "DROP DATABASE", "DELETE FROM",
        "TRUNCATE TABLE",
        ":(){:|:&};:",  # Fork bomb
        "shutdown", "reboot",
        "reg delete", "regedit",
    ]
    
    plan_lower = plan_text.lower()
    for pattern in unsafe_patterns:
        if pattern.lower() in plan_lower:
            issues.append({
                "id": "MICRO-CHECK-005",
                "description": f"Plan contains potentially unsafe operation: '{pattern}'",
                "severity": "blocking",
            })
    
    # =========================================================================
    # Check 5: Reply existence (v2.3: skip for CHAT_ONLY)
    # If plan says "write output" but spec has no generated_reply → problem
    # =========================================================================
    
    reply_content = spec_data.get("sandbox_generated_reply", "")
    
    # Check if plan includes write step but no reply exists
    # v2.3: Skip for CHAT_ONLY mode (no output file expected)
    write_keywords = ["write output", "write reply", "write file", "create output", "save reply"]
    plan_has_write = any(kw in plan_lower for kw in write_keywords)
    
    if output_mode != "chat_only" and plan_has_write and not reply_content:
        issues.append({
            "id": "MICRO-CHECK-006",
            "description": "Plan includes write step but spec has no sandbox_generated_reply - nothing to write",
            "severity": "blocking",
        })
    
    # =========================================================================
    # Build result
    # =========================================================================
    
    passed = len(issues) == 0
    
    if passed:
        summary = "✅ All quickchecks passed"
    else:
        blocking_count = sum(1 for i in issues if i.get("severity") == "blocking")
        summary = f"❌ {blocking_count} blocking issue(s) found"
    
    logger.info(
        "[micro_quickcheck] Result: passed=%s, issues=%d, input=%s, output=%s, reply=%s",
        passed, len(issues),
        bool(input_path), bool(output_path), bool(reply_content)
    )
    
    return MicroQuickcheckResult(passed=passed, issues=issues, summary=summary)


def _generate_micro_execution_plan(spec_data: Dict[str, Any], job_id: str) -> str:
    """
    Generate a minimal execution plan for MICRO jobs.
    
    v2.7 (2026-01-30): MULTI-TARGET READ SUPPORT
    - Handles is_multi_target_read=True for multi-file read operations
    - Displays multi_target_files list instead of single input_path
    - Works with SpecGate's multi-target synthesis
    
    v2.4 (2026-01-25): MODE-AWARE PLAN GENERATION
    - CHAT_ONLY: NO write steps, NO output verification
    - REWRITE_IN_PLACE / APPEND_IN_PLACE: Write to same file
    - SEPARATE_REPLY_FILE: Write to output path
    
    No architecture design needed - just a simple step-by-step plan
    that Overwatcher can execute directly.
    
    Uses the sandbox fields populated by SpecGate.
    """
    # v2.4: Get output mode FIRST - this determines plan structure
    output_mode = (spec_data.get("sandbox_output_mode") or "").strip().lower()
    
    # v2.7: Check for multi-target read mode
    is_multi_target_read = spec_data.get("is_multi_target_read", False)
    multi_target_files = spec_data.get("multi_target_files", [])
    
    logger.info(
        f"[_generate_micro_execution_plan] v2.7 output_mode={repr(output_mode)}, "
        f"is_multi_target_read={is_multi_target_read}, multi_target_count={len(multi_target_files)}"
    )
    
    # v2.7: Get input path - handle multi-target read differently
    if is_multi_target_read and multi_target_files:
        # Multi-target: format as list of files
        input_path = f"(multi-target read: {len(multi_target_files)} files)"
    else:
        # Single file: original behavior
        input_path = (
            spec_data.get("sandbox_input_path") or
            spec_data.get("input_file_path") or
            "(input path not resolved)"
        )
    
    output_path = (
        spec_data.get("sandbox_output_path") or
        spec_data.get("output_file_path") or
        spec_data.get("planned_output_path") or
        "(output path not resolved)"
    )
    
    # Get content from SpecGate's sandbox discovery
    input_excerpt = spec_data.get("sandbox_input_excerpt", "")
    reply_content = spec_data.get("sandbox_generated_reply", "")
    content_type = spec_data.get("sandbox_selected_type", "unknown")
    
    # Get summary/goal from spec
    summary = spec_data.get("goal", spec_data.get("summary", spec_data.get("objective", "Execute task per spec")))
    
    # Build content type line only if meaningful (suppress "unknown")
    content_type_line = ""
    if content_type and content_type.lower() != "unknown":
        content_type_line = f"- **Content Type:** {content_type}\n"
    
    # =========================================================================
    # v2.4: MODE-AWARE PLAN GENERATION
    # =========================================================================
    
    if output_mode == "chat_only":
        # =====================================================================
        # CHAT_ONLY: NO file writes, response returned in chat only
        # =====================================================================
        plan = f"""# Micro-Execution Plan

**Job ID:** {job_id}
**Type:** MICRO_EXECUTION (no architecture required)
**Output Mode:** CHAT_ONLY ⚠️ NO FILE WRITES

## Task Summary
{summary}

## Resolved Paths (by SpecGate)
- **Input:** `{input_path}`
- **Output:** (none - CHAT_ONLY mode)
{content_type_line}

## Execution Steps

1. **Read Input File**
   - Path: `{input_path}`
   - Action: Read file contents

2. **Generate Response**
   - Parse the content
   - Generate response based on file content

3. **Return Response in Chat**
   - ⚠️ **NO FILE WRITE** - Response is returned in chat only
   - The input file will NOT be modified
   - No output file will be created

## Verification
- ✅ Input file was read
- ✅ Response was generated
- ⚠️ **NO OUTPUT FILE** (CHAT_ONLY mode - this is intentional)
"""
        
        # Add input preview if available
        if input_excerpt:
            plan += f"""
## Input Preview
```
{input_excerpt[:500] if input_excerpt else '(content will be read at execution time)'}
```
"""
        
        # Add generated reply
        if reply_content:
            plan += f"""
## Generated Reply (from SpecGate) - WILL BE RETURNED IN CHAT
```
{reply_content}
```

**Note:** This reply will be displayed in chat. NO file will be modified.
"""
        
        plan += """
## Notes
- ⚠️ **CHAT_ONLY MODE ACTIVE**
- The user explicitly requested NO file modifications
- Response will be returned in chat only
- Input file remains unchanged
- No output file will be created

---
✅ **Ready for Overwatcher** - CHAT_ONLY mode: Response will be returned in chat, NO file will be modified.
"""
    
    elif output_mode in ("rewrite_in_place", "append_in_place"):
        # =====================================================================
        # REWRITE/APPEND: Write to same file as input
        # =====================================================================
        mode_desc = "rewrite content" if output_mode == "rewrite_in_place" else "append to file"
        
        plan = f"""# Micro-Execution Plan

**Job ID:** {job_id}
**Type:** MICRO_EXECUTION (no architecture required)
**Output Mode:** {output_mode.upper()}

## Task Summary
{summary}

## Resolved Paths (by SpecGate)
- **Input:** `{input_path}`
- **Output:** `{input_path}` (same file - {output_mode})
{content_type_line}

## Execution Steps

1. **Read Input File**
   - Path: `{input_path}`
   - Action: Read file contents

2. **Process Content**
   - Parse the content
   - Generate response based on file content

3. **Write to Same File ({output_mode.upper()})**
   - Path: `{input_path}`
   - Mode: {output_mode}
   - Action: {mode_desc}

4. **Verify**
   - Confirm file was updated
   - Validate content is correct
"""
        
        # Add input preview if available
        if input_excerpt:
            plan += f"""
## Input Preview
```
{input_excerpt[:500] if input_excerpt else '(content will be read at execution time)'}
```
"""
        
        # Add expected output if SpecGate already generated the reply
        if reply_content:
            plan += f"""
## Generated Reply (from SpecGate)
```
{reply_content}
```

**Note:** SpecGate has already generated this reply. Overwatcher will {mode_desc}.
"""
        else:
            plan += """
## Expected Output
(to be generated by Overwatcher based on input content)
"""
        
        plan += f"""
## Notes
- This is a simple file operation task
- Mode: {output_mode.upper()} - writing to same file
- All paths are pre-resolved by SpecGate
- Overwatcher can execute directly

---
✅ **Ready for Overwatcher** - Say 'Astra, command: send to overwatcher' to execute.
"""
    
    else:
        # =====================================================================
        # SEPARATE_REPLY_FILE or default: Write to output path
        # =====================================================================
        plan = f"""# Micro-Execution Plan

**Job ID:** {job_id}
**Type:** MICRO_EXECUTION (no architecture required)
**Output Mode:** {output_mode.upper() if output_mode else 'SEPARATE_REPLY_FILE'}

## Task Summary
{summary}

## Resolved Paths (by SpecGate)
- **Input:** `{input_path}`
- **Output:** `{output_path}`
{content_type_line}

## Execution Steps

1. **Read Input File**
   - Path: `{input_path}`
   - Action: Read file contents

2. **Process Content**
   - Parse the content
   - Generate response based on file content

3. **Write Output File**
   - Path: `{output_path}`
   - Action: Write generated reply

4. **Verify**
   - Confirm output file exists
   - Validate content is correct
"""
        
        # Add input preview if available
        if input_excerpt:
            plan += f"""
## Input Preview
```
{input_excerpt[:500] if input_excerpt else '(content will be read at execution time)'}
```
"""
        
        # Add expected output if SpecGate already generated the reply
        if reply_content:
            plan += f"""
## Generated Reply (from SpecGate)
```
{reply_content}
```

**Note:** SpecGate has already generated this reply. Overwatcher just needs to write it.
"""
        else:
            plan += """
## Expected Output
(to be generated by Overwatcher based on input content)
"""
        
        plan += """
## Notes
- This is a simple file operation task
- No architectural changes required
- All paths are pre-resolved by SpecGate
- Overwatcher can execute directly

---
✅ **Ready for Overwatcher** - Say 'Astra, command: send to overwatcher' to execute.
"""
    
    return plan


# =============================================================================
# v2.5: SCAN_ONLY Quickcheck and Plan Generation
# =============================================================================

@dataclass
class ScanQuickcheckResult:
    """Result of scan-only quickcheck validation.
    
    v2.5: This is a fast, deterministic validation for SCAN_ONLY jobs.
    NO LLM calls - pure tick-box checks to verify scan spec is valid.
    """
    passed: bool
    issues: List[Dict[str, str]] = field(default_factory=list)
    summary: str = ""


def scan_quickcheck(spec_data: Dict[str, Any], plan_text: str) -> ScanQuickcheckResult:
    """
    Fast deterministic validation for SCAN_ONLY jobs.
    NO LLM calls - pure tick-box checks.
    
    v2.6 HARD SECURITY GATE:
    0. scan_roots MUST be within SAFE_DEFAULT_SCAN_ROOTS (D:\\Orb, D:\\orb-desktop)
       Bare drive letters (D:\\, C:\\) are ALWAYS rejected
       This is a non-negotiable security requirement
    
    v2.5 Checks:
    1. scan_roots exists and is non-empty
    2. scan_terms exists OR scan_targets exists (at least one search criterion)
    3. output_mode is CHAT_ONLY (scan results are reported, not written)
    4. No write operations in plan
    5. write_policy is READ_ONLY (if present)
    
    Returns:
        ScanQuickcheckResult with pass/fail and any issues found
    """
    issues: List[Dict[str, str]] = []
    
    # =========================================================================
    # v2.6 HARD SECURITY GATE - scan_roots MUST be within allowed paths
    # This check runs FIRST and is NON-NEGOTIABLE
    # =========================================================================
    
    scan_roots = spec_data.get("scan_roots", [])
    
    if scan_roots:
        # Validate all scan roots against allowlist
        valid_roots, rejected_roots = validate_scan_roots(scan_roots)
        
        if rejected_roots:
            # BLOCKING: We rejected some paths
            logger.error(
                "[scan_quickcheck] v2.6 SECURITY GATE: Rejected %d scan root(s): %s",
                len(rejected_roots), rejected_roots
            )
            issues.append({
                "id": "SCAN-SECURITY-001",
                "description": (
                    f"SECURITY VIOLATION: The following scan roots are NOT allowed: {rejected_roots}. "
                    f"Scans can ONLY target the sandbox workspace: {SAFE_DEFAULT_SCAN_ROOTS}"
                ),
                "severity": "blocking",
            })
        
        # Check for bare drive letters specifically (belt and suspenders)
        for root in scan_roots:
            normalized = root.replace('/', '\\').rstrip('\\')
            if len(normalized) <= 3:  # "D:", "D:\\", "C:"
                logger.error(
                    "[scan_quickcheck] v2.6 SECURITY GATE: Bare drive letter '%s' BLOCKED",
                    root
                )
                if not any(issue["id"] == "SCAN-SECURITY-002" for issue in issues):
                    issues.append({
                        "id": "SCAN-SECURITY-002",
                        "description": (
                            f"SECURITY VIOLATION: Bare drive letter '{root}' is NOT allowed. "
                            f"This would scan the host PC filesystem. "
                            f"Use specific paths within {SAFE_DEFAULT_SCAN_ROOTS} instead."
                        ),
                        "severity": "blocking",
                    })
        
        logger.info(
            "[scan_quickcheck] v2.6 SECURITY GATE: valid_roots=%s, rejected=%s",
            valid_roots, rejected_roots
        )
    
    # =========================================================================
    # Check 1: scan_roots exists and is non-empty
    # =========================================================================
    if not scan_roots:
        issues.append({
            "id": "SCAN-CHECK-001",
            "description": "scan_roots not specified in spec - no scan target defined",
            "severity": "blocking",
        })
    
    # =========================================================================
    # Check 2: scan_terms or scan_targets exists
    # =========================================================================
    
    scan_terms = spec_data.get("scan_terms", [])
    scan_targets = spec_data.get("scan_targets", [])
    
    if not scan_terms and not scan_targets:
        issues.append({
            "id": "SCAN-CHECK-002",
            "description": "No scan_terms or scan_targets specified - what are we scanning for?",
            "severity": "blocking",
        })
    
    # =========================================================================
    # Check 3: output_mode should be CHAT_ONLY for scan jobs
    # =========================================================================
    
    output_mode = (spec_data.get("sandbox_output_mode") or 
                   spec_data.get("output_mode") or "").lower()
    
    if output_mode and output_mode != "chat_only":
        issues.append({
            "id": "SCAN-CHECK-003",
            "description": f"SCAN_ONLY jobs should use CHAT_ONLY output mode (found: {output_mode})",
            "severity": "warning",
        })
    
    # =========================================================================
    # Check 4: No write operations in plan
    # =========================================================================
    
    plan_lower = plan_text.lower()
    write_patterns = [
        "write file", "create file", "save to", "output to",
        "modify file", "edit file", "update file",
        "sandbox_output_path", "sandbox_generated_reply",
    ]
    
    for pattern in write_patterns:
        if pattern in plan_lower:
            issues.append({
                "id": "SCAN-CHECK-004",
                "description": f"SCAN_ONLY plan should not contain write operations (found: '{pattern}')",
                "severity": "warning",
            })
            break
    
    # =========================================================================
    # Check 5: write_policy should be READ_ONLY if present
    # =========================================================================
    
    write_policy = (spec_data.get("write_policy") or "").lower()
    if write_policy and write_policy != "read_only":
        issues.append({
            "id": "SCAN-CHECK-005",
            "description": f"SCAN_ONLY jobs should have write_policy=READ_ONLY (found: {write_policy})",
            "severity": "warning",
        })
    
    # =========================================================================
    # Build result
    # =========================================================================
    
    blocking_count = sum(1 for i in issues if i.get("severity") == "blocking")
    passed = blocking_count == 0
    
    if passed:
        if issues:
            summary = f"✅ Scan quickcheck passed with {len(issues)} warning(s)"
        else:
            summary = "✅ All scan quickchecks passed"
    else:
        summary = f"❌ {blocking_count} blocking issue(s) found"
    
    logger.info(
        "[scan_quickcheck] Result: passed=%s, issues=%d, scan_roots=%d, scan_terms=%d",
        passed, len(issues), len(scan_roots) if scan_roots else 0, len(scan_terms) if scan_terms else 0
    )
    
    return ScanQuickcheckResult(passed=passed, issues=issues, summary=summary)


def _generate_scan_execution_plan(spec_data: Dict[str, Any], job_id: str) -> str:
    """
    Generate a minimal execution plan for SCAN_ONLY jobs.
    
    v2.5: SCAN_ONLY jobs:
    - Do NOT write any files
    - Do NOT require sandbox_input_path / sandbox_output_path
    - Produce results in chat (CHAT_ONLY output)
    - Enumerate folders/files/code matching specified criteria
    
    Uses the scan fields populated by SpecGate:
    - scan_roots: array of root paths to scan
    - scan_terms: array of search terms
    - scan_targets: what to search ("names", "contents")
    - scan_case_mode: case sensitivity
    - scan_exclusions: directories/patterns to skip
    """
    # Extract scan parameters from spec
    scan_roots = spec_data.get("scan_roots", [])
    scan_terms = spec_data.get("scan_terms", [])
    scan_targets = spec_data.get("scan_targets", ["names", "contents"])  # Default: both
    scan_case_mode = spec_data.get("scan_case_mode", "case_insensitive")
    scan_exclusions = spec_data.get("scan_exclusions", [
        ".git", "node_modules", "dist", "build", ".venv", "__pycache__"
    ])
    scan_content_mode = spec_data.get("scan_content_mode", "text_only")
    
    # Get goal/summary from spec
    goal = spec_data.get("goal", spec_data.get("summary", spec_data.get("objective", "Scan and enumerate matching items")))
    
    # Format scan roots for display
    roots_display = "\n".join([f"  - `{r}`" for r in scan_roots]) if scan_roots else "  - (none specified)"
    
    # Format search terms for display  
    terms_display = ", ".join([f"`{t}`" for t in scan_terms]) if scan_terms else "(none specified)"
    
    # Format targets for display
    targets_display = ", ".join(scan_targets) if scan_targets else "names, contents"
    
    # Format exclusions for display
    exclusions_display = ", ".join([f"`{e}`" for e in scan_exclusions[:5]]) if scan_exclusions else "(none)"
    if len(scan_exclusions) > 5:
        exclusions_display += f" + {len(scan_exclusions) - 5} more"
    
    plan = f"""# Scan Execution Plan

**Job ID:** {job_id}
**Type:** SCAN_ONLY (read-only, no file writes)
**Output Mode:** CHAT_ONLY ⚠️ NO FILE WRITES

## Task Summary
{goal}

## Scan Parameters (from SpecGate)

### Scan Roots
{roots_display}

### Search Terms
{terms_display}

### Search Targets
{targets_display}

### Case Mode
{scan_case_mode}

### Content Mode
{scan_content_mode}

### Exclusions
{exclusions_display}

## Execution Steps

1. **Enumerate Directories**
   - Walk directory tree from each scan root
   - Skip excluded directories ({exclusions_display})
   - Collect folder and file names

2. **Match Folder/File Names**
   - Check each folder/file name against search terms
   - Mode: {scan_case_mode}
   - Record name hits with full path

3. **Scan File Contents** (if "contents" in targets)
   - Read text/code files only (skip binaries)
   - Search for term occurrences
   - Record content hits with path, line number, snippet

4. **Compile Results**
   - Group hits by type (name vs content)
   - Group by path/folder
   - Include context for each hit

5. **Return Report in Chat**
   - ⚠️ **NO FILE WRITE** - Results returned in chat only
   - Format as structured report with full paths
   - Explain why each hit exists

## Expected Output Shape

```
SCAN_REPORT:
├── Name Hits:
│   ├── D:\\path\\to\\folder\\orb-file.txt  (matched: "orb" in filename)
│   └── ...
├── Content Hits:
│   ├── D:\\path\\to\\file.py:42  (snippet: "import orb...")
│   └── ...
└── Summary: X name hits, Y content hits across Z files
```

## Verification
- ✅ All scan roots were traversed
- ✅ Search terms were applied
- ✅ Results were compiled
- ⚠️ **NO OUTPUT FILE** (SCAN_ONLY mode - results in chat only)

## Notes
- ⚠️ **SCAN_ONLY MODE ACTIVE**
- This is a read-only operation
- No files will be created or modified
- Results will be returned in chat only
- Scan does NOT require Overwatcher for execution

---
✅ **Ready for Execution** - This scan job can be executed directly without Overwatcher.
"""
    
    return plan


# =============================================================================
# Artifact Binding (v2.1)
# =============================================================================

# Path template variables for artifact binding
PATH_VARIABLES = {
    "{JOB_ID}": lambda ctx: ctx.get("job_id", "unknown"),
    "{JOB_ROOT}": lambda ctx: os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"),
    "{SANDBOX_DESKTOP}": lambda ctx: "C:/Users/WDAGUtilityAccount/Desktop",
    "{REPO_ROOT}": lambda ctx: ctx.get("repo_root", "."),
}


def _resolve_path_template(template: str, context: Dict[str, Any]) -> str:
    """Resolve path template variables."""
    result = template
    for var, resolver in PATH_VARIABLES.items():
        if var in result:
            result = result.replace(var, str(resolver(context)))
    return result


def _extract_artifact_bindings(spec_data: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract and resolve artifact bindings from spec for Overwatcher.
    
    Returns list of bindings with resolved paths:
    [
        {
            "artifact_id": "output_1",
            "action": "create",
            "path": "/resolved/path/to/file.txt",
            "content_type": "text",
            "content_verbatim": "hello",  # if specified
            "description": "Output file"
        }
    ]
    """
    bindings: List[Dict[str, Any]] = []
    
    # Get outputs from spec
    outputs = spec_data.get("outputs", [])
    if not outputs:
        # Try metadata
        metadata = spec_data.get("metadata", {}) or {}
        outputs = metadata.get("outputs", [])
    
    # Get content preservation fields
    content_verbatim = (
        spec_data.get("content_verbatim") or
        spec_data.get("context", {}).get("content_verbatim") or
        spec_data.get("metadata", {}).get("content_verbatim")
    )
    location = (
        spec_data.get("location") or
        spec_data.get("context", {}).get("location") or
        spec_data.get("metadata", {}).get("location")
    )
    
    for i, output in enumerate(outputs):
        if isinstance(output, str):
            output = {"name": output, "path": "", "description": ""}
        
        name = output.get("name", f"output_{i+1}")
        path = output.get("path", "")
        description = output.get("description", output.get("notes", ""))
        
        # Resolve path
        if path:
            resolved_path = _resolve_path_template(path, context)
        elif location:
            # Use location from content preservation
            resolved_path = _resolve_path_template(location, context)
            if name and not resolved_path.endswith(name):
                resolved_path = os.path.join(resolved_path, name)
        else:
            # Default to job artifacts directory
            resolved_path = os.path.join(
                context.get("job_root", "jobs"),
                "jobs",
                context.get("job_id", "unknown"),
                "outputs",
                name
            )
        
        binding = {
            "artifact_id": f"output_{i+1}",
            "action": "create",
            "path": resolved_path,
            "content_type": _infer_content_type(name),
            "description": description or name,
        }
        
        # Include content_verbatim if this is the primary output
        if i == 0 and content_verbatim:
            binding["content_verbatim"] = content_verbatim
        
        bindings.append(binding)
    
    logger.info("[critical_pipeline] Extracted %d artifact bindings", len(bindings))
    return bindings


def _infer_content_type(filename: str) -> str:
    """Infer content type from filename."""
    ext = os.path.splitext(filename.lower())[1]
    type_map = {
        ".txt": "text",
        ".md": "markdown",
        ".json": "json",
        ".py": "python",
        ".js": "javascript",
        ".html": "html",
        ".css": "css",
        ".yaml": "yaml",
        ".yml": "yaml",
    }
    return type_map.get(ext, "text")


def _build_artifact_binding_prompt(bindings: List[Dict[str, Any]]) -> str:
    """Build prompt section for artifact bindings."""
    if not bindings:
        return ""
    
    lines = [
        "\n## ARTIFACT BINDINGS (for Overwatcher)",
        "",
        "The following artifacts MUST be created with these EXACT paths:",
        ""
    ]
    
    for binding in bindings:
        lines.append(f"- **{binding['artifact_id']}**: `{binding['path']}`")
        lines.append(f"  - Action: {binding['action']}")
        lines.append(f"  - Type: {binding['content_type']}")
        if binding.get("content_verbatim"):
            lines.append(f"  - Content: \"{binding['content_verbatim']}\" (EXACT)")
        lines.append("")
    
    lines.append("Overwatcher will use these bindings to write files. Do NOT invent different paths.")
    
    return "\n".join(lines)


# =============================================================================
# Main Stream Handler
# =============================================================================

async def generate_critical_pipeline_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[Any] = None,
    conversation_id: Optional[str] = None,
    spec_id: Optional[str] = None,
    spec_hash: Optional[str] = None,
    job_id: Optional[str] = None,
):
    """
    Generate SSE stream for Critical Pipeline execution with artifact binding (v2.1).
    
    v2.2: Added micro_quickcheck validation before "Ready for Overwatcher"
    
    Flow:
    1. Load validated spec from DB
    2. Classify job type (MICRO vs ARCHITECTURE)
    3a. MICRO: Generate plan → quickcheck validation → ready for Overwatcher
    3b. ARCHITECTURE: Extract bindings → run Block 4-6 pipeline → stream result
    """
    response_parts = []
    
    model_cfg = _get_pipeline_model_config()
    pipeline_provider = model_cfg["provider"]
    pipeline_model = model_cfg["model"]
    
    try:
        yield "data: " + json.dumps({"type": "token", "content": "⚙️ **Critical Pipeline**\n\n"}) + "\n\n"
        response_parts.append("⚙️ **Critical Pipeline**\n\n")
        
        # =====================================================================
        # Validation
        # =====================================================================
        
        if not _PIPELINE_AVAILABLE:
            error_msg = (
                "❌ **Pipeline modules not available.**\n\n"
                "The high-stakes pipeline modules (app.llm.pipeline.*) failed to import.\n"
            )
            yield "data: " + json.dumps({"type": "token", "content": error_msg}) + "\n\n"
            response_parts.append(error_msg)
            if trace:
                trace.finalize(success=False, error_message="Pipeline modules not available")
            yield "data: " + json.dumps({
                "type": "done", "provider": pipeline_provider, "model": pipeline_model,
                "total_length": sum(len(p) for p in response_parts)
            }) + "\n\n"
            return
        
        if not _SCHEMAS_AVAILABLE:
            error_msg = "❌ **Schema imports failed.** Check backend logs.\n"
            yield "data: " + json.dumps({"type": "token", "content": error_msg}) + "\n\n"
            response_parts.append(error_msg)
            yield "data: " + json.dumps({
                "type": "done", "provider": pipeline_provider, "model": pipeline_model,
                "total_length": sum(len(p) for p in response_parts)
            }) + "\n\n"
            return
        
        # =====================================================================
        # Step 1: Load validated spec
        # =====================================================================
        
        yield "data: " + json.dumps({"type": "token", "content": "📋 **Loading validated spec...**\n"}) + "\n\n"
        response_parts.append("📋 **Loading validated spec...**\n")
        
        db_spec = None
        spec_json = None
        
        if spec_id and _SPECS_SERVICE_AVAILABLE and get_spec:
            try:
                db_spec = get_spec(db, spec_id)
            except Exception as e:
                logger.warning(f"[critical_pipeline] Failed to get spec by ID: {e}")
        
        if not db_spec and _SPECS_SERVICE_AVAILABLE and get_latest_validated_spec:
            try:
                db_spec = get_latest_validated_spec(db, project_id)
            except Exception as e:
                logger.warning(f"[critical_pipeline] Failed to get latest validated spec: {e}")
        
        if not db_spec:
            error_msg = (
                "❌ **No validated spec found.**\n\n"
                "Please complete Spec Gate validation first:\n"
                "1. Describe what you want to build\n"
                "2. Say `Astra, command: how does that look all together`\n"
                "3. Say `Astra, command: critical architecture` to validate\n"
                "4. Once validated, retry `run critical pipeline`\n"
            )
            yield "data: " + json.dumps({"type": "token", "content": error_msg}) + "\n\n"
            response_parts.append(error_msg)
            yield "data: " + json.dumps({
                "type": "done", "provider": pipeline_provider, "model": pipeline_model,
                "total_length": sum(len(p) for p in response_parts)
            }) + "\n\n"
            return
        
        spec_id = db_spec.spec_id
        spec_hash = db_spec.spec_hash
        spec_json = db_spec.content_json
        spec_markdown = db_spec.content_markdown  # v2.10: POT spec markdown for grounded architecture
        
        # Parse spec JSON
        try:
            spec_data = json.loads(spec_json) if isinstance(spec_json, str) else (spec_json or {})
        except Exception:
            spec_data = {}
        
        yield "data: " + json.dumps({"type": "token", "content": f"✅ Spec loaded: `{spec_id[:16]}...`\n"}) + "\n\n"
        response_parts.append(f"✅ Spec loaded: `{spec_id[:16]}...`\n")
        
        # =====================================================================
        # v2.12: MECHANICAL GUARD — pending_evidence blocks all downstream
        # =====================================================================
        # If SpecGate marked this spec as pending_evidence, it means CRITICAL
        # EVIDENCE_REQUESTs are unfulfilled. Proceeding would let architecture
        # or execution "fill in the blanks" with guesses. Hard stop here.
        
        validation_status = spec_data.get("validation_status", "validated")
        
        if validation_status == "pending_evidence":
            block_msg = (
                "\n🚫 **BLOCKED: Spec has unfulfilled CRITICAL evidence requirements**\n\n"
                "SpecGate marked this spec as `pending_evidence` because it contains "
                "CRITICAL EVIDENCE_REQUESTs that haven't been resolved yet.\n\n"
                "**What this means:** The spec references files, patterns, or integration "
                "points that haven't been verified against the actual codebase. Proceeding "
                "now would produce architecture based on guesses, not evidence.\n\n"
                "**What to do:**\n"
                "1. Review the EVIDENCE_REQUESTs in the spec output above\n"
                "2. Gather the requested evidence (inspect files, confirm patterns)\n"
                "3. Re-run SpecGate with the evidence to get a `validated` spec\n"
                "4. Then retry `run critical pipeline`\n\n"
                "_This is a mechanical guard — not a suggestion. The pipeline will not "
                "proceed until evidence is gathered._\n"
            )
            logger.warning(
                "[critical_pipeline] v2.12 MECHANICAL GUARD: validation_status=pending_evidence, "
                "spec_id=%s — refusing to proceed",
                spec_id
            )
            print(f"[critical_pipeline] v2.12 BLOCKED: pending_evidence for spec {spec_id}")
            yield "data: " + json.dumps({"type": "token", "content": block_msg}) + "\n\n"
            response_parts.append(block_msg)
            
            yield "data: " + json.dumps({
                "type": "done",
                "provider": pipeline_provider,
                "model": pipeline_model,
                "total_length": sum(len(p) for p in response_parts),
                "blocked": True,
                "blocked_reason": "pending_evidence",
                "validation_status": validation_status,
                "spec_id": spec_id,
            }) + "\n\n"
            
            if trace:
                trace.finalize(success=False, error_message="Spec has pending_evidence status — CRITICAL ERs unfulfilled")
            return
        
        # Also block on other non-ready statuses
        non_ready_statuses = {"blocked", "error", "needs_clarification"}
        if validation_status in non_ready_statuses:
            block_msg = (
                f"\n🚫 **BLOCKED: Spec status is `{validation_status}`**\n\n"
                f"The spec cannot proceed to Critical Pipeline with status `{validation_status}`.\n"
                "Please resolve the spec issues and re-validate before retrying.\n"
            )
            logger.warning(
                "[critical_pipeline] v2.12 MECHANICAL GUARD: validation_status=%s, "
                "spec_id=%s — refusing to proceed",
                validation_status, spec_id
            )
            yield "data: " + json.dumps({"type": "token", "content": block_msg}) + "\n\n"
            response_parts.append(block_msg)
            
            yield "data: " + json.dumps({
                "type": "done",
                "provider": pipeline_provider,
                "model": pipeline_model,
                "total_length": sum(len(p) for p in response_parts),
                "blocked": True,
                "blocked_reason": validation_status,
                "spec_id": spec_id,
            }) + "\n\n"
            
            if trace:
                trace.finalize(success=False, error_message=f"Spec status is {validation_status}")
            return
        
        logger.info(
            "[critical_pipeline] v2.12 validation_status=%s — proceeding",
            validation_status
        )
        
        # =====================================================================
        # Step 1b: Classify Job Type (v2.2)
        # =====================================================================
        
        job_kind = _classify_job_kind(spec_data, message)
        
        yield "data: " + json.dumps({"type": "token", "content": f"🏷️ **Job Type:** `{job_kind}`\n"}) + "\n\n"
        response_parts.append(f"🏷️ **Job Type:** `{job_kind}`\n")
        
        # =====================================================================
        # MICRO_EXECUTION PATH: Skip architecture, generate minimal plan + quickcheck
        # =====================================================================
        
        if job_kind == JobKind.MICRO_EXECUTION:
            yield "data: " + json.dumps({"type": "token", "content": "\n⚡ **Fast Path:** This is a micro-execution job.\n"}) + "\n\n"
            response_parts.append("\n⚡ **Fast Path:** This is a micro-execution job.\n")
            yield "data: " + json.dumps({"type": "token", "content": "No architecture design required - generating execution plan...\n\n"}) + "\n\n"
            response_parts.append("No architecture design required - generating execution plan...\n\n")
            
            # Create job ID
            if not job_id:
                job_id = f"micro-{uuid4().hex[:8]}"
            
            # =================================================================
            # v2.9: Gather evidence for MICRO jobs too (informed plan generation)
            # =================================================================
            
            micro_evidence = gather_critical_pipeline_evidence(
                spec_data=spec_data,
                message=message,
                include_arch_map=False,  # Micro jobs don't need full arch map
                include_codebase_report=False,  # Keep it light
                include_file_evidence=True,  # DO gather file evidence
            )
            
            if micro_evidence.file_evidence_loaded:
                evidence_msg = f"📚 **File evidence loaded:** {len(micro_evidence.multi_target_files)} file(s)\n"
                yield "data: " + json.dumps({"type": "token", "content": evidence_msg}) + "\n\n"
                response_parts.append(evidence_msg)
                logger.info(
                    "[critical_pipeline] v2.9 MICRO evidence: %d files",
                    len(micro_evidence.multi_target_files)
                )
            
            # Generate minimal execution plan (no LLM call needed)
            micro_plan = _generate_micro_execution_plan(spec_data, job_id)
            
            # =================================================================
            # v2.2: Run quickcheck validation BEFORE showing plan
            # =================================================================
            
            yield "data: " + json.dumps({"type": "token", "content": "🧪 **Running Quickcheck...**\n"}) + "\n\n"
            response_parts.append("🧪 **Running Quickcheck...**\n")
            
            quickcheck_result = micro_quickcheck(spec_data, micro_plan)
            
            if quickcheck_result.passed:
                # Quickcheck PASSED - show plan and mark ready
                yield "data: " + json.dumps({"type": "token", "content": f"{quickcheck_result.summary}\n\n"}) + "\n\n"
                response_parts.append(f"{quickcheck_result.summary}\n\n")
                
                yield "data: " + json.dumps({"type": "token", "content": micro_plan}) + "\n\n"
                response_parts.append(micro_plan)
                
                # Extract artifact bindings for Overwatcher
                binding_context = {
                    "job_id": job_id,
                    "job_root": os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"),
                    "repo_root": os.getenv("REPO_ROOT", "."),
                }
                artifact_bindings = _extract_artifact_bindings(spec_data, binding_context)
                
                # Emit completion event
                yield "data: " + json.dumps({
                    "type": "work_artifacts",
                    "spec_id": spec_id,
                    "job_id": job_id,
                    "job_kind": job_kind,
                    "critique_mode": "quickcheck",
                    "critique_passed": True,
                    "artifact_bindings": artifact_bindings,
                }) + "\n\n"
                
                # Save to memory
                full_response = "".join(response_parts)
                if memory_service and memory_schemas:
                    try:
                        memory_service.create_message(db, memory_schemas.MessageCreate(
                            project_id=project_id, role="assistant", content=full_response,
                            provider="local", model="micro-execution"
                        ))
                    except Exception as e:
                        logger.warning(f"[critical_pipeline] Failed to save to memory: {e}")
                
                if trace:
                    trace.finalize(success=True)
                
                yield "data: " + json.dumps({
                    "type": "done",
                    "provider": "local",
                    "model": "micro-execution",
                    "total_length": len(full_response),
                    "spec_id": spec_id,
                    "job_id": job_id,
                    "job_kind": job_kind,
                    "critique_mode": "quickcheck",
                    "critique_passed": True,
                    "artifact_bindings": len(artifact_bindings),
                }) + "\n\n"
                return  # Exit early - micro job complete
            
            else:
                # Quickcheck FAILED - show issues and do NOT mark ready
                yield "data: " + json.dumps({"type": "token", "content": f"{quickcheck_result.summary}\n\n"}) + "\n\n"
                response_parts.append(f"{quickcheck_result.summary}\n\n")
                
                # List the issues
                for issue in quickcheck_result.issues:
                    issue_msg = f"❌ **{issue['id']}:** {issue['description']}\n"
                    yield "data: " + json.dumps({"type": "token", "content": issue_msg}) + "\n\n"
                    response_parts.append(issue_msg)
                
                # Show the plan anyway for debugging
                yield "data: " + json.dumps({"type": "token", "content": "\n### Generated Plan (for review):\n"}) + "\n\n"
                response_parts.append("\n### Generated Plan (for review):\n")
                yield "data: " + json.dumps({"type": "token", "content": micro_plan}) + "\n\n"
                response_parts.append(micro_plan)
                
                # Show next steps
                fail_msg = """
---
⚠️ **Quickcheck Failed** - Job NOT ready for Overwatcher.

Please check:
1. Did SpecGate resolve the input/output paths correctly?
2. Is the spec complete with sandbox_input_path and sandbox_output_path?
3. If the plan needs to write output, does the spec have a sandbox_generated_reply?

You may need to re-run Spec Gate with more details about the file locations.
"""
                yield "data: " + json.dumps({"type": "token", "content": fail_msg}) + "\n\n"
                response_parts.append(fail_msg)
                
                # Save to memory (even on failure)
                full_response = "".join(response_parts)
                if memory_service and memory_schemas:
                    try:
                        memory_service.create_message(db, memory_schemas.MessageCreate(
                            project_id=project_id, role="assistant", content=full_response,
                            provider="local", model="micro-execution"
                        ))
                    except Exception as e:
                        logger.warning(f"[critical_pipeline] Failed to save to memory: {e}")
                
                if trace:
                    trace.finalize(success=False, error_message="Quickcheck failed")
                
                yield "data: " + json.dumps({
                    "type": "done",
                    "provider": "local",
                    "model": "micro-execution",
                    "total_length": len(full_response),
                    "spec_id": spec_id,
                    "job_id": job_id,
                    "job_kind": job_kind,
                    "critique_mode": "quickcheck",
                    "critique_passed": False,
                    "quickcheck_issues": len(quickcheck_result.issues),
                }) + "\n\n"
                return  # Exit - quickcheck failed
        
        # =====================================================================
        # SCAN_ONLY PATH: Generate scan plan + quickcheck (v2.5)
        # =====================================================================
        
        if job_kind == JobKind.SCAN_ONLY:
            yield "data: " + json.dumps({"type": "token", "content": "\n🔍 **Scan Mode:** Read-only filesystem scan.\n"}) + "\n\n"
            response_parts.append("\n🔍 **Scan Mode:** Read-only filesystem scan.\n")
            yield "data: " + json.dumps({"type": "token", "content": "No architecture design required - generating scan execution plan...\n\n"}) + "\n\n"
            response_parts.append("No architecture design required - generating scan execution plan...\n\n")
            
            # Create job ID
            if not job_id:
                job_id = f"scan-{uuid4().hex[:8]}"
            
            # =================================================================
            # v2.9: Gather evidence for SCAN jobs (filesystem context)
            # =================================================================
            
            scan_evidence = gather_critical_pipeline_evidence(
                spec_data=spec_data,
                message=message,
                include_arch_map=True,  # Scan jobs benefit from arch understanding
                include_codebase_report=False,
                include_file_evidence=False,  # Scan finds files, doesn't need existing evidence
                arch_map_max_lines=300,
            )
            
            if scan_evidence.arch_map_loaded:
                evidence_msg = f"📚 **Architecture context loaded:** {len(scan_evidence.arch_map_content or '')} chars\n"
                yield "data: " + json.dumps({"type": "token", "content": evidence_msg}) + "\n\n"
                response_parts.append(evidence_msg)
                logger.info(
                    "[critical_pipeline] v2.9 SCAN evidence: arch_map=%d chars",
                    len(scan_evidence.arch_map_content or '')
                )
            
            # Generate scan execution plan (no LLM call needed)
            scan_plan = _generate_scan_execution_plan(spec_data, job_id)
            
            # =================================================================
            # v2.5: Run scan quickcheck validation BEFORE showing plan
            # =================================================================
            
            yield "data: " + json.dumps({"type": "token", "content": "🧪 **Running Scan Quickcheck...**\n"}) + "\n\n"
            response_parts.append("🧪 **Running Scan Quickcheck...**\n")
            
            scan_quickcheck_result = scan_quickcheck(spec_data, scan_plan)
            
            if scan_quickcheck_result.passed:
                # Quickcheck PASSED - show plan and mark ready
                yield "data: " + json.dumps({"type": "token", "content": f"{scan_quickcheck_result.summary}\n\n"}) + "\n\n"
                response_parts.append(f"{scan_quickcheck_result.summary}\n\n")
                
                # Show any warnings
                for issue in scan_quickcheck_result.issues:
                    if issue.get("severity") == "warning":
                        warn_msg = f"⚠️ **{issue['id']}:** {issue['description']}\n"
                        yield "data: " + json.dumps({"type": "token", "content": warn_msg}) + "\n\n"
                        response_parts.append(warn_msg)
                
                yield "data: " + json.dumps({"type": "token", "content": scan_plan}) + "\n\n"
                response_parts.append(scan_plan)
                
                # Emit completion event
                yield "data: " + json.dumps({
                    "type": "work_artifacts",
                    "spec_id": spec_id,
                    "job_id": job_id,
                    "job_kind": job_kind,
                    "critique_mode": "quickcheck",
                    "critique_passed": True,
                    "scan_roots": spec_data.get("scan_roots", []),
                    "scan_terms": spec_data.get("scan_terms", []),
                    "artifact_bindings": [],  # SCAN_ONLY has no artifacts
                }) + "\n\n"
                
                # Save to memory
                full_response = "".join(response_parts)
                if memory_service and memory_schemas:
                    try:
                        memory_service.create_message(db, memory_schemas.MessageCreate(
                            project_id=project_id, role="assistant", content=full_response,
                            provider="local", model="scan-only"
                        ))
                    except Exception as e:
                        logger.warning(f"[critical_pipeline] Failed to save to memory: {e}")
                
                if trace:
                    trace.finalize(success=True)
                
                yield "data: " + json.dumps({
                    "type": "done",
                    "provider": "local",
                    "model": "scan-only",
                    "total_length": len(full_response),
                    "spec_id": spec_id,
                    "job_id": job_id,
                    "job_kind": job_kind,
                    "critique_mode": "quickcheck",
                    "critique_passed": True,
                }) + "\n\n"
                return  # Exit early - scan job ready for execution
            
            else:
                # Quickcheck FAILED - show issues and do NOT mark ready
                yield "data: " + json.dumps({"type": "token", "content": f"{scan_quickcheck_result.summary}\n\n"}) + "\n\n"
                response_parts.append(f"{scan_quickcheck_result.summary}\n\n")
                
                # List the issues
                for issue in scan_quickcheck_result.issues:
                    severity_icon = "❌" if issue.get("severity") == "blocking" else "⚠️"
                    issue_msg = f"{severity_icon} **{issue['id']}:** {issue['description']}\n"
                    yield "data: " + json.dumps({"type": "token", "content": issue_msg}) + "\n\n"
                    response_parts.append(issue_msg)
                
                # Show the plan anyway for debugging
                yield "data: " + json.dumps({"type": "token", "content": "\n### Generated Plan (for review):\n"}) + "\n\n"
                response_parts.append("\n### Generated Plan (for review):\n")
                yield "data: " + json.dumps({"type": "token", "content": scan_plan}) + "\n\n"
                response_parts.append(scan_plan)
                
                # Show next steps
                fail_msg = """\n---\n⚠️ **Scan Quickcheck Failed** - Job NOT ready for execution.\n\nPlease check:\n1. Did SpecGate resolve the scan_roots correctly?\n2. Did SpecGate extract the scan_terms from your request?\n3. Is the output_mode set to CHAT_ONLY?\n\nYou may need to re-run Spec Gate with more details about what to scan.\n"""
                yield "data: " + json.dumps({"type": "token", "content": fail_msg}) + "\n\n"
                response_parts.append(fail_msg)
                
                # Save to memory (even on failure)
                full_response = "".join(response_parts)
                if memory_service and memory_schemas:
                    try:
                        memory_service.create_message(db, memory_schemas.MessageCreate(
                            project_id=project_id, role="assistant", content=full_response,
                            provider="local", model="scan-only"
                        ))
                    except Exception as e:
                        logger.warning(f"[critical_pipeline] Failed to save to memory: {e}")
                
                if trace:
                    trace.finalize(success=False, error_message="Scan quickcheck failed")
                
                yield "data: " + json.dumps({
                    "type": "done",
                    "provider": "local",
                    "model": "scan-only",
                    "total_length": len(full_response),
                    "spec_id": spec_id,
                    "job_id": job_id,
                    "job_kind": job_kind,
                    "critique_mode": "quickcheck",
                    "critique_passed": False,
                    "quickcheck_issues": len(scan_quickcheck_result.issues),
                }) + "\n\n"
                return  # Exit - scan quickcheck failed
        
        # =====================================================================
        # ARCHITECTURE PATH: Full pipeline (continues below)
        # =====================================================================
        
        yield "data: " + json.dumps({"type": "token", "content": "\n🏗️ **Architecture Mode:** Full design pipeline required.\n\n"}) + "\n\n"
        response_parts.append("\n🏗️ **Architecture Mode:** Full design pipeline required.\n\n")
        
        # =====================================================================
        # Step 2: Create job ID and extract artifact bindings (v2.1)
        # =====================================================================
        
        if not job_id:
            job_id = f"cp-{uuid4().hex[:8]}"
        
        # Build context for path resolution
        binding_context = {
            "job_id": job_id,
            "job_root": os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"),
            "repo_root": os.getenv("REPO_ROOT", "."),
        }
        
        # Extract artifact bindings
        artifact_bindings = _extract_artifact_bindings(spec_data, binding_context)
        
        yield "data: " + json.dumps({"type": "token", "content": f"📁 **Job ID:** `{job_id}`\n"}) + "\n\n"
        response_parts.append(f"📁 **Job ID:** `{job_id}`\n")
        
        if artifact_bindings:
            binding_msg = f"📦 **Artifact Bindings:** {len(artifact_bindings)} output(s)\n"
            for b in artifact_bindings[:3]:  # Show first 3
                binding_msg += f"  - `{b['path']}`\n"
            if len(artifact_bindings) > 3:
                binding_msg += f"  - ... and {len(artifact_bindings) - 3} more\n"
            yield "data: " + json.dumps({"type": "token", "content": binding_msg}) + "\n\n"
            response_parts.append(binding_msg)
        
        # =====================================================================
        # Step 2b: v2.9 Load COMPREHENSIVE evidence for grounded architecture
        # =====================================================================
        
        yield "data: " + json.dumps({"type": "token", "content": "📚 **Gathering evidence...**\n"}) + "\n\n"
        response_parts.append("📚 **Gathering evidence...**\n")
        
        # v2.9: Use new comprehensive evidence gathering (same powers as SpecGate)
        cp_evidence = gather_critical_pipeline_evidence(
            spec_data=spec_data,
            message=message,
            include_arch_map=True,
            include_codebase_report=True,  # v2.9: Now load codebase report too
            include_file_evidence=True,
            arch_map_max_lines=800,  # v2.9: More context
            codebase_max_lines=500,
        )
        
        evidence_status = []
        if cp_evidence.arch_map_loaded:
            evidence_status.append(f"Architecture map ({len(cp_evidence.arch_map_content or '')} chars)")
        if cp_evidence.codebase_report_loaded:
            evidence_status.append(f"Codebase report ({len(cp_evidence.codebase_report_content or '')} chars)")
        if cp_evidence.file_evidence_loaded:
            evidence_status.append(f"File evidence ({len(cp_evidence.multi_target_files)} files)")
        
        if evidence_status:
            evidence_msg = "✅ **Evidence loaded:** " + ", ".join(evidence_status) + "\n"
            yield "data: " + json.dumps({"type": "token", "content": evidence_msg}) + "\n\n"
            response_parts.append(evidence_msg)
        else:
            yield "data: " + json.dumps({"type": "token", "content": "⚠️ **Limited evidence available**\n"}) + "\n\n"
            response_parts.append("⚠️ **Limited evidence available**\n")
        
        if cp_evidence.errors:
            for err in cp_evidence.errors[:3]:  # Show first 3 errors
                err_msg = f"  ⚠️ {err}\n"
                yield "data: " + json.dumps({"type": "token", "content": err_msg}) + "\n\n"
                response_parts.append(err_msg)
        
        # Format evidence for LLM prompt
        evidence_context = cp_evidence.to_context_string(
            max_arch_chars=12000,  # v2.9: More context for architecture
            max_codebase_chars=8000,
        )
        
        logger.info(
            "[critical_pipeline] v2.9 Evidence gathered: arch=%s, codebase=%s, files=%d, context_len=%d",
            cp_evidence.arch_map_loaded,
            cp_evidence.codebase_report_loaded,
            len(cp_evidence.multi_target_files),
            len(evidence_context)
        )
        
        # =====================================================================
        # Step 3: Build prompt with content preservation and bindings
        # =====================================================================
        
        yield "data: " + json.dumps({"type": "token", "content": "🔧 **Building architecture prompt...**\n\n"}) + "\n\n"
        response_parts.append("🔧 **Building architecture prompt...**\n\n")
        
        # Extract content preservation fields
        content_verbatim = (
            spec_data.get("content_verbatim") or
            spec_data.get("context", {}).get("content_verbatim") or
            spec_data.get("metadata", {}).get("content_verbatim")
        )
        location = (
            spec_data.get("location") or
            spec_data.get("context", {}).get("location") or
            spec_data.get("metadata", {}).get("location")
        )
        scope_constraints = (
            spec_data.get("scope_constraints") or
            spec_data.get("context", {}).get("scope_constraints") or
            spec_data.get("metadata", {}).get("scope_constraints") or
            []
        )
        
        # Build artifact binding prompt section
        binding_prompt = _build_artifact_binding_prompt(artifact_bindings)
        
        # Build system prompt with all context
        # FIXED: Extract actual task description, not generic titles
        original_request = message
        if spec_data:
            # Priority order for finding the actual task:
            # 1. summary - often contains the real task description
            # 2. objective - if it's not generic
            # 3. First input's content/example if it looks like a task
            # 4. Fall back to message
            
            summary = spec_data.get("summary", "")
            objective = spec_data.get("objective", "")
            
            # Check if objective is generic/placeholder
            generic_objectives = [
                "job description", "weaver", "build spec", "create spec",
                "draft", "generated", "placeholder"
            ]
            objective_is_generic = any(
                g in (objective or "").lower() for g in generic_objectives
            ) or len(objective or "") < 20
            
            # Use summary if it's more descriptive
            if summary and len(summary) > len(objective or ""):
                original_request = summary
            elif objective and not objective_is_generic:
                original_request = objective
            elif summary:
                original_request = summary
            else:
                # Try to get from inputs or fall back to message
                inputs = spec_data.get("inputs", [])
                if inputs and isinstance(inputs, list) and len(inputs) > 0:
                    first_input = inputs[0]
                    if isinstance(first_input, dict):
                        input_example = first_input.get("example", "")
                        if input_example and len(input_example) > 20:
                            original_request = f"Task: {input_example}"
                        else:
                            original_request = message
                    else:
                        original_request = message
                else:
                    original_request = message
            
            logger.info(f"[critical_pipeline] Extracted objective: {original_request[:100]}...")
        
        system_prompt = f"""You are Claude Opus, generating a detailed architecture document.

SPEC_ID: {spec_id}
SPEC_HASH: {spec_hash}

You are working from a validated PoT Spec. Your architecture MUST:
1. Address all MUST requirements from the spec
2. Consider all constraints
3. Be buildable by a solo developer on Windows 11
4. Include the SPEC_ID and SPEC_HASH header at the top of your output

## CONTENT PRESERVATION (CRITICAL)
"""
        
        if content_verbatim:
            system_prompt += f"""
**EXACT FILE CONTENT REQUIRED:**
The file content MUST be EXACTLY: "{content_verbatim}"
Do NOT paraphrase, summarize, or modify this content in any way.
"""
        
        if location:
            system_prompt += f"""
**EXACT LOCATION REQUIRED:**
The output MUST be written to: {location}
Use this EXACT path - do not substitute or normalize it.
"""
        
        if scope_constraints:
            system_prompt += f"""
**SCOPE CONSTRAINTS:**
{chr(10).join(f'- {c}' for c in scope_constraints)}
The implementation MUST NOT operate outside these boundaries.
"""
        
        system_prompt += binding_prompt
        
        # v2.11: SPEC CONSTRAINT ENFORCEMENT
        # Extract explicit constraints and "use as-is" directives from spec
        # These are injected as INVIOLABLE rules in the system prompt so the LLM
        # is primed to treat them as hard boundaries BEFORE seeing evidence or spec content
        spec_constraints = []
        
        # Pull from key requirements
        key_reqs = spec_data.get("key_requirements", [])
        if isinstance(key_reqs, list):
            for req in key_reqs:
                if isinstance(req, str):
                    lower = req.lower()
                    if any(kw in lower for kw in [
                        "don't rewrite", "don't rebuild", "as-is", "do not",
                        "use existing", "never", "must not", "phase 1 only",
                        "in-memory", "no disk", "no cross-platform",
                        "don't implement", "don't add", "don't create",
                        "only", "not implement",
                    ]):
                        spec_constraints.append(req)
        
        # Pull from design_preferences
        design_prefs = spec_data.get("design_preferences", [])
        if isinstance(design_prefs, list):
            for pref in design_prefs:
                if isinstance(pref, str):
                    spec_constraints.append(pref)
        
        # Pull from explicit constraints field
        explicit_constraints = spec_data.get("constraints", [])
        if isinstance(explicit_constraints, list):
            for c in explicit_constraints:
                if isinstance(c, str):
                    spec_constraints.append(c)
        
        # Also check nested locations where SpecGate may store constraints
        grounding_constraints = spec_data.get("grounding_data", {}).get("constraints", [])
        if isinstance(grounding_constraints, list):
            for c in grounding_constraints:
                if isinstance(c, str) and c not in spec_constraints:
                    spec_constraints.append(c)
        
        if spec_constraints:
            system_prompt += f"""

## INVIOLABLE SPEC CONSTRAINTS (v2.11)

The following constraints are ABSOLUTE. Your architecture MUST NOT violate any of them.
A DECISION block can NEVER override these — they come directly from the user's spec.
If your architecture would violate any constraint, STOP and redesign.

{chr(10).join(f'- ❌ VIOLATION IF BROKEN: {c}' for c in spec_constraints)}

These are not suggestions. Breaking ANY of these constraints means the architecture is WRONG.
"""
            logger.info(
                "[critical_pipeline] v2.11 Injected %d spec constraints into system prompt",
                len(spec_constraints)
            )
        else:
            logger.info("[critical_pipeline] v2.11 No spec constraints found to inject")
        
        # v2.9: Add comprehensive evidence context (architecture + codebase + files)
        if evidence_context and len(evidence_context) > 50:
            system_prompt += f"""\n\n## CODEBASE EVIDENCE (v2.9 - Comprehensive)\n\nThis is comprehensive evidence gathered from the codebase. Use this to understand:\n- Existing code structure and patterns\n- Available modules and services\n- Integration points\n- Actual file contents for context\n\n{evidence_context}\n\nReference these patterns when designing the architecture.\n"""
        
        system_prompt += """

## OUTPUT FORMAT REQUIREMENTS

Your architecture document MUST include a `## File Inventory` section with two markdown tables:

### New Files
| File | Purpose |
|------|--------|
| `path/to/file.ext` | Brief description |

### Modified Files
| File | Purpose |
|------|--------|
| `path/to/file.ext` | Brief description |

This section is REQUIRED — the downstream executor parses it to know which files to create and modify.

**CRITICAL — MULTI-ROOT PATH RULES:**
This project has TWO separate root directories:
- **Backend** (`D:\Orb`): Python/FastAPI. Paths start with `app/`, `tests/`, `main.py`, `requirements.txt`, etc.
- **Frontend** (`D:\orb-desktop`): Electron/React/TypeScript. Paths MUST use the `orb-desktop/` prefix.

Path format rules:
- Backend files: relative to D:\Orb (e.g. `app/routers/voice.py`, `main.py`, `.env`)
- Frontend files: MUST start with `orb-desktop/` (e.g. `orb-desktop/src/components/VoiceInput.tsx`, `orb-desktop/package.json`)
- NEVER use bare `src/` for frontend files — always prefix with `orb-desktop/`
- The architecture map uses these same conventions — follow them exactly.

If no files are modified, include the table header with no rows.

Generate a complete, detailed architecture document."""
        
        task_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate architecture for:\n\n{original_request}\n\nSpec:\n{json.dumps(spec_data, indent=2)}"},
        ]
        
        # Build LLMTask
        task = LLMTask(
            messages=task_messages,
            job_type=JobType.ARCHITECTURE_DESIGN if hasattr(JobType, 'ARCHITECTURE_DESIGN') else list(JobType)[0],
            attachments=[],
        )
        
        # Build JobEnvelope with artifact bindings in metadata
        envelope = JobEnvelope(
            job_id=job_id,
            session_id=conversation_id or f"session-{uuid4().hex[:8]}",
            project_id=project_id,
            job_type=getattr(Phase4JobType, "APP_ARCHITECTURE", list(Phase4JobType)[0]),
            importance=Importance.CRITICAL,
            data_sensitivity=DataSensitivity.INTERNAL,
            modalities_in=[Modality.TEXT],
            budget=JobBudget(
                max_tokens=16384,
                max_cost_estimate=1.00,
                max_wall_time_seconds=600,
            ),
            output_contract=OutputContract.TEXT_RESPONSE,
            messages=task_messages,
            metadata={
                "spec_id": spec_id,
                "spec_hash": spec_hash,
                "pipeline": "critical",
                # v2.1: Include artifact bindings for Overwatcher
                "artifact_bindings": artifact_bindings,
                "content_verbatim": content_verbatim,
                "location": location,
                "scope_constraints": scope_constraints,
            },
            allow_multi_model_review=True,
            needs_tools=[],
        )
        
        # =====================================================================
        # Step 4: Run the pipeline
        # =====================================================================
        
        yield "data: " + json.dumps({"type": "token", "content": f"🏗️ **Starting Block 4-6 Pipeline with {pipeline_model}...**\n\n"}) + "\n\n"
        response_parts.append(f"🏗️ **Starting Block 4-6 Pipeline with {pipeline_model}...**\n\n")
        
        yield "data: " + json.dumps({"type": "token", "content": "This may take 2-5 minutes. Stages:\n"}) + "\n\n"
        yield "data: " + json.dumps({"type": "token", "content": "  1. 📝 Architecture generation\n"}) + "\n\n"
        yield "data: " + json.dumps({"type": "token", "content": "  2. 🔍 Critique (real blockers only)\n"}) + "\n\n"
        yield "data: " + json.dumps({"type": "token", "content": "  3. ✏️ Revision loop (stops early if clean)\n\n"}) + "\n\n"
        
        yield "data: " + json.dumps({
            "type": "pipeline_started",
            "stage": "critical_pipeline",
            "job_id": job_id,
            "spec_id": spec_id,
            "critique_mode": "deep",
            "artifact_bindings": len(artifact_bindings),
        }) + "\n\n"
        
        try:
            result = await run_high_stakes_with_critique(
                task=task,
                provider_id=pipeline_provider,
                model_id=pipeline_model,
                envelope=envelope,
                job_type_str="architecture_design",
                file_map=None,
                db=db,
                spec_id=spec_id,
                spec_hash=spec_hash,
                spec_json=spec_json,
                spec_markdown=spec_markdown,  # v2.10: Inject full POT spec for grounded architecture
                use_json_critique=True,
            )
            
        except Exception as e:
            logger.exception(f"[critical_pipeline] Pipeline failed: {e}")
            error_msg = f"❌ **Pipeline failed:** {e}\n"
            yield "data: " + json.dumps({"type": "token", "content": error_msg}) + "\n\n"
            response_parts.append(error_msg)
            yield "data: " + json.dumps({
                "type": "done", "provider": pipeline_provider, "model": pipeline_model,
                "total_length": sum(len(p) for p in response_parts), "error": str(e)
            }) + "\n\n"
            return
        
        # =====================================================================
        # Step 5: Stream the result
        # =====================================================================
        
        if not result or not result.content:
            error_msg = "❌ **Pipeline returned empty result.**\n"
            yield "data: " + json.dumps({"type": "token", "content": error_msg}) + "\n\n"
            response_parts.append(error_msg)
            yield "data: " + json.dumps({
                "type": "done", "provider": pipeline_provider, "model": pipeline_model,
                "total_length": sum(len(p) for p in response_parts)
            }) + "\n\n"
            return
        
        routing_decision = getattr(result, 'routing_decision', {}) or {}
        arch_id = routing_decision.get('arch_id', 'unknown')
        final_version = routing_decision.get('final_version', 1)
        critique_passed = routing_decision.get('critique_passed', False)
        blocking_issues = routing_decision.get('blocking_issues', 0)
        
        summary_header = "✅ **Pipeline Complete**\n\n"
        yield "data: " + json.dumps({"type": "token", "content": summary_header}) + "\n\n"
        response_parts.append(summary_header)
        
        summary_details = f"""**Architecture ID:** `{arch_id}`
**Final Version:** v{final_version}
**Critique Mode:** deep (blocker filtering enabled)
**Critique Status:** {"✅ PASSED" if critique_passed else f"⚠️ {blocking_issues} blocking issues"}
**Provider:** {result.provider}
**Model:** {result.model}
**Tokens:** {result.total_tokens:,}
**Cost:** ${result.cost_usd:.4f}
**Artifact Bindings:** {len(artifact_bindings)}

---

"""
        yield "data: " + json.dumps({"type": "token", "content": summary_details}) + "\n\n"
        response_parts.append(summary_details)
        
        # Stream architecture content
        yield "data: " + json.dumps({"type": "token", "content": "### Architecture Document\n\n"}) + "\n\n"
        response_parts.append("### Architecture Document\n\n")
        
        content = result.content
        chunk_size = 200
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            yield "data: " + json.dumps({"type": "token", "content": chunk}) + "\n\n"
            response_parts.append(chunk)
            await asyncio.sleep(0.01)
        
        # =====================================================================
        # Step 6: Emit completion events with artifact bindings
        # =====================================================================
        
        yield "data: " + json.dumps({
            "type": "work_artifacts",
            "spec_id": spec_id,
            "job_id": job_id,
            "arch_id": arch_id,
            "final_version": final_version,
            "critique_mode": "deep",
            "critique_passed": critique_passed,
            "artifact_bindings": artifact_bindings,  # v2.1: Include for Overwatcher
            "artifacts": [
                f"arch_v{final_version}.md",
                f"critique_v{final_version}.json",
            ],
        }) + "\n\n"
        
        if critique_passed:
            next_step = f"""

---
✅ **Ready for Implementation**

Architecture approved with {len(artifact_bindings)} artifact binding(s).
Critique mode: deep (blocker filtering enabled, stops early when clean)

🔧 **Next Step:** Say **'Astra, command: send to overwatcher'** to implement.
"""
        else:
            next_step = f"""

---
⚠️ **Critique Not Fully Passed**

{blocking_issues} blocking issues remain (after filtering for real blockers only).

You may:
- Re-run with updated spec
- Proceed to Overwatcher with caution
"""
        
        yield "data: " + json.dumps({"type": "token", "content": next_step}) + "\n\n"
        response_parts.append(next_step)
        
        # Save to memory
        full_response = "".join(response_parts)
        if memory_service and memory_schemas:
            try:
                memory_service.create_message(db, memory_schemas.MessageCreate(
                    project_id=project_id, role="assistant", content=full_response,
                    provider=pipeline_provider, model=pipeline_model
                ))
            except Exception as e:
                logger.warning(f"[critical_pipeline] Failed to save to memory: {e}")
        
        if trace:
            trace.finalize(success=True)
        
        yield "data: " + json.dumps({
            "type": "done",
            "provider": pipeline_provider,
            "model": pipeline_model,
            "total_length": len(full_response),
            "spec_id": spec_id,
            "job_id": job_id,
            "arch_id": arch_id,
            "final_version": final_version,
            "critique_mode": "deep",
            "critique_passed": critique_passed,
            "artifact_bindings": len(artifact_bindings),
            "tokens": result.total_tokens,
            "cost_usd": result.cost_usd,
        }) + "\n\n"
        
    except Exception as e:
        logger.exception("[critical_pipeline] Stream failed: %s", e)
        if trace:
            trace.finalize(success=False, error_message=str(e))
        yield "data: " + json.dumps({"type": "error", "error": str(e)}) + "\n\n"


__all__ = [
    "generate_critical_pipeline_stream",
    "JobKind",
    "MicroQuickcheckResult",
    "micro_quickcheck",
    "ScanQuickcheckResult",
    "scan_quickcheck",
    # v2.9: Evidence gathering exports
    "CriticalPipelineEvidence",
    "gather_critical_pipeline_evidence",
    "read_file_for_critical_pipeline",
    "list_directory_for_critical_pipeline",
]
