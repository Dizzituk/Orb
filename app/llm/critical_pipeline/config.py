# FILE: app/llm/critical_pipeline/config.py
"""
Critical Pipeline configuration: imports, feature flags, and model resolution.

All try/except import blocks are centralized here so other modules can import
the availability flags and optional references from a single location.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# =============================================================================
# Build Verification
# =============================================================================
CRITICAL_PIPELINE_BUILD_ID = "2026-02-08-v3.1-segment-guard"
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
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    run_high_stakes_with_critique = None
    store_architecture_artifact = None
    get_environment_context = None
    HIGH_STAKES_JOB_TYPES = None
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
    SCHEMAS_AVAILABLE = True
except ImportError as e:
    SCHEMAS_AVAILABLE = False
    LLMTask = None
    JobType = None
    JobEnvelope = None
    Phase4JobType = None
    Importance = None
    DataSensitivity = None
    Modality = None
    JobBudget = None
    OutputContract = None
    logger.warning(f"[critical_pipeline] Schema imports failed: {e}")

# =============================================================================
# Spec Service Imports
# =============================================================================

try:
    from app.specs.service import get_spec, get_latest_validated_spec, get_spec_schema
    SPECS_SERVICE_AVAILABLE = True
except ImportError:
    SPECS_SERVICE_AVAILABLE = False
    get_spec = None
    get_latest_validated_spec = None
    get_spec_schema = None

# =============================================================================
# Evidence Collector Import
# =============================================================================

try:
    from app.pot_spec.evidence_collector import load_evidence, EvidenceBundle
    EVIDENCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[critical_pipeline] Evidence collector not available: {e}")
    EVIDENCE_AVAILABLE = False
    load_evidence = None
    EvidenceBundle = None

# =============================================================================
# Full Evidence Gathering (same as SpecGate)
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
    FULL_EVIDENCE_AVAILABLE = True
    logger.info("[critical_pipeline] Full evidence gathering loaded successfully")
except ImportError as e:
    logger.warning(f"[critical_pipeline] Full evidence gathering not available: {e}")
    FULL_EVIDENCE_AVAILABLE = False
    gather_filesystem_evidence = None
    gather_multi_target_evidence = None
    gather_system_wide_scan_evidence = None
    EvidencePackage = None
    FileEvidence = None
    format_evidence_for_prompt = None
    sandbox_path_exists = None
    sandbox_read_file = None
    sandbox_list_directory = None

# =============================================================================
# Architecture map and codebase report loaders
# =============================================================================

try:
    from app.llm.local_tools.latest_report_resolver import (
        get_latest_architecture_map,
        get_latest_codebase_report_full,
        read_report_content,
    )
    REPORT_RESOLVER_AVAILABLE = True
    logger.info("[critical_pipeline] Report resolver loaded successfully")
except ImportError as e:
    logger.warning(f"[critical_pipeline] Report resolver not available: {e}")
    REPORT_RESOLVER_AVAILABLE = False
    get_latest_architecture_map = None
    get_latest_codebase_report_full = None
    read_report_content = None

# =============================================================================
# Scan Security Imports (HARD SECURITY GATE)
# =============================================================================

try:
    from app.pot_spec.spec_gate_grounded import (
        validate_scan_roots as _validate_scan_roots,
        SAFE_DEFAULT_SCAN_ROOTS,
    )
    SCAN_SECURITY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[critical_pipeline] Scan security functions not available: {e}")
    SCAN_SECURITY_AVAILABLE = False
    SAFE_DEFAULT_SCAN_ROOTS = ["D:\\Orb", "D:\\orb-desktop"]
    _validate_scan_roots = None


def validate_scan_roots(scan_roots):
    """
    Validate scan roots against safe defaults.
    
    Uses the imported validator if available, otherwise falls back to
    a safe local implementation that only allows SAFE_DEFAULT_SCAN_ROOTS.
    """
    if SCAN_SECURITY_AVAILABLE and _validate_scan_roots:
        return _validate_scan_roots(scan_roots)

    # Fallback validation - only allow SAFE_DEFAULT_SCAN_ROOTS
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
    STAGE_MODELS_AVAILABLE = True
except ImportError:
    STAGE_MODELS_AVAILABLE = False
    get_critical_pipeline_config = None

# =============================================================================
# Model Configuration
# =============================================================================


def get_pipeline_model_config() -> dict:
    """Resolve the provider and model for the Critical Pipeline stage."""
    if STAGE_MODELS_AVAILABLE and get_critical_pipeline_config:
        try:
            cfg = get_critical_pipeline_config()
            return {"provider": cfg.provider, "model": cfg.model}
        except Exception:
            pass
    return {
        "provider": os.getenv("CRITICAL_PIPELINE_PROVIDER", "anthropic"),
        "model": os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-5-20251101"),
    }
