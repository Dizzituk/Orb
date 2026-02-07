# FILE: app/llm/pipeline/high_stakes.py
"""High-stakes critique pipeline - Main orchestrator.

Implements Blocks 4, 5, 6 of the PoT (Proof of Thought) system:

Block 4: Architecture generation as versioned artifact with spec traceability
Block 5: Structured JSON critique with blocking/non-blocking issues (critique.py)
Block 6: Revision loop until critique passes (revision.py)

v5.1 (2026-02-06): EVIDENCE FULFILLMENT LOOP (v3.2 contract)
- NEW: Architecture draft wrapped with run_stage_with_evidence()
- EVIDENCE_REQUESTs are now dispatched by orchestrator (file reads, RAG, etc.)
- Fulfilled evidence injected back; LLM re-generates with real data
- After max_loops (env: ASTRA_EVIDENCE_MAX_LOOPS, default 2), force-resolve
- Kill switch: set ASTRA_EVIDENCE_LOOP_ENABLED=0 to revert to single-pass
- Token counts accumulated across loop iterations
- Legacy single-pass path preserved as fallback

v5.0 (2026-02-02): GROUNDED SPEC INJECTION - POT spec as source of truth
- NEW: spec_markdown parameter for full POT spec with grounded evidence
- Architecture LLM now receives the COMPLETE POT spec with file paths, line numbers
- The POT spec IS the instruction set - architecture must follow it
- Removed rigid job_kind checking - spec tells the LLM what type of work this is
- Philosophy: "Ground and trust" - grounded evidence prevents hallucination
- If spec says "Change line 42 of file X", that's exactly what architecture addresses

v4.3 (2026-01-22): CRITICAL FIX - Phantom ENVIRONMENT_CONSTRAINTS Bug
- get_environment_context() now extracts tech_stack FROM SPEC, not hardcoded defaults
- Removed hardcoded React/Electron/FastAPI/SQLite constraints that were causing
  critique to reject valid architectures for phantom requirements
- If spec has no implementation_stack, NO tech_stack constraints are applied
- See PHANTOM_CONSTRAINT_BUG_FIX.md for full details

v4.2 (2026-01-22): CRITICAL FIX - Spec Injection into Architecture Prompt
- Architecture prompt now receives FULL spec content (goal, requirements, constraints)
- implementation_stack field injected with STACK LOCKED warnings when applicable
- Prevents architecture from ignoring user-discussed tech stack
- See CRITICAL_PIPELINE_FAILURE_REPORT.md for why this is needed

v4.1 (2026-01):
- Uses stage_models for provider/model configuration (env-driven)
- ARCHITECTURE_PROVIDER/ARCHITECTURE_MODEL from env controls draft generation

v4.0 (2025-12):
- REFACTORED: Split into 3 files for maintainability:
  - high_stakes.py: Orchestrator, routing, architecture storage (~400 lines)
  - critique.py: Critique callers, prompts, parsing (~420 lines)
  - revision.py: Revision loop, spec-anchored prompts (~380 lines)
- Spec-anchored pipeline prevents drift from reviewer suggestions
- Debug logging throughout for visibility

SPEC ANCHORING:
The PoT Spec serves as the authoritative anchor point throughout the pipeline:
1. Spec Gate creates the spec (user intent → requirements)
2. Claude Opus generates architecture (must include SPEC_ID/SPEC_HASH header)
3. Gemini critiques architecture against spec (verify alignment)
4. Claude Opus revises based on critique, BUT verifies suggestions against spec first
   - Reject suggestions that add/contradict spec requirements
   - This prevents "spec drift" where reviewers inadvertently change scope
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from app.llm.schemas import JobType, LLMResult, LLMTask
from app.jobs.schemas import (
    JobEnvelope,
    JobType as Phase4JobType,
    Importance,
    DataSensitivity,
    Modality,
    JobBudget,
    OutputContract,
)
from app.providers.registry import llm_call as registry_llm_call
from app.llm.job_classifier import compute_modality_flags
from app.llm.gemini_vision import transcribe_video_for_context

# Import from sibling modules
from app.llm.pipeline.critique import (
    call_json_critic,
    store_critique_artifact,
    call_gemini_critic,
    build_critique_prompt,
    GEMINI_CRITIC_MODEL,
)
from app.llm.pipeline.revision import (
    call_revision,
    run_revision_loop,
    call_opus_revision,
    _map_to_phase4_job_type,
    OPUS_REVISION_MAX_TOKENS,
    MAX_REVISION_ITERATIONS,
)
from app.llm.pipeline.critique_schemas import CritiqueResult

# v2.0: Evidence-or-Request Contract prompt
try:
    from app.llm.pipeline.evidence_contract_prompt import EVIDENCE_CONTRACT_PROMPT
    _EVIDENCE_CONTRACT_AVAILABLE = True
except ImportError:
    _EVIDENCE_CONTRACT_AVAILABLE = False
    EVIDENCE_CONTRACT_PROMPT = ""

# v3.2: Evidence fulfillment loop
try:
    from app.llm.pipeline.evidence_loop import (
        run_stage_with_evidence,
        parse_evidence_requests,
        StageResult,
        JobContext,
    )
    _EVIDENCE_LOOP_AVAILABLE = True
except ImportError:
    _EVIDENCE_LOOP_AVAILABLE = False

# Audit logging (Spec §12)
try:
    from app.llm.audit_logger import (
        get_audit_logger,
        RoutingTrace,
        AuditEventType,
    )
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False

# Artefact service (Block 4)
try:
    from app.artefacts.service import ArtefactService, write_architecture_doc
    ARTEFACTS_AVAILABLE = True
except ImportError:
    ARTEFACTS_AVAILABLE = False

# Ledger events (Block 4)
try:
    from app.pot_spec.ledger import (
        emit_arch_created,
        emit_arch_mirror_written,
    )
    from app.pot_spec.service import get_job_artifact_root
    LEDGER_AVAILABLE = True
except ImportError:
    LEDGER_AVAILABLE = False

# Stage 3 spec echo (for verification)
try:
    from app.jobs.stage3_locks import build_spec_echo_instruction
    STAGE3_AVAILABLE = True
except ImportError:
    STAGE3_AVAILABLE = False

# Stage models (env-driven model resolution)
try:
    from app.llm.stage_models import get_architecture_config
    _STAGE_MODELS_AVAILABLE = True
except ImportError:
    _STAGE_MODELS_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

AUDIT_ENABLED = os.getenv("ORB_AUDIT_ENABLED", "1") == "1"

logger = logging.getLogger(__name__)

MIN_CRITIQUE_CHARS = int(os.getenv("ORB_MIN_CRITIQUE_CHARS", "1500"))


def _get_architecture_draft_config() -> tuple[str, str, int, int]:
    """Get architecture draft provider/model from stage_models or env vars AT RUNTIME.
    
    Returns: (provider, model, max_tokens, timeout)
    """
    if _STAGE_MODELS_AVAILABLE:
        try:
            cfg = get_architecture_config()
            return cfg.provider, cfg.model, cfg.max_output_tokens, cfg.timeout_seconds
        except Exception:
            pass
    
    # Fallback to legacy env vars
    provider = os.getenv("ARCHITECTURE_PROVIDER", "anthropic")
    model = os.getenv("ARCHITECTURE_MODEL") or os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-5-20251101")
    max_tokens = int(os.getenv("ARCHITECTURE_MAX_OUTPUT_TOKENS") or os.getenv("OPUS_DRAFT_MAX_TOKENS", "60000"))
    timeout = int(os.getenv("ARCHITECTURE_TIMEOUT_SECONDS") or os.getenv("OPUS_TIMEOUT_SECONDS", "600"))
    return provider, model, max_tokens, timeout


# Legacy exports (for backward compatibility)
OPUS_DRAFT_MAX_TOKENS = int(os.getenv("ARCHITECTURE_MAX_OUTPUT_TOKENS") or os.getenv("OPUS_DRAFT_MAX_TOKENS", "60000"))
OPUS_TIMEOUT_SECONDS = int(os.getenv("ARCHITECTURE_TIMEOUT_SECONDS") or os.getenv("OPUS_TIMEOUT_SECONDS", "600"))

HIGH_STAKES_JOB_TYPES = {
    "architecture_design",
    "security_review",
    "high_stakes_infra",
    "big_architecture",
    "compliance_review",
    "high_stakes_legal",
    "high_stakes_medical",
    "orchestrator",
}


# =============================================================================
# Audit Integration
# =============================================================================

def _maybe_start_trace(
    task: LLMTask,
    envelope: JobEnvelope,
    *,
    job_type_str: str,
    provider_id: str,
    model_id: str,
):
    """Start an audit trace if auditing is enabled."""
    if not (AUDIT_AVAILABLE and AUDIT_ENABLED):
        return None, None

    try:
        audit_logger = get_audit_logger()
        if not audit_logger:
            return None, None
        
        # Extract IDs from envelope for trace
        job_id = str(getattr(envelope, "job_id", "unknown"))
        session_id = str(getattr(envelope, "session_id", "unknown"))
        project_id = int(getattr(envelope, "project_id", 0))
        
        # Create trace with correct dataclass parameters
        trace = RoutingTrace(
            logger=audit_logger,
            request_id=job_id,
            session_id=session_id,
            project_id=project_id,
        )
        
        # Log the routing decision with metadata
        trace.log_routing_decision(
            job_type=job_type_str,
            provider=provider_id,
            model=model_id,
            reason="high_stakes_critique pipeline",
        )
        
        return audit_logger, trace
    except Exception as exc:
        logger.warning(f"[audit] Failed to start trace: {exc}")
        return None, None


def _maybe_complete_trace(audit_logger, trace, *, success: bool = True, error_message: str = "") -> None:
    """Complete an audit trace if one exists."""
    if not trace or not audit_logger:
        return
    try:
        audit_logger.complete_trace(trace, success=success, error_message=error_message)
    except Exception:
        pass


def _trace_step(trace, step: str, **kv) -> None:
    """Log a step/warning to the trace."""
    if not trace:
        return
    try:
        # RoutingTrace doesn't have add_step, use log_warning for step tracking
        trace.log_warning(f"step:{step}", **kv)
    except Exception:
        pass


def _trace_error(trace, step: str, message: str) -> None:
    """Log an error to the trace."""
    if not trace:
        return
    try:
        trace.log_error(step, message)
    except Exception:
        pass


# =============================================================================
# Environment Context
# =============================================================================

def get_environment_context(spec_json: Optional[str] = None) -> Dict[str, Any]:
    """Get environment context from spec, NOT hardcoded defaults.
    
    v1.1 (2026-01-22): CRITICAL FIX - Phantom Constraint Bug
    - Tech stack constraints MUST come from the spec, not hardcoded defaults
    - If spec has implementation_stack, use that
    - If spec has no implementation_stack, DO NOT inject default tech_stack
    - This prevents critique from rejecting architectures for phantom requirements
    
    Args:
        spec_json: The SpecGate JSON spec (contains implementation_stack if user specified)
    
    Returns:
        Environment context dict with deployment info and spec-derived constraints
    """
    # Base deployment context (platform-specific, always included)
    context = {
        "deployment": {
            "type": "single_host",
            "os": "Windows 11",
            "scope": "local_only",
            "network": "LAN",
            "resources": "solo_dev_workstation",
        },
        "constraints": {
            "cloud_services": False,
            "external_hosting": False,
            "multi_user": False,
            "scale": "personal_project",
        },
    }
    
    # ==========================================================================
    # v1.1 CRITICAL FIX: Extract tech_stack FROM SPEC, not hardcoded defaults
    # ==========================================================================
    # The old code had hardcoded React/Electron/FastAPI/SQLite as defaults.
    # This caused critique to reject architectures for not meeting phantom
    # requirements that the user never specified.
    #
    # Now we ONLY include tech_stack if the spec explicitly provides it.
    
    if spec_json:
        try:
            spec_data = json.loads(spec_json) if isinstance(spec_json, str) else spec_json
            impl_stack = spec_data.get("implementation_stack")
            
            if impl_stack and isinstance(impl_stack, dict):
                # Build tech_stack from spec's implementation_stack
                stack_info = {}
                
                if impl_stack.get("language"):
                    stack_info["language"] = impl_stack["language"]
                if impl_stack.get("framework"):
                    stack_info["framework"] = impl_stack["framework"]
                if impl_stack.get("runtime"):
                    stack_info["runtime"] = impl_stack["runtime"]
                
                # Include lock status so critique knows how strict to be
                stack_info["stack_locked"] = impl_stack.get("stack_locked", False)
                stack_info["source"] = impl_stack.get("source", "spec")
                
                if stack_info.get("language") or stack_info.get("framework"):
                    context["tech_stack"] = stack_info
                    logger.info(
                        "[get_environment_context] v1.1 Using spec-defined tech_stack: %s (locked=%s)",
                        stack_info, stack_info.get("stack_locked")
                    )
                    print(f"[DEBUG] [env_context] v1.1 Spec tech_stack: {stack_info}")
                else:
                    logger.info("[get_environment_context] v1.1 Spec has implementation_stack but no language/framework - skipping tech_stack")
            else:
                logger.info("[get_environment_context] v1.1 No implementation_stack in spec - NO tech_stack constraints")
                print("[DEBUG] [env_context] v1.1 No implementation_stack in spec - critique will NOT check tech stack")
        except Exception as e:
            logger.warning("[get_environment_context] v1.1 Failed to parse spec_json: %s", e)
    else:
        logger.info("[get_environment_context] v1.1 No spec_json provided - NO tech_stack constraints")
    
    # NOTE: We do NOT add a default tech_stack!
    # If the user didn't specify a tech stack, critique should NOT enforce one.
    # This is the FIX for the phantom constraint bug.
    
    return context


# =============================================================================
# Routing Helpers
# =============================================================================

def normalize_job_type_for_high_stakes(job_type_str: str, reason: str = "") -> str:
    """Normalize various job type strings to canonical high-stakes types."""
    jt = (job_type_str or "").strip().lower().replace(" ", "_")
    
    # Map common variants
    mappings = {
        "architecture": "architecture_design",
        "arch": "architecture_design",
        "big_arch": "big_architecture",
        "security": "security_review",
        "sec_review": "security_review",
        "infra": "high_stakes_infra",
        "infrastructure": "high_stakes_infra",
        "compliance": "compliance_review",
        "legal": "high_stakes_legal",
        "medical": "high_stakes_medical",
    }
    
    return mappings.get(jt, jt)


def is_high_stakes_job(job_type_str: str) -> bool:
    """Check if job type qualifies for high-stakes pipeline."""
    normalized = normalize_job_type_for_high_stakes(job_type_str)
    return normalized in HIGH_STAKES_JOB_TYPES


def is_opus_model(model_id: str) -> bool:
    """Check if model is an Opus-tier model."""
    return "opus" in (model_id or "").lower()


def is_long_enough_for_critique(text: str) -> bool:
    """Check if response is long enough to warrant critique."""
    return len(text or "") >= MIN_CRITIQUE_CHARS


# =============================================================================
# v3.2: Evidence Loop Adapter for Architecture Draft
# =============================================================================

def _format_fulfilled_evidence(context: "JobContext") -> str:
    """Format fulfilled evidence results for re-injection into stage prompt.

    After the orchestrator dispatches tool calls from EVIDENCE_REQUEST blocks,
    the results are stored in context.fulfilled_evidence. This function formats
    them into a system message so the LLM can incorporate the evidence on its
    next pass.
    """
    parts = []
    parts.append("=" * 60)
    parts.append("FULFILLED EVIDENCE — Orchestrator Results")
    parts.append("=" * 60)
    parts.append("")
    parts.append(
        "The orchestrator has fulfilled the following EVIDENCE_REQUESTs. "
        "Use these results to CITE evidence in your architecture. "
        "Replace the corresponding EVIDENCE_REQUEST blocks with CITED "
        "claims or DECISION blocks as appropriate. "
        "Do NOT re-emit EVIDENCE_REQUESTs for fulfilled items."
    )
    parts.append("")

    for req_id, info in context.fulfilled_evidence.items():
        parts.append(f"--- Evidence for {req_id} ---")
        tools_called = info.get("tools_called", [])
        results = info.get("results", [])
        if tools_called:
            parts.append(f"  Tools called: {', '.join(tools_called)}")
        for i, result in enumerate(results):
            if isinstance(result, dict):
                # Truncate large content payloads
                for key, val in result.items():
                    if isinstance(val, str) and len(val) > 3000:
                        result[key] = val[:3000] + "... [truncated]"
                parts.append(f"  Result {i + 1}: {result}")
            else:
                parts.append(f"  Result {i + 1}: {result}")
        parts.append("")

    parts.append("=" * 60)
    parts.append(
        "NOW: Re-generate the architecture incorporating this evidence. "
        "CITE the evidence using [CITED file=\"...\" lines=\"...\"] tags. "
        "Any claims that are still unresolved should use EVIDENCE_REQUEST "
        "(if more evidence is needed) or DECISION (if you can decide now)."
    )
    parts.append("=" * 60)
    return "\n".join(parts)


def _format_force_resolve(context: "JobContext") -> str:
    """Format force-resolve instructions for unresolved CRITICAL requests.

    After max_loops, any remaining CRITICAL EVIDENCE_REQUESTs with
    fallback_if_not_found=DECISION_ALLOWED must be resolved by the stage.
    The orchestrator NEVER fabricates decisions.
    """
    parts = []
    parts.append("=" * 60)
    parts.append("FORCE RESOLVE — Evidence Not Found")
    parts.append("=" * 60)
    parts.append("")
    parts.append(
        "The following evidence requests could NOT be fulfilled after "
        "exhausting all search loops. You MUST now resolve each one as "
        "either a DECISION block (with rationale and revisit_if) or a "
        "HUMAN_REQUIRED block. Do NOT emit EVIDENCE_REQUEST for these items."
    )
    parts.append("")

    for req_id, info in context.force_resolve.items():
        parts.append(f"--- {req_id} ---")
        parts.append(f"  Original need: {info.get('original_need', 'unknown')}")
        parts.append(f"  Instruction: {info.get('instruction', '')}")
        parts.append("")

    parts.append("=" * 60)
    return "\n".join(parts)


# =============================================================================
# Block 4: Architecture Artifact Storage
# =============================================================================

def _compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content (truncated to 16 chars)."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _utc_iso() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def store_architecture_artifact(
    *,
    db,
    job_id: str,
    project_id: int,
    arch_content: str,
    spec_id: str,
    spec_hash: str,
    arch_version: int = 1,
    model: str = "",
    previous_arch_id: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Store architecture as versioned artifact with spec traceability.
    
    Creates:
    - DB record via ArtefactService (if available)
    - Filesystem mirror at jobs/{job_id}/arch/arch_v{version}.md
    
    Returns (arch_id, arch_hash, path)
    """
    arch_id = str(uuid4())
    arch_hash = _compute_content_hash(arch_content)
    path = ""
    
    # 1. Store in ArtefactService (if available) - uses static methods
    if ARTEFACTS_AVAILABLE and db:
        try:
            ArtefactService.write_artefact(
                db=db,
                project_id=project_id,
                artefact_type="architecture_doc",
                name=f"arch_{job_id}_v{arch_version}",
                content=arch_content,
                metadata={
                    "arch_id": arch_id,
                    "arch_hash": arch_hash,
                    "spec_id": spec_id,
                    "spec_hash": spec_hash,
                    "version": arch_version,
                    "model": model,
                    "previous_arch_id": previous_arch_id,
                },
                created_by_job_id=job_id,
            )
            logger.info(f"[arch] Stored in ArtefactService: {arch_id}")
        except Exception as e:
            logger.warning(f"[arch] ArtefactService storage failed: {e}")
    
    # 2. Write filesystem mirror
    if LEDGER_AVAILABLE:
        try:
            job_root = get_job_artifact_root()
            arch_dir = Path(job_root) / "jobs" / job_id / "arch"
            arch_dir.mkdir(parents=True, exist_ok=True)
            
            path = str(arch_dir / f"arch_v{arch_version}.md")
            Path(path).write_text(arch_content, encoding="utf-8")
            
            # Emit ledger events
            emit_arch_created(
                job_artifact_root=job_root,
                job_id=job_id,
                arch_id=arch_id,
                arch_version=arch_version,
                arch_hash=arch_hash,
                spec_id=spec_id,
                spec_hash=spec_hash,
                model=model,
            )
            
            emit_arch_mirror_written(
                job_artifact_root=job_root,
                job_id=job_id,
                arch_id=arch_id,
                arch_version=arch_version,
                mirror_path=path,
                checksum=arch_hash,
            )
            
            logger.info(f"[arch] Mirror written: {path}")
        except Exception as e:
            logger.warning(f"[arch] Filesystem mirror failed: {e}")
    
    return arch_id, arch_hash, path


# =============================================================================
# Main Pipeline Entry Point
# =============================================================================

async def run_high_stakes_with_critique(
    task: LLMTask,
    provider_id: str,
    model_id: str,
    envelope: JobEnvelope,
    job_type_str: str,
    file_map: Optional[str] = None,
    *,
    # Block 4-6 params (optional, passed from Spec Gate)
    db=None,
    spec_id: Optional[str] = None,
    spec_hash: Optional[str] = None,
    spec_json: Optional[str] = None,
    spec_markdown: Optional[str] = None,  # v5.0: Full POT spec with grounded evidence
    use_json_critique: bool = True,
) -> LLMResult:
    """Run high-stakes critique pipeline.
    
    v5.0: Now accepts spec_markdown - the full POT spec with grounded evidence.
    This is the PRIMARY source of truth with actual file paths, line numbers,
    and specific changes. The architecture LLM follows this spec exactly.
    
    If spec_id/spec_hash are provided (from Spec Gate), uses Block 4-6 pipeline:
    - Stores architecture as versioned artifact
    - Uses JSON critique schema
    - Runs revision loop until pass or max iterations
    
    Otherwise uses legacy prose-based critique.
    """
    logger.info(f"[critic] High-stakes pipeline: job_type={job_type_str} model={model_id}")
    
    audit_logger, trace = _maybe_start_trace(
        task, envelope, job_type_str=job_type_str, provider_id=provider_id, model_id=model_id
    )
    
    # Pre-step: Video transcription
    attachments = task.attachments or []
    flags = compute_modality_flags(attachments)
    video_attachments = flags.get("video_attachments", [])
    
    transcripts_text = ""
    if video_attachments:
        for video_att in video_attachments:
            try:
                video_path = getattr(video_att, "path", None)
                if video_path:
                    transcript = await transcribe_video_for_context(video_path)
                    transcripts_text += f"\n\n=== Video: {video_att.filename} ===\n{transcript}"
            except Exception:
                pass
    
    # Step 1: Generate draft
    draft_messages = list(envelope.messages)
    
    # =========================================================================
    # v5.0: INJECT FULL POT SPEC MARKDOWN (PRIMARY SOURCE OF TRUTH)
    # =========================================================================
    # The POT spec contains GROUNDED evidence: real file paths, real line numbers,
    # real content. This is the instruction set - the architecture must follow it.
    # Grounding IS the safety mechanism - if it says "Change line 42", that's truth.
    
    if spec_markdown:
        pot_spec_instruction = f"""{'='*70}
POT SPEC - AUTHORITATIVE SOURCE OF TRUTH (GROUNDED EVIDENCE)
{'='*70}

The following POT spec contains VERIFIED information:
- Real file paths that have been confirmed to exist
- Real line numbers pointing to actual code
- Real content excerpts from the codebase

Your architecture MUST:
1. Address EVERY item in the "Change" section below
2. NOT modify items in the "Skip" section
3. Follow the exact file paths and line numbers provided
4. NOT invent features, files, or changes beyond this spec
5. Treat ALL sections in this markdown as binding — including Acceptance Criteria,
   Constraints, Evidence Requests, and Implementation Steps. If Acceptance Criteria
   appear in the markdown below, they ARE the authoritative requirements regardless
   of any structured JSON fields. Do NOT claim acceptance criteria are empty or
   missing if they appear in this markdown.

{spec_markdown}

{'='*70}
END OF POT SPEC - Architecture must implement EXACTLY the above
{'='*70}
"""
        draft_messages.append({"role": "system", "content": pot_spec_instruction})
        
        logger.info("[high_stakes] v5.0 Injected FULL POT spec markdown (%d chars)", len(spec_markdown))
        print(f"[DEBUG] [high_stakes] v5.0 POT spec markdown injected ({len(spec_markdown)} chars)")
    
    # =========================================================================
    # v4.2 LEGACY: Extract metadata from spec_json (supplementary to POT spec)
    # =========================================================================
    # This extracts goal, stack, requirements from spec_json.
    # If spec_markdown was provided, this is supplementary context.
    # If spec_markdown was NOT provided, this is the primary anchoring.
    
    if spec_json:
        try:
            spec_data = json.loads(spec_json) if isinstance(spec_json, str) else spec_json
            
            # Build spec anchoring instruction
            spec_anchoring_parts = []
            spec_anchoring_parts.append("="*60)
            spec_anchoring_parts.append("AUTHORITATIVE SPEC (PoT) - YOU MUST HONOR THESE CONSTRAINTS")
            spec_anchoring_parts.append("="*60)
            
            # Goal
            if spec_data.get("goal"):
                spec_anchoring_parts.append(f"\nGOAL: {spec_data.get('goal')}")
            
            # Implementation Stack (v1.1 - CRITICAL for preventing stack drift)
            impl_stack = spec_data.get("implementation_stack")
            if impl_stack and isinstance(impl_stack, dict):
                stack_locked = impl_stack.get("stack_locked", False)
                language = impl_stack.get("language", "")
                framework = impl_stack.get("framework", "")
                runtime = impl_stack.get("runtime", "")
                source = impl_stack.get("source", "user discussion")
                
                spec_anchoring_parts.append("\nIMPLEMENTATION STACK:")
                if language:
                    spec_anchoring_parts.append(f"  Language: {language}")
                if framework:
                    spec_anchoring_parts.append(f"  Framework/Library: {framework}")
                if runtime:
                    spec_anchoring_parts.append(f"  Runtime: {runtime}")
                spec_anchoring_parts.append(f"  Source: {source}")
                
                if stack_locked:
                    spec_anchoring_parts.append("  ⚠️  STACK LOCKED: User explicitly confirmed this stack choice.")
                    spec_anchoring_parts.append("      You MUST use this exact technology stack.")
                    spec_anchoring_parts.append("      Do NOT substitute with alternatives (e.g., don't use Electron for Python+Pygame).")
                else:
                    spec_anchoring_parts.append("  Stack discussed but not locked. Prefer this stack unless there's a strong reason not to.")
                
                print(f"[DEBUG] [high_stakes] v4.2 Injecting implementation_stack: {language}+{framework} (locked={stack_locked})")
            
            # Requirements (MUST/SHOULD/CAN)
            requirements = spec_data.get("requirements", {})
            if requirements:
                must_reqs = requirements.get("must", [])
                should_reqs = requirements.get("should", [])
                if must_reqs:
                    spec_anchoring_parts.append("\nMUST REQUIREMENTS (non-negotiable):")
                    for i, req in enumerate(must_reqs[:10], 1):
                        spec_anchoring_parts.append(f"  {i}. {req}")
                if should_reqs:
                    spec_anchoring_parts.append("\nSHOULD REQUIREMENTS (preferred):")
                    for i, req in enumerate(should_reqs[:5], 1):
                        spec_anchoring_parts.append(f"  {i}. {req}")
            
            # Constraints
            constraints = spec_data.get("constraints", {})
            if constraints:
                spec_anchoring_parts.append("\nCONSTRAINTS:")
                for key, value in list(constraints.items())[:10]:
                    spec_anchoring_parts.append(f"  {key}: {value}")
            
            spec_anchoring_parts.append("\n" + "="*60)
            spec_anchoring_parts.append("YOUR ARCHITECTURE MUST ALIGN WITH THE ABOVE SPEC.")
            spec_anchoring_parts.append("Do NOT add requirements or change stack without explicit user approval.")
            spec_anchoring_parts.append("="*60)
            
            spec_instruction = "\n".join(spec_anchoring_parts)
            draft_messages.append({"role": "system", "content": spec_instruction})
            
            logger.info("[high_stakes] v4.2 Injected spec anchoring into architecture prompt")
            print(f"[DEBUG] [high_stakes] v4.2 Spec anchoring injected ({len(spec_instruction)} chars)")
            
        except Exception as e:
            logger.warning(f"[high_stakes] v4.2 Failed to inject spec: {e}")
            print(f"[DEBUG] [high_stakes] v4.2 Spec injection failed: {e}")
    
    # Inject spec echo instruction for Stage 3 verification
    if spec_id and spec_hash and STAGE3_AVAILABLE:
        spec_echo_instruction = build_spec_echo_instruction(spec_id, spec_hash)
        draft_messages.append({"role": "system", "content": spec_echo_instruction})
    
    if transcripts_text:
        draft_messages.append({"role": "system", "content": f"Video context:\n{transcripts_text.strip()}"})
    
    if file_map:
        draft_messages.append({"role": "system", "content": f"{file_map}\n\nRefer to files using [FILE_X] identifiers."})
    
    # =========================================================================
    # v2.0: EVIDENCE-OR-REQUEST CONTRACT
    # =========================================================================
    # Tells the architecture LLM to CITE evidence for every critical claim,
    # or emit EVIDENCE_REQUEST / DECISION / HUMAN_REQUIRED instead of guessing.
    # Must come AFTER spec injection so the LLM knows what to cite against.
    
    if _EVIDENCE_CONTRACT_AVAILABLE and EVIDENCE_CONTRACT_PROMPT:
        draft_messages.append({"role": "system", "content": EVIDENCE_CONTRACT_PROMPT})
        logger.info("[high_stakes] v2.0 Evidence contract prompt injected (%d chars)", len(EVIDENCE_CONTRACT_PROMPT))
        print(f"[DEBUG] [high_stakes] v2.0 Evidence contract prompt injected ({len(EVIDENCE_CONTRACT_PROMPT)} chars)")
    
    # v4.2: Log full draft messages for debugging
    print(f"[DEBUG] [high_stakes] v4.2 Draft messages: {len(draft_messages)} messages")
    
    if trace:
        _trace_step(trace, 'draft')
    
    # Get architecture config for max_tokens and timeout (use stage_models if available)
    _, _, arch_max_tokens, arch_timeout = _get_architecture_draft_config()
    print(f"[DEBUG] [high_stakes] Draft generation: provider={provider_id}, model={model_id}, max_tokens={arch_max_tokens}")
    
    # =========================================================================
    # v3.2: EVIDENCE FULFILLMENT LOOP
    # =========================================================================
    # If evidence loop is available AND contract prompt is injected, wrap the
    # architecture draft call with run_stage_with_evidence(). This enables:
    #   1. LLM emits EVIDENCE_REQUESTs instead of guessing
    #   2. Orchestrator dispatches tool calls (file reads, RAG, etc.)
    #   3. Evidence results injected back, LLM re-generates with real data
    #   4. After max_loops, unresolved CRITICAL items get force-resolved
    #   5. Final output has CRITICAL_CLAIMS register (only final resolutions)
    #
    # Without this: LLM sees the contract rules but gets one shot, so it
    # emits both EVIDENCE_REQUESTs AND a broken CRITICAL_CLAIMS in one pass.
    
    _use_evidence_loop = (
        _EVIDENCE_LOOP_AVAILABLE
        and _EVIDENCE_CONTRACT_AVAILABLE
        and EVIDENCE_CONTRACT_PROMPT
        and os.getenv("ASTRA_EVIDENCE_LOOP_ENABLED", "1") == "1"
    )
    
    if _use_evidence_loop:
        logger.info("[high_stakes] v3.2 Evidence loop ENABLED for architecture draft")
        print("[DEBUG] [high_stakes] v3.2 Evidence loop ENABLED")
        
        # Accumulate token counts across loop iterations
        _total_prompt_tokens = 0
        _total_completion_tokens = 0
        _total_cost = 0.0
        _raw_response = None
        
        async def _architecture_stage_fn(ctx: JobContext) -> StageResult:
            """Adapter: wrap registry_llm_call as a stage_fn for evidence loop."""
            nonlocal _total_prompt_tokens, _total_completion_tokens, _total_cost, _raw_response
            
            messages = list(draft_messages)
            
            # Inject fulfilled evidence from previous loop iteration
            if ctx.fulfilled_evidence:
                evidence_text = _format_fulfilled_evidence(ctx)
                messages.append({"role": "system", "content": evidence_text})
                logger.info(
                    "[high_stakes] v3.2 Injecting %d fulfilled evidence items (%d chars)",
                    len(ctx.fulfilled_evidence), len(evidence_text),
                )
                print(f"[DEBUG] [high_stakes] v3.2 Fulfilled evidence injected: {len(ctx.fulfilled_evidence)} items")
            
            # Inject force-resolve instructions if max loops exhausted
            if ctx.force_resolve_only and ctx.force_resolve:
                force_text = _format_force_resolve(ctx)
                messages.append({"role": "system", "content": force_text})
                logger.info(
                    "[high_stakes] v3.2 Force-resolve injected for %d items",
                    len(ctx.force_resolve),
                )
                print(f"[DEBUG] [high_stakes] v3.2 Force-resolve injected: {len(ctx.force_resolve)} items")
            
            try:
                result = await registry_llm_call(
                    provider_id=provider_id,
                    model_id=model_id,
                    messages=messages,
                    job_envelope=envelope,
                    max_tokens=arch_max_tokens,
                    timeout_seconds=arch_timeout,
                )
                _total_prompt_tokens += result.usage.prompt_tokens
                _total_completion_tokens += result.usage.completion_tokens
                _total_cost += result.usage.cost_estimate
                _raw_response = result.raw_response
                return StageResult(output=result.content, success=True)
            except Exception as exc:
                return StageResult(output="", success=False, error=str(exc))
        
        # Create job context (evidence bundle can be loaded from collector)
        evidence_bundle = None
        try:
            from app.pot_spec.evidence_collector import load_evidence
            evidence_bundle = load_evidence()
        except Exception:
            pass
        
        job_ctx = JobContext(evidence_bundle=evidence_bundle)
        
        # Run the evidence loop (max_loops=2 by default)
        max_evidence_loops = int(os.getenv("ASTRA_EVIDENCE_MAX_LOOPS", "2"))
        stage_result = await run_stage_with_evidence(
            stage_name="critical",
            stage_fn=_architecture_stage_fn,
            context=job_ctx,
            max_loops=max_evidence_loops,
        )
        
        if not stage_result.success and stage_result.error:
            err_msg = f"High-stakes draft failed: {stage_result.error}"
            if trace:
                _trace_error(trace, 'draft', err_msg)
            _maybe_complete_trace(audit_logger, trace, success=False, error_message=err_msg)
            return LLMResult(
                content=err_msg, provider=provider_id, model=model_id,
                finish_reason="error", error_message=err_msg,
                prompt_tokens=0, completion_tokens=0, total_tokens=0, cost_usd=0.0, raw_response=None,
            )
        
        # Log unresolved HUMAN_REQUIRED items
        if stage_result.unresolved_human_required:
            logger.warning(
                "[high_stakes] v3.2 %d HUMAN_REQUIRED items unresolved",
                len(stage_result.unresolved_human_required),
            )
            print(f"[DEBUG] [high_stakes] v3.2 HUMAN_REQUIRED: {len(stage_result.unresolved_human_required)} items")
        
        draft = LLMResult(
            content=stage_result.output,
            provider=provider_id,
            model=model_id,
            finish_reason="stop",
            error_message=None,
            prompt_tokens=_total_prompt_tokens,
            completion_tokens=_total_completion_tokens,
            total_tokens=_total_prompt_tokens + _total_completion_tokens,
            cost_usd=_total_cost,
            raw_response=_raw_response,
        )
        
        logger.info(
            "[high_stakes] v3.2 Evidence loop complete: %d chars, %d tokens, $%.4f",
            len(draft.content), draft.total_tokens, draft.cost_usd,
        )
        print(f"[DEBUG] [high_stakes] v3.2 Evidence loop complete: {len(draft.content)} chars")
    
    else:
        # =====================================================================
        # Legacy: Single-pass draft (no evidence loop)
        # =====================================================================
        if not _use_evidence_loop and _EVIDENCE_CONTRACT_AVAILABLE:
            logger.info("[high_stakes] v3.2 Evidence loop DISABLED (env or import); single-pass draft")
            print("[DEBUG] [high_stakes] v3.2 Evidence loop disabled, single-pass mode")
        
        try:
            draft_result = await registry_llm_call(
                provider_id=provider_id,
                model_id=model_id,
                messages=draft_messages,
                job_envelope=envelope,
                max_tokens=arch_max_tokens,
                timeout_seconds=arch_timeout,
            )
        except Exception as exc:
            err_msg = f"High-stakes draft failed: {exc}"
            if trace:
                _trace_error(trace, 'draft', err_msg)
            _maybe_complete_trace(audit_logger, trace, success=False, error_message=err_msg)
            return LLMResult(
                content=err_msg, provider=provider_id, model=model_id,
                finish_reason="error", error_message=err_msg,
                prompt_tokens=0, completion_tokens=0, total_tokens=0, cost_usd=0.0, raw_response=None,
            )
        
        draft = LLMResult(
            content=draft_result.content,
            provider=provider_id,
            model=model_id,
            finish_reason="stop",
            error_message=None,
            prompt_tokens=draft_result.usage.prompt_tokens,
            completion_tokens=draft_result.usage.completion_tokens,
            total_tokens=draft_result.usage.total_tokens,
            cost_usd=draft_result.usage.cost_estimate,
            raw_response=draft_result.raw_response,
        )
    
    if trace:
        _trace_step(trace, 'draft_done')
    
    # Check if critique needed
    if not is_long_enough_for_critique(draft.content):
        logger.warning("[critic] Draft too short for critique")
        _maybe_complete_trace(audit_logger, trace, success=True)
        draft.routing_decision = {"job_type": job_type_str, "provider": provider_id, "model": model_id, "reason": "draft too short"}
        return draft
    
    # Extract original request
    user_messages = [m for m in task.messages if m.get("role") == "user"]
    original_request = user_messages[-1].get("content", "") if user_messages else ""
    
    # =========================================================================
    # Block 4-6: Full artifact pipeline (if spec provided)
    # =========================================================================
    if spec_id and spec_hash and use_json_critique:
        logger.info("[critic] Using Block 4-6 artifact pipeline")
        
        job_id = str(envelope.job_id)
        project_id = int(getattr(envelope, "project_id", 0))
        
        # Store initial architecture (Block 4)
        arch_id, arch_hash, _ = store_architecture_artifact(
            db=db,
            job_id=job_id,
            project_id=project_id,
            arch_content=draft.content,
            spec_id=spec_id,
            spec_hash=spec_hash,
            arch_version=1,
            model=model_id,
        )
        
        if trace:
            _trace_step(trace, 'arch_stored', arch_id=arch_id)
        
        # Run revision loop (Block 5 + 6)
        # v1.1 FIX: Pass spec_json to get_environment_context() to avoid phantom constraints
        env_context = get_environment_context(spec_json=spec_json) if job_type_str in HIGH_STAKES_JOB_TYPES else None
        
        # v5.0: Pass spec_markdown to revision loop for grounded critique
        if spec_markdown:
            print(f"[DEBUG] [high_stakes] v5.0 Passing spec_markdown ({len(spec_markdown)} chars) to revision loop")
        
        final_content, final_version, passed, final_critique = await run_revision_loop(
            db=db,
            job_id=job_id,
            project_id=project_id,
            arch_content=draft.content,
            arch_id=arch_id,
            spec_id=spec_id,
            spec_hash=spec_hash,
            spec_json=spec_json,
            spec_markdown=spec_markdown,
            original_request=original_request,
            opus_model_id=model_id,
            envelope=envelope,
            env_context=env_context,
            store_architecture_fn=store_architecture_artifact,
        )
        
        if trace:
            _trace_step(trace, 'revision_loop_done', version=final_version, passed=passed)
        
        _maybe_complete_trace(audit_logger, trace, success=True)
        
        return LLMResult(
            content=final_content,
            provider=provider_id,
            model=model_id,
            finish_reason="stop",
            error_message=None,
            prompt_tokens=draft.prompt_tokens,
            completion_tokens=draft.completion_tokens,
            total_tokens=draft.total_tokens,
            cost_usd=draft.cost_usd,
            raw_response=None,
            routing_decision={
                "job_type": job_type_str,
                "provider": provider_id,
                "model": model_id,
                "reason": f"Block 4-6 pipeline: v{final_version}, passed={passed}",
                "arch_id": arch_id,
                "final_version": final_version,
                "critique_passed": passed,
                "blocking_issues": len(final_critique.blocking_issues),
            },
        )
    
    # =========================================================================
    # Legacy pipeline (prose critique, single revision)
    # =========================================================================
    logger.info("[critic] Using legacy prose critique pipeline")
    
    # Step 2: Critique
    # v1.1 FIX: Pass spec_json to get_environment_context() to avoid phantom constraints
    env_context = get_environment_context(spec_json=spec_json) if job_type_str in HIGH_STAKES_JOB_TYPES else None
    critique = await call_gemini_critic(
        original_task=task,
        draft_result=draft,
        job_type_str=job_type_str,
        envelope=envelope,
        env_context=env_context,
    )
    
    if not critique:
        logger.warning("[critic] Critique failed; returning draft")
        _maybe_complete_trace(audit_logger, trace, success=True)
        draft.routing_decision = {"job_type": job_type_str, "provider": provider_id, "model": model_id, "reason": "critique failed"}
        return draft
    
    if trace:
        _trace_step(trace, 'critique_done')
    
    # Step 3: Revision
    revision = await call_opus_revision(
        original_task=task, draft_result=draft, critique_result=critique,
        opus_model_id=model_id, envelope=envelope
    )
    
    if not revision:
        logger.warning("[critic] Revision failed; returning draft")
        _maybe_complete_trace(audit_logger, trace, success=True)
        draft.routing_decision = {"job_type": job_type_str, "provider": provider_id, "model": model_id, "reason": "revision failed"}
        return draft
    
    if trace:
        _trace_step(trace, 'revision_done')
    
    _maybe_complete_trace(audit_logger, trace, success=True)
    
    revision.routing_decision = {
        "job_type": job_type_str,
        "provider": provider_id,
        "model": revision.model,
        "reason": "Legacy: Opus draft → Gemini critique → Opus revision",
        "critique_pipeline": {
            "draft_tokens": draft.total_tokens,
            "critique_tokens": critique.total_tokens,
            "revision_tokens": revision.total_tokens,
            "total_cost": draft.cost_usd + critique.cost_usd + revision.cost_usd,
        },
    }
    
    return revision


__all__ = [
    # Configuration
    "HIGH_STAKES_JOB_TYPES",
    "MIN_CRITIQUE_CHARS",
    # Routing helpers
    "normalize_job_type_for_high_stakes",
    "is_high_stakes_job",
    "is_opus_model",
    "is_long_enough_for_critique",
    "get_environment_context",
    "_map_to_phase4_job_type",
    # Block 4: Architecture storage
    "store_architecture_artifact",
    # Re-exports from critique.py
    "call_json_critic",
    "store_critique_artifact",
    "call_gemini_critic",
    "build_critique_prompt",
    # Re-exports from revision.py
    "call_revision",
    "run_revision_loop",
    "call_opus_revision",
    # Main entry
    "run_high_stakes_with_critique",
]