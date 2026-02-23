# FILE: app/llm/pipeline/high_stakes.py
"""High-stakes critique pipeline - Main orchestrator.

Implements Blocks 4, 5, 6 of the PoT (Proof of Thought) system:
  Block 4: Architecture generation as versioned artifact with spec traceability
  Block 5: Structured JSON critique with blocking/non-blocking issues (critique.py)
  Block 6: Revision loop until critique passes (revision.py)

Helpers in _high_stakes_helpers.py. Utilities in _high_stakes_utils.py.
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
from app.llm.pipeline._high_stakes_utils import AUDIT_ENABLED, _compute_content_hash, _format_force_resolve, _format_fulfilled_evidence, _maybe_complete_trace, _trace_error, _trace_step, _utc_iso

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
    from app.llm.stage_models import get_critical_pipeline_config as get_architecture_config
    _STAGE_MODELS_AVAILABLE = True
except ImportError:
    _STAGE_MODELS_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

logger = logging.getLogger(__name__)
# =============================================================================
# Helpers (extracted to _high_stakes_helpers.py)
# =============================================================================
from app.llm.pipeline._high_stakes_helpers import (
    MIN_CRITIQUE_CHARS,
    _get_architecture_draft_config,
    OPUS_DRAFT_MAX_TOKENS,
    OPUS_TIMEOUT_SECONDS,
    HIGH_STAKES_JOB_TYPES,
    _maybe_start_trace,
    get_environment_context,
    normalize_job_type_for_high_stakes,
    is_high_stakes_job,
    is_opus_model,
    is_long_enough_for_critique,
    store_architecture_artifact,
)



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
    segment_contract_markdown: Optional[str] = None,  # v5.4 Phase 2B: Interface contract for critique
    segment_file_scope: Optional[List[str]] = None,  # v5.18: File scope for architecture sanitiser
    enrichment_markdown: Optional[str] = None,  # v5.18: AST-extracted symbols for critique/revision
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
6. FILE SIZE CONSTRAINT: Design all output files to be under 20 KB (~500 lines)
   each. Prefer single-responsibility modules with one primary function per file.
   If a file would exceed 20 KB, decompose it into smaller focused modules. Only
   exceed 20 KB if there is a solid reason the logic cannot be decomposed further.
   Do NOT leave large orchestration blobs — split them into thin coordinators that
   call focused sub-modules.

{spec_markdown}

{'='*70}
END OF POT SPEC - Architecture must implement EXACTLY the above
{'='*70}
"""
        draft_messages.append({"role": "system", "content": pot_spec_instruction})
        
        logger.info("[high_stakes] v5.0 Injected FULL POT spec markdown (%d chars)", len(spec_markdown))
        print(f"[DEBUG] [high_stakes] v5.0 POT spec markdown injected ({len(spec_markdown)} chars)")
    
    # =========================================================================
    # v5.5 PHASE 4B: Foundation Templates — inject for greenfield CREATE jobs
    # =========================================================================
    try:
        _spec_data_for_templates = {}
        if spec_json:
            try:
                _spec_data_for_templates = json.loads(spec_json) if isinstance(spec_json, str) else (spec_json or {})
            except Exception:
                pass

        _job_kind = _spec_data_for_templates.get("job_kind", "")
        _impl_stack = _spec_data_for_templates.get("implementation_stack", {})

        # Only inject for architecture/CREATE jobs (not refactors or simple edits)
        if _job_kind in ("architecture", "create", "") and spec_markdown:
            from app.llm.critical_pipeline.foundation_templates import match_templates

            _tech_dict = {}
            if isinstance(_impl_stack, dict):
                _tech_dict = {k: str(v) for k, v in _impl_stack.items() if v}

            _matched = match_templates(
                tech_stack=_tech_dict,
                spec_text=spec_markdown,
                max_templates=4,
            )

            if _matched.count > 0:
                _tmpl_markdown = _matched.format_for_prompt()
                draft_messages.append({"role": "system", "content": _tmpl_markdown})
                logger.info(
                    "[high_stakes] v5.5 Foundation templates injected: %d templates (%d chars)",
                    _matched.count, len(_tmpl_markdown),
                )
                print(f"[DEBUG] [high_stakes] v5.5 FOUNDATION TEMPLATES: {_matched.count} injected")
    except ImportError:
        logger.debug("[high_stakes] v5.5 Foundation templates module not available")
    except Exception as _ft_err:
        logger.warning("[high_stakes] v5.5 Foundation template matching failed (non-fatal): %s", _ft_err)

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
                    stage="architecture",  # v2.2: Cost tracking
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
                stage="architecture",  # v2.2: Cost tracking
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
        
        # v5.18: Sanitise draft BEFORE storing and BEFORE critique loop.
        # This catches hallucinated paths (package self-naming, out-of-scope files,
        # phantom evidence references) so the critique loop doesn't waste iterations
        # trying to fix structurally invalid architectures.
        _sanitised_draft = draft.content
        try:
            from app.orchestrator.architecture_sanitiser import sanitise_architecture
            _sanitised_draft, _san_result = sanitise_architecture(
                arch_text=draft.content,
                file_scope=segment_file_scope,
                segment_id=str(getattr(envelope, 'job_id', 'unknown')),
            )
            if _san_result.had_fixes:
                draft.content = _sanitised_draft
                logger.info(
                    "[high_stakes] v5.18 Sanitiser applied %d fix(es) before critique loop",
                    _san_result.fix_count,
                )
                print(f"[DEBUG] [high_stakes] v5.18 Sanitiser: {_san_result.summary()}")
        except ImportError:
            pass  # Sanitiser not available — continue without it
        except Exception as _san_err:
            logger.warning("[high_stakes] v5.18 Sanitiser error (non-fatal): %s", _san_err)

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
            segment_contract_markdown=segment_contract_markdown,
            enrichment_markdown=enrichment_markdown,
        )
        
        if trace:
            _trace_step(trace, 'revision_loop_done', version=final_version, passed=passed)
        
        _maybe_complete_trace(audit_logger, trace, success=True)
        
        # v2.2: Cache successful architectures for reuse
        if passed:
            try:
                from app.orchestrator.architecture_cache import store_architecture as _store_arch_cache
                _arch_goal = ""
                _arch_files = []
                if spec_json:
                    _spec_data = json.loads(spec_json) if isinstance(spec_json, str) else spec_json
                    _arch_goal = _spec_data.get("goal", "")
                _store_arch_cache(
                    goal=_arch_goal,
                    file_targets=segment_file_scope or [],
                    arch_content=final_content,
                    spec_hash=spec_hash or "",
                    model_used=model_id,
                    critique_passed=True,
                )
            except Exception:
                pass  # Cache store failure is never fatal
        
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
