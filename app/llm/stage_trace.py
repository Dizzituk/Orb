# FILE: app/llm/stage_trace.py
"""
Stage Trace - Centralized tracing for ASTRA command execution.

Provides:
- Correlation ID generation and propagation
- Stage boundary events (ENTER/EXIT)
- Model resolution audit logging
- Failure tracking with context

v1.0 (2026-01): Initial implementation

Usage:
    trace = StageTrace.start("run_critical_pipeline", project_id=18)
    trace.enter_stage("spec_gate", provider="openai", model="gpt-4.1-mini")
    # ... do work ...
    trace.exit_stage("spec_gate", success=True, tokens_used=1500)
    trace.finish(success=True)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Enable verbose console output for debugging
STAGE_TRACE_VERBOSE = os.getenv("STAGE_TRACE_VERBOSE", "1") == "1"

# Enable ledger persistence
STAGE_TRACE_LEDGER = os.getenv("STAGE_TRACE_LEDGER", "1") == "1"


# =============================================================================
# LEDGER HELPERS
# =============================================================================

def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_trace_event(job_id: str, event: dict) -> None:
    """Append trace event to ledger."""
    if not STAGE_TRACE_LEDGER:
        return
    
    try:
        from app.pot_spec.ledger import append_event
        from app.pot_spec.service import get_job_artifact_root
        
        job_root = get_job_artifact_root()
        append_event(job_artifact_root=job_root, job_id=job_id, event=event)
    except Exception as e:
        logger.debug(f"[stage_trace] Failed to append ledger event: {e}")


# =============================================================================
# STAGE ENTRY
# =============================================================================

@dataclass
class StageEntry:
    """Record of a single stage execution."""
    stage_name: str
    entered_at: float
    exited_at: Optional[float] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    env_provider: Optional[str] = None  # What env var said
    env_model: Optional[str] = None     # What env var said
    success: bool = True
    error_message: Optional[str] = None
    tokens_used: int = 0
    cost_estimate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> int:
        if self.exited_at is None:
            return 0
        return int((self.exited_at - self.entered_at) * 1000)
    
    def to_dict(self) -> dict:
        return {
            "stage": self.stage_name,
            "provider": self.provider,
            "model": self.model,
            "env_provider": self.env_provider,
            "env_model": self.env_model,
            "success": self.success,
            "error": self.error_message,
            "tokens": self.tokens_used,
            "cost": self.cost_estimate,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


# =============================================================================
# STAGE TRACE
# =============================================================================

@dataclass
class StageTrace:
    """
    Trace context for a single ASTRA command execution.
    
    Tracks:
    - Correlation ID (command_id)
    - Job ID (if applicable)
    - Project ID
    - All stage entries/exits
    - Final outcome
    """
    command_id: str
    command_type: str
    project_id: Optional[int] = None
    job_id: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    stages: List[StageEntry] = field(default_factory=list)
    current_stage: Optional[StageEntry] = None
    success: bool = True
    error_message: Optional[str] = None
    final_outcome: Optional[str] = None
    
    @classmethod
    def start(
        cls,
        command_type: str,
        project_id: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> "StageTrace":
        """Start a new trace for a command."""
        command_id = f"cmd_{uuid4().hex[:12]}"
        
        trace = cls(
            command_id=command_id,
            command_type=command_type,
            project_id=project_id,
            job_id=job_id,
        )
        
        # Log command start
        if STAGE_TRACE_VERBOSE:
            print(f"[STAGE_TRACE] ═══════════════════════════════════════════════════════")
            print(f"[STAGE_TRACE] COMMAND START: {command_type}")
            print(f"[STAGE_TRACE] command_id={command_id}, project_id={project_id}, job_id={job_id}")
            print(f"[STAGE_TRACE] ═══════════════════════════════════════════════════════")
        
        logger.info(f"[stage_trace] COMMAND_START: type={command_type}, id={command_id}, project={project_id}")
        
        # Emit ledger event
        if job_id:
            _append_trace_event(job_id, {
                "event": "COMMAND_START",
                "command_id": command_id,
                "command_type": command_type,
                "project_id": project_id,
                "job_id": job_id,
                "status": "started",
                "ts": _utc_ts(),
            })
        
        return trace
    
    def set_job_id(self, job_id: str) -> None:
        """Set job ID after creation (e.g., when envelope is created)."""
        self.job_id = job_id
        if STAGE_TRACE_VERBOSE:
            print(f"[STAGE_TRACE] Job ID assigned: {job_id}")
    
    def enter_stage(
        self,
        stage_name: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **metadata,
    ) -> None:
        """Record entering a stage."""
        # Capture what env vars currently say for audit
        env_provider = os.getenv("SPEC_GATE_PROVIDER", "<not set>") if "spec_gate" in stage_name.lower() else None
        env_model = os.getenv("OPENAI_SPEC_GATE_MODEL", "<not set>") if "spec_gate" in stage_name.lower() else None
        
        entry = StageEntry(
            stage_name=stage_name,
            entered_at=time.time(),
            provider=provider,
            model=model,
            env_provider=env_provider,
            env_model=env_model,
            metadata=metadata,
        )
        
        self.current_stage = entry
        
        # Log stage entry
        if STAGE_TRACE_VERBOSE:
            print(f"[STAGE_TRACE] ┌─ ENTER: {stage_name}")
            print(f"[STAGE_TRACE] │  command_id={self.command_id}")
            if provider:
                print(f"[STAGE_TRACE] │  provider={provider}, model={model}")
            if env_provider and env_provider != "<not set>":
                print(f"[STAGE_TRACE] │  env: SPEC_GATE_PROVIDER={env_provider}, OPENAI_SPEC_GATE_MODEL={env_model}")
            if metadata:
                print(f"[STAGE_TRACE] │  metadata={metadata}")
        
        logger.info(f"[stage_trace] STAGE_ENTER: {stage_name} (cmd={self.command_id}, provider={provider}, model={model})")
        
        # Emit ledger event
        if self.job_id:
            _append_trace_event(self.job_id, {
                "event": "STAGE_ENTER",
                "command_id": self.command_id,
                "stage": stage_name,
                "provider": provider,
                "model": model,
                "env_provider": env_provider,
                "env_model": env_model,
                "metadata": metadata,
                "status": "ok",
                "ts": _utc_ts(),
            })
    
    def log_model_call(
        self,
        provider: str,
        model: str,
        purpose: str = "primary",
        tokens_used: int = 0,
        cost_estimate: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Log an LLM model call within the current stage."""
        if STAGE_TRACE_VERBOSE:
            status = "✓" if success else "✗"
            print(f"[STAGE_TRACE] │  {status} LLM_CALL: {provider}/{model} ({purpose})")
            if tokens_used:
                print(f"[STAGE_TRACE] │    tokens={tokens_used}, cost=${cost_estimate:.4f}")
            if error:
                print(f"[STAGE_TRACE] │    ERROR: {error}")
        
        logger.info(f"[stage_trace] LLM_CALL: {provider}/{model} purpose={purpose} success={success}")
        
        # Update current stage metrics
        if self.current_stage:
            self.current_stage.provider = provider
            self.current_stage.model = model
            self.current_stage.tokens_used += tokens_used
            self.current_stage.cost_estimate += cost_estimate
            if not success:
                self.current_stage.success = False
                self.current_stage.error_message = error
        
        # Emit ledger event
        if self.job_id:
            _append_trace_event(self.job_id, {
                "event": "LLM_CALL",
                "command_id": self.command_id,
                "stage": self.current_stage.stage_name if self.current_stage else "unknown",
                "provider": provider,
                "model": model,
                "purpose": purpose,
                "tokens": tokens_used,
                "cost": cost_estimate,
                "success": success,
                "error": error,
                "status": "ok" if success else "error",
                "ts": _utc_ts(),
            })
    
    def exit_stage(
        self,
        stage_name: str,
        success: bool = True,
        error: Optional[str] = None,
        tokens_used: int = 0,
        cost_estimate: float = 0.0,
        **metadata,
    ) -> None:
        """Record exiting a stage."""
        if self.current_stage and self.current_stage.stage_name == stage_name:
            self.current_stage.exited_at = time.time()
            self.current_stage.success = success
            self.current_stage.error_message = error
            self.current_stage.tokens_used += tokens_used
            self.current_stage.cost_estimate += cost_estimate
            self.current_stage.metadata.update(metadata)
            
            # Move to completed stages
            self.stages.append(self.current_stage)
            entry = self.current_stage
            self.current_stage = None
        else:
            # Stage mismatch - create a synthetic entry
            entry = StageEntry(
                stage_name=stage_name,
                entered_at=time.time(),
                exited_at=time.time(),
                success=success,
                error_message=error,
                tokens_used=tokens_used,
                cost_estimate=cost_estimate,
                metadata=metadata,
            )
            self.stages.append(entry)
        
        # Update overall success
        if not success:
            self.success = False
        
        # Log stage exit
        status_icon = "✓" if success else "✗"
        if STAGE_TRACE_VERBOSE:
            print(f"[STAGE_TRACE] └─ EXIT: {stage_name} [{status_icon}] ({entry.duration_ms}ms)")
            if error:
                print(f"[STAGE_TRACE]    ERROR: {error}")
        
        logger.info(f"[stage_trace] STAGE_EXIT: {stage_name} success={success} duration={entry.duration_ms}ms")
        
        # Emit ledger event
        if self.job_id:
            _append_trace_event(self.job_id, {
                "event": "STAGE_EXIT",
                "command_id": self.command_id,
                "stage": stage_name,
                "success": success,
                "error": error,
                "duration_ms": entry.duration_ms,
                "tokens": entry.tokens_used,
                "cost": entry.cost_estimate,
                "metadata": metadata,
                "status": "ok" if success else "error",
                "ts": _utc_ts(),
            })
    
    def record_routing_failure(
        self,
        reason: str,
        expected_handler: Optional[str] = None,
        fallback_action: Optional[str] = None,
    ) -> None:
        """Record a routing failure (e.g., handler not available)."""
        self.success = False
        self.error_message = reason
        
        if STAGE_TRACE_VERBOSE:
            print(f"[STAGE_TRACE] ⚠ ROUTING FAILURE: {reason}")
            if expected_handler:
                print(f"[STAGE_TRACE]   expected_handler={expected_handler}")
            if fallback_action:
                print(f"[STAGE_TRACE]   fallback_action={fallback_action}")
        
        logger.warning(f"[stage_trace] ROUTING_FAILURE: {reason} (cmd={self.command_id})")
        
        # Emit ledger event
        if self.job_id:
            _append_trace_event(self.job_id, {
                "event": "ROUTING_FAILURE",
                "command_id": self.command_id,
                "reason": reason,
                "expected_handler": expected_handler,
                "fallback_action": fallback_action,
                "status": "error",
                "ts": _utc_ts(),
            })
    
    def finish(
        self,
        success: bool = True,
        outcome: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Finish the trace."""
        self.finished_at = time.time()
        self.success = success and self.success  # Don't override if already failed
        self.final_outcome = outcome
        if error:
            self.error_message = error
        
        duration_ms = int((self.finished_at - self.started_at) * 1000)
        total_tokens = sum(s.tokens_used for s in self.stages)
        total_cost = sum(s.cost_estimate for s in self.stages)
        
        # Log summary
        if STAGE_TRACE_VERBOSE:
            status = "SUCCESS" if self.success else "FAILED"
            print(f"[STAGE_TRACE] ═══════════════════════════════════════════════════════")
            print(f"[STAGE_TRACE] COMMAND {status}: {self.command_type}")
            print(f"[STAGE_TRACE] command_id={self.command_id}, duration={duration_ms}ms")
            print(f"[STAGE_TRACE] stages={len(self.stages)}, tokens={total_tokens}, cost=${total_cost:.4f}")
            if self.error_message:
                print(f"[STAGE_TRACE] ERROR: {self.error_message}")
            print(f"[STAGE_TRACE] ═══════════════════════════════════════════════════════")
        
        logger.info(f"[stage_trace] COMMAND_FINISH: {self.command_type} success={self.success} duration={duration_ms}ms")
        
        # Emit summary ledger event
        if self.job_id:
            _append_trace_event(self.job_id, {
                "event": "COMMAND_FINISH",
                "command_id": self.command_id,
                "command_type": self.command_type,
                "project_id": self.project_id,
                "job_id": self.job_id,
                "success": self.success,
                "outcome": self.final_outcome,
                "error": self.error_message,
                "duration_ms": duration_ms,
                "stages_completed": len(self.stages),
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "stage_summary": [s.to_dict() for s in self.stages],
                "status": "ok" if self.success else "error",
                "ts": _utc_ts(),
            })
    
    def to_summary(self) -> dict:
        """Generate summary dict for logging/debugging."""
        duration_ms = int((self.finished_at or time.time()) - self.started_at) * 1000
        return {
            "command_id": self.command_id,
            "command_type": self.command_type,
            "project_id": self.project_id,
            "job_id": self.job_id,
            "success": self.success,
            "error": self.error_message,
            "outcome": self.final_outcome,
            "duration_ms": duration_ms,
            "stages": [s.to_dict() for s in self.stages],
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_env_model_audit() -> dict:
    """Get current env var values for model selection audit."""
    return {
        "SPEC_GATE_PROVIDER": os.getenv("SPEC_GATE_PROVIDER", "<not set>"),
        "OPENAI_SPEC_GATE_MODEL": os.getenv("OPENAI_SPEC_GATE_MODEL", "<not set>"),
        "OPENAI_DEFAULT_MODEL": os.getenv("OPENAI_DEFAULT_MODEL", "<not set>"),
        "ANTHROPIC_OPUS_MODEL": os.getenv("ANTHROPIC_OPUS_MODEL", "<not set>"),
        "ANTHROPIC_SONNET_MODEL": os.getenv("ANTHROPIC_SONNET_MODEL", "<not set>"),
    }


def log_model_resolution(
    stage: str,
    resolved_provider: str,
    resolved_model: str,
    env_vars: Optional[dict] = None,
) -> None:
    """Log model resolution for audit purposes."""
    env_vars = env_vars or get_env_model_audit()
    
    print(f"[MODEL_AUDIT] {stage}: provider={resolved_provider}, model={resolved_model}")
    print(f"[MODEL_AUDIT]   env: {env_vars}")
    
    logger.info(f"[model_audit] {stage}: provider={resolved_provider}, model={resolved_model}, env={env_vars}")


__all__ = [
    "StageTrace",
    "StageEntry",
    "get_env_model_audit",
    "log_model_resolution",
]