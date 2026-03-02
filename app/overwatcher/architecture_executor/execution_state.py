# FILE: app/overwatcher/architecture_executor/execution_state.py
"""
Shared execution state for the architecture executor.

Contains the ExecutionContext dataclass that holds all mutable state
accumulated across the task processing loop. Passed by reference
to sub-modules so they can read and update shared context.

v2.1 (2026-02): Extracted from orchestrator.py monolith.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass
class ExecutionContext:
    """All mutable state for a single architecture execution run."""

    # --- Identity ---
    spec_id: str
    job_id: str
    architecture_content: str
    architecture_path: str
    artifact_root: str

    # --- Sandbox ---
    sandbox_base: str = ""

    # --- LLM config ---
    impl_provider: str = "anthropic"
    impl_model: str = "claude-sonnet-4-5-20250929"
    impl_max_tokens: int = 16384
    llm_call_fn: Optional[Callable] = None

    # --- File operations ---
    new_files: List[Dict[str, Any]] = field(default_factory=list)
    modified_files: List[Dict[str, Any]] = field(default_factory=list)
    total_operations: int = 0

    # --- Cross-file context (v2.3) ---
    job_context: Dict[str, str] = field(default_factory=dict)
    router_registrations: Dict[str, str] = field(default_factory=dict)
    created_file_contents: Dict[str, str] = field(default_factory=dict)

    # --- Import validation (v5.11) ---
    existing_sandbox_files: Set[str] = field(default_factory=set)
    parent_module_files: Set[str] = field(default_factory=set)
    available_modules_evidence: str = ""

    # --- Interface contract ---
    interface_contract: str = ""

    # --- Results ---
    trace: List[Dict[str, Any]] = field(default_factory=list)
    artifacts_written: List[str] = field(default_factory=list)
    files_created: int = 0
    files_modified: int = 0
    files_failed: int = 0
    start_time: float = field(default_factory=time.time)

    # --- Boot check ---
    skip_boot_check: bool = False
    manifest_all_files: Optional[Set[str]] = None

    # --- Scaffold Engine (v1.0) ---
    scaffold_result: Any = None

    def elapsed_ms(self) -> int:
        return int((time.time() - self.start_time) * 1000)

    def add_trace(self, stage: str, status: str, details: Optional[Dict] = None):
        self.trace.append({
            "stage": stage,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
        })
        # Emit to Build Journal (fire-and-forget)
        try:
            from app.experience.journal_writer import emit_from_trace
            import os
            _job_dir = os.path.join(self.artifact_root, "jobs", self.job_id)
            emit_from_trace(
                job_id=self.job_id,
                job_dir=_job_dir,
                trace_stage=stage,
                trace_status=status,
                trace_details=details,
            )
        except Exception:
            pass

    @property
    def total_succeeded(self) -> int:
        return self.files_created + self.files_modified

    @property
    def success(self) -> bool:
        return self.total_succeeded > 0 and self.files_failed == 0


__all__ = ["ExecutionContext"]
