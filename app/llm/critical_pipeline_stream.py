# FILE: app/llm/critical_pipeline_stream.py
# Purpose: Backward-compatibility shim for Critical Pipeline.
# Called-by: app.llm.routing.handler_registry
# Depends-on: app.llm.critical_pipeline
# Last-renovated: 2026-06-11
"""
Backward-compatibility shim for Critical Pipeline.

The implementation has been refactored into the ``app.llm.critical_pipeline``
package (v3.0).  This file re-exports all public symbols so that existing
imports continue to work without modification:

    from app.llm.critical_pipeline_stream import generate_critical_pipeline_stream
    from app.llm.critical_pipeline_stream import JobKind, micro_quickcheck, ...

See app/llm/critical_pipeline/ for the full source.
"""

# Re-export everything from the package
from app.llm.critical_pipeline import (  # noqa: F401
    generate_critical_pipeline_stream,
    JobKind,
    MicroQuickcheckResult,
    micro_quickcheck,
    ScanQuickcheckResult,
    scan_quickcheck,
    CriticalPipelineEvidence,
    gather_critical_pipeline_evidence,
    read_file_for_critical_pipeline,
    list_directory_for_critical_pipeline,
)

__all__ = [
    "generate_critical_pipeline_stream",
    "JobKind",
    "MicroQuickcheckResult",
    "micro_quickcheck",
    "ScanQuickcheckResult",
    "scan_quickcheck",
    "CriticalPipelineEvidence",
    "gather_critical_pipeline_evidence",
    "read_file_for_critical_pipeline",
    "list_directory_for_critical_pipeline",
]
