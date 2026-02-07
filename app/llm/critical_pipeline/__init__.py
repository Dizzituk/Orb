# FILE: app/llm/critical_pipeline/__init__.py
"""
Critical Pipeline package — refactored from critical_pipeline_stream.py (v3.0).

All public symbols are re-exported here for backward compatibility so that
existing imports like:

    from app.llm.critical_pipeline_stream import generate_critical_pipeline_stream

continue to work via the compatibility shim in critical_pipeline_stream.py.
"""

from app.llm.critical_pipeline.stream_handler import generate_critical_pipeline_stream
from app.llm.critical_pipeline.job_classification import JobKind
from app.llm.critical_pipeline.quickcheck_micro import (
    MicroQuickcheckResult,
    micro_quickcheck,
)
from app.llm.critical_pipeline.quickcheck_scan import (
    ScanQuickcheckResult,
    scan_quickcheck,
)
from app.llm.critical_pipeline.evidence import (
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
