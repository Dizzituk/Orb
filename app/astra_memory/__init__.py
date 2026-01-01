# FILE: app/astra_memory/__init__.py
"""
ASTRA Memory System (AstraJob 5)

Three layers of memory:
1. AstraJob-local: Per-AstraJob state, execution timeline, Overwatcher verdicts
2. Global brain: Cross-AstraJob preferences, patterns, lessons
3. Overwatcher state: Risk scores, intervention history, cross-AstraJob patterns

Tables:
- astra_jobs: AstraJob registry with spec/arch links
- astra_job_files: Files touched (Atlas link)
- astra_job_events: Ledger index
- astra_job_chunks: Execution chunks
- astra_overwatch_summary: Per-AstraJob Overwatcher stats
- astra_global_prefs: Cross-AstraJob preferences
- astra_overwatch_patterns: Cross-AstraJob patterns
"""

from app.astra_memory.models import (
    AstraJob,
    JobFile,
    JobEvent,
    JobChunk,
    OverwatchSummary,
    GlobalPref,
    OverwatchPattern,
)

from app.astra_memory.service import (
    # AstraJob lifecycle
    create_job,
    update_job_status,
    link_spec_to_job,
    link_arch_to_job,
    # File tracking
    record_file_touch,
    get_files_for_job,
    get_jobs_for_file,
    # Event projection
    project_event_to_db,
    get_events_for_job,
    # Chunk tracking
    create_chunk,
    update_chunk_status,
    # Overwatcher
    get_or_create_overwatch_summary,
    record_overwatch_intervention,
    record_overwatch_pattern,
    get_patterns_for_file,
    # Global prefs
    set_global_pref,
    get_global_pref,
    get_prefs_for_component,
    # Queries
    get_job,
    get_jobs_by_status,
    get_escalated_jobs,
)

__all__ = [
    # Models
    "AstraJob",
    "JobFile",
    "JobEvent",
    "JobChunk",
    "OverwatchSummary",
    "GlobalPref",
    "OverwatchPattern",
    # Service functions
    "create_job",
    "update_job_status",
    "link_spec_to_job",
    "link_arch_to_job",
    "record_file_touch",
    "get_files_for_job",
    "get_jobs_for_file",
    "project_event_to_db",
    "get_events_for_job",
    "create_chunk",
    "update_chunk_status",
    "get_or_create_overwatch_summary",
    "record_overwatch_intervention",
    "record_overwatch_pattern",
    "get_patterns_for_file",
    "set_global_pref",
    "get_global_pref",
    "get_prefs_for_component",
    "get_job",
    "get_jobs_by_status",
    "get_escalated_jobs",
]
