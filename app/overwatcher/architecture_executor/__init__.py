"""
Architecture Executor Package

Facade module that re-exports all public symbols from the refactored
sub-modules. Existing imports like:

    from app.overwatcher.architecture_executor import run_architecture_execution

continue to work without modification.
"""

# Core orchestrator (the main entry point)
from .orchestrator import run_architecture_execution

# Architecture document parsing
from .parsing import parse_file_inventory, extract_section_for_file

# Context and interface extraction
from .context import (
    _extract_file_interfaces,
    _extract_existing_imports,
    _extract_router_registrations,
    _build_resolved_endpoints,
    _format_job_context,
)

# Constants and build metadata
from .constants import ARCHITECTURE_EXECUTOR_BUILD_ID

# Path resolution
from .path_resolution import _ensure_python_init_files

__all__ = [
    "run_architecture_execution",
    "parse_file_inventory",
    "extract_section_for_file",
    "_extract_file_interfaces",
    "_extract_existing_imports",
    "_extract_router_registrations",
    "_build_resolved_endpoints",
    "_format_job_context",
    "_ensure_python_init_files",
    "ARCHITECTURE_EXECUTOR_BUILD_ID",
]