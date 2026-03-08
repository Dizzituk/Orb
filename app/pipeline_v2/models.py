# FILE: app/pipeline_v2/models.py
"""
ASTRA v2.1 Pipeline Data Models.

Simplified from v2.0. No tiers, no build layers.
Scaffold produces file list, Builder works through them agentically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Scaffold Engine output
# ---------------------------------------------------------------------------

@dataclass
class ScaffoldFile:
    """A file produced by the Scaffold Engine."""
    path: str
    content: str
    is_new: bool = True           # True = CREATE, False = MODIFY stub
    char_count: int = 0


@dataclass
class ScaffoldResult:
    """Complete output of the Scaffold Engine."""
    files: List[ScaffoldFile] = field(default_factory=list)
    directories_created: List[str] = field(default_factory=list)
    total_files: int = 0
    duration_seconds: float = 0.0

    @property
    def file_paths(self) -> List[str]:
        return [f.path for f in self.files]


# ---------------------------------------------------------------------------
# Agentic Builder tracking
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """Record of a single tool call made by the Builder."""
    tool: str               # read_file, write_file, run_shell, screenshot
    args: str = ""          # Brief description of args
    result_summary: str = ""
    success: bool = True
    tokens_used: int = 0


@dataclass
class BuildSession:
    """State of one Agentic Builder session (context window)."""
    session_number: int = 1
    tool_calls: List[ToolCall] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    errors_fixed: List[str] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    handover_summary: str = ""
    completed: bool = False

    @property
    def total_tool_calls(self) -> int:
        return len(self.tool_calls)


@dataclass
class BuildResult:
    """Complete output of the Agentic Builder."""
    success: bool = False
    sessions: List[BuildSession] = field(default_factory=list)
    all_files_written: List[str] = field(default_factory=list)
    total_tool_calls: int = 0
    total_llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_duration_seconds: float = 0.0
    used_fallback: bool = False
    errors: List[str] = field(default_factory=list)
    # v2.1.2: Conversation history for verify→fix continuation
    messages_history: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Verification output
# ---------------------------------------------------------------------------

@dataclass
class VerifyResult:
    """Visual verification result."""
    passed: bool = False
    feedback: str = ""          # Plain language: what's wrong
    screenshot_path: str = ""   # Path to the captured screenshot
    attempt: int = 1


# ---------------------------------------------------------------------------
# Full pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Complete v2.1 pipeline result."""
    success: bool = False
    job_id: str = ""
    scaffold_result: Optional[ScaffoldResult] = None
    build_result: Optional[BuildResult] = None
    verify_results: List[VerifyResult] = field(default_factory=list)
    total_llm_calls: int = 0
    total_duration_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    errors: List[str] = field(default_factory=list)
