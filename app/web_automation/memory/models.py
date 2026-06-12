# FILE: app/web_automation/memory/models.py
# Purpose: Pydantic data shapes for the flow memory system.
# Called-by: app.web_automation.memory, app.web_automation.memory.checks, app.web_automation.memory.diagnostics, app.web_automation.memory.runner (+1 more)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Pydantic data shapes for the flow memory system.

No logic in this file — these types are pure data carriers. Logic lives
in checks.py (evaluation), runner.py (execution), store.py (persistence),
diagnostics.py (failure context).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# ACTIONS — what a step DOES
# =============================================================================

class Action(BaseModel):
    """
    A single tool invocation. The runner dispatches by `kind` to the matching
    tool in action_executor.TOOL_HANDLERS, passing `params` through directly.

    Examples:
        Action(kind='web_click',
               params={'session': 'meta_business', 'x': 432, 'y': 160})
        Action(kind='web_navigate',
               params={'session': 'meta_business',
                       'url': 'https://business.facebook.com/...'} )
        Action(kind='system_keys',
               params={'text': 'C:/path/to/image.png',
                       'press_enter_after': True})
        Action(kind='wait',
               params={'ms': 2000})  # internal — handled by runner
    """
    kind: str
    params: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# CHECKS — pre / post conditions
# =============================================================================

class Check(BaseModel):
    """
    A precondition or postcondition. Polled until it passes or the timeout
    elapses, so transient UI lag doesn't fail the step.

    Supported kinds (see checks.py):
        'dom_includes' / 'dom_excludes'
            expected = list of substrings; matches against web_dom_snapshot
        'url_includes' / 'url_excludes'
            expected = list of substrings; matches against web_current_state url
        'text_includes' / 'text_absent'
            expected = list of substrings; matches against web_extract_text
        'always_pass'
            no-op; useful to skip a precondition
    """
    kind: str
    expected: List[str] = Field(default_factory=list)
    timeout_ms: int = 5000
    poll_interval_ms: int = 500


class CheckResult(BaseModel):
    ok: bool
    kind: str
    expected: List[str] = Field(default_factory=list)
    observed_summary: str = ""
    elapsed_ms: int = 0
    timed_out: bool = False


# =============================================================================
# STEPS — one stage of a flow
# =============================================================================

class Step(BaseModel):
    """
    One stage of a multi-step flow.

    Lifecycle inside the runner:
        1. precondition (if any)  — must pass within timeout
        2. action                  — dispatched to the tool layer
        3. postcondition (if any) — must pass within timeout

    If precondition fails => the PRIOR step's postcondition was a lie
    (it claimed success but the page is not in the expected state).
    Diagnostic guidance steers the agent toward fixing the prior step.

    If action fails => the tool itself raised. Likely transient or a
    parameter problem on THIS step. Don't redo prior steps.

    If postcondition fails => action ran but expected outcome did not
    appear. Either action targeted wrong selector / coordinate, or the
    platform UI has changed and the expected signature is stale. Fix
    this step only.
    """
    step_id: str
    description: str = ""
    session: Optional[str] = None
    precondition: Optional[Check] = None
    action: Action
    postcondition: Optional[Check] = None
    expected_duration_ms: int = 0   # informational; not enforced


class StepResult(BaseModel):
    step_id: str
    description: str = ""
    ok: bool
    phase: str   # "precondition" | "action" | "postcondition" | "complete"
    duration_ms: int = 0
    action_result: Optional[str] = None
    precondition_result: Optional[CheckResult] = None
    postcondition_result: Optional[CheckResult] = None
    error: Optional[str] = None


# =============================================================================
# FLOWS — full task definitions
# =============================================================================

class Flow(BaseModel):
    """
    A named, cached sequence of steps that together complete a platform
    task — e.g. ('meta_business', 'reply_to_top_comment'),
    ('wordpress', 'publish_draft'), ('coursera', 'mark_lesson_complete').

    Versioned so a Meta UI redesign can be tracked: each successful save
    increments version, allowing rollback if a new version regresses.
    """
    platform: str
    task: str
    version: int = 1
    description: str = ""
    steps: List[Step] = Field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    last_run_at: Optional[str] = None
    last_failure_reason: Optional[str] = None


class FlowResult(BaseModel):
    """
    Outcome of running a flow.

    Failure-isolation contract:
        * completed_steps = every step whose postcondition passed.
          These are KNOWN GOOD. The next attempt does NOT need to
          re-debug them.
        * failed_step    = the single step where the runner halted.
          The agent's repair attention belongs HERE only.
        * remaining_step_ids = steps not yet attempted.
        * diagnostic_summary = human-readable failure context including
          what was expected, what was observed, and which phase failed.
    """
    ok: bool
    platform: str
    task: str
    completed_steps: List[StepResult] = Field(default_factory=list)
    failed_step: Optional[StepResult] = None
    remaining_step_ids: List[str] = Field(default_factory=list)
    total_duration_ms: int = 0
    diagnostic_summary: str = ""
