# FILE: app/web_automation/memory/__init__.py
"""
Per-platform flow memory for ASTRA.

Stores, executes, and verifies multi-step interaction flows on platforms
where the system has previously succeeded. Each flow is a sequence of
Steps; each Step has a precondition, an action, and a postcondition.
The runner halts on the first verification failure, so the agent always
knows:

  * which steps confirmed working (their postconditions passed)
  * which step is the actual problem (failed precondition / action /
    postcondition with structured 'observed vs. expected' detail)
  * which steps haven't been attempted yet

This is the failure-isolation property: when something breaks at step 8,
steps 1-7 are *known good*, the agent can focus repair on step 8 only,
and there is no "everything is broken" cliff.

Public API:
  - models.Flow / Step / Action / Check / FlowResult / StepResult
  - store.load_flow / save_flow / list_flows / update_flow_stats
  - runner.run_flow
  - diagnostics.build_diagnostic_summary

Chat tools wrapping this lives in:
  app/debug/executors/flow_actions.py
"""
from app.web_automation.memory.models import (
    Action,
    Check,
    CheckResult,
    Flow,
    FlowResult,
    Step,
    StepResult,
)
from app.web_automation.memory.store import (
    list_flows,
    load_flow,
    save_flow,
    update_flow_stats,
)
from app.web_automation.memory.runner import run_flow
from app.web_automation.memory.diagnostics import build_diagnostic_summary

__all__ = [
    "Action", "Check", "CheckResult",
    "Flow", "FlowResult", "Step", "StepResult",
    "list_flows", "load_flow", "save_flow", "update_flow_stats",
    "run_flow",
    "build_diagnostic_summary",
]
