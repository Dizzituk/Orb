# FILE: app/llm/local_tools/sequential_thinking.py
# Purpose: The `sequentialthinking` tool — a revisable reasoning scratchpad for
#          local agent workers (Nat). Extends effective working memory past the
#          model's small context and forces committed, revisable steps.
# Called-by: app.llm.nat_tool_loop
# Depends-on: (none)
# Last-renovated: 2026-06-19
"""
Sequential-thinking capability for the local worker.

The model writes one numbered thought at a time (thought, thoughtNumber,
isRevision, nextThoughtNeeded, ...). Each call is recorded and acknowledged;
the loop keeps inviting thoughts until the model sets nextThoughtNeeded=false.

This is OFF for the memory jobs (they're simple) but MUST be present and
callable so agentic callers (the 26B) can invoke deliberation. The tool is
defined in the internal `parameters` shape; the vLLM provider converts it to
OpenAI function format on the way out.
"""

from __future__ import annotations

from typing import Any, Dict, List


SEQUENTIAL_THINKING_TOOL: Dict[str, Any] = {
    "name": "sequentialthinking",
    "description": (
        "A scratchpad for step-by-step, revisable reasoning. Write ONE numbered "
        "thought per call. You may revise an earlier thought (set isRevision=true "
        "and revisesThought). Set nextThoughtNeeded=false on your final thought, "
        "then give your answer in plain text. Use this when a problem benefits "
        "from deliberation; skip it for simple lookups."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "thought": {"type": "string", "description": "The current reasoning step."},
            "thoughtNumber": {"type": "integer", "description": "1-based index of this thought."},
            "totalThoughts": {"type": "integer", "description": "Current best estimate of total thoughts."},
            "nextThoughtNeeded": {"type": "boolean", "description": "True if another thought should follow."},
            "isRevision": {"type": "boolean", "description": "True if this revises a previous thought."},
            "revisesThought": {"type": "integer", "description": "Which thoughtNumber this revises."},
        },
        "required": ["thought", "thoughtNumber", "nextThoughtNeeded"],
    },
}


class SequentialThinkingState:
    """Per-job scratchpad. One instance per run_nat_tool_loop call."""

    def __init__(self) -> None:
        self.thoughts: List[Dict[str, Any]] = []

    def record(self, params: Dict[str, Any]) -> str:
        """Record a thought and return a compact JSON acknowledgement string."""
        thought = str(params.get("thought", "")).strip()
        number = params.get("thoughtNumber")
        next_needed = bool(params.get("nextThoughtNeeded", False))
        entry = {
            "thoughtNumber": number,
            "isRevision": bool(params.get("isRevision", False)),
            "revisesThought": params.get("revisesThought"),
            "thought": thought,
        }
        self.thoughts.append(entry)
        # Minimal ack — enough for the model to continue, cheap on tokens.
        return (
            '{"recorded": true, "thoughtNumber": %s, "totalRecorded": %d, '
            '"nextThoughtNeeded": %s}'
            % (number if number is not None else "null", len(self.thoughts),
               "true" if next_needed else "false")
        )

    def next_needed(self, params: Dict[str, Any]) -> bool:
        return bool(params.get("nextThoughtNeeded", False))
