# FILE: app/pot_spec/grounded/simple_create.py
# Purpose: SpecGate CREATE Path - LLM-Grounded Feature Spec Builder
# Called-by: app.pot_spec.grounded._simple_create_review, app.pot_spec.grounded._simple_create_utils_17, app.pot_spec.grounded.spec_runner
# Depends-on: app.pot_spec.governance_rules, app.pot_spec.grounded._simple_create_evidence, app.pot_spec.grounded._simple_create_utils_12, app.pot_spec.grounded._simple_create_utils_13 (+4 more)
# Last-renovated: 2026-06-11
"""
SpecGate CREATE Path - LLM-Grounded Feature Spec Builder

Provides grounded specs for CREATE tasks (new features) by combining
filesystem evidence with LLM analysis using the allocated model.

Flow:
1. Extract meaningful keywords from task description (stopword-filtered)
2. Scan codebase for relevant integration points (semantic, not substring)
3. Detect tech stack and patterns
4. Extract constraints from task description (e.g., "no cloud APIs")
5. Use allocated LLM to analyze evidence and generate intelligent spec
6. Build grounded spec with WHERE + HOW + INTEGRATION + CONSTRAINTS

All helpers, constants, and data structures are in _simple_create_utils_12..17.
Evidence fulfilment loop is in _simple_create_evidence.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.pot_spec.grounded._simple_create_utils_12 import (
    SIMPLE_CREATE_BUILD_ID, _CONTENT_SIGNALS, _CREATE_ANALYSIS_MODEL,
    _EVIDENCE_MAX_LOOPS, _FALLBACK_MODELS, _NEGATIVE_PATH_SEGMENTS,
    _extract_acceptance_from_constraints, _find_file_in_projects,
)
from app.pot_spec.grounded._simple_create_utils_13 import (
    ARCHITECTURAL_FILE_PATTERNS, CONCEPT_KEYWORDS, KEYWORD_STOPWORDS,
    MIN_KEYWORD_LENGTH, NEGATION_PATTERNS, PLACEHOLDER_GOALS,
    _resolve_mentioned_files, _score_integration_point,
)
from app.pot_spec.grounded._simple_create_utils_14 import (
    CONCEPT_DIRECTORY_PATTERNS, _CREATE_ANALYSIS_TIMEOUT,
    _EVIDENCE_MAX_FILE_CHARS, _extract_constraints, _extract_task_keywords,
    _host_list_directory, _sanitize_goal, _suggest_new_files,
)
from app.pot_spec.grounded._simple_create_utils_15 import (
    _extract_patterns, _find_integration_points, _host_read_file,
    _read_text_any_encoding, build_create_spec,
)
from app.pot_spec.grounded._simple_create_utils_16 import (
    CreateEvidence, IntegrationPoint, _detect_tech_stack,
)
from app.pot_spec.grounded._simple_create_utils_17 import (
    TechStack, _run_llm_analysis, build_grounded_create_spec,
)

logger = logging.getLogger(__name__)
print(f"[SIMPLE_CREATE_LOADED] BUILD_ID={SIMPLE_CREATE_BUILD_ID}")


async def _fulfil_evidence_requests(
    llm_analysis: str,
    provider_id: str,
    model_id: str,
    llm_call_func: Callable,
    project_paths: List[str],
    goal: str = "",
    what_to_do: str = "",
) -> str:
    """Fulfil EVIDENCE_REQUEST blocks — delegates to _simple_create_evidence.py."""
    from app.pot_spec.grounded._simple_create_evidence import fulfil_evidence_requests
    return await fulfil_evidence_requests(
        llm_analysis=llm_analysis,
        provider_id=provider_id,
        model_id=model_id,
        llm_call_func=llm_call_func,
        project_paths=project_paths,
        goal=goal,
        what_to_do=what_to_do,
        system_prompt=CREATE_ANALYSIS_SYSTEM_PROMPT,
    )


# v2.1: Import governance rules for prompt injection
try:
    from app.pot_spec.governance_rules import SPEC_GATE_GOVERNANCE_PROMPT as _GOV_PROMPT
except ImportError:
    _GOV_PROMPT = ""

CREATE_ANALYSIS_SYSTEM_PROMPT = """You are an expert software architect producing a tight, actionable spec.
""" + _GOV_PROMPT + """

You receive: a feature description, detected tech stack, discovered integration
points (existing files), and extracted constraints.

PRODUCE THESE SECTIONS ONLY:

## Architecture Overview
2-4 sentences. What this feature adds, how it fits the existing system.
No generic preamble.

## Files to Modify
For each EXISTING file that needs changes:
- Full path
- WHAT changes (1-2 lines)
- WHY (what breaks without it)
Only list files that ACTUALLY need changing for THIS feature.
Do NOT list files just because they exist in the codebase.

## New Files to Create
For each new file:
- Full path
- Purpose (1 line)
- Key contents: list fields/exports/endpoints concretely
Mark each content item: [TEMPLATE] (deterministic scaffold) or [LLM_FILL] (creative).

## Acceptance Criteria
Testable, task-specific. Each criterion must be verifiable by running the app.
Derive from constraints and requirements. No generic filler like "no console errors".

RULES:
- Be SPECIFIC to THIS feature. No generic advice.
- Respect ALL constraints. No cloud if constraints say local-only.
- Reference ONLY files from the integration points list or your evidence.
- Do NOT suggest files outside the stated phase scope.
- Do NOT include raw code snippets — the implementation pipeline reads files itself.
- Do NOT repeat the feature description back. Get straight to architecture.
- Keep the total output under 3000 chars. Concise specs build better code.

FULL-STACK RULE:
- When the feature description mentions UI, visual design, frontend components,
  screenshots, colour schemes, layouts, cards, dashboards, or any user-facing
  elements, you MUST produce BOTH backend AND frontend files.
- A feature with UI requirements is INCOMPLETE without .tsx/.ts component files.
- Check the Tech Stack section: if a frontend framework is detected (React,
  Vue, etc.), frontend files are expected for any feature with a visual element.
- Frontend files follow the project's component structure (e.g. src/components/
  for React projects). Check the integration points for existing patterns.
- If the feature description includes style references (hex colours, dark mode,
  card layouts, etc.), include these as requirements in the Architecture Overview
  so the implementation pipeline preserves them.

PATTERN-INHERITANCE AND DRAFTING RULE (CRITICAL):
This is almost always a modification of an EXISTING project, not greenfield work.
Your draft architecture MUST reference the project's existing patterns and use
ONLY the libraries listed in the Declared Dependencies section of the user prompt.

Patterns to identify before proposing new files (use the Existing Patterns and
Declared Dependencies sections in the user prompt as your starting point, and
emit EVIDENCE_REQUEST blocks for any uncertain assumptions):
- Persistence primitive (SharedPreferences, DataStore, Room, JSON files, etc.) —
  new persistent state should use the same primitive.
- HTTP/networking client (Retrofit, OkHttp, Ktor, requests, httpx, etc.) —
  new networking should follow the same approach.
- State management pattern (StateFlow, LiveData, RxJava, callbacks) —
  new state should match.
- Dependency injection, if any (Hilt, Koin, manual) — new wiring should match.

If a library is NOT in the Declared Dependencies list, you MUST NOT propose code
that imports it. Build the feature using ONLY the primitives the project already
has. If the existing primitives genuinely cannot solve the problem, raise this as
[HUMAN_REQUIRED] rather than silently introducing a new dependency.

Proportionality: prefer the simplest primitive that solves the problem. A queue
holding a few items in a dead zone needs a JSON file, not Room. A counter needs
SharedPreferences, not a database. Match the scale of the solution to the scale
of the problem.

DRAFTING DISCIPLINE (CRITICAL):
You MUST produce a complete draft spec on EVERY round, including round 1. The
draft MUST contain Architecture Overview, Files to Modify, New Files to Create,
and Acceptance Criteria sections — even if some details are still uncertain.

EVIDENCE_REQUEST blocks are for VERIFYING uncertain assumptions in your draft,
not for AVOIDING the draft. It is acceptable to emit ER blocks alongside your
draft to confirm specific facts, but it is NEVER acceptable to emit ONLY ERs
with no draft. A response containing only EVIDENCE_REQUEST blocks and no spec
sections will be treated as a failed round.

If you are uncertain about an existing file's contents, mark your draft proposal
with [VERIFY: <what you assumed>] and emit a targeted EVIDENCE_REQUEST. The
orchestrator will return the file contents and you can refine the draft on the
next round. The loop refines drafts; it does not gate them.

This rule overrides any default architectural pattern from training data. The
project's existing style is the source of truth, not generic best practice.

AMBIGUITY HANDLING:
- If a safe default exists: pick it, mark [DECISION_ALLOWED], move on.
- Only mark [HUMAN_REQUIRED] for genuine ambiguity with no safe default.
- Implementation details are NOT architectural decisions.

EVIDENCE_REQUEST FORMAT:
When you need to examine a file before making architecture decisions, emit:

EVIDENCE_REQUEST:
  id: "ER-NNN"
  severity: "CRITICAL" | "NONCRITICAL"
  need: "What you need to know"
  why: "What breaks if you guess wrong"
  scope:
    roots: ["where to look"]
    max_files: 500
  tool_calls:
    - tool: "sandbox_inspector.read_sandbox_file"
      args: {file_path: "FULL_PATH"}
      expect: "What you expect to find"
  success_criteria: "What counts as having the answer"
  fallback_if_not_found: "DECISION_ALLOWED" | "HUMAN_REQUIRED"

ER RULES:
- Each id must be unique (ER-001, ER-002, etc.)
- ALWAYS use FULL ABSOLUTE PATHS from the integration points list
- Use 'sandbox_inspector.read_sandbox_file' for file reads
- Use 'sandbox_inspector.run_sandbox_discovery_chain' for directory listings
- Block MUST start with 'EVIDENCE_REQUEST:' on its own line, 2-space indent
- Do NOT wrap in markdown code blocks

The orchestrator will read the files and re-prompt you with actual contents."""
