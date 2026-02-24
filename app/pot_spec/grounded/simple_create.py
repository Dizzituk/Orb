# FILE: app/pot_spec/grounded/simple_create.py
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
import re
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

CREATE_ANALYSIS_SYSTEM_PROMPT = """You are an expert software architect analyzing a feature request.
""" + _GOV_PROMPT + """

You will receive:
1. A feature description (from the Weaver stage)
2. Detected tech stack
3. Discovered integration points (existing files)
4. Extracted constraints

YOUR TASK:
Produce a structured analysis with these sections:

## Architecture Overview
Brief description of the feature's architecture (3-5 sentences).

## Implementation Steps
Numbered, actionable implementation steps. Each step should reference specific
files (existing or new) and describe what changes are needed. Order by dependency.

## Files to Modify
For each existing file that needs changes, explain WHAT changes are needed and WHY.

## New Files to Create
For each new file, explain its purpose and key contents.

## Acceptance Criteria
Task-specific, testable acceptance criteria. Include criteria derived from
explicit constraints (e.g., "no network traffic during transcription").

IMPORTANT:
- Be specific to THIS feature, not generic
- Respect ALL constraints listed
- Reference actual integration points provided
- Keep implementation steps concrete and actionable
- Do NOT suggest cloud services if constraints say local-only
- Do NOT suggest files/features outside the stated phase scope

AMBIGUITY HANDLING:
When a design decision has multiple valid approaches, do NOT flag it as
HUMAN_REQUIRED if a safe default exists. Instead:
- Pick the most flexible default
- State it as a DECISION_ALLOWED with the chosen default
- Only flag HUMAN_REQUIRED for genuine ambiguity with no safe default
- Implementation details are NOT architectural decisions — adopt sensible defaults

ENTRYPOINT IDENTIFICATION (CRITICAL):
When generating EVIDENCE_REQUESTs to locate backend entrypoints (e.g., main.py):
- Find the file that instantiates FastAPI() and registers routers via include_router().
- Ignore main.py under /static/, /dist/, /build/, /public/.
- success_criteria MUST include the FastAPI instantiation and include_router lines.
- If multiple main.py files exist, distinguish by content, not path.

CONFIGURATION FILE EVIDENCE:
When the task references external configuration files loaded by service wrappers:
- Include an EVIDENCE_REQUEST to read the configuration file directly.
- Confirm sections, keys, and values match consuming code expectations.

FUNCTION SIGNATURE VERIFICATION (CRITICAL):
When proposing to CALL an existing function:
- MUST emit an EVIDENCE_REQUEST to read the function definition first.
- Verify exact parameter names, required vs optional, return type.
- Do NOT assume parameter names — verify from evidence.

CALLER CHAIN VERIFICATION (CRITICAL):
When proposing to inject code at a specific point:
- Verify WHO calls that function and WHAT data they pass in.
- If your plan depends on a field being available, verify via EVIDENCE_REQUEST.

ER ID UNIQUENESS:
- Every EVIDENCE_REQUEST must have a unique id (ER-001, ER-002, etc.).
- NEVER emit two EVIDENCE_REQUEST blocks with the same id.

EVIDENCE_REQUEST FORMAT (strict YAML):
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
      args: {file_path: "full/path/to/file.py"}
      expect: "What you expect to find"
  success_criteria: "What counts as having the answer"
  fallback_if_not_found: "DECISION_ALLOWED" | "HUMAN_REQUIRED"

CRITICAL FORMATTING RULES:
- Block MUST start with 'EVIDENCE_REQUEST:' on its own line
- Fields MUST be indented with 2 spaces
- id field MUST be quoted: id: "ER-001"
- Do NOT wrap in markdown code blocks or headers

TOOL USAGE:
- Use 'sandbox_inspector.read_sandbox_file' with {file_path: "FULL_PATH"} for reads
- Use 'sandbox_inspector.run_sandbox_discovery_chain' with {anchor: "FULL_PATH"} for listings
- ALWAYS use FULL ABSOLUTE PATHS from the Integration Points list
- Do NOT guess paths — only request files from Integration Points or prior discovery

When you need to examine files, emit EVIDENCE_REQUEST blocks. The orchestrator
will read the files and re-prompt you with actual contents."""
