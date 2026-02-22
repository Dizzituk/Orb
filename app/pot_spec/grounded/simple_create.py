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

v1.0 (2026-02-02): Initial implementation
v1.2 (2026-02-02): Goal validation
v2.0 (2026-02-04): MAJOR FIX — LLM-grounded analysis
    - Added LLM analysis step using the allocated model (was zero-LLM before)
    - Fixed keyword extraction: added stopwords, min length, no garbage matches
    - Fixed integration point discovery: semantic relevance, not substring matching
    - Fixed file suggestions: constraint-aware, removed hardcoded openai_client.py
    - Fixed Implementation Steps: LLM-generated from evidence, not copy-paste
    - Fixed acceptance criteria: task-specific, extracted from constraints
    - Added constraint extraction from weaver output (negation detection)
    - spec_runner now passes provider_id/model_id through to this module
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from app.pot_spec.grounded._simple_create_utils import SIMPLE_CREATE_BUILD_ID, _CONTENT_SIGNALS, _CREATE_ANALYSIS_MODEL, _EVIDENCE_MAX_LOOPS, _FALLBACK_MODELS, _NEGATIVE_PATH_SEGMENTS, _extract_acceptance_from_constraints, _find_file_in_projects
from app.pot_spec.grounded._simple_create_utils import ARCHITECTURAL_FILE_PATTERNS, CONCEPT_KEYWORDS, KEYWORD_STOPWORDS, MIN_KEYWORD_LENGTH, NEGATION_PATTERNS, PLACEHOLDER_GOALS, _resolve_mentioned_files, _score_integration_point
from app.pot_spec.grounded._simple_create_utils import CONCEPT_DIRECTORY_PATTERNS, _CREATE_ANALYSIS_TIMEOUT, _EVIDENCE_MAX_FILE_CHARS, _extract_constraints, _extract_task_keywords, _host_list_directory, _sanitize_goal, _suggest_new_files
from app.pot_spec.grounded._simple_create_utils import _extract_patterns, _find_integration_points, _host_read_file, _read_text_any_encoding, build_create_spec
from app.pot_spec.grounded._simple_create_utils import CreateEvidence, IntegrationPoint, _detect_tech_stack
from app.pot_spec.grounded._simple_create_utils import TechStack, _run_llm_analysis, build_grounded_create_spec
from app.pot_spec.grounded._simple_create_utils import _fulfil_evidence_requests

logger = logging.getLogger(__name__)
print(f"[SIMPLE_CREATE_LOADED] BUILD_ID={SIMPLE_CREATE_BUILD_ID}")

# v4.0: Max evidence fulfilment loops (matches ASTRA_EVIDENCE_MAX_LOOPS convention)
# v4.0: Max chars to read per file during evidence fulfilment

# =============================================================================
# ENV-DRIVEN MODEL OVERRIDE FOR CREATE ANALYSIS
# =============================================================================
# The spec_gate stage allocates a model (often gpt-5.2-pro) for the WHOLE stage,
# but CREATE analysis is structured analysis — doesn't need a pro-tier model.
# A faster model gives better latency without sacrificing quality here.
#
# Set ASTRA_CREATE_ANALYSIS_MODEL to override (e.g., "gpt-5-mini", "gpt-5.2")
# If not set, uses the model allocated by spec_gate_stream.


# =============================================================================
# DATA STRUCTURES
# =============================================================================


# =============================================================================
# STOPWORDS & KEYWORD EXTRACTION (v2.0 — FIXED)
# =============================================================================

# Words that should NEVER be used as search keywords for filename matching

# Minimum keyword length for filename matching (prevents "no", "ui", etc.)

# Map task concepts to search keywords
# v2.0: Only concepts are returned, individual keywords are used for DETECTION only


# =============================================================================
# CONSTRAINT EXTRACTION (v2.0 — NEW)
# =============================================================================

# Patterns that indicate negative constraints


# =============================================================================
# v5.1: PRE-RESOLVE MENTIONED FILENAMES
# =============================================================================
# When the user mentions specific filenames (e.g. `weaver_stream.py`,
# `architecture_executor.py`), resolve them to real paths BEFORE the LLM
# is called. This prevents the LLM from guessing wrong paths in
# EVIDENCE_REQUESTs.


# =============================================================================
# TECH STACK DETECTION
# =============================================================================


# =============================================================================
# INTEGRATION POINT DISCOVERY (v2.0 — FIXED)
# =============================================================================

# v2.0: Only match these SPECIFIC architectural files, not keyword substrings

# v2.0: Concept-to-directory patterns — search for files in relevant directories


# =============================================================================
# v3.6: CONTENT-SIGNAL SCORING for integration point disambiguation
# =============================================================================
# Problem: filename-only matching surfaces false positives like Orb/static/main.py
# alongside the real FastAPI entrypoint D:/Orb/main.py. Both match the
# architectural pattern r'^main\.py$' but only one is architecturally relevant.
#
# Solution: For ambiguous filenames (main.py, index.py, app.py), read a small
# content sample and score based on signals. Negative scores for paths under
# static/dist/build/public directories.


# =============================================================================
# PATTERN EXTRACTION
# =============================================================================


# =============================================================================
# FILE SUGGESTION (v2.0 — FIXED: constraint-aware)
# =============================================================================


# =============================================================================
# HOST-DIRECT FILE READER (v4.0 — SpecGate Evidence Fulfilment)
# =============================================================================
# SpecGate runs BEFORE the sandbox is available, so evidence fulfilment must
# use host-direct filesystem access. This is intentionally local to simple_create
# to make the "SpecGate uses host-direct access" boundary explicit.


# =============================================================================
# EVIDENCE FULFILMENT LOOP (v4.0 — SpecGate Evidence Fulfilment)
# =============================================================================
# Uses parsing/stripping utilities from evidence_loop.py but dispatches file
# reads via host-direct access (not sandbox). This keeps SpecGate independent
# of the sandbox lifecycle while reusing the robust 3-layer YAML parsing.


# =============================================================================
# LLM ANALYSIS (v2.0 — NEW)
# =============================================================================

CREATE_ANALYSIS_SYSTEM_PROMPT = """You are an expert software architect analyzing a feature request.

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
When a design decision has multiple valid approaches (e.g., "should the endpoint
accept multipart/form-data or raw bytes?"), do NOT flag it as HUMAN_REQUIRED if
a safe default exists that covers both options. Instead:
- Pick the most flexible default (e.g., "support both multipart and raw body")
- State it as a DECISION_ALLOWED with the chosen default
- Only flag HUMAN_REQUIRED when there is genuine ambiguity with no safe default,
  or when the choice has significant architectural consequences that cannot be
  reversed without major rework
- Implementation details like exact response field names, optional metadata fields,
  or input format variants are NOT architectural decisions — adopt sensible defaults

ENTRYPOINT IDENTIFICATION (CRITICAL):
When generating EVIDENCE_REQUESTs to locate backend entrypoints (e.g., main.py):
- The goal is to find the file that instantiates FastAPI() (or calls an app factory
  returning FastAPI) and registers routers via include_router().
- Ignore any main.py files under /static/, /dist/, /build/, /public/, or frontend
  project roots. These are NOT the backend entrypoint.
- success_criteria MUST include: "Evidence must include the lines showing
  app = FastAPI(...) (or equivalent factory) and at least one include_router(...)."
- If multiple main.py files exist, the ER must distinguish them by checking content,
  not just path.

CONFIGURATION FILE EVIDENCE:
When the task references external configuration files (e.g., config.ini, .env, YAML)
that are loaded by service wrappers or modules being integrated:
- Include an EVIDENCE_REQUEST to read the configuration file directly.
- Confirm that the sections, keys, and values match what the consuming code expects.
- This prevents runtime mismatches between config parsers and actual config content.
- Fold this into an existing ER if it already reads the consuming module's code,
  or create a dedicated ER if the config file is on a separate path.

FUNCTION SIGNATURE VERIFICATION (CRITICAL):
When your implementation plan proposes CALLING an existing function (e.g., store_embedding(),
search_embeddings(), generate_embedding(), or any other existing function):
- You MUST emit an EVIDENCE_REQUEST to read that function's definition BEFORE proposing
  parameter names or return types in your spec.
- Verify: exact parameter names, required vs optional params, return type/structure.
- Do NOT assume parameter names like 'top_k', 'source_type', 'chunk_index' exist —
  the actual function may use different names (e.g., 'limit', 'content_type', 'idx').
- If your plan says "call X(a=1, b=2)", you must have CITED evidence showing X accepts
  parameters 'a' and 'b'. Otherwise emit an ER to read the function.
- This applies equally to ORM model fields — if you propose filtering by a column,
  verify that column exists on the model.

CALLER CHAIN VERIFICATION (CRITICAL):
When your implementation plan proposes injecting code at a specific point (e.g., adding
a call inside an existing function), you MUST verify:
- WHO calls that function and WHAT data they pass in.
- If your plan depends on a field being available (e.g., project_id in task metadata),
  emit an EVIDENCE_REQUEST to read at least one caller to confirm the field is populated.
- Do NOT propose regex-parsing free text fields to extract structured data that should
  be passed explicitly. If a field isn't available, the spec should say "add this field"
  rather than "parse it out of an unrelated string".

ER ID UNIQUENESS:
- Every EVIDENCE_REQUEST must have a unique id (ER-001, ER-002, etc.).
- NEVER emit two EVIDENCE_REQUEST blocks with the same id.
- If you need to request evidence for two different things, use different ids.

EVIDENCE_REQUEST FORMAT (you MUST use this exact YAML format — the parser is strict):
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
- The block MUST start with 'EVIDENCE_REQUEST:' on its own line (no prefix, no markdown header)
- Fields MUST be indented with 2 spaces under EVIDENCE_REQUEST:
- The id field MUST be quoted: id: "ER-001" (not id: ER-001)
- Do NOT wrap EVIDENCE_REQUESTs in markdown code blocks or headers
- Do NOT use prose-style descriptions — use the YAML structure above

TOOL USAGE:
- Use tool 'sandbox_inspector.read_sandbox_file' with args: {file_path: "FULL_PATH"} for reading files
- Use tool 'sandbox_inspector.run_sandbox_discovery_chain' with args: {anchor: "FULL_PATH"} for listing directories
- ALWAYS use FULL ABSOLUTE PATHS from the Integration Points list provided to you
- Do NOT guess paths. Only request files that appear in the Integration Points list,
  or that you discovered from a previous directory listing or file read.
- If you need to find files not in the Integration Points list, first request a
  directory listing of the parent directory, then request specific files from the results.

When you need to examine files to ground your analysis, emit EVIDENCE_REQUEST blocks
in this exact format. The orchestrator will read the files and re-prompt you with
the actual contents so you can produce a grounded analysis instead of guessing."""


# Default fallback model when the allocated model times out


# =============================================================================
# SPEC BUILDER (v2.0 — FIXED)
# =============================================================================

# v1.2: Placeholder goals that should never be used


# =============================================================================
# MAIN ENTRY POINT (v2.0 — UPDATED)
# =============================================================================
