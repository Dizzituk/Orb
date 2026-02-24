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
from app.pot_spec.grounded._simple_create_utils_12 import SIMPLE_CREATE_BUILD_ID, _CONTENT_SIGNALS, _CREATE_ANALYSIS_MODEL, _EVIDENCE_MAX_LOOPS, _FALLBACK_MODELS, _NEGATIVE_PATH_SEGMENTS, _extract_acceptance_from_constraints, _find_file_in_projects
from app.pot_spec.grounded._simple_create_utils_13 import ARCHITECTURAL_FILE_PATTERNS, CONCEPT_KEYWORDS, KEYWORD_STOPWORDS, MIN_KEYWORD_LENGTH, NEGATION_PATTERNS, PLACEHOLDER_GOALS, _resolve_mentioned_files, _score_integration_point
from app.pot_spec.grounded._simple_create_utils_14 import CONCEPT_DIRECTORY_PATTERNS, _CREATE_ANALYSIS_TIMEOUT, _EVIDENCE_MAX_FILE_CHARS, _extract_constraints, _extract_task_keywords, _host_list_directory, _sanitize_goal, _suggest_new_files
from app.pot_spec.grounded._simple_create_utils_15 import _extract_patterns, _find_integration_points, _host_read_file, _read_text_any_encoding, build_create_spec
from app.pot_spec.grounded._simple_create_utils_16 import CreateEvidence, IntegrationPoint, _detect_tech_stack
from app.pot_spec.grounded._simple_create_utils_17 import TechStack, _run_llm_analysis, build_grounded_create_spec

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


async def _fulfil_evidence_requests(
    llm_analysis: str,
    provider_id: str,
    model_id: str,
    llm_call_func: Callable,
    project_paths: List[str],
    goal: str = "",
    what_to_do: str = "",
) -> str:
    """Fulfil EVIDENCE_REQUEST blocks in the LLM analysis by reading actual files.

    v4.0: Parses ERs from the analysis, reads requested files from the host
    filesystem, then re-prompts the LLM with real evidence so it can produce
    a grounded spec instead of hallucinating architecture.

    Uses parse_evidence_requests() and strip_fulfilled_requests() from
    evidence_loop.py for robust ER parsing (3-layer YAML defence).

    Max loops: _EVIDENCE_MAX_LOOPS (default 2). After exhaustion, remaining
    ERs are force-resolved with FORCED_RESOLUTION markers.

    Returns updated LLM analysis with ERs replaced by RESOLVED_REQUEST or
    FORCED_RESOLUTION markers and real evidence injected.
    """
    try:
        from app.llm.pipeline.evidence_loop import (
            parse_evidence_requests,
            strip_fulfilled_requests,
            strip_forced_stop_requests,
        )
    except ImportError as exc:
        logger.warning("[SPEC_GATE_EVIDENCE] Cannot import evidence_loop: %s — skipping fulfilment", exc)
        print(f"[SPEC_GATE_EVIDENCE] WARNING: evidence_loop import failed: {exc}")
        return llm_analysis

    current_analysis = llm_analysis

    for loop_idx in range(_EVIDENCE_MAX_LOOPS):
        # Parse outstanding EVIDENCE_REQUESTs
        requests = parse_evidence_requests(current_analysis)
        if not requests:
            logger.info("[SPEC_GATE_EVIDENCE] Loop %d/%d: No EVIDENCE_REQUESTs found — done",
                        loop_idx + 1, _EVIDENCE_MAX_LOOPS)
            print(f"[SPEC_GATE_EVIDENCE] Loop {loop_idx + 1}/{_EVIDENCE_MAX_LOOPS}: No ERs — done")
            break

        logger.info("[SPEC_GATE_EVIDENCE] Loop %d/%d: %d EVIDENCE_REQUEST(s): %s",
                    loop_idx + 1, _EVIDENCE_MAX_LOOPS, len(requests),
                    [r.get('id') for r in requests])
        print(f"[SPEC_GATE_EVIDENCE] Loop {loop_idx + 1}/{_EVIDENCE_MAX_LOOPS}: "
              f"{len(requests)} ER(s): {[r.get('id') for r in requests]}")

        # Dispatch file reads for each ER
        fulfilled_ids = set()
        evidence_bundle_parts = []  # Accumulated evidence text for re-prompt

        for req in requests:
            req_id = req.get("id", "UNKNOWN")
            tool_calls = req.get("tool_calls", [])
            need = req.get("need", "")
            er_results = []

            logger.info("[SPEC_GATE_EVIDENCE] Processing %s: need=%s, tools=%d",
                        req_id, need[:80], len(tool_calls))
            # v4.2: Diagnostic logging for tool dispatch debugging
            for _tc_debug in tool_calls:
                print(f"[SPEC_GATE_EVIDENCE] {req_id}: tool='{_tc_debug.get('tool', 'NONE')}' "
                      f"args={dict(_tc_debug.get('args', {}))}")

            for tc in tool_calls:
                tool_name = tc.get("tool", "")
                args = tc.get("args", {})

                # Dispatch host-direct reads
                # v4.1: Extended tool name matching — LLM may use various tool names
                # from the evidence contract. Map them all to host-direct reads.
                if tool_name in ("sandbox_inspector.read_sandbox_file",
                                 "evidence_collector.add_file_read_to_bundle",
                                 "read_file",
                                 "sandbox_inspector.file_exists_in_sandbox",
                                 "arch_query.get_file_signatures"):
                    file_path = args.get("file_path") or args.get("path", "")
                    if file_path:
                        success, content = _host_read_file(
                            file_path,
                            max_chars=args.get("max_chars", _EVIDENCE_MAX_FILE_CHARS),
                            project_paths=project_paths,
                        )
                        er_results.append({
                            "tool": tool_name,
                            "file_path": file_path,
                            "success": success,
                            # v4.7: Removed hardcoded 4000 char cap. The file is already
                            # truncated by _host_read_file at _EVIDENCE_MAX_FILE_CHARS (50k).
                            # The old [:4000] was starving the LLM of evidence — e.g.
                            # search_embeddings() in service.py starts at char 5237,
                            # call_llm_async() in core.py starts at char ~10000.
                            # The re-prompt's _MAX_EVIDENCE_CHARS (40k) is the real backstop.
                            "content": content if success else None,
                            "error": content if not success else None,
                        })
                        logger.info("[SPEC_GATE_EVIDENCE] %s: read_file %s → %s (%d chars)",
                                    req_id, file_path, "OK" if success else "FAIL",
                                    len(content) if success else 0)

                elif tool_name in ("sandbox_inspector.run_sandbox_discovery_chain",
                                   "list_directory"):
                    dir_path = args.get("anchor") or args.get("path", "")
                    if dir_path:
                        success, listing = _host_list_directory(dir_path, project_paths=project_paths)
                        er_results.append({
                            "tool": tool_name,
                            "path": dir_path,
                            "success": success,
                            "content": listing[:2000] if success else None,
                            "error": listing if not success else None,
                        })
                        logger.info("[SPEC_GATE_EVIDENCE] %s: list_dir %s → %s",
                                    req_id, dir_path, "OK" if success else "FAIL")

                elif tool_name in ("evidence_collector.verify_path_exists",
                                   "evidence_collector.find_in_evidence"):
                    # v4.2: Map verify_path_exists to a simple file existence check
                    file_path = args.get("path") or args.get("file_path", "")
                    if file_path:
                        resolved = file_path.replace('/', os.sep).replace('\\', os.sep)
                        if not os.path.exists(resolved) and project_paths:
                            for root in project_paths:
                                candidate = os.path.join(root, resolved)
                                if os.path.exists(candidate):
                                    resolved = candidate
                                    break
                        exists = os.path.exists(resolved)
                        er_results.append({
                            "tool": tool_name,
                            "file_path": resolved,
                            "success": exists,
                            "content": f"Path {'exists' if exists else 'does NOT exist'}: {resolved}" if exists else None,
                            "error": f"Path does not exist: {resolved}" if not exists else None,
                        })
                        logger.info("[SPEC_GATE_EVIDENCE] %s: verify_path %s → %s",
                                    req_id, resolved, "EXISTS" if exists else "NOT FOUND")

                elif tool_name in ("embeddings_service.search_embeddings",
                                   "evidence_collector.add_search_to_bundle",
                                   "arch_query.search_symbols"):
                    # v4.2: Search tools not available at spec stage — map to directory listing
                    # Use query/anchor to find relevant files via listing
                    query = args.get("query") or args.get("anchor", "")
                    search_path = None
                    if project_paths:
                        search_path = project_paths[0]
                        # If query looks like a path fragment, try listing that dir
                        for root in project_paths:
                            candidate = os.path.join(root, query.replace('.', os.sep))
                            if os.path.isdir(candidate):
                                search_path = candidate
                                break
                    if search_path:
                        success, listing = _host_list_directory(search_path, project_paths=project_paths)
                        er_results.append({
                            "tool": tool_name,
                            "path": search_path,
                            "success": success,
                            "content": f"Directory listing for context (search unavailable at spec stage):\n{listing[:2000]}" if success else None,
                            "error": listing if not success else None,
                        })
                    else:
                        er_results.append({
                            "tool": tool_name,
                            "skipped": True,
                            "reason": f"Search tool '{tool_name}' not available at spec stage; no project path to list",
                        })

                else:
                    # Unsupported tool at spec stage — log and skip
                    logger.info("[SPEC_GATE_EVIDENCE] %s: Skipping unsupported tool '%s' (args=%s)",
                                req_id, tool_name, list(args.keys()))
                    print(f"[SPEC_GATE_EVIDENCE] {req_id}: UNSUPPORTED TOOL '{tool_name}' "
                          f"args={list(args.keys())} — skipping")
                    er_results.append({
                        "tool": tool_name,
                        "skipped": True,
                        "reason": f"Tool '{tool_name}' not available at spec stage",
                    })

            # If we got ANY successful reads, mark this ER as fulfilled
            any_success = any(r.get("success") for r in er_results)
            if any_success:
                fulfilled_ids.add(req_id)
                # Build evidence text block for re-prompt
                evidence_text = f"\n### Evidence for {req_id} (need: {need})\n"
                for r in er_results:
                    if r.get("success") and r.get("content"):
                        label = r.get("file_path") or r.get("path", "unknown")
                        evidence_text += f"\n**{label}:**\n```\n{r['content']}\n```\n"
                    elif r.get("error"):
                        evidence_text += f"\n**Error:** {r['error']}\n"
                evidence_bundle_parts.append(evidence_text)
            else:
                # No successful reads — will be force-resolved after loops
                logger.warning("[SPEC_GATE_EVIDENCE] %s: All tool calls failed", req_id)
                print(f"[SPEC_GATE_EVIDENCE] {req_id}: ALL FAILED — "
                      f"results={[(r.get('tool','?'), r.get('error','?')[:100] if r.get('error') else r.get('reason','?')) for r in er_results]}")
                if er_results:  # Had tool calls but all failed
                    evidence_text = f"\n### Evidence for {req_id} — UNAVAILABLE\n"
                    for r in er_results:
                        if r.get("error"):
                            evidence_text += f"- {r.get('tool', '?')}: {r['error']}\n"
                        elif r.get("skipped"):
                            evidence_text += f"- {r.get('tool', '?')}: {r.get('reason', 'skipped')}\n"
                    evidence_bundle_parts.append(evidence_text)

        if not fulfilled_ids and not evidence_bundle_parts:
            logger.info("[SPEC_GATE_EVIDENCE] No evidence gathered — stopping loop")
            print("[SPEC_GATE_EVIDENCE] No evidence gathered — stopping loop")
            break

        # Strip fulfilled ERs → RESOLVED_REQUEST markers
        current_analysis = strip_fulfilled_requests(current_analysis, fulfilled_ids)

        # Re-prompt LLM with the original analysis + real evidence
        # v4.8: Smart evidence prioritisation — instead of blind concatenation
        # and truncation, score each evidence part by relevance to the job
        # and include the most relevant first within the budget.
        _MAX_EVIDENCE_CHARS = 60000  # v4.8: Raised from 40k — full files need room

        def _score_evidence_relevance(ev_text: str, job_goal: str, job_desc: str) -> float:
            """Score how relevant an evidence block is to the job.
            Higher = more relevant = included first.
            Uses keyword overlap between the evidence content and the job description."""
            # Extract meaningful words from goal + description
            import re as _re
            job_words = set(_re.findall(r'[a-z_]{4,}', (job_goal + ' ' + job_desc).lower()))
            # Remove very common words that would match everything
            stop_words = {'that', 'this', 'with', 'from', 'have', 'will', 'been', 'being',
                          'should', 'would', 'could', 'also', 'into', 'when', 'each',
                          'already', 'existing', 'current', 'ensure', 'which', 'their',
                          'them', 'they', 'than', 'then', 'what', 'where', 'were', 'here',
                          'there', 'other', 'some', 'more', 'only', 'same', 'does', 'none',
                          'true', 'false', 'self', 'return', 'import', 'file', 'path',
                          'line', 'string', 'list', 'dict', 'type', 'name', 'value'}
            job_words -= stop_words
            ev_lower = ev_text.lower()
            ev_words = set(_re.findall(r'[a-z_]{4,}', ev_lower))
            ev_words -= stop_words
            if not job_words:
                return 0.0
            # Score = overlap ratio
            overlap = job_words & ev_words
            score = len(overlap) / len(job_words)
            # Bonus for files that contain function definitions (more useful than __init__.py)
            if 'def ' in ev_text or 'async def ' in ev_text:
                score += 0.1
            # Bonus for files that contain class definitions (models, schemas)
            if 'class ' in ev_text:
                score += 0.05
            return score

        # Score and sort evidence parts by relevance
        scored_parts = []
        for part in evidence_bundle_parts:
            relevance = _score_evidence_relevance(part, goal, what_to_do)
            scored_parts.append((relevance, part))
        scored_parts.sort(key=lambda x: x[0], reverse=True)

        # Assemble within budget, most relevant first
        evidence_block = ""
        included_count = 0
        skipped_count = 0
        for relevance, part in scored_parts:
            if len(evidence_block) + len(part) <= _MAX_EVIDENCE_CHARS:
                evidence_block += part + "\n"
                included_count += 1
            else:
                # Try to include a truncated version if there's room for at least 2000 chars
                remaining = _MAX_EVIDENCE_CHARS - len(evidence_block)
                if remaining >= 2000:
                    evidence_block += part[:remaining - 200] + (
                        f"\n\n... [File truncated to fit budget. "
                        f"Relevance score: {relevance:.2f}]\n"
                    )
                    included_count += 1
                else:
                    skipped_count += 1

        if skipped_count > 0:
            evidence_block += (
                f"\n\n[{skipped_count} lower-relevance evidence block(s) omitted to stay within budget. "
                f"Included {included_count} blocks sorted by relevance to the job.]\n"
            )
            print(f"[SPEC_GATE_EVIDENCE] Smart prioritisation: included {included_count}, "
                  f"skipped {skipped_count} (budget={_MAX_EVIDENCE_CHARS})")
        else:
            print(f"[SPEC_GATE_EVIDENCE] All {included_count} evidence blocks fit within budget")

        # v4.4: Determine if this is the final loop
        is_final_loop = (loop_idx >= _EVIDENCE_MAX_LOOPS - 1)

        # v4.4: Cap previous analysis to prevent input explosion
        _MAX_PREV_ANALYSIS_CHARS = 15000
        prev_analysis_text = current_analysis
        if len(prev_analysis_text) > _MAX_PREV_ANALYSIS_CHARS:
            prev_analysis_text = prev_analysis_text[:_MAX_PREV_ANALYSIS_CHARS] + (
                f"\n\n... [Previous analysis truncated. "
                f"Use the evidence below to produce a complete, grounded analysis.]"
            )
            print(f"[SPEC_GATE_EVIDENCE] Previous analysis truncated to {_MAX_PREV_ANALYSIS_CHARS} chars")

        # v4.7: Always re-inject the original job description so the LLM
        # never loses sight of what it's building across evidence loops.
        # Without this, each re-prompt dilutes the original intent until
        # the LLM produces generic boilerplate instead of a job-specific spec.
        job_context = (
            f"--- ORIGINAL JOB (do not lose sight of this) ---\n\n"
            f"Feature Request:\n{goal}\n\n"
            f"Full Description:\n{what_to_do}\n\n"
            f"--- END ORIGINAL JOB ---\n\n"
        )

        if is_final_loop:
            # v4.4: FINAL LOOP — force spec production, no more ERs allowed
            re_prompt = (
                f"You have completed {loop_idx + 1} rounds of evidence gathering. "
                f"The orchestrator has read real files from the codebase for you across "
                f"all rounds.\n\n"
                f"THIS IS YOUR FINAL ROUND. You MUST now produce your complete, grounded "
                f"analysis. Do NOT emit any more EVIDENCE_REQUEST blocks — they will be "
                f"ignored. Use the evidence you have to produce the best possible analysis.\n\n"
                f"For anything you still don't know, use DECISION_ALLOWED with a sensible "
                f"default, or HUMAN_REQUIRED only if truly high-risk.\n\n"
                f"{job_context}"
                f"REQUIRED OUTPUT SECTIONS (all mandatory):\n"
                f"## Architecture Overview\n"
                f"## Implementation Steps (numbered, actionable, referencing real files)\n"
                f"## Files to Modify (with WHAT and WHY for each)\n"
                f"## New Files to Create (or state 'None needed')\n"
                f"## Acceptance Criteria (testable, specific)\n\n"
                f"--- PREVIOUS ANALYSIS ---\n\n"
                f"{prev_analysis_text}\n\n"
                f"--- EVIDENCE FROM CODEBASE ---\n\n"
                f"{evidence_block}\n\n"
                f"--- END EVIDENCE ---\n\n"
                f"Produce your FINAL grounded analysis now. All sections required. "
                f"No EVIDENCE_REQUEST blocks."
            )
            logger.info("[SPEC_GATE_EVIDENCE] FINAL LOOP — forcing spec production")
            print(f"[SPEC_GATE_EVIDENCE] FINAL LOOP — forcing spec production (no more ERs)")
        else:
            re_prompt = (
                f"You previously produced the analysis below, which contained "
                f"EVIDENCE_REQUEST blocks. The orchestrator has now fulfilled "
                f"{len(fulfilled_ids)} of those requests by reading actual files. "
                f"The fulfilled requests have been replaced with RESOLVED_REQUEST markers.\n\n"
                f"{job_context}"
                f"Please revise your analysis using the REAL evidence provided below. "
                f"Replace any assumptions or hallucinated architecture with what the "
                f"actual code shows. Keep all other sections intact. "
                f"If you still need more evidence, you may emit new EVIDENCE_REQUEST blocks "
                f"(with new unique IDs, not reusing resolved ones). Focus your new ERs on "
                f"the most critical gaps \u2014 prioritise files that directly affect the "
                f"implementation over general exploration.\n\n"
                f"--- PREVIOUS ANALYSIS (with RESOLVED_REQUEST markers) ---\n\n"
                f"{prev_analysis_text}\n\n"
                f"--- FULFILLED EVIDENCE ---\n\n"
                f"{evidence_block}\n\n"
                f"--- END EVIDENCE ---\n\n"
                f"Please provide your revised, grounded analysis."
            )

        logger.info("[SPEC_GATE_EVIDENCE] Re-prompting LLM with %d chars of evidence "
                    "(%d ERs fulfilled, %d unfulfilled)",
                    len(evidence_block), len(fulfilled_ids),
                    len(requests) - len(fulfilled_ids))
        print(f"[SPEC_GATE_EVIDENCE] Re-prompting LLM: {len(fulfilled_ids)} fulfilled, "
              f"{len(requests) - len(fulfilled_ids)} remaining")

        try:
            result = await llm_call_func(
                provider_id=provider_id,
                model_id=model_id,
                messages=[{"role": "user", "content": re_prompt}],
                system_prompt=CREATE_ANALYSIS_SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=8192,
                timeout_seconds=_CREATE_ANALYSIS_TIMEOUT,
            )

            if result.is_success() and result.content:
                current_analysis = result.content.strip()
                logger.info("[SPEC_GATE_EVIDENCE] Re-analysis success: %d chars", len(current_analysis))
                print(f"[SPEC_GATE_EVIDENCE] Re-analysis: {len(current_analysis)} chars")
            else:
                error_msg = getattr(result, 'error_message', 'Unknown error')
                logger.warning("[SPEC_GATE_EVIDENCE] Re-analysis LLM failed: %s — keeping current", error_msg)
                print(f"[SPEC_GATE_EVIDENCE] Re-analysis failed: {error_msg} — keeping current")
                break  # Don't loop further if LLM fails

        except Exception as exc:
            logger.warning("[SPEC_GATE_EVIDENCE] Re-analysis exception: %s — keeping current", exc)
            print(f"[SPEC_GATE_EVIDENCE] Re-analysis exception: {exc} — keeping current")
            break

    # Force-resolve any remaining ERs after all loops
    remaining = parse_evidence_requests(current_analysis)
    if remaining:
        remaining_ids = {r.get("id", "UNKNOWN") for r in remaining}
        logger.warning("[SPEC_GATE_EVIDENCE] Force-resolving %d remaining ER(s) after %d loops: %s",
                       len(remaining), _EVIDENCE_MAX_LOOPS, remaining_ids)
        print(f"[SPEC_GATE_EVIDENCE] Force-resolving {len(remaining)} remaining ER(s): {remaining_ids}")
        current_analysis = strip_forced_stop_requests(current_analysis, remaining_ids)

    return current_analysis


# =============================================================================
# LLM ANALYSIS (v2.0 — NEW)
# =============================================================================

# v2.1: Import governance rules for prompt injection
try:
    from app.pot_spec.governance_rules import SPEC_GATE_GOVERNANCE_PROMPT as _GOV_PROMPT
except ImportError:
    _GOV_PROMPT = ""

CREATE_ANALYSIS_SYSTEM_PROMPT = f"""You are an expert software architect analyzing a feature request.
{_GOV_PROMPT}

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
