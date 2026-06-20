# FILE: app/pot_spec/grounded/_simple_create_evidence.py
# Purpose: Evidence fulfilment loop for SpecGate CREATE path.
# Called-by: app.pot_spec.grounded.simple_create
# Depends-on: app.llm.pipeline.evidence_loop, app.pot_spec.grounded._simple_create_utils_12, app.pot_spec.grounded._simple_create_utils_14, app.pot_spec.grounded._simple_create_utils_15
# Last-renovated: 2026-06-11
"""
Evidence fulfilment loop for SpecGate CREATE path.

Parses EVIDENCE_REQUEST blocks from LLM analysis, reads requested files
from host filesystem, and re-prompts LLM with real evidence.
Extracted from simple_create.py.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from app.pot_spec.grounded._simple_create_utils_12 import (
    _EVIDENCE_MAX_LOOPS,
    _find_file_in_projects,
)
from app.pot_spec.grounded._simple_create_utils_14 import (
    _CREATE_ANALYSIS_TIMEOUT,
    _EVIDENCE_MAX_FILE_CHARS,
    _host_list_directory,
)
from app.pot_spec.grounded._simple_create_utils_15 import _host_read_file
from app.pot_spec.grounded._sbx_fs import _sbx_isfile, _sbx_isdir, _sbx_exists
from app.pot_spec.grounded._stage_reasoning import spec_gate_reasoning, spec_gate_timeout_seconds

logger = logging.getLogger(__name__)

# v3.2-fix: Sandbox-aware filesystem checks for codebase paths.
# v4.3: Spec gate uses HOST filesystem, not sandbox.


async def fulfil_evidence_requests(
    llm_analysis: str,
    provider_id: str,
    model_id: str,
    llm_call_func: Callable,
    project_paths: List[str],
    goal: str = "",
    what_to_do: str = "",
    system_prompt: str = "",
) -> str:
    """Fulfil EVIDENCE_REQUEST blocks by reading actual files and re-prompting."""
    try:
        from app.llm.pipeline.evidence_loop import (
            parse_evidence_requests,
            strip_fulfilled_requests,
            strip_forced_stop_requests,
        )
    except ImportError as exc:
        logger.warning("[SPEC_GATE_EVIDENCE] Cannot import evidence_loop: %s", exc)
        return llm_analysis

    current_analysis = llm_analysis

    for loop_idx in range(_EVIDENCE_MAX_LOOPS):
        requests = parse_evidence_requests(current_analysis)
        if not requests:
            logger.info("[SPEC_GATE_EVIDENCE] Loop %d: LLM emitted no ERs — evidence complete", loop_idx + 1)
            print(f"[SPEC_GATE_EVIDENCE] Loop {loop_idx + 1}: LLM satisfied — no more evidence needed")
            break

        logger.info("[SPEC_GATE_EVIDENCE] Loop %d: %d ER(s)", loop_idx + 1, len(requests))

        fulfilled_ids: Set[str] = set()
        evidence_parts: List[str] = []

        for req in requests:
            req_id = req.get("id", "UNKNOWN")
            tool_calls = req.get("tool_calls", [])
            need = req.get("need", "")
            er_results = _dispatch_tool_calls(tool_calls, req_id, project_paths)

            any_success = any(r.get("success") for r in er_results)
            if any_success:
                fulfilled_ids.add(req_id)
            evidence_parts.append(_format_evidence_block(req_id, need, er_results))

        if not fulfilled_ids and not evidence_parts:
            logger.info("[SPEC_GATE_EVIDENCE] Loop %d: No evidence gathered at all — stopping", loop_idx + 1)
            break

        # v4.3: Diminishing returns — if we sent ERs but NONE succeeded,
        # the LLM is fishing in the wrong pond. Stop fishing but give the
        # LLM one final chance to produce a complete analysis with what it has.
        # v5.2: Previously this just 'break'd, which meant the LLM's response
        # (often ONLY ER blocks) got force-resolved to tiny stubs, destroying
        # all architecture analysis. Now we re-prompt as final round.
        total_ers = len(requests)
        if total_ers > 0 and len(fulfilled_ids) == 0:
            logger.info(
                "[SPEC_GATE_EVIDENCE] Loop %d: 0/%d ERs fulfilled — forcing FINAL round",
                loop_idx + 1, total_ers,
            )
            print(f"[SPEC_GATE_EVIDENCE] Loop {loop_idx + 1}: 0/{total_ers} ERs fulfilled — re-prompting as FINAL")
            # Build a final-round prompt with whatever evidence we DO have
            _unavail_block = _prioritise_evidence(evidence_parts, goal, what_to_do)
            _final_prompt = _build_re_prompt(
                current_analysis, _unavail_block, goal, what_to_do,
                loop_idx, fulfilled_ids, requests, is_final=True,
            )
            try:
                _final_result = await llm_call_func(
                    provider_id=provider_id,
                    model_id=model_id,
                    messages=[{"role": "user", "content": _final_prompt}],
                    system_prompt=system_prompt,
                temperature=1.0,  # stripped by registry when the model requires it
                max_tokens=16384,  # floored upward by registry per effort level
                timeout_seconds=max(_CREATE_ANALYSIS_TIMEOUT, spec_gate_timeout_seconds()),
                reasoning=spec_gate_reasoning(),  # v2.8: env-overridable (SPEC_GATE_REASONING_EFFORT)
                )
                if _final_result.is_success() and _final_result.content:
                    current_analysis = _final_result.content.strip()
                    logger.info(
                        "[SPEC_GATE_EVIDENCE] Final round produced %d chars",
                        len(current_analysis),
                    )
                    print(f"[SPEC_GATE_EVIDENCE] Final round: {len(current_analysis)} chars")
            except Exception as exc:
                logger.warning("[SPEC_GATE_EVIDENCE] Final round exception: %s", exc)
            break

        current_analysis = strip_fulfilled_requests(current_analysis, fulfilled_ids)

        # Build evidence block with smart prioritisation
        evidence_block = _prioritise_evidence(evidence_parts, goal, what_to_do)

        is_final = loop_idx >= _EVIDENCE_MAX_LOOPS - 1
        re_prompt = _build_re_prompt(
            current_analysis, evidence_block, goal, what_to_do,
            loop_idx, fulfilled_ids, requests, is_final,
        )

        try:
            result = await llm_call_func(
                provider_id=provider_id,
                model_id=model_id,
                messages=[{"role": "user", "content": re_prompt}],
                system_prompt=system_prompt,
                temperature=1.0,  # stripped by registry when the model requires it
                max_tokens=16384,  # floored upward by registry per effort level
                timeout_seconds=max(_CREATE_ANALYSIS_TIMEOUT, spec_gate_timeout_seconds()),
                reasoning=spec_gate_reasoning(),  # v2.8: env-overridable (SPEC_GATE_REASONING_EFFORT)
            )
            if result.is_success() and result.content:
                current_analysis = result.content.strip()
            else:
                break
        except Exception as exc:
            logger.warning("[SPEC_GATE_EVIDENCE] Re-analysis exception: %s", exc)
            break

    # Force-resolve remaining ERs
    remaining = parse_evidence_requests(current_analysis)
    if remaining:
        remaining_ids = {r.get("id", "UNKNOWN") for r in remaining}
        current_analysis = strip_forced_stop_requests(current_analysis, remaining_ids)

    return current_analysis


# =============================================================================
# TOOL DISPATCH
# =============================================================================

def _dispatch_tool_calls(
    tool_calls: List[Dict],
    req_id: str,
    project_paths: List[str],
) -> List[Dict]:
    """Dispatch host-direct reads for each tool call."""
    results: List[Dict] = []

    READ_TOOLS = {
        "sandbox_inspector.read_sandbox_file",
        "evidence_collector.add_file_read_to_bundle",
        "read_file",
        "sandbox_inspector.file_exists_in_sandbox",
        "arch_query.get_file_signatures",
    }
    LIST_TOOLS = {
        "sandbox_inspector.run_sandbox_discovery_chain",
        "list_directory",
    }
    VERIFY_TOOLS = {
        "evidence_collector.verify_path_exists",
        "evidence_collector.find_in_evidence",
    }
    SEARCH_TOOLS = {
        "embeddings_service.search_embeddings",
        "evidence_collector.add_search_to_bundle",
        "arch_query.search_symbols",
    }

    for tc in tool_calls:
        tool_name = tc.get("tool", "")
        args = tc.get("args", {})

        if tool_name in READ_TOOLS:
            file_path = args.get("file_path") or args.get("path", "")
            if file_path:
                success, content = _host_read_file(
                    file_path,
                    max_chars=args.get("max_chars", _EVIDENCE_MAX_FILE_CHARS),
                    project_paths=project_paths,
                )
                results.append({
                    "tool": tool_name, "file_path": file_path,
                    "success": success,
                    "content": content if success else None,
                    "error": content if not success else None,
                })

        elif tool_name in LIST_TOOLS:
            dir_path = args.get("anchor") or args.get("path", "")
            if dir_path:
                success, listing = _host_list_directory(dir_path, project_paths=project_paths)
                results.append({
                    "tool": tool_name, "path": dir_path,
                    "success": success,
                    "content": listing[:2000] if success else None,
                    "error": listing if not success else None,
                })

        elif tool_name in VERIFY_TOOLS:
            file_path = args.get("path") or args.get("file_path", "")
            if file_path:
                resolved = file_path.replace('/', os.sep).replace('\\', os.sep)
                if not _sbx_exists(resolved) and project_paths:
                    for root in project_paths:
                        candidate = os.path.join(root, resolved)
                        if _sbx_exists(candidate):
                            resolved = candidate
                            break
                exists = _sbx_exists(resolved)
                results.append({
                    "tool": tool_name, "file_path": resolved,
                    "success": exists,
                    "content": f"Path {'exists' if exists else 'NOT found'}: {resolved}" if exists else None,
                    "error": f"Path does not exist: {resolved}" if not exists else None,
                })

        elif tool_name in SEARCH_TOOLS:
            query = args.get("query") or args.get("anchor", "")
            search_path = project_paths[0] if project_paths else None
            if search_path:
                for root in project_paths:
                    candidate = os.path.join(root, query.replace('.', os.sep))
                    if _sbx_isdir(candidate):
                        search_path = candidate
                        break
                success, listing = _host_list_directory(search_path, project_paths=project_paths)
                results.append({
                    "tool": tool_name, "path": search_path,
                    "success": success,
                    "content": f"Directory listing (search unavailable at spec stage):\n{listing[:2000]}" if success else None,
                    "error": listing if not success else None,
                })
            else:
                results.append({"tool": tool_name, "skipped": True, "reason": "No project path"})
        else:
            results.append({"tool": tool_name, "skipped": True, "reason": f"Unsupported: {tool_name}"})

    return results


# =============================================================================
# EVIDENCE FORMATTING
# =============================================================================

def _format_evidence_block(req_id: str, need: str, results: List[Dict]) -> str:
    """Format evidence results into a text block for re-prompting."""
    text = f"\n### Evidence for {req_id} (need: {need})\n"
    any_success = any(r.get("success") for r in results)
    if not any_success:
        text = f"\n### Evidence for {req_id} — UNAVAILABLE\n"
    for r in results:
        if r.get("success") and r.get("content"):
            label = r.get("file_path") or r.get("path", "unknown")
            text += f"\n**{label}:**\n```\n{r['content']}\n```\n"
        elif r.get("error"):
            text += f"- {r.get('tool', '?')}: {r['error']}\n"
        elif r.get("skipped"):
            text += f"- {r.get('tool', '?')}: {r.get('reason', 'skipped')}\n"
    return text


def _score_evidence_relevance(ev_text: str, job_goal: str, job_desc: str) -> float:
    """Score how relevant an evidence block is to the job."""
    job_words = set(re.findall(r'[a-z_]{4,}', (job_goal + ' ' + job_desc).lower()))
    stop_words = {
        'that', 'this', 'with', 'from', 'have', 'will', 'been', 'being',
        'should', 'would', 'could', 'also', 'into', 'when', 'each',
        'already', 'existing', 'current', 'ensure', 'which', 'their',
        'them', 'they', 'than', 'then', 'what', 'where', 'were', 'here',
        'there', 'other', 'some', 'more', 'only', 'same', 'does', 'none',
        'true', 'false', 'self', 'return', 'import', 'file', 'path',
        'line', 'string', 'list', 'dict', 'type', 'name', 'value',
    }
    job_words -= stop_words
    ev_words = set(re.findall(r'[a-z_]{4,}', ev_text.lower())) - stop_words
    if not job_words:
        return 0.0
    score = len(job_words & ev_words) / len(job_words)
    if 'def ' in ev_text or 'async def ' in ev_text:
        score += 0.1
    if 'class ' in ev_text:
        score += 0.05
    return score


def _prioritise_evidence(parts: List[str], goal: str, what_to_do: str) -> str:
    """Score and assemble evidence within budget, most relevant first."""
    _MAX = 60000
    scored = [(ev, _score_evidence_relevance(ev, goal, what_to_do)) for ev in parts]
    scored.sort(key=lambda x: x[1], reverse=True)

    block = ""
    included = 0
    for ev_text, relevance in scored:
        if len(block) + len(ev_text) <= _MAX:
            block += ev_text + "\n"
            included += 1
        else:
            remaining = _MAX - len(block)
            if remaining >= 2000:
                block += ev_text[:remaining - 200] + f"\n\n... [Truncated, relevance={relevance:.2f}]\n"
                included += 1
    return block


# =============================================================================
# RE-PROMPT BUILDING
# =============================================================================

def _build_re_prompt(
    current_analysis: str,
    evidence_block: str,
    goal: str,
    what_to_do: str,
    loop_idx: int,
    fulfilled_ids: Set[str],
    requests: List[Dict],
    is_final: bool,
) -> str:
    """Build the re-prompt for the LLM with evidence."""
    _MAX_PREV = 15000
    prev = current_analysis
    if len(prev) > _MAX_PREV:
        prev = prev[:_MAX_PREV] + "\n\n... [Truncated]"

    job_context = (
        f"--- ORIGINAL JOB ---\n\n"
        f"Feature Request:\n{goal}\n\n"
        f"Full Description:\n{what_to_do}\n\n"
        f"--- END ORIGINAL JOB ---\n\n"
    )

    if is_final:
        return (
            f"You have completed {loop_idx + 1} rounds of evidence gathering. "
            f"THIS IS YOUR FINAL ROUND. Produce your complete analysis. "
            f"Do NOT emit EVIDENCE_REQUEST blocks.\n\n"
            f"{job_context}"
            f"REQUIRED: Architecture Overview, Implementation Steps, "
            f"Files to Modify, New Files, Acceptance Criteria.\n\n"
            f"--- PREVIOUS ANALYSIS ---\n\n{prev}\n\n"
            f"--- EVIDENCE ---\n\n{evidence_block}\n\n"
            f"Produce your FINAL grounded analysis now."
        )
    else:
        return (
            f"The orchestrator has fulfilled {len(fulfilled_ids)} evidence requests. "
            f"Refine your draft using the REAL evidence below.\n\n"
            f"{job_context}"
            f"--- PREVIOUS DRAFT ---\n\n{prev}\n\n"
            f"--- FULFILLED EVIDENCE ---\n\n{evidence_block}\n\n"
            f"REQUIRED OUTPUT: Produce a COMPLETE updated draft on this round, "
            f"containing Architecture Overview, Files to Modify, New Files to Create, "
            f"and Acceptance Criteria — every section, every time. The draft must "
            f"reflect what you learned from the evidence.\n\n"
            f"You MAY emit additional EVIDENCE_REQUEST blocks AFTER your draft if "
            f"the evidence raised new uncertainties — but ONLY for verifying specific "
            f"facts, never as a substitute for the draft itself. A response containing "
            f"only EVIDENCE_REQUEST blocks with no draft sections will be treated as a "
            f"failed round and the loop will be force-terminated."
        )