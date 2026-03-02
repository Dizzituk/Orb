# FILE: app/llm/pipeline/evidence_loop.py
"""
Evidence loop utilities for the Evidence-or-Request Contract.

v2.2 (2026-02-06): TOOL DISPATCH FIXES
- Fix: search_embeddings dispatcher now creates real DB session instead of passing LLM-provided args
- Fix: search_embeddings returns serializable results (SearchResult.__dict__)
- Note: sandbox_inspector import path already fixed in prior patch (app.llm.local_tools.zobie)
- BUILD_ID: 2026-02-06-v2.2-tool-dispatch-fixes

v2.1 (2026-02-06): ROBUST YAML PARSING FIX
- Fix: Windows backslashes in double-quoted YAML strings (D:\\Orb -> invalid \\O escape)
- Fix: Flat indentation where EVIDENCE_REQUEST: has sibling keys instead of children
- Fix: Regex fallback extraction when YAML parsing completely fails
- Three-layer defense: backslash escape -> YAML parse + restructure -> regex fallback
- Fixes: "Failed to parse EVIDENCE_REQUEST block at pos XXXX" errors in pipeline logs
- Fix: Trailing backslash before closing quote (\" edge case)
- BUILD_ID: 2026-02-06-v2.1-robust-yaml-parsing

v2.0 (2026-02-05): Initial implementation

Provides: parsing EVIDENCE_REQUEST blocks, stripping fulfilled/forced requests,
validating output structure (CRITICAL_CLAIMS terminal invariant), tool dispatch
with stage-level allowlisting.
"""

import re
import yaml
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple
from app.llm.pipeline._evidence_loop_utils_2 import JobContext, StageResult, _BLOCK_BOUNDARY_MARKERS, _BUILD_ID, _block_boundary_pattern, _extract_path_from_rag_hit, _request_block_pattern, _restructure_flat_evidence_request
from app.llm.pipeline._evidence_loop_utils_3 import _escape_backslashes_for_yaml, _regex_fallback_extract, _try_parse_block, execute_tool_call, parse_evidence_requests, strip_forced_stop_requests, validate_output_structure, validate_tool_call

logger = logging.getLogger(__name__)
print(f"[EVIDENCE_LOOP_LOADED] BUILD_ID={_BUILD_ID}")


# =============================================================================
# Block boundary markers — shared across all parsing functions
# =============================================================================


# =============================================================================
# YAML robustness helpers (v2.1)
# =============================================================================


# =============================================================================
# Parsing EVIDENCE_REQUEST blocks
# =============================================================================


# =============================================================================
# Stripping / replacing request blocks
# =============================================================================


def strip_fulfilled_requests(output: str, fulfilled_ids: set) -> str:
    """Replace fulfilled EVIDENCE_REQUEST blocks with RESOLVED_REQUEST markers."""
    for req_id in fulfilled_ids:
        pattern = _request_block_pattern(req_id)
        stub = (
            f'RESOLVED_REQUEST:\n'
            f'  id: "{req_id}"\n'
            f'  fulfilled_by: "orchestrator"\n'
            f'  evidence_added_to_bundle: true\n'
        )
        output = pattern.sub(stub, output)
    return output


# =============================================================================
# Output structure validation
# =============================================================================


# =============================================================================
# Tool Dispatch Table
# =============================================================================
# Non-Implementer stages access sandbox ONLY through sandbox_inspector
# (which internally uses sandbox_client with root-resolution, depth limits,
# and classification). Direct sandbox_client.* calls are Implementer-only.

TOOL_DISPATCH = {
    # ── Read-only: specgate, critical, critique, overwatcher ──
    "sandbox_inspector.run_sandbox_discovery_chain": {
        "allowed_stages": ["specgate", "critical", "critique", "overwatcher"],
    },
    "sandbox_inspector.read_sandbox_file": {
        "allowed_stages": ["specgate", "critical", "critique", "overwatcher"],
    },
    "sandbox_inspector.file_exists_in_sandbox": {
        "allowed_stages": ["specgate", "critical", "critique", "overwatcher"],
    },
    "evidence_collector.load_evidence": {
        "allowed_stages": ["specgate", "critical", "critique", "overwatcher"],
    },
    "evidence_collector.add_file_read_to_bundle": {
        "allowed_stages": ["specgate", "critical", "critique", "overwatcher"],
    },
    "evidence_collector.add_search_to_bundle": {
        "allowed_stages": ["specgate", "critical", "critique", "overwatcher"],
    },
    "evidence_collector.find_in_evidence": {
        "allowed_stages": ["specgate", "critical", "critique", "overwatcher"],
    },
    "evidence_collector.verify_path_exists": {
        "allowed_stages": ["specgate", "critical", "critique", "overwatcher"],
    },
    "embeddings_service.search_embeddings": {
        "allowed_stages": ["specgate", "critical", "critique", "overwatcher"],
    },
    "arch_query.search_symbols": {
        "allowed_stages": ["specgate", "critical", "critique", "overwatcher"],
    },
    "arch_query.get_file_signatures": {
        "allowed_stages": ["specgate", "critical", "critique", "overwatcher"],
    },
    # ── Implementer ONLY ──
    "sandbox_client.call_fs_tree":           {"allowed_stages": ["implementer"]},
    "sandbox_client.call_fs_contents":       {"allowed_stages": ["implementer"]},
    "sandbox_client.call_fs_write":          {"allowed_stages": ["implementer"]},
    "sandbox_client.call_fs_write_absolute": {"allowed_stages": ["implementer"]},
}


def tool_allowed(tool_name: str, stage_name: str) -> bool:
    """Check whether a tool call is permitted for the given stage."""
    entry = TOOL_DISPATCH.get(tool_name)
    if not entry:
        return False
    return stage_name in entry["allowed_stages"]


# =============================================================================
# Orchestrator Loop — run_stage_with_evidence()
# =============================================================================


async def run_stage_with_evidence(
    stage_name: str,
    stage_fn,
    context: JobContext,
    max_loops: int = 2,
) -> StageResult:
    """Run a pipeline stage, fulfilling evidence requests up to *max_loops*.

    Flow:
        1. Call ``stage_fn(context)`` to get initial output
        2. Parse EVIDENCE_REQUEST blocks from output
        3. Dispatch tool calls (stage-allowlisted)
        4. Replace fulfilled requests -> RESOLVED_REQUEST stubs
        5. Repeat 1-4 up to *max_loops*
        6. If CRITICAL requests remain with DECISION_ALLOWED -> force-resolve
        7. Final CRITICAL_CLAIMS terminal validation

    Args:
        stage_name: Pipeline stage identifier (e.g. "specgate", "critical")
        stage_fn:   Async callable ``(JobContext) -> StageResult``
        context:    Shared job context accumulating evidence
        max_loops:  Maximum evidence-fulfillment iterations (default 2)

    Returns:
        StageResult with final output and any unresolved HUMAN_REQUIRED items
    """
    context.force_resolve_only = False
    result = await stage_fn(context)

    # ── Evidence fulfillment loop ──
    for loop_idx in range(max_loops):
        requests = parse_evidence_requests(result.output)
        if not requests:
            break  # No requests — stage is done

        logger.info(
            "[evidence_loop] %s loop %d/%d: %d EVIDENCE_REQUEST(s)",
            stage_name, loop_idx + 1, max_loops, len(requests),
        )

        evidence_results: Dict[str, List] = {}
        fulfilled_ids: Set[str] = set()

        for req in requests:
            scope = req.get("scope", {})
            max_files = min(scope.get("max_files", 500), 1000)  # Hard cap

            # Validate tool calls against stage allowlist
            # v2.4: Also filter out empty-arg read calls (Sonnet sometimes sends bare tool names)
            validated_calls = []
            for tc in req.get("tool_calls", []):
                if not tool_allowed(tc.get("tool", ""), stage_name):
                    continue
                # Skip read_sandbox_file with no file_path — harmless no-op
                if tc.get("tool") == "sandbox_inspector.read_sandbox_file":
                    _fp = tc.get("args", {}).get("file_path") or tc.get("args", {}).get("path") or ""
                    if not _fp.strip():
                        logger.debug("[evidence_loop] Filtered empty read_sandbox_file call")
                        continue
                validated_calls.append(tc)

            # Execute validated tool calls
            call_results: List[Dict] = []
            for tc in validated_calls:
                tool_result = await execute_tool_call(
                    tc,
                    stage_name=stage_name,
                    bundle=context.evidence_bundle,
                    max_files=max_files,
                )
                call_results.append(tool_result)

                # AUTO-QUEUE: RAG hit -> confirming file read
                if tc.get("tool") == "embeddings_service.search_embeddings":
                    rag_results = tool_result.get("results", []) if isinstance(tool_result, dict) else []
                    for rag_hit in rag_results:
                        path = _extract_path_from_rag_hit(rag_hit)
                        if path and context.evidence_bundle is not None:
                            try:
                                from app.pot_spec.evidence_collector import add_file_read_to_bundle
                                file_content = add_file_read_to_bundle(
                                    context.evidence_bundle, path,
                                )
                                call_results.append({
                                    "auto_read": path,
                                    "content": file_content[:2000] if file_content else None,
                                })
                            except Exception:
                                pass

            req_id = req.get("id", "UNKNOWN")
            evidence_results[req_id] = call_results
            fulfilled_ids.add(req_id)


        # v4.3: Diminishing returns - if all ERs dispatched but none returned content, stop
        _any_content = any(
            any(r.get("content") or r.get("auto_read") for r in evidence_results.get(req.get("id", ""), []))
            for req in requests
        )
        if requests and not _any_content:
            logger.info(
                "[evidence_loop] %s loop %d: 0/%d ERs produced content - stopping",
                stage_name, loop_idx + 1, len(requests),
            )
            break
        # Strip fulfilled requests -> RESOLVED_REQUEST stubs
        result.output = strip_fulfilled_requests(result.output, fulfilled_ids)

        # Store structured fulfilled evidence for stage re-prompt
        context.fulfilled_evidence = {
            req_id: {
                "tools_called": [
                    tc.get("tool") for r in requests if r.get("id") == req_id
                    for tc in r.get("tool_calls", [])
                ],
                "results": evidence_results.get(req_id, []),
            }
            for req_id in fulfilled_ids
        }
        context.fulfilled_evidence_ids = fulfilled_ids
        context.evidence_results = evidence_results

        # Re-prompt stage with accumulated evidence
        result = await stage_fn(context)

    # ── Handle unresolved CRITICAL claims after all loops ──
    remaining = parse_evidence_requests(result.output)
    critical_remaining = [r for r in remaining if r.get("severity") == "CRITICAL"]

    if critical_remaining:
        logger.warning(
            "[evidence_loop] %s: %d CRITICAL requests unresolved after %d loops",
            stage_name, len(critical_remaining), max_loops,
        )

        for req in critical_remaining:
            fallback = req.get("fallback_if_not_found", "HUMAN_REQUIRED")

            if fallback == "DECISION_ALLOWED":
                # Re-prompt the STAGE to decide (orchestrator NEVER fabricates)
                context.force_resolve_only = True
                context.force_resolve = {
                    req.get("id", "UNKNOWN"): {
                        "instruction": (
                            f"Evidence for {req.get('id')} was not found after "
                            f"{max_loops} search loops. You MUST now output either "
                            f"a DECISION block (with rationale and revisit_if) or "
                            f"a HUMAN_REQUIRED block. Do not emit EVIDENCE_REQUEST."
                        ),
                        "original_need": req.get("need", ""),
                    },
                }
                result = await stage_fn(context)

                # Enforce: if stage STILL emitted EVIDENCE_REQUEST -> HUMAN_REQUIRED
                leftover = parse_evidence_requests(result.output)
                for lr in leftover:
                    if lr.get("severity") == "CRITICAL":
                        result.unresolved_human_required.append({
                            "id": lr.get("id"),
                            "need": lr.get("need", ""),
                            "why": lr.get("why", ""),
                            "scope": lr.get("scope", {}),
                            "searched": lr.get("tool_calls", []),
                        })
                # Strip with FORCED_RESOLUTION stubs
                result.output = strip_forced_stop_requests(
                    result.output,
                    {lr.get("id") for lr in leftover},
                )
                context.force_resolve_only = False

            else:  # HUMAN_REQUIRED
                result.unresolved_human_required.append({
                    "id": req.get("id"),
                    "need": req.get("need", ""),
                    "why": req.get("why", ""),
                    "scope": req.get("scope", {}),
                    "searched": req.get("tool_calls", []),
                })

    # ── Final validation — CRITICAL_CLAIMS must be terminal ──
    if not validate_output_structure(result.output):
        logger.error(
            "[evidence_loop] CRITICAL_CLAIMS is not terminal in %s output",
            stage_name,
        )
        # Don't crash the pipeline — log and continue.
        # The critique stage will catch this as an 'unresolved_critical' blocker.

    return result


__all__ = [
    "parse_evidence_requests",
    "strip_fulfilled_requests",
    "strip_forced_stop_requests",
    "validate_output_structure",
    "TOOL_DISPATCH",
    "tool_allowed",
    "validate_tool_call",
    # YAML helpers (v2.1)
    "_escape_backslashes_for_yaml",
    "_restructure_flat_evidence_request",
    "_regex_fallback_extract",
    "_try_parse_block",
    # Orchestrator loop
    "StageResult",
    "JobContext",
    "execute_tool_call",
    "run_stage_with_evidence",
]
