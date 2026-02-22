import logging
import os
from app.pot_spec.grounded._simple_create_utils import _CREATE_ANALYSIS_TIMEOUT, _EVIDENCE_MAX_FILE_CHARS, _host_list_directory
from app.pot_spec.grounded._simple_create_utils import _EVIDENCE_MAX_LOOPS
from app.pot_spec.grounded._simple_create_utils import _host_read_file
from typing import Callable, List
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


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
