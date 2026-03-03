"""
Step 4: Process all file tasks.

Contains the main per-file task loop with three-strike error handling.
Each file goes through: quarantine check → contract extraction →
prompt build → LLM call → syntax guard → write → verify.

Extracted from orchestrator.py monolith.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ..sandbox_client import SandboxClient
from .constants import MAX_STRIKES_PER_TASK, MODIFY_EDIT_MODE_THRESHOLD
from .execution_state import ExecutionContext
from .step_quarantine import check_quarantine_skip as _check_quarantine_skip
from .step_write import _delegate_write, _record_success
from .context import (
    _read_existing_file,
    _read_source_context,
    _format_job_context,
    _extract_file_interfaces,
    _extract_router_registrations,
)
from .helpers import (
    _extract_llm_content,
    _strip_markdown_fences,
    _sanitise_python_content,
    _check_python_syntax,
)
from .parsing import extract_section_for_file, _extract_verbatim_code_from_architecture
from .arch_code_extractor import extract_code_for_files, ExtractionResult


# ---------------------------------------------------------------------------
# v1.1: Strip implementation briefs before code extraction
# ---------------------------------------------------------------------------
def _strip_brief_sections(arch_text: str) -> str:
    """Remove Implementation Brief sections from architecture text.

    The implementation compiler prepends per-file briefs to the arch doc.
    These contain scaffold code blocks (empty shells) that the code
    extractor must NOT pick up. The real code blocks from the
    architecture document are what we want.

    Strategy: find "## IMPLEMENTATION BRIEFS" and strip everything from
    the start of the document up to (but not including) the real
    architecture content. The real arch always starts with "SPEC_ID:"
    which is the definitive boundary marker.

    v1.2 (2026-03-02): Fixed over-stripping that removed arch content.
    The previous version tracked --- separators but the arch doc itself
    contains --- separators, causing the entire document to be stripped.
    """
    marker = "## IMPLEMENTATION BRIEFS"
    if marker not in arch_text:
        return arch_text

    # The real architecture always starts with SPEC_ID: or SPEC_HASH:
    # Find that boundary and strip everything before it.
    lines = arch_text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("SPEC_ID:") or line.startswith("SPEC_HASH:"):
            remaining = "\n".join(lines[i:])
            return remaining

    # Fallback: if no SPEC_ID found, return unmodified to avoid data loss
    return arch_text

from .arch_code_merge import decide_merge_strategy, MergeDecision
from .path_resolution import _resolve_multi_root_path, _infer_lang_from_path
from .prompts import _parse_edit_pairs
from .source_extraction import _detect_source_files_from_architecture
from .step_task_prompt import (
    extract_contract_block,
    strip_competing_signatures,
    format_strike_error_block,
    prepend_strike_error,
    build_create_prompt,
    build_modify_prompt,
    build_verify_prompt,
    inject_experience_and_rag,
    run_preflight_gate,
)
from .step_task_verify import verify_written_file

logger = logging.getLogger(__name__)


# Quarantine skip check (v5.13) - extracted to step_quarantine.py


# ---------------------------------------------------------------------------
# Single-task strike loop
# ---------------------------------------------------------------------------

async def _process_single_task(
    task: Dict,
    task_index: int,
    ctx: ExecutionContext,
    client: SandboxClient,
    run_implementer_task,
    run_implementer_edit_task,
    extraction_result: Optional[ExtractionResult] = None,
) -> bool:
    """Process one file task with three-strike error handling.

    Returns True if task succeeded, False if it failed.
    """
    file_info = task["info"]
    action = task["action"]
    rel_path = file_info["path"]
    abs_path = _resolve_multi_root_path(rel_path, ctx.sandbox_base)

    logger.info("[arch_exec] [%d/%d] %s: %s", task_index, ctx.total_operations, action.upper(), rel_path)
    print(f"[ARCH_EXEC] [{task_index}/{ctx.total_operations}] {action.upper()}: {rel_path}")

    ctx.add_trace("FILE_TASK_START", "processing", {
        "operation": action,
        "relative_path": rel_path,
        "absolute_path": abs_path,
        "task_number": task_index,
    })

    # Quarantine skip for MODIFY/DELETE
    if action == "modify":
        if _check_quarantine_skip(rel_path, abs_path, file_info, ctx, client):
            ctx.files_modified += 1
            ctx.artifacts_written.append(abs_path)
            return True

    # Three-strike loop
    task_success = False
    last_error: Optional[str] = None
    job_checker_errors: List[str] = []
    structured_sig_mismatches: list = []
    use_edit_mode = False  # v3.2: Initialise before loop to avoid UnboundLocalError
    file_content = None    # v3.2: Same — referenced in _record_success after loop

    for strike in range(1, MAX_STRIKES_PER_TASK + 1):
        logger.info("[arch_exec] %s strike %d/%d", rel_path, strike, MAX_STRIKES_PER_TASK)

        # v2.6: Auto-generated __init__.py — skip LLM
        if rel_path.endswith("__init__.py") and file_info.get("description", "").startswith("v2.6 auto-created"):
            file_content = "# Auto-generated by architecture executor v2.6\n"
            logger.info("[arch_exec] v2.6 Direct-writing __init__.py: %s", rel_path)
            try:
                impl_result = await run_implementer_task(
                    path=abs_path, content=file_content,
                    action="create", ensure_parents=True, client=client,
                )
                if impl_result.success:
                    task_success = True
                else:
                    last_error = f"Implementer write failed for __init__.py: {impl_result.error}"
            except Exception as e:
                last_error = f"__init__.py write exception: {e}"
            break

        # --- v1.0 Code Block Extraction: merge decision ---
        merge_decision = None
        if extraction_result and action == "create" and strike == 1:
            try:

                merge_decision = decide_merge_strategy(
                    extraction_result, rel_path,
                    action=action,
                )

                if merge_decision.skip_llm and merge_decision.content:
                    # Direct use: extracted code IS the file content
                    file_content = merge_decision.content
                    logger.info(
                        "[arch_exec] v1.0 DIRECT_EXTRACTION: %s (%d chars, confidence=%.2f) — %s",
                        rel_path, len(file_content), merge_decision.confidence,
                        merge_decision.reason,
                    )
                    print(
                        f"[ARCH_EXEC] v1.0 DIRECT_EXTRACTION: {rel_path} "
                        f"({len(file_content)} chars) — skipping LLM"
                    )
                    ctx.add_trace("DIRECT_EXTRACTION", "success", {
                        "path": rel_path,
                        "chars": len(file_content),
                        "confidence": merge_decision.confidence,
                        "reason": merge_decision.reason,
                    })

                    # Write directly — same path as verbatim/deterministic extraction
                    try:
                        impl_result = await run_implementer_task(
                            path=abs_path, content=file_content,
                            action=action, ensure_parents=True, client=client,
                        )
                        if impl_result.success:
                            task_success = True
                        else:
                            last_error = f"Direct extraction write failed: {impl_result.error}"
                    except Exception as e:
                        last_error = f"Direct extraction write exception: {e}"
                    break

                elif merge_decision.use_verify_prompt:
                    logger.info(
                        "[arch_exec] v1.0 PREFILL_MODE: %s (confidence=%.2f) — %s",
                        rel_path, merge_decision.confidence, merge_decision.reason,
                    )
                    print(
                        f"[ARCH_EXEC] v1.0 PREFILL_MODE: {rel_path} — "
                        f"LLM will verify extracted code"
                    )
                    ctx.add_trace("PREFILL_MODE", "prepared", {
                        "path": rel_path,
                        "confidence": merge_decision.confidence,
                        "reason": merge_decision.reason,
                    })

            except Exception as me:
                logger.warning(
                    "[arch_exec] v1.0 Merge decision failed for %s (non-fatal): %s",
                    rel_path, me,
                )
                merge_decision = None

        # --- Extract architecture context ---
        file_context = extract_section_for_file(ctx.architecture_content, rel_path)
        if not file_context:
            last_error = f"No architecture context found for {rel_path}"
            logger.warning("[arch_exec] %s", last_error)
            break

        # --- Build cross-file context ---
        job_context_section = _format_job_context(ctx.job_context, ctx.router_registrations)
        use_edit_mode = False
        verbatim_content = None
        file_content = None

        # --- Contract injection (v5.23) ---
        contract_block, contract_sigs, bare_names = extract_contract_block(
            ctx.interface_contract, rel_path,
        )
        if contract_block:
            ctx.add_trace("CONTRACT_INJECT", "injected", {
                "path": rel_path,
                "signatures": contract_sigs,
                "bare_names": bare_names,
                "count": len(contract_sigs) + len(bare_names),
            })

        # Strip competing signatures from arch context
        file_context_for_prompt = strip_competing_signatures(file_context, contract_sigs)

        # --- Build prompt based on action ---
        try:
            if action == "create":
                verbatim_content = _extract_verbatim_code_from_architecture(
                    file_context, rel_path,
                )
                if verbatim_content:
                    print(f"[ARCH_EXEC] v1.13 VERBATIM extraction: {rel_path} ({len(verbatim_content)} chars)")
                    ctx.add_trace("VERBATIM_EXTRACTION", "success", {
                        "path": rel_path, "chars": len(verbatim_content),
                    })

                # v1.0: Use verification prompt if merge decision says prefill
                if (merge_decision and merge_decision.use_verify_prompt
                        and merge_decision.content and strike == 1):
                    user_prompt, system_prompt = build_verify_prompt(
                        rel_path, merge_decision.content,
                        file_context_for_prompt, contract_block,
                        job_context_section, ctx.available_modules_evidence,
                    )
                    ctx.add_trace("VERIFY_PROMPT_USED", "injected", {
                        "path": rel_path,
                        "prefill_chars": len(merge_decision.content),
                    })
                else:
                    user_prompt, system_prompt = build_create_prompt(
                        rel_path, file_context_for_prompt, contract_block,
                        job_context_section, ctx.available_modules_evidence,
                    )

                # Source file context injection (v3.0)
                try:
                    source_files = _detect_source_files_from_architecture(
                        file_section=file_context,
                        architecture_content=ctx.architecture_content,
                        rel_path=rel_path,
                    )
                    if source_files:
                        source_ctx = await _read_source_context(client, source_files, ctx.sandbox_base)
                        if source_ctx:
                            # Insert before the trailing "Output ONLY..." line
                            insert_pos = user_prompt.rfind("Output ONLY")
                            if insert_pos > 0:
                                user_prompt = user_prompt[:insert_pos] + f"{source_ctx}\n\n" + user_prompt[insert_pos:]
                except Exception as e:
                    logger.warning("[arch_exec] v3.0 Source context failed for %s: %s", rel_path, e)

            else:
                # MODIFY: read existing file
                existing_content = await _read_existing_file(client, abs_path)
                if existing_content is None:
                    last_error = f"Cannot read existing file for modification: {abs_path}"
                    logger.error("[arch_exec] %s", last_error)
                    break

                file_char_count = len(existing_content)
                use_edit_mode = file_char_count >= MODIFY_EDIT_MODE_THRESHOLD

                if file_char_count > 150_000:
                    print(f"[ARCH_EXEC] ⚠️ Very large MODIFY: {rel_path} ({file_char_count:,} chars)")
                elif use_edit_mode:
                    print(f"[ARCH_EXEC] v1.13 Large MODIFY: {rel_path} ({file_char_count:,} chars) — using edit mode")
                    ctx.add_trace("MODIFY_EDIT_MODE", "enabled", {"path": rel_path, "chars": file_char_count})

                user_prompt, system_prompt = build_modify_prompt(
                    rel_path, existing_content, file_context_for_prompt,
                    contract_block, job_context_section,
                    ctx.available_modules_evidence, use_edit_mode,
                )

            # --- Verbatim shortcut or LLM call ---
            if verbatim_content and strike == 1:
                file_content = verbatim_content
            else:
                if verbatim_content and strike > 1:
                    verbatim_content = None

                # Prepend strike error if retrying
                if strike > 1 and last_error:
                    error_block = format_strike_error_block(
                        strike, MAX_STRIKES_PER_TASK, last_error, structured_sig_mismatches,
                    )
                    user_prompt = prepend_strike_error(user_prompt, error_block)

                # Inject experience + RAG
                system_prompt = inject_experience_and_rag(
                    system_prompt, rel_path, action, file_context[:200],
                )

                # Pre-flight gate
                run_preflight_gate(ctx.interface_contract, rel_path, user_prompt, system_prompt)

                # LLM call
                llm_result = await ctx.llm_call_fn(
                    provider_id=ctx.impl_provider,
                    model_id=ctx.impl_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=ctx.impl_max_tokens,
                    timeout_seconds=600,
                )
                file_content = _extract_llm_content(llm_result)
                file_content = _strip_markdown_fences(file_content)

                if not file_content or len(file_content.strip()) < 10:
                    last_error = "LLM returned empty/minimal content"
                    logger.warning("[arch_exec] Strike %d: %s for %s", strike, last_error, rel_path)
                    ctx.add_trace("FILE_TASK_STRIKE", f"strike_{strike}", {"path": rel_path, "error": last_error})
                    continue

                logger.info("[arch_exec] LLM generated %d chars for %s (strike %d)", len(file_content), rel_path, strike)

        except Exception as e:
            last_error = f"LLM call failed: {e}"
            logger.exception("[arch_exec] Strike %d: %s", strike, last_error)
            ctx.add_trace("FILE_TASK_STRIKE", f"strike_{strike}", {"path": rel_path, "error": last_error})
            continue

        # --- Python syntax guard (v1.1) ---
        if rel_path.endswith('.py') and not use_edit_mode:
            file_content, sanitise_warnings = _sanitise_python_content(file_content, rel_path)
            all_prose = False
            for sw in sanitise_warnings:
                logger.warning("[arch_exec] %s", sw)
                if "ALL_PROSE_REJECTED" in sw:
                    all_prose = True
                ctx.add_trace("SANITISE_PYTHON", "stripped_preamble", {"path": rel_path, "warning": sw[:300]})

            if all_prose or not (file_content and file_content.strip()):
                last_error = (
                    f"ALL_PROSE_REJECTED: You wrote markdown/architecture "
                    f"instructions instead of Python code for {rel_path}. "
                    f"Output ONLY valid Python source code."
                )
                job_checker_errors.append(last_error)
                ctx.add_trace("ALL_PROSE_REJECTED", f"strike_{strike}", {"path": rel_path, "error": last_error})
                continue

            syntax_error = _check_python_syntax(file_content, rel_path)
            if syntax_error:
                last_error = f"Syntax guard FAILED: {syntax_error}"
                job_checker_errors.append(last_error)
                ctx.add_trace("SYNTAX_GUARD_FAIL", f"strike_{strike}", {"path": rel_path, "error": syntax_error[:500]})
                continue

        # --- Delegate write ---
        try:
            impl_result = await _delegate_write(
                abs_path, rel_path, file_content, action,
                use_edit_mode, client, ctx,
                run_implementer_task, run_implementer_edit_task,
            )
            if not impl_result.success:
                last_error = f"Implementer write failed: {impl_result.error}"
                ctx.add_trace("FILE_TASK_STRIKE", f"strike_{strike}", {"path": rel_path, "error": last_error})
                continue
        except Exception as e:
            last_error = f"Implementer exception: {e}"
            logger.exception("[arch_exec] Strike %d: %s", strike, last_error)
            ctx.add_trace("FILE_TASK_STRIKE", f"strike_{strike}", {"path": rel_path, "error": last_error})
            continue

        # --- Independent verification ---
        from .sandbox_ops import _verify_file_via_sandbox
        verify = _verify_file_via_sandbox(client, abs_path, expected_min_chars=10)
        if not verify["valid"]:
            last_error = f"Overwatcher verification failed: {verify['error'] or 'file too short/missing'}"
            ctx.add_trace("FILE_TASK_STRIKE", f"strike_{strike}", {"path": rel_path, "error": last_error})
            continue

        # --- Three-layer verification pipeline ---
        vr = await verify_written_file(
            rel_path, file_content, ctx, strike, job_checker_errors,
        )
        if not vr.passed:
            last_error = vr.error or "Verification failed"
            if vr.job_checker_error:
                job_checker_errors.append(vr.job_checker_error)
            structured_sig_mismatches = vr.structured_sig_mismatches
            ctx.add_trace("FILE_TASK_STRIKE", f"strike_{strike}", {"path": rel_path, "error": last_error})
            continue

        # SUCCESS
        task_success = True
        break

    # --- Record result ---
    if task_success:
        await _record_success(ctx, client, rel_path, abs_path, action, file_content, use_edit_mode)
    else:
        ctx.files_failed += 1
        logger.error(
            "[arch_exec] ✗ %s %s FAILED after %d strikes: %s",
            action.upper(), rel_path, MAX_STRIKES_PER_TASK, last_error,
        )
        print(f"[ARCH_EXEC] ✗ {action.upper()} {rel_path} FAILED: {last_error}")
        ctx.add_trace("FILE_TASK_FAILED", "exhausted_strikes", {
            "path": rel_path, "strikes": MAX_STRIKES_PER_TASK, "last_error": last_error,
        })

    return task_success


# Write delegation & success recording — extracted to step_write.py


# ---------------------------------------------------------------------------
# Public entry point — process all tasks
# ---------------------------------------------------------------------------

async def process_all_tasks(
    ctx: ExecutionContext,
    client: SandboxClient,
) -> None:
    """Process all file tasks (creates then modifies) with the strike loop."""
    from app.overwatcher.implementer import run_implementer_task, run_implementer_edit_task

    # v1.0: Ensure frontend root exists if any tasks target it
    from .path_resolution import ensure_frontend_root_exists
    from .constants import FRONTEND_PREFIX, FRONTEND_ROOT
    _has_frontend_tasks = any(
        f.get('path', '').startswith(FRONTEND_PREFIX)
        or f.get('path', '').startswith('src/')
        or f.get('path', '').startswith('public/')
        for f in (ctx.new_files + ctx.modified_files)
    )
    if _has_frontend_tasks:
        _fe_ok = ensure_frontend_root_exists(client)
        if _fe_ok:
            logger.info("[arch_exec] v1.0 Frontend root confirmed: %s", FRONTEND_ROOT)
        else:
            logger.warning(
                "[arch_exec] v1.0 Frontend root %s unavailable "
                "-- frontend files may fail", FRONTEND_ROOT,
            )

    all_tasks = (
        [{"info": f, "action": "create"} for f in ctx.new_files]
        + [{"info": f, "action": "modify"} for f in ctx.modified_files]
    )
    create_count = len(ctx.new_files)

    # v1.0 Code Block Extraction: extract code from arch doc BEFORE task loop
    extraction_result = None
    if ctx.architecture_content:
        try:
            all_file_paths = [
                f["path"] for f in (ctx.new_files + ctx.modified_files)
            ]
            # v1.1: Strip implementation briefs to avoid scaffold contamination
            clean_arch = _strip_brief_sections(ctx.architecture_content)
            extraction_result = extract_code_for_files(
                clean_arch, all_file_paths,
            )
            if extraction_result.file_count > 0:
                logger.info(
                    "[arch_exec] v1.0 CODE_EXTRACTION: %d/%d files have "
                    "extractable code from architecture",
                    extraction_result.file_count, len(all_file_paths),
                )
                print(
                    f"[ARCH_EXEC] v1.0 CODE_EXTRACTION: "
                    f"{extraction_result.file_count}/{len(all_file_paths)} "
                    f"files have extractable architecture code"
                )
                ctx.add_trace("CODE_EXTRACTION", "success", {
                    "files_with_code": extraction_result.file_count,
                    "total_files": len(all_file_paths),
                    "warnings": extraction_result.warnings[:5],
                })
            else:
                logger.info("[arch_exec] v1.0 CODE_EXTRACTION: no extractable code found")
        except Exception as ce:
            logger.warning("[arch_exec] v1.0 Code extraction failed (non-fatal): %s", ce)
            extraction_result = None

    for i, task in enumerate(all_tasks, 1):
        await _process_single_task(
            task, i, ctx, client,
            run_implementer_task, run_implementer_edit_task,
            extraction_result=extraction_result,
        )

        # v2.5: Two-pass boundary — after all CREATEs, refresh interfaces
        if i == create_count and ctx.created_file_contents:
            logger.info(
                "[arch_exec] v2.5 Two-pass: re-extracting interfaces from %d created files",
                len(ctx.created_file_contents),
            )
            for path, content in ctx.created_file_contents.items():
                try:
                    refreshed = _extract_file_interfaces(path, content)
                    ctx.job_context[path] = refreshed
                except Exception as e:
                    logger.warning("[arch_exec] v2.5 Two-pass extraction failed for %s: %s", path, e)
            ctx.add_trace("TWO_PASS_CONTEXT_REFRESH", "success", {
                "files_refreshed": list(ctx.created_file_contents.keys()),
            })
