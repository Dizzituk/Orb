"""
Write delegation and success recording for architecture executor tasks.

Handles file write dispatch (edit mode vs full rewrite) and success
recording (counter updates, interface extraction, context refresh).

Extracted from step_process_task.py for file size compliance.
"""
from __future__ import annotations

import logging
from typing import Optional

from ..sandbox_client import SandboxClient
from .execution_state import ExecutionContext
from .context import (
    _read_existing_file,
    _extract_file_interfaces,
    _extract_router_registrations,
)
from .prompts import _parse_edit_pairs
from .path_resolution import _infer_lang_from_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Write delegation
# ---------------------------------------------------------------------------

async def _delegate_write(
    abs_path, rel_path, file_content, action,
    use_edit_mode, client, ctx,
    run_implementer_task, run_implementer_edit_task,
):
    """Delegate the file write to the Implementer (edit mode or full rewrite)."""
    if use_edit_mode:
        edit_pairs = _parse_edit_pairs(file_content)
        if edit_pairs is None:
            logger.warning("[arch_exec] v1.13 Edit pair parsing failed for %s — falling back", rel_path)
            ctx.add_trace("EDIT_PARSE_FALLBACK", "parse_failed", {"path": rel_path})
            return await run_implementer_task(
                path=abs_path, content=file_content,
                action=action, ensure_parents=True, client=client,
            )

        print(f"[ARCH_EXEC] v1.13 Applying {len(edit_pairs)} targeted edits to {rel_path}")
        edit_result = await run_implementer_edit_task(
            path=abs_path, edits=edit_pairs, client=client,
        )

        class _Adapter:
            def __init__(self, er):
                self.success = er.success
                self.chars_written = er.chars_after
                self.verified = er.verified
                self.error = er.error

        if edit_result.edits_failed > 0:
            logger.warning(
                "[arch_exec] v1.13 %d/%d edits failed for %s",
                edit_result.edits_failed,
                edit_result.edits_applied + edit_result.edits_failed,
                rel_path,
            )
            ctx.add_trace("EDIT_PARTIAL", "some_failed", {
                "path": rel_path,
                "applied": edit_result.edits_applied,
                "failed": edit_result.edits_failed,
            })
        return _Adapter(edit_result)

    return await run_implementer_task(
        path=abs_path, content=file_content,
        action=action, ensure_parents=True, client=client,
    )


# ---------------------------------------------------------------------------
# Success recording
# ---------------------------------------------------------------------------

async def _record_success(
    ctx: ExecutionContext,
    client: SandboxClient,
    rel_path: str,
    abs_path: str,
    action: str,
    file_content: str,
    use_edit_mode: bool,
) -> None:
    """Record a successful task: update counters, extract interfaces."""
    if action == "create":
        ctx.files_created += 1
        ctx.created_file_contents[rel_path] = file_content
    else:
        ctx.files_modified += 1
        if use_edit_mode:
            try:
                actual = await _read_existing_file(client, abs_path)
                if actual:
                    file_content = actual
            except Exception:
                pass
        if rel_path.endswith('.py'):
            try:
                regs = _extract_router_registrations(file_content)
                if regs:
                    ctx.router_registrations.update(regs)
            except Exception:
                pass

    ctx.artifacts_written.append(abs_path)
    ctx.existing_sandbox_files.add(rel_path.replace("\\", "/"))

    try:
        summary = _extract_file_interfaces(rel_path, file_content)
        ctx.job_context[rel_path] = summary
    except Exception as e:
        logger.warning("[arch_exec] v2.3 Interface extraction failed for %s: %s", rel_path, e)

    # v3.2: Log content preview so we can see what was actually written
    _content_len = len(file_content) if file_content else 0
    _preview_lines = (file_content or '').split('\n')[:15]
    _preview = '\n'.join(f'    | {ln}' for ln in _preview_lines)
    _truncated = ' (truncated)' if file_content and len(file_content.split('\n')) > 15 else ''
    logger.info(
        "[arch_exec] ✓ %s %s (%d chars)\n%s%s",
        action.upper(), rel_path, _content_len, _preview, _truncated,
    )
    print(f"[ARCH_EXEC] ✓ {action.upper()} {rel_path} ({_content_len} chars)")
    print(f"[ARCH_EXEC] CONTENT PREVIEW ({rel_path}):\n{_preview}{_truncated}")
    ctx.add_trace("FILE_TASK_SUCCESS", action, {
        "path": rel_path, "absolute_path": abs_path,
        "content_chars": _content_len,
        "content_preview": '\n'.join(_preview_lines[:10]),
        "job_context_files": list(ctx.job_context.keys()),
    })


