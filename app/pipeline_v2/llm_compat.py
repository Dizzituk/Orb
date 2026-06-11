# FILE: app/pipeline_v2/llm_compat.py
"""
JOB 16 (2026-06-10) - legacy-signature compatibility adapters.

app/optimize/boot_recovery.py imports run_agentic_loop - an API from an
older design that never existed in llm_tools, so smart boot-repair was
silently dying on ImportError inside its try/except every time it fired.
This module provides that legacy signature as a thin adapter onto the real
run_tool_loop, in its own file so llm_tools stays at single responsibility
(and under the size ceiling).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


async def run_agentic_loop(
    prompt: str,
    root_path: str = "",
    language: str = "python",
    model: str = "openai/gpt-5.4",
    max_tool_calls: int = 20,
    emit: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Legacy boot-recovery entrypoint.

    Args mirror the old design: model is "provider/model_id". Returns
    {"files_written": int, "text": str} - the shape boot_recovery expects.
    """
    from app.pipeline_v2.llm_tools import run_tool_loop

    _emit = emit or (lambda m: None)
    provider, _, model_id = model.partition("/")
    if not model_id:
        provider, model_id = "openai", provider

    files_written = 0

    def _on_tool(name: str, args: Dict) -> None:
        nonlocal files_written
        if name in ("write_file", "edit_file"):
            files_written += 1
        _emit(f"   [repair] {name}: {str(args.get('path', args.get('cmd', '')))[:80]}")

    texts: List[str] = []
    system = (
        f"You are a surgical repair agent for a {language} codebase rooted at {root_path or 'the project root'}. "
        "Diagnose the reported failure, fix it with the smallest possible edits "
        "(prefer edit_file with exact unique anchors over rewriting files), "
        "verify with run_shell, then summarise what you changed."
    )
    try:
        await run_tool_loop(
            system_prompt=system,
            initial_user_message=prompt,
            provider=provider or "openai",
            model=model_id,
            max_iterations=max_tool_calls,
            max_tokens=16384,
            on_tool_call=_on_tool,
            on_text=lambda t: texts.append(t),
        )
    except Exception as exc:
        logger.warning("[llm_compat] run_agentic_loop failed: %s", exc)
        return {"files_written": files_written, "text": f"repair loop error: {exc}"}

    return {"files_written": files_written, "text": "\n".join(texts)[-4000:]}
