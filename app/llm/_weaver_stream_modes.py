# FILE: app/llm/_weaver_stream_modes.py
# Purpose: Weaver stream: mode-specific prompt building.
# Called-by: app.llm.weaver_stream
# Depends-on: app.llm._weaver_prompts, app.llm._weaver_stream_utils_12, app.llm._weaver_stream_utils_16, app.llm.weaver_rules_engine (+1 more)
# Last-renovated: 2026-06-11
"""
Weaver stream: mode-specific prompt building.

Builds (system_prompt, user_prompt, start_message) for each weave mode:
micro-task, refactor, update, and create.
Extracted from weaver_stream.py for modularity.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.llm._weaver_stream_utils_12 import _format_ramble
from app.llm._weaver_stream_utils_16 import MICRO_TASK_SYSTEM_PROMPT, REFACTOR_TASK_SYSTEM_PROMPT
from app.llm._weaver_prompts import WEAVER_UPDATE_SYSTEM_PROMPT, WEAVER_CREATE_SYSTEM_PROMPT


def build_weave_prompt(
    *,
    is_micro_task: bool,
    is_refactor_task: bool,
    is_update_mode: bool,
    ramble_text: str,
    blocking_questions: List[str],
    new_user_messages: List[Dict],
    checkpoint: Optional[Dict],
    confirmed_prefs: Dict,
    execution_mode: str,
    total_message_count: int,
) -> Tuple[str, str, str]:
    """Return (system_prompt, user_prompt, start_message) for the current mode."""

    prefs_context = _build_prefs_context(confirmed_prefs)
    exec_context = f"\n\nExecution mode (extracted from meta-phrases): {execution_mode}" if execution_mode else ""

    if is_micro_task:
        return _build_micro_prompt(ramble_text, blocking_questions)
    elif is_refactor_task:
        return _build_refactor_prompt(ramble_text)
    elif is_update_mode:
        return _build_update_prompt(
            new_user_messages, checkpoint, prefs_context, exec_context,
        )
    else:
        return _build_create_prompt(
            ramble_text, total_message_count, prefs_context, exec_context,
        )


def _build_prefs_context(confirmed_prefs: Dict) -> str:
    if not confirmed_prefs:
        return ""
    lines = [f"- {k.title()}: {v}" for k, v in confirmed_prefs.items()]
    return "\n\nUser's confirmed design preferences:\n" + "\n".join(lines)


def _build_micro_prompt(ramble_text: str, blocking_questions: List[str]) -> Tuple[str, str, str]:
    blocker = ""
    if blocking_questions:
        blocker = "\n\nBLOCKING QUESTIONS (must include in output):\n" + "\n".join(f"- {q}" for q in blocking_questions)

    return (
        MICRO_TASK_SYSTEM_PROMPT,
        f"User request:\n\n{ramble_text}{blocker}\n\nProduce the minimal job outline:",
        "**Quick task detected...**\n\n",
    )


def _build_refactor_prompt(ramble_text: str) -> Tuple[str, str, str]:
    return (
        REFACTOR_TASK_SYSTEM_PROMPT,
        f"User request:\n\n{ramble_text}\n\nProduce the refactor job outline:",
        "**Refactor/rename task detected...**\n\n",
    )


def _build_update_prompt(
    new_user_messages: List[Dict],
    checkpoint: Optional[Dict],
    prefs_context: str,
    exec_context: str,
) -> Tuple[str, str, str]:
    # Handle compaction for long update message lists
    new_ramble = _format_ramble(new_user_messages)
    previous_output = checkpoint["last_output"] if checkpoint else ""

    # v1.1 (Job 8): Pre-classify new content
    preclassified_block = ""
    try:
        from app.llm.weaver_rules_engine import classify_conversation
        from app.llm.weaver_rules_inject import format_preclassified_block
        _classified = classify_conversation(new_ramble)
        preclassified_block = format_preclassified_block(_classified)
        if preclassified_block:
            preclassified_block = f"\n\n{preclassified_block}\n"
    except Exception:
        pass

    user_prompt = (
        f"Previous job description:\n\n{previous_output}\n\n"
        f"New requirements from user (extract and add EVERY feature):\n\n"
        f"{new_ramble}\n{prefs_context}{exec_context}"
        f"{preclassified_block}\n\n"
        f"Output the complete updated job description with all new features added:"
    )
    start_msg = (
        f"**Updating your job description...**\n\n"
        f"Incorporating {len(new_user_messages)} new requirement(s) from you.\n\n"
    )
    return WEAVER_UPDATE_SYSTEM_PROMPT, user_prompt, start_msg


def _build_create_prompt(
    ramble_text: str,
    total_message_count: int,
    prefs_context: str,
    exec_context: str,
) -> Tuple[str, str, str]:
    # v1.1 (Job 8): Pre-classify conversation content
    preclassified_block = ""
    try:
        from app.llm.weaver_rules_engine import classify_conversation
        from app.llm.weaver_rules_inject import format_preclassified_block
        _classified = classify_conversation(ramble_text)
        preclassified_block = format_preclassified_block(_classified)
        if preclassified_block:
            preclassified_block = f"\n\n{preclassified_block}\n"
    except Exception:
        pass

    user_prompt = (
        f"Organize this conversation into a job description:\n\n"
        f"{ramble_text}{prefs_context}{exec_context}"
        f"{preclassified_block}\n\n"
        f"Remember:\n"
        f"- Include ALL requirements the user stated (don't drop anything)\n"
        f"- Preserve any ambiguities (list them, don't resolve them)\n"
        f"- Keep What and Outcome DIFFERENT (no duplication)\n"
        f"- Code-answerable gaps go in \"SpecGate must resolve\" (NOT questions for user)\n"
        f"- Only put genuinely subjective/preference questions in \"Questions for user\"\n"
        f"- When in doubt, it's a SpecGate directive, not a user question\n"
        f"- Preserve the user's domain terminology\n"
        f"- If pre-classified items are provided above, use them as your starting point"
    )
    start_msg = (
        f"**Organizing your thoughts...**\n\n"
        f"Analyzing {total_message_count} messages to create a job description.\n\n"
    )
    return WEAVER_CREATE_SYSTEM_PROMPT, user_prompt, start_msg
