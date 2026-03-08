# FILE: app/llm/_weaver_stream_prepare.py
"""
Weaver stream: Steps 1-4 message gathering, dedup, and state loading.

Extracts and prepares conversation messages for weaving.
Returns a WeaverPrepResult with all data needed by Steps 5+.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from sqlalchemy.orm import Session

from app.llm._weaver_stream_utils_12 import _format_ramble
from app.llm._weaver_stream_utils_13 import _hash_messages
from app.llm._weaver_stream_utils_14 import _hash_message, _normalize_typos
from app.llm._weaver_stream_utils_15 import _extract_meta_mode, _extract_vision_context, _gather_ramble_messages
from app.llm._weaver_stream_utils_16 import _is_vision_context
from app.llm._weaver_substantive_filter import _is_substantive_assistant_content

logger = logging.getLogger(__name__)


@dataclass
class WeaverPrepResult:
    """All state produced by Steps 1-4."""
    all_messages: List[Dict] = field(default_factory=list)
    filtered_messages: List[Dict] = field(default_factory=list)
    relevant_messages: List[Dict] = field(default_factory=list)
    new_user_messages: List[Dict] = field(default_factory=list)
    ramble_text: str = ""
    vision_context: str = ""
    confirmed_prefs: Dict = field(default_factory=dict)
    current_user_hashes: Set[str] = field(default_factory=set)
    checkpoint: Optional[Dict] = None
    is_update_mode: bool = False
    total_message_count: int = 0
    execution_mode: str = ""
    # Sentinel: if set, yield this message and return early
    early_exit_message: str = ""


def prepare_weaver_messages(
    db: Session,
    project_id: int,
    pending_user_message: Optional[str],
    captured_answers: Optional[Dict[str, str]],
) -> WeaverPrepResult:
    """Execute Steps 1-4: gather, filter, dedup, load state."""
    result = WeaverPrepResult()

    # Import flow state functions
    try:
        from app.llm.spec_flow_state import (
            save_confirmed_design_prefs,
            get_confirmed_design_prefs,
            get_weave_checkpoint,
            get_woven_user_hashes,
        )
        _flow = True
    except ImportError:
        _flow = False
        get_confirmed_design_prefs = None
        get_weave_checkpoint = None
        get_woven_user_hashes = None
        save_confirmed_design_prefs = None

    # STEP 1: Gather all messages
    all_messages = _gather_ramble_messages(db, project_id)

    # v4.1.0: Inject pending user message
    if pending_user_message and pending_user_message.strip():
        pending_msg = {"role": "user", "content": pending_user_message.strip()}
        pending_hash = _hash_message(pending_msg)
        existing_hashes = _hash_messages(all_messages)
        if pending_hash not in existing_hashes:
            all_messages.append(pending_msg)

    if not all_messages:
        result.early_exit_message = (
            "**No conversation to weave**\n\n"
            "I don't see any recent messages to organize into a job description.\n\n"
            "Share what you want to build or change, then say "
            "`how does that look all together` again."
        )
        return result

    result.all_messages = all_messages
    result.total_message_count = len(all_messages)

    # STEP 2: Extract meta-mode phrases
    filtered_messages, extracted_modes = _extract_meta_mode(all_messages)
    from app.llm._weaver_stream_utils_12 import _format_execution_mode
    result.execution_mode = _format_execution_mode(extracted_modes)

    # STEP 2b: Typo normalization
    for i, msg in enumerate(filtered_messages):
        if msg.get("role") == "user" and msg.get("content"):
            normalized = _normalize_typos(msg["content"])
            if normalized != msg["content"]:
                filtered_messages[i] = {**msg, "content": normalized}

    result.filtered_messages = filtered_messages

    # STEP 3: Load prefs, hashes, checkpoint
    confirmed_prefs = {}
    if _flow and get_confirmed_design_prefs:
        confirmed_prefs = get_confirmed_design_prefs(project_id)

    if captured_answers:
        confirmed_prefs.update(captured_answers)
        if _flow and save_confirmed_design_prefs:
            save_confirmed_design_prefs(project_id, captured_answers)

    result.confirmed_prefs = confirmed_prefs

    woven_hashes: Set[str] = set()
    if _flow and get_woven_user_hashes:
        woven_hashes = get_woven_user_hashes(project_id)

    checkpoint = None
    if _flow and get_weave_checkpoint:
        checkpoint = get_weave_checkpoint(project_id)
    result.checkpoint = checkpoint

    # STEP 4: Hash-based dedup
    vision_context = _extract_vision_context(filtered_messages)
    result.vision_context = vision_context

    # v4.3: Include assistant messages with substantive technical content,
    # not just vision context.  This captures rich responses from video+tool
    # pipelines (Gemini codebase analysis, extracted specs, CSS patterns)
    # that the Weaver needs to synthesise into the job description.
    relevant_messages = [
        m for m in filtered_messages
        if m.get("role") == "user"
        or (
            m.get("role") == "assistant"
            and (
                _is_vision_context(m.get("content", ""))
                or _is_substantive_assistant_content(m.get("content", ""))
            )
        )
    ]
    result.relevant_messages = relevant_messages

    user_messages_only = [m for m in filtered_messages if m.get("role") == "user"]
    current_user_hashes = _hash_messages(user_messages_only)
    result.current_user_hashes = current_user_hashes

    new_user_hashes = current_user_hashes - woven_hashes
    result.new_user_messages = [
        m for m in user_messages_only if _hash_message(m) in new_user_hashes
    ]

    is_update_mode = bool(woven_hashes) and checkpoint is not None and checkpoint.get("last_output")
    result.is_update_mode = is_update_mode

    if is_update_mode and not result.new_user_messages:
        result.early_exit_message = (
            "**Nothing new to weave**\n\n"
            "I don't see any new requirements from you since the last weave.\n\n"
            "Add more details to your conversation, then say "
            "`how does that look all together` again."
        )
        return result

    # Format ramble text (with optional compaction — caller handles async)
    result.ramble_text = _format_ramble(relevant_messages)

    return result
