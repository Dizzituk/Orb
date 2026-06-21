# FILE: app/llm/chat_tool_registry.py
# Purpose: Chat tool eligibility gate + tier-aware tool-declaration assembly.
# Called-by: app.llm.chat_tool_loop (re-export shim; importers resolve through it)
# Depends-on: app.debug.tool_definitions, app.debug.gemini_finance_tools, app.debug.gemini_expense_tools, app.tools.chat_bridge (+more, all lazy)
# Last-renovated: 2026-06-21
"""
Chat tool registry — split out of chat_tool_loop.py (SPLIT BATCH 9, 2026-06-21).

Single responsibility: decide which provider/model gets tools (is_tool_eligible
+ the _TRUSTED_MODELS_* sets), convert internal tool dicts to Anthropic
input_schema format (_to_anthropic_tool_format), and assemble the tier-aware
tool-declaration list (get_chat_tools). Code moved VERBATIM from chat_tool_loop.py;
that module re-exports every name here so all importers resolve unchanged.
"""
from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Models that get tool access in chat mode
# v1.2: Gemini models listed for future tool loop support.
# Currently only Anthropic models run the full tool loop (stream_anthropic
# accepts tools). Gemini models get codebase context via chat_codebase_reader
# pre-loading instead. When stream_gemini gains tool support, flip the
# provider gate below.
_TRUSTED_MODELS_ANTHROPIC = {
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
}

# Gemini models that WOULD get tool access once stream_gemini supports it.
# For now these use pre-loaded codebase context (chat_codebase_reader).
_TRUSTED_MODELS_GOOGLE = {
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools",
}

# v11.0: OpenAI models — now tool-eligible via _stream_with_tools_openai.
# v12.0 (2026-06-17): the GPT-5 family is tool-eligible by PREFIX (see
# is_tool_eligible), not just this set. The set stays for documentation; the
# prefix guard means an ENV model swap to any new gpt-5* id (e.g. gpt-5.5) can
# never silently strip tools on the phone/bridge or normal-chat paths (spec 3.2).
# NOTE: the Debug /stream/chat path does not gate on this -- it loads write-tier
# tools directly -- so this guard protects the OTHER OpenAI tool paths.
_TRUSTED_MODELS_OPENAI = {
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-turbo",
    "gpt-5.5",
    "gpt-5.5-mini",
}


def is_tool_eligible(provider: str, model: str) -> bool:
    """Check if this provider/model combo supports the tool execution loop.

    v1.3: Gemini 3.1 Pro customtools now supported via stream_gemini tools param.
    v11.0: OpenAI models now tool-eligible (tool loop + streaming delta accumulation).
    """
    if provider == "anthropic":
        return model in _TRUSTED_MODELS_ANTHROPIC
    if provider in ("google", "gemini"):
        return model in _TRUSTED_MODELS_GOOGLE
    if provider == "openai":
        # Prefix guard: any gpt-5* OpenAI model is tool-eligible, so an ENV model
        # change can never silently disable tools (spec 3.2 offender #3).
        return model in _TRUSTED_MODELS_OPENAI or str(model).lower().startswith("gpt-5")
    return False


def _to_anthropic_tool_format(tool: Dict) -> Dict:
    """Convert internal tool format to Anthropic API format.

    Internal:  {"name", "description", "parameters": {"type": "object", "properties": ...}}
    Anthropic: {"name", "description", "input_schema": {"type": "object", "properties": ...}}
    """
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool.get("parameters", tool.get("input_schema", {"type": "object", "properties": {}})),
    }


# Tool tiers for different contexts
TOOL_TIER_READ = "read"       # Chat mode: read_file, list_files, search_files, read_logs, read_pipeline_state
TOOL_TIER_WRITE = "write"     # Debug/agentic: adds write_file, edit_file, run_command


def get_chat_tools(tier: str = TOOL_TIER_READ) -> List[Dict]:
    """Get tool definitions in Anthropic API format for the given tier.

    Tiers:
        read  — Chat mode. Can explore codebase but not modify anything.
        write — Debug/agentic. Full access including writes and commands.

    Returns tools in Anthropic API format (input_schema, not parameters).
    """
    from app.debug.tool_definitions import get_phase1_tools, get_phase2_tools

    if tier == TOOL_TIER_WRITE:
        raw_tools = get_phase2_tools()
    else:
        raw_tools = get_phase1_tools()

    tools = [_to_anthropic_tool_format(t) for t in raw_tools]
    # Work-day ledger tools - let any chat start/finish/reconcile a delivery
    # day straight to the DB (the 'Work' tab), via the same declarations the
    # image loop uses. Without these, "starting work, 11, 1500 miles" has no
    # tool to call and the model wrongly reaches for write_file / edit_file.
    try:
        from app.debug.gemini_finance_tools import FINANCE_TOOL_DECLARATIONS
        tools.extend(_to_anthropic_tool_format(t) for t in FINANCE_TOOL_DECLARATIONS)
    except Exception as exc:
        logger.warning("[chat_tool_loop] finance tools unavailable: %s", exc)
    # Expense logging tool ("log £120 fuel") - its own module, same wiring.
    try:
        from app.debug.gemini_expense_tools import EXPENSE_TOOL_DECLARATIONS
        tools.extend(_to_anthropic_tool_format(t) for t in EXPENSE_TOOL_DECLARATIONS)
    except Exception as exc:
        logger.warning("[chat_tool_loop] expense tools unavailable: %s", exc)
    # Central-registry tools (2026-06-12): the Session-7 families (documents/
    # editor, podcasts, email, streams, movie night, displays) plus sentinel/
    # vehicle register in app.tools.registry, which this loop never read.
    # Without this merge the chat model has no editor access and answers
    # "upload it here" while the document sits open in the pane.
    try:
        from app.tools.chat_bridge import get_registry_chat_declarations
        _existing = {t.get("name") for t in tools}
        tools.extend(get_registry_chat_declarations(exclude_names=_existing))
    except Exception as exc:
        logger.warning("[chat_tool_loop] registry tools unavailable: %s", exc)
    # Debug orchestrator: ONLY the write tier (the Debug-tab brain) gets the
    # spawn_agents delegation tool, so ordinary chat / sub-agents never see it.
    if tier == TOOL_TIER_WRITE:
        try:
            from app.debug.orchestrator.spawn_tool import SPAWN_AGENTS_TOOL
            tools.append(_to_anthropic_tool_format(SPAWN_AGENTS_TOOL))
        except Exception as exc:
            logger.warning("[chat_tool_loop] spawn_agents tool unavailable: %s", exc)
        # Sandbox boot/self-heal: start/stop/status the clone + read its boot log.
        # These act on the SANDBOX CLONE only (never the host) -- the host-boot gate
        # in filesystem.py refuses host boots, so this is the sanctioned boot path.
        try:
            from app.sandbox.tools import SANDBOX_TOOLS
            from app.debug.sandbox_boot_tool import (
                READ_SANDBOX_BOOT_TOOL, INSPECT_SANDBOX_BOOT_TOOL,
            )
            from app.debug.sandbox_selfheal import SELFHEAL_SANDBOX_BOOT_TOOL
            for _sbx_tool in SANDBOX_TOOLS:
                tools.append(_to_anthropic_tool_format(_sbx_tool))
            tools.append(_to_anthropic_tool_format(READ_SANDBOX_BOOT_TOOL))
            # inspect_sandbox_boot = the FULL visual loop (backend + screenshot/OCR).
            tools.append(_to_anthropic_tool_format(INSPECT_SANDBOX_BOOT_TOOL))
            # selfheal_sandbox_boot = autonomous fix->reboot->re-verify loop (Job 03).
            tools.append(_to_anthropic_tool_format(SELFHEAL_SANDBOX_BOOT_TOOL))
        except Exception as exc:
            logger.warning("[chat_tool_loop] sandbox tools unavailable: %s", exc)
    # Phase-level narration (debug-visibility spec): attach the required `narration`
    # object to the phase-dispatch tools (spawn_agents + sandbox boot/inspect) so the
    # brain must declare its intent (and reflect) at the move boundary. Leaf tools are
    # untouched. Done here -- one place -- so the protected sandbox schemas stay intact.
    try:
        from app.debug.orchestrator.phase_narration import inject_narration_schema
        tools = [inject_narration_schema(t) for t in tools]
    except Exception as exc:
        logger.warning("[chat_tool_loop] narration schema inject unavailable: %s", exc)
    return tools
