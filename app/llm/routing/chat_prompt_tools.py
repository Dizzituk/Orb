# FILE: app/llm/routing/chat_prompt_tools.py
# Purpose: Prompt/tool decoration helpers (tab data, grounding gate, tool injection) for chat routing (split from chat_routing.py).
# Called-by: app.llm.routing.chat_routing
# Depends-on: app.grounding.chat_integration, app.llm.routing.chat_intent_detection, app.llm.routing.chat_model_selection (+ lazy more)
# Last-renovated: 2026-06-21
from __future__ import annotations
import logging
from typing import Any
from .chat_intent_detection import (
    detect_codebase_exploration as _detect_codebase_exploration,
    is_builds_context as _is_builds_context,
)
from .chat_model_selection import set_sticky_model as _set_sticky_model

# Grounding Gate (v1.1)
try:
    from app.grounding.chat_integration import run_grounding_sync
    _GROUNDING_AVAILABLE = True
except ImportError:
    _GROUNDING_AVAILABLE = False
    run_grounding_sync = None
    logging.warning("[chat_routing] Grounding gate not available")

logger = logging.getLogger(__name__)


def _inject_tab_data(ui_ctx: Any, full_context: str, db: Session) -> str:
    """Inject live tab data (e.g. portfolio positions) + editor-pane state."""
    # Editor pane (2026-06-12): unconditional — the user may be on any tab
    # while a document sits open in the editor pane next to the chat.
    try:
        from app.llm.routing.ui_context_data import fetch_editor_state_block
        _editor_block = fetch_editor_state_block()
        if _editor_block:
            full_context += f"\n\n{_editor_block}"
            print(f"[CHAT_MODE] Editor state injected: {len(_editor_block)} chars")
    except Exception as e:
        print(f"[CHAT_MODE] Editor state injection failed: {e}")
    if not (ui_ctx and getattr(ui_ctx, 'job_type', None)):
        return full_context
    try:
        from app.llm.routing.ui_context_data import fetch_tab_data
        tab_data = fetch_tab_data(ui_ctx.job_type, db)
        if tab_data:
            full_context += f"\n\n{tab_data}"
            print(f"[CHAT_MODE] Tab data injected for {ui_ctx.job_type}: {len(tab_data)} chars")
    except Exception as e:
        print(f"[CHAT_MODE] Tab data injection failed: {e}")
    return full_context


def _run_grounding_gate(req: Any, system_prompt: str, label: str = "CHAT_MODE") -> str:
    """Run the grounding gate if available.  Returns the (possibly modified) system prompt."""
    if not (_GROUNDING_AVAILABLE and run_grounding_sync is not None):
        return system_prompt
    try:
        system_prompt, _grounding_meta = run_grounding_sync(
            message=req.message,
            system_prompt=system_prompt,
            context={"user_id": getattr(req, "user_id", "default")},
        )
        if _grounding_meta.get("grounding_applied"):
            print(
                f"[{label}] Grounding gate ACTIVE: "
                f"category={_grounding_meta.get('category')}, "
                f"sources={_grounding_meta.get('source_count')}, "
                f"domain={_grounding_meta.get('domain_hint')}"
            )
        else:
            print(
                f"[{label}] Grounding gate: no grounding needed "
                f"(category={_grounding_meta.get('category', 'n/a')}, "
                f"reason={_grounding_meta.get('reason', 'personal')})"
            )
    except Exception as e:
        print(f"[{label}] Grounding gate error (non-fatal): {e}")
    return system_prompt


# Tool role prompt blocks (shared between chat and normal routing)
_TOOL_ROLE_BLOCK = (
    "\n\n## TOOL ACCESS\n"
    "You have tool access for exploring the codebase AND writing to user folders.\n\n"
    "CODEBASE TOOLS (read-only): read_file, list_files, search_files, read_logs, search_my_files, read_user_file\n"
    "USER FILE TOOLS (read+write): get_user_folders, write_user_file\n"
    "Use get_user_folders to discover real folder paths, then write_user_file to save files there.\n\n"
    "IMPORTANT: Actually USE the tools. Do not just say you will — call them.\n\n"
    "YOUR ROLE: You are a RESEARCHER, PLANNER, and ASSISTANT.\n"
    "- Explore files, read code, understand patterns, discover architecture\n"
    "- Report your findings as text in the chat — describe what you found\n"
    "- When the user asks you to create a file in their personal folders\n"
    "  (Documents, Pictures, Desktop, etc.), call get_user_folders then write_user_file.\n"
    "- When asked to plan or spec, USE tools first to inspect the codebase,\n"
    "  then produce a detailed implementation plan based on real file contents\n"
    "- Present file paths, structures, and what needs to change\n\n"
    "DO NOT:\n"
    "- Try to create, write, or modify ASTRA codebase files — those go through the sandbox\n"
    "- Dump raw file contents — summarise and highlight relevant patterns\n"
    "- Say you will do something without actually calling the tools to do it\n\n"
    "GOOD OUTPUT: Call tools to explore, then present findings and plans.\n"
    "BAD OUTPUT: Saying 'I will inspect...' without calling any tools.\n"
)

_WEB_SEARCH_PROMPT = (
    "\n\n## WEB SEARCH TOOL\n"
    "You have access to a web_search tool. Use it when the user asks you to\n"
    "research, look up, find pricing, get current information, or anything\n"
    "that requires knowledge you do not have.\n"
    "IMPORTANT: Actually CALL the web_search tool. Do not just say you will.\n"
    "Call it with a specific search query and use the results in your response.\n"
)

_CHAT_TOOLS_OVERRIDE = (
    "   - You CAN: read files, write files, execute code, explore directories\n"
)
_CHAT_TOOLS_REPLACEMENT = (
    "   - Codebase files have been PRE-LOADED into your context below.\n"
    "   - You do NOT have codebase tool access. Do NOT generate tool_call blocks for file operations.\n"
    "   - Do NOT call execute_command or shell commands.\n"
    "   - Reference the [CODEBASE CONTEXT] files directly in your response.\n"
)


def _inject_tools(
    provider: str,
    model: str,
    req: Any,
    system_prompt: str,
    codebase_gather_pending: bool = False,
    codebase_ctx: str = "",
) -> tuple[list | None, str]:
    """Inject tool definitions + prompt blocks.  Returns (tools_list, updated_system_prompt)."""
    _chat_tools = None

    try:
        from app.llm.chat_tool_loop import is_tool_eligible, get_chat_tools

        # If model lacks tool access but context needs it, swap to tool-capable model
        if not is_tool_eligible(provider, model):
            from app.memory.complexity import DEEP_KEYWORDS, _count_keyword_hits
            _deep_hits = _count_keyword_hits(req.message.lower(), DEEP_KEYWORDS)
            _needs_tools = (
                _is_builds_context(req)
                or _detect_codebase_exploration(req.message)
                or _deep_hits >= 1
            )
            if _needs_tools:
                import os as _os
                _tool_provider = _os.getenv("TOOL_CHAT_PROVIDER", "google")
                _tool_model = _os.getenv("TOOL_CHAT_MODEL")
                if not _tool_model:
                    try:
                        from app.llm.frontier_models import get_role_model as _get_role_model
                        _tool_model = _get_role_model("MULTIMODAL")[1]
                    except Exception:
                        _tool_model = ""
                if is_tool_eligible(_tool_provider, _tool_model):
                    print(f"[TOOLS] Context needs tools but {provider}/{model} has none — "
                          f"swapping to {_tool_provider}/{_tool_model}")
                    provider = _tool_provider
                    model = _tool_model
                    _set_sticky_model(req.project_id, provider, model)

        if is_tool_eligible(provider, model):
            _chat_tools = get_chat_tools()
            print(f"[TOOLS] Tool access ENABLED for {provider}/{model} ({len(_chat_tools)} tools)")
            system_prompt += _TOOL_ROLE_BLOCK
        else:
            if codebase_gather_pending and codebase_ctx:
                if _CHAT_TOOLS_OVERRIDE in system_prompt:
                    system_prompt = system_prompt.replace(_CHAT_TOOLS_OVERRIDE, _CHAT_TOOLS_REPLACEMENT)
                    print("[TOOLS] Capability layer overridden for non-trusted model")
    except ImportError:
        print("[TOOLS] chat_tool_loop not available")

    # Universal web search — ALL models get web_search as a tool
    try:
        from app.debug.tool_definitions import get_universal_tools
        from app.llm.chat_tool_loop import _to_anthropic_tool_format
        _universal = [_to_anthropic_tool_format(t) for t in get_universal_tools()]
        if _chat_tools is not None:
            _existing_names = {t.get("name") for t in _chat_tools}
            for ut in _universal:
                if ut["name"] not in _existing_names:
                    _chat_tools.append(ut)
            print(f"[TOOLS] Universal tools merged: {len(_chat_tools)} total")
        else:
            _chat_tools = _universal
            system_prompt += _WEB_SEARCH_PROMPT
            print(f"[TOOLS] Universal web_search tool injected for {provider}/{model}")
    except ImportError as _uie:
        print(f"[TOOLS] Universal tools not available: {_uie}")

    return _chat_tools, system_prompt
