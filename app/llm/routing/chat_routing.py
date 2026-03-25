# FILE: app/llm/routing/chat_routing.py
"""
Chat and normal routing handlers for stream routing.

v1.1 (2026-03-03): Integrated Grounding Gate — claims-dependent and contested
    queries now force a web search before the LLM responds.
v1.0 (2026-01-20): Extracted from stream_router.py for modularity.

This module provides:
- `handle_chat_mode()` - Lightweight chat routing (now with grounding gate)
- `handle_normal_routing()` - Standard job-type routing (now with grounding gate)
- `handle_legacy_triggers()` - Fallback for when translation layer unavailable
"""

from __future__ import annotations

import logging
from typing import List, Optional, Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.memory import service as memory_service
# Import from pipeline.high_stakes to avoid circular import with router.py
from app.llm.pipeline.high_stakes import is_high_stakes_job, is_opus_model
from app.llm.streaming import get_available_streaming_provider, get_available_streaming_providers

from app.llm.stream_utils import (
    DEFAULT_MODELS,
    classify_job_type,
    select_provider_for_job_type,
)

from app.memory.complexity import classify_complexity

from app.llm.legacy_triggers import (
    is_zobie_map_trigger,
    is_archmap_trigger,
    is_update_arch_trigger,
    is_introspection_trigger,
    is_sandbox_trigger,
)

from .handler_registry import (
    # Availability flags
    _LOCAL_TOOLS_AVAILABLE,
    _SANDBOX_AVAILABLE,
    _INTROSPECTION_AVAILABLE,
    _RAG_STREAM_AVAILABLE,
    # Handlers
    generate_sse_stream,
    generate_sandbox_stream,
    generate_introspection_stream,
    generate_local_architecture_map_stream,
    generate_local_zobie_map_stream,
    generate_update_architecture_stream,
    generate_rag_query_stream,
    generate_high_stakes_critique_stream,
)

from .prompt_builders import (
    build_system_prompt,
    build_messages,
    build_full_context,
)

from .rag_fallback import is_architecture_query

# v1.1: Grounding Gate — forces web search for claims-dependent queries
try:
    from app.grounding.chat_integration import run_grounding_sync
    _GROUNDING_AVAILABLE = True
except ImportError:
    _GROUNDING_AVAILABLE = False
    run_grounding_sync = None
    logging.warning("[chat_routing] Grounding gate not available")

logger = logging.getLogger(__name__)

# v10.0: Session-level model stickiness.
# Once a conversation upgrades to a more capable model (e.g. Gemini 3.1 Pro),
# hold that model for subsequent messages in the same project session.
# Prevents downgrade to Flash on follow-up messages classified as simpler.
# Cleared when the app restarts or a new project is selected.
_session_model_cache: dict[int, tuple[str, str]] = {}  # project_id -> (provider, model)

def _get_sticky_model(project_id: int) -> tuple[str, str] | None:
    return _session_model_cache.get(project_id)

def _set_sticky_model(project_id: int, provider: str, model: str) -> None:
    _session_model_cache[project_id] = (provider, model)

import re as _re

# v10.1: Detect file creation intent — routes to GPT-5.4 for HTML/code generation.
_FILE_CREATION_PATTERNS = _re.compile(
    r'(?:create|build|make|generate|write|design|return|produce|come\s+up\s+with|put\s+together)\s+'
    r'(?:me\s+)?(?:a\s+|an\s+|the\s+)?'
    r'(?:html|webpage|web\s*page|website|landing\s*page|page|file|document)',
    _re.IGNORECASE,
)

# v11.0: Detect builds context — user is on Builds tab or talking about app building
_BUILDS_KEYWORDS = _re.compile(
    r'\b(add|modify|update|extend|change|build|create|implement|work\s+on)\b'
    r'.*\b(app|copilot|co-?pilot|android|driver|feature|element|screen|module|component)\b',
    _re.IGNORECASE,
)

# v11.3: Detect APK build + deploy to cloud intent (TIGHTENED)
#
# v11.1 patterns were too broad — "app" + "phone" anywhere in a sentence
# would trigger a build. Now requires:
#   1. Imperative/request language ("build me", "make an", "create the", etc.)
#      OR the "Astra, command:" prefix
#   2. The word "apk" specifically (not generic "app")
#   3. Excludes questions ("how do", "can we", "what if", "why does", etc.)
#
# Examples that SHOULD trigger:
#   "Build me an APK and upload to cloud"
#   "Make an APK and put it on the drive"
#   "Astra, command: create APK"
#   "Deploy the APK to the phone"
#
# Examples that should NOT trigger:
#   "How do we compile an APK?"
#   "I want to make the app feel more responsive on the phone"
#   "Can you explain how APK builds work?"
#   "The bridge app connects to the phone via cloud"

# Question patterns — if the message looks like a question, it's not a request
_QUESTION_PATTERN = _re.compile(
    r'^\s*(?:how|what|why|when|where|which|can\s+(?:you|we)|could\s+(?:you|we)|'
    r'would\s+it|is\s+(?:it|there|this)|are\s+(?:there|these)|do\s+(?:we|you)|'
    r'does\s+(?:it|this|the)|should\s+(?:we|I)|explain|tell\s+me\s+about|'
    r'what\s+(?:is|are|does|if)|describe)\b',
    _re.IGNORECASE,
)

# Imperative request: verb + "apk" (with optional words between)
# Requires "apk" specifically — not "app", "bridge", etc.
_BUILD_DEPLOY_PATTERN = _re.compile(
    r'\b(build|create|make|compile|generate|assemble|deploy|upload|push|put|drop|send)\b'
    r'.{0,40}'  # Max 40 chars between verb and "apk" — prevents cross-sentence matches
    r'\bapk\b',
    _re.IGNORECASE,
)

# Reverse: "apk" then action — "the APK, upload it to cloud"
_BUILD_DEPLOY_PATTERN_ALT = _re.compile(
    r'\bapk\b'
    r'.{0,40}'
    r'\b(to\s+(?:the\s+)?(?:cloud|proton|drive|phone)|upload|deploy|install|push)\b',
    _re.IGNORECASE,
)

# Explicit "Astra, command:" prefix — always trust this as a genuine request
_ASTRA_CMD_BUILD = _re.compile(
    r'(?:astra|orb),?\s*(?:command:?)?\s*(?:build|create|make|compile|deploy|upload)\b.*\bapk\b',
    _re.IGNORECASE,
)

def _detect_build_deploy_intent(message: str) -> bool:
    """Check if the user is REQUESTING an APK build (not just discussing it).

    Returns True only for imperative requests containing 'apk'.
    Returns False for questions, general discussion, or sentences that
    just happen to mention build-related words near 'app' or 'phone'.
    """
    msg = message.strip()

    # Explicit Astra command — always trust
    if _ASTRA_CMD_BUILD.search(msg):
        return True

    # Questions are never build requests
    if _QUESTION_PATTERN.search(msg):
        return False

    # Check for imperative request patterns with "apk"
    return bool(_BUILD_DEPLOY_PATTERN.search(msg) or _BUILD_DEPLOY_PATTERN_ALT.search(msg))

def _is_builds_context(req: Any) -> bool:
    """Check if the user is in a builds context (tab + project intent)."""
    ui_ctx = getattr(req, 'ui_context', None)
    on_builds_tab = (
        ui_ctx is not None
        and getattr(ui_ctx, 'job_type', '') == 'project_builds'
    )
    has_builds_intent = bool(_BUILDS_KEYWORDS.search(req.message))
    return on_builds_tab and has_builds_intent


def _detect_file_creation_intent(message: str) -> bool:
    """Check if the user is asking for a file to be created."""
    return bool(_FILE_CREATION_PATTERNS.search(message))

# v12.0: Detect codebase exploration / planning intent — needs tool access
_CODEBASE_EXPLORE_PATTERNS = _re.compile(
    r'(?:'
    r'(?:inspect|examine|explore|look\s+at|have\s+a\s+look|read|review|scan|map|check)\s+'
    r'(?:\w+\s+){0,3}(?:codebase|code|source|app|project|architecture|files?|structure|tree)'
    r'|'
    r'(?:implementation|architecture|feature)\s*plan'
    r'|'
    r'(?:plan\s+(?:out|of\s+action)|spec\s+out|come\s+up\s+with\s+a\s+plan)'
    r'|'
    r'(?:what\s+(?:files?|code)\s+(?:exists?|is\s+there|do\s+we\s+have))'
    r'|'
    r'(?:current\s+(?:state|architecture|structure)\s+of)'
    r'|'
    r'(?:(?:every|each|all)\s+files?\b)'
    r')',
    _re.IGNORECASE,
)

def _detect_codebase_exploration(message: str) -> bool:
    """Check if the user wants ASTRA to explore/inspect a codebase.

    These requests need tool access to actually read files — without it
    the model will say 'I will inspect...' but never actually do it.
    """
    return bool(_CODEBASE_EXPLORE_PATTERNS.search(message))

# v10.3: Detect image generation intent — routes to image generation pipeline.
# v3.1: Added chart/graph/infographic/plot for data-driven image requests.
_IMAGE_GEN_PATTERNS = _re.compile(
    r'(?:create|draw|make|generate|design|paint|sketch|render|produce|build|compile|visuali[sz]e|need|plot|put\s+together)\s+'
    r'(?:me\s+|yourself\s+)?(?:a\s+|an\s+|the\s+|another\s+)?'
    r'(?:new\s+)?(?:image|picture|photo|illustration|avatar|icon|graphic|artwork|portrait|visual|banner|thumbnail|logo|cover|chart|graph|infographic|plot|diagram)',
    _re.IGNORECASE,
)

def _detect_image_gen_intent(message: str) -> bool:
    """Check if the user is asking for an image to be generated."""
    return bool(_IMAGE_GEN_PATTERNS.search(message))

# v10.4: Detect image refinement intent (user wants to modify a previous image)
_IMAGE_REFINE_PATTERNS = _re.compile(
    r'(?:change|modify|adjust|tweak|fix|redo|again\s+but|same\s+but|less\s+\w+|more\s+\w+|make\s+it|try\s+again|not\s+(?:quite|right)|too\s+\w+)',
    _re.IGNORECASE,
)

def _detect_image_refinement(message: str) -> bool:
    """Check if the user is asking to refine a previous image."""
    return bool(_IMAGE_REFINE_PATTERNS.search(message))

def _last_assistant_was_image(project_id: int, db) -> bool:
    """Check if the most recent assistant message was a Nano Banana image generation."""
    try:
        msgs = memory_service.get_messages(db, project_id, limit=3)
        for msg in reversed(msgs):
            if msg.role == 'assistant' and msg.model == 'nano-banana-2':
                return True
            if msg.role == 'assistant':
                return False  # Last assistant message wasn't an image
    except Exception:
        pass
    return False

# v10.2: Elevated models that should persist via history-based stickiness
_ELEVATED_MODELS = {'gpt-5.4', 'gpt-5.4-turbo', 'claude-opus-4-6', 'claude-opus-4-5'}

def _infer_sticky_from_history(project_id: int, db) -> tuple[str, str] | None:
    """Check last assistant message in history for an elevated model.
    Falls back when the in-memory cache is empty (e.g. after app restart)."""
    try:
        msgs = memory_service.get_messages(db, project_id, limit=5)
        for msg in reversed(msgs):
            if msg.role == 'assistant' and msg.model and msg.model in _ELEVATED_MODELS:
                prov = msg.provider or 'openai'
                return (prov, msg.model)
    except Exception:
        pass
    return None

# =============================================================================
# CHAT MODE HANDLER
# =============================================================================

def handle_chat_mode(
    req: Any,  # StreamRequest
    project: Any,  # Project model
    db: Session,
    trace: Any,
) -> StreamingResponse:
    """
    Handle CHAT mode - lightweight model, no commands.
    
    v4.8: Uses stage_models for provider/model selection with debug logging.
    
    Args:
        req: StreamRequest with project_id, message, etc.
        project: Project ORM object
        db: Database session
        trace: Audit trace
    
    Returns:
        StreamingResponse for chat
    """
    print(f"[CHAT_MODE] Handling chat for project={req.project_id}, message={req.message[:50]}...")
    
    # v6.1: Always persist the user's message BEFORE any confirmation gate.
    # Without this, if the confirmation gate fires, the message is never saved
    # and downstream handlers (Weaver) can't find the conversation.
    try:
        from app.memory import schemas as _mem_schemas
        memory_service.create_message(
            db,
            _mem_schemas.MessageCreate(
                project_id=req.project_id,
                role="user",
                content=req.message,
                provider="system",
            ),
        )
    except Exception as e:
        print(f"[CHAT_MODE] Failed to persist user message: {e}")
    
    # Build context
    full_context = build_full_context(db, req.project_id, req.message, req.use_semantic_search)
    # v0.14.0: Inject uploaded file content into context.
    # When a file is uploaded via the debug panel, the extracted text must be
    # available to the chat model. The background knowledge hook runs async and
    # may not be done yet, so we extract synchronously here.
    _file_local_path = getattr(req, "file_upload_local_path", None)
    _file_name = getattr(req, "file_upload_name", None)
    if _file_local_path and _file_name:
        try:
            from app.llm.file_analyzer import extract_text as _extract_text
            _file_text, _file_err = _extract_text(file_path=_file_local_path, filename=_file_name)
            if _file_text:
                _preview = _file_text[:50000]  # Cap at 50KB for context
                _upload_block = (
                    f"=== UPLOADED FILE: {_file_name} ===\n"
                    f"{_preview}\n"
                    f"=== END FILE ===\n\n"
                )
                full_context = _upload_block + (full_context or "")
                print(f"[CHAT_MODE] Injected uploaded file content: {_file_name} ({len(_file_text)} chars)")
            elif _file_err:
                print(f"[CHAT_MODE] File extraction failed for {_file_name}: {_file_err}")
        except Exception as _fex:
            print(f"[CHAT_MODE] File injection error: {_fex}")


    # v7.0: Pre-gather codebase context for trusted models.
    # Reads files from the sandbox (read-only) via RAG-guided discovery.
    # This gives Opus/Gemini 3.1 actual codebase knowledge in standard chat.
    try:
        from app.llm.routing.chat_codebase_reader import (
            gather_codebase_context,
        )
        # We don't know the final model yet (complexity may upgrade it),
        # so we store the message and gather later, after model selection.
        _codebase_gather_pending = True
    except ImportError:
        _codebase_gather_pending = False
    
    # v5.5: Frontend model override (from model switcher dropdown)
    if req.provider and req.model:
        provider = req.provider
        model = req.model
        print(f"[CHAT_MODE] Using frontend override: provider={provider}, model={model}")
        _set_sticky_model(req.project_id, provider, model)
    elif _is_builds_context(req):
        # v11.0: Builds tab + project modification intent → GPT-5.4
        import os as _os
        provider = _os.getenv("BUILD_CHAT_PROVIDER", "openai")
        model = _os.getenv("BUILD_CHAT_MODEL", "gpt-5.4")
        print(f"[CHAT_MODE] Builds context detected -> {provider}/{model}")
        _set_sticky_model(req.project_id, provider, model)
    elif _detect_file_creation_intent(req.message):
        # v10.1: File creation requests → GPT-5.4 with tools (best for HTML/code generation)
        import os as _os
        provider = _os.getenv("FILE_CREATION_PROVIDER", "openai")
        model = _os.getenv("FILE_CREATION_MODEL", "gpt-5.4")
        print(f"[CHAT_MODE] File creation detected -> {provider}/{model}")
        _set_sticky_model(req.project_id, provider, model)
    elif _detect_image_gen_intent(req.message) or (
        _detect_image_refinement(req.message)
        and _last_assistant_was_image(req.project_id, db)
    ):
        # v3.1: Image generation now routed through translation layer dispatch table.
        # The GENERATE_IMAGE intent handles GPT Image 1.5 (primary) + Nano Banana (fallback),
        # with optional web search + Plotly chart rendering for data-driven requests.
        # We return None here so stream_router falls through to the translation layer.
        print(f"[CHAT_MODE] Image generation detected -> routing to translation layer (GENERATE_IMAGE intent)")

        from app.llm.image_router import generate_image_stream

        return StreamingResponse(
            generate_image_stream(
                project_id=req.project_id,
                message=req.message,
                db=db,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    elif _get_sticky_model(req.project_id):
        # v10.0: Session model stickiness — but explicit model
        # requests OVERRIDE the sticky model.
        from app.memory.complexity import _detect_explicit_model_request
        _first_line = req.message.lower().strip().split("\n")[0]
        _explicit = (
            _detect_explicit_model_request(_first_line)
            or _detect_explicit_model_request(req.message.lower())
        )
        if _explicit:
            # User explicitly asked for a different model — override sticky
            provider = _explicit["provider"]
            model = _explicit["model"]
            _set_sticky_model(req.project_id, provider, model)
            print(f"[CHAT_MODE] EXPLICIT override of sticky: {provider}/{model}")
        else:
            provider, model = _get_sticky_model(req.project_id)
            print(f"[CHAT_MODE] Using sticky model from session: {provider}/{model}")
    elif _infer_sticky_from_history(req.project_id, db):
        # v10.2: History-based stickiness
        from app.memory.complexity import _detect_explicit_model_request
        _first_line = req.message.lower().strip().split("\n")[0]
        _explicit = (
            _detect_explicit_model_request(_first_line)
            or _detect_explicit_model_request(req.message.lower())
        )
        if _explicit:
            provider = _explicit["provider"]
            model = _explicit["model"]
            _set_sticky_model(req.project_id, provider, model)
            print(f"[CHAT_MODE] EXPLICIT override of history sticky: {provider}/{model}")
        else:
            provider, model = _infer_sticky_from_history(req.project_id, db)
            _set_sticky_model(req.project_id, provider, model)
            print(f"[CHAT_MODE] Restored sticky model from history: {provider}/{model}")
    else:
        # v5.7: Run complexity classifier to decide model tier.
        complexity = classify_complexity(
            query=req.message,
            intent=None,
            attachments=getattr(req, 'attachments', None),
        )
        print(f"[CHAT_MODE] Complexity: tier={complexity.tier}, target={complexity.model_target}, "
              f"confidence={complexity.confidence}, signals={complexity.signals}")
        
        import os as _os
        _chat_provider = _os.getenv("CHAT_PROVIDER", "openai")  # v2.3: default to OpenAI
        _chat_model = _os.getenv("CHAT_MODEL", "gpt-5-mini")  # v2.3: GPT-5-mini for chat

        _skip_confirm = getattr(req, 'ui_context', None) is not None

        # v13.0: Four-tier model selection based on task complexity
        # Tier 1: GPT-5.4-mini + tools — read and report (exploration only)
        # Tier 2: GPT-5.4 + tools    — think and create (reasoning + exploration)
        # Tier 3: Gemini 3.1 Pro      — see and understand (multimodal/media)
        # Tier 4: Claude Opus 4.6     — architect and plan (large-scale design)
        #
        # Tools are always available for any tool-eligible model.
        # The tier picks the brain, not the tools.

        _signals = complexity.signals
        _arch_hits = _signals.get("arch_hits", 0)
        _explore_hits = _signals.get("explore_hits", 0)
        _reasoning_hits = _signals.get("reasoning_hits", 0)

        if complexity.tier == "multimodal":
            # ── TIER 3: Multimodal — media attached, needs vision ──
            attachments = getattr(req, 'attachments', None) or []
            provider = "google"
            model = _os.getenv("MULTIMODAL_MODEL", "gemini-3.1-pro-preview-customtools")
            print(f"[CHAT_MODE] TIER 3 (multimodal): {len(attachments)} attachments -> {provider}/{model}")
            _set_sticky_model(req.project_id, provider, model)

        elif _arch_hits >= 2 or (_arch_hits >= 1 and _reasoning_hits >= 2):
            # ── TIER 4: Architecture — large-scale design, needs deepest reasoning ──
            provider = _os.getenv("ARCHITECT_PROVIDER", "anthropic")
            model = _os.getenv("ARCHITECT_MODEL", "claude-opus-4-6")
            print(f"[CHAT_MODE] TIER 4 (architecture): arch={_arch_hits}, reasoning={_reasoning_hits} -> {provider}/{model}")
            _set_sticky_model(req.project_id, provider, model)
            if not _skip_confirm:
                try:
                    from app.llm.routing.confirmation_gate import (
                        should_confirm_model_escalation,
                        format_confirmation_sse,
                    )
                    confirm_req = should_confirm_model_escalation(
                        from_tier="lookup", to_tier="architect",
                        confidence=complexity.confidence,
                        message=req.message,
                    )
                    if confirm_req:
                        async def _confirm_stream():
                            import json as _json
                            yield format_confirmation_sse(confirm_req)
                            yield f"data: {_json.dumps({'type': 'done', 'provider': 'local', 'model': 'confirmation_gate', 'total_length': 0})}\n\n"
                        return StreamingResponse(
                            _confirm_stream(),
                            media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                        )
                except ImportError:
                    pass
            else:
                print(f"[CHAT_MODE] Confirmation gate skipped (chat panel request)")

        elif complexity.tier == "deep" and _reasoning_hits >= 1:
            # ── TIER 2: Reasoning + exploration — needs thinking power ──
            provider = _os.getenv("REASONING_PROVIDER", "openai")
            model = _os.getenv("REASONING_MODEL", "gpt-5.4")
            print(f"[CHAT_MODE] TIER 2 (reasoning+tools): explore={_explore_hits}, reasoning={_reasoning_hits} -> {provider}/{model}")
            _set_sticky_model(req.project_id, provider, model)

        elif complexity.tier == "deep" and _explore_hits >= 1:
            # ── TIER 1: Exploration only — read and report, mini is enough ──
            provider = _os.getenv("CHAT_PROVIDER", "openai")
            model = _os.getenv("CHAT_MODEL", "gpt-5.4-mini")
            print(f"[CHAT_MODE] TIER 1 (exploration): explore={_explore_hits}, arch={_arch_hits} -> {provider}/{model}")

        elif complexity.tier == "reasoning":
            # ── TIER 2: Pure reasoning (no exploration keywords) ──
            provider = _os.getenv("REASONING_PROVIDER", "openai")
            model = _os.getenv("REASONING_MODEL", "gpt-5.4")
            print(f"[CHAT_MODE] TIER 2 (reasoning): reasoning={_reasoning_hits} -> {provider}/{model}")
            _set_sticky_model(req.project_id, provider, model)

        # ── Explicit model requests — no confirmation needed ──
        elif complexity.tier.startswith("explicit_"):
            from app.memory.complexity import EXPLICIT_MODEL_PATTERNS
            for _mk, _mc in EXPLICIT_MODEL_PATTERNS.items():
                if _mc["tier"] == complexity.tier:
                    provider = _mc["provider"]
                    model = _mc["model"]
                    break
            else:
                provider = _chat_provider
                model = _chat_model
            print(f"[CHAT_MODE] EXPLICIT model request: {provider}/{model}")
            _set_sticky_model(req.project_id, provider, model)

        else:
            # ── TIER 1: Default — lightweight chat, mini handles it ──
            provider = _chat_provider
            model = _chat_model
            print(f"[CHAT_MODE] TIER 1 (default {complexity.tier}): {provider}/{model}")
    
    # v3.2: Check provider availability — but DON'T silently swap provider
    # while keeping the original model (that causes openai+gemini mismatches).
    providers_available = get_available_streaming_providers()
    print(f"[CHAT_MODE] Provider availability: {providers_available}")
    
    if not providers_available.get(provider, False):
        # Provider key not available — try to find a working alternative
        available = get_available_streaming_provider()
        if available:
            print(f"[CHAT_MODE] Provider '{provider}' not available, falling back to {available} WITH its default model")
            provider = available
            # CRITICAL: Also switch the model to match the new provider.
            # Without this, we'd send e.g. 'gemini-2.5-flash' to the OpenAI API.
            import os as _os2
            if provider == "google":
                model = _os2.getenv("CHAT_MODEL", "gemini-2.5-flash")
            elif provider == "anthropic":
                model = _os2.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-sonnet-4-6")
            elif provider == "openai":
                model = _os2.getenv("OPENAI_DEFAULT_MODEL", "gpt-5-mini")  # v2.3
        else:
            print(f"[CHAT_MODE] WARNING: No providers available at all")
    
    # Build messages
    # v6.0: If chat panel sent its own conversation history, use that instead of DB history.
    # This prevents the panel from inheriting the main chat's conversation thread.
    panel_hist = getattr(req, 'panel_history', None)
    if panel_hist and isinstance(panel_hist, list) and len(panel_hist) > 0:
        # Use panel's local history — already [{role, content}] format
        messages = []
        for entry in panel_hist[-20:]:  # Cap at 20 messages
            role = entry.get('role', 'user')
            content = entry.get('content', '')
            if role in ('user', 'assistant') and content:
                messages.append({'role': role, 'content': content})
        messages.append({'role': 'user', 'content': req.message})
        print(f"[CHAT_MODE] Using panel history: {len(messages)-1} prior messages + current")
    else:
        messages = build_messages(
            message=req.message,
            project_id=req.project_id,
            db=db,
            include_history=req.include_history,
            history_limit=req.history_limit,
        )
    
    # Build system prompt (includes capability layer + UI context)
    ui_ctx = getattr(req, 'ui_context', None)
    
    # v6.0: Inject live tab data (e.g. portfolio positions) into context
    if ui_ctx and getattr(ui_ctx, 'job_type', None):
        try:
            from app.llm.routing.ui_context_data import fetch_tab_data
            tab_data = fetch_tab_data(ui_ctx.job_type, db)
            if tab_data:
                full_context += f"\n\n{tab_data}"
                print(f"[CHAT_MODE] Tab data injected for {ui_ctx.job_type}: {len(tab_data)} chars")
        except Exception as e:
            print(f"[CHAT_MODE] Tab data injection failed: {e}")
    
    # v7.0: Gather codebase context for trusted models (sandbox read-only)
    codebase_ctx = ""
    if _codebase_gather_pending:
        try:
            codebase_ctx = gather_codebase_context(
                message=req.message, model=model, db=db,
            )
            if codebase_ctx:
                full_context += f"\n\n{codebase_ctx}"
                print(f"[CHAT_MODE] Codebase context injected: {len(codebase_ctx)} chars")
        except Exception as e:
            print(f"[CHAT_MODE] Codebase context failed (non-fatal): {e}")
    
    system_prompt = build_system_prompt(project, full_context, ui_context=ui_ctx)
    
    # v1.1: GROUNDING GATE — intercept claims-dependent queries
    _grounding_meta = {}
    if _GROUNDING_AVAILABLE and run_grounding_sync is not None:
        try:
            system_prompt, _grounding_meta = run_grounding_sync(
                message=req.message,
                system_prompt=system_prompt,
                context={"user_id": getattr(req, "user_id", "default")},
            )
            if _grounding_meta.get("grounding_applied"):
                print(
                    f"[CHAT_MODE] Grounding gate ACTIVE: "
                    f"category={_grounding_meta.get('category')}, "
                    f"sources={_grounding_meta.get('source_count')}, "
                    f"domain={_grounding_meta.get('domain_hint')}"
                )
            else:
                print(
                    f"[CHAT_MODE] Grounding gate: no grounding needed "
                    f"(category={_grounding_meta.get('category', 'n/a')}, "
                    f"reason={_grounding_meta.get('reason', 'personal')})"
                )
        except Exception as e:
            print(f"[CHAT_MODE] Grounding gate error (non-fatal): {e}")
    
    # v8.0: Give trusted models real tool access instead of stripping it.
    # When a trusted model (Opus etc.) is selected, provide actual tools
    # so it can read files, list dirs, search, run commands for real.
    # The pre-loaded codebase context is STILL injected (gives a head start)
    # but the model can now explore further on its own.
    _chat_tools = None
    try:
        from app.llm.chat_tool_loop import is_tool_eligible, get_chat_tools
        # v12.0: If the model lacks tool access but the context needs it
        # (builds tab, codebase exploration, architecture planning), swap
        # to an Anthropic model that has full tool loop support.
        if not is_tool_eligible(provider, model):
            # v12.1: Check if context needs tools — builds tab, codebase exploration,
            # or any request that hit deep keywords (filesystem/app exploration)
            from app.memory.complexity import DEEP_KEYWORDS, _count_keyword_hits
            _deep_hits = _count_keyword_hits(req.message.lower(), DEEP_KEYWORDS)
            _needs_tools = (
                _is_builds_context(req)
                or _detect_codebase_exploration(req.message)
                or _deep_hits >= 2
            )
            if _needs_tools:
                import os as _os
                _tool_provider = _os.getenv("TOOL_CHAT_PROVIDER", "google")
                _tool_model = _os.getenv("TOOL_CHAT_MODEL", "gemini-3.1-pro-preview-customtools")
                if is_tool_eligible(_tool_provider, _tool_model):
                    print(f"[CHAT_MODE] Context needs tools but {provider}/{model} has none — "
                          f"swapping to {_tool_provider}/{_tool_model}")
                    provider = _tool_provider
                    model = _tool_model
                    _set_sticky_model(req.project_id, provider, model)
        if is_tool_eligible(provider, model):
            _chat_tools = get_chat_tools()
            print(f"[CHAT_MODE] Tool access ENABLED for {provider}/{model} ({len(_chat_tools)} tools)")
            # v8.2: Inject tool role into system prompt for tool-enabled chat
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
            system_prompt += _TOOL_ROLE_BLOCK
        else:
            # Non-trusted models: strip codebase tool claims to prevent hallucination
            if _codebase_gather_pending and codebase_ctx:
                _CHAT_TOOLS_OVERRIDE = (
                    "   - You CAN: read files, write files, execute code, explore directories\n"
                )
                _CHAT_TOOLS_REPLACEMENT = (
                    "   - Codebase files have been PRE-LOADED into your context below.\n"
                    "   - You do NOT have codebase tool access. Do NOT generate tool_call blocks for file operations.\n"
                    "   - Do NOT call execute_command or shell commands.\n"
                    "   - Reference the [CODEBASE CONTEXT] files directly in your response.\n"
                )
                if _CHAT_TOOLS_OVERRIDE in system_prompt:
                    system_prompt = system_prompt.replace(
                        _CHAT_TOOLS_OVERRIDE, _CHAT_TOOLS_REPLACEMENT,
                    )
                    print("[CHAT_MODE] Capability layer overridden for non-trusted model")
    except ImportError:
        print("[CHAT_MODE] chat_tool_loop not available")
    
    # v2.3: Universal web search — ALL models get web_search as a tool.
    # This is a safety net so the model can actually search the web even
    # when the translation layer misses a research intent.
    try:
        from app.debug.tool_definitions import get_universal_tools
        from app.llm.chat_tool_loop import _to_anthropic_tool_format
        _universal = [_to_anthropic_tool_format(t) for t in get_universal_tools()]
        if _chat_tools is not None:
            # Trusted model: add web_search to existing codebase tools
            _existing_names = {t.get("name") for t in _chat_tools}
            for ut in _universal:
                if ut["name"] not in _existing_names:
                    _chat_tools.append(ut)
            print(f"[CHAT_MODE] Universal tools merged: {len(_chat_tools)} total")
        else:
            # Non-trusted model: give web_search only
            _chat_tools = _universal
            _WEB_SEARCH_PROMPT = (
                "\n\n## WEB SEARCH TOOL\n"
                "You have access to a web_search tool. Use it when the user asks you to\n"
                "research, look up, find pricing, get current information, or anything\n"
                "that requires knowledge you do not have.\n"
                "IMPORTANT: Actually CALL the web_search tool. Do not just say you will.\n"
                "Call it with a specific search query and use the results in your response.\n"
            )
            system_prompt += _WEB_SEARCH_PROMPT
            print(f"[CHAT_MODE] Universal web_search tool injected for {provider}/{model}")
    except ImportError as _uie:
        print(f"[CHAT_MODE] Universal tools not available: {_uie}")
    
    if ui_ctx:
        print(f"[CHAT_MODE] UI context injected: view={ui_ctx.view_type}, job={ui_ctx.job_type}, label={ui_ctx.label}")
    
    print(f"[CHAT_MODE] Calling generate_sse_stream: provider={provider}, model={model}, messages={len(messages)}")
    
    return StreamingResponse(
        generate_sse_stream(
            project_id=req.project_id,
            message=req.message,
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            db=db,
            trace=trace,
            enable_reasoning=req.enable_reasoning,
            tools=_chat_tools,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# =============================================================================
# NORMAL ROUTING HANDLER
# =============================================================================

def handle_normal_routing(
    req: Any,  # StreamRequest
    project: Any,  # Project model
    db: Session,
    trace: Any,
) -> StreamingResponse:
    """
    Handle normal job-type routing with RAG fallback.
    
    v4.12: Includes RAG fallback for architecture queries.
    
    Args:
        req: StreamRequest with project_id, message, etc.
        project: Project ORM object
        db: Database session
        trace: Audit trace
    
    Returns:
        StreamingResponse for the routed job
    """
    
    # =========================================================================
    # RAG FALLBACK: Detect architecture questions when translation layer fails
    # =========================================================================
    if _RAG_STREAM_AVAILABLE and is_architecture_query(req.message):
        print(f"[NORMAL_ROUTING] RAG fallback: detected architecture query")
        print(f"[NORMAL_ROUTING]   message={req.message[:80]}...")
        return StreamingResponse(
            generate_rag_query_stream(
                project_id=req.project_id,
                message=req.message,
                db=db,
                trace=trace,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
    # Build context
    full_context = build_full_context(db, req.project_id, req.message, req.use_semantic_search)
    
    # Job continuation
    if req.continue_job_id and req.job_state == "needs_spec_clarification":
        provider = "anthropic"
        model = DEFAULT_MODELS["anthropic_opus"]
        messages = build_messages(req.message, req.project_id, db, req.include_history, req.history_limit)
        system_prompt = build_system_prompt(project, full_context)
        
        return StreamingResponse(
            generate_high_stakes_critique_stream(
                project_id=req.project_id,
                message=req.message,
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                messages=messages,
                full_context=full_context,
                job_type_str="architecture_design",
                db=db,
                trace=trace,
                enable_reasoning=req.enable_reasoning,
                continue_job_id=req.continue_job_id,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
    # Normal job classification
    job_type = classify_job_type(req.message, req.job_type or "")
    job_type_value = job_type.value
    
    # Determine provider/model
    if req.provider and req.model:
        provider, model = req.provider, req.model
    elif req.provider:
        provider = req.provider
        model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])
    else:
        provider, model = select_provider_for_job_type(job_type)
    
    # v3.2: Provider availability check with matched model fallback
    providers_available = get_available_streaming_providers()
    if not providers_available.get(provider, False):
        available = get_available_streaming_provider()
        if not available:
            raise HTTPException(status_code=503, detail="No LLM provider available")
        print(f"[NORMAL_ROUTING] Provider '{provider}' not available, falling back to {available}")
        provider = available
        model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS.get("google", "gemini-2.5-flash"))
    
    # Build messages and system prompt
    messages = build_messages(req.message, req.project_id, db, req.include_history, req.history_limit)
    
    # v7.0: Inject codebase context for trusted models (same as handle_chat_mode)
    _nr_codebase_ctx = ""
    try:
        from app.llm.routing.chat_codebase_reader import (
            gather_codebase_context, is_trusted_model,
        )
        if is_trusted_model(model):
            _nr_codebase_ctx = gather_codebase_context(
                message=req.message, model=model, db=db,
            )
            if _nr_codebase_ctx:
                full_context += f"\n\n{_nr_codebase_ctx}"
                print(f"[NORMAL_ROUTING] Codebase context injected: {len(_nr_codebase_ctx)} chars")
    except Exception as e:
        print(f"[NORMAL_ROUTING] Codebase context failed (non-fatal): {e}")
    
    system_prompt = build_system_prompt(project, full_context)
    
    # v1.1: GROUNDING GATE — intercept claims-dependent queries in normal routing
    if _GROUNDING_AVAILABLE and run_grounding_sync is not None:
        try:
            system_prompt, _nr_grounding = run_grounding_sync(
                message=req.message,
                system_prompt=system_prompt,
                context={"user_id": getattr(req, "user_id", "default")},
            )
            if _nr_grounding.get("grounding_applied"):
                print(
                    f"[NORMAL_ROUTING] Grounding gate ACTIVE: "
                    f"category={_nr_grounding.get('category')}, "
                    f"sources={_nr_grounding.get('source_count')}"
                )
        except Exception as e:
            print(f"[NORMAL_ROUTING] Grounding gate error (non-fatal): {e}")
    
    # v8.0: Give trusted models real tool access in normal routing too
    _nr_tools = None
    try:
        from app.llm.chat_tool_loop import is_tool_eligible, get_chat_tools
        if is_tool_eligible(provider, model):
            _nr_tools = get_chat_tools()
            print(f"[NORMAL_ROUTING] Tool access ENABLED for {provider}/{model} ({len(_nr_tools)} tools)")
            # v8.2: Same tool role for normal routing (includes user file writes)
            _TOOL_ROLE_BLOCK = (
                "\n\n## TOOL ACCESS\n"
                "You have tool access for exploring the codebase AND writing to user folders.\n\n"
                "CODEBASE TOOLS (read-only): read_file, list_files, search_files, read_logs, search_my_files, read_user_file\n"
                "USER FILE TOOLS (read+write): get_user_folders, write_user_file\n"
                "Use get_user_folders to discover real folder paths, then write_user_file to save files there.\n\n"
                "YOUR ROLE: You are a RESEARCHER and ASSISTANT.\n"
                "- Explore files, read code, understand patterns, discover design tokens\n"
                "- Report your findings as text in the chat -- describe what you found\n"
                "- Present component structures, CSS variables, layout patterns, file paths\n"
                "- When the user asks to save/create a file, use get_user_folders + write_user_file\n"
                "- Codebase research will be picked up by the Weaver to create accurate build specs\n\n"
                "DO NOT:\n"
                "- Generate code blocks, full file contents, or implementation files for the codebase\n"
                "- Try to create, write, or modify ASTRA codebase files -- those go through the sandbox\n"
                "- Dump raw file contents -- summarise and highlight the relevant patterns\n"
            )
            system_prompt += _TOOL_ROLE_BLOCK
        elif _nr_codebase_ctx:
            _TOOLS_LINE = "   - You CAN: read files, write files, execute code, explore directories\n"
            _TOOLS_REPLACE = (
                "   - Codebase files have been PRE-LOADED into your context below.\n"
                "   - You do NOT have codebase tool access. Do NOT generate tool_call blocks for file operations.\n"
                "   - Do NOT call execute_command or shell commands.\n"
                "   - Reference the [CODEBASE CONTEXT] files directly in your response.\n"
            )
            if _TOOLS_LINE in system_prompt:
                system_prompt = system_prompt.replace(_TOOLS_LINE, _TOOLS_REPLACE)
                print("[NORMAL_ROUTING] Capability layer overridden for non-trusted model")
    except ImportError:
        print("[NORMAL_ROUTING] chat_tool_loop not available")
    
    # v2.3: Universal web search — ALL models get web_search as a tool.
    try:
        from app.debug.tool_definitions import get_universal_tools
        from app.llm.chat_tool_loop import _to_anthropic_tool_format
        _universal = [_to_anthropic_tool_format(t) for t in get_universal_tools()]
        if _nr_tools is not None:
            _existing_names = {t.get("name") for t in _nr_tools}
            for ut in _universal:
                if ut["name"] not in _existing_names:
                    _nr_tools.append(ut)
            print(f"[NORMAL_ROUTING] Universal tools merged: {len(_nr_tools)} total")
        else:
            _nr_tools = _universal
            _WEB_SEARCH_PROMPT = (
                "\n\n## WEB SEARCH TOOL\n"
                "You have access to a web_search tool. Use it when the user asks you to\n"
                "research, look up, find pricing, get current information, or anything\n"
                "that requires knowledge you do not have.\n"
                "IMPORTANT: Actually CALL the web_search tool. Do not just say you will.\n"
                "Call it with a specific search query and use the results in your response.\n"
            )
            system_prompt += _WEB_SEARCH_PROMPT
            print(f"[NORMAL_ROUTING] Universal web_search tool injected for {provider}/{model}")
    except ImportError as _uie:
        print(f"[NORMAL_ROUTING] Universal tools not available: {_uie}")
    
    # High-stakes routing
    if provider == "anthropic" and is_opus_model(model) and is_high_stakes_job(job_type_value):
        return StreamingResponse(
            generate_high_stakes_critique_stream(
                project_id=req.project_id,
                message=req.message,
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                messages=messages,
                full_context=full_context,
                job_type_str=job_type_value,
                db=db,
                trace=trace,
                enable_reasoning=req.enable_reasoning,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
    # Normal stream
    return StreamingResponse(
        generate_sse_stream(
            project_id=req.project_id,
            message=req.message,
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            db=db,
            trace=trace,
            enable_reasoning=req.enable_reasoning,
            tools=_nr_tools,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# =============================================================================
# LEGACY TRIGGERS HANDLER
# =============================================================================

def handle_legacy_triggers(
    req: Any,  # StreamRequest
    db: Session,
    trace: Any,
) -> Optional[StreamingResponse]:
    """
    Handle legacy triggers when translation layer unavailable.
    
    Args:
        req: StreamRequest with project_id and message
        db: Database session
        trace: Audit trace
    
    Returns:
        StreamingResponse if trigger matched, None otherwise
    """
    sse_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    
    if _SANDBOX_AVAILABLE and is_sandbox_trigger(req.message):
        return StreamingResponse(
            generate_sandbox_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    
    if is_update_arch_trigger(req.message) and _LOCAL_TOOLS_AVAILABLE:
        return StreamingResponse(
            generate_update_architecture_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    
    if is_archmap_trigger(req.message) and _LOCAL_TOOLS_AVAILABLE:
        return StreamingResponse(
            generate_local_architecture_map_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    
    if is_zobie_map_trigger(req.message) and _LOCAL_TOOLS_AVAILABLE:
        return StreamingResponse(
            generate_local_zobie_map_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    
    if _INTROSPECTION_AVAILABLE and is_introspection_trigger(req.message):
        return StreamingResponse(
            generate_introspection_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    
    return None


__all__ = [
    "handle_chat_mode",
    "handle_normal_routing",
    "handle_legacy_triggers",
]
