# FILE: app/llm/project_scoping_stream.py
# Purpose: Project Scoping Conversation Handler.
# Called-by: app.llm._stream_router_utils, app.llm.routing._cd_dispatch_table
# Depends-on: app.llm.routing.chat_routing, app.llm.streaming, app.memory, app.memory.schemas (+4 more)
# Last-renovated: 2026-06-11
"""
Project Scoping Conversation Handler.

When PROJECT_SCOPE_START fires, this handler takes over the chat and runs
a multi-turn conversation to gather project details. It escalates to the
deep chat model (GPT-5.4) and asks one question at a time.

Once enough information is gathered, it creates the project folder,
registers a dynamic BuildTargetProfile, and sets the ProjectSession.

v1.0 (2026-03-11): Initial implementation.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import AsyncIterator, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _sse(data: dict) -> bytes:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n".encode("utf-8")


def _deep_chat_model() -> str:
    """Resolve the deep-chat model: CHAT_DEEP_MODEL env override, else the REASONING role."""
    env_model = os.getenv("CHAT_DEEP_MODEL")
    if env_model:
        return env_model
    try:
        from app.llm.frontier_models import get_role_model
        return get_role_model("REASONING")[1]
    except Exception:
        return ""


async def generate_project_scoping_stream(
    *,
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[object] = None,
    conversation_id: str = "",
) -> AsyncIterator[bytes]:
    """Stream handler for PROJECT_SCOPE_START.

    Enters a conversational scoping flow. On first entry, sets up
    the ProjectSession and asks the first question. On subsequent
    messages while scoping_active, continues the conversation.
    """
    from app.shared_context.project_session import get_project_session

    provider = os.getenv("CHAT_DEEP_PROVIDER", "openai")
    model = _deep_chat_model()

    # Always key by project_id for consistency
    scope_key = str(project_id)
    session = get_project_session(scope_key)

    yield _sse({"type": "metadata", "provider": provider, "model": model})

    # ── First entry: start scoping ──
    if not session.scoping_active and not session.is_set:
        session.start_scoping()
        session.add_scoping_message("user", message)

        # Set sticky model so even if flow falls through, chat uses GPT-5.4
        try:
            from app.llm.routing.chat_routing import _set_sticky_model
            _set_sticky_model(project_id, provider, model)
        except Exception:
            pass

        opening = (
            "Let's scope out your new project. I'll ask you a few questions "
            "one at a time so we get a clear picture before building anything.\n\n"
            "**What kind of project do you want to create?**\n"
            "- Android app\n"
            "- Web app\n"
            "- ASTRA feature (backend or frontend)\n"
            "- Something else\n"
        )

        yield _sse({"type": "token", "content": opening})
        session.add_scoping_message("assistant", opening)

        # Persist to memory so it shows in chat history
        _persist_message(db, project_id, "assistant", opening, provider, model)

        yield _sse({"type": "done", "provider": provider, "model": model})
        return

    # ── Continue scoping: feed message to LLM with context ──
    session.add_scoping_message("user", message)

    # Check if we have enough info to summarise
    scoping_summary = _try_extract_project_info(session.scoping_messages)

    if scoping_summary and scoping_summary.get("ready"):
        # We have enough — present summary and ask for confirmation
        summary_text = _format_project_summary(scoping_summary)
        yield _sse({"type": "token", "content": summary_text})
        session.add_scoping_message("assistant", summary_text)
        _persist_message(db, project_id, "assistant", summary_text, provider, model)
        yield _sse({"type": "done", "provider": provider, "model": model})
        return

    # Not enough info yet — ask the next question via LLM
    system_prompt = _build_scoping_system_prompt(session.scoping_messages)

    try:
        from app.llm.streaming import stream_openai, stream_anthropic, stream_gemini

        stream_fns = {
            "openai": stream_openai,
            "anthropic": stream_anthropic,
            "google": stream_gemini,
        }
        stream_fn = stream_fns.get(provider)
        if stream_fn is None:
            yield _sse({"type": "token", "content": f"Streaming not available for provider: {provider}"})
            yield _sse({"type": "done", "provider": provider, "model": model})
            return

        llm_messages = [
            {"role": "system", "content": system_prompt},
        ]
        # Add scoping history as conversation
        for msg in session.scoping_messages:
            llm_messages.append({"role": msg["role"], "content": msg["content"]})

        chunks = []
        async for chunk in stream_fn(messages=llm_messages, model=model):
            content = None
            if isinstance(chunk, dict):
                if chunk.get("type") == "error":
                    yield _sse({"type": "token", "content": f"\n⚠️ LLM error: {chunk.get('message', '')}"})
                    continue
                content = chunk.get("text") or chunk.get("content")
                if chunk.get("type") == "metadata":
                    continue
            elif hasattr(chunk, "choices") and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    content = delta.content
            if content:
                chunks.append(content)
                yield _sse({"type": "token", "content": content})

        response_text = "".join(chunks).strip()
        if response_text:
            session.add_scoping_message("assistant", response_text)
            _persist_message(db, project_id, "assistant", response_text, provider, model)

    except Exception as e:
        logger.exception("[project_scoping] LLM streaming failed")
        yield _sse({"type": "token", "content": f"\nScoping error: {e}"})

    yield _sse({"type": "done", "provider": provider, "model": model})


async def handle_scoping_confirmation(
    *,
    project_id: int,
    message: str,
    db: Session,
    conversation_id: str = "",
) -> AsyncIterator[bytes]:
    """Handle confirmation of a scoped project — create folder + register profile."""
    from app.shared_context.project_session import get_project_session
    from app.pipeline_v2.target_registry import register_profile
    from app.pipeline_v2.build_targets import BuildTargetProfile

    provider = os.getenv("CHAT_DEEP_PROVIDER", "openai")
    model = _deep_chat_model()

    scope_key = str(project_id)
    session = get_project_session(scope_key)
    yield _sse({"type": "metadata", "provider": provider, "model": model})

    info = _try_extract_project_info(session.scoping_messages)
    if not info or not info.get("ready"):
        yield _sse({"type": "token", "content": "I don't have enough information to create the project yet. Let's continue scoping."})
        yield _sse({"type": "done", "provider": provider, "model": model})
        return

    proj_name = info.get("name", "Untitled")
    proj_type = info.get("type", "android")
    proj_desc = info.get("description", "")
    proj_id = _slugify(proj_name)

    # Determine paths and config based on project type
    if proj_type in ("android", "android app"):
        proj_root = f"D:/Astra Android Folder/{proj_name.replace(' ', '')}"
        language = "kotlin"
        build_system = "gradle"
        framework = "jetpack-compose"
        source_root = f"app/src/main/java/com/astra/{proj_id.replace('-', '')}"
        package_name = f"com.astra.{proj_id.replace('-', '')}"
        arch_pattern = "mvvm"
        file_ext = ".kt"
    elif proj_type in ("web", "web app"):
        proj_root = f"D:/Astra Projects/{proj_name.replace(' ', '')}"
        language = "typescript"
        build_system = "npm"
        framework = "react"
        source_root = "src"
        package_name = proj_id
        arch_pattern = "component-page"
        file_ext = ".tsx"
    elif proj_type in ("backend", "astra feature"):
        proj_root = "D:/Orb"
        language = "python"
        build_system = "pip"
        framework = "fastapi"
        source_root = "app"
        package_name = "app"
        arch_pattern = "module-router"
        file_ext = ".py"
    else:
        proj_root = f"D:/Astra Projects/{proj_name.replace(' ', '')}"
        language = "python"
        build_system = "pip"
        framework = "generic"
        source_root = "src"
        package_name = proj_id
        arch_pattern = "modular"
        file_ext = ".py"

    # Create the project folder
    try:
        os.makedirs(proj_root.replace("/", os.sep), exist_ok=True)
        yield _sse({"type": "token", "content": f"📁 Created project folder: `{proj_root}`\n"})
    except Exception as e:
        yield _sse({"type": "token", "content": f"⚠️ Could not create folder: {e}\n"})

    # Build commands based on project type
    _JAVA_HOME = r'C:\Program Files\Android\Android Studio\jbr'
    _SET_JAVA = f'$env:JAVA_HOME = "{_JAVA_HOME}" ; '

    if build_system == "gradle":
        _syntax_cmd = (
            f'{_SET_JAVA}'
            f'cd "{proj_root}" ; '
            f'.\\gradlew.bat compileDebugKotlin 2>&1'
        )
        _build_cmd = (
            f'{_SET_JAVA}'
            f'cd "{proj_root}" ; '
            f'.\\gradlew.bat assembleDebug 2>&1'
        )
        _clean_cmd = (
            f'{_SET_JAVA}'
            f'cd "{proj_root}" ; '
            f'.\\gradlew.bat clean 2>&1'
        )
    elif build_system == "npm":
        _syntax_cmd = f'cd "{proj_root}" ; npx tsc --noEmit 2>&1'
        _build_cmd = f'cd "{proj_root}" ; npx tsc --noEmit 2>&1'
        _clean_cmd = ""
    else:
        _syntax_cmd = ""
        _build_cmd = ""
        _clean_cmd = ""

    # Register the BuildTargetProfile
    profile = BuildTargetProfile(
        project_id=proj_id,
        project_name=proj_name,
        project_root=proj_root,
        language=language,
        build_system=build_system,
        framework=framework,
        source_root=source_root,
        package_name=package_name,
        architecture_pattern=arch_pattern,
        file_extension=file_ext,
        syntax_check_cmd=_syntax_cmd,
        build_cmd=_build_cmd,
        clean_cmd=_clean_cmd,
    )
    register_profile(profile)
    yield _sse({"type": "token", "content": f"✅ Registered build target: **{proj_name}** (`{proj_id}`)\n"})

    # Set the ProjectSession
    session.finish_scoping(
        project_id=proj_id,
        project_name=proj_name,
        project_type=proj_type,
        project_root=proj_root,
    )
    yield _sse({"type": "token", "content": f"🎯 Project session active — pipeline will target `{proj_root}`\n\n"})

    # v1.1 (2026-07-04, JOB 4): durable target wiring. ProjectSession dies
    # with the process, so also stamp the BuildProject row — SpecGate's
    # Path-2 lookup (build_target_id -> get_profile) then survives backend
    # restarts, paired with the JOB 1 registry persistence. Must run AFTER
    # finish_scoping so get_or_create's stale-target check sees the new
    # session and archives any old build project first.
    try:
        from app.builds.pipeline_bridge import get_or_create_build_project
        _bp = get_or_create_build_project(db, project_id, brief=proj_desc or proj_name)
        if _bp is not None:
            if _bp.build_target_id != proj_id:
                _bp.build_target_id = proj_id
                _bp.target_path = proj_root
                db.commit()
            logger.info(
                "[project_scoping] BuildProject %s bound to target %s (%s)",
                _bp.id, proj_id, proj_root,
            )
            yield _sse({"type": "token", "content": f"🔗 Build project bound to target `{proj_id}` (survives restarts)\n"})
    except Exception as e:
        logger.warning("[project_scoping] BuildProject target stamp failed (non-fatal): %s", e)

    yield _sse({"type": "token", "content": (
        f"Project **{proj_name}** is ready.\n"
        f"- Type: {proj_type}\n"
        f"- Language: {language}\n"
        f"- Framework: {framework}\n"
        f"- Root: `{proj_root}`\n\n"
        f"Start describing what you want to build, then say **'send that to weaver'** "
        f"when you're ready to generate the spec.\n"
    )})

    _persist_message(db, project_id, "assistant",
        f"Project '{proj_name}' created at {proj_root}. Ready for spec.",
        provider, model)

    yield _sse({"type": "done", "provider": provider, "model": model})


# ═══════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════

def _build_scoping_system_prompt(scoping_messages: list) -> str:
    """Build the system prompt for the scoping conversation LLM."""
    answered = set()
    for msg in scoping_messages:
        content = msg.get("content", "").lower()
        if msg["role"] == "user":
            if any(w in content for w in ("android", "web", "backend", "frontend", "astra feature")):
                answered.add("type")
            if len(content.split()) <= 5 and "type" in answered and "name" not in answered:
                answered.add("name")
            if len(content.split()) > 15:
                answered.add("description")

    return (
        "You are ASTRA, an AI development platform. You are in PROJECT SCOPING mode.\n\n"
        "Your job is to gather a DETAILED project brief that will be used to generate "
        "a full technical specification and build plan. The more detail you gather now, "
        "the better the automated build will be.\n\n"
        "Ask ONE question at a time.\n\n"
        "Information needed (in this order):\n"
        "1. Project type (Android app, web app, ASTRA feature) — "
        + ("✅ answered" if "type" in answered else "❓ ask first") + "\n"
        "2. Project name (2-4 words, NOT just 'Astra') — "
        + ("✅ answered" if "name" in answered else "❓ ask next") + "\n"
        "3. DETAILED description — this is the MOST IMPORTANT part.\n"
        "   Ask: 'Tell me everything about what this app does — features, user flows, "
        "how it connects to other systems, voice features, file handling, UI layout. "
        "The more detail the better, this goes straight into the build spec.'\n"
        + ("   ✅ answered\n" if "description" in answered else "   ❓ ask after name\n")
        + "4. Follow-up questions — dig into specifics based on what they said:\n"
        "   - Connection/networking (how does it reach the backend?)\n"
        "   - Platform constraints (offline support? background services?)\n"
        "   - Key integrations (APIs, services, hardware)\n\n"
        "Once you have rich detail (the user has given at least one long descriptive answer), "
        "summarise EVERYTHING as a structured brief and ask: "
        "'Shall I create this project and set up the build target?'\n\n"
        "RULES:\n"
        "- Ask ONE question per response\n"
        "- Keep YOUR questions short, but ENCOURAGE long detailed answers from the user\n"
        "- NEVER ask for a 'one-line description' — ask for the FULL picture\n"
        "- When the user gives detail, acknowledge key points then ask a targeted follow-up\n"
        "- Be conversational, not formal\n"
        "- 'Astra' alone is NOT a valid project name. Ask for something specific.\n"
        "- Do NOT re-ask questions you already have answers to\n"
        "- Once you have type + name + detailed description, move to confirmation\n"
    )

def _try_extract_project_info(messages: list) -> Optional[dict]:
    """Try to extract project info from the scoping conversation.

    Returns dict with 'type', 'name', 'description', 'ready' if we have enough.
    Returns None or {'ready': False} if not enough info yet.
    """
    proj_type = None
    proj_name = None
    proj_desc = None

    _type_msg_idx = None  # Track which message set the type

    for msg_idx, msg in enumerate(messages):
        if msg["role"] != "user":
            continue
        content = msg["content"].strip()
        lower = content.lower()

        # Detect type
        if proj_type is None:
            if any(w in lower for w in ("android app", "android", "kotlin", "phone app")):
                proj_type = "android"
                _type_msg_idx = msg_idx
            elif any(w in lower for w in ("web app", "website", "web", "react")):
                proj_type = "web"
                _type_msg_idx = msg_idx
            elif any(w in lower for w in ("backend", "astra feature", "astra itself", "python")):
                proj_type = "backend"
                _type_msg_idx = msg_idx

        # Detect name — look for short capitalised phrases or quoted names
        # Skip the same message that set the type ("Android app" is a type, not a name)
        if proj_name is None and proj_type is not None and msg_idx != _type_msg_idx:
            # Quoted name
            quoted = re.findall(r'"([^"]+)"|\u201c([^\u201d]+)\u201d|\'([^\']+)\'', content)
            if quoted:
                proj_name = next((q for q in quoted[0] if q), None)
            # Short response (1-5 words) that looks like a name
            elif len(content.split()) <= 5:
                # Don't grab type answers, confirmations, or common words as names
                # Skip single-word type/confirmation answers
                _cleaned = lower.strip().strip(".")
                skip_exact = (
                    "android", "web", "backend", "frontend", "yes", "no",
                    "astra", "astro", "aster",  # bare wake word only
                    "app", "feature", "it", "the", "a", "an",
                    "something", "sure", "yeah", "okay", "ok",
                    "android app", "web app",  # type answers
                )
                if not any(w == _cleaned for w in skip_exact):
                    if any(c.isupper() for c in content):
                        proj_name = content.strip().strip(".")

        # Detect description — longer user messages (8+ words)
        # Can be captured even without a name — name gets asked separately
        if proj_desc is None and proj_type and len(content.split()) > 8:
            proj_desc = content

    if proj_type and proj_name and proj_desc:
        return {
            "type": proj_type,
            "name": proj_name,
            "description": proj_desc,
            "ready": True,
        }

    # If we have type + description but no name, still not ready
    # The LLM will ask for the name specifically

    return {
        "type": proj_type,
        "name": proj_name,
        "description": proj_desc,
        "ready": False,
    }


def _format_project_summary(info: dict) -> str:
    """Format a project summary for user confirmation."""
    return (
        f"Here's what I've got:\n\n"
        f"- **Type:** {info.get('type', 'unknown')}\n"
        f"- **Name:** {info.get('name', 'unnamed')}\n"
        f"- **Description:** {info.get('description', 'none')}\n\n"
        f"Shall I create this project and set up the build target? "
        f"Say **yes** to confirm or tell me what to change.\n"
    )


def _slugify(name: str) -> str:
    """Convert a project name to a slug ID."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug or "untitled"


def _persist_message(
    db: Session,
    project_id: int,
    role: str,
    content: str,
    provider: str,
    model: str,
) -> None:
    """Persist a message to conversation history."""
    try:
        from app.memory import service as memory_service
        from app.memory import schemas as memory_schemas
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role=role,
                content=content,
                provider=provider,
                model=model,
            ),
        )
    except Exception as e:
        logger.warning("[project_scoping] Failed to persist message: %s", e)
