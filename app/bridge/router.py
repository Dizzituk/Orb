# FILE: app/bridge/router.py
"""
Astra Bridge API — endpoints for the Android companion app.

Provides login, chat (with project persistence), project listing,
message history, and health check.

v1.0 (2026-03-12): Initial — /bridge/login, /bridge/chat, /bridge/health.
v2.0 (2026-03-13): Project-integrated chat. Chats from the phone app now
    appear in Recent Chats on desktop. Added /bridge/projects,
    /bridge/projects/{id}/messages, project_id support in /bridge/chat.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import config as auth_config
from app.db import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/bridge", tags=["Bridge"])
security = HTTPBearer(auto_error=False)

# v4.0: Pending desktop navigation (bridge -> desktop tab switch)
_pending_navigation: Optional[dict] = None


def _push_desktop_navigation(domain: str):
    """Queue a navigation event for the desktop to pick up."""
    global _pending_navigation
    _DOMAIN_TO_JOB = {
        "finance": "accounts",
        "investments": "investments",
        "content": "content",
        "social": "social_media",
        "lifestyle": "health_fitness",
        "debug": "debug",
        "education": "education",
        "builds": "project_builds",
    }
    _pending_navigation = {
        "domain": domain,
        "job_type": _DOMAIN_TO_JOB.get(domain, domain),
        "timestamp": datetime.now().isoformat(),
    }
    logger.info("[bridge] Queued desktop navigation: %s -> %s", domain, _DOMAIN_TO_JOB.get(domain, domain))


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class BridgeLoginRequest(BaseModel):
    password: str


class BridgeLoginResponse(BaseModel):
    session_token: str
    message: str


class BridgeChatRequest(BaseModel):
    message: str
    project_id: Optional[int] = None  # If set, continues an existing chat


class BridgeChatResponse(BaseModel):
    reply: str
    project_id: int  # Always returned so the phone can continue the chat
    project_name: str
    domain: Optional[str] = None  # v3.0: detected domain (investments, finance, etc.)


class BridgeProjectOut(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    type: Optional[str] = None
    created_at: str
    updated_at: str


class BridgeMessageOut(BaseModel):
    id: int
    role: str
    content: str
    provider: Optional[str] = None
    model: Optional[str] = None
    created_at: str


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

async def _require_bridge_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> bool:
    """Validate session token from the bridge app."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required. Use /bridge/login first.",
        )
    token = credentials.credentials
    if auth_config.validate_session(token):
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired session. Please log in again via /bridge/login.",
    )


# ---------------------------------------------------------------------------
# Model selection & LLM helpers (v5.0)
# ---------------------------------------------------------------------------

# Domains that warrant a more capable model than gpt-5-mini
_ESCALATED_DOMAINS = {"builds", "debug"}


def _select_bridge_model(
    message: str,
    translation_result,
    domain_context: str,
) -> tuple:
    """Pick provider/model based on domain and message complexity.

    Escalation rules:
      1. BUILDS or DEBUG domain → GPT-5.4 (needs reasoning + code awareness)
      2. Complexity classifier says 'deep' → env-driven deep model
      3. Complexity classifier says 'reasoning' → env-driven chat model
      4. Default → gpt-5-mini (fast, cheap, good for quick queries)

    Returns (provider, model) tuple.
    """
    # Check domain-based escalation first
    _domain = None
    if (translation_result
            and translation_result.resolved_intent
            and translation_result.resolved_intent.value.startswith("DOMAIN_")):
        try:
            from app.llm.translation_routing import intent_to_routing_info
            info = intent_to_routing_info(translation_result.resolved_intent)
            _domain = info.get("domain") if info else None
        except Exception:
            pass

    if _domain in _ESCALATED_DOMAINS:
        _prov = os.getenv("BUILD_CHAT_PROVIDER", "openai")
        _mod = os.getenv("BUILD_CHAT_MODEL", "gpt-5.4")
        logger.info("[bridge] Domain '%s' -> escalated to %s/%s", _domain, _prov, _mod)
        return (_prov, _mod)

    # Run complexity classifier
    try:
        from app.memory.complexity import classify_complexity
        complexity = classify_complexity(query=message, intent=None)

        if complexity.tier == "deep":
            _prov = os.getenv("CHAT_DEEP_PROVIDER", "openai")
            _mod = os.getenv("CHAT_DEEP_MODEL", "gpt-5.4")
            logger.info("[bridge] Complexity 'deep' -> %s/%s", _prov, _mod)
            return (_prov, _mod)

        if complexity.tier == "reasoning":
            _prov = os.getenv("CHAT_PROVIDER", "openai")
            _mod = os.getenv("CHAT_MODEL", "gpt-5-mini")
            logger.info("[bridge] Complexity 'reasoning' -> %s/%s", _prov, _mod)
            return (_prov, _mod)
    except Exception as e:
        logger.debug("[bridge] Complexity classifier unavailable: %s", e)

    # Default: fast model for quick queries
    _prov = "openai"
    _mod = os.getenv("CHAT_QUICK_MODEL", "gpt-5-mini")
    logger.info("[bridge] Default model -> %s/%s", _prov, _mod)
    return (_prov, _mod)


def _build_bridge_system_prompt(domain_context: str) -> str:
    """Build the system prompt for bridge chat.

    Keeps it concise (phone context) but includes domain data when available.
    """
    prompt = (
        "You are Astra, a helpful AI assistant. "
        "The user is speaking from their phone via the Astra Bridge app. "
        "Keep responses concise — they may be driving. "
        "Be warm, direct, and useful."
    )
    if domain_context:
        prompt += (
            "\n\nYou have access to the following real-time data from ASTRA's systems. "
            "Use it to answer the user's question accurately:\n\n"
            + domain_context
        )
    return prompt


async def _call_llm(provider: str, model: str, messages: list) -> str:
    """Call the LLM via the appropriate provider.

    Supports openai, anthropic, and google providers so bridge chat
    can use any model the desktop supports.
    """
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=2048,
        )
        return response.choices[0].message.content or "No response generated."

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        # Anthropic expects system as a separate param
        system_msg = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg += m["content"] + "\n"
            else:
                chat_messages.append(m)
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system_msg.strip(),
            messages=chat_messages,
        )
        return response.content[0].text if response.content else "No response generated."

    if provider == "google":
        import google.generativeai as genai
        gmodel = genai.GenerativeModel(model)
        # Convert to Gemini format
        history = []
        system_text = ""
        for m in messages:
            if m["role"] == "system":
                system_text += m["content"] + "\n"
            elif m["role"] == "user":
                history.append({"role": "user", "parts": [m["content"]]})
            elif m["role"] == "assistant":
                history.append({"role": "model", "parts": [m["content"]]})
        # Prepend system as first user message if present
        if system_text and history:
            history[0]["parts"].insert(0, system_text.strip() + "\n\n")
        chat = gmodel.start_chat(history=history[:-1] if history else [])
        last_msg = history[-1]["parts"][0] if history else "Hello"
        response = chat.send_message(last_msg)
        return response.text or "No response generated."

    raise ValueError(f"Unknown provider: {provider}")

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/login", response_model=BridgeLoginResponse)
async def bridge_login(req: BridgeLoginRequest):
    """Log in from the bridge app. Returns a session token."""
    if not auth_config.is_auth_configured():
        raise HTTPException(400, "Password not configured on desktop.")

    result = auth_config.login(req.password)
    if not result:
        raise HTTPException(401, "Invalid password")

    return BridgeLoginResponse(
        session_token=result["session_token"],
        message="Connected to ASTRA backend",
    )


@router.get("/projects", response_model=List[BridgeProjectOut])
async def bridge_list_projects(
    limit: int = 30,
    db: Session = Depends(get_db),
    _auth: bool = Depends(_require_bridge_auth),
):
    """List recent chat projects — same data that appears in desktop Recent Chats."""
    from app.memory.service import list_projects

    projects = list_projects(db)
    result = []
    for p in projects[:limit]:
        result.append(BridgeProjectOut(
            id=p.id,
            name=p.name,
            description=p.description,
            type=p.type if hasattr(p, "type") else None,
            created_at=p.created_at.isoformat() if p.created_at else "",
            updated_at=p.updated_at.isoformat() if p.updated_at else "",
        ))
    return result


@router.get("/projects/{project_id}/messages", response_model=List[BridgeMessageOut])
async def bridge_get_messages(
    project_id: int,
    limit: int = 100,
    db: Session = Depends(get_db),
    _auth: bool = Depends(_require_bridge_auth),
):
    """Get message history for a project. Same messages visible on desktop."""
    from app.memory.service import get_project, list_messages

    project = get_project(db, project_id)
    if not project:
        raise HTTPException(404, f"Project {project_id} not found")

    messages = list_messages(db, project_id, limit=limit)
    return [
        BridgeMessageOut(
            id=m.id,
            role=m.role,
            content=m.content,
            provider=m.provider if hasattr(m, "provider") else None,
            model=m.model if hasattr(m, "model") else None,
            created_at=m.created_at.isoformat() if m.created_at else "",
        )
        for m in messages
    ]


@router.post("/chat", response_model=BridgeChatResponse)
async def bridge_chat(
    req: BridgeChatRequest,
    db: Session = Depends(get_db),
    _auth: bool = Depends(_require_bridge_auth),
):
    """Send a chat message from the bridge app.

    If project_id is provided, continues that conversation.
    If not, creates a new project (appears in Recent Chats on desktop).
    Both the user message and assistant reply are saved to the project.
    """
    from app.memory.service import get_project, create_message
    from app.memory.schemas import ProjectCreate, MessageCreate
    from app.memory._service_utils_2 import create_project

    # Resolve or create project
    project = None
    if req.project_id:
        project = get_project(db, req.project_id)
        if not project:
            raise HTTPException(404, f"Project {req.project_id} not found")

    if not project:
        # Create a new project — name from first few words of the message
        name_preview = req.message[:50].strip()
        if len(req.message) > 50:
            name_preview += "..."
        project = create_project(db, ProjectCreate(
            name=name_preview,
            description="Chat from Astra Bridge (phone)",
            type="bridge",
        ))
        logger.info("[bridge] Created project %d: %s", project.id, project.name)

    # Save user message
    create_message(db, MessageCreate(
        project_id=project.id,
        role="user",
        content=req.message,
        provider="bridge",
        model="phone-input",
    ))

    # Get conversation history for context
    from app.memory._service_utils_2 import list_messages
    history = list_messages(db, project.id, limit=20)
    history_messages = [
        {"role": m.role, "content": m.content}
        for m in history
        if m.role in ("user", "assistant")
    ]

    # v3.0: Route through translation layer for domain awareness
    domain_context = ""
    translation_result = None
    try:
        from app.llm.translation_routing import route_via_translation_layer
        translation_result = route_via_translation_layer(req.message)
        if (translation_result
                and translation_result.resolved_intent
                and translation_result.resolved_intent.value.startswith("DOMAIN_")):
            from app.llm.translation_routing import intent_to_routing_info
            domain_info = intent_to_routing_info(translation_result.resolved_intent)
            if domain_info and domain_info.get("type") == "domain_chat":
                from app.llm.routing.domain_context import get_domain_context
                domain_context = get_domain_context(domain_info["domain"], db)
                logger.info("[bridge] Domain detected: %s (%d chars context)",
                           domain_info["domain"], len(domain_context))
    except Exception as e:
        logger.debug("[bridge] Translation layer unavailable: %s", e)

    # v5.0: Model selection — domain-aware + complexity-based escalation
    provider, model = _select_bridge_model(req.message, translation_result, domain_context)
    reply = ""

    try:
        system_prompt = _build_bridge_system_prompt(domain_context)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history_messages[-20:])

        reply = await _call_llm(provider, model, messages)
    except Exception as e:
        logger.error("[bridge] Chat failed (%s/%s): %s", provider, model, e)
        reply = f"Error: {e}"

    # Save assistant reply
    create_message(db, MessageCreate(
        project_id=project.id,
        role="assistant",
        content=reply,
        provider=provider,
        model=model,
    ))

    # Determine detected domain for phone app
    _detected_domain = None
    if domain_context:
        try:
            _detected_domain = domain_info.get("domain")
        except Exception:
            pass

    # v4.0: If a domain was detected, push navigation to desktop
    if _detected_domain:
        _push_desktop_navigation(_detected_domain)

    return BridgeChatResponse(
        reply=reply,
        project_id=project.id,
        project_name=project.name,
        domain=_detected_domain,
    )


@router.get("/pending-navigation")
async def bridge_pending_navigation():
    """Poll for pending navigation events from the bridge app.
    
    The desktop frontend polls this endpoint to check if the phone
    app has requested a tab switch. Returns and clears the pending event.
    """
    global _pending_navigation
    if _pending_navigation:
        event = _pending_navigation
        _pending_navigation = None
        return {"pending": True, **event}
    return {"pending": False}


# ---------------------------------------------------------------------------
# TTS proxy endpoints (v5.0)
# Proxy to the TTS microservice on port 8001 so AstraBridge only needs
# one backend URL. All TTS traffic goes through the bridge auth layer.
# ---------------------------------------------------------------------------

TTS_BASE = "http://127.0.0.1:8001"


class BridgeTTSRequest(BaseModel):
    text: str
    voice_name: Optional[str] = None
    speed: Optional[float] = None


@router.get("/tts/voices")
async def bridge_tts_voices(
    _auth: bool = Depends(_require_bridge_auth),
):
    """List available TTS voices (proxied from TTS microservice)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{TTS_BASE}/tts/voices")
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error("[bridge] TTS voices proxy failed: %s", e)
        raise HTTPException(502, f"TTS service unavailable: {e}")


@router.post("/tts/speak")
async def bridge_tts_speak(
    req: BridgeTTSRequest,
    _auth: bool = Depends(_require_bridge_auth),
):
    """Synthesise speech (proxied from TTS microservice).

    Returns MP3 audio bytes.
    """
    import httpx
    payload = {"text": req.text}
    if req.voice_name:
        payload["voice_name"] = req.voice_name
    if req.speed is not None:
        payload["speed"] = req.speed

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{TTS_BASE}/tts/speak",
                json=payload,
            )
            resp.raise_for_status()
            # Return the audio bytes with correct content type
            from fastapi.responses import Response
            return Response(
                content=resp.content,
                media_type=resp.headers.get("content-type", "audio/mpeg"),
            )
    except Exception as e:
        logger.error("[bridge] TTS speak proxy failed: %s", e)
        raise HTTPException(502, f"TTS service unavailable: {e}")


@router.post("/tts/voices/select")
async def bridge_tts_select_voice(
    voice_name: str,
    _auth: bool = Depends(_require_bridge_auth),
):
    """Set the active TTS voice (proxied from TTS microservice)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{TTS_BASE}/tts/voices/select",
                json={"voice_name": voice_name},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error("[bridge] TTS voice select proxy failed: %s", e)
        raise HTTPException(502, f"TTS service unavailable: {e}")


@router.post("/tts/preview")
async def bridge_tts_preview(
    voice_name: str,
    _auth: bool = Depends(_require_bridge_auth),
):
    """Preview a TTS voice with sample text (proxied from TTS microservice)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{TTS_BASE}/tts/preview",
                json={"voice_name": voice_name},
            )
            resp.raise_for_status()
            from fastapi.responses import Response
            return Response(
                content=resp.content,
                media_type=resp.headers.get("content-type", "audio/mpeg"),
            )
    except Exception as e:
        logger.error("[bridge] TTS preview proxy failed: %s", e)
        raise HTTPException(502, f"TTS service unavailable: {e}")

@router.get("/health")
async def bridge_health():
    """Health check for the bridge app (no auth required)."""
    return {"status": "ok", "service": "astra-bridge-api"}


# ═══════════════════════════════════════════════════════════════════
# Crash Report Receiver
# ═══════════════════════════════════════════════════════════════════

@router.post("/crash-report")
async def receive_crash_report(request: Request):
    """Receive crash reports from AstraBridge Android app and email them."""
    import json
    try:
        body = await request.json()
        report = body.get("report", "No report content")
        app_name = body.get("app", "Unknown")
        timestamp = body.get("timestamp", 0)

        logger.info("[bridge] Received crash report from %s (%d chars)", app_name, len(report))

        # Try to send via Proton Mail
        email_sent = False
        try:
            from app.cloud.proton_mail import send_email
            subject = f"[CRASH] {app_name} — {report.split(chr(10))[1] if chr(10) in report else 'Unknown crash'}"
            await send_email(
                to_address=None,  # Uses default recipient (your email)
                subject=subject[:120],
                body=report,
            )
            email_sent = True
            logger.info("[bridge] Crash report emailed successfully")
        except Exception as mail_err:
            logger.warning("[bridge] Could not email crash report: %s", mail_err)

        # Also save locally as backup
        import os
        crash_dir = os.path.join("D:\\Orb", "logs", "crash_reports")
        os.makedirs(crash_dir, exist_ok=True)
        crash_file = os.path.join(crash_dir, f"crash_{int(timestamp) or 'unknown'}.txt")
        with open(crash_file, "w") as f:
            f.write(report)
        logger.info("[bridge] Crash report saved to %s", crash_file)

        return {
            "received": True,
            "emailed": email_sent,
            "saved": crash_file,
        }
    except Exception as e:
        logger.error("[bridge] Crash report processing failed: %s", e)
        return {"received": False, "error": str(e)}
