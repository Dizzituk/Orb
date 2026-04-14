# FILE: app/bridge/router.py
"""
Astra Bridge API — endpoints for the Android companion app.

v15.0 (2026-04-06): MAJOR REFACTOR — modularised into:
    - schemas.py:     Pydantic models + auth dependency
    - llm_helpers.py: Model selection, system prompt, direct LLM call
    - tts_proxy.py:   All TTS proxy endpoints (separate sub-router)
    - router.py:      Chat endpoints, projects, health, crash report (this file)

v7.1 (2026-04-06): Web search on chat-and-speak route
v7.0 (2026-04-06): Capability honesty enforcement
v5.0: Model selection + multi-provider LLM calls
v4.0: Desktop navigation push from bridge
v3.0: Translation layer + domain awareness
v2.0: Project-integrated chat
v1.0: Initial bridge API
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db import get_db

from .schemas import (
    BridgeLoginRequest,
    BridgeLoginResponse,
    BridgeChatRequest,
    BridgeChatResponse,
    BridgeProjectOut,
    BridgeMessageOut,
    require_bridge_auth,
)
from .llm_helpers import (
    push_desktop_navigation,
    pop_pending_navigation,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/bridge", tags=["Bridge"])


@router.post("/login", response_model=BridgeLoginResponse)
async def bridge_login(req: BridgeLoginRequest):
    from app.auth import config as auth_config

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
    _auth: bool = Depends(require_bridge_auth),
):
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
    _auth: bool = Depends(require_bridge_auth),
):
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
    _auth: bool = Depends(require_bridge_auth),
):
    from app.memory.service import create_message
    from app.memory.schemas import MessageCreate
    from app.memory._service_utils_2 import list_messages

    project = _resolve_or_create_project(req, db)

    create_message(db, MessageCreate(
        project_id=project.id, role="user",
        content=req.message, provider="bridge", model="phone-input",
    ))

    history = list_messages(db, project.id, limit=20)
    history_messages = [
        {"role": m.role, "content": m.content}
        for m in history if m.role in ("user", "assistant")
    ]

    domain_context, translation_result, domain_info = _run_translation(req.message, db)

    from app.bridge.capability_honesty import (
        is_unsupported_on_bridge, get_unsupported_message,
    )

    resolved_intent = (
        translation_result.resolved_intent.value
        if translation_result and translation_result.resolved_intent
        else None
    )

    if is_unsupported_on_bridge(resolved_intent):
        reply = get_unsupported_message(resolved_intent)
        logger.info("[bridge] Blocked unsupported intent: %s", resolved_intent)
        create_message(db, MessageCreate(
            project_id=project.id, role="assistant",
            content=reply, provider="bridge", model="capability-gate",
        ))
        return BridgeChatResponse(
            reply=reply, project_id=project.id, project_name=project.name,
            domain=domain_info.get("domain", "") if domain_info else "",
        )

    web_search_context, search_executed, search_succeeded, early_reply = (
        await _run_web_search(resolved_intent, translation_result, req.message)
    )

    if early_reply:
        from app.memory.service import create_message as _cm
        _cm(db, MessageCreate(
            project_id=project.id, role="assistant",
            content=early_reply, provider="bridge", model="search-gate",
        ))
        return BridgeChatResponse(
            reply=early_reply, project_id=project.id, project_name=project.name,
            domain=domain_info.get("domain", "") if domain_info else "",
        )

    from app.bridge.capability_layer import run_astra_chat

    result = await run_astra_chat(
        message=req.message,
        project_id=project.id,
        history=history_messages,
        db=db,
        source="bridge",
        domain_context=domain_context,
        translation_result=translation_result,
        web_search_context=web_search_context,
        search_executed=search_executed,
        search_succeeded=search_succeeded,
    )

    reply = result["reply"]
    provider = result["provider"]
    model = result["model"]

    create_message(db, MessageCreate(
        project_id=project.id, role="assistant",
        content=reply, provider=provider, model=model,
    ))

    _detected_domain = domain_info.get("domain") if domain_info else None
    if _detected_domain:
        push_desktop_navigation(_detected_domain)

    return BridgeChatResponse(
        reply=reply, project_id=project.id,
        project_name=project.name, domain=_detected_domain,
    )


@router.post("/chat-and-speak")
async def bridge_chat_and_speak(
    req: BridgeChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
):
    from app.memory.service import create_message
    from app.memory.schemas import MessageCreate
    from app.memory._service_utils_2 import list_messages
    from urllib.parse import quote as url_quote

    project = _resolve_or_create_project(req, db)

    idempotency_key = request.headers.get("X-Idempotency-Key")
    history_before_save = list_messages(db, project.id, limit=100)
    existing_user_message = next(
        (
            message for message in reversed(history_before_save)
            if message.role == "user" and message.content == req.message
        ),
        None,
    ) if idempotency_key else None

    if existing_user_message is None:
        create_message(db, MessageCreate(
            project_id=project.id, role="user",
            content=req.message, provider="bridge", model="phone-input",
        ))

    history = list_messages(db, project.id, limit=20)
    history_messages = [
        {"role": m.role, "content": m.content}
        for m in history if m.role in ("user", "assistant")
    ]

    from app.bridge.capability_honesty import (
        is_unsupported_on_bridge, get_unsupported_message,
    )

    domain_context, translation_result, domain_info = _run_translation(req.message, db)

    resolved_intent = (
        translation_result.resolved_intent.value
        if translation_result and translation_result.resolved_intent
        else None
    )

    if is_unsupported_on_bridge(resolved_intent):
        honest_reply = get_unsupported_message(resolved_intent)
        logger.info("[bridge] chat-and-speak: blocked unsupported %s", resolved_intent)
        create_message(db, MessageCreate(
            project_id=project.id, role="assistant",
            content=honest_reply, provider="bridge", model="capability-gate",
        ))

    web_search_context, search_executed, search_succeeded, honest_early_reply = (
        await _run_web_search(resolved_intent, translation_result, req.message)
    )

    from app.bridge.capability_layer import run_astra_chat
    from app.bridge.chat_and_speak import _synthesise_sentence, _split_into_sentences

    if is_unsupported_on_bridge(resolved_intent):
        full_text = get_unsupported_message(resolved_intent)
        provider = "bridge"
        model = "capability-gate"
    elif honest_early_reply:
        full_text = honest_early_reply
        provider = "bridge"
        model = "search-gate"
    else:
        result = await run_astra_chat(
            message=req.message,
            project_id=project.id,
            history=history_messages,
            db=db,
            source="bridge-tts",
            domain_context=domain_context,
            translation_result=translation_result,
            web_search_context=web_search_context,
            search_executed=search_executed,
            search_succeeded=search_succeeded,
        )
        full_text = result["reply"]
        provider = result["provider"]
        model = result["model"]

    project_id = project.id
    project_name = project.name

    assistant_message = create_message(db, MessageCreate(
        project_id=project_id, role="assistant",
        content=full_text, provider=provider, model=model,
    ))

    async def generate_audio():
        sentences = _split_into_sentences(full_text)
        for sentence in sentences:
            try:
                mp3_bytes = await _synthesise_sentence(sentence)
                yield mp3_bytes
            except Exception as e:
                logger.error("[bridge] TTS failed for sentence: %s", e)

    return StreamingResponse(
        generate_audio(),
        media_type="audio/mpeg",
        headers={
            "X-Project-Id": str(project_id),
            "X-Project-Name": url_quote(project_name),
            "X-Full-Text": url_quote(full_text[:8000]),
            "X-Message-Id": str(assistant_message.id),
            "Transfer-Encoding": "chunked",
        },
    )


@router.get("/pending-navigation")
async def bridge_pending_navigation():
    event = pop_pending_navigation()
    if event:
        return {"pending": True, **event}
    return {"pending": False}


@router.get("/health")
async def bridge_health():
    return {"status": "ok", "service": "astra-bridge-api"}


@router.post("/crash-report")
async def receive_crash_report(request: Request):
    import os
    try:
        body = await request.json()
        report = body.get("report", "No report content")
        app_name = body.get("app", "Unknown")
        timestamp = body.get("timestamp", 0)

        logger.info("[bridge] Received crash report from %s (%d chars)", app_name, len(report))

        email_sent = False
        try:
            from app.cloud.proton_mail import send_email
            subject = f"[CRASH] {app_name} — {report.split(chr(10))[1] if chr(10) in report else 'Unknown crash'}"
            await send_email(
                to_address=None,
                subject=subject[:120],
                body=report,
            )
            email_sent = True
            logger.info("[bridge] Crash report emailed successfully")
        except Exception as mail_err:
            logger.warning("[bridge] Could not email crash report: %s", mail_err)

        crash_dir = os.path.join("D:\\Orb", "logs", "crash_reports")
        os.makedirs(crash_dir, exist_ok=True)
        crash_file = os.path.join(crash_dir, f"crash_{int(timestamp) or 'unknown'}.txt")
        with open(crash_file, "w") as f:
            f.write(report)
        logger.info("[bridge] Crash report saved to %s", crash_file)

        return {"received": True, "emailed": email_sent, "saved": crash_file}
    except Exception as e:
        logger.error("[bridge] Crash report processing failed: %s", e)
        return {"received": False, "error": str(e)}


@router.post("/upload")
async def bridge_upload(
    request: Request,
    db: Session = Depends(get_db),
    auth=Depends(require_bridge_auth),
):
    """Accept a raw file upload from the Astra Bridge OutboundQueueDrainer.
    
    Bridge posts raw bytes with:
      - Authorization: Bearer <token>
      - X-Idempotency-Key: <uuid>   (dedup on resend after transient failure)
      - Content-Type: application/octet-stream
    
    Returns:
      - 200 {id, bytes, path} on success
      - 409 if this idempotency key was already accepted
      - 401/403 handled upstream by require_bridge_auth
    
    Files are written to D:/Orb/uploads/bridge/ and survive restarts.
    v1.0 (2026-04-13): initial implementation so Bridge attachments
    actually reach the server. Previously Bridge POSTed to /upload (404).
    """
    import os
    import uuid
    import json
    from datetime import datetime, timezone
    
    idempotency_key = request.headers.get("X-Idempotency-Key", "").strip()
    if not idempotency_key:
        # Generate one so the client can still succeed even if header was missing.
        idempotency_key = uuid.uuid4().hex
    
    upload_dir = os.path.join("D:\\\\Orb", "uploads", "bridge")
    os.makedirs(upload_dir, exist_ok=True)
    ledger_path = os.path.join(upload_dir, "_ledger.jsonl")
    
    # Dedup: if idempotency_key is already in the ledger, return 409.
    try:
        if os.path.isfile(ledger_path):
            with open(ledger_path, "r", encoding="utf-8") as lf:
                for line in lf:
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    if entry.get("idempotency_key") == idempotency_key:
                        logger.info("[bridge] upload duplicate for key=%s", idempotency_key[:12])
                        raise HTTPException(status_code=409, detail="duplicate")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("[bridge] ledger read failed, proceeding anyway: %s", e)
    
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty body")
    
    file_id = uuid.uuid4().hex
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{file_id}.bin"
    out_path = os.path.join(upload_dir, filename)
    with open(out_path, "wb") as f:
        f.write(body)
    
    entry = {
        "id": file_id,
        "idempotency_key": idempotency_key,
        "filename": filename,
        "path": out_path,
        "bytes": len(body),
        "received_at": datetime.now(timezone.utc).isoformat(),
        "content_type": request.headers.get("Content-Type", ""),
    }
    with open(ledger_path, "a", encoding="utf-8") as lf:
        lf.write(json.dumps(entry) + "\n")
    
    logger.info(
        "[bridge] upload received: id=%s bytes=%d key=%s",
        file_id, len(body), idempotency_key[:12],
    )
    
    return {"id": file_id, "bytes": len(body), "path": out_path}


def _resolve_or_create_project(req: BridgeChatRequest, db: Session):
    from app.memory.service import get_project
    from app.memory.schemas import ProjectCreate
    from app.memory._service_utils_2 import create_project, get_project_by_name

    project = None
    if req.project_id:
        project = get_project(db, req.project_id)
        if not project:
            raise HTTPException(404, f"Project {req.project_id} not found")

    if not project:
        name_preview = req.message[:50].strip()
        if len(req.message) > 50:
            name_preview += "..."
        existing = get_project_by_name(db, name_preview)
        if existing:
            project = existing
            logger.info("[bridge] Reusing existing project %d: %s", project.id, project.name)
        else:
            project = create_project(db, ProjectCreate(
                name=name_preview,
                description="Chat from Astra Bridge (phone)",
                type="bridge",
            ))
            logger.info("[bridge] Created project %d: %s", project.id, project.name)
    return project


def _run_translation(message: str, db: Session) -> tuple:
    domain_context = ""
    translation_result = None
    domain_info = None
    try:
        from app.translation import translate_message_sync
        from app.translation.modes import UIContext
        bridge_ctx = UIContext(in_job_config=True)
        translation_result = translate_message_sync(message, ui_context=bridge_ctx)
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
        logger.info("[bridge] Translation layer error: %s", e)
    return domain_context, translation_result, domain_info


async def _run_web_search(
    resolved_intent: str | None,
    translation_result,
    message: str,
) -> tuple:
    from app.bridge.capability_honesty import get_search_failed_message

    web_search_context = ""
    search_executed = False
    search_succeeded = False
    early_reply = None

    if resolved_intent not in ("WEB_SEARCH", "DEEP_RESEARCH"):
        return web_search_context, search_executed, search_succeeded, early_reply

    search_executed = True
    try:
        from app.llm.web_search import search_and_answer, WebSearchRequest
        search_query = (
            (translation_result.extracted_context or {}).get("extracted_query", "")
            or message
        )
        logger.info("[bridge] Web search triggered: %s", search_query[:80])
        search_result = await search_and_answer(
            WebSearchRequest(query=search_query, max_results=5)
        )
        if search_result and search_result.ok:
            search_succeeded = True
            sources_text = "\n".join(
                f"- [{s.title}]({s.url}): {s.snippet}"
                for s in search_result.sources[:5]
            )
            web_search_context = (
                f"## Web Search Results for: {search_result.query}\n\n"
                f"{search_result.answer}\n\n"
                f"Sources:\n{sources_text}\n"
            )
            logger.info("[bridge] Web search: %d sources, %d chars",
                       len(search_result.sources), len(web_search_context))
        else:
            logger.info("[bridge] Web search returned no results")
            early_reply = get_search_failed_message()
    except Exception as ws_err:
        logger.info("[bridge] Web search failed: %s", ws_err)
        early_reply = get_search_failed_message()

    return web_search_context, search_executed, search_succeeded, early_reply
