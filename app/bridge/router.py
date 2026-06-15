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
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db import get_db

from .schemas import (
    BridgeLoginRequest,
    BridgeLoginResponse,
    BridgeChatRequest,
    BridgeChatResponse,
    BridgeProjectOut,
    BridgeArtifactRef,
    BridgePinRequest,
    BridgeMessageOut,
    require_bridge_auth,
)
from .llm_helpers import (
    push_desktop_navigation,
    pop_pending_navigation,
)
from .uploads_store import save_upload, is_duplicate

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/bridge", tags=["Bridge"])

def _process_artifacts(content: str) -> tuple[str, list[BridgeArtifactRef]]:
    """Extract artifact markers from message content.

    Returns (stripped_content, refs). The stripped content has the markers
    removed so the phone displays clean text in the chat bubble and TTS
    speaks clean audio. The refs are enriched with size + mime via stat
    on the actual file and given absolute /bridge/artifacts URLs so the
    phone can fetch them via the auth-gated endpoint.

    Empty refs list when the content has no markers, which is the
    overwhelmingly common case. Both outputs are safe to pass to clients
    even when no artifacts exist.
    """
    from app.bridge.artifacts import (
        extract_artifacts, strip_artifacts, enrich_artifact_ref,
    )
    refs = extract_artifacts(content)
    if not refs:
        return content, []
    stripped = strip_artifacts(content)
    out: list[BridgeArtifactRef] = []
    for r in refs:
        enriched = enrich_artifact_ref(r)
        out.append(BridgeArtifactRef(
            kind=enriched.kind,
            filename=enriched.filename,
            size_bytes=enriched.size_bytes,
            mime_type=enriched.mime_type,
            url=f"/bridge/artifacts/{enriched.kind}/{enriched.filename}",
        ))
    return stripped, out



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
            pinned=bool(getattr(p, "pinned", False)),
        ))
    return result

@router.post("/projects/{project_id}/pin", response_model=BridgeProjectOut)
async def bridge_pin_project(
    project_id: int,
    req: BridgePinRequest,
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
):
    """Set the pinned state of a project.

    Idempotent: the caller declares the desired final state via req.pinned
    (rather than this endpoint toggling). Returns the updated project so the
    client can reconcile its view without re-fetching the full list.

    Used by the Bridge phone app's long-press pin gesture; the desktop
    sees the change on its next /projects fetch because pin state lives on
    the Project row itself, not in any per-client cache.
    """
    from app.memory.service import update_project
    from app.memory.schemas import ProjectUpdate

    project = update_project(db, project_id, ProjectUpdate(pinned=req.pinned))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return BridgeProjectOut(
        id=project.id,
        name=project.name,
        description=project.description,
        type=getattr(project, "type", None),
        created_at=project.created_at.isoformat() if project.created_at else "",
        updated_at=project.updated_at.isoformat() if project.updated_at else "",
        pinned=bool(getattr(project, "pinned", False)),
    )


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
    out: list[BridgeMessageOut] = []
    for m in messages:
        # Per-message artifact extraction. Cheap when content has no marker
        # (early-exits inside extract_artifacts). On marker-bearing messages
        # we strip the marker so the phone shows clean text plus the chip.
        stripped, refs = _process_artifacts(m.content)
        out.append(BridgeMessageOut(
            id=m.id,
            role=m.role,
            content=stripped,
            provider=m.provider if hasattr(m, "provider") else None,
            model=m.model if hasattr(m, "model") else None,
            created_at=m.created_at.isoformat() if m.created_at else "",
            attachments=refs,
        ))
    return out


@router.get("/artifacts/{kind}/{filename}")
async def bridge_get_artifact(
    kind: str,
    filename: str,
    _auth: bool = Depends(require_bridge_auth),
):
    """Stream a generated artifact (image, etc.) to the phone.

    Auth-gated download endpoint backing the chip's tap action. The kind
    and filename come from a marker emitted by some earlier turn (see
    app.bridge.artifacts.ARTIFACT_MARKER_RE for the syntax) — the phone
    constructs this URL from the BridgeArtifactRef the server returned.

    Safety: resolve_artifact_path enforces the safe-path check (charset
    re-validation + traversal block + "inside base dir" assertion +
    is_file). Any failure path returns 404 without revealing which check
    fired so attackers cannot probe the filesystem layout.

    Returns a streamed FileResponse so multi-MB images don't get pulled
    into memory; OkHttp on the phone handles the chunked download.
    """
    from app.bridge.artifacts import resolve_artifact_path
    path = resolve_artifact_path(kind, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    import mimetypes
    mime, _ = mimetypes.guess_type(str(path))
    return FileResponse(
        path=str(path),
        media_type=mime or "application/octet-stream",
        filename=filename,
    )


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
    # Phase 3: scan for identity facts (DOB, birthplace, location, etc.)
    # Fire-and-forget — writes to Tier 1 identity store if matched.
    from app.bridge.identity_hook import capture_from_bridge_message
    capture_from_bridge_message(req.message, source_label=f"bridge:{project.id}")

    # Phase 7: capture fragment for long-term associative memory.
    # Every message embedded + clustered into themes; decays over time.
    from app.bridge.identity_hook import capture_fragment_from_bridge
    capture_fragment_from_bridge(req.message, project_id=project.id)

    create_message(db, MessageCreate(
        project_id=project.id, role="user",
        content=req.message, provider="bridge", model="phone-input",
    ))

    history = list_messages(db, project.id, limit=20)
    history_messages = [
        {"role": m.role, "content": m.content}
        for m in history if m.role in ("user", "assistant")
    ]

    # Build-command parity (Phase 1): "build the apk" and its "yes"/"no" voice
    # confirmation are handled here, before translation, so the phone triggers
    # the same host build flow the desktop uses. The build runs async; the
    # completion is delivered (and spoken) via the /bridge/missed-replies poller.
    from app.bridge.build_actions import maybe_handle_build_turn
    build_turn = maybe_handle_build_turn(req.message, project.id, db)
    if build_turn is not None:
        bt_message = create_message(db, MessageCreate(
            project_id=project.id, role="assistant",
            content=build_turn.reply, provider="bridge", model=build_turn.model,
        ))
        return BridgeChatResponse(
            reply=build_turn.reply, project_id=project.id,
            project_name=project.name, domain="",
            message_id=bt_message.id,
        )

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
        gate_message = create_message(db, MessageCreate(
            project_id=project.id, role="assistant",
            content=reply, provider="bridge", model="capability-gate",
        ))
        return BridgeChatResponse(
            reply=reply, project_id=project.id, project_name=project.name,
            domain=domain_info.get("domain", "") if domain_info else "",
            message_id=gate_message.id,
        )

    web_search_context, search_executed, search_succeeded, early_reply = (
        await _run_web_search(resolved_intent, translation_result, req.message)
    )

    if early_reply:
        from app.memory.service import create_message as _cm
        early_message = _cm(db, MessageCreate(
            project_id=project.id, role="assistant",
            content=early_reply, provider="bridge", model="search-gate",
        ))
        return BridgeChatResponse(
            reply=early_reply, project_id=project.id, project_name=project.name,
            domain=domain_info.get("domain", "") if domain_info else "",
            message_id=early_message.id,
        )

    from app.bridge.capability_layer import run_astra_chat
    from app.bridge.attachment_describe import augment_user_message

    # Same attachment handling as bridge_chat_and_speak: prepend vision /
    # document text to the LLM-input message, keep req.message clean for
    # the DB record and identity hook. No-op when attachment_ids is empty.
    llm_input_message = augment_user_message(req, history_messages)

    result = await run_astra_chat(
        message=llm_input_message,
        project_id=project.id,
        history=history_messages,
        db=db,
        source="bridge",
        domain_context=domain_context,
        translation_result=translation_result,
        web_search_context=web_search_context,
        search_executed=search_executed,
        search_succeeded=search_succeeded,
        raw_message=req.message,
    )

    reply = result["reply"]
    provider = result["provider"]
    model = result["model"]

    assistant_message = create_message(db, MessageCreate(
        project_id=project.id, role="assistant",
        content=reply, provider=provider, model=model,
    ))

    # v2026-06-10: session + summary tracking for bridge conversations.
    # The hook previously only ran on the desktop stream path, so phone
    # chats never got conversation_sessions rows or rolling summaries.
    try:
        from app.memory.integration import record_session_activity
        record_session_activity(
            project_id=project.id, provider=provider,
            model=model, db_session=db,
        )
    except Exception as _sess_err:
        logger.debug("[bridge] session tracking failed: %s", _sess_err)

    _detected_domain = domain_info.get("domain") if domain_info else None
    if _detected_domain:
        push_desktop_navigation(_detected_domain)

    # Process artifact markers: strip from reply for clean display, build
    # BridgeArtifactRef list for the response. The DB still holds the raw
    # reply (with markers) so history reload can re-emit the same chips.
    display_reply, attachments = _process_artifacts(reply)

    # Phone-action directives ([[astra:...]] markers — see directives.py):
    # stripped from the visible reply, forwarded for the app to execute.
    from .directives import extract_directives, strip_directives
    directives = extract_directives(display_reply)
    if directives:
        display_reply = strip_directives(display_reply)

    return BridgeChatResponse(
        reply=display_reply, project_id=project.id,
        project_name=project.name, domain=_detected_domain,
        attachments=attachments,
        directives=directives,
        message_id=assistant_message.id,
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

    from . import tts_cache
    from .chat_speak_stream import build_audio_response, replay_cached_reply

    # Idempotent retry: a key we've already answered re-serves the original
    # reply (text headers + cached audio) — no second LLM run, no duplicate
    # user/assistant rows, no replay-ghost. Checked BEFORE any side effects.
    idempotency_key = request.headers.get("X-Idempotency-Key")
    if idempotency_key:
        replayed_id = tts_cache.idem_get(idempotency_key)
        if replayed_id is not None:
            # await (2026-06-13): replay now waits out an in-flight
            # disconnect-continuation instead of double-writing its .part.
            replayed = await replay_cached_reply(replayed_id, db, _process_artifacts)
            if replayed is not None:
                logger.info("[bridge] chat-and-speak idempotent replay (key=%s -> msg %s)",
                            idempotency_key[:12], replayed_id)
                return replayed

    project = _resolve_or_create_project(req, db)
    # Phase 3: scan for identity facts (DOB, birthplace, location, etc.)
    # Fire-and-forget — writes to Tier 1 identity store if matched.
    from app.bridge.identity_hook import capture_from_bridge_message
    capture_from_bridge_message(req.message, source_label=f"bridge:{project.id}")

    # Phase 7: capture fragment for long-term associative memory.
    # Every message embedded + clustered into themes; decays over time.
    from app.bridge.identity_hook import capture_fragment_from_bridge
    capture_fragment_from_bridge(req.message, project_id=project.id)

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

    # Build-command parity (Phase 1): handle "build the apk" + voice confirmation
    # here, before translation, so the phone triggers the same host build flow as
    # the desktop. The build runs async; completion arrives via /missed-replies.
    # The acknowledgement is spoken via the normal sentence-TTS streaming path.
    from app.bridge.build_actions import maybe_handle_build_turn
    build_turn = maybe_handle_build_turn(req.message, project.id, db)
    if build_turn is not None:
        bt_msg = create_message(db, MessageCreate(
            project_id=project.id, role="assistant",
            content=build_turn.reply, provider="bridge", model=build_turn.model,
        ))
        if idempotency_key:
            tts_cache.idem_put(idempotency_key, bt_msg.id)
        return build_audio_response(
            message_id=bt_msg.id,
            display_text=build_turn.reply,
            project_id=project.id,
            project_name=project.name,
            attachments_payload="[]",
        )

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

    if is_unsupported_on_bridge(resolved_intent):
        full_text = get_unsupported_message(resolved_intent)
        provider = "bridge"
        model = "capability-gate"
    elif honest_early_reply:
        full_text = honest_early_reply
        provider = "bridge"
        model = "search-gate"
    else:
        # Augment with image descriptions / document text if any attachments
        # were uploaded. req.message stays untouched so identity_hook,
        # translation, and the DB record all see the user's actual words.
        # See attachment_describe.augment_user_message for the contract;
        # the helper is shared with bridge_chat (text-only path) so both
        # endpoints have identical attachment handling.
        from app.bridge.attachment_describe import augment_user_message
        llm_input_message = augment_user_message(req, history_messages)

        result = await run_astra_chat(
            message=llm_input_message,
            project_id=project.id,
            history=history_messages,
            db=db,
            source="bridge-tts",
            domain_context=domain_context,
            translation_result=translation_result,
            web_search_context=web_search_context,
            search_executed=search_executed,
            search_succeeded=search_succeeded,
            raw_message=req.message,
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

    # Map the client's idempotency key to this reply so a retried request
    # (OkHttp resend after connection loss) replays instead of re-running.
    if idempotency_key:
        tts_cache.idem_put(idempotency_key, assistant_message.id)

    # v2026-06-10: session + summary tracking (see bridge_chat above).
    try:
        from app.memory.integration import record_session_activity
        record_session_activity(
            project_id=project_id, provider=provider,
            model=model, db_session=db,
        )
    except Exception as _sess_err:
        logger.debug("[bridge] session tracking failed: %s", _sess_err)

    # Process artifact markers. The DB row keeps the raw full_text
    # (with markers) so history reload re-emits the chips; the streaming
    # response uses display_text for the X-Full-Text header and TTS so
    # the marker syntax never reaches the user. attachments_payload is
    # JSON-serialised into the X-Artifacts header for the phone to parse.
    import json
    display_text, attachments = _process_artifacts(full_text)
    attachments_payload = json.dumps([a.model_dump() for a in attachments])

    # Phone-action directives ([[astra:...]] — see directives.py): stripped
    # before the text reaches the screen or the voice, sent via X-Directives.
    from .directives import extract_directives, strip_directives, directives_payload
    directives = extract_directives(display_text)
    if directives:
        display_text = strip_directives(display_text)

    # Stream sentence-MP3s to the phone while teeing them into the
    # per-message tts_cache; survives client disconnects (see
    # chat_speak_stream.build_audio_response for the full contract).
    return build_audio_response(
        message_id=assistant_message.id,
        display_text=display_text,
        project_id=project_id,
        project_name=project_name,
        attachments_payload=attachments_payload,
        directives_json=directives_payload(directives),
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
    auth=Depends(require_bridge_auth),
):
    """Accept a raw file upload from the Astra Bridge OutboundQueueDrainer.

    Bridge posts raw bytes with:
      - Authorization: Bearer <token>
      - X-Idempotency-Key: <uuid>   (dedup on resend after transient failure)
      - Content-Type: image/jpeg etc. (used to pick the saved-file extension)

    Returns 200 {id, bytes, path} on success; 409 on duplicate idempotency
    key; 401/403 from require_bridge_auth on bad auth.

    Files land under the user's Pictures folder (see uploads_store.UPLOADS_ROOT)
    with their real extension, tracked by a shared JSONL ledger. All path
    resolution and writes are delegated to app.bridge.uploads_store so the
    endpoint stays small and the destination is easy to change.

    v2.0 (2026-05-24): delegated to uploads_store; destination changed from
    D:/Orb/uploads/bridge/ to OneDrive\\Pictures\\Astra Mobile Uploads so
    files are visible to the user and indexed by the drive watcher.
    v1.0 (2026-04-13): initial implementation so Bridge attachments actually
    reach the server. Previously Bridge POSTed to /upload (404).
    """
    import uuid

    idempotency_key = request.headers.get("X-Idempotency-Key", "").strip()
    if not idempotency_key:
        # Generate one so the client can still succeed if the header was missing.
        idempotency_key = uuid.uuid4().hex

    if is_duplicate(idempotency_key):
        logger.info("[bridge] upload duplicate for key=%s", idempotency_key[:12])
        raise HTTPException(status_code=409, detail="duplicate")

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty body")

    record = save_upload(
        body=body,
        content_type=request.headers.get("Content-Type", ""),
        idempotency_key=idempotency_key,
        original_filename=request.headers.get("X-Original-Filename", "").strip(),
    )

    logger.info(
        "[bridge] upload received: id=%s bytes=%d key=%s path=%s",
        record.id, record.bytes, idempotency_key[:12], record.path,
    )

    return {"id": record.id, "bytes": record.bytes, "path": record.path}


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
