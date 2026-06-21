# FILE: app/bridge/router.py
"""
Astra Bridge API - endpoints for the Android companion app.

v16.0 (2026-06-21): batch-3 split - the 4 shared chat-pipeline helpers moved
    to chat_helpers.py (pure leaf, re-exported below so lazy importers in
    chat_speak_stream.py / missed_replies.py resolve unchanged); the /chat body
    moved to chat_endpoints.run_bridge_chat (lazy-delegated, like /chat-and-speak).
v15.0 (2026-04-06): modularised into schemas.py / llm_helpers.py / tts_proxy.py / router.py.
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

# Shared chat-pipeline helpers were extracted to chat_helpers.py (batch 3,
# 2026-06-21); re-export them so `from .router import <helper>` keeps resolving
# for chat_speak_stream.py and missed_replies.py (lazy importers) with zero edits.
from .chat_helpers import (
    _process_artifacts,
    _resolve_or_create_project,
    _run_translation,
    _run_web_search,
)


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
    # Body lives in chat_endpoints.run_bridge_chat (batch-3 split). The lazy
    # import mirrors bridge_chat_and_speak's delegation to chat_speak_stream,
    # so there is no router<->chat_endpoints module-load cycle.
    from .chat_endpoints import run_bridge_chat
    return await run_bridge_chat(req, db)


@router.post("/chat-and-speak")
async def bridge_chat_and_speak(
    req: BridgeChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
):
    # Body lives in chat_speak_stream.run_chat_and_speak so the in-flight
    # idempotency claim releases in a finally without re-indenting this module.
    from .chat_speak_stream import run_chat_and_speak
    return await run_chat_and_speak(req, request, db)


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
