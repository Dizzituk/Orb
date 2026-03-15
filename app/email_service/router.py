# FILE: app/email_service/router.py
"""
Email API endpoints — Proton Mail via Proton Bridge (IMAP/SMTP).

Endpoints:
  GET  /email/status         — Check Bridge connection + credentials
  GET  /email/folders        — List mailbox folders
  GET  /email/inbox          — Fetch recent emails
  GET  /email/read           — Read a single email by UID
  GET  /email/search         — Search emails
  POST /email/send           — Send an email

v1.0 (2026-03-14): Initial implementation.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.auth import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/email", tags=["Email"], dependencies=[Depends(require_auth)])


# ── Request models ──

class SendEmailRequest(BaseModel):
    to: str | List[str]
    subject: str
    body_text: str = ""
    body_html: str = ""
    cc: Optional[List[str]] = None
    bcc: Optional[List[str]] = None


# ── Endpoints ──

@router.get("/status")
def email_status():
    """Check if Proton Bridge is connected and credentials are configured."""
    from app.email_service.imap_service import check_connection
    return check_connection()


@router.get("/folders")
def email_folders():
    """List available email folders."""
    from app.email_service.imap_service import list_folders
    try:
        folders = list_folders()
        return {"folders": folders}
    except Exception as e:
        return {"error": str(e)}


@router.get("/inbox")
def email_inbox(
    folder: str = Query("INBOX", description="Mailbox folder"),
    limit: int = Query(20, ge=1, le=100),
    since_days: Optional[int] = Query(None, description="Only emails from last N days"),
):
    """Fetch recent emails from a folder."""
    from app.email_service.imap_service import fetch_inbox
    try:
        emails = fetch_inbox(folder=folder, limit=limit, since_days=since_days)
        return {"folder": folder, "count": len(emails), "emails": emails}
    except Exception as e:
        return {"error": str(e)}


@router.get("/read")
def email_read(
    uid: str = Query(..., description="Email UID"),
    folder: str = Query("INBOX"),
):
    """Read a single email with full body."""
    from app.email_service.imap_service import read_email
    try:
        result = read_email(folder, uid)
        if result:
            return result
        return {"error": "Email not found"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/search")
def email_search(
    q: str = Query(..., description="Search query"),
    folder: str = Query("INBOX"),
    limit: int = Query(20, ge=1, le=100),
):
    """Search emails by subject or body content."""
    from app.email_service.imap_service import search_emails
    try:
        results = search_emails(query=q, folder=folder, limit=limit)
        return {"query": q, "folder": folder, "count": len(results), "emails": results}
    except Exception as e:
        return {"error": str(e)}


@router.post("/send")
def email_send(req: SendEmailRequest):
    """Send an email via Proton Bridge SMTP."""
    from app.email_service.smtp_service import send_email
    return send_email(
        to=req.to,
        subject=req.subject,
        body_text=req.body_text,
        body_html=req.body_html,
        cc=req.cc,
        bcc=req.bcc,
    )
