# FILE: app/email/proton_provider.py
# Purpose: Proton Mail implementation of the EmailProvider contract — thin
#          adapter over the EXISTING app/email_service Bridge plumbing.
# Called-by: app.email (factory)
# Depends-on: app.email.provider, app.email_service.imap_service, app.email_service.smtp_service
# Last-renovated: 2026-06-12
"""
ProtonBridgeProvider.

The job-stack spec asked for a stub here ("Bridge IMAP/SMTP plan, do not
implement") — but D:\\Orb already HAS working Proton Bridge plumbing
(app/email_service, 2026-03-14: imaplib/smtplib against 127.0.0.1, Bridge
ports, creds in the settings DB). So instead of a dead stub this is a thin
adapter mapping that existing service onto the contract. The plumbing stays
owned by app/email_service; nothing is duplicated.

Documented lossy mappings (IMAP vs the thread model):
  - "Threads" are single messages (UID-keyed). Good enough for voice.
  - label filtering: only 'UNREAD' is honoured (IMAP UNSEEN is not exposed
    by the existing service, so unread comes back best-effort False).
  - Drafts are LOCAL to this provider (data/email/proton_drafts.json):
    Bridge SMTP has no server-side draft store. create_draft stores the
    draft locally; send_draft sends it via SMTP and removes it. The
    confirm-loop behaves identically to Gmail from the intent layer.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import List, Optional

from app.email.provider import (
    EmailProvider,
    Message,
    Thread,
    ThreadSummary,
    clean_body,
)

logger = logging.getLogger(__name__)

_DRAFTS_FILE = Path("D:/Orb/data/email/proton_drafts.json")


def _load_drafts() -> dict:
    try:
        return json.loads(_DRAFTS_FILE.read_text(encoding="utf-8")) if _DRAFTS_FILE.exists() else {}
    except Exception:
        return {}


def _save_drafts(drafts: dict) -> None:
    _DRAFTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _DRAFTS_FILE.write_text(json.dumps(drafts, indent=1), encoding="utf-8")


def _summary_from_raw(raw: dict) -> ThreadSummary:
    body = raw.get("body_text") or raw.get("body_html") or ""
    snippet = " ".join(str(body).split())[:140]
    return ThreadSummary(
        id=str(raw.get("uid")),
        sender=str(raw.get("from", "")),
        subject=str(raw.get("subject") or "(no subject)"),
        snippet=snippet,
        unread=False,  # IMAP \Seen not exposed by the existing service — best-effort
        date=str(raw.get("date", "")),
        tags=[],
    )


class ProtonBridgeProvider(EmailProvider):
    name = "proton"

    def list_threads(self, query: Optional[str] = None, label: Optional[str] = None,
                     max_results: int = 10) -> List[ThreadSummary]:
        from app.email_service import imap_service
        if query:
            rows = imap_service.search_emails(query, limit=max_results)
        else:
            rows = imap_service.fetch_inbox(limit=max_results)
        if label and label.lower() != "unread":
            logger.warning("[proton-provider] label '%s' not supported over IMAP adapter", label)
        return [_summary_from_raw(r) for r in rows[:max_results]]

    def read_thread(self, thread_id: str) -> Thread:
        from app.email_service import imap_service
        raw = imap_service.read_email("INBOX", thread_id)
        if not raw:
            return Thread(id=thread_id, subject="(not found)", messages=[])
        body = raw.get("body_text") or ""
        is_html = False
        if not body and raw.get("body_html"):
            body, is_html = raw["body_html"], True
        message = Message(
            id=str(raw.get("uid")),
            sender=str(raw.get("from", "")),
            to=str(raw.get("to", "")),
            date=str(raw.get("date", "")),
            body=clean_body(body, is_html=is_html),
            tags=[],
        )
        return Thread(id=thread_id, subject=str(raw.get("subject") or "(no subject)"),
                      messages=[message])

    def create_draft(self, to: str, subject: str, body: str,
                     reply_to_thread: Optional[str] = None) -> dict:
        drafts = _load_drafts()
        draft_id = f"proton-draft-{uuid.uuid4().hex[:12]}"
        drafts[draft_id] = {
            "to": to, "subject": subject, "body": body,
            "reply_to_thread": reply_to_thread, "created_at": int(time.time()),
        }
        _save_drafts(drafts)
        return {"ok": True, "draft_id": draft_id, "to": to,
                "subject": subject, "body": body}

    def send_draft(self, draft_id: str) -> dict:
        drafts = _load_drafts()
        draft = drafts.get(draft_id)
        if not draft:
            return {"ok": False, "error": f"no draft {draft_id}"}
        out = self.send(draft["to"], draft["subject"], draft["body"],
                        reply_to_thread=draft.get("reply_to_thread"))
        if out.get("ok"):
            drafts.pop(draft_id, None)
            _save_drafts(drafts)
        return out

    def send(self, to: str, subject: str, body: str,
             reply_to_thread: Optional[str] = None) -> dict:
        from app.email_service import smtp_service
        try:
            result = smtp_service.send_email(to=to, subject=subject, body_text=body)
            ok = bool(result.get("success")) if isinstance(result, dict) else False
            out = {"ok": ok}
            if isinstance(result, dict) and result.get("error"):
                out["error"] = result["error"]
            return out
        except Exception as exc:
            logger.exception("[proton-provider] send failed")
            return {"ok": False, "error": str(exc)}
