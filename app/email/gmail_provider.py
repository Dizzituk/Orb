# FILE: app/email/gmail_provider.py
# Purpose: Gmail implementation of the EmailProvider contract (Gmail API v1,
#          threads/drafts/send with proper reply headers).
# Called-by: app.email (factory)
# Depends-on: app.email.provider, app.email.gmail_auth, google-api-python-client
# Last-renovated: 2026-06-12
"""
GmailProvider.

Maps the generic contract onto the Gmail API:
  list_threads  -> users.threads.list + per-thread metadata get
  read_thread   -> users.threads.get(format=full), bodies -> plain text
  create_draft  -> users.drafts.create (reply headers + threadId when replying)
  send_draft    -> users.drafts.send
  send          -> users.messages.send
Labels surface as generic tags; 'UNREAD' is the cross-provider one.
"""
from __future__ import annotations

import base64
import logging
from email.message import EmailMessage
from typing import List, Optional

from app.email.provider import (
    EmailProvider,
    Message,
    Thread,
    ThreadSummary,
    clean_body,
)

logger = logging.getLogger(__name__)


class GmailNotAuthenticated(Exception):
    """Raised when no valid OAuth token exists — tool layer reports setup steps."""


class GmailProvider(EmailProvider):
    name = "gmail"

    # ── plumbing ────────────────────────────────────────────────────────

    def _service(self):
        from app.email.gmail_auth import get_gmail_credentials
        creds = get_gmail_credentials()
        if not creds:
            raise GmailNotAuthenticated(
                "Gmail isn't connected — run the Gmail setup (setup_email_auth) first."
            )
        from googleapiclient.discovery import build
        return build("gmail", "v1", credentials=creds, cache_discovery=False)

    @staticmethod
    def _header(headers: List[dict], name: str) -> str:
        for h in headers or []:
            if str(h.get("name", "")).lower() == name.lower():
                return str(h.get("value", ""))
        return ""

    @staticmethod
    def _walk_for_body(payload: dict) -> tuple[str, bool]:
        """Prefer text/plain anywhere in the part tree; fall back to text/html."""
        plain, html = "", ""

        def walk(part: dict):
            nonlocal plain, html
            mime = str(part.get("mimeType", ""))
            data = (part.get("body") or {}).get("data")
            if data and mime == "text/plain" and not plain:
                plain = _b64url_decode(data)
            elif data and mime == "text/html" and not html:
                html = _b64url_decode(data)
            for child in part.get("parts") or []:
                walk(child)

        walk(payload or {})
        if plain:
            return plain, False
        return html, True

    def _build_mime(self, to: str, subject: str, body: str,
                    reply_headers: Optional[dict] = None) -> EmailMessage:
        msg = EmailMessage()
        msg["To"] = to
        msg["Subject"] = subject
        if reply_headers:
            if reply_headers.get("message_id"):
                msg["In-Reply-To"] = reply_headers["message_id"]
                msg["References"] = (
                    (reply_headers.get("references", "") + " " + reply_headers["message_id"]).strip()
                )
        msg.set_content(body)
        return msg

    def _reply_context(self, service, thread_id: str) -> dict:
        """Last message's Message-ID/References/Subject for clean reply threading."""
        thread = service.users().threads().get(
            userId="me", id=thread_id, format="metadata",
            metadataHeaders=["Message-ID", "References", "Subject", "From", "Reply-To"],
        ).execute()
        messages = thread.get("messages") or []
        if not messages:
            return {}
        headers = (messages[-1].get("payload") or {}).get("headers") or []
        subject = self._header(headers, "Subject")
        if subject and not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"
        return {
            "message_id": self._header(headers, "Message-ID"),
            "references": self._header(headers, "References"),
            "subject": subject,
            "reply_to": self._header(headers, "Reply-To") or self._header(headers, "From"),
        }

    @staticmethod
    def _raw(msg: EmailMessage) -> str:
        return base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")

    # ── contract ────────────────────────────────────────────────────────

    def list_threads(self, query: Optional[str] = None, label: Optional[str] = None,
                     max_results: int = 10) -> List[ThreadSummary]:
        service = self._service()
        kwargs: dict = {"userId": "me", "maxResults": max_results}
        if query:
            kwargs["q"] = query
        if label:
            kwargs["labelIds"] = [label.upper() if label.lower() == "unread" else label]
        listing = service.users().threads().list(**kwargs).execute()
        out: List[ThreadSummary] = []
        for row in (listing.get("threads") or [])[:max_results]:
            t = service.users().threads().get(
                userId="me", id=row["id"], format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            ).execute()
            messages = t.get("messages") or []
            if not messages:
                continue
            first, last = messages[0], messages[-1]
            headers = (first.get("payload") or {}).get("headers") or []
            label_ids = {lab for m in messages for lab in (m.get("labelIds") or [])}
            out.append(ThreadSummary(
                id=str(t.get("id")),
                sender=self._header(headers, "From"),
                subject=self._header(headers, "Subject") or "(no subject)",
                snippet=str(last.get("snippet") or ""),
                unread="UNREAD" in label_ids,
                date=self._header(
                    ((last.get("payload") or {}).get("headers") or headers), "Date"),
                tags=sorted(label_ids),
            ))
        return out

    def read_thread(self, thread_id: str) -> Thread:
        service = self._service()
        t = service.users().threads().get(
            userId="me", id=thread_id, format="full"
        ).execute()
        messages: List[Message] = []
        subject = ""
        for m in t.get("messages") or []:
            payload = m.get("payload") or {}
            headers = payload.get("headers") or []
            subject = subject or self._header(headers, "Subject")
            raw_body, is_html = self._walk_for_body(payload)
            messages.append(Message(
                id=str(m.get("id")),
                sender=self._header(headers, "From"),
                to=self._header(headers, "To"),
                date=self._header(headers, "Date"),
                body=clean_body(raw_body, is_html=is_html),
                tags=sorted(m.get("labelIds") or []),
            ))
        return Thread(id=str(t.get("id")), subject=subject or "(no subject)",
                      messages=messages)

    def create_draft(self, to: str, subject: str, body: str,
                     reply_to_thread: Optional[str] = None) -> dict:
        service = self._service()
        reply_headers = None
        thread_id = None
        if reply_to_thread:
            ctx = self._reply_context(service, reply_to_thread)
            reply_headers = ctx
            thread_id = reply_to_thread
            subject = subject or ctx.get("subject") or ""
            to = to or ctx.get("reply_to") or ""
        msg = self._build_mime(to, subject, body, reply_headers)
        payload: dict = {"message": {"raw": self._raw(msg)}}
        if thread_id:
            payload["message"]["threadId"] = thread_id
        draft = service.users().drafts().create(userId="me", body=payload).execute()
        return {"ok": True, "draft_id": str(draft.get("id")),
                "to": to, "subject": subject, "body": body}

    def send_draft(self, draft_id: str) -> dict:
        service = self._service()
        sent = service.users().drafts().send(
            userId="me", body={"id": draft_id}
        ).execute()
        return {"ok": True, "id": str(sent.get("id"))}

    def send(self, to: str, subject: str, body: str,
             reply_to_thread: Optional[str] = None) -> dict:
        service = self._service()
        reply_headers = None
        payload: dict = {}
        if reply_to_thread:
            ctx = self._reply_context(service, reply_to_thread)
            reply_headers = ctx
            subject = subject or ctx.get("subject") or ""
            to = to or ctx.get("reply_to") or ""
            payload["threadId"] = reply_to_thread
        msg = self._build_mime(to, subject, body, reply_headers)
        payload["raw"] = self._raw(msg)
        sent = service.users().messages().send(userId="me", body=payload).execute()
        return {"ok": True, "id": str(sent.get("id"))}


def _b64url_decode(data: str) -> str:
    try:
        return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode(
            "utf-8", errors="replace")
    except Exception:
        return ""
