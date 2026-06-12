# FILE: app/email_service/imap_service.py
# Purpose: Proton Mail IMAP service — reads email via Proton Bridge.
# Called-by: app.email_service.router
# Depends-on: app.settings.service
# Last-renovated: 2026-06-11
"""
Proton Mail IMAP service — reads email via Proton Bridge.

Proton Bridge runs locally and exposes IMAP on 127.0.0.1:1143.
Requires: Proton Bridge installed + logged in + Bridge password configured.

ASTRA stores the Bridge password in settings (encrypted in DB).

v1.0 (2026-03-14): Initial — fetch inbox, read email, search, folders.
"""
from __future__ import annotations

import email
import imaplib
import logging
import os
from datetime import datetime, timedelta
from email.header import decode_header
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Proton Bridge defaults
IMAP_HOST = os.getenv("PROTON_BRIDGE_IMAP_HOST", "127.0.0.1")
IMAP_PORT = int(os.getenv("PROTON_BRIDGE_IMAP_PORT", "1143"))


def _get_credentials() -> tuple[str, str]:
    """Get email credentials from environment or settings DB."""
    username = os.getenv("PROTON_EMAIL", "")
    password = os.getenv("PROTON_BRIDGE_PASSWORD", "")

    if not username or not password:
        # Try loading from ASTRA settings
        try:
            from app.settings.service import get_setting_value
            username = username or get_setting_value("proton_email") or ""
            password = password or get_setting_value("proton_bridge_password") or ""
        except Exception:
            pass

    return username, password


def _connect() -> imaplib.IMAP4:
    """Connect to Proton Bridge IMAP server."""
    username, password = _get_credentials()
    if not username or not password:
        raise ConnectionError(
            "Proton Bridge credentials not configured. "
            "Set PROTON_EMAIL and PROTON_BRIDGE_PASSWORD in Settings."
        )

    try:
        # Proton Bridge uses STARTTLS on port 1143
        conn = imaplib.IMAP4(IMAP_HOST, IMAP_PORT)
        conn.starttls()
        conn.login(username, password)
        return conn
    except imaplib.IMAP4.error as e:
        raise ConnectionError(f"IMAP login failed: {e}")
    except OSError as e:
        raise ConnectionError(
            f"Cannot connect to Proton Bridge at {IMAP_HOST}:{IMAP_PORT}. "
            f"Is Proton Bridge running? Error: {e}"
        )


def _decode_header_value(value: str) -> str:
    """Decode a MIME-encoded email header."""
    if not value:
        return ""
    decoded_parts = decode_header(value)
    result = []
    for part, charset in decoded_parts:
        if isinstance(part, bytes):
            result.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            result.append(str(part))
    return " ".join(result)


def _parse_email(msg_data: bytes, uid: str) -> Dict[str, Any]:
    """Parse a raw email into a structured dict."""
    msg = email.message_from_bytes(msg_data)

    # Extract body
    body_text = ""
    body_html = ""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain" and not body_text:
                payload = part.get_payload(decode=True)
                if payload:
                    body_text = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
            elif ct == "text/html" and not body_html:
                payload = part.get_payload(decode=True)
                if payload:
                    body_html = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            ct = msg.get_content_type()
            text = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
            if ct == "text/html":
                body_html = text
            else:
                body_text = text

    # Parse date
    date_str = msg.get("Date", "")
    parsed_date = ""
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(date_str)
        parsed_date = dt.isoformat()
    except Exception:
        parsed_date = date_str

    return {
        "uid": uid,
        "subject": _decode_header_value(msg.get("Subject", "")),
        "from": _decode_header_value(msg.get("From", "")),
        "to": _decode_header_value(msg.get("To", "")),
        "date": parsed_date,
        "date_raw": date_str,
        "body_text": body_text[:5000],  # Truncate for safety
        "body_html": body_html[:10000],
        "has_attachments": any(
            part.get_content_disposition() == "attachment"
            for part in (msg.walk() if msg.is_multipart() else [])
        ),
    }


def check_connection() -> Dict[str, Any]:
    """Check if Proton Bridge is reachable and credentials work."""
    try:
        conn = _connect()
        # Get mailbox list
        status, folders = conn.list()
        folder_names = []
        if status == "OK":
            for f in folders:
                if isinstance(f, bytes):
                    parts = f.decode("utf-8", errors="replace").split(' "/" ')
                    if len(parts) >= 2:
                        folder_names.append(parts[-1].strip('"'))
        conn.logout()
        return {
            "connected": True,
            "host": IMAP_HOST,
            "port": IMAP_PORT,
            "folders": folder_names,
            "message": "Connected to Proton Mail via Bridge",
        }
    except ConnectionError as e:
        return {"connected": False, "error": str(e)}
    except Exception as e:
        return {"connected": False, "error": f"Unexpected error: {e}"}


def fetch_inbox(
    folder: str = "INBOX",
    limit: int = 20,
    since_days: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch recent emails from a folder.

    Returns list of email dicts with subject, from, date, body preview.
    """
    conn = _connect()
    try:
        status, _ = conn.select(folder, readonly=True)
        if status != "OK":
            return []

        # Build search criteria
        criteria = "ALL"
        if since_days:
            since = (datetime.now() - timedelta(days=since_days)).strftime("%d-%b-%Y")
            criteria = f'(SINCE {since})'

        status, data = conn.search(None, criteria)
        if status != "OK":
            return []

        msg_ids = data[0].split()
        # Take the most recent N
        recent_ids = msg_ids[-limit:] if len(msg_ids) > limit else msg_ids
        recent_ids.reverse()  # Newest first

        emails = []
        for msg_id in recent_ids:
            status, msg_data = conn.fetch(msg_id, "(UID RFC822)")
            if status != "OK" or not msg_data or not msg_data[0]:
                continue

            # Extract UID
            uid = msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)
            if isinstance(msg_data[0], tuple) and len(msg_data[0]) >= 2:
                raw = msg_data[0][1]
                # Try to extract UID from response
                uid_line = msg_data[0][0]
                if isinstance(uid_line, bytes) and b"UID" in uid_line:
                    try:
                        uid = uid_line.decode().split("UID")[1].split()[0].strip()
                    except Exception:
                        pass
                emails.append(_parse_email(raw, uid))

        return emails
    finally:
        conn.logout()


def read_email(folder: str, uid: str) -> Optional[Dict[str, Any]]:
    """Read a single email by UID with full body."""
    conn = _connect()
    try:
        conn.select(folder, readonly=True)
        status, data = conn.uid("fetch", uid, "(RFC822)")
        if status != "OK" or not data or not data[0]:
            return None
        if isinstance(data[0], tuple) and len(data[0]) >= 2:
            return _parse_email(data[0][1], uid)
        return None
    finally:
        conn.logout()


def search_emails(
    query: str,
    folder: str = "INBOX",
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Search emails by subject or body content."""
    conn = _connect()
    try:
        conn.select(folder, readonly=True)

        # IMAP search — try subject first, then body
        criteria = f'(OR SUBJECT "{query}" BODY "{query}")'
        status, data = conn.search(None, criteria)
        if status != "OK":
            return []

        msg_ids = data[0].split()
        recent_ids = msg_ids[-limit:] if len(msg_ids) > limit else msg_ids
        recent_ids.reverse()

        emails = []
        for msg_id in recent_ids:
            status, msg_data = conn.fetch(msg_id, "(UID RFC822)")
            if status == "OK" and msg_data and msg_data[0]:
                uid = msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)
                if isinstance(msg_data[0], tuple) and len(msg_data[0]) >= 2:
                    emails.append(_parse_email(msg_data[0][1], uid))

        return emails
    finally:
        conn.logout()


def list_folders() -> List[str]:
    """List all available email folders."""
    conn = _connect()
    try:
        status, folders = conn.list()
        names = []
        if status == "OK":
            for f in folders:
                if isinstance(f, bytes):
                    parts = f.decode("utf-8", errors="replace").split(' "/" ')
                    if len(parts) >= 2:
                        names.append(parts[-1].strip('"'))
        return names
    finally:
        conn.logout()
