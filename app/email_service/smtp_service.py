# FILE: app/email_service/smtp_service.py
# Purpose: Proton Mail SMTP service — sends email via Proton Bridge.
# Called-by: app.email_service.router
# Depends-on: app.settings.service
# Last-renovated: 2026-06-11
"""
Proton Mail SMTP service — sends email via Proton Bridge.

Proton Bridge exposes SMTP on 127.0.0.1:1025 with STARTTLS.
Uses the same Bridge password as IMAP.

v1.0 (2026-03-14): Initial — send plain text and HTML emails.
"""
from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SMTP_HOST = os.getenv("PROTON_BRIDGE_SMTP_HOST", "127.0.0.1")
SMTP_PORT = int(os.getenv("PROTON_BRIDGE_SMTP_PORT", "1025"))


def _get_credentials() -> tuple[str, str]:
    """Get email credentials from environment or settings DB."""
    username = os.getenv("PROTON_EMAIL", "")
    password = os.getenv("PROTON_BRIDGE_PASSWORD", "")
    if not username or not password:
        try:
            from app.settings.service import get_setting_value
            username = username or get_setting_value("proton_email") or ""
            password = password or get_setting_value("proton_bridge_password") or ""
        except Exception:
            pass
    return username, password


def send_email(
    to: str | List[str],
    subject: str,
    body_text: str = "",
    body_html: str = "",
    cc: Optional[List[str]] = None,
    bcc: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Send an email via Proton Bridge SMTP.

    Args:
        to: Recipient(s) — single address or list.
        subject: Email subject line.
        body_text: Plain text body.
        body_html: HTML body (optional, sent as alternative).
        cc: CC recipients.
        bcc: BCC recipients.

    Returns dict with success status and message_id or error.
    """
    username, password = _get_credentials()
    if not username or not password:
        return {
            "success": False,
            "error": "Proton Bridge credentials not configured.",
        }

    # Normalise recipients
    if isinstance(to, str):
        to = [to]
    all_recipients = list(to) + (cc or []) + (bcc or [])

    # Build message
    msg = MIMEMultipart("alternative")
    msg["From"] = username
    msg["To"] = ", ".join(to)
    msg["Subject"] = subject
    if cc:
        msg["Cc"] = ", ".join(cc)

    if body_text:
        msg.attach(MIMEText(body_text, "plain", "utf-8"))
    if body_html:
        msg.attach(MIMEText(body_html, "html", "utf-8"))
    elif not body_text:
        msg.attach(MIMEText("", "plain", "utf-8"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(username, password)
            server.sendmail(username, all_recipients, msg.as_string())

        logger.info("[email] Sent to %s: %s", ", ".join(to), subject)
        return {
            "success": True,
            "to": to,
            "subject": subject,
            "message": f"Email sent to {', '.join(to)}",
        }
    except smtplib.SMTPAuthenticationError as e:
        return {"success": False, "error": f"SMTP auth failed: {e}"}
    except OSError as e:
        return {
            "success": False,
            "error": f"Cannot connect to Proton Bridge SMTP at {SMTP_HOST}:{SMTP_PORT}. "
                     f"Is Bridge running? {e}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
