# FILE: app/email/gmail_auth.py
# Purpose: Gmail OAuth2 — installed-app flow with local callback (port 8091),
#          token cached in data/email/gmail_token.json.
# Called-by: app.email.gmail_provider, app.tools.email_tools (setup/status)
# Depends-on: google-auth, google-auth-oauthlib (already in venv from Drive/YouTube work)
# Last-renovated: 2026-06-12
"""
Gmail OAuth2 — mirrors the house pattern (gdrive_auth 8089, youtube_auth 8090;
Gmail takes 8091). Client id/secret from env GMAIL_CLIENT_ID/GMAIL_CLIENT_SECRET
with the settings-DB fallback used by the Proton service. Scopes cover read,
compose (drafts) and send — no broader.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

TOKEN_PATH = Path("D:/Orb/data/email/gmail_token.json")
REDIRECT_PORT = 8091

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
]


def _get_oauth_client() -> Tuple[Optional[str], Optional[str]]:
    """Client id/secret from env, falling back to the settings DB."""
    client_id = os.getenv("GMAIL_CLIENT_ID")
    client_secret = os.getenv("GMAIL_CLIENT_SECRET")
    if not client_id or not client_secret:
        try:
            from app.settings.service import get_setting_value
            client_id = client_id or get_setting_value("gmail_client_id") or None
            client_secret = client_secret or get_setting_value("gmail_client_secret") or None
        except Exception:
            pass
    return client_id, client_secret


def get_gmail_credentials():
    """Authenticated google Credentials (auto-refresh) or None."""
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
    except ImportError:
        logger.error("[gmail-auth] google-auth not installed")
        return None

    if not TOKEN_PATH.exists():
        return None
    try:
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    except Exception as exc:
        logger.warning("[gmail-auth] failed to load token: %s", exc)
        return None

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_token(creds)
        except Exception as exc:
            logger.warning("[gmail-auth] token refresh failed: %s", exc)
            return None
    return creds if creds and creds.valid else None


def check_auth_status() -> dict:
    """Is Gmail connected? Includes setup hints when it isn't."""
    client_id, _ = _get_oauth_client()
    creds = get_gmail_credentials()
    if creds:
        try:
            from googleapiclient.discovery import build
            service = build("gmail", "v1", credentials=creds, cache_discovery=False)
            profile = service.users().getProfile(userId="me").execute()
            return {
                "authenticated": True,
                "email_address": profile.get("emailAddress"),
                "has_client_id": bool(client_id),
            }
        except Exception as exc:
            logger.warning("[gmail-auth] auth check failed: %s", exc)
            return {"authenticated": False, "error": str(exc),
                    "has_client_id": bool(client_id)}
    return {
        "authenticated": False,
        "has_client_id": bool(client_id),
        "has_token": TOKEN_PATH.exists(),
        "needs_setup": not bool(client_id),
    }


def start_auth_flow() -> dict:
    """Open the Google consent screen in the system browser; callback on 8091."""
    client_id, client_secret = _get_oauth_client()
    if not client_id or not client_secret:
        return {
            "error": "no_credentials",
            "message": (
                "Gmail OAuth credentials not configured. Create an OAuth "
                "client (Desktop app) in Google Cloud Console and set "
                "GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET."
            ),
        }
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        return {"error": "missing_package", "message": "google-auth-oauthlib not installed"}

    flow = InstalledAppFlow.from_client_config(
        {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [f"http://127.0.0.1:{REDIRECT_PORT}"],
            }
        },
        scopes=SCOPES,
    )

    def _run_flow():
        try:
            creds = flow.run_local_server(
                port=REDIRECT_PORT,
                prompt="consent",
                access_type="offline",
                open_browser=True,
            )
            _save_token(creds)
            logger.info("[gmail-auth] OAuth complete — Gmail connected")
        except Exception as exc:
            logger.error("[gmail-auth] OAuth flow failed: %s", exc)

    threading.Thread(target=_run_flow, daemon=True).start()
    return {
        "started": True,
        "message": "Google sign-in opened in the browser — sign in and allow Gmail access.",
    }


def revoke_auth() -> dict:
    if TOKEN_PATH.exists():
        TOKEN_PATH.unlink()
        logger.info("[gmail-auth] token revoked")
    return {"revoked": True}


def _save_token(creds) -> None:
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
    logger.info("[gmail-auth] token saved to %s", TOKEN_PATH)
