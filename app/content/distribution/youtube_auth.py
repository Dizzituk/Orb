# FILE: app/content/distribution/youtube_auth.py
"""
YouTube OAuth2 — in-app flow for Electron desktop.

Mirrors the Google Drive auth pattern (drive_auth_service.py)
but uses YouTube-specific scopes and a separate token file.

Flow:
1. Backend generates auth URL
2. Frontend opens it in system browser
3. User logs into Google, clicks Allow
4. Google redirects to http://127.0.0.1:8090
5. Backend catches the code, exchanges for tokens, stores them
6. Frontend polls /content/youtube/auth/status until authenticated

Port 8090 chosen to avoid conflict with Drive OAuth (8089).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

TOKEN_PATH = Path("D:/Orb/data/content/youtube_token.json")
REDIRECT_PORT = 8090

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]


def _get_oauth_client() -> Tuple[Optional[str], Optional[str]]:
    """Get YouTube OAuth client ID and secret from environment."""
    client_id = os.getenv("YOUTUBE_CLIENT_ID")
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET")
    return client_id, client_secret


def get_youtube_credentials():
    """Get authenticated YouTube credentials object.

    Returns a google.oauth2.credentials.Credentials instance
    or None if not authenticated.
    """
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
    except ImportError:
        logger.error("[youtube-auth] google-auth not installed")
        return None

    if not TOKEN_PATH.exists():
        return None

    try:
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    except Exception as e:
        logger.warning("[youtube-auth] Failed to load token: %s", e)
        return None

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_token(creds)
        except Exception as e:
            logger.warning("[youtube-auth] Token refresh failed: %s", e)
            return None

    if not creds or not creds.valid:
        return None

    return creds


def check_auth_status() -> dict:
    """Check if YouTube OAuth is authenticated."""
    client_id, _ = _get_oauth_client()

    creds = get_youtube_credentials()
    if creds:
        # Try to fetch channel info to confirm auth works
        try:
            from googleapiclient.discovery import build

            service = build("youtube", "v3", credentials=creds)
            resp = service.channels().list(
                part="snippet", mine=True,
            ).execute()

            items = resp.get("items", [])
            if items:
                channel = items[0]["snippet"]
                return {
                    "authenticated": True,
                    "channel_name": channel.get("title", "Unknown"),
                    "channel_id": items[0].get("id"),
                    "has_client_id": bool(client_id),
                }
        except Exception as e:
            logger.warning("[youtube-auth] Auth check failed: %s", e)
            return {
                "authenticated": False,
                "error": str(e),
                "has_client_id": bool(client_id),
            }

    return {
        "authenticated": False,
        "has_client_id": bool(client_id),
        "has_token": TOKEN_PATH.exists(),
        "needs_setup": not bool(client_id),
    }


def start_auth_flow() -> dict:
    """Start OAuth flow with a local callback server on port 8090.

    Returns status dict. The backend spins up a temp server to
    catch the redirect. Opens the auth URL in system browser.
    """
    client_id, client_secret = _get_oauth_client()
    if not client_id or not client_secret:
        return {
            "error": "no_credentials",
            "message": (
                "YouTube OAuth credentials not configured. "
                "Go to ASTRA Settings and set YouTube Client ID and Secret."
            ),
        }

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        return {
            "error": "missing_package",
            "message": "google-auth-oauthlib not installed",
        }

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
            logger.info("[youtube-auth] OAuth complete — YouTube connected")
        except Exception as e:
            logger.error("[youtube-auth] OAuth flow failed: %s", e)

    thread = threading.Thread(target=_run_flow, daemon=True)
    thread.start()

    return {
        "started": True,
        "message": (
            "Google sign-in opened in your browser. "
            "Sign in and allow YouTube access."
        ),
        "poll": "/content/youtube/auth/status",
    }


def revoke_auth() -> dict:
    """Revoke YouTube OAuth access and delete stored token."""
    if TOKEN_PATH.exists():
        TOKEN_PATH.unlink()
        logger.info("[youtube-auth] Token revoked")
    return {"revoked": True}


def _save_token(creds) -> None:
    """Save credentials to token file."""
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TOKEN_PATH, "w") as f:
        f.write(creds.to_json())
    logger.info("[youtube-auth] Token saved to %s", TOKEN_PATH)
