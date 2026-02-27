# FILE: app/finance/services/drive_auth_service.py
"""
Google Drive OAuth2 — in-app flow for Electron desktop.

Flow:
1. Backend generates auth URL
2. Frontend opens it in system browser (or Electron BrowserWindow)
3. User logs into Google, clicks Allow
4. Google redirects to http://127.0.0.1:8089
5. Backend catches the code, exchanges for tokens, stores them
6. Frontend polls /finance/drive/status until authenticated

No manual config file downloads needed. OAuth client credentials
are stored in ASTRA's encrypted settings DB.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TOKEN_PATH = Path("D:/Orb/config/google_drive_token.json")
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Default OAuth client for ASTRA Desktop (can be overridden in settings)
# For a desktop app, the client_secret is not truly secret (Google documents this).
# Users can replace with their own client ID in ASTRA Settings.
_DEFAULT_CLIENT_ID = None
_DEFAULT_CLIENT_SECRET = None


def _get_oauth_client():
    """Get OAuth client ID and secret from env, settings, or credentials.json."""
    client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID", _DEFAULT_CLIENT_ID)
    client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", _DEFAULT_CLIENT_SECRET)

    # Fallback: read from google_credentials.json
    if not client_id:
        creds_path = Path("D:/Orb/config/google_credentials.json")
        if creds_path.exists():
            try:
                import json
                data = json.loads(creds_path.read_text())
                installed = data.get("installed", {})
                client_id = installed.get("client_id")
                client_secret = installed.get("client_secret")
            except Exception as e:
                logger.warning("[drive] Failed to read credentials.json: %s", e)

    return client_id, client_secret


def get_drive_service():
    """Get an authenticated Google Drive service instance."""
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError:
        logger.error("[drive] google-api-python-client not installed")
        return None

    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_token(creds)
        except Exception as e:
            logger.warning("[drive] Token refresh failed: %s", e)
            creds = None

    if not creds or not creds.valid:
        return None

    return build("drive", "v3", credentials=creds)


def check_auth_status() -> dict:
    """Check if Google Drive is authenticated."""
    client_id, _ = _get_oauth_client()

    service = get_drive_service()
    if service:
        try:
            about = service.about().get(fields="user").execute()
            email = about.get("user", {}).get("emailAddress", "unknown")
            return {
                "authenticated": True,
                "email": email,
                "has_client_id": bool(client_id),
            }
        except Exception as e:
            return {"authenticated": False, "error": str(e), "has_client_id": bool(client_id)}

    return {
        "authenticated": False,
        "has_client_id": bool(client_id),
        "has_token": TOKEN_PATH.exists(),
        "needs_setup": not bool(client_id),
    }


def get_auth_url() -> dict:
    """Generate the Google OAuth URL for the user to visit.
    
    Returns {"auth_url": "https://accounts.google.com/..."} 
    The frontend should open this in the system browser.
    """
    client_id, client_secret = _get_oauth_client()
    if not client_id or not client_secret:
        return {
            "error": "no_client_id",
            "message": (
                "Google OAuth Client ID not configured. "
                "Go to ASTRA Settings > API Keys and add your Google OAuth Client ID and Secret. "
                "Get these from console.cloud.google.com > APIs & Services > Credentials > "
                "Create OAuth Client ID (Desktop app type). Enable the Google Drive API first."
            ),
        }

    try:
        from google_auth_oauthlib.flow import Flow
    except ImportError:
        return {"error": "missing_package", "message": "google-auth-oauthlib not installed"}

    flow = Flow.from_client_config(
        {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://127.0.0.1:8089"],
            }
        },
        scopes=SCOPES,
        redirect_uri="http://127.0.0.1:8089",
    )

    auth_url, state = flow.authorization_url(
        access_type="offline",
        prompt="consent",
    )

    # Store flow state for later exchange
    _store_flow_state(state, flow)

    return {"auth_url": auth_url, "state": state}


def start_auth_with_local_server() -> dict:
    """Start OAuth flow with a local callback server.
    
    Returns the auth URL. The backend spins up a temp server on port 8089
    to catch the redirect. The frontend opens the URL in a browser.
    """
    client_id, client_secret = _get_oauth_client()
    if not client_id or not client_secret:
        return get_auth_url()  # Returns the setup instructions

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
                "redirect_uris": ["http://127.0.0.1:8089"],
            }
        },
        scopes=SCOPES,
    )

    # Run in background thread so it doesn't block the API
    def _run_flow():
        try:
            creds = flow.run_local_server(
                port=8089,
                prompt="consent",
                access_type="offline",
                open_browser=True,
            )
            _save_token(creds)
            logger.info("[drive] OAuth complete — Google Drive connected")
        except Exception as e:
            logger.error("[drive] OAuth flow failed: %s", e)

    thread = threading.Thread(target=_run_flow, daemon=True)
    thread.start()

    return {
        "started": True,
        "message": "Google sign-in opened in your browser. Sign in and click Allow.",
        "poll": "/finance/drive/status",
    }


# ── Flow state storage (for manual code exchange) ──

_flow_states: dict = {}


def _store_flow_state(state: str, flow):
    _flow_states[state] = flow


def exchange_auth_code(code: str, state: str = "") -> dict:
    """Exchange an authorization code for tokens (manual flow)."""
    flow = _flow_states.pop(state, None)
    if not flow:
        # Recreate flow
        client_id, client_secret = _get_oauth_client()
        if not client_id:
            return {"error": "no_client_id"}
        from google_auth_oauthlib.flow import Flow
        flow = Flow.from_client_config(
            {
                "installed": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["http://127.0.0.1:8089"],
                }
            },
            scopes=SCOPES,
            redirect_uri="http://127.0.0.1:8089",
        )

    flow.fetch_token(code=code)
    _save_token(flow.credentials)
    return {"authenticated": True, "message": "Google Drive connected!"}


def revoke_auth() -> dict:
    """Revoke Google Drive access."""
    if TOKEN_PATH.exists():
        TOKEN_PATH.unlink()
    return {"revoked": True}


def _save_token(creds):
    """Save credentials to token file."""
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TOKEN_PATH, "w") as f:
        f.write(creds.to_json())
    logger.info("[drive] Token saved to %s", TOKEN_PATH)

