# FILE: app/cloud/gdrive_auth.py
# Purpose: Google Drive OAuth2 authentication flow.
# Called-by: app.cloud.router
# Depends-on: app.db, app.settings.service
# Last-renovated: 2026-06-11
"""
Google Drive OAuth2 authentication flow.

Reuses the same Google Cloud project as YouTube OAuth.
Adds `drive.file` scope so ASTRA can upload/manage its own files.

Run this script once to authenticate:
    python -m app.cloud.gdrive_auth

v1.0 (2026-03-14): Initial — one-time OAuth flow for Drive file access.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_TOKEN_PATH = Path("D:/Orb/data/google_drive_token.json")
_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def run_auth_flow() -> bool:
    """Run the OAuth2 flow for Google Drive access.
    
    Opens a browser for the user to sign in and grant Drive access.
    Saves the token to data/google_drive_token.json.
    """
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("ERROR: google-auth-oauthlib not installed.")
        print("Run: pip install google-auth-oauthlib google-api-python-client --break-system-packages")
        return False

    # Get client credentials from ASTRA settings
    client_id = os.getenv("YOUTUBE_CLIENT_ID", "")
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        # Try loading from settings DB
        try:
            from app.settings.service import sync_all_keys
            from app.db import get_db_session
            db = get_db_session()
            try:
                sync_all_keys(db)
            finally:
                db.close()
            client_id = os.getenv("YOUTUBE_CLIENT_ID", "")
            client_secret = os.getenv("YOUTUBE_CLIENT_SECRET", "")
        except Exception:
            pass

    if not client_id or not client_secret:
        print("ERROR: No Google OAuth client credentials found.")
        print("Set YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET in ASTRA Settings.")
        return False

    # Build client config from env vars (same as YouTube uses)
    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost", "urn:ietf:wg:oauth:2.0:oob"],
        }
    }

    flow = InstalledAppFlow.from_client_config(client_config, _SCOPES)
    creds = flow.run_local_server(port=8090, prompt="consent")

    # Save token
    _TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_TOKEN_PATH, "w") as f:
        f.write(creds.to_json())

    print(f"Google Drive authenticated! Token saved to {_TOKEN_PATH}")
    return True


def check_auth() -> dict:
    """Check if Google Drive OAuth is configured and valid."""
    if not _TOKEN_PATH.exists():
        return {"authenticated": False, "message": "No token found. Run: python -m app.cloud.gdrive_auth"}

    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request

        creds = Credentials.from_authorized_user_file(str(_TOKEN_PATH), _SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(_TOKEN_PATH, "w") as f:
                f.write(creds.to_json())

        if creds.valid:
            return {"authenticated": True, "message": "Google Drive connected"}
        else:
            return {"authenticated": False, "message": "Token expired and refresh failed"}
    except Exception as e:
        return {"authenticated": False, "message": str(e)}


if __name__ == "__main__":
    # When run directly, do the auth flow
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    # Load env
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
    
    success = run_auth_flow()
    if success:
        print("\nYou can now use 'Astra, build the APK and put it in the cloud'")
    else:
        print("\nAuth failed. Check the error above.")
