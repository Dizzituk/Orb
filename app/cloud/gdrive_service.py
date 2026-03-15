# FILE: app/cloud/gdrive_service.py
"""
Google Drive cloud service via Google Drive API.

Uses service account or OAuth credentials for read/write access.
Fallback provider when Proton Drive writes are unavailable.

v1.0 (2026-03-14): Initial — upload, list, download via Google API.
"""
from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Google Drive folder ID for APK uploads (created on first use)
_APK_FOLDER_NAME = "ASTRA APKs"


def _get_drive_service():
    """Get an authenticated Google Drive API service instance."""
    try:
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
    except ImportError:
        raise ImportError(
            "Google API client not installed. Run: "
            "pip install google-api-python-client google-auth-oauthlib --break-system-packages"
        )

    # Try loading credentials from ASTRA's stored YouTube OAuth tokens
    # (same Google account, Drive scope can be added)
    token_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "google_drive_token.json")
    creds = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(token_path, "w") as f:
                f.write(creds.to_json())

    if not creds or not creds.valid:
        raise ConnectionError(
            "Google Drive not authenticated. Run the OAuth flow first."
        )

    return build("drive", "v3", credentials=creds)


def _get_or_create_folder(service, folder_name: str, parent_id: str = "root") -> str:
    """Find or create a folder in Google Drive. Returns folder ID."""
    # Search for existing folder
    query = (
        f"name = '{folder_name}' and "
        f"'{parent_id}' in parents and "
        f"mimeType = 'application/vnd.google-apps.folder' and "
        f"trashed = false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    if files:
        return files[0]["id"]

    # Create the folder
    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = service.files().create(body=metadata, fields="id").execute()
    logger.info("[gdrive] Created folder '%s' (id=%s)", folder_name, folder["id"])
    return folder["id"]


async def upload_file(
    local_path: str,
    cloud_filename: str,
    folder_name: str = "ASTRA APKs",
) -> Dict[str, Any]:
    """Upload a file to Google Drive."""
    from googleapiclient.http import MediaFileUpload

    if not Path(local_path).exists():
        return {"success": False, "error": f"File not found: {local_path}"}

    try:
        service = _get_drive_service()
        folder_id = _get_or_create_folder(service, folder_name)

        # Check if file already exists in folder (overwrite)
        query = f"name = '{cloud_filename}' and '{folder_id}' in parents and trashed = false"
        existing = service.files().list(q=query, fields="files(id)").execute()
        existing_files = existing.get("files", [])

        media = MediaFileUpload(local_path, resumable=True)
        metadata = {"name": cloud_filename, "parents": [folder_id]}

        if existing_files:
            # Update existing file
            file_id = existing_files[0]["id"]
            result = service.files().update(
                fileId=file_id, media_body=media, fields="id, name, size, webViewLink"
            ).execute()
        else:
            # Create new file
            result = service.files().create(
                body=metadata, media_body=media, fields="id, name, size, webViewLink"
            ).execute()

        size = Path(local_path).stat().st_size
        logger.info("[gdrive] Uploaded %s -> %s/%s (%d bytes)", local_path, folder_name, cloud_filename, size)

        return {
            "success": True,
            "provider": "google_drive",
            "file_id": result.get("id"),
            "file_name": result.get("name"),
            "web_link": result.get("webViewLink"),
            "size": size,
            "cloud_path": f"{folder_name}/{cloud_filename}",
        }
    except ImportError as e:
        return {"success": False, "error": str(e)}
    except ConnectionError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.warning("[gdrive] Upload failed: %s", e)
        return {"success": False, "error": str(e)}


async def list_files(folder_name: str = "", max_results: int = 50) -> List[Dict[str, Any]]:
    """List files in Google Drive, optionally within a folder."""
    try:
        service = _get_drive_service()

        if folder_name:
            folder_id = _get_or_create_folder(service, folder_name)
            query = f"'{folder_id}' in parents and trashed = false"
        else:
            query = "'root' in parents and trashed = false"

        results = service.files().list(
            q=query,
            pageSize=max_results,
            fields="files(id, name, mimeType, size, modifiedTime, webViewLink)",
            orderBy="modifiedTime desc",
        ).execute()

        return [
            {
                "name": f.get("name", ""),
                "id": f.get("id", ""),
                "size": int(f.get("size", 0)) if f.get("size") else 0,
                "mod_time": f.get("modifiedTime", ""),
                "mime_type": f.get("mimeType", ""),
                "is_dir": f.get("mimeType") == "application/vnd.google-apps.folder",
                "web_link": f.get("webViewLink", ""),
            }
            for f in results.get("files", [])
        ]
    except Exception as e:
        logger.warning("[gdrive] List failed: %s", e)
        return []


def is_configured() -> bool:
    """Check if Google Drive API access is available."""
    try:
        _get_drive_service()
        return True
    except Exception:
        return False
