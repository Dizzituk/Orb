# FILE: app/finance/services/drive_browser_service.py
"""
Google Drive file browser — list folders, list files, download file bytes.
Used by van finance PDF import, credit card statement import, etc.
"""
from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def list_root_folders() -> list[dict]:
    """List top-level folders in the user's Drive."""
    from app.finance.services.drive_auth_service import get_drive_service

    service = get_drive_service()
    results = service.files().list(
        q="'root' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id, name, modifiedTime)",
        orderBy="name",
        pageSize=50,
    ).execute()
    return [
        {"id": f["id"], "name": f["name"], "modified": f.get("modifiedTime")}
        for f in results.get("files", [])
    ]


def list_folder_contents(folder_id: str, file_types: Optional[list[str]] = None) -> dict:
    """List subfolders and files in a Drive folder.
    
    Args:
        folder_id: Google Drive folder ID
        file_types: Optional filter e.g. ['pdf', 'csv', 'png', 'jpg']
    
    Returns:
        {"folders": [...], "files": [...]}
    """
    from app.finance.services.drive_auth_service import get_drive_service

    service = get_drive_service()

    # Get subfolders
    folders_q = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    folder_results = service.files().list(
        q=folders_q,
        fields="files(id, name, modifiedTime)",
        orderBy="name",
        pageSize=100,
    ).execute()

    folders = [
        {"id": f["id"], "name": f["name"], "modified": f.get("modifiedTime"), "type": "folder"}
        for f in folder_results.get("files", [])
    ]

    # Get files
    file_q_parts = [f"'{folder_id}' in parents", "trashed=false"]
    
    if file_types:
        # Build MIME type filter
        mime_map = {
            "pdf": "application/pdf",
            "csv": "text/csv",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
        }
        mime_clauses = []
        for ft in file_types:
            mime = mime_map.get(ft.lower())
            if mime:
                mime_clauses.append(f"mimeType='{mime}'")
            # Also match by file extension in name
            mime_clauses.append(f"name contains '.{ft.lower()}'")
        if mime_clauses:
            file_q_parts.append(f"({' or '.join(mime_clauses)})")
    else:
        file_q_parts.append("mimeType!='application/vnd.google-apps.folder'")

    file_results = service.files().list(
        q=" and ".join(file_q_parts),
        fields="files(id, name, mimeType, size, modifiedTime)",
        orderBy="modifiedTime desc",
        pageSize=50,
    ).execute()

    files = [
        {
            "id": f["id"],
            "name": f["name"],
            "mime": f.get("mimeType"),
            "size": int(f.get("size", 0)),
            "modified": f.get("modifiedTime"),
            "type": "file",
        }
        for f in file_results.get("files", [])
    ]

    return {"folders": folders, "files": files}


def download_file_bytes(file_id: str) -> bytes:
    """Download a file from Google Drive and return its bytes."""
    from app.finance.services.drive_auth_service import get_drive_service

    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)

    buffer = io.BytesIO()
    from googleapiclient.http import MediaIoBaseDownload
    downloader = MediaIoBaseDownload(buffer, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    buffer.seek(0)
    return buffer.read()


def search_files(query: str, file_types: Optional[list[str]] = None) -> list[dict]:
    """Search across all of Drive for files matching a query."""
    from app.finance.services.drive_auth_service import get_drive_service

    service = get_drive_service()

    q_parts = [f"name contains '{query}'", "trashed=false"]
    if file_types:
        mime_map = {
            "pdf": "application/pdf",
            "csv": "text/csv",
        }
        type_clauses = []
        for ft in file_types:
            m = mime_map.get(ft.lower())
            if m:
                type_clauses.append(f"mimeType='{m}'")
        if type_clauses:
            q_parts.append(f"({' or '.join(type_clauses)})")

    results = service.files().list(
        q=" and ".join(q_parts),
        fields="files(id, name, mimeType, size, modifiedTime, parents)",
        orderBy="modifiedTime desc",
        pageSize=20,
    ).execute()

    return [
        {
            "id": f["id"],
            "name": f["name"],
            "mime": f.get("mimeType"),
            "size": int(f.get("size", 0)),
            "modified": f.get("modifiedTime"),
        }
        for f in results.get("files", [])
    ]

