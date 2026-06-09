# FILE: app/bridge/uploads_store.py
"""
Bridge upload storage layer.

Single responsibility: persist files arriving on POST /bridge/upload to a
human-navigable folder under the user's Pictures directory, append a ledger
entry, and provide lookup by upload ID.

Design notes:
  - Files land in PICTURES_ROOT / "Astra Mobile Uploads" / YYYY-MM / <ts>_<uuid>.<ext>.
    That path is already watched by the drive content indexer, so uploads
    become searchable automatically with no extra wiring.
  - Extension is derived from the Content-Type header so saved files keep
    their real type (.jpg / .png / .pdf) instead of the opaque .bin used
    by the older inline implementation.
  - A single JSONL ledger at PICTURES_ROOT / "Astra Mobile Uploads" / "_ledger.jsonl"
    tracks every upload by id and idempotency_key. Dedup is the caller's job;
    we expose helpers (`is_duplicate`, `find_by_id`) so the endpoint can stay slim.
  - Nothing here knows about FastAPI, auth, or HTTP. Pure storage.

v1.0 (2026-05-24): extracted from inline bridge_upload handler in router.py
so the endpoint can stay small and the destination/format can be tested
without booting the API.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Resolve the user's OneDrive Pictures folder. Falls back to a sane default
# if the env var isn't set so tests on machines without OneDrive still work.
_PICTURES_ENV = os.environ.get("BRIDGE_UPLOADS_PICTURES_ROOT")
if _PICTURES_ENV:
    PICTURES_ROOT = _PICTURES_ENV
else:
    _userprofile = os.environ.get("USERPROFILE", os.path.expanduser("~"))
    PICTURES_ROOT = os.path.join(_userprofile, "OneDrive", "Pictures")

UPLOADS_ROOT = os.path.join(PICTURES_ROOT, "Astra Mobile Uploads")
LEDGER_PATH = os.path.join(UPLOADS_ROOT, "_ledger.jsonl")


# ---------------------------------------------------------------------------
# MIME -> extension mapping
# ---------------------------------------------------------------------------

# Only the formats we expect from a phone. Anything else falls through to .bin
# so the file is still saved; we just can't pretty-name it.
_MIME_EXT = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/heic": "heic",
    "image/heif": "heif",
    "application/pdf": "pdf",
    "text/plain": "txt",
}


def extension_for(content_type: str) -> str:
    """Return a sensible file extension (no dot) for a Content-Type header."""
    if not content_type:
        return "bin"
    # Strip any "; charset=..." suffix.
    ct = content_type.split(";", 1)[0].strip().lower()
    return _MIME_EXT.get(ct, "bin")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UploadRecord:
    id: str
    idempotency_key: str
    path: str
    bytes: int
    content_type: str
    received_at: str
    original_filename: str = ""

    def to_jsonl(self) -> str:
        return json.dumps({
            "id": self.id,
            "idempotency_key": self.idempotency_key,
            "path": self.path,
            "bytes": self.bytes,
            "content_type": self.content_type,
            "received_at": self.received_at,
            "original_filename": self.original_filename,
        }) + "\n"


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def _month_dir(now: datetime) -> str:
    return os.path.join(UPLOADS_ROOT, now.strftime("%Y-%m"))


def _ensure_dirs(month_dir: str) -> None:
    os.makedirs(month_dir, exist_ok=True)


def is_duplicate(idempotency_key: str) -> bool:
    """Return True if this idempotency key has been seen before.

    Safe to call before the ledger or its parent directory exist; returns
    False in that case. Read errors are logged but treated as "not duplicate"
    so a damaged ledger never blocks a fresh upload.
    """
    if not idempotency_key or not os.path.isfile(LEDGER_PATH):
        return False
    try:
        with open(LEDGER_PATH, "r", encoding="utf-8") as lf:
            for line in lf:
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if entry.get("idempotency_key") == idempotency_key:
                    return True
    except Exception as e:
        logger.warning("[uploads_store] ledger read failed: %s", e)
    return False


def find_by_id(file_id: str) -> Optional[UploadRecord]:
    """Look up a previously stored upload by its UUID."""
    if not file_id or not os.path.isfile(LEDGER_PATH):
        return None
    try:
        with open(LEDGER_PATH, "r", encoding="utf-8") as lf:
            for line in lf:
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if entry.get("id") == file_id:
                    return UploadRecord(
                        id=entry["id"],
                        idempotency_key=entry.get("idempotency_key", ""),
                        path=entry["path"],
                        bytes=int(entry.get("bytes", 0)),
                        content_type=entry.get("content_type", ""),
                        received_at=entry.get("received_at", ""),
                        original_filename=entry.get("original_filename", ""),
                    )
    except Exception as e:
        logger.warning("[uploads_store] ledger read failed during lookup: %s", e)
    return None


def save_upload(body: bytes, content_type: str, idempotency_key: str, original_filename: Optional[str] = None) -> UploadRecord:
    """Persist `body` under the current YYYY-MM folder and append a ledger row.

    The caller is expected to have already checked `is_duplicate` and to have
    a non-empty body. We do not re-validate here; we just write.

    Returns the UploadRecord. Filename pattern: <YYYYMMDD_HHMMSS>_<uuid>.<ext>
    so files sort chronologically when listed in Explorer.
    """
    now = datetime.now(timezone.utc)
    month_dir = _month_dir(now)
    _ensure_dirs(month_dir)

    file_id = uuid.uuid4().hex
    ext = extension_for(content_type)
    logger.info(
        "[uploads_store] save_upload: content_type=%r -> ext=%r bytes=%d key=%s",
        content_type, ext, len(body), (idempotency_key or '')[:12],
    )
    ts = now.strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{file_id}.{ext}"
    out_path = os.path.join(month_dir, filename)

    with open(out_path, "wb") as f:
        f.write(body)

    record = UploadRecord(
        id=file_id,
        idempotency_key=idempotency_key or "",
        path=out_path,
        bytes=len(body),
        content_type=content_type or "",
        received_at=now.isoformat(),
        original_filename=(original_filename or ""),
    )

    # Ensure the ledger directory exists (it should, after _ensure_dirs above,
    # but the ledger lives at UPLOADS_ROOT, not in the month subfolder).
    os.makedirs(UPLOADS_ROOT, exist_ok=True)
    with open(LEDGER_PATH, "a", encoding="utf-8") as lf:
        lf.write(record.to_jsonl())

    return record


def recent_records(limit: int = 10, image_only: bool = True) -> "list[UploadRecord]":
    """Most-recent-first list of uploads from the ledger.

    The ledger is append-ordered (chronological), so the tail is newest; we
    reverse and cap. Used by the deterministic uploaded-screenshot logger to
    grab the shots the user just sent without the chat model juggling UUIDs.
    """
    if not os.path.isfile(LEDGER_PATH):
        return []
    out: "list[UploadRecord]" = []
    try:
        with open(LEDGER_PATH, "r", encoding="utf-8") as lf:
            for line in lf:
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                ct = (entry.get("content_type") or "").lower()
                if image_only and not ct.startswith("image/"):
                    continue
                out.append(UploadRecord(
                    id=entry.get("id", ""),
                    idempotency_key=entry.get("idempotency_key", ""),
                    path=entry.get("path", ""),
                    bytes=int(entry.get("bytes", 0)),
                    content_type=entry.get("content_type", ""),
                    received_at=entry.get("received_at", ""),
                    original_filename=entry.get("original_filename", ""),
                ))
    except Exception as e:
        logger.warning("[uploads_store] recent_records read failed: %s", e)
    out.reverse()
    return out[:max(0, limit)]
