# FILE: app/documents/storage.py
# Purpose: Save discipline for the editor — atomic write (temp + replace)
#          with a one-time .bak of the original file.
# Called-by: app.documents.router
# Depends-on: stdlib
# Last-renovated: 2026-06-12
"""
Atomic save + first-save backup.

The .bak sits next to the original (report.xlsx -> report.xlsx.bak) and is
written ONCE — the first time the editor saves over an existing file. Later
saves never touch it, so the pre-ASTRA original always survives.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def backup_path(path: str) -> Path:
    return Path(str(path) + ".bak")


def ensure_first_save_backup(path: str) -> bool:
    """Copy original -> .bak if the original exists and no .bak does yet."""
    source = Path(path)
    bak = backup_path(path)
    if source.exists() and not bak.exists():
        shutil.copy2(source, bak)
        logger.info("[documents] first-save backup: %s", bak)
        return True
    return False


def atomic_write_via(path: str, writer) -> None:
    """
    writer(temp_path) produces the new file at temp_path; we then replace
    the target in one os.replace. The temp lives in the SAME directory so
    the replace is same-volume atomic.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=target.suffix + ".tmp",
        dir=str(target.parent))
    os.close(fd)
    try:
        writer(temp_name)
        os.replace(temp_name, str(target))
    finally:
        if os.path.exists(temp_name):
            try:
                os.remove(temp_name)
            except OSError:
                pass
