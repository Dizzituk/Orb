# FILE: scripts/legal_case/_bootstrap.py
"""
Shared bootstrap for the legal_case_* scripts.

Unlocks ASTRA's encrypted key store and syncs API keys to os.environ,
so downstream modules (app.llm.gemini_vision, app.styling.*) just work.

This is a copy of the same pattern tts_server.py uses; factored out here
so each script doesn't duplicate it. Safe to call multiple times.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_DONE = False


def bootstrap_astra() -> None:
    """Idempotent bootstrap: master key -> DB unlock -> key sync."""
    global _DONE
    if _DONE:
        return

    orb_root = Path(r"D:\Orb")
    if str(orb_root) not in sys.path:
        sys.path.insert(0, str(orb_root))

    # Make sure we're running in Orb's working directory so relative
    # paths in config etc. resolve the same way the app does.
    os.chdir(str(orb_root))

    # 1. Master key from Windows Credential Manager (same helper the TTS
    #    microservice uses).
    if not os.getenv("ORB_MASTER_KEY"):
        from app.voice.tts_server import _read_master_key_from_credential_manager
        key = _read_master_key_from_credential_manager()
        if not key:
            raise RuntimeError(
                "ORB_MASTER_KEY not in env and not found in Windows Credential Manager "
                "under 'OrbMasterKey/default'. Cannot decrypt API key store."
            )
        os.environ["ORB_MASTER_KEY"] = key

    # 2. Validate master key and decrypt crypto subsystem.
    from app.crypto import require_master_key_or_exit
    require_master_key_or_exit()

    # 3. Sync all stored keys into os.environ so downstream modules pick
    #    them up via plain os.getenv().
    from app.db import init_db, SessionLocal
    init_db()
    from app.settings.service import sync_all_to_env

    db = SessionLocal()
    try:
        count = sync_all_to_env(db)
    finally:
        db.close()

    logger.info("[legal_case._bootstrap] Synced %d API keys from DB to env", count)
    _DONE = True
