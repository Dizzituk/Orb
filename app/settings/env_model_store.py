# FILE: app/settings/env_model_store.py
# Purpose: Atomic .env model-var operations — targeted line rewrite/append,
#          write-temp-then-replace, parse validation, and hot reload with a
#          model-var diff (LANE D Tasks 2a/2b plumbing).
# Called-by: app.settings.model_settings
# Depends-on: python-dotenv, app.settings.service (key re-sync), app.db
# Last-renovated: 2026-07-02
"""
.env is the canonical model store (Lane A). This module is the ONLY writer
the settings API uses:

* set_env_vars({...}) — rewrites the LAST occurrence of each var in place
  (python-dotenv gives later lines precedence, so the last occurrence is the
  effective one and earlier lane blocks stay untouched), else appends under
  the "## UI-MANAGED MODELS" marker. Written to a temp file first, parse-
  validated with dotenv_values, then os.replace'd (atomic on same volume).
  A dated backup goes to D:\\QUARANTINE\\_snapshots first (quarantine rule).

* reload_env() — load_dotenv(override=True) then re-syncs the DB-managed API
  keys over the top (the encrypted settings_api_keys store must keep winning
  over anything stale in .env), returning a name: old->new diff of model vars.
"""
from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from dotenv import dotenv_values, load_dotenv

logger = logging.getLogger(__name__)

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
_UI_MARKER = "## UI-MANAGED MODELS"
_BACKUP_DIR = Path(r"D:\QUARANTINE\_snapshots")

# Vars considered "model vars" for the reload diff (never API keys).
_MODEL_VAR_MARKERS = ("MODEL", "PROVIDER")
_MODEL_VAR_PREFIXES = ("ASTRA_FALLBACK_CHAIN_", "ASTRA_TOOL_TRUSTED_MODELS_",
                       "ASTRA_CODEBASE_READER_MODELS", "ASTRA_SMALL_MODELS")


def is_model_var(name: str) -> bool:
    upper = name.upper()
    if "KEY" in upper or "SECRET" in upper or "TOKEN" in upper:
        return False
    return any(m in upper for m in _MODEL_VAR_MARKERS) or upper.startswith(_MODEL_VAR_PREFIXES)


def _validate_value(value: str) -> None:
    """Refuse values that could break .env line structure."""
    if "\n" in value or "\r" in value or "\x00" in value:
        raise ValueError("value must be a single line")
    if len(value) > 500:
        raise ValueError("value too long")


def _backup_env() -> Optional[Path]:
    try:
        _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = _BACKUP_DIR / f"env.ui-backup-{stamp}"
        shutil.copy2(ENV_PATH, dest)
        return dest
    except Exception as exc:  # backup failure must not block the write
        logger.warning("[env_model_store] .env backup failed: %s", exc)
        return None


def set_env_vars(pairs: Dict[str, str]) -> Dict[str, Tuple[Optional[str], str]]:
    """Atomically rewrite/append vars in .env. Returns {var: (old, new)}.

    Raises ValueError on bad input or if the rewritten file fails to parse
    back with the expected values (in which case .env is left untouched).
    """
    if not ENV_PATH.exists():
        raise ValueError(f".env not found at {ENV_PATH}")
    for var, value in pairs.items():
        if not var or not var.replace("_", "").isalnum():
            raise ValueError(f"bad var name: {var!r}")
        _validate_value(value)

    before = dotenv_values(str(ENV_PATH))
    text = ENV_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()

    for var, value in pairs.items():
        newline = f"{var}={value}"
        idxs = [i for i, l in enumerate(lines)
                if l.startswith(f"{var}=") or l.startswith(f"{var} =")]
        if idxs:
            lines[idxs[-1]] = newline  # last occurrence is the effective one
        else:
            try:
                marker_idx = lines.index(_UI_MARKER)
                lines.insert(marker_idx + 1, newline)
            except ValueError:
                lines.extend(["", _UI_MARKER, newline])

    new_text = "\n".join(lines) + "\n"
    tmp = ENV_PATH.with_name(".env.laned-tmp")
    tmp.write_text(new_text, encoding="utf-8")
    try:
        check = dotenv_values(str(tmp))
        for var, value in pairs.items():
            if check.get(var) != value:
                raise ValueError(
                    f".env rewrite validation failed for {var} "
                    f"(parsed {check.get(var)!r}, expected {value!r}) — aborted"
                )
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    _backup_env()
    os.replace(tmp, ENV_PATH)  # atomic same-volume swap
    diff = {var: (before.get(var), value) for var, value in pairs.items()}
    logger.info("[env_model_store] .env updated: %s",
                {k: f"{o!r} -> {n!r}" for k, (o, n) in diff.items()})
    return diff


def reload_env() -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """Re-run load_dotenv(override=True), re-sync DB API keys, diff model vars.

    Because frontier_models reads env ON ACCESS, changed model vars take
    effect on the next request — no restart (import-time consumers excepted;
    see model_roles.RESTART_GATED).
    """
    before = {k: v for k, v in os.environ.items() if is_model_var(k)}
    load_dotenv(ENV_PATH, override=True)

    # The encrypted settings_api_keys store must keep winning over .env.
    try:
        from app.db import SessionLocal
        from app.settings.service import sync_all_to_env
        db = SessionLocal()
        try:
            sync_all_to_env(db)
        finally:
            db.close()
    except Exception as exc:
        logger.warning("[env_model_store] API-key re-sync after reload failed: %s", exc)

    after = {k: v for k, v in os.environ.items() if is_model_var(k)}
    diff = {
        k: (before.get(k), after.get(k))
        for k in sorted(set(before) | set(after))
        if before.get(k) != after.get(k)
    }
    if diff:
        logger.info("[env_model_store] reload diff: %s",
                    {k: f"{o!r} -> {n!r}" for k, (o, n) in diff.items()})
    else:
        logger.info("[env_model_store] reload: no model-var changes")
    return diff


__all__ = ["ENV_PATH", "is_model_var", "set_env_vars", "reload_env"]
