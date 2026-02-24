# FILE: app/settings/service.py
"""
API Key Settings Service.

Core logic for managing API keys:
- Store encrypted in DB
- Sync to os.environ so existing code (providers, etc.) works unchanged
- Verify keys against their respective APIs
- Mask keys for safe display
"""
import os
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from sqlalchemy.orm import Session

from app.settings.models import ApiKeyEntry, API_KEY_REGISTRY

logger = logging.getLogger(__name__)


def mask_key(value: str) -> str:
    """
    Mask an API key for safe display.
    Shows first 4 and last 4 characters only.
    E.g., "sk-abc123...xyz789"
    """
    if not value or len(value) <= 10:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


def set_api_key(
    db: Session,
    key_name: str,
    key_value: str,
) -> ApiKeyEntry:
    """
    Store or update an API key.
    Encrypts in DB and syncs to os.environ immediately.
    """
    if key_name not in API_KEY_REGISTRY:
        raise ValueError(
            f"Unknown key name '{key_name}'. "
            f"Valid keys: {list(API_KEY_REGISTRY.keys())}"
        )

    # Strip whitespace (common paste error)
    key_value = key_value.strip()

    if not key_value:
        raise ValueError("API key value cannot be empty")

    # Upsert: update if exists, create if not
    entry = db.query(ApiKeyEntry).get(key_name)
    if entry:
        entry.key_value = key_value
        entry.active = True
        entry.updated_at = datetime.now(timezone.utc)
        entry.verification_status = "unknown"
    else:
        entry = ApiKeyEntry(
            key_name=key_name,
            key_value=key_value,
            active=True,
            verification_status="unknown",
        )
        db.add(entry)

    db.commit()
    db.refresh(entry)

    # Sync to environment variable
    _sync_to_env(key_name, key_value)

    logger.info(
        f"[settings] API key '{key_name}' stored and synced "
        f"({mask_key(key_value)})"
    )
    return entry


def remove_api_key(
    db: Session,
    key_name: str,
) -> bool:
    """Remove an API key from DB and environment."""
    entry = db.query(ApiKeyEntry).get(key_name)
    if not entry:
        return False

    db.delete(entry)
    db.commit()

    # Remove from environment
    reg = API_KEY_REGISTRY.get(key_name)
    if reg:
        env_var = reg["env_var"]
        os.environ.pop(env_var, None)

    logger.info(f"[settings] API key '{key_name}' removed")
    return True


def toggle_api_key(
    db: Session,
    key_name: str,
    active: bool,
) -> Optional[ApiKeyEntry]:
    """Enable or disable an API key without deleting it."""
    entry = db.query(ApiKeyEntry).get(key_name)
    if not entry:
        return None

    entry.active = active
    entry.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(entry)

    if active:
        # Re-sync to env
        _sync_to_env(key_name, entry.key_value)
    else:
        # Remove from env
        reg = API_KEY_REGISTRY.get(key_name)
        if reg:
            os.environ.pop(reg["env_var"], None)

    logger.info(f"[settings] API key '{key_name}' {'enabled' if active else 'disabled'}")
    return entry


def get_all_keys_status(db: Session) -> List[Dict[str, Any]]:
    """
    Get status of all known API keys.
    Shows which are set, masked values, and verification status.
    """
    stored = {
        e.key_name: e
        for e in db.query(ApiKeyEntry).all()
    }

    results = []
    for key_name, reg in API_KEY_REGISTRY.items():
        entry = stored.get(key_name)

        # Check if key exists in env (could be from .env file)
        env_var = reg["env_var"]
        env_value = os.getenv(env_var)

        if entry and entry.active:
            source = "settings"
            masked = mask_key(entry.key_value)
            is_set = True
        elif env_value:
            source = "env_file"
            masked = mask_key(env_value)
            is_set = True
        else:
            source = None
            masked = None
            is_set = False

        results.append({
            "key_name": key_name,
            "display_name": reg["display_name"],
            "description": reg["description"],
            "required": reg["required"],
            "is_set": is_set,
            "source": source,
            "masked_value": masked,
            "active": entry.active if entry else (is_set if env_value else False),
            "verification_status": (
                entry.verification_status if entry else "unknown"
            ),
            "last_verified_at": (
                entry.last_verified_at.isoformat()
                if entry and entry.last_verified_at else None
            ),
            "url": reg["url"],
            "prefix_hint": reg["prefix_hint"],
        })

    return results


def get_key_value(db: Session, key_name: str) -> Optional[str]:
    """
    Get the actual (decrypted) API key value.
    Falls back to os.getenv if not in DB.
    """
    entry = db.query(ApiKeyEntry).get(key_name)
    if entry and entry.active:
        return entry.key_value

    reg = API_KEY_REGISTRY.get(key_name)
    if reg:
        return os.getenv(reg["env_var"])

    return None


# ═══════════════════════════════════════════════════
# ENV SYNC
# ═══════════════════════════════════════════════════

def _sync_to_env(key_name: str, value: str) -> None:
    """Sync a key to os.environ so existing code picks it up."""
    reg = API_KEY_REGISTRY.get(key_name)
    if not reg:
        return

    env_var = reg["env_var"]
    os.environ[env_var] = value
    logger.debug(f"[settings] Synced {key_name} → ${env_var}")


def sync_all_to_env(db: Session) -> int:
    """
    Sync all active DB-stored keys to os.environ.
    Called at startup to overlay DB keys on top of .env values.
    DB keys take priority over .env file.
    """
    entries = (
        db.query(ApiKeyEntry)
        .filter(ApiKeyEntry.active == True)
        .all()
    )

    count = 0
    for entry in entries:
        _sync_to_env(entry.key_name, entry.key_value)
        count += 1

    if count:
        logger.info(f"[settings] Synced {count} API keys from DB to env")
    return count


# ═══════════════════════════════════════════════════
# KEY VERIFICATION
# ═══════════════════════════════════════════════════

async def verify_api_key(
    db: Session,
    key_name: str,
) -> Dict[str, Any]:
    """
    Test an API key by making a lightweight API call.
    Updates verification status in DB.
    """
    entry = db.query(ApiKeyEntry).get(key_name)
    value = entry.key_value if entry else os.getenv(
        API_KEY_REGISTRY.get(key_name, {}).get("env_var", "")
    )

    if not value:
        return {"key_name": key_name, "status": "not_set"}

    status = "unknown"
    detail = ""

    try:
        if key_name == "openai":
            status, detail = await _verify_openai(value)
        elif key_name == "anthropic":
            status, detail = await _verify_anthropic(value)
        elif key_name == "google":
            status, detail = await _verify_google(value)
        elif key_name == "brave_search":
            status, detail = await _verify_brave(value)
        else:
            status = "skipped"
            detail = "No verification available for this key type"

    except Exception as e:
        status = "error"
        detail = str(e)

    # Update DB record
    if entry:
        entry.verification_status = status
        entry.last_verified_at = datetime.now(timezone.utc)
        db.commit()

    return {
        "key_name": key_name,
        "status": status,
        "detail": detail,
    }


async def _verify_openai(key: str) -> tuple:
    """Verify OpenAI key with a models list call."""
    import httpx
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        if resp.status_code == 200:
            return "valid", "Key accepted"
        elif resp.status_code == 401:
            return "invalid", "Authentication failed"
        else:
            return "error", f"HTTP {resp.status_code}"


async def _verify_anthropic(key: str) -> tuple:
    """Verify Anthropic key with a lightweight call."""
    import httpx
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        if resp.status_code in (200, 201):
            return "valid", "Key accepted"
        elif resp.status_code == 401:
            return "invalid", "Authentication failed"
        else:
            return "error", f"HTTP {resp.status_code}"


async def _verify_google(key: str) -> tuple:
    """Verify Google API key with a models list call."""
    import httpx
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
        )
        if resp.status_code == 200:
            return "valid", "Key accepted"
        elif resp.status_code in (400, 403):
            return "invalid", "Key rejected"
        else:
            return "error", f"HTTP {resp.status_code}"


async def _verify_brave(key: str) -> tuple:
    """Verify Brave Search API key."""
    import httpx
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": key},
            params={"q": "test"},
        )
        if resp.status_code == 200:
            return "valid", "Key accepted"
        elif resp.status_code in (401, 403):
            return "invalid", "Authentication failed"
        else:
            return "error", f"HTTP {resp.status_code}"
