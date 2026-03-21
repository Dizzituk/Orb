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
            "category": reg.get("category", "other"),
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
        # v3.2 diagnostic: confirm key was actually written
        reg = API_KEY_REGISTRY.get(entry.key_name)
        if reg:
            env_var = reg["env_var"]
            written = os.getenv(env_var, "")
            logger.info(
                f"[settings] Synced {entry.key_name} -> {env_var}: "
                f"{len(written)} chars, bool={bool(written)}"
            )

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
        elif key_name == "trading212_api_key":
            secret = _get_companion_key(db, "trading212_api_secret")
            status, detail = await _verify_trading212(value, secret)
        elif key_name == "coinmarketcap":
            status, detail = await _verify_coinmarketcap(value)
        elif key_name == "meta_access_token":
            status, detail = await _verify_meta(value)
        elif key_name == "pexels":
            status, detail = await _verify_pexels(value)
        elif key_name == "pixabay":
            status, detail = await _verify_pixabay(value)

        elif key_name == "fal_ai":
            status, detail = await _verify_fal_ai(value)
        elif key_name == "heygen":
            status, detail = await _verify_heygen(value)
        elif key_name in (
            "youtube_client_id", "youtube_client_secret", "youtube_api_key",
            "meta_app_id", "meta_app_secret",
            "tiktok_client_key", "tiktok_client_secret",
            "trading212_api_secret",
            "quickbooks_client_id", "quickbooks_client_secret",
            "quickbooks_refresh_token", "quickbooks_realm_id",
            "heygen_avatar_id", "heygen_voice_id",
        ):
            # These are credential components verified via their primary key
            status = "stored"
            detail = "Credential stored — verified via primary key"
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


async def _verify_trading212(api_key: str, api_secret: Optional[str]) -> tuple:
    """
    Verify Trading 212 credentials.
    Uses Basic Auth (Base64 of key:secret) against account cash endpoint.
    """
    if not api_secret:
        return "incomplete", "API Secret not set — both key and secret are required"

    import httpx
    import base64

    credentials = base64.b64encode(
        f"{api_key}:{api_secret}".encode("utf-8")
    ).decode("utf-8")

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://live.trading212.com/api/v0/equity/account/cash",
            headers={"Authorization": f"Basic {credentials}"},
        )
        if resp.status_code == 200:
            return "valid", "Credentials accepted"
        elif resp.status_code in (401, 403):
            return "invalid", "Authentication failed"
        else:
            return "error", f"HTTP {resp.status_code}"


async def _verify_coinmarketcap(key: str) -> tuple:
    """Verify CoinMarketCap API key with a lightweight call."""
    import httpx
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://pro-api.coinmarketcap.com/v1/key/info",
            headers={
                "X-CMC_PRO_API_KEY": key,
                "Accepts": "application/json",
            },
        )
        if resp.status_code == 200:
            data = resp.json()
            plan = data.get("data", {}).get("plan", {}).get("plan_name", "unknown")
            return "valid", f"Key accepted — {plan} plan"
        elif resp.status_code == 401:
            return "invalid", "Authentication failed"
        else:
            return "error", f"HTTP {resp.status_code}"


async def _verify_meta(token: str) -> tuple:
    """
    Verify Meta (Facebook/Instagram) access token.
    Calls /me on the Graph API to confirm token validity.
    """
    import httpx
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://graph.facebook.com/v19.0/me",
            params={"access_token": token},
        )
        if resp.status_code == 200:
            data = resp.json()
            name = data.get("name", "unknown")
            return "valid", f"Token valid — {name}"
        elif resp.status_code == 401:
            return "invalid", "Token expired or invalid"
        else:
            error = resp.json().get("error", {}).get("message", "")
            return "error", error or f"HTTP {resp.status_code}"


async def _verify_pexels(key: str) -> tuple:
    """Verify Pexels API key with a curated photos request."""
    import httpx
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://api.pexels.com/v1/curated?per_page=1",
            headers={"Authorization": key},
        )
        if resp.status_code == 200:
            return "valid", "Key accepted"
        elif resp.status_code == 401:
            return "invalid", "Authentication failed"
        else:
            return "error", f"HTTP {resp.status_code}"


async def _verify_pixabay(key: str) -> tuple:
    """Verify Pixabay API key with a simple video search."""
    import httpx
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://pixabay.com/api/videos/",
            params={"key": key, "q": "test", "per_page": 3},
        )
        if resp.status_code == 200:
            data = resp.json()
            hits = len(data.get("hits", []))
            return "valid", f"Key accepted ({hits} test results)"
        elif resp.status_code == 401:
            return "invalid", "Authentication failed"
        elif resp.status_code == 429:
            return "error", "Rate limited — key is valid but try again later"
        else:
            return "error", f"HTTP {resp.status_code}"



async def _verify_fal_ai(key: str) -> tuple:
    """Verify fal.ai API key with the pricing endpoint."""
    import httpx
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://api.fal.ai/v1/models/pricing",
            headers={"Authorization": f"Key {key}"},
        )
        if resp.status_code == 200:
            return "valid", "Key accepted"
        elif resp.status_code == 401:
            return "invalid", "Authentication failed"
        else:
            return "error", f"HTTP {resp.status_code}"


async def _verify_heygen(key: str) -> tuple:
    """Verify HeyGen API key with an avatars list request."""
    import httpx
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://api.heygen.com/v2/avatars",
            headers={"X-Api-Key": key},
        )
        if resp.status_code == 200:
            return "valid", "Key accepted"
        elif resp.status_code in (401, 403):
            return "invalid", f"Key rejected ({resp.status_code})"
        else:
            return "error", f"HTTP {resp.status_code}"


def _get_companion_key(db: Session, key_name: str) -> Optional[str]:
    """Get a companion key value (e.g., secret for a key pair)."""
    entry = db.query(ApiKeyEntry).get(key_name)
    if entry and entry.active:
        return entry.key_value

    reg = API_KEY_REGISTRY.get(key_name)
    if reg:
        return os.getenv(reg["env_var"])

    return None
