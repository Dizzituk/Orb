# FILE: app/settings/models.py
"""
Settings Database Models.

API keys stored encrypted at rest using ASTRA's master key.
The EncryptedText type handles encryption/decryption transparently.
"""
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, Boolean, Text

from app.db import Base
from app.crypto import EncryptedText


def _now() -> datetime:
    return datetime.now(timezone.utc)


# Known API key definitions with metadata
API_KEY_REGISTRY = {
    "openai": {
        "display_name": "OpenAI",
        "description": "Chat, embeddings, DALL-E image generation, Sora video generation",
        "env_var": "OPENAI_API_KEY",
        "required": True,
        "url": "https://platform.openai.com/api-keys",
        "prefix_hint": "sk-",
    },
    "anthropic": {
        "display_name": "Anthropic",
        "description": "Claude models for pipeline draft writing and analysis",
        "env_var": "ANTHROPIC_API_KEY",
        "required": False,
        "url": "https://console.anthropic.com/settings/keys",
        "prefix_hint": "sk-ant-",
    },
    "google": {
        "display_name": "Google (Gemini)",
        "description": "Gemini Pro/Flash for vision, video analysis, content scout",
        "env_var": "GOOGLE_API_KEY",
        "required": True,
        "url": "https://aistudio.google.com/apikey",
        "prefix_hint": "AI",
    },
    "brave_search": {
        "display_name": "Brave Search",
        "description": "Web search API for research and fact-checking",
        "env_var": "BRAVE_SEARCH_API_KEY",
        "required": False,
        "url": "https://brave.com/search/api/",
        "prefix_hint": "BSA",
    },
    "youtube_client_id": {
        "display_name": "YouTube Client ID",
        "description": "OAuth Client ID for YouTube Data API v3 uploads",
        "env_var": "YOUTUBE_CLIENT_ID",
        "required": False,
        "url": "https://console.cloud.google.com/apis/credentials",
        "prefix_hint": "",
    },
    "youtube_client_secret": {
        "display_name": "YouTube Client Secret",
        "description": "OAuth Client Secret for YouTube Data API v3",
        "env_var": "YOUTUBE_CLIENT_SECRET",
        "required": False,
        "url": "https://console.cloud.google.com/apis/credentials",
        "prefix_hint": "",
    },
    "instagram_access_token": {
        "display_name": "Instagram Access Token",
        "description": "Long-lived token for Instagram Graph API publishing",
        "env_var": "INSTAGRAM_ACCESS_TOKEN",
        "required": False,
        "url": "https://developers.facebook.com/tools/explorer/",
        "prefix_hint": "",
    },
    "instagram_account_id": {
        "display_name": "Instagram Account ID",
        "description": "Instagram Business/Creator account ID",
        "env_var": "INSTAGRAM_ACCOUNT_ID",
        "required": False,
        "url": "https://developers.facebook.com/tools/explorer/",
        "prefix_hint": "",
    },
}


class ApiKeyEntry(Base):
    """
    An API key stored encrypted in the database.

    The key_value column uses EncryptedText — automatic
    encryption on write, decryption on read via the master key.
    """
    __tablename__ = "settings_api_keys"

    # key_name is the lookup identifier (e.g., "openai", "google")
    key_name = Column(String, primary_key=True)

    # The actual API key — encrypted at rest
    key_value = Column(EncryptedText, nullable=False)

    # Whether this key is currently active (allows disabling without deleting)
    active = Column(Boolean, default=True)

    # Metadata
    last_verified_at = Column(DateTime, nullable=True)
    verification_status = Column(String, nullable=True)  # valid | invalid | unknown

    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)
