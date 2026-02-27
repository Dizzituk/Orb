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
# Categories: llm, search, social, finance
API_KEY_REGISTRY = {
    # ═══════════════════════════════════════════
    # LLM PROVIDERS
    # ═══════════════════════════════════════════
    "openai": {
        "display_name": "OpenAI",
        "description": "Chat, embeddings, DALL-E image generation, Sora video generation",
        "env_var": "OPENAI_API_KEY",
        "required": True,
        "category": "llm",
        "url": "https://platform.openai.com/api-keys",
        "prefix_hint": "sk-",
    },
    "anthropic": {
        "display_name": "Anthropic",
        "description": "Claude models for pipeline draft writing and analysis",
        "env_var": "ANTHROPIC_API_KEY",
        "required": False,
        "category": "llm",
        "url": "https://console.anthropic.com/settings/keys",
        "prefix_hint": "sk-ant-",
    },
    "google": {
        "display_name": "Google (Gemini)",
        "description": "Gemini Pro/Flash for vision, video analysis, content scout",
        "env_var": "GOOGLE_API_KEY",
        "required": True,
        "category": "llm",
        "url": "https://aistudio.google.com/apikey",
        "prefix_hint": "AI",
    },
    # ═══════════════════════════════════════════
    # SEARCH
    # ═══════════════════════════════════════════
    "brave_search": {
        "display_name": "Brave Search",
        "description": "Web search API for research and fact-checking",
        "env_var": "BRAVE_SEARCH_API_KEY",
        "required": False,
        "category": "search",
        "url": "https://brave.com/search/api/",
        "prefix_hint": "BSA",
    },
    # ═══════════════════════════════════════════
    # SOCIAL / CONTENT DISTRIBUTION
    # ═══════════════════════════════════════════
    "youtube_client_id": {
        "display_name": "YouTube Client ID",
        "description": "OAuth Client ID for YouTube Data API v3 (same Google Cloud project as Gemini)",
        "env_var": "YOUTUBE_CLIENT_ID",
        "required": False,
        "category": "social",
        "url": "https://console.cloud.google.com/apis/credentials",
        "prefix_hint": "",
    },
    "youtube_client_secret": {
        "display_name": "YouTube Client Secret",
        "description": "OAuth Client Secret for YouTube Data API v3",
        "env_var": "YOUTUBE_CLIENT_SECRET",
        "required": False,
        "category": "social",
        "url": "https://console.cloud.google.com/apis/credentials",
        "prefix_hint": "",
    },
    "youtube_api_key": {
        "display_name": "YouTube API Key",
        "description": "API key for public YouTube data reads (search, video info)",
        "env_var": "YOUTUBE_API_KEY",
        "required": False,
        "category": "social",
        "url": "https://console.cloud.google.com/apis/credentials",
        "prefix_hint": "AI",
    },
    "google_oauth_client_id": {
        "display_name": "Google Drive OAuth Client ID",
        "description": "OAuth Client ID for Google Drive access (Desktop app type). Create at console.cloud.google.com > Credentials",
        "env_var": "GOOGLE_OAUTH_CLIENT_ID",
        "required": False,
        "category": "integrations",
        "url": "https://console.cloud.google.com/apis/credentials",
        "prefix_hint": "",
    },
    "google_oauth_client_secret": {
        "display_name": "Google Drive OAuth Client Secret",
        "description": "OAuth Client Secret for Google Drive access",
        "env_var": "GOOGLE_OAUTH_CLIENT_SECRET",
        "required": False,
        "category": "integrations",
        "url": "https://console.cloud.google.com/apis/credentials",
        "prefix_hint": "",
    },
    "meta_app_id": {
        "display_name": "Meta App ID",
        "description": "Meta Developer App ID — covers Facebook + Instagram via Graph API",
        "env_var": "META_APP_ID",
        "required": False,
        "category": "social",
        "url": "https://developers.facebook.com/apps/",
        "prefix_hint": "",
    },
    "meta_app_secret": {
        "display_name": "Meta App Secret",
        "description": "Meta Developer App Secret for Graph API auth",
        "env_var": "META_APP_SECRET",
        "required": False,
        "category": "social",
        "url": "https://developers.facebook.com/apps/",
        "prefix_hint": "",
    },
    "meta_access_token": {
        "display_name": "Meta Access Token",
        "description": "Long-lived User Access Token for Facebook + Instagram publishing",
        "env_var": "META_ACCESS_TOKEN",
        "required": False,
        "category": "social",
        "url": "https://developers.facebook.com/tools/explorer/",
        "prefix_hint": "",
    },
    "tiktok_client_key": {
        "display_name": "TikTok Client Key",
        "description": "Client Key from TikTok Developer Portal for Content Posting API",
        "env_var": "TIKTOK_CLIENT_KEY",
        "required": False,
        "category": "social",
        "url": "https://developers.tiktok.com/",
        "prefix_hint": "",
    },
    "facebook_page_id": {
        "display_name": "Facebook Page ID",
        "description": "Your Facebook Page ID — found in Page Settings > About. Required for publishing.",
        "env_var": "FACEBOOK_PAGE_ID",
        "required": False,
        "category": "social",
        "url": "https://www.facebook.com/settings/?tab=pages",
        "prefix_hint": "",
    },
    "tiktok_client_secret": {
        "display_name": "TikTok Client Secret",
        "description": "Client Secret from TikTok Developer Portal",
        "env_var": "TIKTOK_CLIENT_SECRET",
        "required": False,
        "category": "social",
        "url": "https://developers.tiktok.com/",
        "prefix_hint": "",
    },
    # ═══════════════════════════════════════════
    # FINANCE / BUSINESS
    # ═══════════════════════════════════════════
    "trading212_api_key": {
        "display_name": "Trading 212 API Key",
        "description": "API Key for portfolio data, positions, trade history (beta)",
        "env_var": "TRADING212_API_KEY",
        "required": False,
        "category": "finance",
        "url": "https://helpcentre.trading212.com/hc/en-us/articles/14584770928157",
        "prefix_hint": "",
    },
    "trading212_api_secret": {
        "display_name": "Trading 212 API Secret",
        "description": "API Secret — shown only once on generation, store securely",
        "env_var": "TRADING212_API_SECRET",
        "required": False,
        "category": "finance",
        "url": "https://helpcentre.trading212.com/hc/en-us/articles/14584770928157",
        "prefix_hint": "",
    },
    "coinmarketcap": {
        "display_name": "CoinMarketCap",
        "description": "Crypto market data — prices, rankings, portfolio tracking (free tier available)",
        "env_var": "COINMARKETCAP_API_KEY",
        "required": False,
        "category": "finance",
        "url": "https://pro.coinmarketcap.com/signup",
        "prefix_hint": "",
    },
    "quickbooks_client_id": {
        "display_name": "QuickBooks Client ID",
        "description": "OAuth Client ID from Intuit Developer Portal for accounting API",
        "env_var": "QUICKBOOKS_CLIENT_ID",
        "required": False,
        "category": "finance",
        "url": "https://developer.intuit.com/app/developer/dashboard",
        "prefix_hint": "",
    },
    "quickbooks_client_secret": {
        "display_name": "QuickBooks Client Secret",
        "description": "OAuth Client Secret from Intuit Developer Portal",
        "env_var": "QUICKBOOKS_CLIENT_SECRET",
        "required": False,
        "category": "finance",
        "url": "https://developer.intuit.com/app/developer/dashboard",
        "prefix_hint": "",
    },
    "quickbooks_refresh_token": {
        "display_name": "QuickBooks Refresh Token",
        "description": "OAuth refresh token — auto-generated after first auth, valid 100 days",
        "env_var": "QUICKBOOKS_REFRESH_TOKEN",
        "required": False,
        "category": "finance",
        "url": "https://developer.intuit.com/app/developer/dashboard",
        "prefix_hint": "",
    },
    "quickbooks_realm_id": {
        "display_name": "QuickBooks Realm ID",
        "description": "Company ID returned after OAuth — identifies which QuickBooks company to access",
        "env_var": "QUICKBOOKS_REALM_ID",
        "required": False,
        "category": "finance",
        "url": "https://developer.intuit.com/app/developer/dashboard",
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

