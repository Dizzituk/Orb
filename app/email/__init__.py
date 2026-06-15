# FILE: app/email/__init__.py
# Purpose: Provider-agnostic email surface — the factory that picks the
#          configured EmailProvider (gmail | proton | registered test fakes).
# Called-by: app.tools.email_tools
# Depends-on: app.email.provider, app.email.gmail_provider, app.email.proton_provider
# Last-renovated: 2026-06-12
"""
Email module (provider-agnostic).

Config: ASTRA_EMAIL_PROVIDER env var, falling back to the settings-DB key
'email_provider', defaulting to 'gmail'. Swapping provider is a config
change — zero changes to intents or UI (that seam is the whole point).

register_provider() lets tests (and future providers) plug into the same
factory without touching this file's built-ins.

NOTE: app/email_service (Proton Bridge IMAP/SMTP plumbing, 2026-03-14) is a
separate, older module — the ProtonBridgeProvider here adapts it onto the
contract rather than duplicating it.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Dict

from app.email.provider import EmailProvider  # noqa: F401  (re-export for callers)

logger = logging.getLogger(__name__)

_PROVIDER_FACTORIES: Dict[str, Callable[[], EmailProvider]] = {}
_instances: Dict[str, EmailProvider] = {}


def register_provider(name: str, factory: Callable[[], EmailProvider]) -> None:
    """Plug a provider into the factory (used by tests to prove the seam)."""
    _PROVIDER_FACTORIES[name.lower()] = factory
    _instances.pop(name.lower(), None)


def _builtin_factories() -> None:
    if "gmail" not in _PROVIDER_FACTORIES:
        def _gmail() -> EmailProvider:
            from app.email.gmail_provider import GmailProvider
            return GmailProvider()
        _PROVIDER_FACTORIES["gmail"] = _gmail
    if "proton" not in _PROVIDER_FACTORIES:
        def _proton() -> EmailProvider:
            from app.email.proton_provider import ProtonBridgeProvider
            return ProtonBridgeProvider()
        _PROVIDER_FACTORIES["proton"] = _proton


def configured_provider_name() -> str:
    name = (os.getenv("ASTRA_EMAIL_PROVIDER") or "").strip().lower()
    if not name:
        try:
            from app.settings.service import get_setting_value
            name = (get_setting_value("email_provider") or "").strip().lower()
        except Exception:
            name = ""
    return name or "gmail"


def get_provider(name: str | None = None) -> EmailProvider:
    """The configured provider (cached one instance per provider name)."""
    _builtin_factories()
    key = (name or configured_provider_name()).lower()
    if key not in _PROVIDER_FACTORIES:
        raise ValueError(
            f"unknown email provider '{key}' — known: {sorted(_PROVIDER_FACTORIES)}"
        )
    if key not in _instances:
        _instances[key] = _PROVIDER_FACTORIES[key]()
    return _instances[key]
