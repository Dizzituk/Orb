# Purpose: registry utils 4
# Called-by: app.investments.chat_router, app.providers.registry
# Depends-on: app.providers._registry_utils_3, app.providers.registry
# Last-renovated: 2026-06-11
from __future__ import annotations
from app.providers._registry_utils_3 import LlmUsage
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class LlmCallStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    INVALID_REQUEST = "invalid_request"
    TOOL_ERROR = "tool_error"
    TOOL_LOOP_EXCEEDED = "tool_loop_exceeded"

@dataclass
class LlmCallResult:
    status: LlmCallStatus
    provider_id: str
    model_id: str
    content: str = ""
    usage: LlmUsage = field(default_factory=LlmUsage)
    raw_response: Optional[dict] = None
    error_message: Optional[str] = None

    def is_success(self) -> bool:
        return self.status == LlmCallStatus.SUCCESS

_registry = None  # Module-level singleton (initialised lazily by get_provider_registry)


def get_provider_registry() -> ProviderRegistry:
    from .registry import ProviderRegistry
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry

def is_provider_available(provider_id: str) -> bool:
    return get_provider_registry().is_provider_available(provider_id)
