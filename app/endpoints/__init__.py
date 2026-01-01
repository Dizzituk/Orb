# FILE: app/endpoints/__init__.py
"""
Endpoint routers for Orb API.

Refactored from main.py for better organization.
"""

from fastapi import APIRouter

from .chat import router as chat_router
from .chat_attachments import router as chat_attachments_router
from .direct_llm import router as direct_llm_router

# Combined router for all endpoints
router = APIRouter()
router.include_router(chat_router)
router.include_router(chat_attachments_router)
router.include_router(direct_llm_router)

__all__ = [
    "router",
    "chat_router",
    "chat_attachments_router",
    "direct_llm_router",
]
