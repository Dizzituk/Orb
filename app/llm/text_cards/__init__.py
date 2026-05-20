# FILE: app/llm/text_cards/__init__.py
"""
Text card rendering package.

Deterministic Pillow-based renderer for quote graphics, social posts, and
other "text on a styled background" outputs. Bypasses diffusion image
models entirely — they cannot render exact text reliably.

Public surface:
    render_text_card(spec)   -> dict (image_gen-compatible result)
    extract_card_content(...) -> dict | None
    is_text_card_request(...) -> bool

v1.0 (2026-05-01): Initial implementation. Triggered by image_router when
                    the user wants a text-on-background image and a quote
                    is recoverable from recent context (HTML artifact or
                    prior assistant message).
"""
from __future__ import annotations

from app.llm.text_cards.renderer import render_text_card
from app.llm.text_cards.classifier import is_text_card_request
from app.llm.text_cards.extractor import extract_card_content
from app.llm.text_cards.styles import get_style, list_styles

__all__ = [
    "render_text_card",
    "is_text_card_request",
    "extract_card_content",
    "get_style",
    "list_styles",
]
