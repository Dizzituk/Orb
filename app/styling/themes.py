# FILE: app/styling/themes.py
"""
Shared theme presets for all styled file creators (docx, pdf, xlsx, html).

Two presets:
  - astra_default : modern, slightly opinionated palette. For client-facing
    documents, proposals, reports, summaries. Adds colour and visual hierarchy.
  - astra_minimal : black on white, no decoration. For legal evidence,
    formal letters, anything where 'designed' would undercut credibility.

Auto-selection: pick_theme(name_or_path) inspects the filename/project name
for keywords like 'legal', 'evidence', 'letter', 'solicitor' and chooses
astra_minimal automatically. Everything else gets astra_default.
"""
from __future__ import annotations

from typing import Dict, Any


ASTRA_DEFAULT: Dict[str, Any] = {
    "name": "astra_default",
    # Fonts (system fallbacks if Inter/JetBrains Mono not installed)
    "font_heading": "Inter, Calibri, Arial, sans-serif",
    "font_body": "Inter, Calibri, Arial, sans-serif",
    "font_mono": "JetBrains Mono, Consolas, Courier New, monospace",
    # Single-name fallbacks for libraries that don't accept stacks
    "font_heading_single": "Calibri",
    "font_body_single": "Calibri",
    "font_mono_single": "Consolas",
    # Palette (hex without '#' for openpyxl compatibility, with '#' for HTML/PDF)
    "colour_primary": "0B5FFF",
    "colour_accent": "00C2A8",
    "colour_text": "1A1F2E",
    "colour_muted": "6B7280",
    "colour_rule": "E5E7EB",
    "colour_zebra": "F8FAFC",
    "colour_bg": "FFFFFF",
    # Layout
    "page_margin_cm": 2.0,
    "max_width_px": 820,
    # Type scale
    "heading_sizes_pt": [22, 16, 13, 11],  # H1..H4
    "body_size_pt": 11,
    "small_size_pt": 9,
    # Tables
    "table_header_fill": "0B5FFF",
    "table_header_text": "FFFFFF",
    "table_zebra": True,
    # Decoration
    "show_cover_page": True,
    "show_page_numbers": True,
    "show_generated_footer": True,
}


ASTRA_MINIMAL: Dict[str, Any] = {
    "name": "astra_minimal",
    "font_heading": "Times New Roman, Georgia, serif",
    "font_body": "Times New Roman, Georgia, serif",
    "font_mono": "Consolas, Courier New, monospace",
    "font_heading_single": "Times New Roman",
    "font_body_single": "Times New Roman",
    "font_mono_single": "Consolas",
    "colour_primary": "000000",
    "colour_accent": "000000",
    "colour_text": "000000",
    "colour_muted": "555555",
    "colour_rule": "CCCCCC",
    "colour_zebra": "F4F4F4",
    "colour_bg": "FFFFFF",
    "page_margin_cm": 2.5,
    "max_width_px": 720,
    "heading_sizes_pt": [16, 13, 12, 11],
    "body_size_pt": 11,
    "small_size_pt": 9,
    "table_header_fill": "E0E0E0",
    "table_header_text": "000000",
    "table_zebra": False,
    "show_cover_page": False,
    "show_page_numbers": True,
    "show_generated_footer": True,
}


THEMES: Dict[str, Dict[str, Any]] = {
    "astra_default": ASTRA_DEFAULT,
    "astra_minimal": ASTRA_MINIMAL,
    # Aliases
    "default": ASTRA_DEFAULT,
    "minimal": ASTRA_MINIMAL,
}


# Filename / project keywords that trigger the minimal theme automatically.
# Anything legal-or-formal-looking goes plain.
_MINIMAL_KEYWORDS = (
    "legal", "evidence", "solicitor", "lawyer", "court", "tribunal",
    "letter", "formal", "statement", "affidavit", "witness",
    "complaint", "grievance", "claim", "case",
)


def pick_theme(name_or_path: str = "", explicit: str = "auto") -> Dict[str, Any]:
    """Resolve a theme by explicit name or auto-detect from a filename/path.

    Args:
        name_or_path: filename or path to inspect for keywords
        explicit: theme name, or 'auto' to detect from name_or_path

    Returns:
        A theme dict (one of THEMES values).
    """
    if explicit and explicit != "auto":
        return THEMES.get(explicit.lower(), ASTRA_DEFAULT)

    haystack = (name_or_path or "").lower()
    for kw in _MINIMAL_KEYWORDS:
        if kw in haystack:
            return ASTRA_MINIMAL

    return ASTRA_DEFAULT


def hex_with_hash(value: str) -> str:
    """Return a hex colour with a leading '#' for HTML/CSS use."""
    value = (value or "").lstrip("#")
    return f"#{value}" if value else "#000000"
