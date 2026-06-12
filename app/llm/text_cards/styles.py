# FILE: app/llm/text_cards/styles.py
# Purpose: Style presets for text card rendering.
# Called-by: app.llm.text_cards.renderer
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Style presets for text card rendering.

Each preset describes how to paint a 1080x1080 (or other) text card:
    - background: gradient stops (linear, diagonal) + optional radial highlights
    - text: colour, font weight preference, max font size, line spacing
    - layout: padding, alignment, max line width as fraction of canvas

Adding a new style = add one dict here. The renderer is style-agnostic and
reads everything from these specs.

v1.0 (2026-05-01): Initial four presets — premium, light_gradient, dark,
                   minimal. Mirrors the design vocabulary the chat layer
                   already uses ("light gradient", "premium", "dark mode").
"""
from __future__ import annotations

from typing import Any


# Each style is a flat dict — no nested classes, no inheritance. Keep it
# trivially serialisable so we can echo the spec back in logs / debug.
_STYLES: dict[str, dict[str, Any]] = {
    # The "make it premium" target — warm cream → cool grey-blue gradient,
    # bold near-black text, soft radial highlights for depth. This is the
    # visual identity used in the original Instagram-square HTML artifact.
    "premium": {
        "background": {
            "type": "linear_gradient",
            "angle_deg": 135,
            "stops": [
                (0.00, (247, 244, 239)),  # warm cream
                (0.35, (232, 221, 209)),  # warm beige
                (0.70, (217, 230, 234)),  # cool pale blue
                (1.00, (245, 239, 230)),  # warm cream again
            ],
            "radial_highlights": [
                # (cx_frac, cy_frac, radius_frac, alpha_0_255)
                (0.10, 0.10, 0.45, 115),
                (0.90, 0.90, 0.40, 80),
            ],
        },
        "text": {
            "colour": (17, 17, 17),
            "weight_preference": ["arialbd.ttf", "arial.ttf", "segoeuib.ttf"],
            "max_font_size": 96,
            "min_font_size": 44,
            "line_spacing": 1.08,
            "letter_spacing_px": -2,
        },
        "layout": {
            "padding_frac": 0.10,
            "max_text_width_frac": 0.80,
            "alignment": "center",
            "vertical_align": "center",
        },
    },

    # The simpler "light gradient" — closer to the user's original verbal
    # spec ("nice light gradient + black punchy text"). Less designed, more
    # immediate. Single warm-to-cool sweep, no radial highlights.
    "light_gradient": {
        "background": {
            "type": "linear_gradient",
            "angle_deg": 135,
            "stops": [
                (0.00, (250, 248, 244)),
                (1.00, (224, 232, 240)),
            ],
            "radial_highlights": [],
        },
        "text": {
            "colour": (15, 15, 15),
            "weight_preference": ["arialbd.ttf", "arial.ttf", "segoeuib.ttf"],
            "max_font_size": 92,
            "min_font_size": 44,
            "line_spacing": 1.10,
            "letter_spacing_px": -1,
        },
        "layout": {
            "padding_frac": 0.10,
            "max_text_width_frac": 0.80,
            "alignment": "center",
            "vertical_align": "center",
        },
    },

    # Dark mode — deep charcoal gradient with subtle blue undertone, near-
    # white text. Use for serious / political / late-night-feed posts.
    "dark": {
        "background": {
            "type": "linear_gradient",
            "angle_deg": 135,
            "stops": [
                (0.00, (18, 18, 22)),
                (0.50, (24, 28, 36)),
                (1.00, (14, 16, 20)),
            ],
            "radial_highlights": [
                (0.20, 0.15, 0.35, 35),
            ],
        },
        "text": {
            "colour": (242, 242, 244),
            "weight_preference": ["arialbd.ttf", "arial.ttf", "segoeuib.ttf"],
            "max_font_size": 96,
            "min_font_size": 44,
            "line_spacing": 1.08,
            "letter_spacing_px": -2,
        },
        "layout": {
            "padding_frac": 0.10,
            "max_text_width_frac": 0.80,
            "alignment": "center",
            "vertical_align": "center",
        },
    },

    # Minimal — flat off-white, pure black text. No gradient, no flourish.
    # Useful when you want the words to do all the work.
    "minimal": {
        "background": {
            "type": "flat",
            "colour": (248, 248, 246),
            "radial_highlights": [],
        },
        "text": {
            "colour": (10, 10, 10),
            "weight_preference": ["arialbd.ttf", "arial.ttf", "segoeuib.ttf"],
            "max_font_size": 92,
            "min_font_size": 44,
            "line_spacing": 1.12,
            "letter_spacing_px": -1,
        },
        "layout": {
            "padding_frac": 0.12,
            "max_text_width_frac": 0.78,
            "alignment": "center",
            "vertical_align": "center",
        },
    },
}


# Aliases so user prompts that say "gradient" or "white background" find
# the right preset without exact matching.
_ALIASES: dict[str, str] = {
    "default": "premium",
    "gradient": "light_gradient",
    "light": "light_gradient",
    "white": "minimal",
    "flat": "minimal",
    "clean": "minimal",
    "dark_mode": "dark",
    "black": "dark",
}


def get_style(name: str | None) -> dict[str, Any]:
    """Resolve a style name (or alias) to a full style spec.

    Returns the 'premium' preset if the name is None, empty, or unknown.
    Style names are matched case-insensitively.
    """
    if not name:
        return _STYLES["premium"]
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in _STYLES:
        return _STYLES[key]
    if key in _ALIASES:
        return _STYLES[_ALIASES[key]]
    return _STYLES["premium"]


def list_styles() -> list[str]:
    """Return all canonical style names (excluding aliases)."""
    return sorted(_STYLES.keys())


__all__ = ["get_style", "list_styles"]
