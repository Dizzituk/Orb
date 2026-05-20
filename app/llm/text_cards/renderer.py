# FILE: app/llm/text_cards/renderer.py
"""
Pillow-based text card renderer.

Takes a structured spec and produces a PNG that exactly matches what was
asked for — no diffusion, no hallucinated text, no random "DISCIPLINE
TODAY / SUCCESS TOMORROW" filler. Returns a dict shaped identically to
image_gen.generate_image so the router can treat it as a drop-in result.

Pipeline:
    1. Build canvas with chosen background (gradient or flat)
    2. Apply optional radial highlights for depth
    3. Find the best available bold font on disk
    4. Auto-fit text to canvas (try max_font_size, shrink if it overflows)
    5. Wrap text into lines respecting max width
    6. Centre vertically and horizontally
    7. Save to the resolved image output dir

v1.0 (2026-05-01): Initial implementation. Gradient generation is pure
                   Python — slow on huge canvases but fine for 1080x1080
                   (~30ms on a modern laptop).
"""
from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.llm.image_output_dir import get_image_output_dir
from app.llm.text_cards.styles import get_style
from app.llm.text_cards._gradient import paint_background
from app.llm.text_cards._typesetter import layout_text, find_font_path

logger = logging.getLogger(__name__)


def render_text_card(
    text: str,
    style_name: str | None = None,
    width: int = 1080,
    height: int = 1080,
    output_filename: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> Optional[dict]:
    """Render a text card to PNG.

    Args:
        text: The main quote / headline text. Newlines are respected as
              hard line breaks; otherwise text wraps automatically.
        style_name: Style preset name. See styles.list_styles() for options.
                    None / unknown → 'premium'.
        width, height: Canvas size in pixels. Default 1080x1080 (Instagram).
        output_filename: Override generated filename.
        subtitle: Optional smaller text shown below the main quote.

    Returns:
        Dict matching image_gen.generate_image's schema, or None on failure.
    """
    if not text or not text.strip():
        logger.warning("[text_card_renderer] Empty text — refusing to render")
        return None

    style = get_style(style_name)

    try:
        from PIL import Image
    except ImportError:
        logger.error("[text_card_renderer] Pillow not installed")
        return None

    try:
        canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
        paint_background(canvas, style["background"])
        _draw_text_block(canvas, text, subtitle, style)
    except Exception as e:
        logger.error("[text_card_renderer] Compose failed: %s", e, exc_info=True)
        return None

    try:
        return _save_to_output(canvas, text, output_filename)
    except Exception as e:
        logger.error("[text_card_renderer] Save failed: %s", e, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Text block rendering
# ---------------------------------------------------------------------------

def _draw_text_block(canvas, text: str, subtitle: str | None, style: dict[str, Any]) -> None:
    """Lay out the main quote (and optional subtitle) onto the canvas in place."""
    from PIL import ImageDraw

    text_spec = style["text"]
    layout_spec = style["layout"]

    width, height = canvas.size
    pad = int(min(width, height) * layout_spec["padding_frac"])
    max_text_width = int(width * layout_spec["max_text_width_frac"])

    font_path = find_font_path(text_spec["weight_preference"])
    if not font_path:
        logger.warning(
            "[text_card_renderer] No bold font found from %s; falling back to PIL default",
            text_spec["weight_preference"],
        )

    main_layout = layout_text(
        text=text.strip(),
        font_path=font_path,
        max_width=max_text_width,
        max_font_size=text_spec["max_font_size"],
        min_font_size=text_spec["min_font_size"],
        line_spacing=text_spec["line_spacing"],
        letter_spacing_px=text_spec["letter_spacing_px"],
    )

    sub_layout = None
    if subtitle and subtitle.strip():
        sub_max_size = max(
            text_spec["min_font_size"],
            int(main_layout["font_size"] * 0.55),
        )
        sub_min_size = max(20, int(text_spec["min_font_size"] * 0.6))
        sub_layout = layout_text(
            text=subtitle.strip(),
            font_path=font_path,
            max_width=max_text_width,
            max_font_size=sub_max_size,
            min_font_size=sub_min_size,
            line_spacing=text_spec["line_spacing"],
            letter_spacing_px=int(text_spec["letter_spacing_px"] * 0.5),
        )

    # Vertical centring — combined block height of main + gap + subtitle.
    sub_gap = int(main_layout["font_size"] * 0.4) if sub_layout else 0
    total_h = main_layout["block_height"] + sub_gap + (
        sub_layout["block_height"] if sub_layout else 0
    )
    available_h = height - 2 * pad
    if total_h > available_h:
        # Shrink main text further if combined block doesn't fit. Rare —
        # only happens with very long subtitles — but worth handling.
        scale = available_h / total_h
        new_max = max(text_spec["min_font_size"], int(main_layout["font_size"] * scale))
        main_layout = layout_text(
            text=text.strip(),
            font_path=font_path,
            max_width=max_text_width,
            max_font_size=new_max,
            min_font_size=text_spec["min_font_size"],
            line_spacing=text_spec["line_spacing"],
            letter_spacing_px=text_spec["letter_spacing_px"],
        )
        total_h = main_layout["block_height"] + sub_gap + (
            sub_layout["block_height"] if sub_layout else 0
        )

    y = (height - total_h) // 2
    draw = ImageDraw.Draw(canvas)

    _draw_lines(draw, main_layout, text_spec["colour"], width, y)
    y += main_layout["block_height"] + sub_gap

    if sub_layout:
        # Subtitle uses a slightly muted colour by default — same hue, ~92% alpha
        # baked into the RGB.
        muted = _muted_colour(text_spec["colour"], 0.78)
        _draw_lines(draw, sub_layout, muted, width, y)


def _draw_lines(draw, layout: dict[str, Any], colour: tuple, canvas_w: int, y: int) -> None:
    """Draw each laid-out line, horizontally centred on the canvas."""
    for line in layout["lines"]:
        line_w = line["width"]
        x = (canvas_w - line_w) // 2
        # Letter-spaced text: render each glyph individually if spacing != 0.
        # Pillow's ImageDraw.text doesn't support tracking, so we walk the
        # string. For zero/tiny spacing the simple path is fine and faster.
        if abs(layout["letter_spacing_px"]) <= 1:
            draw.text((x, y), line["text"], font=layout["font"], fill=colour)
        else:
            _draw_tracked(draw, line["text"], x, y, layout["font"],
                          colour, layout["letter_spacing_px"])
        y += line["height"]


def _draw_tracked(draw, text: str, x: int, y: int, font, colour: tuple, spacing_px: int) -> None:
    """Draw text glyph-by-glyph with custom letter spacing."""
    for ch in text:
        draw.text((x, y), ch, font=font, fill=colour)
        bbox = font.getbbox(ch)
        glyph_w = bbox[2] - bbox[0]
        x += glyph_w + spacing_px


def _muted_colour(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """Lighten/darken a colour toward 50% grey for muted accents."""
    r, g, b = rgb
    target = 128
    return (
        int(r + (target - r) * (1 - factor)),
        int(g + (target - g) * (1 - factor)),
        int(b + (target - b) * (1 - factor)),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _save_to_output(canvas, text: str, output_filename: str | None) -> dict:
    """Persist the canvas to the image output dir; return image_gen-compatible dict."""
    import base64
    from io import BytesIO

    output_dir = get_image_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not output_filename:
        h = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:8]
        ts = datetime.now(timezone.utc).strftime("%H%M%S")
        output_filename = f"card-{h}-{ts}.png"

    filepath = output_dir / output_filename

    # PNG for crisp text. JPEG would compress the edges and look fuzzy on phones.
    canvas.save(filepath, format="PNG", optimize=True)

    # Build the same shape image_gen.generate_image returns so the router
    # can plug us in without any branching downstream.
    buf = BytesIO()
    canvas.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    b64 = base64.b64encode(image_bytes).decode("ascii")

    logger.info(
        "[text_card_renderer] Saved %s (%d bytes)",
        filepath, len(image_bytes),
    )

    return {
        "path": str(filepath),
        "filename": output_filename,
        "size_bytes": len(image_bytes),
        "base64_data": b64,
        "mime_type": "image/png",
        "prompt": text,
        "text": text,
    }


__all__ = ["render_text_card"]
