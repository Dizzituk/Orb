# FILE: app/content/production/static_gen.py
"""
Static Content Generator (Spec Section 7.5).

Produces non-video content formats using deterministic templates:
- Instagram carousels (5-8 slides via Pillow)
- Blog post HTML formatting
- Thumbnail generation

All template-driven — AI provides content, rendering is deterministic.
"""
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/content/output")


# ─── CAROUSEL DEFAULTS ───

CAROUSEL_DEFAULTS = {
    "width": 1080,
    "height": 1350,  # 4:5 aspect ratio
    "bg_colour": (26, 26, 46),  # Dark navy (#1A1A2E)
    "text_colour": (255, 255, 255),
    "accent_colour": (233, 69, 96),  # #E94560
    "font_size_title": 48,
    "font_size_body": 36,
    "font_size_small": 24,
    "padding": 80,
    "line_spacing": 12,
}


def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Get a font, falling back to default if custom not available."""
    # Try common system fonts
    font_names = [
        "arial.ttf", "Arial.ttf",
        "arialbd.ttf", "Arial Bold.ttf",
        "calibri.ttf", "Calibri.ttf",
        "segoeui.ttf", "Segoe UI.ttf",
    ]
    if bold:
        font_names = [
            "arialbd.ttf", "Arial Bold.ttf",
            "calibrib.ttf", "Calibri Bold.ttf",
            "segoeuib.ttf", "Segoe UI Bold.ttf",
        ] + font_names

    for name in font_names:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue

    # Fallback to default
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except (OSError, IOError):
        return ImageFont.load_default()


def _wrap_text(
    draw: ImageDraw.Draw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> List[str]:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        width = bbox[2] - bbox[0]

        if width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    return lines


# ═══════════════════════════════════════════════════
# CAROUSEL GENERATION
# ═══════════════════════════════════════════════════

def generate_carousel(
    piece_id: str,
    slides_content: List[Dict[str, str]],
    style: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Generate Instagram carousel slides.

    slides_content: List of dicts with 'title' and 'body' keys.
    Returns list of file paths for generated slide images.

    Each slide has:
    - Title text (bold, larger)
    - Body text (regular, wrapped)
    - Slide number indicator
    - Consistent branding
    """
    s = {**CAROUSEL_DEFAULTS, **(style or {})}
    output_dir = OUTPUT_DIR / piece_id / "carousel"
    os.makedirs(output_dir, exist_ok=True)

    font_title = _get_font(s["font_size_title"], bold=True)
    font_body = _get_font(s["font_size_body"])
    font_small = _get_font(s["font_size_small"])

    total_slides = len(slides_content)
    paths = []

    for i, slide_data in enumerate(slides_content):
        img = Image.new("RGB", (s["width"], s["height"]), s["bg_colour"])
        draw = ImageDraw.Draw(img)

        pad = s["padding"]
        max_text_width = s["width"] - (pad * 2)
        y_cursor = pad

        # Accent bar at top
        draw.rectangle(
            [(0, 0), (s["width"], 6)],
            fill=s["accent_colour"],
        )
        y_cursor += 30

        # Slide number
        slide_num = f"{i + 1}/{total_slides}"
        draw.text(
            (s["width"] - pad, y_cursor),
            slide_num,
            font=font_small,
            fill=s["accent_colour"],
            anchor="ra",
        )
        y_cursor += 50

        # Title
        title = slide_data.get("title", "")
        if title:
            title_lines = _wrap_text(draw, title, font_title, max_text_width)
            for line in title_lines:
                draw.text(
                    (pad, y_cursor),
                    line,
                    font=font_title,
                    fill=s["text_colour"],
                )
                bbox = draw.textbbox((0, 0), line, font=font_title)
                y_cursor += (bbox[3] - bbox[1]) + s["line_spacing"]
            y_cursor += 30

        # Accent divider line
        draw.rectangle(
            [(pad, y_cursor), (pad + 80, y_cursor + 4)],
            fill=s["accent_colour"],
        )
        y_cursor += 30

        # Body text
        body = slide_data.get("body", "")
        if body:
            body_lines = _wrap_text(draw, body, font_body, max_text_width)
            for line in body_lines:
                draw.text(
                    (pad, y_cursor),
                    line,
                    font=font_body,
                    fill=s["text_colour"],
                )
                bbox = draw.textbbox((0, 0), line, font=font_body)
                y_cursor += (bbox[3] - bbox[1]) + s["line_spacing"]

        # Bottom accent bar
        draw.rectangle(
            [(0, s["height"] - 6), (s["width"], s["height"])],
            fill=s["accent_colour"],
        )

        # Save
        slide_path = str(output_dir / f"slide_{i + 1:02d}.png")
        img.save(slide_path, "PNG", quality=95)
        paths.append(slide_path)

    logger.info(
        f"[static_gen] Generated {len(paths)} carousel slides for {piece_id}"
    )
    return paths


# ═══════════════════════════════════════════════════
# BLOG POST HTML
# ═══════════════════════════════════════════════════

def format_blog_html(
    title: str,
    body_markdown: str,
    author: str = "Taz",
    series_name: Optional[str] = None,
) -> str:
    """
    Convert draft text to basic HTML blog format.
    Simple, clean, SEO-friendly structure.
    """
    # Basic markdown-to-HTML conversion
    import re

    html_body = body_markdown

    # Headers
    html_body = re.sub(
        r"^### (.+)$", r"<h3>\1</h3>", html_body, flags=re.MULTILINE
    )
    html_body = re.sub(
        r"^## (.+)$", r"<h2>\1</h2>", html_body, flags=re.MULTILINE
    )
    html_body = re.sub(
        r"^# (.+)$", r"<h1>\1</h1>", html_body, flags=re.MULTILINE
    )

    # Bold and italic
    html_body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html_body)
    html_body = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html_body)

    # Paragraphs (double newlines)
    paragraphs = html_body.split("\n\n")
    html_body = "\n".join(
        f"<p>{p.strip()}</p>" if not p.strip().startswith("<h") else p.strip()
        for p in paragraphs
        if p.strip()
    )

    series_tag = (
        f'<span class="series-tag">{series_name}</span>'
        if series_name else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
</head>
<body>
<article>
{series_tag}
<h1>{title}</h1>
<p class="author">By {author}</p>
{html_body}
</article>
</body>
</html>"""


def save_blog_html(
    piece_id: str,
    title: str,
    body: str,
    series_name: Optional[str] = None,
) -> str:
    """Generate and save blog post HTML. Returns file path."""
    html = format_blog_html(title, body, series_name=series_name)
    output_dir = OUTPUT_DIR / piece_id
    os.makedirs(output_dir, exist_ok=True)
    path = str(output_dir / "blog_post.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"[static_gen] Saved blog HTML: {path}")
    return path
