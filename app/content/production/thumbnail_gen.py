# FILE: app/content/production/thumbnail_gen.py
"""
Thumbnail Generator — extract frame + overlay title text.

Creates YouTube-ready thumbnails by:
1. Extracting the most visually interesting frame via FFmpeg
2. Overlaying the video title in bold Impact font
3. Adding a semi-transparent dark gradient for readability
4. Outputting at YouTube's recommended 1280x720

No API calls needed — pure FFmpeg + Pillow.
"""
import os
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont, ImageFilter

logger = logging.getLogger(__name__)

# YouTube thumbnail specs
THUMB_WIDTH = 1280
THUMB_HEIGHT = 720

# Font config — Impact is the classic YouTube thumbnail font
FONT_PATHS = [
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _find_font(size: int) -> ImageFont.FreeTypeFont:
    """Find a bold font available on the system."""
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def _extract_frame(
    video_path: str,
    output_path: str,
    timestamp: Optional[float] = None,
) -> bool:
    """Extract a single frame from a video using FFmpeg.

    If no timestamp given, picks a frame at 30% through
    (usually more interesting than the first frame).
    """
    if timestamp is None:
        # Get duration and pick 30% mark
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "json", video_path,
                ],
                capture_output=True, text=True, timeout=15,
            )
            data = json.loads(result.stdout)
            duration = float(data["format"]["duration"])
            timestamp = duration * 0.3
        except Exception:
            timestamp = 2.0  # fallback to 2 seconds

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(timestamp),
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",
                output_path,
            ],
            capture_output=True, timeout=15,
        )
        return os.path.exists(output_path)
    except Exception as e:
        logger.error("[thumbnail] Frame extraction failed: %s", e)
        return False


def _wrap_text(
    text: str, font: ImageFont.FreeTypeFont, max_width: int,
) -> list:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test = (current_line + " " + word).strip()
        bbox = font.getbbox(test)
        if bbox[2] <= max_width:
            current_line = test
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


def generate_thumbnail(
    video_path: str,
    title: str,
    output_path: str,
    frame_timestamp: Optional[float] = None,
    accent_color: str = "#7c3aed",
) -> Optional[str]:
    """
    Generate a YouTube thumbnail from a video.

    1. Extract frame from video
    2. Resize/crop to 1280x720
    3. Add dark gradient overlay
    4. Overlay title text in Impact
    5. Save as PNG

    Returns output path or None on failure.
    """
    # Step 1: Extract frame
    temp_frame = output_path.replace(".png", "_frame.jpg")
    if not _extract_frame(video_path, temp_frame, frame_timestamp):
        logger.error("[thumbnail] Could not extract frame")
        return None

    try:
        img = Image.open(temp_frame).convert("RGB")

        # Step 2: Resize to cover 1280x720 then center-crop
        img = _resize_cover(img, THUMB_WIDTH, THUMB_HEIGHT)

        # Step 3: Add dark gradient at bottom for text readability
        img = _add_gradient(img)

        # Step 4: Add accent bar at top
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            [(0, 0), (THUMB_WIDTH, 6)],
            fill=accent_color,
        )

        # Step 5: Overlay title text
        _draw_title(draw, title, THUMB_WIDTH, THUMB_HEIGHT)

        # Step 6: Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, "PNG", quality=95)

        # Cleanup temp frame
        try:
            os.remove(temp_frame)
        except OSError:
            pass

        logger.info("[thumbnail] Generated: %s", output_path)
        return output_path

    except Exception as e:
        logger.error("[thumbnail] Generation failed: %s", e)
        return None


def _resize_cover(
    img: Image.Image, target_w: int, target_h: int,
) -> Image.Image:
    """Resize image to cover target dimensions, then center crop."""
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))

    return img


def _add_gradient(img: Image.Image) -> Image.Image:
    """Add a dark gradient on the bottom half for text readability."""
    gradient = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(gradient)

    w, h = img.size
    # Gradient from transparent at middle to dark at bottom
    for y in range(h // 3, h):
        alpha = int(200 * (y - h // 3) / (h - h // 3))
        draw.line([(0, y), (w, y)], fill=(0, 0, 0, alpha))

    img = img.convert("RGBA")
    img = Image.alpha_composite(img, gradient)
    return img.convert("RGB")


def _draw_title(
    draw: ImageDraw.Draw,
    title: str,
    width: int,
    height: int,
) -> None:
    """Draw the title text with outline for maximum readability."""
    # Size the font to fill roughly 70% of width
    font_size = 72
    font = _find_font(font_size)

    # Wrap text
    max_text_width = int(width * 0.85)
    lines = _wrap_text(title.upper(), font, max_text_width)

    # If more than 3 lines, reduce font size
    while len(lines) > 3 and font_size > 36:
        font_size -= 4
        font = _find_font(font_size)
        lines = _wrap_text(title.upper(), font, max_text_width)

    # Calculate total text height
    line_height = font_size + 8
    total_height = len(lines) * line_height

    # Position text at bottom third
    y_start = height - total_height - 40

    for i, line in enumerate(lines):
        bbox = font.getbbox(line)
        text_w = bbox[2] - bbox[0]
        x = (width - text_w) // 2
        y = y_start + i * line_height

        # Draw outline (black stroke)
        outline_width = 3
        for ox in range(-outline_width, outline_width + 1):
            for oy in range(-outline_width, outline_width + 1):
                if ox == 0 and oy == 0:
                    continue
                draw.text(
                    (x + ox, y + oy), line,
                    font=font, fill=(0, 0, 0),
                )

        # Draw main text (white)
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
