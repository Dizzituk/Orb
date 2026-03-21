# FILE: app/content/video_pipeline/thumbnail.py
"""
AI Thumbnail Generator — GPT Image background + Pillow text overlay.

1. Extracts a still from the avatar segment
2. Sends it to GPT Image 1 with a thumbnail design prompt
3. Overlays title text with Pillow (pixel-perfect control)
4. NO text in the AI image — text cut-off is eliminated
"""
import base64
import json
import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


async def generate_ai_thumbnail(
    scene_plan, edl, title: str, output_path: str,
) -> bool:
    """Generate a YouTube thumbnail using AI image generation."""
    try:
        # Find the first avatar segment's video file
        avatar_path = None
        for i, seg in enumerate(scene_plan.segments):
            if seg.requires_avatar and i < len(edl.segments):
                avatar_path = edl.segments[i].source_path
                break

        if not avatar_path or not os.path.exists(avatar_path):
            logger.warning("[thumbnail] No avatar segment found")
            return False

        # Get duration and extract frame at 40%
        try:
            r = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "json", avatar_path,
                ],
                capture_output=True, text=True, timeout=10,
            )
            dur = float(json.loads(r.stdout)["format"]["duration"])
            seek = dur * 0.4
        except Exception:
            seek = 1.5

        temp_frame = output_path.replace(".png", "_avatar_frame.png")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(seek),
                "-i", avatar_path,
                "-frames:v", "1",
                "-q:v", "2",
                temp_frame,
            ],
            capture_output=True, timeout=10,
        )
        if not os.path.exists(temp_frame):
            return False

        # Encode avatar frame as base64
        with open(temp_frame, "rb") as f:
            avatar_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = (
            "Create a professional YouTube thumbnail background "
            "(1280x720). Feature the AI presenter from the attached "
            "image on the RIGHT side of the frame. The LEFT side "
            "should be empty dark space for text overlay later. "
            "DO NOT add any text, titles, or words to the image. "
            "STYLE: Dramatic cinematic lighting, dark tech background "
            "with subtle glowing elements, circuit patterns, "
            "depth of field. Professional and clean."
        )

        from openai import AsyncOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("[thumbnail] No OPENAI_API_KEY")
            return False

        client = AsyncOpenAI(api_key=api_key)
        response = await client.images.edit(
            model="gpt-image-1",
            image=open(temp_frame, "rb"),
            prompt=prompt,
            size="1536x1024",
        )

        if response.data and response.data[0].b64_json:
            img_bytes = base64.b64decode(response.data[0].b64_json)
            from PIL import Image, ImageDraw, ImageFont
            import io
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize((1280, 720), Image.LANCZOS)

            draw = ImageDraw.Draw(img)
            safe_title = title.replace('"', '').strip()
            font = None
            for fp in [
                "C:/Windows/Fonts/impact.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
            ]:
                if os.path.exists(fp):
                    font = ImageFont.truetype(fp, 72)
                    break
            if not font:
                font = ImageFont.load_default()

            # Word wrap into lines that fit left 55% of frame
            title_upper = safe_title.upper()
            max_w = int(1280 * 0.55)
            words = title_upper.split()
            lines = []
            current = ""
            for word in words:
                test = (current + " " + word).strip()
                bbox = font.getbbox(test)
                if bbox[2] <= max_w:
                    current = test
                else:
                    if current:
                        lines.append(current)
                    current = word
            if current:
                lines.append(current)

            # Draw text with outline
            line_h = 80
            total_text_h = len(lines) * line_h
            y_start = (720 - total_text_h) // 2
            x_pos = 50

            for i, line in enumerate(lines[:5]):
                y = y_start + i * line_h
                for ox in range(-3, 4):
                    for oy in range(-3, 4):
                        if ox == 0 and oy == 0:
                            continue
                        draw.text(
                            (x_pos + ox, y + oy), line,
                            font=font, fill=(0, 0, 0),
                        )
                draw.text(
                    (x_pos, y), line,
                    font=font, fill=(255, 255, 255),
                )

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path, "PNG")
            logger.info(f"[thumbnail] AI-generated thumbnail: {output_path}")
            try:
                os.remove(temp_frame)
            except OSError:
                pass
            return True

        logger.warning("[thumbnail] AI image generation returned no data")
        return False

    except Exception as e:
        logger.warning(f"[thumbnail] AI thumbnail failed: {e}")
        return False
