# FILE: app/drive/thumbnail.py
"""
Thumbnail generation for image files.
Creates small base64-encoded previews for the Drive UI.
"""

import base64
import io
from pathlib import Path
from typing import Optional

# Cache thumbnails in-memory to avoid regenerating
_thumb_cache: dict[str, str] = {}
_THUMB_MAX_SIZE = 200  # px
_CACHE_MAX = 500  # max cached thumbnails


def generate_thumbnail(filepath: Path, size: int = _THUMB_MAX_SIZE) -> Optional[str]:
    """
    Generate a base64-encoded JPEG thumbnail for an image file.
    Returns a data-URI string or None if generation fails.
    """
    cache_key = f"{filepath}:{size}"
    if cache_key in _thumb_cache:
        return _thumb_cache[cache_key]

    try:
        from PIL import Image

        with Image.open(filepath) as img:
            # Convert to RGB for JPEG output (handles PNGs with alpha)
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")

            img.thumbnail((size, size), Image.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            data_uri = f"data:image/jpeg;base64,{b64}"

            # Cache it
            if len(_thumb_cache) < _CACHE_MAX:
                _thumb_cache[cache_key] = data_uri

            return data_uri
    except Exception:
        return None


def clear_thumb_cache():
    """Clear the in-memory thumbnail cache."""
    _thumb_cache.clear()
