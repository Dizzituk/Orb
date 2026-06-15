# FILE: app/memory/enrichment/id3_reader.py
# Purpose: Zero-dependency MP3 tag reader (ID3v2 + ID3v1) for music enrichment.
# Called-by: app.memory.enrichment.media_enricher
# Depends-on: stdlib only
# Last-renovated: 2026-06-12
"""
Zero-dependency MP3 tag reader.

Music enrichment policy (memory architecture directive, 2026-06-12): tags come
from file METADATA only — never from an audio-listening model. "unknown" is an
acceptable value. The venv has no mutagen/tinytag and the corpus is small, so
a minimal hand-rolled reader keeps us at zero new dependencies.

Supports: ID3v2.2/2.3/2.4 text frames (title/artist/album/genre) with ID3v1
fallback. Non-MP3 audio returns all-unknown rather than guessing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Standard ID3v1 genre list (indices 0-79 of the canonical table)
_ID3V1_GENRES = (
    "Blues,Classic Rock,Country,Dance,Disco,Funk,Grunge,Hip-Hop,Jazz,Metal,"
    "New Age,Oldies,Other,Pop,R&B,Rap,Reggae,Rock,Techno,Industrial,"
    "Alternative,Ska,Death Metal,Pranks,Soundtrack,Euro-Techno,Ambient,"
    "Trip-Hop,Vocal,Jazz+Funk,Fusion,Trance,Classical,Instrumental,Acid,"
    "House,Game,Sound Clip,Gospel,Noise,Alt. Rock,Bass,Soul,Punk,Space,"
    "Meditative,Instrumental Pop,Instrumental Rock,Ethnic,Gothic,Darkwave,"
    "Techno-Industrial,Electronic,Pop-Folk,Eurodance,Dream,Southern Rock,"
    "Comedy,Cult,Gangsta Rap,Top 40,Christian Rap,Pop/Funk,Jungle,"
    "Native American,Cabaret,New Wave,Psychedelic,Rave,Showtunes,Trailer,"
    "Lo-Fi,Tribal,Acid Punk,Acid Jazz,Polka,Retro,Musical,Rock & Roll,Hard Rock"
).split(",")

# Frame ids per ID3v2 version: (v2.2 3-char, v2.3/2.4 4-char)
_FRAME_MAP = {
    "title": ("TT2", "TIT2"),
    "artist": ("TP1", "TPE1"),
    "album": ("TAL", "TALB"),
    "genre": ("TCO", "TCON"),
}


def _syncsafe(data: bytes) -> int:
    """Decode a 4-byte syncsafe integer (7 bits per byte)."""
    return (data[0] << 21) | (data[1] << 14) | (data[2] << 7) | data[3]


def _decode_text(payload: bytes) -> Optional[str]:
    """Decode an ID3v2 text frame payload (leading encoding byte)."""
    if not payload:
        return None
    enc, body = payload[0], payload[1:]
    try:
        if enc == 0:
            text = body.decode("latin-1", errors="replace")
        elif enc == 1:
            text = body.decode("utf-16", errors="replace")
        elif enc == 2:
            text = body.decode("utf-16-be", errors="replace")
        else:
            text = body.decode("utf-8", errors="replace")
    except Exception:
        return None
    text = text.strip("\x00").strip()
    return text or None


def _clean_genre(raw: Optional[str]) -> Optional[str]:
    """Resolve '(13)'-style numeric genre refs to names."""
    if not raw:
        return None
    txt = raw.strip()
    if txt.startswith("(") and ")" in txt:
        num, _, rest = txt[1:].partition(")")
        if num.isdigit():
            idx = int(num)
            name = _ID3V1_GENRES[idx] if idx < len(_ID3V1_GENRES) else None
            return (rest.strip() or name) or None
    if txt.isdigit() and int(txt) < len(_ID3V1_GENRES):
        return _ID3V1_GENRES[int(txt)]
    return txt or None


def _read_id3v2(raw: bytes) -> Dict[str, Optional[str]]:
    """Parse the ID3v2 tag at the start of the file, if present."""
    out: Dict[str, Optional[str]] = {}
    if len(raw) < 10 or raw[:3] != b"ID3":
        return out

    major = raw[3]
    tag_size = _syncsafe(raw[6:10])
    end = min(10 + tag_size, len(raw))
    pos = 10

    # Skip extended header if flagged (v2.3+)
    if raw[5] & 0x40 and major >= 3 and pos + 4 <= end:
        ext_size = (
            _syncsafe(raw[pos:pos + 4]) if major == 4
            else int.from_bytes(raw[pos:pos + 4], "big")
        )
        pos += ext_size + (4 if major == 3 else 0)

    want = {
        (ids[0] if major == 2 else ids[1]): field
        for field, ids in _FRAME_MAP.items()
    }
    header_len = 6 if major == 2 else 10

    while pos + header_len <= end and len(out) < len(_FRAME_MAP):
        if major == 2:
            frame_id = raw[pos:pos + 3].decode("latin-1", errors="replace")
            size = int.from_bytes(raw[pos + 3:pos + 6], "big")
            payload_at = pos + 6
        else:
            frame_id = raw[pos:pos + 4].decode("latin-1", errors="replace")
            size_bytes = raw[pos + 4:pos + 8]
            size = (
                _syncsafe(size_bytes) if major == 4
                else int.from_bytes(size_bytes, "big")
            )
            payload_at = pos + 10

        if not frame_id.strip("\x00") or size <= 0 or payload_at + size > end:
            break

        field = want.get(frame_id)
        if field and field not in out:
            value = _decode_text(raw[payload_at:payload_at + size])
            if value:
                out[field] = value

        pos = payload_at + size

    return out


def _read_id3v1(raw: bytes) -> Dict[str, Optional[str]]:
    """Parse the 128-byte ID3v1 tag at the end of the file, if present."""
    out: Dict[str, Optional[str]] = {}
    if len(raw) < 128:
        return out
    tag = raw[-128:]
    if tag[:3] != b"TAG":
        return out

    def _field(data: bytes) -> Optional[str]:
        text = data.decode("latin-1", errors="replace").strip("\x00").strip()
        return text or None

    out["title"] = _field(tag[3:33])
    out["artist"] = _field(tag[33:63])
    out["album"] = _field(tag[63:93])
    genre_idx = tag[127]
    if genre_idx < len(_ID3V1_GENRES):
        out["genre"] = _ID3V1_GENRES[genre_idx]
    return {k: v for k, v in out.items() if v}


def read_audio_tags(path) -> Dict[str, str]:
    """Read title/artist/album/genre from an audio file's metadata.

    Returns all four keys, "unknown" where metadata is absent or the
    format is unsupported. Never raises.
    """
    result = {"title": "unknown", "artist": "unknown",
              "album": "unknown", "genre": "unknown"}
    try:
        p = Path(path)
        if not p.exists() or p.suffix.lower() != ".mp3":
            return result
        raw = p.read_bytes()
        tags = _read_id3v2(raw)
        v1 = _read_id3v1(raw)
        for key in result:
            value = tags.get(key) or v1.get(key)
            if key == "genre":
                value = _clean_genre(value)
            if value:
                result[key] = value
    except Exception as exc:
        logger.debug("[id3] tag read failed for %s: %s", path, exc)
    return result


__all__ = ["read_audio_tags"]
