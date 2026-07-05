# FILE: app/content/video_pipeline/caption_align.py
# Purpose: Word-synced burned captions for shorts (wires in the orphan captions.py).
# Called-by: app.content.video_pipeline.shorts_orchestrator, tests.test_caption_align
# Depends-on: app.services.model_manager, app.content.production.captions
# Last-renovated: 2026-07-02
"""
Word-synced captions (jobspec Job 7).

Transcribes the rendered HeyGen mp4's audio with faster-whisper
word-level timestamps, groups the words into 2-4 word chunks, and feeds
the (until now unimported) app/content/production/captions.py to produce
a styled 9:16 ASS track, burn it into a master mp4, and drop a plain SRT
alongside for platforms that accept subtitle uploads.

We call the raw WhisperModel through model_manager rather than the
segment-only FasterWhisperService wrapper, because word_timestamps=True
isn't exposed there and we don't want to perturb the live voice path.
The model is loaded GPU-aware via model_manager and unloaded after.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.content.production.captions import (
    CaptionEntry, burn_captions, save_ass, save_srt,
)

logger = logging.getLogger(__name__)

# 9:16 caption style: big, bold, high-contrast, and lifted well clear of
# the platform UI overlays (progress bar, action rail, caption sheet).
SHORTS_CAPTION_STYLE: Dict[str, Any] = {
    "font": "Arial",
    "font_size": 44,
    "primary_colour": "&H00FFFFFF",   # white (ASS BBGGRR)
    "outline_colour": "&H00000000",   # black outline
    "outline_width": 4,
    "shadow_depth": 1,
    "alignment": 2,                    # bottom-centre
    "margin_v": 320,                   # generous — clears TikTok/IG bottom UI
    "bold": True,
}

_VIDEO_W = 1080
_VIDEO_H = 1920


def _transcribe_words(video_path: str) -> List[Dict[str, float]]:
    """Return [{word, start, end}] via faster-whisper word timestamps."""
    from app.services.model_manager import get_model_manager

    mm = get_model_manager()
    if not mm.is_loaded():
        mm.load_model()
    model = mm.get_model()
    if model is None:
        raise RuntimeError("whisper model unavailable (model_manager returned None)")

    words: List[Dict[str, float]] = []
    segments, _info = model.transcribe(
        video_path,
        word_timestamps=True,
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
    )
    for seg in segments:
        for w in (getattr(seg, "words", None) or []):
            token = (w.word or "").strip()
            if token:
                words.append({"word": token, "start": float(w.start), "end": float(w.end)})
    return words


def _group_words(
    words: List[Dict[str, float]], min_n: int = 2, max_n: int = 4
) -> List[CaptionEntry]:
    """Chunk word list into 2-4 word captions carrying real word timing."""
    captions: List[CaptionEntry] = []
    idx = 1
    i = 0
    n = len(words)
    while i < n:
        chunk = words[i:i + max_n]
        # Avoid a lonely trailing 1-word caption when we can help it.
        if 0 < (n - (i + len(chunk))) < min_n and len(chunk) > min_n:
            chunk = words[i:i + max_n - 1]
        start = chunk[0]["start"]
        end = chunk[-1]["end"]
        if end <= start:
            end = start + 0.4
        text = " ".join(w["word"] for w in chunk)
        captions.append(CaptionEntry(
            index=idx,
            start_seconds=start,
            end_seconds=end,
            text=text.upper() if SHORTS_CAPTION_STYLE["bold"] else text,
        ))
        idx += 1
        i += len(chunk)
    return captions


def _fallback_from_script(script_text: str, duration_s: float) -> List[CaptionEntry]:
    """Even-split captions if transcription yields nothing (last resort)."""
    words = (script_text or "").split()
    if not words or duration_s <= 0:
        return []
    per = 4
    chunks = [words[i:i + per] for i in range(0, len(words), per)]
    slot = duration_s / max(len(chunks), 1)
    out: List[CaptionEntry] = []
    for i, ch in enumerate(chunks):
        text = " ".join(ch)
        out.append(CaptionEntry(
            index=i + 1,
            start_seconds=i * slot,
            end_seconds=(i + 1) * slot,
            text=text.upper() if SHORTS_CAPTION_STYLE["bold"] else text,
        ))
    return out


def align_and_burn(
    video_path: str,
    out_dir: str,
    *,
    slug: str = "short",
    script_text: str = "",
    duration_s: float = 0.0,
    unload_after: bool = True,
) -> Dict[str, Any]:
    """Transcribe -> group -> styled ASS -> burn + SRT. Returns paths.

    Returns {ok, burned_path, srt_path, ass_path, caption_count, error?}.
    If burning fails, burned_path falls back to the input (captions still
    saved as SRT so nothing is lost).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        words = _transcribe_words(video_path)
    except Exception as e:
        logger.warning("[caption_align] transcription failed: %s", e)
        words = []
    finally:
        if unload_after:
            try:
                from app.services.model_manager import get_model_manager
                get_model_manager().unload_model()
            except Exception:
                pass

    captions = _group_words(words) if words else _fallback_from_script(script_text, duration_s)
    if not captions:
        return {"ok": False, "error": "no captions produced (no words, no script)",
                "burned_path": video_path, "srt_path": "", "ass_path": "", "caption_count": 0}

    ass_path = str(out / f"{slug}.ass")
    srt_path = str(out / f"{slug}.srt")
    burned_path = str(out / f"{slug}_captioned.mp4")

    save_ass(captions, ass_path, style=SHORTS_CAPTION_STYLE,
             video_width=_VIDEO_W, video_height=_VIDEO_H)
    save_srt(captions, srt_path)

    burned_ok = burn_captions(video_path, ass_path, burned_path)
    if not burned_ok:
        logger.warning("[caption_align] burn failed; delivering uncaptioned master + SRT")
        burned_path = video_path

    return {
        "ok": bool(burned_ok),
        "burned_path": burned_path,
        "srt_path": srt_path,
        "ass_path": ass_path,
        "caption_count": len(captions),
    }
