# FILE: app/bridge/tts_cache.py
# Purpose: Persistent per-message TTS audio cache (files + manifest + LRU eviction + idempotency map).
# Called-by: app.bridge.router (chat-and-speak tee), app.bridge.tts_audio (serve/ensure), app.bridge.missed_replies
# Depends-on: stdlib only (json, threading, pathlib)
# Last-renovated: 2026-06-11
"""
Per-message TTS audio cache on the PC.

Every synthesised assistant reply is assembled into data/tts_cache/{message_id}.mp3
while it streams to the phone, so audio is NEVER re-synthesised for a message
that already has it. A small manifest.json tracks size + last access for LRU
eviction (size cap via ORB_TTS_CACHE_MAX_MB, default 3072 MB). idempotency.json
maps client X-Idempotency-Key -> assistant message id so a retried
chat-and-speak request re-serves the original reply instead of re-running
LLM + TTS (the replay-ghost / double-reply fix).

Files in flight are written as {message_id}.part and renamed on finalize;
a .part present without a final file means synthesis is still running.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ORB_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(os.getenv("ORB_TTS_CACHE_DIR", str(_ORB_ROOT / "data" / "tts_cache")))
MAX_BYTES = int(os.getenv("ORB_TTS_CACHE_MAX_MB", "3072")) * 1024 * 1024
AUDIO_EXT = "mp3"
_IDEM_CAP = 1000          # max idempotency entries kept
_PART_GRACE_SECS = 3600   # never evict a .part younger than this

_MANIFEST = CACHE_DIR / "manifest.json"
_IDEMPOTENCY = CACHE_DIR / "idempotency.json"
_lock = threading.Lock()  # guards manifest/idempotency read-modify-write


def _ensure_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("[tts_cache] could not read %s: %s", path.name, e)
    return {}


def _save_json(path: Path, data: dict) -> None:
    _ensure_dir()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=0), encoding="utf-8")
    os.replace(tmp, path)


# ── Paths and lookups ───────────────────────────────────────────────────

def final_path(message_id: int) -> Path:
    return CACHE_DIR / f"{int(message_id)}.{AUDIO_EXT}"


def part_path(message_id: int) -> Path:
    return CACHE_DIR / f"{int(message_id)}.part"


def get_cached_path(message_id: int) -> Optional[Path]:
    """Path to the complete cached audio for a message, or None."""
    p = final_path(message_id)
    return p if p.exists() and p.stat().st_size > 0 else None


def is_pending(message_id: int) -> bool:
    """True when synthesis for this message is still writing its .part file."""
    return part_path(message_id).exists() and not final_path(message_id).exists()


def has_audio(message_id: int) -> bool:
    return get_cached_path(message_id) is not None


def touch(message_id: int) -> None:
    """Record an access for LRU purposes."""
    with _lock:
        manifest = _load_json(_MANIFEST)
        entry = manifest.get(str(int(message_id)))
        if entry is not None:
            entry["last_access"] = time.time()
            _save_json(_MANIFEST, manifest)


# ── VBR (Xing) duration header ──────────────────────────────────────────
# Chatterbox sentence-MP3s are MPEG2 Layer3, 24 kHz mono, VBR — and the
# encoder's priming frame is 8 kbps. The concatenated per-message file has
# no Xing header, so any player that falls back to constant-bitrate
# duration estimation from the FIRST frame (ExoPlayer's Mp3Extractor does)
# reports size/8kbps ≈ 7x the real length. That one defect caused the
# 2026-06-12 road bugs: ~15-min duration marker on a ~2-min reply with a
# phantom dead-air tail, AND playback jumping back near the start when the
# phone swaps from the growing stream to the final file (seekTo() maps the
# position through the same wrong seek map).
#
# Fix (2026-06-13): every NEW stream/cache file begins with a fixed-size
# placeholder Xing frame — a valid, silent MPEG2-L3 mono frame — written
# IDENTICALLY to the live HTTP stream and the .part file so byte offsets
# stay aligned for HTTP Range resume. finalize() patches the real frame
# count / byte total / seek TOC into the on-disk copy IN PLACE (offsets
# never shift). The phone re-fetches the first 192 bytes after completion
# to adopt the patched header before its seekable swap.

_XING_FRAME_LEN = 192                    # MPEG2 L3 @ 64 kbps, 24 kHz: 72000*64/24000
_XING_HEADER4 = b"\xff\xf3\x84\xc4"      # MPEG2, Layer3, 64 kbps, 24 kHz, mono
_XING_TAG_OFFSET = 13                    # 4 header + 9 side-info bytes (MPEG2 mono)
_BITRATES_V2L3 = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0)
_SAMPLERATES_V2 = (22050, 24000, 16000)


def xing_placeholder() -> bytes:
    """A silent MPEG2-L3 frame carrying an all-zero-field Xing tag.

    Parsers that understand Xing skip it (zero flags = no usable fields →
    they fall back to today's behaviour); ones that don't decode ~24 ms of
    silence. Either way it is inert until finalize() patches real values in.
    """
    frame = bytearray(_XING_FRAME_LEN)
    frame[0:4] = _XING_HEADER4
    frame[_XING_TAG_OFFSET:_XING_TAG_OFFSET + 4] = b"Xing"
    return bytes(frame)


def _walk_frames(data: bytes, start: int):
    """Yield (offset, frame_len, duration_s) for MPEG2-L3 frames in data."""
    i, n = start, len(data)
    while i + 4 <= n:
        if data[i] == 0xFF and data[i + 1] in (0xF2, 0xF3):  # sync + MPEG2 + L3
            b2 = data[i + 2]
            br = _BITRATES_V2L3[(b2 >> 4) & 0xF]
            sr_idx = (b2 >> 2) & 3
            if br and sr_idx != 3:
                sr = _SAMPLERATES_V2[sr_idx]
                flen = (72000 * br) // sr + ((b2 >> 1) & 1)
                if flen > 4 and i + flen <= n:
                    yield i, flen, 576.0 / sr
                    i += flen
                    continue
        i += 1


def patch_xing_header(path: Path) -> bool:
    """Write real totals + seek TOC into the placeholder frame, in place.

    No-op (False) for files that don't begin with our placeholder — old
    cache entries keep their old behaviour. Never moves bytes: HTTP Range
    offsets handed out during streaming stay valid.
    """
    import struct
    try:
        data = path.read_bytes()
    except OSError:
        return False
    if len(data) <= _XING_FRAME_LEN or not data.startswith(_XING_HEADER4):
        return False
    if data[_XING_TAG_OFFSET:_XING_TAG_OFFSET + 4] != b"Xing":
        return False
    frames = list(_walk_frames(data, _XING_FRAME_LEN))
    if not frames:
        return False
    total_bytes = len(data)
    cumulative = []                      # (start_time_s, byte_offset) per frame
    elapsed = 0.0
    for off, _flen, dur in frames:
        cumulative.append((elapsed, off))
        elapsed += dur
    toc = bytearray(100)
    fi = 0
    for pct in range(100):               # single forward pass, both sorted
        target = elapsed * (pct / 100.0)
        while fi < len(cumulative) - 1 and cumulative[fi + 1][0] <= target:
            fi += 1
        toc[pct] = min(255, (cumulative[fi][1] * 256) // total_bytes)
    payload = (
        b"Xing"
        + struct.pack(">I", 0x0007)      # FRAMES | BYTES | TOC
        + struct.pack(">I", len(frames))
        + struct.pack(">I", total_bytes)
        + bytes(toc)
    )
    with open(path, "r+b") as fh:
        fh.seek(_XING_TAG_OFFSET)
        fh.write(payload)
    return True


# ── Writer (tee target for the streaming synthesis loop) ───────────────

class TtsCacheWriter:
    """Appends MP3 chunks to {id}.part; finalize() promotes it to {id}.mp3.

    abort() keeps the .part on disk (synthesis may be continued later by a
    background task); discard() removes it outright.
    """

    def __init__(self, message_id: int):
        self.message_id = int(message_id)
        self._path = part_path(self.message_id)
        self._bytes = 0
        self._closed = False
        _ensure_dir()
        # "wb" not "ab" (2026-06-13): a fresh writer must own the file from
        # byte 0 — appending onto an orphaned .part from a crashed attempt
        # produced a file that no longer matched the bytes streamed to the
        # phone, corrupting Range resume. Disconnect continuation reuses
        # THIS instance's handle, so nothing legitimate appended cross-instance.
        self._fh = open(self._path, "wb")

    def start_header(self) -> bytes:
        """Write the placeholder Xing frame at byte 0 of a fresh file.

        Returns the header bytes so streaming callers can yield the SAME
        bytes into the live HTTP response (file and stream must stay
        byte-identical for Range resume); b"" if anything is already written.
        """
        if self._closed or self._bytes:
            return b""
        hdr = xing_placeholder()
        self._fh.write(hdr)
        self._bytes += len(hdr)
        return hdr

    def add_chunk(self, chunk: bytes) -> None:
        if self._closed or not chunk:
            return
        self._fh.write(chunk)
        self._bytes += len(chunk)

    def finalize(self) -> Optional[Path]:
        if self._closed:
            return get_cached_path(self.message_id)
        self._close_fh()
        # Patch the real duration/seek data into the placeholder header
        # (2026-06-13) BEFORE promotion, so every reader of the final file
        # sees a truthful Xing frame. See module comment above.
        try:
            patch_xing_header(self._path)
        except Exception as e:
            logger.warning("[tts_cache] xing patch failed for %s: %s", self.message_id, e)
        final = final_path(self.message_id)
        try:
            os.replace(self._path, final)
        except OSError as e:
            logger.error("[tts_cache] finalize failed for %s: %s", self.message_id, e)
            return None
        size = final.stat().st_size
        with _lock:
            manifest = _load_json(_MANIFEST)
            manifest[str(self.message_id)] = {
                "ext": AUDIO_EXT,
                "bytes": size,
                "created": time.time(),
                "last_access": time.time(),
            }
            _save_json(_MANIFEST, manifest)
        evict_to_cap()
        logger.info("[tts_cache] cached message %s (%d KB)", self.message_id, size // 1024)
        return final

    def abort(self) -> None:
        """Stop writing but KEEP the .part for a later continuation."""
        self._close_fh()
        logger.info("[tts_cache] writer aborted for %s (%d bytes kept as .part)",
                    self.message_id, self._bytes)

    def discard(self) -> None:
        self._close_fh()
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            pass

    def _close_fh(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._fh.close()
            except Exception:
                pass


def open_writer(message_id: int) -> TtsCacheWriter:
    return TtsCacheWriter(message_id)


# ── LRU eviction ────────────────────────────────────────────────────────

def evict_to_cap(max_bytes: int = MAX_BYTES) -> int:
    """Delete least-recently-accessed audio until the cache fits the cap.

    Returns number of files evicted. .part files younger than the grace
    window are exempt (synthesis may still be running).
    """
    _ensure_dir()
    evicted = 0
    with _lock:
        manifest = _load_json(_MANIFEST)
        files = []
        total = 0
        now = time.time()
        for f in CACHE_DIR.iterdir():
            if f.suffix == ".json" or f.suffix == ".tmp" or not f.is_file():
                continue
            if f.suffix == ".part" and (now - f.stat().st_mtime) < _PART_GRACE_SECS:
                total += f.stat().st_size
                continue
            meta = manifest.get(f.stem, {})
            last = meta.get("last_access", f.stat().st_mtime)
            size = f.stat().st_size
            total += size
            files.append((last, size, f))
        if total <= max_bytes:
            return 0
        files.sort(key=lambda t: t[0])  # oldest access first
        for last, size, f in files:
            if total <= max_bytes:
                break
            try:
                f.unlink()
                manifest.pop(f.stem, None)
                total -= size
                evicted += 1
            except OSError as e:
                logger.warning("[tts_cache] evict failed for %s: %s", f.name, e)
        if evicted:
            _save_json(_MANIFEST, manifest)
            logger.info("[tts_cache] evicted %d file(s), %d MB now used",
                        evicted, total // (1024 * 1024))
    return evicted


# ── Idempotency map (X-Idempotency-Key -> assistant message id) ────────

def idem_get(key: str) -> Optional[int]:
    if not key:
        return None
    with _lock:
        entry = _load_json(_IDEMPOTENCY).get(key)
    return int(entry["message_id"]) if entry else None


def idem_put(key: str, message_id: int) -> None:
    if not key:
        return
    with _lock:
        data = _load_json(_IDEMPOTENCY)
        data[key] = {"message_id": int(message_id), "ts": time.time()}
        if len(data) > _IDEM_CAP:
            for stale in sorted(data, key=lambda k: data[k].get("ts", 0))[: len(data) - _IDEM_CAP]:
                data.pop(stale, None)
        _save_json(_IDEMPOTENCY, data)
