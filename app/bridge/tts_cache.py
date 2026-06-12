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
        self._fh = open(self._path, "ab")  # append: continue an earlier partial

    def add_chunk(self, chunk: bytes) -> None:
        if self._closed or not chunk:
            return
        self._fh.write(chunk)
        self._bytes += len(chunk)

    def finalize(self) -> Optional[Path]:
        if self._closed:
            return get_cached_path(self.message_id)
        self._close_fh()
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
