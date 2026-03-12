# FILE: app/voice/tts_server.py
"""
Google Cloud TTS microservice — FastAPI on port 8001.

Uses the Google Cloud TTS REST API with API key auth (no service account needed).
Serves audio to the Electron frontend via ttsApi.ts.

Endpoints:
  GET  /ping              — Health check
  POST /tts/speak         — Synthesise text → MP3 audio
  POST /tts/preview       — Preview a voice with sample text
  GET  /tts/voices        — List available voices + current selection
  POST /tts/voices/select — Set the active voice

v1.0 (2026-02-24): Initial implementation.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
except ImportError:
    pass  # dotenv not required if env vars set externally

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Ensure TTS logs are visible in the console (standalone service)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)

TTS_BUILD_ID = "2026-02-28-v1.1-error-logging"

# ── Config ──────────────────────────────────────────────────────────────

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TTS_API_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
VOICES_API_URL = "https://texttospeech.googleapis.com/v1/voices"

# Default voice — British English, male, WaveNet quality
DEFAULT_VOICE = "en-GB-WaveNet-B"
DEFAULT_LANGUAGE = "en-GB"
DEFAULT_SPEED = 1.0

# Supported language prefixes (en-GB first, then en-US)
SUPPORTED_LANGUAGES = ["en-GB", "en-US"]

# Tier display order (best first)
TIER_ORDER = ["Chirp3-HD", "Chirp-HD", "Studio", "Wavenet", "Neural2", "News", "Standard"]

# Cached voice list from Google API
_voice_cache: list = []
_voice_cache_time: float = 0
VOICE_CACHE_TTL = 3600  # Refresh every hour

# ── State (file-persisted) ──────────────────────────────────────────────

_VOICE_PREFS_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'tts_prefs.json')


def _load_voice_pref() -> str:
    """Load persisted voice selection from disk."""
    try:
        if os.path.exists(_VOICE_PREFS_FILE):
            with open(_VOICE_PREFS_FILE, 'r') as f:
                data = json.load(f)
            voice = data.get("selected_voice", "")
            if voice:
                logger.info("[tts] Loaded persisted voice: %s", voice)
                return voice
    except Exception as e:
        logger.warning("[tts] Could not load voice pref: %s", e)
    return DEFAULT_VOICE


def _save_voice_pref(voice: str) -> None:
    """Persist voice selection to disk."""
    try:
        os.makedirs(os.path.dirname(_VOICE_PREFS_FILE), exist_ok=True)
        with open(_VOICE_PREFS_FILE, 'w') as f:
            json.dump({"selected_voice": voice}, f)
        logger.info("[tts] Saved voice pref: %s", voice)
    except Exception as e:
        logger.warning("[tts] Could not save voice pref: %s", e)


_selected_voice = _load_voice_pref()

# ── Bootstrap: sync API keys from encrypted DB ─────────────────────────
# The TTS server runs as a separate process from the main backend.
# API keys live in the encrypted DB (master key via Credential Manager),
# not in .env. We must sync them to os.environ before serving.

def _read_master_key_from_credential_manager() -> Optional[str]:
    """Read ORB_MASTER_KEY from Windows Credential Manager (same store as Electron/keytar)."""
    import ctypes
    import ctypes.wintypes as wt

    class CREDENTIAL(ctypes.Structure):
        _fields_ = [
            ('Flags', wt.DWORD), ('Type', wt.DWORD),
            ('TargetName', wt.LPWSTR), ('Comment', wt.LPWSTR),
            ('LastWritten', wt.FILETIME), ('CredentialBlobSize', wt.DWORD),
            ('CredentialBlob', ctypes.POINTER(ctypes.c_byte)),
            ('Persist', wt.DWORD), ('AttributeCount', wt.DWORD),
            ('Attributes', ctypes.c_void_p), ('TargetAlias', wt.LPWSTR),
            ('UserName', wt.LPWSTR),
        ]

    PCREDENTIAL = ctypes.POINTER(CREDENTIAL)
    cred = PCREDENTIAL()
    # keytar stores credentials as 'ServiceName/AccountName'
    ok = ctypes.windll.advapi32.CredReadW(
        'OrbMasterKey/default', 1, 0, ctypes.byref(cred),
    )
    if not ok:
        return None
    blob = ctypes.string_at(
        cred.contents.CredentialBlob, cred.contents.CredentialBlobSize,
    )
    ctypes.windll.advapi32.CredFree(cred)
    return blob.decode('utf-8')


def _sync_api_keys():
    """Load API keys from encrypted DB into os.environ.

    The TTS server runs as a standalone process (not spawned by Electron),
    so ORB_MASTER_KEY isn't in the environment. We read it directly from
    Windows Credential Manager, then unlock the DB and sync keys.
    """
    try:
        # If master key isn't already in env, read from Credential Manager
        if not os.getenv('ORB_MASTER_KEY'):
            key = _read_master_key_from_credential_manager()
            if key:
                os.environ['ORB_MASTER_KEY'] = key
                logger.info("[tts] Loaded master key from Credential Manager")
            else:
                logger.warning("[tts] No master key in env or Credential Manager")
                return

        from app.crypto import require_master_key_or_exit
        require_master_key_or_exit()

        from app.db import init_db, SessionLocal
        init_db()

        from app.settings.service import sync_all_to_env
        db = SessionLocal()
        count = sync_all_to_env(db)
        db.close()
        logger.info("[tts] Synced %d API keys from encrypted DB", count)
    except Exception as e:
        logger.warning("[tts] Could not sync API keys from DB: %s — falling back to .env", e)

_sync_api_keys()

# ── FastAPI app ─────────────────────────────────────────────────────────

app = FastAPI(title="ASTRA TTS Service", version=TTS_BUILD_ID)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ──────────────────────────────────────────────────────

class SpeakRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = None


class PreviewRequest(BaseModel):
    voice: str


class SelectVoiceRequest(BaseModel):
    voice: str


# ── Voice catalogue (live from Google API) ──────────────────────────────

def _extract_tier(voice_name: str) -> str:
    """Extract tier from voice ID, e.g. 'en-GB-Chirp3-HD-Achird' → 'Chirp3-HD'."""
    # Remove language prefix (e.g. 'en-GB-')
    parts = voice_name.split("-", 2)
    if len(parts) < 3:
        return "Other"
    remainder = parts[2]  # e.g. 'Chirp3-HD-Achird' or 'Wavenet-B'
    for tier in TIER_ORDER:
        if remainder.startswith(tier):
            return tier
    return "Other"


def _make_display_name(voice_name: str, gender: str) -> str:
    """Create a friendly display name from the voice ID."""
    tier = _extract_tier(voice_name)
    # Extract the letter/name suffix
    parts = voice_name.split("-")
    suffix = parts[-1] if parts else ""
    lang = "-".join(parts[:2]) if len(parts) >= 2 else ""
    region = "British" if lang == "en-GB" else "US" if lang == "en-US" else lang
    gender_label = "Male" if gender == "MALE" else "Female"
    return f"{region} {gender_label} — {tier} {suffix}"


async def _fetch_voices_from_google() -> list:
    """Fetch all supported English voices from Google TTS API."""
    global _voice_cache, _voice_cache_time
    import time

    now = time.time()
    if _voice_cache and (now - _voice_cache_time) < VOICE_CACHE_TTL:
        return _voice_cache

    api_key = os.getenv("GOOGLE_API_KEY", "") or GOOGLE_API_KEY
    if not api_key:
        return []

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{VOICES_API_URL}?key={api_key}"
            )
        if resp.status_code != 200:
            logger.error("[tts] Failed to fetch voices: %d", resp.status_code)
            return _voice_cache  # Return stale cache on error

        all_voices = resp.json().get("voices", [])

        catalogue = []
        for v in all_voices:
            langs = v.get("languageCodes", [])
            # Only include supported languages
            matching_lang = None
            for lang in SUPPORTED_LANGUAGES:
                if lang in langs:
                    matching_lang = lang
                    break
            if not matching_lang:
                continue

            voice_id = v["name"]
            gender = v.get("ssmlGender", "NEUTRAL")
            tier = _extract_tier(voice_id)
            display = _make_display_name(voice_id, gender)

            catalogue.append({
                "name": display,
                "voice_id": voice_id,
                "gender": gender,
                "tier": tier,
                "language": matching_lang,
            })

        # Sort: by language (en-GB first), then tier order, then gender, then name
        def sort_key(v):
            lang_idx = SUPPORTED_LANGUAGES.index(v["language"]) if v["language"] in SUPPORTED_LANGUAGES else 99
            tier_idx = TIER_ORDER.index(v["tier"]) if v["tier"] in TIER_ORDER else 99
            return (lang_idx, tier_idx, v["gender"], v["voice_id"])

        catalogue.sort(key=sort_key)
        _voice_cache = catalogue
        _voice_cache_time = now
        logger.info("[tts] Fetched %d voices from Google API", len(catalogue))
        return catalogue

    except Exception as e:
        logger.error("[tts] Voice fetch error: %s", e)
        return _voice_cache


# ── Google TTS call ─────────────────────────────────────────────────────

async def _synthesize(text: str, voice_name: str, speed: float = 1.0) -> bytes:
    """Call Google Cloud TTS REST API and return MP3 bytes."""
    # Re-read API key every call — the key may have been synced from DB
    # after this module was imported (settings sync runs during startup).
    api_key = os.getenv("GOOGLE_API_KEY", "") or GOOGLE_API_KEY
    if not api_key:
        raise HTTPException(500, "GOOGLE_API_KEY not set in environment")

    # Determine language code from voice name (e.g. "en-GB-WaveNet-B" → "en-GB")
    parts = voice_name.split("-")
    lang_code = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else DEFAULT_LANGUAGE

    payload = {
        "input": {"text": text},
        "voice": {
            "languageCode": lang_code,
            "name": voice_name,
        },
        "audioConfig": {
            "audioEncoding": "MP3",
            "speakingRate": max(0.25, min(4.0, speed)),
        },
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{TTS_API_URL}?key={api_key}",
            json=payload,
        )

    if resp.status_code != 200:
        detail = resp.text[:300]
        logger.error("[tts] Google API error %d: %s", resp.status_code, detail)
        raise HTTPException(resp.status_code, f"Google TTS error: {detail}")

    audio_b64 = resp.json().get("audioContent", "")
    if not audio_b64:
        raise HTTPException(500, "Google TTS returned empty audio")

    return base64.b64decode(audio_b64)


# ── Endpoints ───────────────────────────────────────────────────────────

@app.get("/ping")
async def ping():
    return {"status": "ok", "service": "tts", "build": TTS_BUILD_ID}


@app.post("/tts/speak")
async def speak(req: SpeakRequest):
    """Synthesise text to speech. Returns MP3 audio."""
    try:
        if not req.text.strip():
            raise HTTPException(400, "Empty text")

        voice = req.voice or _selected_voice
        speed = req.speed if req.speed is not None else DEFAULT_SPEED

        logger.info("[tts] Speak: %d chars, voice=%s, speed=%.1f", len(req.text), voice, speed)
        audio = await _synthesize(req.text, voice, speed)

        return Response(content=audio, media_type="audio/mpeg")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[tts] Speak FAILED: %s: %s", type(e).__name__, e, exc_info=True)
        raise HTTPException(500, f"TTS speak error: {type(e).__name__}: {e}")


@app.post("/tts/preview")
async def preview(req: PreviewRequest):
    """Preview a voice with sample text."""
    try:
        sample = "Hello, I'm ASTRA. This is how I sound with this voice."
        audio = await _synthesize(sample, req.voice, DEFAULT_SPEED)
        return Response(content=audio, media_type="audio/mpeg")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[tts] Preview FAILED: %s: %s", type(e).__name__, e, exc_info=True)
        raise HTTPException(500, f"TTS preview error: {type(e).__name__}: {e}")


@app.get("/tts/voices")
async def list_voices():
    """List available voices and current selection."""
    catalogue = await _fetch_voices_from_google()
    voices = []
    for v in catalogue:
        voices.append({
            "name": v["name"],
            "voice_id": v["voice_id"],
            "gender": v["gender"],
            "tier": v["tier"],
            "selected": v["voice_id"] == _selected_voice,
        })
    return {"voices": voices, "selected": _selected_voice}


@app.post("/tts/voices/select")
async def select_voice(req: SelectVoiceRequest):
    """Set the active voice."""
    global _selected_voice

    # Accept any voice ID — Google will validate it on speak
    _selected_voice = req.voice
    _save_voice_pref(req.voice)
    logger.info("[tts] Voice selected: %s", _selected_voice)
    return {"selected": _selected_voice}


# ── Standalone entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
