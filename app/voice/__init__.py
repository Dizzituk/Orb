# app/voice/__init__.py
"""
Voice-to-text subsystem for ASTRA.

Provides local speech-to-text transcription via faster-whisper,
wake word detection, and audio pipeline management.

All audio is processed in-memory only — nothing is written to disk.
"""
