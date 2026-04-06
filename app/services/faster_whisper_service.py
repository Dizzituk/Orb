"""
faster-whisper implementation of TranscriptionService.

No top-level torch import — GPU detection uses ctranslate2 or lazy torch import.
Reads model config from ModelManager which sources from D:\LocalAI\config.ini.
"""
import io
import logging
import time
import wave
from typing import Optional, List

import numpy as np

from app.services.transcription_service import (
    TranscriptionSegment,
    TranscriptionResult,
    TranscriptionService,
)
from app.services.model_manager import get_model_manager

logger = logging.getLogger(__name__)


class FasterWhisperService(TranscriptionService):
    """faster-whisper implementation of TranscriptionService."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ):
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type

    def load_model(self) -> None:
        """Load model via ModelManager (lazy, singleton)."""
        mm = get_model_manager()
        mm.load_model(
            model_name=self._model_name,
            device=self._device,
            compute_type=self._compute_type,
        )

    def unload_model(self) -> None:
        get_model_manager().unload_model()

    def is_loaded(self) -> bool:
        return get_model_manager().is_loaded()

    def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        vad_filter: bool = True,
        vad_parameters: Optional[dict] = None,
    ) -> TranscriptionResult:
        """Transcribe audio bytes (WAV file, raw PCM16, or raw float32) to text."""
        mm = get_model_manager()

        # Auto-load if not loaded
        if not mm.is_loaded():
            logger.info("[faster_whisper] Model not loaded, loading now...")
            self.load_model()

        model = mm.get_model()
        if model is None:
            raise RuntimeError("Model failed to load")

        t0 = time.time()

        # Decode audio bytes to float32 numpy array
        audio = self._decode_audio(audio_bytes)

        if len(audio) == 0:
            return TranscriptionResult(text="", language=language or "en")

        # Default VAD params — threshold lowered from 0.5 to 0.3
        # to avoid filtering out speech in noisy environments
        # (e.g. phone mic while driving, wind noise, road noise).
        if vad_parameters is None:
            vad_parameters = {
                "threshold": 0.3,
                "min_speech_duration_ms": 200,
                "min_silence_duration_ms": 500,
            }

        logger.info(
            "[faster_whisper] Transcribing %d samples (%.1fs), lang=%s, vad=%s",
            len(audio), len(audio) / 16000, language, vad_filter,
        )

        segments_iter, info = model.transcribe(
            audio,
            language=language or "en",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters if vad_filter else None,
            condition_on_previous_text=False,
        )

        # Collect segments
        result_segments: List[TranscriptionSegment] = []
        text_parts: List[str] = []

        for seg in segments_iter:
            result_segments.append(TranscriptionSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                confidence=seg.avg_logprob,
            ))
            text_parts.append(seg.text.strip())

        full_text = " ".join(text_parts)
        elapsed = time.time() - t0

        logger.info(
            "[faster_whisper] Done: '%s' (%d segments, %.1fs)",
            full_text[:80], len(result_segments), elapsed,
        )

        return TranscriptionResult(
            text=full_text,
            language=info.language if info else (language or "en"),
            segments=result_segments,
            duration=info.duration if info else None,
        )

    @staticmethod
    def _resample_to_16khz(audio: np.ndarray, source_rate: int) -> np.ndarray:
        if source_rate == 16000 or len(audio) == 0:
            return audio.astype(np.float32, copy=False)

        target_len = max(1, int(round(len(audio) * 16000 / source_rate)))
        return np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    @classmethod
    def _decode_wav(cls, audio_bytes: bytes) -> np.ndarray:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        if sampwidth == 1:
            audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sampwidth == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        return cls._resample_to_16khz(audio, framerate)

    @staticmethod
    def _is_likely_raw_float32(audio_bytes: bytes) -> bool:
        return len(audio_bytes) >= 4 and len(audio_bytes) % 4 == 0

    @staticmethod
    def _is_likely_raw_pcm16(audio_bytes: bytes) -> bool:
        return len(audio_bytes) >= 2 and len(audio_bytes) % 2 == 0

    @classmethod
    def _decode_audio(cls, audio_bytes: bytes) -> np.ndarray:
        """Decode audio bytes to float32 numpy array at 16kHz mono.

        Uses ffmpeg as universal decoder — handles WAV, m4a, AAC, MP3,
        OGG, 3GP, MPEG-4, and any other format ffmpeg supports.
        Falls back to raw PCM if ffmpeg fails.
        """
        # Try ffmpeg first — handles ALL container formats
        try:
            return cls._decode_with_ffmpeg(audio_bytes)
        except Exception as ffmpeg_err:
            logger.warning("[faster_whisper] ffmpeg decode failed: %s — trying fallbacks", ffmpeg_err)

        # Fallback: WAV container
        if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
            return cls._decode_wav(audio_bytes)

        # Fallback: raw float32 PCM
        if len(audio_bytes) % 4 == 0 and len(audio_bytes) > 1000:
            try:
                return np.frombuffer(audio_bytes, dtype=np.float32)
            except ValueError:
                pass

        # Fallback: raw PCM16
        if len(audio_bytes) % 2 == 0 and len(audio_bytes) > 1000:
            try:
                return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            except ValueError:
                pass

        raise ValueError(
            f"Unsupported audio format: {len(audio_bytes)} bytes, "
            f"header={audio_bytes[:12].hex() if len(audio_bytes) >= 12 else 'too short'}"
        )

    @classmethod
    def _decode_with_ffmpeg(cls, audio_bytes: bytes) -> np.ndarray:
        """Use ffmpeg to convert any audio format to mono 16kHz float32 PCM."""
        import subprocess
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", tmp_path,
                    "-f", "f32le",
                    "-ac", "1",
                    "-ar", "16000",
                    "pipe:1",
                ],
                capture_output=True,
                timeout=15,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")[:300]
                raise RuntimeError(f"ffmpeg exit {result.returncode}: {stderr}")

            pcm_bytes = result.stdout
            if len(pcm_bytes) < 100:
                raise RuntimeError(f"ffmpeg produced only {len(pcm_bytes)} bytes")

            audio = np.frombuffer(pcm_bytes, dtype=np.float32)
            logger.info(
                "[faster_whisper] ffmpeg decoded %d bytes → %d samples (%.1fs at 16kHz)",
                len(audio_bytes), len(audio), len(audio) / 16000,
            )
            return audio
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def get_status(self):
        mm = get_model_manager()
        return mm.get_status()