"""
faster-whisper implementation of TranscriptionService.

No top-level torch import — GPU detection uses ctranslate2 or lazy torch import.
Reads model config from ModelManager which sources from D:\\LocalAI\\config.ini.
"""
import io
import logging
import time
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

        # Default VAD params
        if vad_parameters is None:
            vad_parameters = {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
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
    def _decode_audio(audio_bytes: bytes) -> np.ndarray:
        """Decode audio bytes to float32 numpy array at 16kHz mono.

        Handles:
        - WAV files (any sample rate / channels — resampled to 16kHz mono)
        - Raw PCM16 bytes
        - Raw float32 bytes
        """
        import io
        import wave

        # Try WAV first
        if audio_bytes[:4] == b'RIFF':
            try:
                with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    framerate = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())

                    if sampwidth == 2:
                        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sampwidth == 4:
                        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                    else:
                        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

                    # Mono mixdown
                    if n_channels > 1:
                        audio = audio.reshape(-1, n_channels).mean(axis=1)

                    # Resample to 16kHz if needed
                    if framerate != 16000:
                        target_len = int(len(audio) * 16000 / framerate)
                        audio = np.interp(
                            np.linspace(0, len(audio) - 1, target_len),
                            np.arange(len(audio)),
                            audio,
                        ).astype(np.float32)

                    return audio
            except Exception as e:
                logger.warning("[faster_whisper] WAV decode failed, trying raw: %s", e)

        # Try raw PCM16
        if len(audio_bytes) % 2 == 0:
            try:
                return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            except ValueError:
                pass

        # Try raw float32
        return np.frombuffer(audio_bytes, dtype=np.float32)

    def get_status(self):
        mm = get_model_manager()
        return mm.get_status()
