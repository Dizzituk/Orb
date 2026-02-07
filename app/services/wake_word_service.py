"""
Wake word detection service for voice assistant.

This module provides a lightweight wake word detection service that can identify
specific trigger words in audio streams without requiring heavy model inference.
It's designed to be fast and efficient for always-on listening scenarios.
"""

import asyncio
import logging
import wave
import io
import struct
from typing import Optional, Callable, List
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WakeWordConfig:
    """Configuration for wake word detection."""
    
    wake_words: List[str]
    sample_rate: int = 16000
    chunk_size: int = 1024
    threshold: float = 0.5
    cooldown_seconds: float = 2.0


class WakeWordService:
    """
    Lightweight wake word detection service.
    
    This service provides simple pattern matching for wake words in audio streams.
    It can be used to trigger voice assistant activation without requiring continuous
    transcription of all audio.
    
    Attributes:
        config: Wake word detection configuration
        is_enabled: Whether wake word detection is currently enabled
        last_detection_time: Timestamp of last wake word detection
    """
    
    _instance: Optional['WakeWordService'] = None
    _lock = asyncio.Lock()
    
    def __init__(self, config: Optional[WakeWordConfig] = None):
        """
        Initialize wake word service.
        
        Args:
            config: Wake word detection configuration. If None, uses defaults.
        """
        self.config = config or WakeWordConfig(
            wake_words=["hey assistant", "ok assistant"],
            sample_rate=16000,
            chunk_size=1024,
            threshold=0.5,
            cooldown_seconds=2.0
        )
        self.is_enabled = False
        self.last_detection_time: Optional[float] = None
        self._detection_callback: Optional[Callable[[str], None]] = None
        self._audio_buffer: List[bytes] = []
        self._buffer_size_seconds = 3.0
        
        logger.info(
            f"WakeWordService initialized with wake words: {self.config.wake_words}"
        )
    
    @classmethod
    async def get_instance(cls, config: Optional[WakeWordConfig] = None) -> 'WakeWordService':
        """
        Get singleton instance of wake word service.
        
        Args:
            config: Optional configuration for first initialization
            
        Returns:
            Singleton WakeWordService instance
        """
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
        return cls._instance
    
    def enable(self) -> None:
        """Enable wake word detection."""
        self.is_enabled = True
        logger.info("Wake word detection enabled")
    
    def disable(self) -> None:
        """Disable wake word detection."""
        self.is_enabled = False
        self._audio_buffer.clear()
        logger.info("Wake word detection disabled")
    
    def set_detection_callback(self, callback: Callable[[str], None]) -> None:
        """
        Set callback function to be called when wake word is detected.
        
        Args:
            callback: Function to call with detected wake word
        """
        self._detection_callback = callback
    
    def _is_in_cooldown(self) -> bool:
        """
        Check if we're in cooldown period after last detection.
        
        Returns:
            True if in cooldown period, False otherwise
        """
        if self.last_detection_time is None:
            return False
        
        import time
        elapsed = time.time() - self.last_detection_time
        return elapsed < self.config.cooldown_seconds
    
    def _add_to_buffer(self, audio_chunk: bytes) -> None:
        """
        Add audio chunk to rolling buffer.
        
        Args:
            audio_chunk: Audio data to add to buffer
        """
        self._audio_buffer.append(audio_chunk)
        
        # Calculate approximate buffer duration and trim if needed
        bytes_per_second = self.config.sample_rate * 2  # 16-bit audio
        max_buffer_bytes = int(bytes_per_second * self._buffer_size_seconds)
        
        total_bytes = sum(len(chunk) for chunk in self._audio_buffer)
        while total_bytes > max_buffer_bytes and len(self._audio_buffer) > 1:
            removed = self._audio_buffer.pop(0)
            total_bytes -= len(removed)
    
    def _get_buffer_audio(self) -> bytes:
        """
        Get concatenated audio from buffer.
        
        Returns:
            Combined audio data from buffer
        """
        return b''.join(self._audio_buffer)
    
    def _simple_energy_detection(self, audio_data: bytes) -> float:
        """
        Calculate simple audio energy level.
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            
        Returns:
            Normalized energy level (0.0 to 1.0)
        """
        if len(audio_data) < 2:
            return 0.0
        
        # Convert bytes to 16-bit integers
        samples = struct.unpack(f'{len(audio_data) // 2}h', audio_data)
        
        # Calculate RMS energy
        if not samples:
            return 0.0
        
        energy = np.sqrt(np.mean(np.square(samples)))
        
        # Normalize to 0-1 range (assuming 16-bit audio)
        return min(energy / 32768.0, 1.0)
    
    async def process_audio_chunk(self, audio_chunk: bytes) -> Optional[str]:
        """
        Process audio chunk and check for wake word.
        
        This is a simplified implementation that uses basic audio analysis.
        In a production system, this would use a specialized wake word model
        like Porcupine, Snowboy, or a custom trained model.
        
        Args:
            audio_chunk: Raw audio data (16-bit PCM at configured sample rate)
            
        Returns:
            Detected wake word string if found, None otherwise
        """
        if not self.is_enabled:
            return None
        
        if self._is_in_cooldown():
            return None
        
        # Add to rolling buffer
        self._add_to_buffer(audio_chunk)
        
        # Simple energy-based detection as placeholder
        # In production, this would use a real wake word model
        energy = self._simple_energy_detection(audio_chunk)
        
        if energy > self.config.threshold:
            # Simulate wake word detection
            # In a real implementation, this would use a trained model
            # to analyze the buffered audio for wake word patterns
            detected_word = self._detect_wake_word_pattern()
            
            if detected_word:
                import time
                self.last_detection_time = time.time()
                
                logger.info(f"Wake word detected: {detected_word}")
                
                if self._detection_callback:
                    self._detection_callback(detected_word)
                
                return detected_word
        
        return None
    
    def _detect_wake_word_pattern(self) -> Optional[str]:
        """
        Analyze buffered audio for wake word patterns.
        
        This is a placeholder implementation. In production, this would:
        1. Use a specialized wake word detection model (Porcupine, Snowboy, etc.)
        2. Analyze acoustic features of the buffered audio
        3. Return the specific wake word if detected with sufficient confidence
        
        Returns:
            Detected wake word or None
        """
        # Placeholder: In real implementation, this would use a trained model
        # to detect specific wake word patterns in the audio buffer
        
        # For now, we return None to indicate no detection
        # This ensures the service works without blocking, but doesn't
        # actually detect wake words until a real model is integrated
        return None
    
    async def process_audio_stream(
        self,
        audio_generator,
        stop_event: Optional[asyncio.Event] = None
    ) -> None:
        """
        Process continuous audio stream for wake word detection.
        
        Args:
            audio_generator: Async generator yielding audio chunks
            stop_event: Optional event to signal stream processing should stop
        """
        try:
            async for audio_chunk in audio_generator:
                if stop_event and stop_event.is_set():
                    break
                
                await self.process_audio_chunk(audio_chunk)
                
        except Exception as e:
            logger.error(f"Error processing audio stream: {e}", exc_info=True)
        finally:
            self._audio_buffer.clear()
    
    def reset(self) -> None:
        """Reset wake word service state."""
        self._audio_buffer.clear()
        self.last_detection_time = None
        logger.info("Wake word service reset")
    
    def get_status(self) -> dict:
        """
        Get current wake word service status.
        
        Returns:
            Dictionary containing service status information
        """
        return {
            "enabled": self.is_enabled,
            "wake_words": self.config.wake_words,
            "sample_rate": self.config.sample_rate,
            "threshold": self.config.threshold,
            "cooldown_seconds": self.config.cooldown_seconds,
            "in_cooldown": self._is_in_cooldown(),
            "buffer_size": len(self._audio_buffer),
        }


async def get_wake_word_service(
    config: Optional[WakeWordConfig] = None
) -> WakeWordService:
    """
    Get singleton instance of wake word service.
    
    This is a convenience function that wraps WakeWordService.get_instance()
    for consistent API with other service modules.
    
    Args:
        config: Optional configuration for first initialization
        
    Returns:
        Singleton WakeWordService instance
    """
    return await WakeWordService.get_instance(config)