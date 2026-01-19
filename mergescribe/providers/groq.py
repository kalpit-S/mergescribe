"""
Groq Whisper API provider for cloud transcription.
"""

import io
import time
from typing import Optional

import numpy as np
import soundfile as sf

from . import Provider
from ..types import TranscriptionResult


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convert numpy audio array to WAV bytes."""
    audio_int16 = (audio * 32767).astype(np.int16)
    buffer = io.BytesIO()
    sf.write(buffer, audio_int16, sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.getvalue()


class GroqProvider(Provider):
    """
    Cloud transcription using Groq's Whisper API.

    Fast cloud-based transcription with low latency.
    """

    name = "groq"

    def __init__(self, api_key: str, model: str = "whisper-large-v3"):
        self.api_key = api_key
        self.model = model
        self.client = None

    def initialize(self) -> None:
        """Create Groq client."""
        try:
            from groq import Groq

            self.client = Groq(api_key=self.api_key)
            print(f"[{self.name}] Initialized")

        except Exception as e:
            print(f"[{self.name}] Failed to initialize: {e}")
            self.client = None

    def transcribe(self, audio: np.ndarray, mic_name: str = "") -> TranscriptionResult:
        """
        Transcribe audio using Groq Whisper API.

        Args:
            audio: Audio data (16kHz, mono, float32)
            mic_name: Microphone name for metadata

        Returns:
            TranscriptionResult with text and timing
        """
        start = time.time()
        text = ""

        if self.client is None:
            return TranscriptionResult(
                text="",
                provider=self.name,
                mic=mic_name,
                latency_ms=0,
            )

        try:
            # Convert to WAV bytes
            audio_bytes = _audio_to_wav_bytes(audio)
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"

            # Call Groq API
            response = self.client.audio.transcriptions.create(
                file=audio_file,
                model=self.model,
                temperature=0.0,
            )

            text = response.text

        except Exception as e:
            print(f"[{self.name}] Transcription error: {e}")
            text = ""

        latency_ms = int((time.time() - start) * 1000)

        return TranscriptionResult(
            text=text,
            provider=self.name,
            mic=mic_name,
            latency_ms=latency_ms,
        )

    def shutdown(self) -> None:
        """Close client."""
        self.client = None
        print(f"[{self.name}] Shutdown")
