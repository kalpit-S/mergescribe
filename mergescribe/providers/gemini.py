"""
Google Gemini API provider for cloud transcription via OpenRouter.
"""

import base64
import io
import time
from typing import Optional

import numpy as np
import requests
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


class GeminiProvider(Provider):
    """
    Cloud transcription using Google Gemini via OpenRouter.

    Multimodal model that can handle audio transcription.
    Uses OpenRouter API for access.
    """

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "google/gemini-2.5-flash",
        prompt: str = "Transcribe this speech exactly as spoken."
    ):
        self.api_key = api_key  # OpenRouter API key
        self.model = model if "/" in model else f"google/{model}"
        self.prompt = prompt
        self._initialized = False

    def initialize(self) -> None:
        """Validate API key."""
        if not self.api_key:
            print(f"[{self.name}] No API key provided")
            return

        self._initialized = True
        print(f"[{self.name}] Initialized (model: {self.model})")

    def transcribe(self, audio: np.ndarray, mic_name: str = "") -> TranscriptionResult:
        """
        Transcribe audio using Gemini via OpenRouter.

        Args:
            audio: Audio data (16kHz, mono, float32)
            mic_name: Microphone name for metadata

        Returns:
            TranscriptionResult with text and timing
        """
        start = time.time()
        text = ""

        if not self._initialized:
            return TranscriptionResult(
                text="",
                provider=self.name,
                mic=mic_name,
                latency_ms=0,
            )

        try:
            # Convert audio to base64-encoded WAV
            audio_bytes = _audio_to_wav_bytes(audio)
            base64_audio = base64.b64encode(audio_bytes).decode("utf-8")

            # Build request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {"data": base64_audio, "format": "wav"},
                            },
                        ],
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 4000,
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=20,
            )
            response.raise_for_status()
            result = response.json()

            text = result["choices"][0]["message"]["content"] or ""

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
        """Nothing to clean up."""
        self._initialized = False
        print(f"[{self.name}] Shutdown")
