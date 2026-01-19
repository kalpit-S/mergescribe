"""
Parakeet MLX provider for local transcription.

Uses Apple Silicon optimizations via MLX framework.
"""

import gc
import threading
import time
from typing import Optional

import numpy as np

from . import Provider
from ..types import TranscriptionResult


class ParakeetProvider(Provider):
    """
    Local transcription using Parakeet MLX model.

    The model is loaded once on initialize() and kept in memory.
    Thread-safe via lock (MLX models aren't thread-safe).
    """

    name = "parakeet"

    def __init__(self):
        self.model = None
        self.preprocessor_config = None
        self._lock = threading.Lock()
        self._transcription_count = 0
        self._CACHE_CLEAR_INTERVAL = 10  # Clear MLX cache every N transcriptions

    def initialize(self) -> None:
        """Load Parakeet model weights."""
        try:
            import mlx.core as mx
            from parakeet_mlx import from_pretrained
            from parakeet_mlx.audio import get_logmel

            print(f"[{self.name}] Loading model...")
            self.model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
            self.preprocessor_config = self.model.preprocessor_config

            # Warmup inference to ensure model is fully loaded
            dummy_audio = mx.array(np.zeros(1600, dtype=np.float32))  # 0.1s at 16kHz
            mel = get_logmel(dummy_audio, self.preprocessor_config)
            _ = self.model.generate(mel)

            print(f"[{self.name}] Initialized")

        except Exception as e:
            print(f"[{self.name}] Failed to initialize: {e}")
            self.model = None

    def transcribe(self, audio: np.ndarray, mic_name: str = "") -> TranscriptionResult:
        """
        Transcribe audio using Parakeet MLX.

        Args:
            audio: Audio data (16kHz, mono, float32)
            mic_name: Microphone name for metadata

        Returns:
            TranscriptionResult with text and timing
        """
        start = time.time()
        text = ""

        with self._lock:
            if self.model is None:
                return TranscriptionResult(
                    text="",
                    provider=self.name,
                    mic=mic_name,
                    latency_ms=0,
                )

            try:
                import mlx.core as mx
                from parakeet_mlx.audio import get_logmel

                # Ensure audio is float32
                audio_data = audio.astype(np.float32)

                # Resample if needed (model expects 16kHz)
                target_sr = self.preprocessor_config.sample_rate
                if len(audio_data) > 0:
                    # Check if we need to resample based on expected audio length
                    # In production, we control sample rate at capture time
                    # But for testing, audio may be at different sample rates
                    pass

                audio_mx = mx.array(audio_data)
                mel = get_logmel(audio_mx, self.preprocessor_config)

                alignments = self.model.generate(mel)
                text = "".join([seg.text for seg in alignments])

                # Clean up to prevent memory accumulation
                del audio_mx
                del mel
                del alignments

                # Clear memory cache periodically (not every call - expensive)
                self._transcription_count += 1
                if self._transcription_count >= self._CACHE_CLEAR_INTERVAL:
                    self._transcription_count = 0
                    if hasattr(mx, "clear_cache"):
                        mx.clear_cache()
                    elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                        mx.metal.clear_cache()

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
        """Unload model weights."""
        with self._lock:
            self.model = None
            self.preprocessor_config = None

        gc.collect()
        print(f"[{self.name}] Shutdown")
