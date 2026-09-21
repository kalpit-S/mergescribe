"""
Tests for mergescribe providers.

These tests verify the new provider implementations work correctly.
"""

import os
import pytest
import numpy as np
import soundfile as sf


def load_test_audio_as_array(target_sr: int = 16000):
    """Load test audio file as numpy array at target sample rate.

    Args:
        target_sr: Target sample rate (default 16kHz to match AudioEngine)
    """
    test_file = os.path.join(os.path.dirname(__file__), "testing_file.wav")
    if not os.path.exists(test_file):
        pytest.skip("Test audio file not found")

    audio, sample_rate = sf.read(test_file)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Resample to target rate if needed
    if sample_rate != target_sr:
        from scipy import signal
        num_samples = int(len(audio) * target_sr / sample_rate)
        audio = signal.resample(audio, num_samples)

    # Ensure float32
    audio = audio.astype(np.float32)

    return audio


class TestParakeetProvider:
    """Tests for Parakeet MLX provider."""

    def test_initialization(self):
        """Test provider initializes correctly."""
        try:
            from mergescribe.providers.parakeet import ParakeetProvider

            provider = ParakeetProvider()
            provider.initialize()

            # Model should be loaded (or None if not available)
            if provider.model is not None:
                assert provider.preprocessor_config is not None

            provider.shutdown()

        except ImportError as e:
            pytest.skip(f"Parakeet MLX not available: {e}")

    def test_transcription(self):
        """Test transcription produces output."""
        try:
            from mergescribe.providers.parakeet import ParakeetProvider

            audio = load_test_audio_as_array()
            provider = ParakeetProvider()
            provider.initialize()

            if provider.model is None:
                pytest.skip("Parakeet model failed to load")

            result = provider.transcribe(audio, mic_name="test_mic")

            assert result.provider == "parakeet"
            assert result.mic == "test_mic"
            assert isinstance(result.text, str)
            assert len(result.text) > 0
            assert result.latency_ms > 0

            # Check for expected words
            text_lower = result.text.lower()
            assert any(word in text_lower for word in ["testing", "one", "two", "three"])

            provider.shutdown()

        except ImportError as e:
            pytest.skip(f"Parakeet MLX not available: {e}")


class TestOpenRouterSTTProvider:
    """Tests for OpenRouter STT model routing."""

    def test_dedicated_stt_models_use_transcription_endpoint(self):
        from mergescribe.providers.openrouter_stt import OpenRouterSTTProvider

        models = [
            "microsoft/mai-transcribe-1.5",
            "mistralai/voxtral-mini-transcribe",
            "qwen/qwen3-asr-flash-2026-02-10",
        ]

        for model in models:
            provider = OpenRouterSTTProvider(api_key="test", model=model)
            assert provider._use_stt_endpoint is True

    def test_retries_ssl_transport_error_with_fresh_connection(self, monkeypatch):
        from mergescribe.providers import openrouter_stt
        from mergescribe.providers.openrouter_stt import OpenRouterSTTProvider

        attempts = []
        closes = []

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"text": "ok"}

        class FakeSession:
            def __init__(self):
                self.headers = {}

            def post(self, *args, **kwargs):
                attempts.append((args, kwargs))
                if len(attempts) == 1:
                    raise openrouter_stt.requests.exceptions.SSLError("EOF")
                return FakeResponse()

            def close(self):
                closes.append(True)

        monkeypatch.setattr(openrouter_stt.requests, "Session", FakeSession)
        monkeypatch.setattr(openrouter_stt.time, "sleep", lambda _: None)

        provider = OpenRouterSTTProvider(api_key="test", model="microsoft/mai-transcribe-1.5")

        assert provider._call_stt_endpoint(b"wav") == "ok"
        assert len(attempts) == 2
        assert len(closes) == 2


class TestAudioConversion:
    """Tests for audio conversion utilities."""

    def test_wav_conversion(self):
        """Test numpy to WAV bytes conversion."""
        from mergescribe.providers.openrouter_stt import _audio_to_wav_bytes

        # Create test audio (1 second of silence)
        audio = np.zeros(16000, dtype=np.float32)
        wav_bytes = _audio_to_wav_bytes(audio, sample_rate=16000)

        assert isinstance(wav_bytes, bytes)
        assert len(wav_bytes) > 0

        # Should be a valid WAV file (starts with RIFF)
        assert wav_bytes[:4] == b"RIFF"

    def test_wav_conversion_preserves_content(self):
        """Test that conversion doesn't corrupt audio data."""
        import io
        from mergescribe.providers.openrouter_stt import _audio_to_wav_bytes

        # Create test audio with a sine wave
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz sine

        wav_bytes = _audio_to_wav_bytes(audio)

        # Read it back
        audio_back, sr = sf.read(io.BytesIO(wav_bytes))

        assert sr == 16000
        # Allow some precision loss from int16 conversion
        np.testing.assert_allclose(audio, audio_back, atol=1e-4)


class TestReasoningDefaults:
    def test_routing_suffix_inherits_the_base_model_default(self):
        """":nitro" is the same model; it must not silently re-enable reasoning."""
        from mergescribe.correct import OPENROUTER_NO_REASONING_MODELS

        base = "openai/gpt-5.6-luna"
        assert base in OPENROUTER_NO_REASONING_MODELS
        assert f"{base}:nitro".split(":", 1)[0] in OPENROUTER_NO_REASONING_MODELS
