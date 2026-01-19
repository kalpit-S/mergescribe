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


class TestGroqProvider:
    """Tests for Groq Whisper provider."""

    def test_initialization_without_key(self):
        """Test provider handles missing API key gracefully."""
        from mergescribe.providers.groq import GroqProvider

        provider = GroqProvider(api_key="")
        provider.initialize()

        # Should not crash, but client should be None
        # (actual behavior depends on groq library)
        provider.shutdown()

    def test_initialization_with_invalid_key(self):
        """Test provider handles invalid API key."""
        from mergescribe.providers.groq import GroqProvider

        provider = GroqProvider(api_key="invalid_key")
        provider.initialize()
        provider.shutdown()

    @pytest.mark.skipif(
        not os.environ.get("GROQ_API_KEY"),
        reason="GROQ_API_KEY not set"
    )
    def test_transcription(self):
        """Test transcription with valid API key."""
        from mergescribe.providers.groq import GroqProvider

        audio = load_test_audio_as_array()
        api_key = os.environ.get("GROQ_API_KEY", "")

        provider = GroqProvider(api_key=api_key)
        provider.initialize()

        result = provider.transcribe(audio, mic_name="test_mic")

        assert result.provider == "groq"
        assert result.mic == "test_mic"
        assert isinstance(result.text, str)
        assert len(result.text) > 0
        assert result.latency_ms > 0

        text_lower = result.text.lower()
        assert any(word in text_lower for word in ["testing", "one", "two", "three"])

        provider.shutdown()


class TestGeminiProvider:
    """Tests for Gemini provider via OpenRouter."""

    def test_initialization_without_key(self):
        """Test provider handles missing API key gracefully."""
        from mergescribe.providers.gemini import GeminiProvider

        provider = GeminiProvider(api_key="")
        provider.initialize()

        # Should not crash
        assert not provider._initialized
        provider.shutdown()

    @pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set"
    )
    def test_transcription(self):
        """Test transcription with valid API key."""
        from mergescribe.providers.gemini import GeminiProvider

        audio = load_test_audio_as_array()
        api_key = os.environ.get("OPENROUTER_API_KEY", "")

        provider = GeminiProvider(api_key=api_key)
        provider.initialize()

        result = provider.transcribe(audio, mic_name="test_mic")

        assert result.provider == "gemini"
        assert result.mic == "test_mic"
        assert isinstance(result.text, str)
        assert len(result.text) > 0
        assert result.latency_ms > 0

        text_lower = result.text.lower()
        assert any(word in text_lower for word in ["testing", "one", "two", "three"])

        provider.shutdown()


class TestProviderRegistry:
    """Tests for provider registry."""

    def test_registry_creation(self):
        """Test registry can be created."""
        from mergescribe.providers import ProviderRegistry

        registry = ProviderRegistry()
        assert len(registry.providers) == 0
        registry.shutdown()

    def test_registry_parallel_transcription(self):
        """Test registry can run providers in parallel."""
        try:
            from mergescribe.providers import ProviderRegistry
            from mergescribe.providers.parakeet import ParakeetProvider

            audio = load_test_audio_as_array()

            registry = ProviderRegistry()

            # Only test with parakeet if available
            try:
                provider = ParakeetProvider()
                provider.initialize()
                if provider.model is not None:
                    registry.providers["parakeet"] = provider
            except ImportError:
                pytest.skip("No providers available for test")

            if not registry.providers:
                pytest.skip("No providers available for test")

            results = registry.transcribe_all(audio, mic_name="test_mic")

            assert len(results) > 0
            for result in results:
                assert result.text  # Non-empty text

            registry.shutdown()

        except ImportError as e:
            pytest.skip(f"Dependencies not available: {e}")


class TestAudioConversion:
    """Tests for audio conversion utilities."""

    def test_wav_conversion(self):
        """Test numpy to WAV bytes conversion."""
        from mergescribe.providers.groq import _audio_to_wav_bytes

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
        from mergescribe.providers.groq import _audio_to_wav_bytes

        # Create test audio with a sine wave
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz sine

        wav_bytes = _audio_to_wav_bytes(audio)

        # Read it back
        audio_back, sr = sf.read(io.BytesIO(wav_bytes))

        assert sr == 16000
        # Allow some precision loss from int16 conversion
        np.testing.assert_allclose(audio, audio_back, atol=1e-4)
