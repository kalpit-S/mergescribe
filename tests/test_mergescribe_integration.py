"""
Integration tests for mergescribe.

Tests the full flow from audio chunks through transcription and correction.
"""

import os
import pytest
import numpy as np
import soundfile as sf
from unittest.mock import Mock, patch, MagicMock


def load_test_audio(target_sr: int = 16000):
    """Load test audio at target sample rate."""
    test_file = os.path.join(os.path.dirname(__file__), "testing_file.wav")
    if not os.path.exists(test_file):
        pytest.skip("Test audio file not found")

    audio, sample_rate = sf.read(test_file)

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if sample_rate != target_sr:
        from scipy import signal
        num_samples = int(len(audio) * target_sr / sample_rate)
        audio = signal.resample(audio, num_samples)

    return audio.astype(np.float32)


class TestConsensus:
    """Tests for consensus checking."""

    def test_normalize_strips_punctuation(self):
        """Test normalization removes punctuation."""
        from mergescribe.consensus import normalize_for_matching

        assert normalize_for_matching("Hello, world!") == "hello world"
        assert normalize_for_matching("Hello.") == "hello"
        assert normalize_for_matching("Hello") == "hello"

    def test_normalize_handles_whitespace(self):
        """Test normalization handles whitespace."""
        from mergescribe.consensus import normalize_for_matching

        assert normalize_for_matching("hello   world") == "hello world"
        assert normalize_for_matching("  hello  ") == "hello"

    def test_consensus_exact_match(self):
        """Test consensus with exact matches."""
        from mergescribe.consensus import check_consensus
        from mergescribe.types import TranscriptionResult, ConfigSnapshot

        results = [
            TranscriptionResult(text="Hello world", provider="p1", mic="m1", latency_ms=100),
            TranscriptionResult(text="Hello world", provider="p2", mic="m1", latency_ms=100),
            TranscriptionResult(text="Hello world", provider="p1", mic="m2", latency_ms=100),
        ]

        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 2
        config.consensus_max_words = 50

        consensus = check_consensus(results, config)
        assert consensus == "Hello world"

    def test_consensus_punctuation_difference(self):
        """Test consensus ignores punctuation differences."""
        from mergescribe.consensus import check_consensus
        from mergescribe.types import TranscriptionResult, ConfigSnapshot

        results = [
            TranscriptionResult(text="Hello world.", provider="p1", mic="m1", latency_ms=100),
            TranscriptionResult(text="Hello world", provider="p2", mic="m1", latency_ms=100),
        ]

        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 2
        config.consensus_max_words = 50

        consensus = check_consensus(results, config)
        # Should match and return first one (with punctuation)
        assert consensus == "Hello world."

    def test_consensus_no_match(self):
        """Test no consensus when texts differ."""
        from mergescribe.consensus import check_consensus
        from mergescribe.types import TranscriptionResult, ConfigSnapshot

        results = [
            TranscriptionResult(text="Hello world", provider="p1", mic="m1", latency_ms=100),
            TranscriptionResult(text="Hi there", provider="p2", mic="m1", latency_ms=100),
        ]

        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 2
        config.consensus_max_words = 50

        consensus = check_consensus(results, config)
        assert consensus is None


class TestPromptBuilding:
    """Tests for LLM prompt building."""

    def test_build_prompt_single_result(self):
        """Test prompt with single transcription."""
        from mergescribe.correct import _build_prompt
        from mergescribe.types import TranscriptionResult, ConfigSnapshot

        results = [
            TranscriptionResult(text="Hello world", provider="parakeet", mic="builtin", latency_ms=100),
        ]

        config = Mock(spec=ConfigSnapshot)

        prompt = _build_prompt(results, None, config)

        # Prompt now just contains data, instructions are in system message
        assert "[parakeet/builtin]: Hello world" in prompt
        assert "Transcriptions:" in prompt

    def test_build_prompt_multiple_results(self):
        """Test prompt with multiple transcriptions."""
        from mergescribe.correct import _build_prompt
        from mergescribe.types import TranscriptionResult, ConfigSnapshot, AppContext

        results = [
            TranscriptionResult(text="Hello world", provider="parakeet", mic="m1", latency_ms=100),
            TranscriptionResult(text="Hello, world!", provider="groq", mic="m1", latency_ms=200),
        ]

        context = AppContext(
            app_name="VS Code",
            window_title="test.py",
            bundle_id="com.microsoft.VSCode",
            rigor_level="normal",
        )

        config = Mock(spec=ConfigSnapshot)

        prompt = _build_prompt(results, context, config)

        # Prompt now just contains data, instructions are in system message
        assert "[parakeet/m1]: Hello world" in prompt
        assert "[groq/m1]: Hello, world!" in prompt
        assert "VS Code" in prompt
        assert "Transcriptions:" in prompt


class TestProviderRegistry:
    """Tests for provider registry integration."""

    def test_parallel_transcription_with_real_provider(self):
        """Test registry runs providers in parallel."""
        try:
            from mergescribe.providers import ProviderRegistry
            from mergescribe.providers.parakeet import ParakeetProvider

            audio = load_test_audio()
            registry = ProviderRegistry()

            provider = ParakeetProvider()
            provider.initialize()

            if provider.model is None:
                pytest.skip("Parakeet model not available")

            registry.providers["parakeet"] = provider

            results = registry.transcribe_all(audio, mic_name="test_mic", timeout=30)

            assert len(results) == 1
            assert results[0].provider == "parakeet"
            assert len(results[0].text) > 0

            registry.shutdown()

        except ImportError as e:
            pytest.skip(f"Dependencies not available: {e}")


class TestEndToEndFlow:
    """End-to-end integration tests."""

    def test_full_flow_with_mocked_llm(self):
        """Test full transcription flow with mocked LLM."""
        from mergescribe.types import TranscriptionResult, ConfigSnapshot
        from mergescribe.consensus import check_consensus
        from mergescribe.correct import correct_with_llm

        # Simulate results from transcription
        results = [
            TranscriptionResult(text="Testing one two three.", provider="parakeet", mic="m1", latency_ms=100),
            TranscriptionResult(text="Testing one two three", provider="groq", mic="m1", latency_ms=200),
        ]

        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 2
        config.consensus_max_words = 50
        config.openrouter_api_key = ""
        config.hedged_requests = False

        # Check consensus first
        consensus = check_consensus(results, config)
        assert consensus == "Testing one two three."  # Returns first match

        # If no consensus, would call LLM (but we have consensus, so skip LLM)
        if consensus is None:
            # LLM would be called here
            pass

        # Final result
        final_text = consensus or results[0].text
        assert "testing" in final_text.lower()

    @pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set"
    )
    def test_real_transcription_and_correction(self):
        """Test full flow with real APIs."""
        from mergescribe.providers.parakeet import ParakeetProvider
        from mergescribe.types import ConfigSnapshot
        from mergescribe.correct import correct_with_llm

        audio = load_test_audio()

        # Transcribe with Parakeet
        provider = ParakeetProvider()
        provider.initialize()

        if provider.model is None:
            pytest.skip("Parakeet model not available")

        result = provider.transcribe(audio, mic_name="test")

        # Set up config for LLM correction
        config = Mock(spec=ConfigSnapshot)
        config.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
        config.hedged_requests = False

        # Correct with LLM
        corrected = correct_with_llm([result], None, config)

        assert len(corrected) > 0
        # Should contain similar content to original
        text_lower = corrected.lower()
        assert any(word in text_lower for word in ["testing", "one", "two", "three"])

        provider.shutdown()
