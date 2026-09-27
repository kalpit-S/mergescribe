"""
Integration tests for mergescribe.

Tests the full flow from audio chunks through transcription and correction.
"""

import os
import pytest
import numpy as np
import soundfile as sf
from unittest.mock import Mock


def metal_available() -> bool:
    """True when MLX can run the real model on worker threads.

    Parakeet runs on worker threads, and MLX's CPU fallback has no stream in
    them ("There is no Stream(cpu, 1) in current thread"). GitHub's macOS
    runners report Metal as available yet still land on that CPU path, so CI
    is ruled out by name rather than trusted to answer honestly.
    """
    if os.environ.get("CI"):
        return False
    try:
        import mlx.core as mx
        return mx.metal.is_available()
    except Exception:
        return False


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

        consensus = check_consensus(results, config)
        assert consensus == "Hello world"

    def test_same_provider_two_mics_is_not_consensus(self):
        """One model agreeing with itself across mics proves nothing about model bias."""
        from mergescribe.consensus import check_consensus
        from mergescribe.types import TranscriptionResult, ConfigSnapshot

        results = [
            TranscriptionResult(text="Ableton routing", provider="parakeet", mic="mbp", latency_ms=100),
            TranscriptionResult(text="Ableton routing", provider="parakeet", mic="solocast", latency_ms=100),
        ]

        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 2

        assert check_consensus(results, config) is None

    def test_two_providers_agreeing_is_consensus(self):
        """Distinct models agreeing is real cross-model evidence."""
        from mergescribe.consensus import check_consensus
        from mergescribe.types import TranscriptionResult, ConfigSnapshot

        results = [
            TranscriptionResult(text="Ship it today", provider="parakeet", mic="mbp", latency_ms=100),
            TranscriptionResult(text="Ship it today", provider="fish-audio", mic="mbp", latency_ms=100),
        ]

        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 2

        assert check_consensus(results, config) == "Ship it today"

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

        consensus = check_consensus(results, config)
        # Should match and return first one (with punctuation)
        assert consensus == "Hello world."

    def test_agreeing_on_speech_with_filler_is_still_consensus(self):
        """Whether to drop the "um" is the correction's job, not a reason to keep waiting."""
        from mergescribe.consensus import check_consensus
        from mergescribe.types import TranscriptionResult, ConfigSnapshot

        results = [TranscriptionResult(text="Um, ship it today", provider=p, mic="m", latency_ms=1) for p in ("a", "b")]
        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 2
        assert check_consensus(results, config) == "Um, ship it today"

    def test_a_long_agreement_is_still_consensus(self):
        from mergescribe.consensus import check_consensus
        from mergescribe.types import TranscriptionResult, ConfigSnapshot

        long = " ".join(["word"] * 40)
        results = [TranscriptionResult(text=long, provider=p, mic="m", latency_ms=1) for p in ("a", "b")]
        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 2
        assert check_consensus(results, config) == long

    def test_punctuation_that_splits_a_word_is_not_ignored(self):
        """ "re-sign" and "resign" are different words; commas and full stops are not."""
        from mergescribe.consensus import normalize_for_matching as norm

        assert norm("re-sign the lease") != norm("resign the lease")
        assert norm("Hello, world.") == norm("hello world")
        assert norm("Don't ship it") == norm("dont ship it")

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

        consensus = check_consensus(results, config)
        assert consensus is None


class TestPromptBuilding:
    """Tests for LLM prompt building."""

    def test_build_prompt_single_result(self):
        """Test prompt with single transcription."""
        from mergescribe.correct import _build_prompt
        from mergescribe.types import TranscriptionResult

        results = [
            TranscriptionResult(text="Hello world", provider="parakeet", mic="builtin", latency_ms=100),
        ]

        prompt = _build_prompt(results, None)

        # Prompt now just contains data, instructions are in system message
        assert "[parakeet/builtin]: Hello world" in prompt
        assert "Transcriptions:" in prompt

    def test_build_prompt_multiple_results(self):
        """Test prompt with multiple transcriptions."""
        from mergescribe.correct import _build_prompt
        from mergescribe.types import TranscriptionResult, AppContext

        results = [
            TranscriptionResult(text="Hello world", provider="parakeet", mic="m1", latency_ms=100),
            TranscriptionResult(text="Hello, world!", provider="groq", mic="m1", latency_ms=200),
        ]

        context = AppContext(
            app_name="VS Code",
            window_title="test.py",
            bundle_id="com.microsoft.VSCode",
        )

        prompt = _build_prompt(results, context)

        # Prompt now just contains data, instructions are in system message
        assert "[parakeet/m1]: Hello world" in prompt
        assert "[groq/m1]: Hello, world!" in prompt
        assert "VS Code" in prompt
        assert "Transcriptions:" in prompt


class TestEndToEndFlow:
    """End-to-end integration tests."""

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
