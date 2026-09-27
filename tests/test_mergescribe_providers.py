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
            if provider.model is None:
                pytest.skip("Parakeet model not available")
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

    def test_non_speech_annotations_are_stripped(self):
        """A raw transcript can be typed as it stands, so sound tags must not reach it."""
        from mergescribe.providers.openrouter_stt import strip_annotations

        fish = "<|speaker:0|> So I was [pause] thinking [typing sound] we should ship it."
        assert strip_annotations(fish) == "So I was thinking we should ship it."
        assert strip_annotations("[BLANK_AUDIO]") == ""
        assert strip_annotations("Testing, 1, 2, 3.") == "Testing, 1, 2, 3."

    def _sent(self, monkeypatch, model, keyterms):
        """The JSON body a transcription request sends."""
        from mergescribe.providers import openrouter_stt
        from mergescribe.providers.openrouter_stt import OpenRouterSTTProvider

        sent = []

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"text": "ok"}

        class FakeSession:
            headers: dict = {}

            def post(self, url, json=None, **kwargs):
                sent.append(json)
                return FakeResponse()

            def close(self):
                pass

        monkeypatch.setattr(openrouter_stt.requests, "Session", FakeSession)
        OpenRouterSTTProvider(api_key="k", model=model, keyterms=keyterms)._call_stt_endpoint(b"wav")
        return sent[0]

    def test_assemblyai_is_primed_with_the_known_terms(self, monkeypatch):
        body = self._sent(monkeypatch, "assemblyai/universal-3-5-pro", lambda: ["Claude", "Postgres"])
        assert body["provider"] == {"options": {"assemblyai": {"keyterms_prompt": ["Claude", "Postgres"]}}}

    def test_other_models_get_no_provider_options(self, monkeypatch):
        body = self._sent(monkeypatch, "microsoft/mai-transcribe-2", lambda: ["Claude"])
        assert "provider" not in body

    def test_no_terms_or_a_broken_source_sends_none(self, monkeypatch):
        def broken():
            raise OSError("vocabulary unreadable")

        assert "provider" not in self._sent(monkeypatch, "assemblyai/universal-3-5-pro", lambda: [])
        assert "provider" not in self._sent(monkeypatch, "assemblyai/universal-3-5-pro", broken)

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


class TestReasoning:
    """Correction asks for reasoning off, or as little as the model allows."""

    STREAM = [b'data: {"choices":[{"delta":{"content":"Ship it."}}]}', b"data: [DONE]"]

    def _session(self, monkeypatch, replies):
        from mergescribe import correct

        sent = []

        class Reply:
            def __init__(self, status, text="", lines=()):
                self.status_code, self.text, self._lines = status, text, list(lines)

            def iter_lines(self):
                return iter(self._lines)

        class Session:
            def post(self, url, json=None, **kwargs):
                sent.append(json["reasoning"]["effort"])
                return Reply(*replies.pop(0))

        monkeypatch.setattr(correct, "_openrouter_session", Session())
        monkeypatch.setattr(correct, "_NEEDS_SOME_REASONING", set())
        return sent

    def _config(self, model="google/gemini-3.8-flash", effort=""):
        from unittest.mock import Mock

        from mergescribe.types import ConfigSnapshot

        config = Mock(spec=ConfigSnapshot)
        config.openrouter_api_key = "k"
        config.openrouter_correction_model = model
        config.openrouter_correction_reasoning_effort = effort
        config.openrouter_correction_provider_order = []
        config.openrouter_correction_allow_fallbacks = True
        return config

    def test_reasoning_is_off_unless_chosen(self, monkeypatch):
        from mergescribe.correct import _call_openrouter

        sent = self._session(monkeypatch, [(200, "", self.STREAM)])
        assert _call_openrouter("p", "s", self._config()) == "Ship it."
        assert sent == ["none"]

    def test_a_model_that_refuses_off_gets_minimal_and_it_is_remembered(self, monkeypatch):
        from mergescribe.correct import _call_openrouter

        refused = (400, '{"error":{"message":"Reasoning is mandatory for this endpoint"}}')
        sent = self._session(monkeypatch, [refused, (200, "", self.STREAM), (200, "", self.STREAM)])
        assert _call_openrouter("p", "s", self._config()) == "Ship it."
        assert _call_openrouter("p", "s", self._config("google/gemini-3.8-flash:nitro")) == "Ship it."
        assert sent == ["none", "minimal", "minimal"]   # one refusal, then straight to minimal

    def test_a_chosen_effort_is_sent_as_chosen(self, monkeypatch):
        from mergescribe.correct import _call_openrouter

        sent = self._session(monkeypatch, [(200, "", self.STREAM)])
        _call_openrouter("p", "s", self._config(effort="low"))
        assert sent == ["low"]

    def test_other_errors_are_not_mistaken_for_a_refusal(self, monkeypatch):
        from mergescribe.correct import _call_openrouter

        sent = self._session(monkeypatch, [(400, '{"error":{"message":"invalid model"}}')])
        assert _call_openrouter("p", "s", self._config()) == ""
        assert sent == ["none"]


class TestPromptParts:
    def _result(self, text, provider, chunk):
        from mergescribe.types import TranscriptionResult

        return TranscriptionResult(text=text, provider=provider, mic="m", latency_ms=0, chunk=chunk)

    def test_a_phrase_said_in_two_parts_is_kept_twice_in_order(self):
        from mergescribe.correct import _build_prompt

        prompt = _build_prompt([self._result("part two", "a", 2), self._result("ship it", "a", 1),
                                self._result("ship it", "b", 1), self._result("ship it", "a", 3)], None)
        body = prompt.split("Transcriptions", 1)[1]
        assert body.count("ship it") == 2          # once per part it was said in; duplicates within a part merge
        assert body.index("Part 1") < body.index("Part 2") < body.index("Part 3")
        assert body.index("part two") > body.index("Part 2") > body.index("ship it")

    def test_one_part_reads_as_before(self):
        from mergescribe.correct import _build_prompt

        prompt = _build_prompt([self._result("ship it", "a", 1), self._result("Ship it.", "b", 1)], None)
        assert "Part" not in prompt and "[a/m]: ship it" in prompt


class TestABrokenStream:
    """Tokens are typed as they arrive, so a retry must never type them again."""

    def _correct(self, monkeypatch, replies):
        from unittest.mock import Mock

        from mergescribe import correct
        from mergescribe.types import ConfigSnapshot, TranscriptionResult

        calls = []

        def fake_stream(prompt, system, config, on_delta=None, *args, **kwargs):
            text, complete, emit = replies[len(calls)]
            calls.append(on_delta is not None)
            if on_delta is not None:
                for token in emit:
                    on_delta(token)
            return text, complete

        monkeypatch.setattr(correct, "_stream_openrouter", fake_stream)
        config = Mock(spec=ConfigSnapshot)
        config.openrouter_api_key = "k"
        config.openrouter_correction_model = "m"
        config.learn_vocabulary = False
        typed = []
        result = correct.correct_with_llm(
            [TranscriptionResult(text="ship it friday", provider="p", mic="m", latency_ms=0)],
            None, config, on_delta=typed.append)
        return result, "".join(typed), calls

    def test_the_rest_is_typed_when_a_second_answer_continues_the_first(self, monkeypatch):
        result, typed, calls = self._correct(monkeypatch, [
            ("Ship it", False, ["Ship ", "it"]),          # the stream dies after two tokens
            ("Ship it Friday.", True, []),               # asked again, without streaming
        ])
        assert typed == "Ship it Friday." and result == "Ship it Friday."
        assert calls == [True, False]

    def test_an_answer_that_does_not_continue_it_is_handed_over_whole(self, monkeypatch):
        from mergescribe.correct import CorrectionInterrupted

        with pytest.raises(CorrectionInterrupted) as caught:
            self._correct(monkeypatch, [("Ship it", False, ["Ship ", "it"]), ("We ship Friday.", True, [])])
        assert caught.value.shown == "Ship it" and caught.value.complete == "We ship Friday."

    def test_a_failure_before_anything_was_typed_just_retries(self, monkeypatch):
        result, typed, calls = self._correct(monkeypatch, [("", False, []), ("Ship it Friday.", True, ["Ship it Friday."])])
        assert typed == "Ship it Friday." and calls == [True, True]
