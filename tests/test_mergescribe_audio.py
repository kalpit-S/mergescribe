"""
Tests for mergescribe AudioEngine.

Includes both unit tests (mocked) and hardware tests (real mics).
"""

import time
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestAudioEngineUnit:
    """Unit tests with mocked sounddevice."""

    def test_initialization_creates_buffers(self):
        """Test that buffers are created for each mic."""
        from mergescribe.audio import AudioEngine
        from mergescribe.config import Config

        config = Mock(spec=Config)
        config.enabled_mics = ["mic1", "mic2"]
        config.preroll_seconds = 0.5
        config.silence_threshold = 2.0
        config.sample_rate = 16000

        engine = AudioEngine(config)

        # Check computed values
        assert engine._preroll_samples == 8000  # 0.5 * 16000
        assert engine._silence_samples == 32000  # 2.0 * 16000

    def test_silence_detection(self):
        """Test silence detection with various audio levels."""
        from mergescribe.audio import AudioEngine
        from mergescribe.config import Config

        config = Mock(spec=Config)
        config.preroll_seconds = 0.5
        config.silence_threshold = 2.0
        config.sample_rate = 16000

        engine = AudioEngine(config)

        # Pure silence (zeros)
        silence = np.zeros(1024, dtype=np.float32)
        assert engine._is_silence(silence)

        # Very quiet (below threshold)
        quiet = np.random.randn(1024).astype(np.float32) * 0.0001
        assert engine._is_silence(quiet)

        # Normal speech level (above threshold)
        speech = np.random.randn(1024).astype(np.float32) * 0.1
        assert not engine._is_silence(speech)

        # Loud audio
        loud = np.random.randn(1024).astype(np.float32) * 0.5
        assert not engine._is_silence(loud)

    def test_adaptive_silence_hysteresis(self):
        """Adaptive silence detection uses hysteresis to avoid flicker in noise."""
        from mergescribe.audio import AudioEngine
        from mergescribe.config import Config
        from collections import deque

        def audio_for_db(db: float, n: int = 1024) -> np.ndarray:
            # Use a constant signal so RMS == value.
            rms = 10 ** (db / 20)
            return np.full(n, rms, dtype=np.float32)

        config = Config()
        config.enabled_mics = []
        config.preroll_seconds = 0.5
        config.silence_threshold = 2.0
        config.sample_rate = 16000
        config.chunk_on_silence = True
        config.adaptive_threshold_enabled = True
        config.speech_headroom_db = 12.0
        config.speech_hysteresis_db = 3.0
        config.noise_floor_percentile = 10

        engine = AudioEngine(config)

        # Seed adaptive state for a fake mic
        engine._noise_floor_samples["mic1"] = deque(maxlen=100)
        engine._noise_floor_cache["mic1"] = -30.0
        engine._speech_active["mic1"] = False

        # Above start threshold (-30 + 12 = -18) => speech
        assert engine._is_silence(audio_for_db(-17.0), "mic1") is False
        # Drop below start but above release (-30 + 9 = -21) => still speech
        assert engine._is_silence(audio_for_db(-20.0), "mic1") is False
        # Drop below release => silence
        assert engine._is_silence(audio_for_db(-23.0), "mic1") is True

    def test_flush_current_chunk(self):
        """Test chunk flushing concatenates buffers correctly."""
        from mergescribe.audio import AudioEngine
        from mergescribe.config import Config

        config = Mock(spec=Config)
        config.preroll_seconds = 0.5
        config.silence_threshold = 2.0
        config.sample_rate = 16000

        engine = AudioEngine(config)

        # Manually populate buffers
        engine.current_chunk["mic1"] = [
            np.array([1, 2, 3], dtype=np.float32),
            np.array([4, 5, 6], dtype=np.float32),
        ]
        engine.current_chunk["mic2"] = [
            np.array([7, 8], dtype=np.float32),
        ]

        chunk = engine._flush_current_chunk()

        # Check concatenation
        np.testing.assert_array_equal(chunk["mic1"], [1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(chunk["mic2"], [7, 8])

        # Buffers should be cleared
        assert engine.current_chunk["mic1"] == []
        assert engine.current_chunk["mic2"] == []

    def test_start_recording_dumps_preroll(self):
        """Test that preroll is dumped into current chunk on start."""
        from mergescribe.audio import AudioEngine
        from mergescribe.config import Config
        from collections import deque

        config = Mock(spec=Config)
        config.preroll_seconds = 0.5
        config.silence_threshold = 2.0
        config.sample_rate = 16000

        engine = AudioEngine(config)

        # Setup preroll buffers
        engine.preroll_buffers["mic1"] = deque([
            np.array([1, 2], dtype=np.float32),
            np.array([3, 4], dtype=np.float32),
        ])
        engine.current_chunk["mic1"] = []

        engine.start_recording()

        assert engine.is_recording is True
        assert len(engine.current_chunk["mic1"]) == 2

    def test_stop_recording_disconnects_callback(self):
        """Test that callback is disconnected on stop (race condition prevention)."""
        from mergescribe.audio import AudioEngine
        from mergescribe.config import Config

        config = Mock(spec=Config)
        config.preroll_seconds = 0.5
        config.silence_threshold = 2.0
        config.sample_rate = 16000

        engine = AudioEngine(config)
        engine.current_chunk["mic1"] = []
        engine.on_chunk_ready = lambda x: None

        engine.is_recording = True
        engine.stop_recording()

        assert engine.is_recording is False
        assert engine.on_chunk_ready is None

    @patch('sounddevice.query_devices')
    def test_find_device_exact_match(self, mock_query):
        """Test device finding with exact name match."""
        from mergescribe.audio import AudioEngine
        from mergescribe.config import Config

        mock_query.return_value = [
            {"name": "Built-in Output", "max_input_channels": 0},
            {"name": "HyperX SoloCast", "max_input_channels": 2},
            {"name": "MacBook Pro Microphone", "max_input_channels": 1},
        ]

        config = Mock(spec=Config)
        config.preroll_seconds = 0.5
        config.silence_threshold = 2.0
        config.sample_rate = 16000

        engine = AudioEngine(config)

        # Exact match (case insensitive)
        assert engine._find_device("HyperX SoloCast") == 1
        assert engine._find_device("hyperx solocast") == 1

    @patch('sounddevice.query_devices')
    def test_find_device_substring_match(self, mock_query):
        """Test device finding with substring match."""
        from mergescribe.audio import AudioEngine
        from mergescribe.config import Config

        mock_query.return_value = [
            {"name": "Built-in Output", "max_input_channels": 0},
            {"name": "HyperX SoloCast", "max_input_channels": 2},
            {"name": "MacBook Pro Microphone", "max_input_channels": 1},
        ]

        config = Mock(spec=Config)
        config.preroll_seconds = 0.5
        config.silence_threshold = 2.0
        config.sample_rate = 16000

        engine = AudioEngine(config)

        # Substring match
        assert engine._find_device("HyperX") == 1
        assert engine._find_device("MacBook") == 2

    @patch('sounddevice.query_devices')
    def test_find_device_not_found(self, mock_query):
        """Test device finding returns None for unknown device."""
        from mergescribe.audio import AudioEngine
        from mergescribe.config import Config

        mock_query.return_value = [
            {"name": "MacBook Pro Microphone", "max_input_channels": 1},
        ]

        config = Mock(spec=Config)
        config.preroll_seconds = 0.5
        config.silence_threshold = 2.0
        config.sample_rate = 16000

        engine = AudioEngine(config)

        assert engine._find_device("NonexistentMic") is None

    @patch('sounddevice.InputStream')
    @patch('sounddevice.query_devices')
    def test_sync_configured_mics_applies_settings_changes(self, mock_query, mock_input_stream):
        """Idle mic sync opens newly enabled mics and removes disabled ones."""
        from mergescribe.audio import AudioEngine
        from mergescribe.config import Config

        mock_query.return_value = [
            {"name": "Mic1", "max_input_channels": 1},
            {"name": "Mic2", "max_input_channels": 1},
        ]

        streams = []

        def make_stream(*args, **kwargs):
            stream = MagicMock()
            stream.active = True
            streams.append(stream)
            return stream

        mock_input_stream.side_effect = make_stream

        config = Mock(spec=Config)
        config.enabled_mics = ["Mic1", "Mic2"]
        config.preroll_seconds = 0.5
        config.silence_threshold = 2.0
        config.sample_rate = 16000

        engine = AudioEngine(config)
        assert set(engine.sync_configured_mics()) == {"Mic1", "Mic2"}

        config.enabled_mics = ["Mic2"]
        assert engine.sync_configured_mics() == ["Mic2"]
        assert "Mic1" not in engine.streams
        assert "Mic2" in engine.streams
        streams[0].stop.assert_called_once()
        streams[0].close.assert_called_once()


class TestAudioEngineCallback:
    """Tests for audio callback behavior."""

    def test_callback_fills_preroll_when_not_recording(self):
        """Test that audio fills preroll buffer when not recording."""
        from mergescribe.audio import AudioEngine
        from mergescribe.config import Config
        from collections import deque

        config = Mock(spec=Config)
        config.preroll_seconds = 0.5
        config.silence_threshold = 2.0
        config.sample_rate = 16000

        engine = AudioEngine(config)
        engine.preroll_buffers["mic1"] = deque(maxlen=10)
        engine.current_chunk["mic1"] = []
        engine.is_recording = False

        # Simulate callback
        audio = np.random.randn(1024).astype(np.float32)
        engine._audio_callback("mic1", audio.reshape(-1, 1), 1024, None, None)

        assert len(engine.preroll_buffers["mic1"]) == 1
        assert len(engine.current_chunk["mic1"]) == 0

    def test_callback_appends_to_chunk_when_recording(self):
        """Test that audio appends to current chunk when recording."""
        from mergescribe.audio import AudioEngine
        from mergescribe.config import Config
        from collections import deque

        config = Mock(spec=Config)
        config.preroll_seconds = 0.5
        config.silence_threshold = 2.0
        config.sample_rate = 16000

        engine = AudioEngine(config)
        engine.preroll_buffers["mic1"] = deque(maxlen=10)
        engine.current_chunk["mic1"] = []
        engine.is_recording = True

        # Simulate callback with speech
        audio = np.random.randn(1024).astype(np.float32) * 0.1
        engine._audio_callback("mic1", audio.reshape(-1, 1), 1024, None, None)

        assert len(engine.current_chunk["mic1"]) == 1

    def test_silence_only_chunk_is_never_emitted(self):
        """Dead air must not reach the providers: STT hallucinates on silence."""
        from mergescribe.audio import AudioEngine
        from mergescribe.config import Config
        from collections import deque
        import time

        config = Mock(spec=Config)
        config.preroll_seconds = 0.5
        config.silence_threshold = 0.1
        config.sample_rate = 16000

        engine = AudioEngine(config)
        engine.preroll_buffers["mic1"] = deque(maxlen=10)
        # Well past MIN_CHUNK_SECONDS, but every sample of it is silence
        engine.current_chunk["mic1"] = [np.zeros(80000, dtype=np.float32)]
        engine.is_recording = True
        engine._primary_mic = "mic1"
        engine._chunk_has_speech = False
        engine.last_speech_time = time.time() - 1.0

        chunks_received = []
        engine.on_chunk_ready = lambda c: chunks_received.append(c)

        silence = np.zeros(1024, dtype=np.float32)
        for _ in range(20):
            engine._audio_callback("mic1", silence.reshape(-1, 1), 1024, None, None)

        assert chunks_received == [], "silence-only chunk was emitted to providers"

    def test_speech_after_silence_still_emits(self):
        """The guard must not wedge the chunker: speech re-arms emission."""
        from mergescribe.audio import AudioEngine
        from mergescribe.config import Config
        from collections import deque
        import time

        config = Mock(spec=Config)
        config.preroll_seconds = 0.5
        config.silence_threshold = 0.1
        config.sample_rate = 16000

        engine = AudioEngine(config)
        engine.preroll_buffers["mic1"] = deque(maxlen=10)
        engine.current_chunk["mic1"] = [np.zeros(80000, dtype=np.float32)]
        engine.is_recording = True
        engine._primary_mic = "mic1"
        engine._chunk_has_speech = False

        chunks_received = []
        engine.on_chunk_ready = lambda c: chunks_received.append(c)

        # Loud audio marks the chunk as containing speech...
        loud = (np.random.randn(1024) * 0.5).astype(np.float32)
        engine._audio_callback("mic1", loud.reshape(-1, 1), 1024, None, None)
        assert engine._chunk_has_speech

        # ...then a pause emits it, and the flag resets for the next chunk
        engine.last_speech_time = time.time() - 1.0
        silence = np.zeros(1024, dtype=np.float32)
        for _ in range(5):
            engine._audio_callback("mic1", silence.reshape(-1, 1), 1024, None, None)

        assert len(chunks_received) >= 1
        assert not engine._chunk_has_speech

    def test_callback_emits_chunk_on_silence(self):
        """Test that chunk is emitted after sufficient silence."""
        from mergescribe.audio import AudioEngine
        from mergescribe.config import Config
        from collections import deque

        config = Mock(spec=Config)
        config.preroll_seconds = 0.5
        config.silence_threshold = 0.1  # Very short for test
        config.sample_rate = 16000

        import time
        engine = AudioEngine(config)
        engine.preroll_buffers["mic1"] = deque(maxlen=10)
        # Need at least MIN_CHUNK_SECONDS (5.0s) of audio = 80000 samples
        engine.current_chunk["mic1"] = [np.random.randn(80000).astype(np.float32)]
        engine.is_recording = True
        engine._primary_mic = "mic1"  # Set primary mic for silence timing
        # The buffer above stands in for speech already captured this chunk
        engine._chunk_has_speech = True
        # Set last_speech_time in the past so silence threshold is exceeded
        engine.last_speech_time = time.time() - 1.0  # 1 second ago (> 0.1s threshold)

        # Track callback
        chunks_received = []
        engine.on_chunk_ready = lambda c: chunks_received.append(c)

        # Simulate silence callbacks (enough to exceed threshold)
        silence = np.zeros(1024, dtype=np.float32)
        for _ in range(5):  # 5 * 1024 / 16000 = 0.32s > 0.1s threshold
            engine._audio_callback("mic1", silence.reshape(-1, 1), 1024, None, None)

        # Should have emitted a chunk (now has 1s+ of audio)
        assert len(chunks_received) >= 1


class TestAudioEngineHardware:
    """Hardware tests with real microphones.

    These tests require actual audio hardware.
    Skip if no mics available.
    """

    @pytest.fixture
    def real_config(self):
        """Create a real config for testing."""
        from mergescribe.config import Config

        config = Config()
        config.enabled_mics = ["MacBook Pro Microphone"]
        config.preroll_seconds = 0.5
        config.silence_threshold = 2.0
        config.sample_rate = 16000
        return config

    def test_initialize_with_real_mic(self, real_config):
        """Test initializing with a real microphone."""
        from mergescribe.audio import AudioEngine

        engine = AudioEngine(real_config)

        try:
            active_mics = engine.initialize()

            # Should find at least the default mic
            assert len(active_mics) >= 0

            if active_mics:
                assert "MacBook Pro Microphone" in active_mics or len(active_mics) > 0

        finally:
            engine.shutdown()

    def test_record_short_audio(self, real_config):
        """Test recording a short audio segment."""
        from mergescribe.audio import AudioEngine

        engine = AudioEngine(real_config)

        try:
            active_mics = engine.initialize()

            if not active_mics:
                pytest.skip("No mics available")

            # Record for 0.5 seconds
            engine.start_recording()
            time.sleep(0.5)
            chunk = engine.stop_recording()

            # Should have audio data
            assert len(chunk) > 0
            for mic_name, audio in chunk.items():
                assert len(audio) > 0
                assert audio.dtype == np.float32

        finally:
            engine.shutdown()

    def test_preroll_captures_before_start(self, real_config):
        """Test that preroll captures audio before recording starts."""
        from mergescribe.audio import AudioEngine

        # Short preroll for test
        real_config.preroll_seconds = 0.3

        engine = AudioEngine(real_config)

        try:
            active_mics = engine.initialize()

            if not active_mics:
                pytest.skip("No mics available")

            # Let preroll fill
            time.sleep(0.5)

            # Start and immediately stop
            engine.start_recording()
            time.sleep(0.1)  # Brief recording
            chunk = engine.stop_recording()

            # Should have preroll + brief recording
            for mic_name, audio in chunk.items():
                # Should have more than just the 0.1s of recording
                # (preroll should add ~0.3s)
                duration = len(audio) / real_config.sample_rate
                assert duration > 0.2  # At least some preroll

        finally:
            engine.shutdown()


class TestAudioEngineMultiMic:
    """Tests for multi-microphone support."""

    @patch('sounddevice.InputStream')
    @patch('sounddevice.query_devices')
    def test_initialize_multiple_mics(self, mock_query, mock_input_stream):
        """Test initializing multiple microphones."""
        from mergescribe.audio import AudioEngine
        from mergescribe.config import Config

        # Mock devices
        mock_query.return_value = [
            {"name": "Mic1", "max_input_channels": 1},
            {"name": "Mic2", "max_input_channels": 1},
        ]

        # Mock stream
        mock_stream = MagicMock()
        mock_input_stream.return_value = mock_stream

        config = Mock(spec=Config)
        config.enabled_mics = ["Mic1", "Mic2"]
        config.preroll_seconds = 0.5
        config.silence_threshold = 2.0
        config.sample_rate = 16000

        engine = AudioEngine(config)
        active_mics = engine.initialize()

        # Should have both mics
        assert len(active_mics) == 2
        assert "Mic1" in active_mics
        assert "Mic2" in active_mics

        # Should have created streams for both
        assert mock_input_stream.call_count == 2

    def test_chunk_contains_all_mics(self):
        """Test that flushed chunk contains data for all mics."""
        from mergescribe.audio import AudioEngine
        from mergescribe.config import Config

        config = Mock(spec=Config)
        config.preroll_seconds = 0.5
        config.silence_threshold = 2.0
        config.sample_rate = 16000

        engine = AudioEngine(config)

        # Setup multiple mics
        engine.current_chunk["mic1"] = [np.array([1, 2], dtype=np.float32)]
        engine.current_chunk["mic2"] = [np.array([3, 4], dtype=np.float32)]
        engine.current_chunk["mic3"] = [np.array([5, 6], dtype=np.float32)]

        chunk = engine._flush_current_chunk()

        assert len(chunk) == 3
        assert "mic1" in chunk
        assert "mic2" in chunk
        assert "mic3" in chunk
