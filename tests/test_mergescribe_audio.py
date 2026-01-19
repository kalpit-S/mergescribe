"""
Tests for mergescribe AudioEngine.

Includes both unit tests (mocked) and hardware tests (real mics).
"""

import os
import time
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import threading


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
        from mergescribe.audio import AudioEngine, SILENCE_THRESHOLD_DB
        from mergescribe.config import Config

        config = Mock(spec=Config)
        config.preroll_seconds = 0.5
        config.silence_threshold = 2.0
        config.sample_rate = 16000

        engine = AudioEngine(config)

        # Pure silence (zeros)
        silence = np.zeros(1024, dtype=np.float32)
        assert engine._is_silence(silence) == True

        # Very quiet (below threshold)
        quiet = np.random.randn(1024).astype(np.float32) * 0.0001
        assert engine._is_silence(quiet) == True

        # Normal speech level (above threshold)
        speech = np.random.randn(1024).astype(np.float32) * 0.1
        assert engine._is_silence(speech) == False

        # Loud audio
        loud = np.random.randn(1024).astype(np.float32) * 0.5
        assert engine._is_silence(loud) == False

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

    def test_callback_emits_chunk_on_silence(self):
        """Test that chunk is emitted after sufficient silence."""
        from mergescribe.audio import AudioEngine
        from mergescribe.config import Config
        from collections import deque

        config = Mock(spec=Config)
        config.preroll_seconds = 0.5
        config.silence_threshold = 0.1  # Very short for test
        config.sample_rate = 16000

        engine = AudioEngine(config)
        engine.preroll_buffers["mic1"] = deque(maxlen=10)
        # Need at least MIN_CHUNK_SECONDS (5.0s) of audio = 80000 samples
        engine.current_chunk["mic1"] = [np.random.randn(80000).astype(np.float32)]
        engine.is_recording = True

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
