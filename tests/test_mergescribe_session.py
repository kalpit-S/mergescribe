"""
Tests for mergescribe Session and SessionManager.

Tests the recording session lifecycle, chunk handling, and coordination.
"""

import time
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import threading
from concurrent.futures import ThreadPoolExecutor


class TestSession:
    """Tests for Session class."""

    def create_session(self, **kwargs):
        """Create a test session with mocks."""
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import ConfigSnapshot
        from uuid import uuid4

        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 2
        config.consensus_max_words = 15
        config.cache_enabled = False
        config.hedged_requests = False
        config.openrouter_api_key = ""

        defaults = {
            "id": uuid4(),
            "config_snapshot": config,
            "providers": Mock(),
            "output_lock": threading.Lock(),
            "on_complete": Mock(),
            "history": TranscriptionHistory(),
        }
        defaults.update(kwargs)
        return Session(**defaults)

    def test_session_start_captures_context(self):
        """Test that session start captures app context."""
        with patch('mergescribe.session.get_app_context') as mock_ctx:
            from mergescribe.types import AppContext
            mock_ctx.return_value = AppContext(
                app_name="Test App",
                window_title="Test Window",
                bundle_id="com.test.app",
                rigor_level="normal"
            )

            session = self.create_session()
            session.start()

            assert session.is_active is True
            assert session.start_time > 0
            assert session.context.app_name == "Test App"

    def test_session_aggregation_uses_consensus(self):
        """Test result aggregation prefers consensus."""
        from mergescribe.types import TranscriptionResult

        session = self.create_session()

        # Add chunk results - first has consensus, second doesn't
        session.chunk_results = [
            # Chunk 1: has consensus
            ([
                TranscriptionResult(text="Hello world", provider="p1", mic="m1", latency_ms=100),
                TranscriptionResult(text="Hello world", provider="p2", mic="m1", latency_ms=100),
            ], "Hello world"),
            # Chunk 2: no consensus
            ([
                TranscriptionResult(text="How are you", provider="p1", mic="m1", latency_ms=100),
                TranscriptionResult(text="How you are", provider="p2", mic="m1", latency_ms=100),
            ], None),
        ]

        chunk_texts, all_results = session._aggregate_results()

        assert len(chunk_texts) == 2
        assert chunk_texts[0] == "Hello world"  # Consensus
        assert chunk_texts[1] == "How are you"  # First result (no consensus)
        assert len(all_results) == 4

    def test_session_empty_chunk_ignored(self):
        """Test that empty chunks are ignored."""
        session = self.create_session()

        # Empty chunk
        empty_chunk = {"mic1": np.array([], dtype=np.float32)}
        session.on_chunk_ready(empty_chunk)

        # No futures should be pending
        assert len(session.pending_futures) == 0

    def test_session_chunk_creates_futures(self):
        """Test that chunk creates transcription futures."""
        session = self.create_session()

        # Mock providers
        mock_provider = Mock()
        mock_provider.name = "test_provider"
        mock_provider.transcribe = Mock(return_value=Mock(text="test"))
        session.providers.values = Mock(return_value=[mock_provider])

        # Non-empty chunk
        chunk = {"mic1": np.random.randn(1000).astype(np.float32)}
        session.on_chunk_ready(chunk)

        # Give it time to submit
        time.sleep(0.1)

        # Should have pending futures
        assert len(session.pending_futures) >= 1


class TestSessionManager:
    """Tests for SessionManager class."""

    def test_start_session_creates_session(self):
        """Test that start_session creates and returns a session."""
        from mergescribe.session import SessionManager
        from mergescribe.types import ConfigSnapshot

        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 2
        config.consensus_max_words = 15
        config.cache_enabled = False

        manager = SessionManager(
            config_snapshot_fn=lambda: config,
            providers=Mock(),
        )

        session = manager.start_session()

        assert session is not None
        assert manager.active_session == session

    def test_start_session_rejects_when_busy(self):
        """Test that new session is rejected when one is active."""
        from mergescribe.session import SessionManager
        from mergescribe.types import ConfigSnapshot

        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 2
        config.consensus_max_words = 15
        config.cache_enabled = False

        manager = SessionManager(
            config_snapshot_fn=lambda: config,
            providers=Mock(),
        )

        with patch('mergescribe.session.play_busy_sound') as mock_sound:
            # Start first session
            session1 = manager.start_session()
            session1.is_active = True  # Simulate active session

            # Try to start second
            session2 = manager.start_session()

            assert session2 is None
            mock_sound.assert_called_once()

    def test_is_busy_when_session_active(self):
        """Test is_busy returns True when session is active."""
        from mergescribe.session import SessionManager
        from mergescribe.types import ConfigSnapshot

        config = Mock(spec=ConfigSnapshot)
        manager = SessionManager(
            config_snapshot_fn=lambda: config,
            providers=Mock(),
        )

        assert manager.is_busy() is False

        session = manager.start_session()
        session.is_active = True

        assert manager.is_busy() is True

    def test_session_completion_clears_active(self):
        """Test that session completion clears active session."""
        from mergescribe.session import SessionManager
        from mergescribe.types import ConfigSnapshot

        config = Mock(spec=ConfigSnapshot)
        manager = SessionManager(
            config_snapshot_fn=lambda: config,
            providers=Mock(),
        )

        session = manager.start_session()
        assert manager.active_session == session

        # Simulate session completion
        manager._on_session_complete(session)

        assert manager.active_session is None


class TestSessionTranscription:
    """Tests for session transcription flow."""

    def test_transcription_with_consensus(self):
        """Test that consensus is detected correctly."""
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import ConfigSnapshot, TranscriptionResult
        from uuid import uuid4

        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 2
        config.consensus_max_words = 15

        mock_provider1 = Mock()
        mock_provider1.name = "p1"
        mock_provider1.transcribe = Mock(return_value=TranscriptionResult(
            text="Hello world", provider="p1", mic="mic1", latency_ms=100
        ))

        mock_provider2 = Mock()
        mock_provider2.name = "p2"
        mock_provider2.transcribe = Mock(return_value=TranscriptionResult(
            text="Hello world", provider="p2", mic="mic1", latency_ms=100
        ))

        mock_registry = Mock()
        mock_registry.values = Mock(return_value=[mock_provider1, mock_provider2])

        session = Session(
            id=uuid4(),
            config_snapshot=config,
            providers=mock_registry,
            output_lock=threading.Lock(),
            on_complete=Mock(),
            history=TranscriptionHistory(),
        )

        # Run transcription
        chunk = {"mic1": np.random.randn(1000).astype(np.float32)}
        session._transcribe_chunk_with_consensus(chunk)

        # Should have results with consensus
        assert len(session.chunk_results) == 1
        results, consensus = session.chunk_results[0]
        assert consensus == "Hello world"

    def test_transcription_without_consensus(self):
        """Test handling when no consensus is reached."""
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import ConfigSnapshot, TranscriptionResult
        from uuid import uuid4

        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 2
        config.consensus_max_words = 15

        mock_provider1 = Mock()
        mock_provider1.name = "p1"
        mock_provider1.transcribe = Mock(return_value=TranscriptionResult(
            text="Hello world", provider="p1", mic="mic1", latency_ms=100
        ))

        mock_provider2 = Mock()
        mock_provider2.name = "p2"
        mock_provider2.transcribe = Mock(return_value=TranscriptionResult(
            text="Hi there", provider="p2", mic="mic1", latency_ms=100
        ))

        mock_registry = Mock()
        mock_registry.values = Mock(return_value=[mock_provider1, mock_provider2])

        session = Session(
            id=uuid4(),
            config_snapshot=config,
            providers=mock_registry,
            output_lock=threading.Lock(),
            on_complete=Mock(),
            history=TranscriptionHistory(),
        )

        # Run transcription
        chunk = {"mic1": np.random.randn(1000).astype(np.float32)}
        session._transcribe_chunk_with_consensus(chunk)

        # Should have results without consensus
        assert len(session.chunk_results) == 1
        results, consensus = session.chunk_results[0]
        assert consensus is None
        assert len(results) == 2


class TestSessionOutput:
    """Tests for session output handling."""

    def test_output_checks_window(self):
        """Test that output verifies window hasn't changed."""
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import ConfigSnapshot, AppContext
        from uuid import uuid4

        config = Mock(spec=ConfigSnapshot)
        session = Session(
            id=uuid4(),
            config_snapshot=config,
            providers=Mock(),
            output_lock=threading.Lock(),
            on_complete=Mock(),
            history=TranscriptionHistory(),
        )

        # Set initial context
        session.context = AppContext(
            app_name="App1",
            window_title="Window1",
            bundle_id="com.app1",
            rigor_level="normal"
        )

        with patch('mergescribe.session.get_app_context') as mock_ctx:
            with patch('mergescribe.session.type_text') as mock_type:
                # Same window
                mock_ctx.return_value = AppContext(
                    app_name="App1",
                    window_title="Window1",
                    bundle_id="com.app1",
                    rigor_level="normal"
                )

                session._output("Hello")

                mock_type.assert_called_once_with("Hello")

    def test_output_copies_clipboard_on_window_change(self):
        """Test that output copies to clipboard if window changed."""
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import ConfigSnapshot, AppContext
        from uuid import uuid4

        config = Mock(spec=ConfigSnapshot)
        session = Session(
            id=uuid4(),
            config_snapshot=config,
            providers=Mock(),
            output_lock=threading.Lock(),
            on_complete=Mock(),
            history=TranscriptionHistory(),
        )

        # Set initial context
        session.context = AppContext(
            app_name="App1",
            window_title="Window1",
            bundle_id="com.app1",
            rigor_level="normal"
        )

        with patch('mergescribe.session.get_app_context') as mock_ctx:
            with patch('mergescribe.session.copy_to_clipboard') as mock_copy:
                with patch('mergescribe.session.notify') as mock_notify:
                    with patch('mergescribe.session.type_text') as mock_type:
                        # Different window
                        mock_ctx.return_value = AppContext(
                            app_name="App2",
                            window_title="Window2",
                            bundle_id="com.app2",  # Different bundle_id
                            rigor_level="normal"
                        )

                        session._output("Hello")

                        mock_copy.assert_called_once_with("Hello")
                        mock_notify.assert_called_once()
                        mock_type.assert_not_called()


class TestSessionFinalization:
    """Tests for session finalization."""

    def test_finalize_calls_complete_callback(self):
        """Test that finalization calls the completion callback."""
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import ConfigSnapshot, TranscriptionResult, AppContext
        from uuid import uuid4

        config = Mock(spec=ConfigSnapshot)
        config.cache_enabled = False

        on_complete = Mock()

        session = Session(
            id=uuid4(),
            config_snapshot=config,
            providers=Mock(),
            output_lock=threading.Lock(),
            on_complete=on_complete,
            history=TranscriptionHistory(),
        )

        session.context = AppContext(
            app_name="App",
            window_title="Window",
            bundle_id="com.app",
            rigor_level="normal"
        )

        # Add some results
        session.chunk_results = [
            ([TranscriptionResult(text="Hello", provider="p1", mic="m1", latency_ms=100)], "Hello"),
        ]

        with patch('mergescribe.session.get_app_context') as mock_ctx:
            with patch('mergescribe.session.type_text'):
                mock_ctx.return_value = session.context

                # Run finalization
                session._finalize_impl({})

                # Callback should be called
                on_complete.assert_called_once_with(session)

    def test_fast_path_single_chunk_consensus(self):
        """Test fast path when single chunk has consensus."""
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import ConfigSnapshot, TranscriptionResult, AppContext
        from uuid import uuid4

        config = Mock(spec=ConfigSnapshot)
        config.cache_enabled = False

        session = Session(
            id=uuid4(),
            config_snapshot=config,
            providers=Mock(),
            output_lock=threading.Lock(),
            on_complete=Mock(),
            history=TranscriptionHistory(),
        )

        session.context = AppContext(
            app_name="App",
            window_title="Window",
            bundle_id="com.app",
            rigor_level="normal"
        )

        # Single chunk with consensus
        session.chunk_results = [
            ([TranscriptionResult(text="Hello world", provider="p1", mic="m1", latency_ms=100)], "Hello world"),
        ]

        with patch('mergescribe.session.get_app_context') as mock_ctx:
            with patch('mergescribe.session.type_text') as mock_type:
                with patch.object(session, '_output') as mock_output:
                    mock_ctx.return_value = session.context

                    # Run finalization
                    session._finalize_impl({})

                    # Should use fast path - output consensus directly
                    mock_output.assert_called_once_with("Hello world")
