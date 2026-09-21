"""
Tests for mergescribe Session and SessionManager.

Tests the recording session lifecycle, chunk handling, and coordination.
"""

import time
import numpy as np
from unittest.mock import Mock, patch
import threading


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
        config.enabled_mics = ["mic1"]
        config.provider_deadline_multiplier = 1.0
        config.provider_deadline_min_ms = 900
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
                bundle_id="com.test.app"
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
        config.enabled_mics = ["mic1"]
        config.provider_deadline_multiplier = 1.0
        config.provider_deadline_min_ms = 900
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
        config.enabled_mics = ["mic1"]
        config.provider_deadline_multiplier = 1.0
        config.provider_deadline_min_ms = 900
        config.cache_enabled = False

        manager = SessionManager(
            config_snapshot_fn=lambda: config,
            providers=Mock(),
        )

        with patch('mergescribe.session.play_busy_sound') as mock_sound:
            # Start first session
            session1 = manager.start_session()
            session1.is_capturing = True  # Simulate a session holding the mic

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
        session.is_capturing = True

        assert manager.is_busy() is True

    def test_bookkeeping_failure_does_not_strand_the_session(self):
        """A throw in metrics/training must still release the session, or the app goes deaf."""
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import ConfigSnapshot
        from uuid import uuid4

        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 2
        config.consensus_max_words = 15
        config.enabled_mics = ["mic1"]
        config.provider_deadline_multiplier = 1.0
        config.provider_deadline_min_ms = 900
        config.sample_rate = 16000
        config.training_enabled = False
        config.edit_feedback_enabled = False

        metrics = Mock()
        metrics.log = Mock(side_effect=RuntimeError("disk full"))

        completed = []
        session = Session(
            id=uuid4(),
            config_snapshot=config,
            providers=Mock(**{"values.return_value": []}),
            output_lock=threading.Lock(),
            on_complete=completed.append,
            history=TranscriptionHistory(),
            metrics=metrics,
        )
        session.is_active = True
        session.is_capturing = True
        session.start_time = time.time()

        # Empty chunk: nothing to transcribe, so this exercises the teardown path
        session._finalize_impl({"mic1": np.array([], dtype=np.float32)})

        assert session.is_active is False, "session left active after bookkeeping threw"
        assert session.is_capturing is False
        assert completed == [session], "on_complete never fired; manager would stay wedged"

    def test_finalizing_session_does_not_block_a_new_recording(self):
        """The mic is free at key release; the ~2s LLM+typing tail must not deafen the app."""
        from mergescribe.session import SessionManager
        from mergescribe.types import ConfigSnapshot

        config = Mock(spec=ConfigSnapshot)
        config.enabled_mics = ["mic1"]
        manager = SessionManager(config_snapshot_fn=lambda: config, providers=Mock())

        first = manager.start_session()
        first.is_capturing = True
        first.is_active = True
        assert manager.is_busy() is True

        # Key released: still correcting and typing, but no longer holding the mic
        first.is_capturing = False
        assert first.is_active is True
        assert manager.is_busy() is False

        with patch('mergescribe.session.play_busy_sound') as mock_sound:
            second = manager.start_session()
        assert second is not None, "new recording was rejected while the previous only finalized"
        assert second is not first
        mock_sound.assert_not_called()

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

    def _deadline_session(self, multiplier, min_ms, slow_seconds, observer=None, run=True):
        """A fast provider and a straggler, wired for the deadline tests."""
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import ConfigSnapshot, TranscriptionResult
        from uuid import uuid4

        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 99      # never short-circuit on consensus
        config.consensus_max_words = 15
        config.enabled_mics = ["mic1"]
        config.provider_deadline_multiplier = multiplier
        config.provider_deadline_min_ms = min_ms

        def make(name, delay):
            p = Mock()
            p.name = name
            p.single_instance = False

            def transcribe(audio, mic, _n=name, _d=delay):
                time.sleep(_d)
                return TranscriptionResult(text=_n, provider=_n, mic=mic, latency_ms=int(_d * 1000))
            p.transcribe = transcribe
            return p

        registry = Mock()
        registry.values = Mock(return_value=[make("fast", 0.0), make("slow", slow_seconds)])
        session = Session(
            id=uuid4(),
            config_snapshot=config,
            providers=registry,
            output_lock=threading.Lock(),
            on_complete=Mock(),
            history=TranscriptionHistory(),
            observer=observer,
        )
        if not run:
            return session, 0.0
        chunk = {"mic1": np.random.randn(1000).astype(np.float32)}
        started = time.monotonic()
        session._transcribe_chunk_with_consensus(chunk)
        return session, time.monotonic() - started

    def test_start_announces_the_streams_that_will_run(self):
        observer = Mock()
        session, _ = self._deadline_session(1.0, 120, 0.0, observer=observer, run=False)
        session.config_snapshot.field_routing_enabled = False
        with patch.object(type(session), "_capture_context"):
            session.start(mics=["mic1"])
        observer.streams_planned.assert_called_once_with(
            str(session.id), ["fast/mic1", "slow/mic1"])

    def test_observer_sees_streams_land_and_the_straggler_dropped(self):
        observer = Mock()
        session, _ = self._deadline_session(1.0, 120, 3.0, observer=observer)
        sid = str(session.id)
        observer.stream_landed.assert_called_once_with(sid, "fast/mic1", False)
        observer.stream_dropped.assert_called_once_with(sid, "slow/mic1")

    def test_the_final_chunk_is_marked_and_followed_by_done(self):
        observer = Mock()
        session, _ = self._deadline_session(1.0, 1500, 0.05, observer=observer, run=False)
        session._transcribe_remaining({"mic1": np.random.randn(1000).astype(np.float32)})
        sid = str(session.id)
        events = [(c[0], c[1]) for c in observer.method_calls]
        assert ("stream_landed", (sid, "fast/mic1", True)) in events
        assert ("stream_landed", (sid, "slow/mic1", True)) in events
        assert events[-1] == ("transcription_done", (sid,))

    def test_a_broken_observer_never_breaks_transcription(self):
        observer = Mock()
        observer.stream_landed.side_effect = RuntimeError("HUD gone")
        session, _ = self._deadline_session(0.0, 120, 0.0, observer=observer)
        results, _ = session.chunk_results[0]
        assert sorted(r.provider for r in results) == ["fast", "slow"]

    def test_deadline_abandons_a_straggler(self):
        """The chunk should not pay for the slowest provider's tail."""
        session, elapsed = self._deadline_session(multiplier=1.0, min_ms=120, slow_seconds=3.0)

        results, _ = session.chunk_results[0]
        assert [r.provider for r in results] == ["fast"]
        assert elapsed < 1.0, f"waited {elapsed:.2f}s for an abandoned provider"

    def test_deadline_disabled_waits_for_everyone(self):
        """multiplier 0 must restore the old always-wait behaviour."""
        session, _ = self._deadline_session(multiplier=0.0, min_ms=120, slow_seconds=0.35)

        results, _ = session.chunk_results[0]
        assert sorted(r.provider for r in results) == ["fast", "slow"]

    def test_min_grace_protects_a_normally_slow_provider(self):
        """A provider inside the grace window is kept even though it is slowest."""
        session, _ = self._deadline_session(multiplier=1.0, min_ms=1500, slow_seconds=0.3)

        results, _ = session.chunk_results[0]
        assert sorted(r.provider for r in results) == ["fast", "slow"]

    def test_deadline_never_fires_before_any_result(self):
        """If everything is slow there is nothing to compare against; wait."""
        session, _ = self._deadline_session(multiplier=1.0, min_ms=50, slow_seconds=0.0)

        results, _ = session.chunk_results[0]
        assert len(results) == 2

    def test_serialized_provider_runs_on_primary_mic_only(self):
        """A locked provider must not be queued onto every mic: it lands on the critical path."""
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import ConfigSnapshot, TranscriptionResult
        from uuid import uuid4

        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 2
        config.consensus_max_words = 15
        config.enabled_mics = ["mic_a", "mic_b"]
        config.provider_deadline_multiplier = 1.0
        config.provider_deadline_min_ms = 900

        calls = []

        def make(name, single):
            p = Mock()
            p.name = name
            p.single_instance = single
            p.transcribe = lambda audio, mic, _n=name: (
                calls.append((_n, mic)),
                TranscriptionResult(text="hi", provider=_n, mic=mic, latency_ms=1),
            )[1]
            return p

        registry = Mock()
        registry.values = Mock(return_value=[make("local", True), make("cloud", False)])

        session = Session(
            id=uuid4(),
            config_snapshot=config,
            providers=registry,
            output_lock=threading.Lock(),
            on_complete=Mock(),
            history=TranscriptionHistory(),
        )
        chunk = {
            "mic_a": np.random.randn(1000).astype(np.float32),
            "mic_b": np.random.randn(1000).astype(np.float32),
        }
        session._transcribe_chunk_with_consensus(chunk)

        assert ("local", "mic_a") in calls
        assert ("local", "mic_b") not in calls
        assert len([c for c in calls if c[0] == "cloud"]) == 2

    def test_primary_mic_follows_enabled_mics_order(self):
        """Reordering enabled_mics chooses which mic the serialized provider gets."""
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import ConfigSnapshot
        from uuid import uuid4

        config = Mock(spec=ConfigSnapshot)
        config.enabled_mics = ["mic_b", "mic_a"]
        local = Mock(single_instance=True)
        local.name = "local"
        cloud = Mock(single_instance=False)
        cloud.name = "cloud"
        providers = Mock()
        providers.values.return_value = [local, cloud]
        session = Session(
            id=uuid4(),
            config_snapshot=config,
            providers=providers,
            output_lock=threading.Lock(),
            on_complete=Mock(),
            history=TranscriptionHistory(),
        )

        def plan(mics):
            return sorted(f"{p.name}/{m}" for m, p in session.stream_plan(mics))

        assert plan(["mic_a", "mic_b"]) == ["cloud/mic_a", "cloud/mic_b", "local/mic_b"]
        # An unplugged primary falls through to whatever produced audio
        assert plan(["mic_a"]) == ["cloud/mic_a", "local/mic_a"]

    def test_transcription_with_consensus(self):
        """Test that consensus is detected correctly."""
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import ConfigSnapshot, TranscriptionResult
        from uuid import uuid4

        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 2
        config.consensus_max_words = 15
        config.enabled_mics = ["mic1"]
        config.provider_deadline_multiplier = 1.0
        config.provider_deadline_min_ms = 900

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
        config.enabled_mics = ["mic1"]
        config.provider_deadline_multiplier = 1.0
        config.provider_deadline_min_ms = 900

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
            bundle_id="com.app1"
        )

        with patch('mergescribe.session.get_app_context') as mock_ctx:
            with patch('mergescribe.session.type_text') as mock_type:
                # Same window
                mock_ctx.return_value = AppContext(
                    app_name="App1",
                    window_title="Window1",
                    bundle_id="com.app1"
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
            bundle_id="com.app1"
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
            bundle_id="com.app"
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
            bundle_id="com.app"
        )

        # Single chunk with consensus
        session.chunk_results = [
            ([TranscriptionResult(text="Hello world", provider="p1", mic="m1", latency_ms=100)], "Hello world"),
        ]

        with patch('mergescribe.session.get_app_context') as mock_ctx:
            with patch('mergescribe.session.type_text'):
                with patch.object(session, '_output') as mock_output:
                    mock_ctx.return_value = session.context

                    # Run finalization
                    session._finalize_impl({})

                    # Should use fast path - output consensus directly
                    mock_output.assert_called_once_with("Hello world")


class TestNoSpeechGuard:
    """A stray tap must not reach the transcriber."""

    def _session(self):
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import ConfigSnapshot
        from uuid import uuid4
        import threading
        cfg = Mock(spec=ConfigSnapshot)
        cfg.sample_rate = 16000
        return Session(id=uuid4(), config_snapshot=cfg, providers=Mock(),
                       output_lock=threading.Lock(), on_complete=lambda s: None,
                       history=TranscriptionHistory())

    def test_empty_chunk_rejected(self):
        assert self._session()._has_speech({}) is False

    def test_too_short_rejected(self):
        s = self._session()
        blip = (np.random.randn(1600) * 0.3).astype(np.float32)  # 0.1s, loud
        assert s._has_speech({"mic1": blip}) is False

    def test_long_but_silent_rejected(self):
        s = self._session()
        quiet = (np.random.randn(32000) * 0.0001).astype(np.float32)  # 2s of near-silence
        assert s._has_speech({"mic1": quiet}) is False

    def test_real_speech_accepted(self):
        s = self._session()
        speech = (np.random.randn(32000) * 0.1).astype(np.float32)  # 2s at a normal level
        assert s._has_speech({"mic1": speech}) is True

    def test_one_good_mic_is_enough(self):
        s = self._session()
        speech = (np.random.randn(32000) * 0.1).astype(np.float32)
        dead = np.zeros(32000, dtype=np.float32)
        assert s._has_speech({"dead": dead, "good": speech}) is True


class TestFinalizeSteps:
    """Covers the stages _finalize_impl delegates to, including the output path."""

    def _session(self, **overrides):
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import ConfigSnapshot, AppContext
        from uuid import uuid4

        config = Mock(spec=ConfigSnapshot)
        config.sample_rate = 16000
        config.custom_instructions = "be nice"
        config.training_enabled = False
        config.edit_feedback_enabled = False
        config.enabled_mics = ["mic1"]
        config.space_between_dictations = False
        for k, v in overrides.items():
            setattr(config, k, v)

        session = Session(
            id=uuid4(),
            config_snapshot=config,
            providers=Mock(**{"values.return_value": []}),
            output_lock=threading.Lock(),
            on_complete=Mock(),
            history=TranscriptionHistory(),
        )
        session.start_time = time.time()
        session.context = AppContext(app_name="Editor", window_title="t", bundle_id="com.x")
        return session

    def test_collect_final_audio_accumulates_per_mic(self):
        session = self._session()
        session.all_audio["mic1"] = [np.zeros(10, dtype=np.float32)]
        session._collect_final_audio({
            "mic1": np.ones(1600, dtype=np.float32),
            "mic2": np.ones(800, dtype=np.float32),
            "mic3": np.array([], dtype=np.float32),   # silent mic contributes nothing
        })
        assert len(session.all_audio["mic1"]) == 2
        assert len(session.all_audio["mic2"]) == 1
        assert "mic3" not in session.all_audio

    def test_collect_final_audio_copies_the_buffer(self):
        """The engine reuses its arrays; storing a reference would corrupt training data."""
        session = self._session()
        buf = np.ones(100, dtype=np.float32)
        session._collect_final_audio({"mic1": buf})
        buf[:] = 0.0
        assert session.all_audio["mic1"][0].sum() == 100

    def test_history_context_stays_empty(self):
        """Regression guard: feeding prior dictations back made the model echo them."""
        session = self._session()
        assert session._correction_kwargs()["history_context"] == ""

    def test_stream_correction_types_tokens_and_returns_them(self):
        from mergescribe.types import TranscriptionResult

        session = self._session()
        session.field_targets = None          # routing off: no TARGET line to strip
        typed = []

        def fake_correct(results, context, config, on_delta=None, **kw):
            for token in ("Hello ", "world", "!"):
                on_delta(token)
            return "Hello world!"

        with patch("mergescribe.correct.correct_with_llm", fake_correct), \
             patch("mergescribe.session.type_text", side_effect=typed.append):
            out = session._stream_correction(
                [TranscriptionResult(text="hello world", provider="p", mic="m", latency_ms=1)])

        assert out == "Hello world!"
        assert "".join(typed) == "Hello world!"
        assert session.output_method == "streamed"

    def test_invented_line_breaks_are_flattened(self):
        """A typed newline is a Return keystroke, which sends the message in chat apps."""
        from mergescribe.types import TranscriptionResult

        session = self._session()
        session.field_targets = None
        typed = []

        def fake_correct(results, context, config, on_delta=None, **kw):
            for token in ("Okay cool.", "\n\n", "Also, one more thing."):
                on_delta(token)
            return ""

        with patch("mergescribe.correct.correct_with_llm", fake_correct), \
             patch("mergescribe.session.type_text", side_effect=typed.append):
            out = session._stream_correction(
                [TranscriptionResult(text="x", provider="p", mic="m", latency_ms=1)])

        assert "\n" not in out
        assert out == "Okay cool. Also, one more thing."
        assert not any("\n" in t for t in typed), "a newline reached the keyboard"

    def test_continuing_a_recent_dictation_leads_with_a_space(self):
        """Back-to-back recordings otherwise run their words together."""
        from mergescribe.types import TranscriptionResult

        session = self._session(space_between_dictations=True)
        session.field_targets = None
        session.history.add("earlier words", destination=session._output_destination())

        with patch("mergescribe.correct.correct_with_llm",
                   lambda *a, on_delta=None, **k: on_delta("Ship it.") or ""), \
             patch("mergescribe.session.type_text"):
            out = session._stream_correction(
                [TranscriptionResult(text="x", provider="p", mic="m", latency_ms=1)])
        assert out == " Ship it."

    def test_a_fresh_dictation_has_no_stray_space(self):
        """Nothing to append to, so the text should start clean and end clean."""
        from mergescribe.types import TranscriptionResult

        session = self._session(space_between_dictations=True)
        session.field_targets = None   # history empty: not a continuation

        with patch("mergescribe.correct.correct_with_llm",
                   lambda *a, on_delta=None, **k: on_delta("Ship it.") or ""), \
             patch("mergescribe.session.type_text"):
            out = session._stream_correction(
                [TranscriptionResult(text="x", provider="p", mic="m", latency_ms=1)])
        assert out == "Ship it."

    def test_a_stale_dictation_is_not_continued(self):
        """A recording minutes later is a new thought, not a continuation."""
        session = self._session(space_between_dictations=True)
        session.history.add("long ago", destination=session._output_destination())
        # Backdate the entry past the continuation window
        ts, text, dest = session.history._entries[-1]
        session.history._entries[-1] = (ts - 600, text, dest)
        assert session._continues_previous() is False

    def test_a_different_destination_is_not_continued(self):
        session = self._session(space_between_dictations=True)
        session.history.add("elsewhere", destination="Some Other App")
        assert session._continues_previous() is False

    def test_a_failed_correction_leaves_no_stray_space(self):
        """The separator is typed with the first word, so a failure types nothing at all."""
        from mergescribe.types import TranscriptionResult

        session = self._session(space_between_dictations=True)
        session.history.add("earlier", destination=session._output_destination())
        session.field_targets = None
        assert session._continues_previous()
        with patch("mergescribe.correct.correct_with_llm",
                   lambda *a, on_delta=None, **k: ""), \
             patch("mergescribe.session.type_text") as typer:
            assert session._stream_correction(
                [TranscriptionResult(text="x", provider="p", mic="m", latency_ms=1)]) == ""
        typer.assert_not_called()

    def test_stream_correction_reports_failure_as_empty(self):
        """A failed correction must be distinguishable so the raw transcript can be used."""
        from mergescribe.types import TranscriptionResult

        session = self._session()
        session.field_targets = None
        with patch("mergescribe.correct.correct_with_llm",
                   lambda *a, on_delta=None, **k: ""), \
             patch("mergescribe.session.type_text"):
            assert session._stream_correction(
                [TranscriptionResult(text="x", provider="p", mic="m", latency_ms=1)]) == ""

    def test_output_goes_to_clipboard_when_the_window_changed(self):
        from mergescribe.types import TranscriptionResult, AppContext

        session = self._session()
        session.field_targets = None
        elsewhere = AppContext(app_name="Other", window_title="w", bundle_id="com.other")

        with patch("mergescribe.session.get_app_context", return_value=elsewhere), \
             patch("mergescribe.correct.correct_with_llm", lambda *a, **k: "corrected text"), \
             patch("mergescribe.session.copy_to_clipboard") as clip, \
             patch("mergescribe.session.notify"), \
             patch("mergescribe.session.type_text") as typer:
            session._correct_and_output(
                [TranscriptionResult(text="raw", provider="p", mic="m", latency_ms=1)], "raw")

        clip.assert_called_once_with("corrected text")
        typer.assert_not_called()
        assert session.output_method == "clipboard"

    def test_history_records_the_raw_transcript_not_the_correction(self):
        """Storing corrected text would feed the model its own phrasing."""
        from mergescribe.types import TranscriptionResult, AppContext

        session = self._session()
        session.field_targets = None
        elsewhere = AppContext(app_name="Other", window_title="w", bundle_id="com.other")

        with patch("mergescribe.session.get_app_context", return_value=elsewhere), \
             patch("mergescribe.correct.correct_with_llm", lambda *a, **k: "Polished output."), \
             patch("mergescribe.session.copy_to_clipboard"), \
             patch("mergescribe.session.notify"):
            session._correct_and_output(
                [TranscriptionResult(text="um raw", provider="p", mic="m", latency_ms=1)],
                "um raw")

        recent = " ".join(text for _, text, _ in session.history._entries)
        assert "um raw" in recent
        assert "Polished" not in recent


class TestInFlightSpeech:
    """The discard for stray taps must never swallow speech that is mid-transcription."""

    def _session(self, delay=0.3):
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import ConfigSnapshot, TranscriptionResult
        from uuid import uuid4

        config = Mock(spec=ConfigSnapshot)
        config.sample_rate = 16000
        config.consensus_threshold = 2
        config.consensus_max_words = 15
        config.enabled_mics = ["mic1"]
        config.provider_deadline_multiplier = 1.0
        config.provider_deadline_min_ms = 2000
        config.training_enabled = False
        config.edit_feedback_enabled = False

        calls = []

        def transcribe(audio, mic):
            calls.append(len(audio))
            time.sleep(delay)
            return TranscriptionResult(text="ship the fix today", provider="p", mic=mic, latency_ms=1)

        provider = Mock()
        provider.name = "p"
        provider.single_instance = False
        provider.transcribe = transcribe
        session = Session(
            id=uuid4(), config_snapshot=config,
            providers=Mock(**{"values.return_value": [provider]}),
            output_lock=threading.Lock(), on_complete=Mock(),
            history=TranscriptionHistory(), metrics=Mock(),
        )
        session.start_time = time.time()
        return session, calls

    @staticmethod
    def _speech(seconds=1.0):
        return {"mic1": (np.random.randn(int(16000 * seconds)) * 0.3).astype(np.float32)}

    @staticmethod
    def _silence(seconds=0.4):
        return {"mic1": np.zeros(int(16000 * seconds), dtype=np.float32)}

    def _discarded(self, session):
        return any(c.args and c.args[0] == "session_discarded" for c in session.metrics.log.call_args_list)

    def test_speech_still_being_transcribed_survives_a_quiet_release(self):
        """The exact log sequence: 30s chunk emitted, release 0.4s later on silence."""
        from mergescribe.session import Session

        session, _ = self._session(delay=0.3)
        session.on_chunk_ready(self._speech())          # in flight for 0.3s
        with patch.object(Session, "_correct_and_output") as correct:
            session._finalize_impl(self._silence())     # released immediately

        assert not self._discarded(session), "in-flight speech was thrown away"
        correct.assert_called_once()
        assert "ship the fix today" in correct.call_args.args[1]

    def test_the_silent_tail_is_not_sent_to_the_recognizers(self):
        """Recognizers invent words on silence; only the real speech should go out."""
        from mergescribe.session import Session

        session, calls = self._session(delay=0.0)
        session.on_chunk_ready(self._speech())
        with patch.object(Session, "_correct_and_output"):
            session._finalize_impl(self._silence())
        assert len(calls) == 1

    def test_a_stray_tap_is_still_discarded(self):
        from mergescribe.session import Session

        session, calls = self._session()
        with patch.object(Session, "_correct_and_output") as correct:
            session._finalize_impl(self._silence(0.2))
        assert self._discarded(session)
        correct.assert_not_called()
        assert calls == []


class TestCalledOff:
    """The model says "type nothing" with a marker; an empty reply still means failure."""

    def _run(self, tokens, *, same_window=True, continuing=False):
        from mergescribe.session import Session
        from mergescribe.types import AppContext, TranscriptionResult
        helper = TestFinalizeSteps()
        session = helper._session(space_between_dictations=continuing)
        session.metrics = Mock()
        session.field_targets = None
        if continuing:
            session.history.add("earlier", destination=session._output_destination())
        here = session.context
        elsewhere = AppContext(app_name="Other", window_title="w", bundle_id="com.other")

        def fake_correct(results, context, config, on_delta=None, **kw):
            for token in tokens:
                if on_delta:
                    on_delta(token)
            return "".join(tokens)

        with patch("mergescribe.correct.correct_with_llm", fake_correct), \
             patch("mergescribe.session.get_app_context",
                   return_value=here if same_window else elsewhere), \
             patch("mergescribe.session.type_text") as typer, \
             patch("mergescribe.session.copy_to_clipboard") as clip, \
             patch("mergescribe.session.notify"), \
             patch.object(Session, "_output") as fallback:
            session._correct_and_output(
                [TranscriptionResult(text="raw words", provider="p", mic="m", latency_ms=1)],
                "raw words")
        return session, typer, clip, fallback

    def _called_off(self, session):
        return any(c.args and c.args[0] == "session_discarded" and c.kwargs.get("reason") == "called_off"
                   for c in session.metrics.log.call_args_list)

    def test_the_marker_types_nothing_and_does_not_fall_back(self):
        """Falling back would type the raw transcript, "scratch all that" included."""
        session, typer, _, fallback = self._run(["[no", "thing]"])
        typer.assert_not_called()
        fallback.assert_not_called()
        assert self._called_off(session)

    def test_called_off_leaves_no_stray_separator(self):
        session, typer, _, _ = self._run(["[nothing]"], continuing=True)
        typer.assert_not_called()

    def test_the_clipboard_path_honours_the_marker(self):
        session, _, clip, _ = self._run(["[nothing]"], same_window=False)
        clip.assert_not_called()
        assert self._called_off(session)

    def test_bracketed_text_that_is_not_the_marker_still_types(self):
        session, typer, _, _ = self._run(["[WIP] ", "ship it"])
        assert "".join(c.args[0] for c in typer.call_args_list) == "[WIP] ship it"
        assert not self._called_off(session)
