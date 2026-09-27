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
        # Neither may touch the real desktop: no synthetic Cmd+C, no real window.
        with patch('mergescribe.session.get_app_context') as mock_ctx, \
                patch('mergescribe.session.detect_selected_text', return_value=None), \
                patch('mergescribe.session.capture_origin', return_value=None):
            from mergescribe.types import AppContext
            mock_ctx.return_value = AppContext(
                app_name="Test App",
                window_title="Test Window",
                bundle_id="com.test.app"
            )

            session = self.create_session()
            session.start()

            assert session.is_capturing is True
            assert session.start_time > 0
            assert session.context.app_name == "Test App"

    def test_session_aggregation_uses_consensus(self):
        """Test result aggregation prefers consensus."""
        from mergescribe.types import TranscriptionResult

        session = self.create_session()

        # Add chunk results - first has consensus, second doesn't
        session.chunk_results = [
            # Chunk 1: has consensus
            (1, [
                TranscriptionResult(text="Hello world", provider="p1", mic="m1", latency_ms=100),
                TranscriptionResult(text="Hello world", provider="p2", mic="m1", latency_ms=100),
            ], "Hello world"),
            # Chunk 2: no consensus
            (2, [
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

class TestSessionManager:
    """Tests for SessionManager class."""

    def test_start_session_creates_session(self):
        """Test that start_session creates and returns a session."""
        from mergescribe.session import SessionManager
        from mergescribe.types import ConfigSnapshot

        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 2
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
        session.is_capturing = True
        session.start_time = time.time()

        # Empty chunk: nothing to transcribe, so this exercises the teardown path
        session._finalize_impl({"mic1": np.array([], dtype=np.float32)})

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
        assert manager.is_busy() is True

        # Key released: still correcting and typing, but no longer holding the mic
        first.is_capturing = False
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
        _, results, _ = session.chunk_results[0]
        assert sorted(r.provider for r in results) == ["fast", "slow"]

    def test_deadline_abandons_a_straggler(self):
        """The chunk should not pay for the slowest provider's tail."""
        session, elapsed = self._deadline_session(multiplier=1.0, min_ms=120, slow_seconds=3.0)

        _, results, _ = session.chunk_results[0]
        assert [r.provider for r in results] == ["fast"]
        assert elapsed < 1.0, f"waited {elapsed:.2f}s for an abandoned provider"

    def test_deadline_disabled_waits_for_everyone(self):
        """multiplier 0 must restore the old always-wait behaviour."""
        session, _ = self._deadline_session(multiplier=0.0, min_ms=120, slow_seconds=0.35)

        _, results, _ = session.chunk_results[0]
        assert sorted(r.provider for r in results) == ["fast", "slow"]

    def test_min_grace_protects_a_normally_slow_provider(self):
        """A provider inside the grace window is kept even though it is slowest."""
        session, _ = self._deadline_session(multiplier=1.0, min_ms=1500, slow_seconds=0.3)

        _, results, _ = session.chunk_results[0]
        assert sorted(r.provider for r in results) == ["fast", "slow"]

    def test_deadline_never_fires_before_any_result(self):
        """If everything is slow there is nothing to compare against; wait."""
        session, _ = self._deadline_session(multiplier=1.0, min_ms=50, slow_seconds=0.0)

        _, results, _ = session.chunk_results[0]
        assert len(results) == 2

    def test_serialized_provider_runs_on_primary_mic_only(self):
        """A locked provider must not be queued onto every mic: it lands on the critical path."""
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import ConfigSnapshot, TranscriptionResult
        from uuid import uuid4

        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 2
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
        _, results, consensus = session.chunk_results[0]
        assert consensus == "Hello world"

    def test_transcription_without_consensus(self):
        """Test handling when no consensus is reached."""
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import ConfigSnapshot, TranscriptionResult
        from uuid import uuid4

        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 2
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
        _, results, consensus = session.chunk_results[0]
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


class TestBackToWhereItStarted:
    """Dictating while reading something else: the words go where the dictation started."""

    def _session(self):
        from mergescribe.context import Origin
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import AppContext, ConfigSnapshot
        from uuid import uuid4

        config = Mock(spec=ConfigSnapshot)
        config.edit_feedback_enabled = False
        session = Session(id=uuid4(), config_snapshot=config, providers=Mock(),
                          output_lock=threading.Lock(), on_complete=Mock(),
                          history=TranscriptionHistory())
        session.start_time = session.finalize_start_time = time.time()
        session.context = AppContext(app_name="Claude", window_title="chat", bundle_id="com.anthropic")
        session.origin = Origin(pid=1, app=object(), window=object())
        return session

    def _output(self, here, comes_back):
        session = self._session()
        with patch("mergescribe.session.at_origin", return_value=here), \
             patch("mergescribe.session.return_to", return_value=comes_back) as back, \
             patch("mergescribe.session.type_text") as typer, \
             patch("mergescribe.session.copy_to_clipboard") as clip, \
             patch("mergescribe.session.notify"):
            session._output("Ship it Friday.")
        return typer, clip, back

    def test_still_there_it_just_types(self):
        typer, clip, back = self._output(here=True, comes_back=True)
        typer.assert_called_once_with("Ship it Friday.")
        back.assert_not_called()
        clip.assert_not_called()

    def test_moved_on_it_brings_the_window_back_and_types_there(self):
        typer, clip, back = self._output(here=False, comes_back=True)
        back.assert_called_once()
        typer.assert_called_once_with("Ship it Friday.")
        clip.assert_not_called()

    def test_a_window_that_cannot_come_back_gets_the_clipboard(self):
        """Closed since: nothing to type into, so nothing is typed anywhere else."""
        typer, clip, back = self._output(here=False, comes_back=False)
        typer.assert_not_called()
        clip.assert_called_once_with("Ship it Friday.")

    def test_the_correction_streams_into_the_window_it_brought_back(self):
        from mergescribe.types import TranscriptionResult

        session = self._session()
        with patch("mergescribe.session.at_origin", return_value=False), \
             patch("mergescribe.session.return_to", return_value=True), \
             patch.object(type(session), "_stream_correction", return_value="corrected") as streamed, \
             patch.object(type(session), "_clipboard_correction") as clipboard:
            session._correct_and_output(
                [TranscriptionResult(text="raw", provider="p", mic="m", latency_ms=1)], "raw")
        streamed.assert_called_once()
        clipboard.assert_not_called()


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
            (1, [TranscriptionResult(text="Hello", provider="p1", mic="m1", latency_ms=100)], "Hello"),
        ]

        with patch('mergescribe.session.get_app_context') as mock_ctx:
            with patch('mergescribe.session.type_text'):
                mock_ctx.return_value = session.context

                # Run finalization
                session._finalize_impl({})

                # Callback should be called
                on_complete.assert_called_once_with(session)

    def test_agreed_text_still_goes_through_the_judge_or_correction(self):
        """Consensus is not a licence to type: "um, ship it" can be agreed on too."""
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
            (1, [TranscriptionResult(text="Hello world", provider="p1", mic="m1", latency_ms=100)], "Hello world"),
        ]

        with patch('mergescribe.session.get_app_context') as mock_ctx, \
                patch('mergescribe.session.type_text'), \
                patch.object(session, '_output') as typed_as_is, \
                patch.object(session, '_correct_and_output') as corrected:
            mock_ctx.return_value = session.context
            session._finalize_impl({})

        # Recognizers agreeing settles the words, not whether they need tidying:
        # that is the judge's and the correction model's call.
        typed_as_is.assert_not_called()
        corrected.assert_called_once()


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

    def _edited(self, output_method, llm_model=None):
        """Run the edit watch to an "edited" outcome and return the corpus row written."""
        from mergescribe.types import LLMCorrectionResult

        session = self._session(edit_feedback_enabled=True)
        session.output_method = output_method
        if llm_model:
            session.llm_result = LLMCorrectionResult(text="t", provider="openrouter", model=llm_model,
                                                     input_tokens_est=1, latency_ms=1.0)
        rows, callbacks = [], []
        with patch("mergescribe.feedback.watch_for_edits",
                   side_effect=lambda text, cb, **kw: callbacks.append(cb)), \
                patch("mergescribe.feedback.record_correction", side_effect=rows.append):
            session._start_edit_watch("ship it friday")
            callbacks[0]("edited", 8.0, "ship it friday", "ship it Friday")
        return rows[0]

    def test_a_fixed_correction_records_the_model_that_wrote_it(self):
        row = self._edited("streamed", llm_model="openai/gpt-6-luna")
        assert row["output_method"] == "streamed" and row["correction_model"] == "openai/gpt-6-luna"

    def test_a_fixed_raw_transcript_names_no_correction_model(self):
        """Typed by the judge or consensus: whatever was wrong came from a recognizer."""
        row = self._edited("judged", llm_model="openai/gpt-6-luna")   # the correction ran but lost
        assert row["output_method"] == "judged" and row["correction_model"] == ""

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

    def test_earlier_dictations_never_reach_the_prompt(self):
        """Regression guard: feeding prior dictations back made the model echo them."""
        from mergescribe.types import TranscriptionResult

        session = self._session()
        session.config_snapshot.openrouter_api_key = "k"
        session.config_snapshot.learn_vocabulary = False
        session.history.add("the migration plan from this morning", destination="Editor")
        prompts = []
        with patch("mergescribe.correct._stream_openrouter",
                   side_effect=lambda prompt, *a, **k: (prompts.append(prompt), ("Ship it.", True))[1]):
            session._clipboard_correction(
                [TranscriptionResult(text="ship it", provider="p", mic="m", latency_ms=1)])
        assert prompts and "migration plan" not in prompts[0]

    def test_stream_correction_types_tokens_and_returns_them(self):
        from mergescribe.types import TranscriptionResult

        session = self._session()
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
        with patch("mergescribe.correct.correct_with_llm",
                   lambda *a, on_delta=None, **k: ""), \
             patch("mergescribe.session.type_text"):
            assert session._stream_correction(
                [TranscriptionResult(text="x", provider="p", mic="m", latency_ms=1)]) == ""

    def test_output_goes_to_clipboard_when_the_window_changed(self):
        from mergescribe.types import TranscriptionResult, AppContext

        session = self._session()
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


class TestPartsInOrder:
    """A long dictation arrives in parts; they must come out in the order they were spoken."""

    def _session(self, delays):
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import ConfigSnapshot, TranscriptionResult
        from uuid import uuid4

        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 99
        config.enabled_mics = ["mic1"]
        config.provider_deadline_multiplier = 1.0
        config.provider_deadline_min_ms = 2000
        config.sample_rate = 16000

        provider = Mock()
        provider.name = "p"
        provider.single_instance = False

        def transcribe(audio, mic):
            part = int(audio[0])           # each part's audio says which part it is
            time.sleep(delays[part])
            return TranscriptionResult(text=f"part {part}", provider="p", mic=mic, latency_ms=0)
        provider.transcribe = transcribe
        registry = Mock()
        registry.values = Mock(return_value=[provider])
        return Session(id=uuid4(), config_snapshot=config, providers=registry,
                       output_lock=threading.Lock(), on_complete=Mock(), history=TranscriptionHistory())

    def test_an_earlier_part_that_finishes_last_still_comes_first(self):
        """The 30s part is still transcribing when the short tail comes back."""
        session = self._session({1: 0.3, 2: 0.0})
        session.on_chunk_ready({"mic1": np.full(1600, 1.0, dtype=np.float32)})
        session._transcribe_remaining({"mic1": np.full(1600, 2.0, dtype=np.float32)})
        texts, results = session._aggregate_results()
        assert texts == ["part 1", "part 2"]
        assert [r.text for r in results] == ["part 1", "part 2"]


class TestSelectionIsKnownBeforeRouting:
    def test_a_short_command_waits_for_the_selection_before_deciding(self):
        """Selection capture (a synthetic Cmd+C) can outlast a fast transcription."""
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import ConfigSnapshot, TranscriptionResult
        from uuid import uuid4

        config = Mock(spec=ConfigSnapshot)
        config.sample_rate = 16000
        config.training_enabled = False
        config.edit_feedback_enabled = False
        session = Session(id=uuid4(), config_snapshot=config, providers=Mock(),
                          output_lock=threading.Lock(), on_complete=Mock(), history=TranscriptionHistory())
        session.start_time = time.time()

        def capture():
            time.sleep(0.3)
            session.selected_text = "the selected paragraph"
        session._context_thread = threading.Thread(target=capture)
        session._context_thread.start()

        result = TranscriptionResult(text="make this more formal", provider="p", mic="m", latency_ms=1)
        with patch.object(type(session), "_has_speech", return_value=True), \
             patch.object(type(session), "_transcribe_remaining"), \
             patch.object(type(session), "_aggregate_results", return_value=(["make this more formal"], [result])), \
             patch.object(type(session), "_run_edit_mode") as edit, \
             patch.object(type(session), "_correct_and_output") as dictate:
            session._finalize_impl({"mic1": np.ones(1600, dtype=np.float32)})
        edit.assert_called_once_with("make this more formal")
        dictate.assert_not_called()


class TestStreamingIntoTheField:
    """What happens to streamed text when the world changes under it."""

    def _session(self):
        from mergescribe.context import Origin
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import AppContext, ConfigSnapshot
        from uuid import uuid4

        config = Mock(spec=ConfigSnapshot)
        config.judge_enabled = False
        config.space_between_dictations = False
        config.custom_instructions = ""
        session = Session(id=uuid4(), config_snapshot=config, providers=Mock(),
                          output_lock=threading.Lock(), on_complete=Mock(), history=TranscriptionHistory())
        session.context = AppContext(app_name="Claude", window_title="chat", bundle_id="com.anthropic")
        session.origin = Origin(pid=1, app=object(), window=object())
        return session

    def _stream(self, session, correction, here=(True,)):
        from mergescribe.types import TranscriptionResult

        typed, clipboard = [], []
        answers = iter(list(here) + [here[-1]] * 50)
        with patch("mergescribe.correct.correct_with_llm", correction), \
             patch("mergescribe.session.at_origin", side_effect=lambda origin: next(answers)), \
             patch("mergescribe.session.type_text", side_effect=typed.append), \
             patch("mergescribe.session.copy_to_clipboard", side_effect=clipboard.append), \
             patch("mergescribe.session.notify"):
            result = session._stream_correction(
                [TranscriptionResult(text="ship it friday", provider="p", mic="m", latency_ms=1)])
        return result, "".join(typed), clipboard

    def test_moving_to_another_window_stops_typing_and_hands_over_the_rest(self):
        def correction(results, context, config, on_delta=None, **kwargs):
            for token in ("Ship ", "it ", "Friday."):
                on_delta(token)
                time.sleep(0.12)        # past the 100ms between focus checks

        session = self._session()
        result, typed, clipboard = self._stream(session, correction, here=(True, False))
        assert typed == "Ship "
        assert clipboard == ["it Friday."]
        assert session._partial_output is True

    def test_a_correction_cut_off_after_typing_puts_the_whole_text_on_the_clipboard(self):
        from mergescribe.correct import CorrectionInterrupted

        def correction(results, context, config, on_delta=None, **kwargs):
            on_delta("Ship it")
            raise CorrectionInterrupted("Ship it", "We ship Friday.")

        result, typed, clipboard = self._stream(self._session(), correction)
        assert typed == "Ship it" and result == "Ship it"
        assert clipboard == ["We ship Friday."]


class TestOutputOrder:
    def test_a_later_dictation_waits_for_the_earlier_one(self):
        from mergescribe.session import OutputOrder

        order, events = OutputOrder(), []
        first, second = order.take(), order.take()

        def later():
            order.wait(second)
            events.append("second types")
        thread = threading.Thread(target=later)
        thread.start()
        time.sleep(0.1)
        events.append("first types")
        order.done(first)
        thread.join(1.0)
        assert events == ["first types", "second types"]

    def test_finishing_out_of_order_does_not_skip_a_turn(self):
        from mergescribe.session import OutputOrder

        order = OutputOrder(patience=0.2)
        a, b, c = order.take(), order.take(), order.take()
        order.done(b)                      # b gave up waiting and finished early
        started = time.monotonic()
        order.wait(c)                      # a is still going: c must still wait
        assert time.monotonic() - started >= 0.15
        order.done(a)
        started = time.monotonic()
        order.wait(c)
        assert time.monotonic() - started < 0.05

    def test_a_session_that_fails_still_gives_up_its_turn(self):
        from mergescribe.session import OutputOrder, Session, TranscriptionHistory
        from mergescribe.types import ConfigSnapshot
        from uuid import uuid4

        order = OutputOrder(patience=5.0)
        config = Mock(spec=ConfigSnapshot)
        config.training_enabled = False
        session = Session(id=uuid4(), config_snapshot=config, providers=Mock(), output_lock=threading.Lock(),
                          on_complete=Mock(), history=TranscriptionHistory(), output_order=order)
        session.start_time = time.time()
        with patch.object(type(session), "_finalize_impl", side_effect=lambda chunk: session._teardown()):
            session.finalize({})
        time.sleep(0.1)
        started = time.monotonic()
        order.wait(order.take())
        assert time.monotonic() - started < 0.5


class TestWaitingForRecognizers:
    def _session(self, replies, min_ms, multiplier=1.0):
        from mergescribe.session import Session, TranscriptionHistory
        from mergescribe.types import ConfigSnapshot, TranscriptionResult
        from uuid import uuid4

        config = Mock(spec=ConfigSnapshot)
        config.consensus_threshold = 99
        config.enabled_mics = ["mic1"]
        config.provider_deadline_multiplier = multiplier
        config.provider_deadline_min_ms = min_ms

        def make(name, delay, text):
            p = Mock()
            p.name, p.single_instance = name, False
            p.transcribe = lambda audio, mic: (time.sleep(delay),
                                               TranscriptionResult(text=text, provider=name, mic=mic, latency_ms=0))[1]
            return p
        registry = Mock()
        registry.values = Mock(return_value=[make(*r) for r in replies])
        return Session(id=uuid4(), config_snapshot=config, providers=registry, output_lock=threading.Lock(),
                       on_complete=Mock(), history=TranscriptionHistory())

    def test_an_empty_answer_does_not_start_the_clock_on_the_others(self):
        """A recognizer that heard nothing is not evidence the rest are late."""
        session = self._session([("blank", 0.0, ""), ("slow", 0.6, "ship it friday")], min_ms=300)
        session._transcribe_chunk_with_consensus({"mic1": np.ones(1600, dtype=np.float32)})
        (_, results, _), = session.chunk_results
        assert "ship it friday" in [r.text for r in results]

    def test_the_hard_limit_holds_after_a_first_answer(self):
        import mergescribe.session as session_module

        session = self._session([("fast", 0.2, "ship it"), ("slow", 1.5, "ship it friday")], min_ms=0, multiplier=10.0)
        with patch.object(session_module, "_CHUNK_HARD_TIMEOUT", 0.5):
            started = time.monotonic()
            session._transcribe_chunk_with_consensus({"mic1": np.ones(1600, dtype=np.float32)})
        assert time.monotonic() - started < 1.0
