"""
Tests for the recording HUD.

Only the pure logic and the failure behaviour are covered — the AppKit panel
itself needs a GUI session. What matters here is that the HUD can never take
the app down with it, since it is pure decoration over the recording path.
"""

from unittest.mock import Mock, patch


class TestLevelNormalization:
    def test_maps_relative_to_the_noise_floor(self):
        """A hot mic and a quiet one should read the same on the meter."""
        from mergescribe.ui.hud import normalize_level

        quiet = normalize_level(-35.0, -50.0)   # 15dB over a quiet floor
        loud = normalize_level(-5.0, -20.0)     # 15dB over a loud floor
        assert quiet == loud

    def test_clamps_to_unit_range(self):
        from mergescribe.ui.hud import normalize_level

        assert normalize_level(-200.0, -50.0) == 0.0
        assert normalize_level(50.0, -50.0) == 1.0
        assert 0.0 <= normalize_level(-40.0, -50.0) <= 1.0

    def test_survives_non_finite_input(self):
        """log10 of digital silence is -inf; it must not reach AppKit as NaN."""
        from mergescribe.ui.hud import normalize_level

        assert normalize_level(float("-inf"), -50.0) == 0.0
        assert normalize_level(float("nan"), None) == 0.0

    def test_falls_back_before_a_noise_floor_exists(self):
        from mergescribe.ui.hud import normalize_level

        assert normalize_level(-60.0, None) == 0.0
        assert normalize_level(-30.0, None) == 0.5


class TestMeterBallistics:
    def test_attack_is_fast_and_release_is_slow(self):
        """A meter that tracks RMS symmetrically reads as noise, not a voice."""
        from mergescribe.ui.hud import smooth_level

        rise = smooth_level(1.0, 0.0)
        fall = smooth_level(0.0, 1.0)
        assert rise > 0.5, "should jump most of the way to a peak in one frame"
        assert fall > 0.5, "should still be high one frame after silence"
        assert rise > (1.0 - fall), "attack must outpace release"

    def test_converges_on_a_held_level(self):
        from mergescribe.ui.hud import smooth_level

        v = 0.0
        for _ in range(40):
            v = smooth_level(0.7, v)
        assert abs(v - 0.7) < 0.01

    def test_stays_in_range(self):
        from mergescribe.ui.hud import smooth_level

        v = 0.5
        for target in (0.0, 1.0, 0.3, 1.0, 0.0):
            for _ in range(10):
                v = smooth_level(target, v)
                assert 0.0 <= v <= 1.0


class TestSprings:
    def test_settles_on_its_target(self):
        from mergescribe.ui.hud import Spring

        spring = Spring(0.0, (190.0, 17.0))
        spring.target = 1.0
        for _ in range(120):
            spring.step(1 / 60)
        assert abs(spring.value - 1.0) < 0.01
        assert abs(spring.velocity) < 0.05

    def test_an_underdamped_entrance_overshoots_a_little(self):
        """The overshoot is the "pop"; more than a few percent reads as wobble."""
        from mergescribe.ui.hud import Spring, _PRESENCE_SPRING

        spring = Spring(0.0, _PRESENCE_SPRING)
        spring.target = 1.0
        peak = 0.0
        for _ in range(120):
            spring.step(1 / 60)
            peak = max(peak, spring.value)
        assert 1.0 < peak < 1.12

    def test_retargeting_keeps_momentum(self):
        """An interrupted animation bends toward its new goal instead of restarting."""
        from mergescribe.ui.hud import Spring

        spring = Spring(0.0, (190.0, 17.0))
        spring.target = 1.0
        for _ in range(6):
            spring.step(1 / 60)
        rising = spring.value
        spring.target = 0.0
        spring.step(1 / 60)
        assert spring.value > rising


class TestHUDAnimation:
    def _run(self, anim, seconds, level=0.0):
        for _ in range(int(seconds * 60)):
            anim.step(1 / 60, level)

    def test_appears_then_springs_away(self):
        from mergescribe.ui.hud import HUDAnimation

        anim = HUDAnimation()
        anim.set_status("recording")
        self._run(anim, 0.8)
        assert anim.alpha > 0.99 and not anim.hidden
        anim.set_status("idle")
        self._run(anim, 0.8)
        assert anim.hidden

    def test_silence_breathes_and_speech_swells(self):
        from mergescribe.ui.hud import HUDAnimation, _STRAND_AMPLITUDE

        def tallest(level):
            anim = HUDAnimation()
            anim.set_status("recording")
            self._run(anim, 0.5, level)
            return max(abs(y) for _, _, offsets in anim.strand_shapes() for y in offsets)

        quiet, loud = tallest(0.0), tallest(1.0)
        assert 0.0 < quiet < 0.15 * _STRAND_AMPLITUDE
        assert loud > 4 * quiet

    def test_strands_taper_to_nothing_at_both_ends(self):
        from mergescribe.ui.hud import HUDAnimation

        anim = HUDAnimation()
        anim.set_status("recording")
        self._run(anim, 0.5, 1.0)
        for _, _, offsets in anim.strand_shapes():
            assert abs(offsets[0]) < 1e-9 and abs(offsets[-1]) < 1e-9

    def test_releasing_merges_the_strands_into_one(self):
        """While listening the strands differ; while correcting they are one line."""
        from mergescribe.ui.hud import HUDAnimation

        def spread(anim):
            shapes = [offsets for _, _, offsets in anim.strand_shapes(0.5)]
            return max(abs(a - b) for column in zip(*shapes) for a in column for b in column)

        anim = HUDAnimation()
        anim.set_status("recording")
        self._run(anim, 1.0, 0.7)
        assert spread(anim) > 3.0
        anim.set_status("processing")
        self._run(anim, 1.5)
        assert spread(anim) < 0.1

    def test_the_pulse_starts_off_the_left_edge(self):
        from mergescribe.ui.hud import HUDAnimation

        anim = HUDAnimation()
        anim.set_status("recording")
        anim.set_status("processing")
        anim.step(1 / 60)
        assert anim.pulse_position < 0.0

    def test_widens_for_words_and_a_new_recording_starts_compact(self):
        from mergescribe.ui.hud import HUDAnimation

        anim = HUDAnimation()
        anim.set_status("recording")
        anim.set_transcript("so I was thinking")
        self._run(anim, 0.6)
        assert anim.width.value > 0.95
        anim.set_status("idle")
        anim.set_status("recording")     # pressed again before the fade finished
        assert anim.transcript == ""
        self._run(anim, 0.6)
        assert anim.width.value < 0.05

    def test_a_stalled_frame_does_not_fling_anything(self):
        from mergescribe.ui.hud import HUDAnimation

        anim = HUDAnimation()
        anim.set_status("recording")
        anim.step(5.0, 1.0)
        assert 0.0 <= anim.presence.value <= 1.0
        assert all(abs(y) < 50 for _, _, offsets in anim.strand_shapes() for y in offsets)


class TestHonestStrands:
    """The strands are the streams the session reported, doing what they did."""

    STREAMS = ["parakeet/built-in", "mai/built-in", "mai/solocast"]

    def _recording(self, streams=STREAMS):
        from mergescribe.ui.hud import HUDAnimation

        anim = HUDAnimation()
        anim.set_status("recording")
        anim.streams_planned("s1", streams)
        for _ in range(30):
            anim.step(1 / 60, 0.6)
        return anim

    def _run(self, anim, seconds):
        for _ in range(int(seconds * 60)):
            anim.step(1 / 60)

    def _strand(self, anim, key):
        return next(s for s in anim.strands if s.key == key)

    def test_one_strand_per_stream(self):
        assert len(self._recording(self.STREAMS[:2]).strands) == 2
        assert len(self._recording(self.STREAMS).strands) == 3

    def test_a_chunk_landing_mid_recording_flashes_without_joining(self):
        anim = self._recording()
        anim.stream_landed("s1", "mai/built-in", final=False)
        strand = self._strand(anim, "mai/built-in")
        assert strand.flash == 1.0
        self._run(anim, 0.5)
        assert strand.join.value < 0.05

    def test_final_results_join_the_line_one_at_a_time(self):
        """After release, the fast recognizer joins first and the slow one keeps working."""
        anim = self._recording()
        anim.set_status("processing")
        anim.stream_landed("s1", "parakeet/built-in", final=True)
        self._run(anim, 0.6)
        assert self._strand(anim, "parakeet/built-in").join.value > 0.95
        assert self._strand(anim, "mai/built-in").join.value < 0.05
        assert anim.correcting < 0.05          # still transcribing: no pulse yet

    def test_a_stream_left_behind_fades_out(self):
        anim = self._recording()
        anim.set_status("processing")
        anim.stream_dropped("s1", "mai/solocast")
        self._run(anim, 0.6)
        assert self._strand(anim, "mai/solocast").presence.value < 0.05

    def test_missing_one_chunk_while_recording_only_dims(self):
        anim = self._recording()
        anim.stream_dropped("s1", "mai/solocast")
        self._run(anim, 0.6)
        assert 0.2 < self._strand(anim, "mai/solocast").presence.value < 0.5

    def test_transcription_done_joins_the_rest_and_starts_the_pulse(self):
        """Streams cancelled by consensus were agreed for, so they join too."""
        anim = self._recording()
        anim.set_status("processing")
        anim.stream_landed("s1", "parakeet/built-in", final=True)
        anim.stream_dropped("s1", "mai/solocast")
        anim.transcription_done("s1")
        self._run(anim, 0.8)
        assert self._strand(anim, "mai/built-in").join.value > 0.95
        assert self._strand(anim, "mai/solocast").presence.value < 0.05
        assert anim.correcting > 0.95

    def test_a_finished_session_reporting_late_is_ignored(self):
        """Session A can still be typing when B starts recording."""
        anim = self._recording()
        anim.set_status("recording")
        anim.streams_planned("s2", self.STREAMS)
        anim.stream_landed("s1", "parakeet/built-in", final=True)
        anim.transcription_done("s1")
        anim.partial_text("s1", "old words")
        self._run(anim, 0.5)
        assert self._strand(anim, "parakeet/built-in").join.value < 0.05
        assert anim.correcting == 0.0
        assert anim.transcript == ""

    def test_without_reports_releasing_merges_everything(self):
        """Nothing will say when streams finish, so don't wait for it."""
        from mergescribe.ui.hud import HUDAnimation

        anim = HUDAnimation()
        anim.set_status("recording")
        anim.set_status("processing")
        self._run(anim, 0.8)
        assert all(s.join.value > 0.95 for s in anim.strands)
        assert anim.correcting > 0.95


class TestTailText:
    def test_short_text_passes_through_whole(self):
        from mergescribe.ui.hud import tail_text

        assert tail_text("hello there") == "hello there"

    def test_empty_stays_empty_so_the_hud_stays_compact(self):
        from mergescribe.ui.hud import tail_text

        assert tail_text("") == ""
        assert tail_text(None) == ""
        assert tail_text("   \n  ") == ""

    def test_keeps_the_end_not_the_start(self):
        """You want the words just spoken, not the ones from a minute ago."""
        from mergescribe.ui.hud import tail_text

        text = " ".join(f"word{i}" for i in range(100))
        tail = tail_text(text)
        assert "word99" in tail
        assert "word0 " not in tail
        assert tail.startswith("\u2026")

    def test_does_not_cut_a_word_in_half(self):
        from mergescribe.ui.hud import tail_text

        tail = tail_text("supercalifragilistic " * 20, budget=30)
        body = tail.lstrip("\u2026 ")
        assert body.split()[0] in ("supercalifragilistic", "")

    def test_collapses_whitespace_from_joined_chunks(self):
        """Chunk texts are joined with spaces and can arrive padded."""
        from mergescribe.ui.hud import tail_text

        assert tail_text("  one   two \n three ") == "one two three"

    def test_respects_the_budget(self):
        from mergescribe.ui.hud import tail_text

        for n in (10, 40, 200):
            assert len(tail_text("a bb ccc dddd " * 50, budget=n)) <= n + 2


class TestFacadeSafety:
    def test_disabled_hud_never_touches_appkit(self):
        from mergescribe.ui.hud import RecordingHUD

        hud = RecordingHUD(enabled=False)
        with patch.object(RecordingHUD, "_call_on_main") as marshal:
            hud.set_status("recording")
        marshal.assert_not_called()

    def test_a_failure_disables_the_hud_instead_of_raising(self):
        """Recording must continue even if the HUD is broken."""
        from mergescribe.ui.hud import RecordingHUD

        hud = RecordingHUD()
        with patch.object(RecordingHUD, "_call_on_main",
                          side_effect=RuntimeError("no window server")):
            hud.set_status("recording")   # must not raise
        assert hud._unavailable is True

        with patch.object(RecordingHUD, "_call_on_main") as marshal:
            hud.set_status("recording")
        marshal.assert_not_called()

    def test_hidden_status_does_not_build_a_panel(self):
        """main() sets 'idle' before the event loop; that must not create a window."""
        from mergescribe.ui.hud import RecordingHUD

        hud = RecordingHUD()
        with patch.object(RecordingHUD, "_ensure_controller") as ensure:
            hud._apply_status("idle")
        ensure.assert_not_called()

    def test_transcript_before_the_panel_exists_is_dropped(self):
        """A chunk can land before any status showed the HUD; that must not build one."""
        from mergescribe.ui.hud import RecordingHUD

        hud = RecordingHUD()
        with patch.object(RecordingHUD, "_ensure_controller") as ensure:
            hud._apply_event("partial_text", ("s1", "some words"))
        ensure.assert_not_called()

    def test_a_disabled_hud_ignores_session_reports(self):
        from mergescribe.ui.hud import RecordingHUD

        hud = RecordingHUD(enabled=False)
        with patch.object(RecordingHUD, "_call_on_main") as marshal:
            hud.stream_landed("s1", "parakeet/mic", True)
            hud.partial_text("s1", "words")
        marshal.assert_not_called()

    def test_shutdown_without_a_panel_is_a_no_op(self):
        from mergescribe.ui.hud import RecordingHUD

        RecordingHUD().shutdown()   # must not raise

    def test_level_source_errors_are_swallowed(self):
        """The meter reads a live engine attribute; a stale reference must not crash."""
        from mergescribe.ui.hud import RecordingHUD

        hud = RecordingHUD(level_source=Mock(side_effect=RuntimeError("gone")))
        with patch.object(RecordingHUD, "_call_on_main"):
            hud.set_status("recording")


class TestMenuBarDeference:
    """The menu bar should stop flashing red only while the HUD is actually carrying state."""

    def test_live_hud_claims_the_status(self):
        from mergescribe.ui.hud import RecordingHUD

        assert RecordingHUD().is_showing is True

    def test_disabled_hud_hands_status_back(self):
        from mergescribe.ui.hud import RecordingHUD

        assert RecordingHUD(enabled=False).is_showing is False

    def test_failed_hud_hands_status_back(self):
        """A broken HUD must not leave the app with no indication at all."""
        from mergescribe.ui.hud import RecordingHUD

        hud = RecordingHUD()
        with patch.object(RecordingHUD, "_call_on_main", side_effect=RuntimeError("boom")):
            hud.set_status("recording")
        assert hud.is_showing is False


class TestAudioEngineFeed:
    def test_engine_exposes_a_level_for_the_meter(self):
        from mergescribe.audio import AudioEngine
        from mergescribe.config import Config

        config = Mock(spec=Config)
        config.preroll_seconds = 1.0
        config.sample_rate = 16000
        config.silence_threshold = 1.2
        engine = AudioEngine(config)
        assert engine.current_level == 0.0
