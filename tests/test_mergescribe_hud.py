"""
Tests for the recording HUD.

Only the pure logic and the failure behaviour are covered — the AppKit panel
itself needs a GUI session. What matters here is that the HUD can never take
the app down with it, since it is pure decoration over the recording path.
"""

from unittest.mock import Mock, patch

import numpy as np


def _aurora(anim, width=120.0, track=None):
    """This frame's aurora as floats in 0..1, (rows, columns, RGBA), 1pt per pixel."""
    from mergescribe.ui.hud import aurora_rgba

    return aurora_rgba(anim, width, 40.0, track_pt=track).astype(float) / 255.0


def _lift(anim, **kwargs):
    """How much brighter the brightest column in the middle of the strip is than the dimmest."""
    columns = _aurora(anim, **kwargs)[..., 3].sum(axis=0)[30:90]
    return columns.max() / columns.min()


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
        from mergescribe.ui.hud import HUDAnimation

        def light(level):
            anim = HUDAnimation()
            anim.set_status("recording")
            self._run(anim, 0.5, level)
            return _aurora(anim)[..., 3]

        quiet, loud = light(0.0), light(1.0)
        assert quiet.sum() > 0.0, "silence should breathe, not go dark"
        assert loud[:20].sum() > 8 * quiet[:20].sum(), "a voice sends the rays up the capsule"

    def test_the_aurora_fades_to_nothing_at_both_ends(self):
        from mergescribe.ui.hud import HUDAnimation

        anim = HUDAnimation()
        anim.set_status("recording")
        self._run(anim, 0.5, 1.0)
        alpha = _aurora(anim)[..., 3]
        assert alpha[:, 0].max() < 0.02 and alpha[:, -1].max() < 0.02
        assert alpha[:, 60].max() > 0.5

    def test_releasing_settles_the_curtains_into_one_hem(self):
        """While listening each curtain's hem wanders; once merged they lie on one line."""
        from mergescribe.ui.hud import HUDAnimation

        def wander(anim):
            """How much the brightest row moves across the middle of the strip."""
            return np.argmax(_aurora(anim)[:, 30:90, 3], axis=0).std()

        anim = HUDAnimation()
        anim.set_status("recording")
        self._run(anim, 1.0, 0.7)
        assert wander(anim) > 1.0
        anim.set_status("processing")
        self._run(anim, 1.5)
        assert wander(anim) < 0.3

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
        assert _aurora(anim)[0, :, 3].max() < 0.3, "the rays should not be flung to the top"


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


class TestSkippedCorrection:
    """A dictation typed as it stood should not look like one being corrected."""

    def _anim(self):
        from mergescribe.ui.hud import HUDAnimation

        anim = HUDAnimation()
        anim.set_status("recording")
        anim.streams_planned("s1", ["parakeet/built-in", "mai/built-in"])
        for _ in range(30):
            anim.step(1 / 60, 0.6)
        anim.set_status("processing")
        anim.transcription_done("s1")
        return anim

    def test_the_pulse_never_starts(self):
        anim = self._anim()
        anim.correction_skipped("s1")
        for _ in range(60):
            anim.step(1 / 60)
        assert anim.pulse.value < 0.2
        assert anim.skipped is True

    def test_it_flashes_and_settles(self):
        anim = self._anim()
        anim.correction_skipped("s1")
        bright = _aurora(anim)[..., 3].sum()
        for _ in range(60):
            anim.step(1 / 60)
        settled = _aurora(anim)[..., 3].sum()
        assert bright > settled * 1.5, "the flash should be visible, then fade"

    def test_a_stale_session_cannot_flash(self):
        anim = self._anim()
        anim.correction_skipped("other")
        assert anim.skipped is False

    def test_a_new_recording_clears_it(self):
        anim = self._anim()
        anim.correction_skipped("s1")
        anim.set_status("idle")
        for _ in range(60):
            anim.step(1 / 60)
        anim.reset()
        assert anim.skipped is False and anim.flash == 0.0


class TestTokenRipples:
    """The hem's light should move at the rate words land, and stall when the stream does."""

    def _correcting(self):
        from mergescribe.ui.hud import HUDAnimation

        anim = HUDAnimation()
        anim.set_status("recording")
        anim.streams_planned("s1", ["parakeet/built-in"])
        anim.set_status("processing")
        anim.transcription_done("s1")
        for _ in range(20):
            anim.step(1 / 60)
        return anim

    def test_a_token_sends_a_ripple_that_travels(self):
        anim = self._correcting()
        anim.token_typed("s1")
        starts = list(anim._ripples)
        for _ in range(20):
            anim.step(1 / 60)
        assert anim._ripples[0] > starts[0], "the ripple should move along the line"

    def test_a_stalled_stream_goes_quiet(self):
        """No tokens, no movement: a stalled correction must look stalled."""
        anim = self._correcting()
        anim.token_typed("s1")
        for _ in range(180):      # three seconds of silence from the model
            anim.step(1 / 60)
        assert anim._ripples == []
        assert _lift(anim) < 1.05, "the hem should be evenly lit, with nothing travelling"

    def test_a_burst_of_tokens_is_spread_out(self):
        """Five tokens in one packet should not stack into one lump."""
        anim = self._correcting()
        for _ in range(5):
            anim.token_typed("s1")
        assert len(set(anim._ripples)) == 5
        assert max(anim._ripples) - min(anim._ripples) > 0.1

    def test_ripples_are_capped(self):
        anim = self._correcting()
        for _ in range(50):
            anim.token_typed("s1")
        assert len(anim._ripples) <= 8

    def test_an_unreported_stream_still_sweeps(self):
        """Nothing calls token_typed in the preview; the light must still move."""
        anim = self._correcting()
        for _ in range(30):
            anim.step(1 / 60)
        assert _lift(anim) > 1.3

    def test_a_words_light_runs_the_track_the_shimmer_does(self):
        """
        Ripple positions are fractions of the whole content, strip and words
        together, so the light on the hem and the shimmer over the words are
        one travelling thing rather than two that disagree.
        """
        anim = self._correcting()
        anim.token_typed("s1")
        for _ in range(7):
            anim.step(1 / 60)
        (p,) = anim._ripples
        for track in (480.0, 800.0):
            brightest = np.argmax(_aurora(anim, track=track)[..., 3].sum(axis=0))
            assert abs(brightest - p * track) < 3.0

    def test_the_shimmer_follows_the_ripples(self):
        anim = self._correcting()
        anim.token_typed("s1")
        first = anim.pulse_track
        for _ in range(20):
            anim.step(1 / 60)
        assert anim.pulse_track > first, "the words should light up as the line moves"

    def test_the_shimmer_leaves_when_the_stream_does(self):
        anim = self._correcting()
        anim.token_typed("s1")
        for _ in range(180):
            anim.step(1 / 60)
        assert anim.pulse_track < 0, "nothing to shimmer once the ripples are gone"

    def test_a_stale_session_cannot_ripple(self):
        anim = self._correcting()
        anim.token_typed("other")
        assert anim._ripples == []


class TestDiscarded:
    def test_a_called_off_dictation_draws_itself_in(self):
        from mergescribe.ui.hud import HUDAnimation

        anim = HUDAnimation()
        anim.set_status("recording")
        anim.streams_planned("s1", ["parakeet/built-in"])
        anim.set_status("processing")
        anim.dictation_discarded("s1")
        for _ in range(45):
            anim.step(1 / 60)
        assert anim.discarded is True
        assert anim.collapse.value > 0.9

    def test_a_new_recording_reopens_it(self):
        from mergescribe.ui.hud import HUDAnimation

        anim = HUDAnimation()
        anim.set_status("recording")
        anim.streams_planned("s1", ["parakeet/built-in"])
        anim.dictation_discarded("s1")
        for _ in range(30):
            anim.step(1 / 60)
        anim.set_status("recording")
        assert anim.collapse.value == 0.0 and anim.discarded is False


class TestCapsuleAndLight:
    """The dot that opens into a capsule, and the light along its edge."""

    def _listening(self, level=0.0, frames=40):
        from mergescribe.ui.hud import HUDAnimation

        anim = HUDAnimation()
        anim.set_status("recording")
        anim.streams_planned("s1", ["parakeet/built-in"])
        for _ in range(frames):
            anim.step(1 / 60, level)
        return anim

    def test_it_opens_from_a_dot(self):
        from mergescribe.ui.hud import HUDAnimation

        anim = HUDAnimation()
        anim.set_status("recording")
        assert anim.morph == 0.0
        anim.step(1 / 60)
        assert 0.0 < anim.morph < 0.5
        for _ in range(60):
            anim.step(1 / 60)
        assert abs(anim.morph - 1.0) < 0.05

    def test_the_contents_wait_for_the_capsule(self):
        """Strands appearing inside a 40pt dot would be crushed into a smudge."""
        from mergescribe.ui.hud import HUDAnimation

        anim = HUDAnimation()
        anim.set_status("recording")
        anim.step(1 / 60)
        assert anim.content_alpha == 0.0
        for _ in range(60):
            anim.step(1 / 60)
        assert anim.content_alpha > 0.95

    def test_the_edge_is_nearly_dark_in_silence_and_lights_with_a_voice(self):
        quiet = self._listening(level=0.0).edge_light()[0]
        loud = self._listening(level=1.0).edge_light()[0]
        assert quiet < 0.05
        assert loud > 5 * quiet

    def test_the_sweep_runs_faster_once_released(self):
        listening = self._listening(frames=60)
        start = listening.edge_phase
        for _ in range(60):
            listening.step(1 / 60)
        drift = (listening.edge_phase - start) % 1.0

        working = self._listening(frames=60)
        working.set_status("processing")
        for _ in range(40):
            working.step(1 / 60)
        start = working.edge_phase
        for _ in range(60):
            working.step(1 / 60)
        assert (working.edge_phase - start) % 1.0 > 3 * drift

    def test_a_skipped_correction_flares_the_whole_edge_and_stops_the_sweep(self):
        anim = self._listening(frames=60)
        anim.set_status("processing")
        anim.transcription_done("s1")
        anim.correction_skipped("s1")
        steady, _, sweep = anim.edge_light()
        assert steady > 0.8 and sweep == 0.0

    def test_a_discarded_dictation_puts_the_light_out(self):
        anim = self._listening(level=0.8, frames=60)
        anim.dictation_discarded("s1")
        for _ in range(60):
            anim.step(1 / 60)
        steady, _, sweep = anim.edge_light()
        assert steady < 0.05 and sweep < 0.05
        assert anim.content_alpha < 0.05

    def test_the_aurora_climbs_with_your_voice(self):
        def highest_lit_row(anim):
            return np.nonzero((_aurora(anim)[..., 3] > 0.25).any(axis=1))[0].min()

        quiet = highest_lit_row(self._listening(level=0.0))
        loud = highest_lit_row(self._listening(level=1.0))
        assert loud + 10 < quiet, "a louder voice should light rays nearer the top"


class TestAuroraImage:
    """The image handed to the GPU each frame."""

    def _merged(self):
        from mergescribe.ui.hud import HUDAnimation

        anim = HUDAnimation()
        anim.set_status("recording")
        anim.set_status("processing")
        for _ in range(90):
            anim.step(1 / 60)
        return anim

    def test_padding_and_scale(self):
        from mergescribe.ui.hud import HUDAnimation, aurora_rgba

        anim = HUDAnimation()
        anim.set_status("recording")
        for _ in range(30):
            anim.step(1 / 60, 0.5)
        image = aurora_rgba(anim, 112.0, 40.0, pad_pt=12.0)
        assert image.shape == (64, 136, 4) and image.dtype == np.uint8
        assert not image[:12].any() and not image[:, :12].any(), "the pad is clear, for the glow"
        assert aurora_rgba(anim, 112.0, 40.0, scale=2.0).shape == (80, 224, 4)

    def test_it_is_premultiplied(self):
        """Core Image blurs premultiplied pixels; colour brighter than its alpha would halo."""
        image = _aurora(self._merged()).reshape(-1, 4)
        assert (image[:, :3] <= image[:, 3:] + 1 / 255).all()

    def test_the_settled_hem_is_mint(self):
        """The curtains' greens, cyans and violets agree on one colour once merged."""
        pixel = _aurora(self._merged())[:, 60]
        r, g, b, _ = pixel[np.argmax(pixel[:, 3])]
        assert g > b > r

    def test_nothing_is_drawn_before_the_capsule_opens(self):
        from mergescribe.ui.hud import HUDAnimation

        anim = HUDAnimation()
        anim.set_status("recording")
        anim.step(1 / 60)
        assert not _aurora(anim)[..., 3].any()
