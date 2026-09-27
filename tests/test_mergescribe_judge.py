"""
Tests for skipping correction when the transcript needs none.

The judge is a network call on the critical path of every dictation, so what
matters most is that it cannot do harm: off by default, silent on failure, and
incapable of holding up the correction model that is already running.
"""

import json
import time
from unittest.mock import Mock, patch

from mergescribe.types import ConfigSnapshot, TranscriptionResult


def _config(**overrides):
    config = Mock(spec=ConfigSnapshot)
    config.judge_enabled = True
    config.judge_model = "~typesafe/jev-latest"
    config.judge_timeout_ms = 900
    config.openrouter_api_key = "sk-or-test"
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def _results(*texts):
    return [TranscriptionResult(text=t, provider=f"p{i}", mic="m", latency_ms=1)
            for i, t in enumerate(texts)]


def _reply(filler=0.05, restart=0.05, misheard=0.05, punct=0.05, none=0.05, pick="0",
           count=4, per=None):
    """Jev's answer: every job scored the same for every transcript, unless per overrides one."""
    probabilities = {str(i): (0.9 if str(i) == pick else 0.02) for i in range(count)}
    probabilities["none"] = none
    answers = {"best": {"choice": pick, "probabilities": probabilities, "confidence": 0.9}}
    for i in range(count):
        scores = {"filler": filler, "restart": restart, "misheard": misheard, "punct": punct}
        scores.update((per or {}).get(i, {}))
        answers.update({f"{job}_{i}": {"noul": v} for job, v in scores.items()})
    body = {"model": "typesafe/jev-1.13", "answers": answers}
    response = Mock()
    response.read.return_value = json.dumps(body).encode()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    return response


class TestSafety:
    """Nothing here may cost a dictation."""

    def test_off_by_default(self):
        """Read the declared defaults, not this machine's settings file."""
        from dataclasses import fields

        from mergescribe.config import DEFAULT_CONFIG
        from mergescribe.judge import judge_transcripts

        declared = {f.name: f.default for f in fields(ConfigSnapshot)}
        assert declared["judge_enabled"] is False
        assert DEFAULT_CONFIG["judge_enabled"] is False

        with patch("urllib.request.urlopen") as opened:
            assert judge_transcripts(_results("hi"), _config(judge_enabled=False)) is None
        opened.assert_not_called()

    def test_a_config_stub_cannot_switch_it_on(self):
        """Mock attributes are truthy; only a real True may open a socket."""
        from mergescribe.judge import judge_transcripts

        config = Mock(spec=ConfigSnapshot)   # judge_enabled is a Mock, not True
        with patch("urllib.request.urlopen") as opened:
            assert judge_transcripts(_results("hi"), config) is None
        opened.assert_not_called()

    def test_no_api_key_means_no_call(self):
        from mergescribe.judge import judge_transcripts

        with patch("urllib.request.urlopen") as opened:
            assert judge_transcripts(_results("hi"), _config(openrouter_api_key="")) is None
        opened.assert_not_called()

    def test_network_failure_is_silent(self):
        from mergescribe.judge import judge_transcripts

        with patch("urllib.request.urlopen", side_effect=TimeoutError("slow")):
            assert judge_transcripts(_results("hi"), _config()) is None

    def test_an_unexpected_reply_is_silent(self):
        from mergescribe.judge import judge_transcripts

        response = Mock()
        response.read.return_value = b'{"answers": {"filler": {}}}'
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with patch("urllib.request.urlopen", return_value=response):
            assert judge_transcripts(_results("hi"), _config()) is None

    def test_the_timeout_survives_a_junk_setting(self):
        from mergescribe.judge import _timeout_seconds

        assert _timeout_seconds(_config(judge_timeout_ms=250)) == 0.25
        assert _timeout_seconds(_config(judge_timeout_ms="fast")) == 0.9


class TestVerdict:
    def test_a_clean_transcript_is_cleared_for_typing(self):
        from mergescribe.judge import judge_transcripts

        with patch("urllib.request.urlopen", return_value=_reply()):
            verdict = judge_transcripts(_results("Ship it Friday."), _config())
        assert verdict.clean is True
        assert verdict.text == "Ship it Friday."
        assert verdict.scores["filler"] == 0.05

    def test_filler_sends_it_to_the_correction_model(self):
        from mergescribe.judge import judge_transcripts

        with patch("urllib.request.urlopen", return_value=_reply(filler=0.9)):
            assert judge_transcripts(_results("um, ship it"), _config()).clean is False

    def test_a_self_correction_sends_it_to_the_correction_model(self):
        from mergescribe.judge import judge_transcripts

        with patch("urllib.request.urlopen", return_value=_reply(restart=0.8)):
            assert judge_transcripts(_results("Tuesday, no wait, Friday"), _config()).clean is False

    def test_a_suspected_mishearing_sends_it_to_the_correction_model(self):
        """The one failure that would actually hurt: typing a word that was misheard."""
        from mergescribe.judge import judge_transcripts

        with patch("urllib.request.urlopen", return_value=_reply(misheard=0.95)):
            assert judge_transcripts(_results("the new series"), _config()).clean is False

    def test_a_punctuation_or_grammar_fix_sends_it_to_the_correction_model(self):
        from mergescribe.judge import judge_transcripts

        with patch("urllib.request.urlopen", return_value=_reply(punct=0.8)):
            assert judge_transcripts(_results("ship it friday"), _config()).clean is False

    def test_none_of_them_right_sends_it_to_the_correction_model(self):
        """The recognizers disagree and Jev trusts none of them: the correction should merge."""
        from mergescribe.judge import judge_transcripts

        with patch("urllib.request.urlopen", return_value=_reply(none=0.7)):
            assert judge_transcripts(_results("the new series", "the new Siri"), _config()).clean is False

    def test_the_verdict_is_about_the_transcript_it_picks(self):
        """
        Jev scores each question on its own, so the jobs are asked of every
        transcript and the verdict reads the scores of the one it would type.
        """
        from mergescribe.judge import judge_transcripts

        results = _results("um, ship it Friday", "Ship it Friday.")
        with patch("urllib.request.urlopen", return_value=_reply(pick="1", per={0: {"filler": 0.95}})):
            verdict = judge_transcripts(results, _config())
        assert verdict.clean is True and verdict.text == "Ship it Friday."
        with patch("urllib.request.urlopen", return_value=_reply(pick="0", per={0: {"filler": 0.95}})):
            assert judge_transcripts(results, _config()).clean is False

    def test_every_job_is_asked_of_every_transcript_in_one_call(self):
        from mergescribe.judge import JOBS, questions

        asked = questions(3)
        assert set(asked["best"]["criteria"]) == {"0", "1", "2", "none"}
        assert {f"{job}_{i}" for job in JOBS for i in range(3)} <= set(asked)

    def test_it_types_the_transcript_the_judge_picked(self):
        from mergescribe.judge import judge_transcripts

        with patch("urllib.request.urlopen", return_value=_reply(pick="1")):
            verdict = judge_transcripts(_results("the new series", "the new Siri"), _config())
        # Candidates are ordered longest first, so index 1 is the shorter one.
        assert verdict.text == "the new Siri"

    def test_an_out_of_range_choice_falls_back_to_the_first(self):
        from mergescribe.judge import judge_transcripts

        with patch("urllib.request.urlopen", return_value=_reply(pick="7")):
            assert judge_transcripts(_results("only one"), _config()).text == "only one"

    def test_identical_transcripts_are_offered_once(self):
        from mergescribe.judge import _candidates

        assert _candidates(_results("Ship it.", "ship it.", "")) == ["Ship it."]


class TestPreferences:
    """Typing a transcript as it stands must not skip what the speaker asked for."""

    SLACK = ("Slack", "general - Acme - Slack", "com.tinyspeck.slackmacgap")
    RULE = "In Slack, write everything in lowercase."

    def _sent(self, config, context=None):
        from mergescribe.judge import judge_transcripts
        from mergescribe.types import AppContext

        with patch("urllib.request.urlopen", return_value=_reply()) as urlopen:
            judge_transcripts(_results("Ship it Friday."), config, AppContext(*context) if context else None)
        return json.loads(urlopen.call_args.args[0].data)

    def test_they_are_asked_about_where_the_text_is_going(self):
        body = self._sent(_config(custom_instructions=self.RULE), self.SLACK)
        asked = body["questions"]["prefs_0"]["instructions"]
        assert self.RULE in asked and "Slack (window: general - Acme - Slack)" in asked

    def test_they_stay_out_of_the_shared_state(self):
        """In the state they made every other question more cautious; see PREFERENCES."""
        body = self._sent(_config(custom_instructions=self.RULE), self.SLACK)
        assert body["state"] == {"transcripts": ["Ship it Friday."]}

    def test_no_preferences_means_no_question(self):
        for config in (_config(custom_instructions="  "), _config()):   # blank, and a stub's non-string
            assert not any(k.startswith("prefs_") for k in self._sent(config, self.SLACK)["questions"])

    def test_an_unknown_window_is_still_asked(self):
        body = self._sent(_config(custom_instructions=self.RULE))
        assert "an unknown app" in body["questions"]["prefs_0"]["instructions"]

    def test_a_replaced_correction_prompt_counts_as_a_preference(self):
        """The four jobs are the default prompt's; a prompt of your own can ask for more."""
        from mergescribe.correct import DEFAULT_SYSTEM_CONTEXT
        from mergescribe.judge import preferences

        own = "Translate what I say into Spanish."
        assert preferences(_config(system_prompt=own, custom_instructions=self.RULE)) == f"{own}\n\n{self.RULE}"
        assert preferences(_config(system_prompt=own, custom_instructions="")) == own
        # The default, blank or a stub's non-string: nothing beyond the jobs.
        for prompt in (DEFAULT_SYSTEM_CONTEXT, "", Mock()):
            assert preferences(_config(system_prompt=prompt, custom_instructions=self.RULE)) == self.RULE

    def test_a_broken_preference_sends_it_to_the_correction_model(self):
        from mergescribe.judge import judge_transcripts

        with patch("urllib.request.urlopen", return_value=_reply(per={0: {"prefs": 0.9}})):
            verdict = judge_transcripts(_results("Ship it Friday."), _config(custom_instructions=self.RULE))
        assert verdict.clean is False and verdict.scores["prefs"] == 0.9


class TestRace:
    """The judge and the correction model are asked together; first usable answer wins."""

    def _session(self, **overrides):
        import tests.test_mergescribe_session as session_tests

        settings = {"judge_enabled": True, "judge_timeout_ms": 900,
                    "space_between_dictations": False, **overrides}
        session = session_tests.TestFinalizeSteps()._session(**settings)
        session.metrics = Mock()
        return session

    def _run(self, verdict, correction_delay=0.0, tokens=("corrected text",)):
        from mergescribe.judge import Verdict   # noqa: F401  (documents the shape)

        session = self._session()
        typed = []

        def slow_correction(results, context, config, on_delta=None, **kw):
            time.sleep(correction_delay)
            for token in tokens:
                if on_delta:
                    on_delta(token)
            return "".join(tokens)

        with patch("mergescribe.judge.judge_transcripts", return_value=verdict), \
             patch("mergescribe.correct.correct_with_llm", slow_correction), \
             patch("mergescribe.session.type_text", side_effect=typed.append):
            result = session._stream_correction(_results("raw words"))
        return session, result, "".join(typed)

    def test_a_clean_verdict_types_the_transcript_and_drops_the_correction(self):
        from mergescribe.judge import Verdict

        verdict = Verdict(clean=True, text="raw words", scores={"filler": 0.1}, latency_ms=240)
        session, result, typed = self._run(verdict, correction_delay=0.4)
        assert typed == "raw words"
        assert result == "raw words"
        assert session.output_method == "judged"

    def test_a_dirty_verdict_leaves_the_correction_alone(self):
        from mergescribe.judge import Verdict

        verdict = Verdict(clean=False, text="raw words", scores={"filler": 0.9})
        _, result, typed = self._run(verdict)
        assert typed == "corrected text"
        assert result == "corrected text"

    def test_no_judge_means_the_old_path(self):
        _, result, typed = self._run(None)
        assert typed == "corrected text"
        assert result == "corrected text"

    def test_a_late_verdict_never_types_over_the_correction(self):
        """The correction model got there first; two copies must not land."""
        from mergescribe.judge import Verdict

        late = Verdict(clean=True, text="raw words", scores={}, latency_ms=800)

        def slow_judge(*a, **k):
            time.sleep(0.25)
            return late

        session = self._session()
        typed = []
        with patch("mergescribe.judge.judge_transcripts", slow_judge), \
             patch("mergescribe.correct.correct_with_llm",
                   lambda *a, on_delta=None, **k: (on_delta("corrected text"), "corrected text")[1]), \
             patch("mergescribe.session.type_text", side_effect=typed.append):
            result = session._stream_correction(_results("raw words"))
        assert "".join(typed) == "corrected text"
        assert result == "corrected text"

    def test_the_verdict_is_logged_either_way(self):
        from mergescribe.judge import Verdict

        session, _, _ = self._run(Verdict(clean=True, text="raw words",
                                          scores={"filler": 0.02}, latency_ms=210))
        events = [c for c in session.metrics.log.call_args_list if c.args[0] == "judge"]
        assert events and events[0].kwargs["clean"] is True
        assert events[0].kwargs["latency_ms"] == 210

    def test_a_hanging_judge_does_not_hold_up_typing(self):
        """A judge that never answers must not add its timeout to the dictation."""
        from mergescribe.types import TranscriptionResult   # noqa: F401

        session = self._session(judge_timeout_ms=5000)
        typed = []

        def hanging_judge(*a, **k):
            time.sleep(3.0)
            return None

        started = time.monotonic()
        with patch("mergescribe.judge.judge_transcripts", hanging_judge), \
             patch("mergescribe.correct.correct_with_llm",
                   lambda *a, on_delta=None, **k: (on_delta("corrected text"), "corrected text")[1]), \
             patch("mergescribe.session.type_text", side_effect=typed.append):
            session._stream_correction(_results("raw words"))
        assert "".join(typed) == "corrected text"
        assert time.monotonic() - started < 1.0, "typing waited on the judge"


class TestMultiChunk:
    """Across chunks the transcripts are consecutive, not alternative."""

    def _session(self, chunks):
        import tests.test_mergescribe_session as session_tests
        from mergescribe.types import TranscriptionResult

        session = session_tests.TestFinalizeSteps()._session(
            judge_enabled=True, judge_timeout_ms=900, space_between_dictations=False)
        session.metrics = Mock()
        for i in range(chunks):
            session.chunk_results.append(
                (i + 1, [TranscriptionResult(text=f"chunk {i}", provider="p", mic="m", latency_ms=1)], None))
        return session

    def test_a_multi_chunk_dictation_never_reaches_the_judge(self):
        """Otherwise a clean verdict would type one chunk and drop the others."""
        from mergescribe.judge import judge_transcripts   # noqa: F401

        session = self._session(chunks=3)
        with patch("mergescribe.judge.judge_transcripts") as asked:
            box = session._start_judge(_results("chunk 0", "chunk 1", "chunk 2"))
        assert box["done"].is_set()
        assert box.get("v") is None
        asked.assert_not_called()

    def test_a_single_chunk_dictation_still_asks(self):
        session = self._session(chunks=1)
        with patch("mergescribe.judge.judge_transcripts", return_value=None) as asked:
            box = session._start_judge(_results("just the one"))
            box["done"].wait(1.0)
        asked.assert_called_once()
        assert asked.call_args.args[2] is session.context   # where it is going, for the preferences
