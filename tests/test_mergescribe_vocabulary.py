"""
Tests for corrections handed to the model as vocabulary evidence.

Code decides only what counts as repeated. Judging what a correction means is
left to the model, and one test pins that the code no longer tries.
"""

import json


def row(session, typed, corrected, ts=0.0):
    return {"session_id": session, "typed": typed, "corrected": corrected, "ts": ts}


class TestRepeatedCorrections:
    def test_a_correction_made_in_two_dictations_is_kept(self):
        """The Claude case: heard as "cloud", fixed twice."""
        from mergescribe.vocabulary import repeated_corrections

        rows = [row("s1", "thoughts about cloud today", "thoughts about Claude today"),
                row("s2", "as good as cloud is", "as good as Claude is")]
        assert repeated_corrections(rows) == [(["cloud"], "Claude", 2)]

    def test_different_mishearings_of_one_word_add_up(self):
        from mergescribe.vocabulary import repeated_corrections

        rows = [row("s1", "about cloud", "about Claude"), row("s2", "the Claud model", "the Claude model")]
        assert repeated_corrections(rows) == [(["cloud", "Claud"], "Claude", 2)]

    def test_a_single_correction_is_not_enough(self):
        from mergescribe.vocabulary import repeated_corrections

        assert repeated_corrections([row("s1", "about cloud", "about Claude")]) == []

    def test_two_fixes_in_one_dictation_count_once(self):
        from mergescribe.vocabulary import repeated_corrections

        rows = [row("s1", "cloud and cloud", "Claude and Claude"), row("s1", "cloud", "Claude")]
        assert repeated_corrections(rows) == []

    def test_case_and_punctuation_changes_carry_nothing(self):
        from mergescribe.vocabulary import repeated_corrections

        rows = [row("s1", "okay sure", "Okay, sure."), row("s2", "okay sure", "Okay, sure.")]
        assert repeated_corrections(rows) == []

    def test_long_rewrites_are_left_out_of_the_prompt(self):
        from mergescribe.vocabulary import repeated_corrections

        long_before = "we should probably go and look at it"
        long_after = "I will review the deployment logs later tonight"
        rows = [row("s1", long_before, long_after), row("s2", long_before, long_after)]
        assert repeated_corrections(rows) == []

    def test_code_does_not_judge_what_a_correction_means(self):
        """Ordinary words used to be filtered by a dictionary; that call is the model's now."""
        from mergescribe.vocabulary import repeated_corrections

        rows = [row("s1", "over their", "over there"), row("s2", "over their", "over there")]
        assert repeated_corrections(rows) == [(["their"], "there", 2)]

    def test_most_established_first_and_capped(self):
        from mergescribe.vocabulary import repeated_corrections

        rows = [row(f"a{i}", "for post grass", "for Postgres") for i in range(3)]
        rows += [row(f"b{i}", "about cloud", "about Claude") for i in range(2)]
        assert [after for _, after, _ in repeated_corrections(rows)] == ["Postgres", "Claude"]
        assert [after for _, after, _ in repeated_corrections(rows, limit=1)] == ["Postgres"]


class TestPrompt:
    def test_nothing_learned_adds_nothing(self):
        from mergescribe.vocabulary import vocabulary_prompt

        assert vocabulary_prompt([]) == ""

    def test_corrections_are_evidence_not_replacements(self):
        from mergescribe.vocabulary import vocabulary_prompt

        text = vocabulary_prompt([(["cloud", "Claud"], "Claude", 2)])
        assert '"cloud" or "Claud" corrected to "Claude" (2 dictations)' in text
        assert "not a list of replacements" in text
        assert "only where the audio" in text


class TestLearnedCorrectionsFile:
    def test_reads_the_corpus_and_notices_new_corrections(self, tmp_path):
        import os
        from mergescribe import vocabulary

        corpus = tmp_path / "corrections.jsonl"
        corpus.write_text(json.dumps(row("s1", "about cloud", "about Claude")) + "\n")
        assert vocabulary.learned_corrections(corpus) == []

        with corpus.open("a") as f:
            f.write(json.dumps(row("s2", "the Claud model", "the Claude model")) + "\n")
        stat = corpus.stat()
        os.utime(corpus, (stat.st_atime, stat.st_mtime + 5))   # invalidate the cache
        assert vocabulary.learned_corrections(corpus) == [(["cloud", "Claud"], "Claude", 2)]

    def test_missing_corpus_is_harmless(self, tmp_path):
        from mergescribe.vocabulary import learned_corrections

        assert learned_corrections(tmp_path / "nope.jsonl") == []


class TestPromptWiring:
    def _config(self, learn):
        from unittest.mock import Mock
        from mergescribe.types import ConfigSnapshot

        config = Mock(spec=ConfigSnapshot)
        config.system_prompt = ""
        config.routing_instructions = ""
        config.learn_vocabulary = learn
        return config

    def test_learned_corrections_reach_the_system_prompt(self):
        from unittest.mock import patch
        from mergescribe.correct import build_system_prompt

        with patch("mergescribe.vocabulary.learned_corrections",
                   return_value=[(["cloud"], "Claude", 2)]):
            prompt = build_system_prompt(self._config(True))
        assert '"cloud" corrected to "Claude"' in prompt

    def test_the_switch_turns_it_off(self):
        from unittest.mock import patch
        from mergescribe.correct import build_system_prompt

        with patch("mergescribe.vocabulary.learned_corrections",
                   return_value=[(["cloud"], "Claude", 2)]):
            prompt = build_system_prompt(self._config(False))
        assert "Claude" not in prompt
