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


class TestTruncationArtifacts:
    """Word diffs cut inside words; the leftovers are not vocabulary."""

    def test_a_fragment_of_the_old_word_is_not_a_correction(self):
        from mergescribe.vocabulary import repeated_corrections

        rows = [row("s1", "look at them okay", "look at the okay"),
                row("s2", "send them now", "send the now")]
        assert repeated_corrections(rows) == []

    def test_a_dropped_negation_is_never_learned(self):
        """"not" -> "no" would teach the model to weaken negations."""
        from mergescribe.vocabulary import repeated_corrections

        rows = [row("s1", "it is not ready", "it is no ready"),
                row("s2", "we are not sure", "we are no sure")]
        assert repeated_corrections(rows) == []

    def test_a_real_respelling_still_counts(self):
        from mergescribe.vocabulary import repeated_corrections

        rows = [row("s1", "deploy it to cooper netties", "deploy it to Kubernetes"),
                row("s2", "the cube or netties cluster", "the Kubernetes cluster")]
        assert [after for _, after, _ in repeated_corrections(rows)] == ["Kubernetes"]

    def test_an_acronym_the_recognizer_mangles_still_counts(self):
        from mergescribe.vocabulary import repeated_corrections

        rows = [row("s1", "parse the Jason file", "parse the JSON file"),
                row("s2", "return Jason here", "return JSON here")]
        assert [after for _, after, _ in repeated_corrections(rows)] == ["JSON"]

    def test_spelling_a_word_out_more_fully_is_kept(self):
        """"Claud" -> "Claude" adds a letter; that is the whole point."""
        from mergescribe.vocabulary import repeated_corrections

        rows = [row("s1", "ask Claud about it", "ask Claude about it"),
                row("s2", "Claud said so", "Claude said so")]
        assert [after for _, after, _ in repeated_corrections(rows)] == ["Claude"]


class TestSuppliedVocabulary:
    """Terms handed in from outside - by a nightly job over what the speaker reads, say."""

    def test_one_term_per_line_with_comments(self, tmp_path):
        from mergescribe.vocabulary import supplied_terms

        path = tmp_path / "vocabulary.txt"
        path.write_text("# from reading, most important first\nPostgres\n\n  Kubernetes   operator  # a phrase\n")
        assert supplied_terms(path) == ["Postgres", "Kubernetes operator"]

    def test_a_missing_file_is_no_terms(self, tmp_path):
        from mergescribe.vocabulary import supplied_terms

        assert supplied_terms(tmp_path / "nope.txt") == []

    def test_known_terms_put_learned_first_and_repeat_nothing(self, tmp_path):
        from mergescribe.vocabulary import known_terms

        corrections = tmp_path / "corrections.jsonl"
        corrections.write_text("\n".join(json.dumps(r) for r in [
            row("s1", "thoughts about cloud today", "thoughts about Claude today"),
            row("s2", "as good as cloud is", "as good as Claude is")]))
        supplied = tmp_path / "vocabulary.txt"
        supplied.write_text("claude\nPostgres\none two three four five six seven\n")
        assert known_terms(corrections, supplied) == ["Claude", "Postgres"]   # 7 words is too long a term

    def test_the_prompt_takes_only_the_head_of_a_long_list(self):
        from mergescribe.vocabulary import MAX_PROMPT_TERMS, supplied_prompt

        prompt = supplied_prompt([f"term{i}" for i in range(MAX_PROMPT_TERMS + 50)])
        assert f"term{MAX_PROMPT_TERMS - 1}," in prompt or f"term{MAX_PROMPT_TERMS - 1}." in prompt
        assert f"term{MAX_PROMPT_TERMS}" not in prompt
        assert supplied_prompt([]) == ""


class TestAnyScript:
    """Letters are letters in every script, not only a to z."""

    def test_distinct_non_latin_corrections_stay_distinct(self):
        from mergescribe.vocabulary import repeated_corrections

        rows = [row("s1", "去北经", "去北京"), row("s2", "在上还", "在上海")]
        learned = repeated_corrections(rows, min_sessions=1)
        assert sorted(after for _, after, _ in learned) == ["去北京", "在上海"]

    def test_a_case_only_change_is_not_a_correction_in_any_script(self):
        from mergescribe.feedback import is_app_transform

        assert is_app_transform("Café Ümlaut", "café ümlaut")
        assert not is_app_transform("去北经", "去北京")
