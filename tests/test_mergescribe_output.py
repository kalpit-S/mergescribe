

class TestDictationFilter:
    """Line breaks must never reach the keyboard: Return sends in chat apps."""

    def test_flattens_breaks_across_token_boundaries(self):
        from mergescribe.output import DictationFilter

        f = DictationFilter()
        out = "".join(f.feed(t) for t in ["Okay cool.", "\n\n", "Also, it'd be"])
        assert out == "Okay cool. Also, it'd be"

    def test_collapses_a_break_next_to_existing_spaces(self):
        from mergescribe.output import DictationFilter

        f = DictationFilter()
        assert "".join(f.feed(t) for t in ["one ", "\n\n", " two"]) == "one two"

    def test_swallows_leading_whitespace(self):
        """Dictation shouldn't start by nudging the cursor along."""
        from mergescribe.output import DictationFilter

        assert DictationFilter().feed("   hello") == "hello"

    def test_handles_carriage_returns(self):
        from mergescribe.output import DictationFilter

        assert DictationFilter().feed("a\r\nb") == "a b"

    def test_passes_ordinary_text_through_untouched(self):
        from mergescribe.output import DictationFilter

        text = "Nothing special here, just a sentence."
        assert DictationFilter().feed(text) == text

    def test_empty_tokens_are_safe(self):
        from mergescribe.output import DictationFilter

        f = DictationFilter()
        assert f.feed("") == ""
        assert f.feed("\n") == ""


class TestNothingMarker:
    """The model's "type nothing" reply must be recognized without delaying real text."""

    def test_ordinary_text_passes_on_the_first_token(self):
        from mergescribe.output import NothingMarker

        gate = NothingMarker()
        assert gate.feed("Ship ") == "Ship "
        assert gate.feed("it.") == "it."
        assert not gate.called_off

    def test_the_marker_split_across_tokens_is_caught(self):
        from mergescribe.output import NothingMarker

        gate = NothingMarker()
        assert [gate.feed(t) for t in ("[no", "th", "ing]")] == ["", "", ""]
        assert gate.called_off
        assert gate.flush() == ""

    def test_case_and_surrounding_space_do_not_matter(self):
        from mergescribe.output import NothingMarker, is_nothing

        gate = NothingMarker()
        gate.feed("  [Nothing]")
        assert gate.called_off
        assert is_nothing(" [NOTHING] ")
        assert not is_nothing("nothing to see here")

    def test_bracketed_text_is_released_once_it_diverges(self):
        from mergescribe.output import NothingMarker

        gate = NothingMarker()
        assert gate.feed("[no") == ""
        assert gate.feed("te] fix") == "[note] fix"
        assert not gate.called_off

    def test_a_reply_that_stops_mid_prefix_is_released(self):
        from mergescribe.output import NothingMarker

        gate = NothingMarker()
        gate.feed("[not")
        assert gate.flush() == "[not"

    def test_the_default_prompt_asks_for_the_marker_this_gate_expects(self):
        from mergescribe.correct import DEFAULT_SYSTEM_CONTEXT
        from mergescribe.output import NOTHING_MARKER

        assert f"exactly {NOTHING_MARKER}" in DEFAULT_SYSTEM_CONTEXT
