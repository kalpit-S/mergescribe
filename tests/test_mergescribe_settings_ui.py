"""
Tests for the pure helpers behind the settings window.

The window itself needs a display, but the number formatting and parsing it
relies on can be checked directly, and that is where a silent bug lived.
"""


class TestNumberRendering:
    def test_whole_numbers_keep_their_zeros(self):
        """450 used to render as "45"."""
        from mergescribe.ui.settings import _trim

        assert _trim(450, 0) == "450"
        assert _trim(10, 0) == "10"
        assert _trim(100, 0) == "100"

    def test_trailing_decimal_zeros_are_dropped(self):
        from mergescribe.ui.settings import _trim

        assert _trim(2.5, 2) == "2.5"
        assert _trim(2.0, 2) == "2"
        assert _trim(12.0, 1) == "12"

    def test_showing_then_parsing_a_value_never_changes_it(self):
        """Tabbing through a field must save exactly what was already there."""
        from mergescribe.ui.settings import _clamped, _trim

        for value, lo, hi, places in (
            (450, 150, 1000, 0),     # double-tap window, milliseconds
            (1.2, 0.2, 10.0, 2),     # split after silence, seconds
            (12.0, 1.0, 30.0, 1),    # speech threshold, dB
            (0.25, 0.2, 10.0, 2),
        ):
            assert _clamped(_trim(value, places), lo, hi) == value


class TestClamp:
    def test_clamps_into_range(self):
        from mergescribe.ui.settings import _clamped

        assert _clamped("99", 0.2, 10.0) == 10.0
        assert _clamped("-5", 0.2, 10.0) == 0.2
        assert _clamped(" 3.5 ", 0.2, 10.0) == 3.5

    def test_rejects_garbage_instead_of_guessing(self):
        from mergescribe.ui.settings import _clamped

        assert _clamped("not a number", 0.0, 1.0) is None
        assert _clamped("", 0.0, 1.0) is None
        assert _clamped(None, 0.0, 1.0) is None


class TestListParsing:
    def test_model_ids_split_on_spaces_and_newlines(self):
        from mergescribe.ui.settings import _words

        assert _words("a/b c/d\ne/f  a/b") == ["a/b", "c/d", "e/f"]

    def test_lines_drop_blanks(self):
        from mergescribe.ui.settings import _lines

        assert _lines("Slack\n\n  Claude \n") == ["Slack", "Claude"]
