"""Tests for output-routing target parsing (pure logic, no AX)."""

from mergescribe.fields import TargetStreamParser, parse_target_prefix


class TestParseTargetPrefix:
    def test_target_line_stripped(self):
        target, rest = parse_target_prefix("TARGET: f3\nHello world")
        assert target == "f3"
        assert rest == "Hello world"

    def test_target_focused(self):
        target, rest = parse_target_prefix("TARGET: focused\nHi")
        assert target == "focused"
        assert rest == "Hi"

    def test_no_target(self):
        target, rest = parse_target_prefix("Hello world")
        assert target is None
        assert rest == "Hello world"

    def test_target_mentioned_mid_text_not_parsed(self):
        text = "The word TARGET: f1 appears mid-sentence"
        target, rest = parse_target_prefix(text)
        assert target is None
        assert rest == text

    def test_leading_whitespace(self):
        target, rest = parse_target_prefix("  TARGET: f2\nText")
        assert target == "f2"
        assert rest == "Text"

    def test_target_only_no_newline(self):
        target, rest = parse_target_prefix("TARGET: f1")
        assert target == "f1"
        assert rest == ""


class TestTargetStreamParser:
    def _run(self, tokens):
        targets = []
        parser = TargetStreamParser(on_target=targets.append)
        out = "".join(parser.feed(t) for t in tokens)
        out += parser.flush()
        return targets, out

    def test_target_then_text(self):
        targets, out = self._run(["TARGET:", " f3\n", "Hello", " world"])
        assert targets == ["f3"]
        assert out == "Hello world"

    def test_single_token_stream(self):
        targets, out = self._run(["TARGET: f1\nHi there"])
        assert targets == ["f1"]
        assert out == "Hi there"

    def test_no_target_passthrough(self):
        targets, out = self._run(["Hello", " world, this is a longer sentence without routing."])
        assert targets == [None]
        assert out == "Hello world, this is a longer sentence without routing."

    def test_short_output_without_target(self):
        targets, out = self._run(["Yes."])
        assert targets == [None]
        assert out == "Yes."

    def test_on_target_called_exactly_once(self):
        targets, _ = self._run(["TARGET: focused\n", "a", "b", "c"])
        assert targets == ["focused"]

    def test_text_starting_like_target_word(self):
        # "TAR" prefix-matches TARGET: so it buffers, then resolves as plain text
        targets, out = self._run(["TAR", "DIS is a word\nmore text"])
        assert targets == [None]
        assert out == "TARDIS is a word\nmore text"

    def test_empty_stream(self):
        targets, out = self._run([])
        assert targets == [None]
        assert out == ""


class TestAppAllowed:
    def test_empty_allowlist_allows_all(self):
        from mergescribe.fields import _app_allowed
        assert _app_allowed("Anything", [])
        assert _app_allowed("Anything", None)

    def test_substring_match_case_insensitive(self):
        from mergescribe.fields import _app_allowed
        allowed = ["warp", "Claude", "Slack"]
        assert _app_allowed("Warp", allowed)
        assert _app_allowed("claude", allowed)
        assert not _app_allowed("Finder", allowed)

    def test_partial_names_match_one_directional(self):
        from mergescribe.fields import _app_allowed
        assert _app_allowed("Google Chrome", ["Chrome"])
        # Entry longer than the app name must NOT match the bare app —
        # profile-scoped entries rely on this.
        assert not _app_allowed("Google Chrome", ["Google Chrome - Kalpit"])

    def test_profile_scoped_window_matching(self):
        from mergescribe.fields import _matches
        assert _matches("ChatGPT - Google Chrome - Kalpit", ["Google Chrome - Kalpit"])
        assert not _matches("Dashboard - Google Chrome - Work", ["Google Chrome - Kalpit"])
