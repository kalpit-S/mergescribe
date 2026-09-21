"""Tests for post-output edit detection (pure logic, no AX)."""

from mergescribe.feedback import classify, diff_span

TYPED = "Can you check the retry logic for post grass and rerun the migrations?"


def span_of(baseline, typed):
    start = baseline.rfind(typed)
    return (start, start + len(typed))


class TestClassify:
    def test_unchanged(self):
        base = f"draft: {TYPED}"
        assert classify(base, base, span_of(base, TYPED))[0] == "unchanged"

    def test_cleared(self):
        base = TYPED
        assert classify(base, "", span_of(base, TYPED))[0] == "cleared"

    def test_unreadable(self):
        base = TYPED
        assert classify(base, None, span_of(base, TYPED))[0] == "unreadable"

    def test_word_corrected(self):
        base = TYPED
        current = TYPED.replace("post grass", "Postgres")
        outcome, corrected = classify(base, current, span_of(base, TYPED))
        assert outcome == "edited"
        assert "Postgres" in corrected

    def test_typing_after_is_not_an_edit(self):
        """The bug that made the first implementation useless."""
        base = TYPED
        current = TYPED + " Also check the after-hours flow."
        assert classify(base, current, span_of(base, TYPED))[0] == "unchanged"

    def test_typing_before_is_not_an_edit(self):
        base = TYPED
        current = "Earlier note.\n" + TYPED
        assert classify(base, current, span_of(base, TYPED))[0] == "unchanged"

    def test_second_dictation_into_same_field_is_not_an_edit(self):
        """Dictating twice into one field is normal usage, not a correction."""
        base = f"previous sentence. {TYPED}"
        current = base + " A whole new dictated sentence appended here."
        assert classify(base, current, span_of(base, TYPED))[0] == "unchanged"

    def test_edit_survives_surrounding_changes(self):
        base = f"intro. {TYPED} outro."
        current = f"CHANGED INTRO. {TYPED.replace('post grass', 'Postgres')} different outro."
        outcome, corrected = classify(base, current, span_of(base, TYPED))
        assert outcome == "edited"
        assert "Postgres" in corrected


class TestDiffSpan:
    def test_reports_no_change_outside_span(self):
        base = "hello " + TYPED
        current = "goodbye " + TYPED
        changed, _ = diff_span(base, current, span_of(base, TYPED))
        assert changed is False

    def test_reports_change_inside_span(self):
        base = TYPED
        current = TYPED.replace("rerun", "re-run")
        changed, corrected = diff_span(base, current, span_of(base, TYPED))
        assert changed is True
        assert "re-run" in corrected


class TestPlaceholderRejection:
    """A sent message leaves the composer showing its placeholder, not an edit."""

    def _span(self, baseline, typed):
        start = baseline.rfind(typed)
        return (start, start + len(typed))

    def test_placeholder_after_send_is_not_an_edit(self):
        typed = ("And as far as the rollout goes, we already have the staging environment "
                 "set up, so verifying the change should not take very long at all.")
        outcome, corrected = classify(typed, "Ask ChatGPT", self._span(typed, typed))
        assert outcome == "replaced"
        assert corrected == ""

    def test_short_placeholder_variants(self):
        typed = "Yeah, I just set up the new workstation and noticed a few things were broken."
        for placeholder in ("Ask Gemini", "Follow up", "GPT", "Message #general"):
            outcome, _ = classify(typed, placeholder, self._span(typed, typed))
            assert outcome == "replaced", f"{placeholder!r} should not count as an edit"

    def test_genuine_small_edit_still_captured(self):
        typed = "Can you check the retry logic for post grass and rerun the migrations?"
        fixed = typed.replace("post grass", "Postgres")
        outcome, corrected = classify(typed, fixed, self._span(typed, typed))
        assert outcome == "edited"
        assert "Postgres" in corrected

    def test_trimming_half_the_text_is_a_replacement(self):
        typed = "First sentence that is fairly long. Second sentence also long enough."
        outcome, _ = classify(typed, "First sentence", self._span(typed, typed))
        assert outcome == "replaced"


class TestAppTransformRejection:
    """The app rewriting text is not the user correcting it."""

    def _span(self, t):
        return (0, len(t))

    def test_url_encoding_rejected(self):
        typed = "Best TVs to use as monitors in twenty twenty six"
        outcome, _ = classify(typed, typed.replace(" ", "+"), self._span(typed))
        assert outcome == "reformatted"

    def test_case_folding_rejected(self):
        typed = "The pings indicate the requests go to U.S. servers"
        outcome, _ = classify(typed, typed.lower(), self._span(typed))
        assert outcome == "reformatted"

    def test_punctuation_strip_rejected(self):
        typed = "Can you just add a tab to this page?"
        outcome, _ = classify(typed, "Can you just add a tab to this page", self._span(typed))
        assert outcome == "reformatted"

    def test_real_word_change_still_captured(self):
        typed = "Based on the pings the requests are going to US servers"
        fixed = "Based on the pings the requests go to US servers"
        outcome, corrected = classify(typed, fixed, self._span(typed))
        assert outcome == "edited"
        assert "go to" in corrected

    def test_vocabulary_fix_still_captured(self):
        typed = "check the retry logic for post grass today"
        fixed = "check the retry logic for Postgres today"
        outcome, corrected = classify(typed, fixed, self._span(typed))
        assert outcome == "edited"
        assert "Postgres" in corrected
