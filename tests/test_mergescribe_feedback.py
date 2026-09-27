"""Tests for post-output edit detection (pure logic, no AX)."""

import random
import time

from mergescribe.feedback import _opcodes, classify, diff_span

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

    def test_a_field_that_no_longer_exists_is_gone(self):
        base = TYPED
        assert classify(base, None, span_of(base, TYPED))[0] == "gone"

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


class TestLargeFields:
    """A field can hold a whole document; diffing it must stay cheap."""

    def _document(self, words=9000):
        rng = random.Random(1)
        vocab = "the of and to in is you that it was for on are as with they at be this have".split()
        return " ".join(rng.choice(vocab) for _ in range(words))    # ~40k characters

    def test_an_edit_inside_a_long_document_is_found_quickly(self):
        doc = self._document()
        base = doc[:20000] + " " + TYPED + " " + doc[20000:]
        current = base.replace("post grass", "Postgres")
        started = time.perf_counter()
        outcome, corrected = classify(base, current, span_of(base, TYPED))
        assert outcome == "edited" and "Postgres" in corrected
        assert time.perf_counter() - started < 0.5, "a quadratic diff here froze the HUD"

    def test_edits_elsewhere_in_a_long_document_are_not_ours(self):
        doc = self._document()
        base = doc[:20000] + " " + TYPED + " " + doc[20000:]
        current = "A new first line. " + base + " And a new last one."
        assert classify(base, current, span_of(base, TYPED)) == ("unchanged", "")

    def test_a_document_rewritten_everywhere_gives_up_rather_than_stalling(self):
        doc = self._document()
        base = doc[:20000] + " " + TYPED + " " + doc[20000:]
        current = base.replace(" the ", " a ").replace("post grass", "Postgres")
        started = time.perf_counter()
        assert classify(base, current, span_of(base, TYPED)) == ("unmeasured", "")
        assert time.perf_counter() - started < 0.5

    def test_trimmed_opcodes_rebuild_the_new_text(self):
        rng = random.Random(7)
        for _ in range(200):
            a = "".join(rng.choice("ab c") for _ in range(rng.randint(0, 30)))
            b = "".join(rng.choice("ab c") for _ in range(rng.randint(0, 30)))
            rebuilt = "".join(a[i1:i2] if tag == "equal" else b[j1:j2]
                              for tag, i1, i2, j1, j2 in _opcodes(a, b))
            assert rebuilt == b, (a, b)


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
        typed = "Quiet hiking trails near the coast in early spring"
        outcome, _ = classify(typed, typed.replace(" ", "+"), self._span(typed))
        assert outcome == "reformatted"

    def test_case_folding_rejected(self):
        typed = "The trace shows the uploads land on U.K. storage"
        outcome, _ = classify(typed, typed.lower(), self._span(typed))
        assert outcome == "reformatted"

    def test_punctuation_strip_rejected(self):
        typed = "Could you move the chart onto the second slide?"
        outcome, _ = classify(typed, "Could you move the chart onto the second slide", self._span(typed))
        assert outcome == "reformatted"

    def test_real_word_change_still_captured(self):
        typed = "From the trace the backups are going to EU storage"
        fixed = "From the trace the backups go to EU storage"
        outcome, corrected = classify(typed, fixed, self._span(typed))
        assert outcome == "edited"
        assert "go to" in corrected

    def test_vocabulary_fix_still_captured(self):
        typed = "check the retry logic for post grass today"
        fixed = "check the retry logic for Postgres today"
        outcome, corrected = classify(typed, fixed, self._span(typed))
        assert outcome == "edited"
        assert "Postgres" in corrected



class TestReadField:
    """ChatGPT rebuilds its composer on send; Claude clears it in place."""

    def _patch(self, values, focused=None, pids=None):
        from unittest.mock import patch
        import mergescribe.feedback as fb

        return (patch.object(fb, "_ax_get", lambda el, attr: values.get(el)),
                patch.object(fb, "focused_element", lambda: focused),
                patch.object(fb, "_pid", lambda el: (pids or {}).get(el)))

    def _read(self, values, focused=None, pids=None, target="old", pid=100):
        from mergescribe.feedback import read_field

        a, b, c = self._patch(values, focused, pids)
        with a, b, c:
            return read_field(target, pid)

    def test_a_live_field_reads_normally(self):
        assert self._read({"old": "hello"}) == ("old", "hello")

    def test_a_rebuilt_composer_after_send_reads_as_its_successor(self):
        """The fresh composer is empty, so this classifies as cleared - like Claude."""
        element, value = self._read({"old": None, "new": ""}, focused="new",
                                    pids={"new": 100})
        assert (element, value) == ("new", "")
        assert classify(TYPED, value, span_of(TYPED, TYPED))[0] == "cleared"

    def test_a_rerender_mid_edit_still_captures_the_edit(self):
        edited = TYPED.replace("post grass", "Postgres")
        element, value = self._read({"old": None, "new": edited}, focused="new",
                                    pids={"new": 100})
        outcome, corrected = classify(TYPED, value, span_of(TYPED, TYPED))
        assert element == "new" and outcome == "edited" and "Postgres" in corrected

    def test_focus_in_another_app_means_the_field_is_gone(self):
        """Never classify a different app's text against our dictation."""
        element, value = self._read({"old": None, "elsewhere": "unrelated text"},
                                    focused="elsewhere", pids={"elsewhere": 999})
        assert value is None
        assert classify(TYPED, value, span_of(TYPED, TYPED))[0] == "gone"

    def test_nothing_focused_means_gone(self):
        assert self._read({"old": None}, focused=None)[1] is None
