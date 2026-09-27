"""Tests for knowing, and returning to, the window a dictation started in."""

from unittest.mock import Mock, patch

from mergescribe import context
from mergescribe.context import Origin


def _frontmost(pid):
    app = Mock()
    app.processIdentifier.return_value = pid
    workspace = Mock()
    workspace.frontmostApplication.return_value = app
    # PyObjC won't let a selector be mocked; the class is looked up at call time.
    return patch("AppKit.NSWorkspace", Mock(sharedWorkspace=Mock(return_value=workspace)))


class TestAtOrigin:
    def test_the_same_app_and_window(self):
        window = object()
        with _frontmost(7), patch.object(context, "_ax", return_value=window):
            assert context.at_origin(Origin(pid=7, app=object(), window=window))

    def test_another_app(self):
        with _frontmost(8):
            assert not context.at_origin(Origin(pid=7, app=object(), window=None))

    def test_another_window_of_the_same_app(self):
        """Two Chrome windows: the one being read is not the chat."""
        with _frontmost(7), patch.object(context, "_ax", return_value=object()):
            assert not context.at_origin(Origin(pid=7, app=object(), window=object()))


class TestReturnTo:
    def test_brings_the_window_back_and_waits_for_it(self):
        window = object()
        answers = iter([False, False, False, True])
        with patch.object(context, "at_origin", side_effect=lambda origin: next(answers)), \
             patch("ApplicationServices.AXUIElementSetAttributeValue") as set_attribute, \
             patch("ApplicationServices.AXUIElementPerformAction") as act, \
             patch.object(context, "_RETURN_SETTLE", 0.0):
            assert context.return_to(Origin(pid=7, app=object(), window=window)) is True
        assert [c.args[1] for c in set_attribute.call_args_list] == ["AXFrontmost", "AXMain"]
        act.assert_called_once_with(window, "AXRaise")

    def test_a_window_that_never_comes_back_is_reported(self):
        """Closed since: the caller must not type into whatever is in front."""
        with patch.object(context, "at_origin", return_value=False), \
             patch("ApplicationServices.AXUIElementSetAttributeValue"), \
             patch("ApplicationServices.AXUIElementPerformAction"), \
             patch.object(context, "_RETURN_TIMEOUT", 0.1):
            assert context.return_to(Origin(pid=7, app=object(), window=object())) is False

    def test_already_there_touches_nothing(self):
        with patch.object(context, "at_origin", return_value=True), \
             patch("ApplicationServices.AXUIElementSetAttributeValue") as set_attribute:
            assert context.return_to(Origin(pid=7, app=object(), window=object())) is True
        set_attribute.assert_not_called()
