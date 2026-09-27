"""Tests for turning other audio down while dictating through speakers."""

import pytest


class FakeOutput:
    """A default output device: its kind, and a volume that may step."""

    def __init__(self, transport="bltn", source="ispk", volume=0.5, step=0.0):
        self.kind_ = (transport, source)
        self.level = volume
        self.step = step

    def device(self):
        return 76

    def kind(self, device):
        return self.kind_

    def volume(self, device):
        return self.level

    def set_volume(self, device, value):
        self.level = round(value / self.step) * self.step if self.step else value


@pytest.mark.parametrize("transport, source, loud", [
    ("bltn", "ispk", True),    # the MacBook's own speakers
    ("bltn", "hdpn", False),   # something in its headphone jack
    ("hdmi", "", True),        # a display's speakers
    ("dprt", "", True),
    ("airp", "", True),        # AirPlay
    ("blue", "", False),       # AirPods, and most Bluetooth output
    ("usb ", "", False),       # a headset or an interface: a guess either way, so leave it
])
def test_what_counts_as_playing_out_loud(transport, source, loud):
    from mergescribe.ducking import plays_out_loud

    assert plays_out_loud(transport, source) is loud


def test_speakers_are_turned_down_and_back_up():
    from mergescribe.ducking import DUCK_TO, Ducker

    output = FakeOutput(volume=0.6)
    ducker = Ducker(output)
    ducker.duck()
    assert output.level == pytest.approx(0.6 * DUCK_TO)
    ducker.restore()
    assert output.level == pytest.approx(0.6)


def test_headphones_are_left_alone():
    from mergescribe.ducking import Ducker

    output = FakeOutput(transport="blue", volume=0.6)
    Ducker(output).duck()
    assert output.level == 0.6


def test_a_volume_changed_mid_dictation_is_kept():
    """Reaching for the volume while talking is a choice; restoring over it would undo it."""
    from mergescribe.ducking import Ducker

    output = FakeOutput(volume=0.6)
    ducker = Ducker(output)
    ducker.duck()
    output.level = 0.9
    ducker.restore()
    assert output.level == 0.9


def test_a_device_that_steps_its_volume_still_restores():
    from mergescribe.ducking import Ducker

    output = FakeOutput(volume=0.5, step=1 / 16)     # 0.125 requested, 0.125 given; 0.1 would round
    ducker = Ducker(output)
    ducker.duck()
    ducker.restore()
    assert output.level == 0.5


def test_ducking_twice_restores_the_original():
    """A toggle-mode press while already ducked must not duck the ducked volume."""
    from mergescribe.ducking import Ducker

    output = FakeOutput(volume=0.8)
    ducker = Ducker(output)
    ducker.duck()
    ducker.duck()
    ducker.restore()
    assert output.level == pytest.approx(0.8)


def test_muted_or_missing_output_is_a_no_op():
    from mergescribe.ducking import Ducker

    silent = FakeOutput(volume=0.0)
    Ducker(silent).duck()
    assert silent.level == 0.0
    Ducker(output=None).restore()      # never raises


def test_the_transcription_catalogue_is_parsed_and_unreachable_is_empty(monkeypatch):
    from unittest.mock import Mock

    import requests

    from mergescribe.ui.settings_store import fetch_transcription_models

    reply = Mock()
    reply.json.return_value = {"data": [{"id": "microsoft/mai-transcribe-2", "name": "Microsoft AI: MAI-Transcribe 2"},
                                        {"id": "x/plain"}]}
    monkeypatch.setattr(requests, "get", lambda *a, **k: reply)
    assert fetch_transcription_models() == [("microsoft/mai-transcribe-2", "MAI-Transcribe 2"), ("x/plain", "x/plain")]

    def offline(*a, **k):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(requests, "get", offline)
    assert fetch_transcription_models() == []
