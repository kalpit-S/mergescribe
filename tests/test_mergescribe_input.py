import time

import pytest


pytest.importorskip("pynput")

from pynput.keyboard import Key

from mergescribe.config import Config
from mergescribe.input import InputController


def _make_controller(*, threshold: float, toggle_timeout: float = 5.0):
    config = Config()
    config.trigger_key = "alt_r"
    config.double_tap_threshold = threshold
    config.toggle_mode_timeout = toggle_timeout

    events: list[str] = []
    controller = InputController(config)
    controller.on_start_recording = lambda: events.append("start")
    controller.on_stop_recording = lambda: events.append("stop")
    return controller, events


def test_single_short_tap_stops_after_double_tap_window():
    controller, events = _make_controller(threshold=0.08)

    controller.on_key_press(Key.alt_r)
    time.sleep(0.01)
    controller.on_key_release(Key.alt_r)

    assert events == ["start"]
    assert controller.state == "recording"

    time.sleep(0.10)
    assert events == ["start", "stop"]
    assert controller.state == "idle"


def test_hold_stops_immediately_on_release():
    controller, events = _make_controller(threshold=0.05)

    controller.on_key_press(Key.alt_r)
    time.sleep(0.08)
    controller.on_key_release(Key.alt_r)

    assert events == ["start", "stop"]
    assert controller.state == "idle"


def test_double_tap_enters_toggle_and_third_press_stops():
    controller, events = _make_controller(threshold=0.12, toggle_timeout=5.0)

    # Tap 1
    controller.on_key_press(Key.alt_r)
    time.sleep(0.01)
    controller.on_key_release(Key.alt_r)

    # Tap 2 within threshold -> toggle mode, no stop in between
    time.sleep(0.02)
    controller.on_key_press(Key.alt_r)
    controller.on_key_release(Key.alt_r)

    assert events == ["start"]
    assert controller.state == "toggle_recording"

    # Tap 3 stops (even if very soon)
    controller.on_key_press(Key.alt_r)
    assert events == ["start", "stop"]
    assert controller.state == "idle"

    # Ensure no delayed stop fires later
    time.sleep(0.20)
    assert events == ["start", "stop"]

