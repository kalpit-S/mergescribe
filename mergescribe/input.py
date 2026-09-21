"""
Input controller for hotkey state machine.

Handles hold-to-record, double-tap-to-toggle, and emergency reset.
"""

import threading
import time
from typing import Optional, Callable, Literal

from .config import Config


InputState = Literal["idle", "recording", "toggle_recording"]


class InputController:
    """
    Translates raw key events into recording intents.

    Handles:
    - Hold-to-record: Press and hold trigger key
    - Double-tap-to-toggle: Quick double-tap starts continuous recording
    - Emergency reset: Shift+Esc always stops everything

    Usage:
        controller = InputController(config)
        controller.on_start_recording = start_fn
        controller.on_stop_recording = stop_fn

        # Wire to pynput listener
        listener = keyboard.Listener(
            on_press=controller.on_key_press,
            on_release=controller.on_key_release
        )
    """

    def __init__(self, config: Config):
        self.config = config
        self.state: InputState = "idle"
        self.last_press_time: float = 0.0
        self._lock = threading.Lock()
        self._toggle_timer: Optional[threading.Timer] = None
        self._pending_stop_timer: Optional[threading.Timer] = None
        self._awaiting_second_tap: bool = False
        self._trigger_press_started_at: Optional[float] = None

        # Callbacks
        self.on_start_recording: Optional[Callable[[], None]] = None
        self.on_stop_recording: Optional[Callable[[], None]] = None
        self.on_emergency_reset: Optional[Callable[[], None]] = None

        # Key tracking
        self._trigger_key_pressed = False
        self._shift_pressed = False

    def on_key_press(self, key) -> None:
        """
        Handle key press events.

        Args:
            key: pynput key object
        """
        from pynput.keyboard import Key

        # Track modifier keys
        if key == Key.shift or key == Key.shift_r:
            self._shift_pressed = True
            return

        # Check for emergency reset (Shift + Esc)
        if key == Key.esc and self._shift_pressed:
            self._emergency_reset()
            return

        # Check for trigger key (F17 or configurable)
        if not self._is_trigger_key(key):
            return

        with self._lock:
            if self._trigger_key_pressed:
                return  # Already pressed, ignore

            self._trigger_key_pressed = True
            now = time.time()
            self._trigger_press_started_at = now

            if self.state == "toggle_recording":
                # In toggle mode, press stops recording (always, regardless of timing)
                self._stop_recording()
            elif (
                self.state == "recording"
                and self._awaiting_second_tap
                and now - self.last_press_time < self.config.double_tap_threshold
            ):
                # Second tap within threshold -> enter toggle mode without stopping/processing in between
                self._enter_toggle_mode()
            elif self.state == "idle":
                self._start_recording()

            self.last_press_time = now

    def on_key_release(self, key) -> None:
        """
        Handle key release events.

        Args:
            key: pynput key object
        """
        from pynput.keyboard import Key

        # Track modifier keys
        if key == Key.shift or key == Key.shift_r:
            self._shift_pressed = False
            return

        # Check for trigger key
        if not self._is_trigger_key(key):
            return

        with self._lock:
            self._trigger_key_pressed = False
            now = time.time()

            if self.state == "recording":
                # If this was a short tap, wait for the double-tap window to expire
                # before stopping. This prevents a first tap from triggering
                # processing and blocking the second tap.
                press_started_at = self._trigger_press_started_at or now
                press_duration = now - press_started_at
                if press_duration >= self.config.double_tap_threshold:
                    self._stop_recording()
                else:
                    remaining = self.config.double_tap_threshold - press_duration
                    self._arm_pending_stop(remaining)
            # Toggle mode: release does nothing
            self._trigger_press_started_at = None

    def _is_trigger_key(self, key) -> bool:
        """Check if key is the recording trigger."""
        from pynput.keyboard import Key, KeyCode

        # Get configured trigger key (e.g., "alt_r", "f17")
        trigger_name = self.config.trigger_key

        # Check if it's a named Key attribute
        if hasattr(Key, trigger_name):
            if key == getattr(Key, trigger_name):
                return True

        # Also handle special cases for F17 as KeyCode (vk code 64 on macOS)
        if trigger_name.lower() == "f17" and isinstance(key, KeyCode):
            if hasattr(key, 'vk') and key.vk == 64:
                return True

        return False

    def _start_recording(self) -> None:
        """Start recording (must hold lock)."""
        self._awaiting_second_tap = False
        self._cancel_pending_stop_timer()
        self.state = "recording"
        if self.on_start_recording:
            self.on_start_recording()

    def _stop_recording(self) -> None:
        """Stop recording (must hold lock)."""
        self.state = "idle"
        self._awaiting_second_tap = False
        self._cancel_pending_stop_timer()
        self._cancel_toggle_timer()
        if self.on_stop_recording:
            self.on_stop_recording()

    def _enter_toggle_mode(self) -> None:
        """Enter toggle recording mode (must hold lock)."""
        if self.state == "recording":
            # Transition from hold-mode recording into toggle recording without restarting
            self.state = "toggle_recording"
            self._awaiting_second_tap = False
            self._cancel_pending_stop_timer()
        elif self.state == "idle":
            # Allow entering toggle from idle for callers, but on_key_press should
            # generally start recording first.
            self.state = "toggle_recording"
            self._awaiting_second_tap = False
            self._cancel_pending_stop_timer()
            if self.on_start_recording:
                self.on_start_recording()
        else:
            return

        # Safety timeout
        self._cancel_toggle_timer()
        self._toggle_timer = threading.Timer(
            self.config.toggle_mode_timeout,
            self._toggle_timeout
        )
        self._toggle_timer.daemon = True
        self._toggle_timer.start()

    def _toggle_timeout(self) -> None:
        """Called when toggle mode times out."""
        with self._lock:
            if self.state == "toggle_recording":
                print("Toggle mode timeout - stopping recording")
                self._stop_recording()

    def _cancel_toggle_timer(self) -> None:
        """Cancel any pending toggle timeout."""
        if self._toggle_timer:
            self._toggle_timer.cancel()
            self._toggle_timer = None

    def _arm_pending_stop(self, delay_seconds: float) -> None:
        """Arm a delayed stop to allow a double-tap to convert to toggle mode."""
        self._awaiting_second_tap = True
        self._cancel_pending_stop_timer()
        self._pending_stop_timer = threading.Timer(delay_seconds, self._pending_stop_timeout)
        self._pending_stop_timer.daemon = True
        self._pending_stop_timer.start()

    def _pending_stop_timeout(self) -> None:
        """Called when the double-tap window expires after a short tap."""
        with self._lock:
            self._pending_stop_timer = None
            if self.state == "recording" and self._awaiting_second_tap:
                self._stop_recording()

    def _cancel_pending_stop_timer(self) -> None:
        """Cancel any pending delayed stop."""
        if self._pending_stop_timer:
            self._pending_stop_timer.cancel()
            self._pending_stop_timer = None

    def _emergency_reset(self) -> None:
        """Emergency reset - stop everything."""
        with self._lock:
            self.state = "idle"
            self._trigger_key_pressed = False
            self._cancel_toggle_timer()
            self._cancel_pending_stop_timer()
            self._awaiting_second_tap = False
            self._trigger_press_started_at = None

        if self.on_emergency_reset:
            self.on_emergency_reset()
        elif self.on_stop_recording:
            self.on_stop_recording()

        print("Emergency reset triggered")
