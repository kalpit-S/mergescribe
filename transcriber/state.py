import threading
import time
from typing import Optional

from config_manager import ConfigManager


class GlobalState:
    """Holds mutable runtime state for the transcription workflow."""

    def __init__(self) -> None:
        self.recording_in_progress = False
        self.audio_buffer = []
        self.transcription_history = []
        self.turbo_mode = False
        self.processing_lock = threading.Lock()
        self.active_threads = []
        self.stream = None
        self.selected_text: Optional[str] = None
        self.audio_hardware_busy = False
        self.current_sample_rate = None
        self.current_channels = None
        self.current_device_index = None
        self.last_mic_name = None
        self.recording_start_time = None


_state = GlobalState()


def get_global_state() -> GlobalState:
    return _state


def is_debug() -> bool:
    return bool(ConfigManager().get_value("DEBUG_MODE"))


def reset_state() -> None:
    """Reset global state, stopping audio streams and clearing buffers."""
    state = get_global_state()
    with state.processing_lock:
        state.recording_in_progress = False
        state.audio_hardware_busy = False
        state.audio_buffer = []
        state.selected_text = None
        state.transcription_history = []
        state.current_sample_rate = None
        state.current_channels = None
        state.current_device_index = None
        state.last_mic_name = None
        state.recording_start_time = None

        if state.stream:
            try:
                state.stream.stop()
                state.stream.close()
            except Exception as exc:  # noqa: BLE001 - we log and continue cleanup
                print(f"Error closing audio stream: {exc}")
            finally:
                state.stream = None

        for thread in state.active_threads:
            if thread.is_alive():
                print("Warning: Thread still running during reset")
        state.active_threads = []


def cleanup() -> None:
    reset_state()
    time.sleep(0.1)
