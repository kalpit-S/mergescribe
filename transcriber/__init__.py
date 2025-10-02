"""High-level interface for the transcription workflow."""

from text_editing import detect_selected_text

from . import audio
from .audio import get_audio_as_bytes  # noqa: F401
from .correction import (  # noqa: F401
    build_correction_prompt,
    correct_with_openrouter,
    edit_text_with_openrouter,
    get_application_context,
)
from .history import build_context_from_history  # noqa: F401
from .processing import process_in_thread
from .state import GlobalState, cleanup, get_global_state, is_debug, reset_state

__all__ = [
    "GlobalState",
    "build_context_from_history",
    "build_correction_prompt",
    "cleanup",
    "configure_pyautogui",
    "correct_with_openrouter",
    "detect_and_store_selected_text",
    "edit_text_with_openrouter",
    "get_application_context",
    "get_audio_as_bytes",
    "get_global_state",
    "is_debug",
    "process_in_thread",
    "start_recording",
    "stop_recording_and_process",
    "reset_state",
]


def configure_pyautogui() -> None:
    import pyautogui

    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.01


def detect_and_store_selected_text() -> None:
    state_obj = get_global_state()
    state_obj.selected_text = detect_selected_text()

    if is_debug() and state_obj.selected_text:
        preview = state_obj.selected_text[:50]
        ellipsis = "..." if len(state_obj.selected_text) > 50 else ""
        print(f"[DEBUG] Selected text for editing: \"{preview}{ellipsis}\"")


def start_recording() -> None:
    audio.start_recording()


def stop_recording_and_process() -> None:
    state_obj = get_global_state()
    if is_debug():
        print("[DEBUG] stop_recording_and_process called")

    with state_obj.processing_lock:
        if is_debug():
            print("[DEBUG] Acquired processing lock")
            if not state_obj.recording_in_progress:
                print(
                    "[DEBUG] Warning: stop_recording_and_process called but "
                    "recording_in_progress is already False"
                )
        state_obj.recording_in_progress = False
        audio.stop_active_stream()
        if is_debug():
            print("[DEBUG] Released processing lock, starting background processing")

    process_in_thread()

    if is_debug():
        print("[DEBUG] stop_recording_and_process completed")
