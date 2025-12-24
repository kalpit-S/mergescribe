import threading
import time
from typing import Tuple

from fast_text_input import copy_result_to_clipboard, type_fast
from config_manager import ConfigManager
from text_editing import replace_selected_text

from .audio import prepare_audio_data
from .correction import (
    correct_with_openrouter,
    correct_with_openrouter_streaming,
    edit_text_with_openrouter,
    get_application_context,
)
from .history import build_context_from_history
from .providers import collect_transcriptions, compare_transcriptions
from .state import get_global_state, is_debug


def process_in_thread() -> None:
    """Spawn a background thread to process the current recording."""
    state = get_global_state()
    alive_threads = []
    for thread in state.active_threads:
        if thread.is_alive():
            alive_threads.append(thread)
        elif is_debug():
            print(f"[DEBUG] Cleaning up completed thread: {thread}")
    state.active_threads = alive_threads

    if is_debug():
        print(f"[DEBUG] Active threads before starting new one: {len(state.active_threads)}")

    thread = threading.Thread(target=_process_recording, daemon=True)
    state.active_threads.append(thread)
    thread.start()

    if is_debug():
        print(f"[DEBUG] Started new processing thread, total active: {len(state.active_threads)}")


def _process_recording() -> None:
    """Process the buffered audio, dispatching to providers and post-processing."""
    state = get_global_state()
    if is_debug():
        print("[DEBUG] _process_recording started")

    try:
        processing_start_time = time.time()

        if not state.audio_buffer:
            print("No audio recorded.")
            return

        audio_data, preprocessing_time = prepare_audio_data()
        is_text_editing_mode = state.selected_text is not None
        _print_mode_info(is_text_editing_mode, state)

        transcriptions, provider_timings = collect_transcriptions(audio_data, state)
        if not transcriptions:
            print("❌ No transcriptions received.")
            return

        if len(transcriptions) > 1:
            compare_transcriptions(transcriptions)

        final_result, openrouter_time, mode_description = _process_results(
            transcriptions,
            is_text_editing_mode,
            state,
        )

        _show_results(
            final_result,
            processing_start_time,
            preprocessing_time,
            provider_timings,
            openrouter_time,
            is_text_editing_mode,
            state,
            mode_description,
        )

        print(f"Result: {final_result}")
    except Exception as exc:  # noqa: BLE001 - unexpected processing failure
        print(f"Error processing recording: {exc}")
        if is_debug():
            import traceback

            traceback.print_exc()
    finally:
        _cleanup_processing_thread()


def _print_mode_info(is_text_editing_mode: bool, state) -> None:
    if is_text_editing_mode:
        print("📝 Text editing mode detected")
    else:
        print("🎤 Normal transcription mode")

    if is_debug():
        print(f"[DEBUG] Processing {len(state.audio_buffer)} audio chunks")
        if is_text_editing_mode and state.selected_text:
            preview = state.selected_text[:100]
            if len(state.selected_text) > 100:
                preview += "..."
            print(f"[DEBUG] Selected text: \"{preview}\"")


def _process_results(transcriptions, is_text_editing_mode: bool, state):
    if is_text_editing_mode:
        return _handle_text_editing(transcriptions, state.selected_text)
    return _handle_normal_transcription(transcriptions, state)


def _handle_text_editing(transcriptions, selected_text: str) -> Tuple[str, float, str]:
    voice_command = transcriptions[0][1] if transcriptions else ""

    if is_debug():
        print(f"[DEBUG] Voice command: \"{voice_command}\"")

    editing_start = time.time()
    edited_text = edit_text_with_openrouter(selected_text, voice_command)
    editing_time = time.time() - editing_start

    replace_selected_text(edited_text)

    return edited_text, editing_time, f'Text Editing: "{voice_command}"'


def _handle_normal_transcription(transcriptions, state):
    if state.turbo_mode:
        # Turbo mode: skip OpenRouter entirely and just type the best raw provider output.
        best_transcription = transcriptions[0][1]
        openrouter_time = 0.0
        if is_debug():
            print(f"[DEBUG] Turbo mode: using raw result from {transcriptions[0][0]}")

        state.transcription_history.append((time.time(), best_transcription))
        type_fast(best_transcription)
        if bool(ConfigManager().get_value("AUTO_COPY_RESULT_TO_CLIPBOARD")):
            # Convenience: let users paste instantly if an app blocks synthetic typing.
            copy_result_to_clipboard(best_transcription)
        return best_transcription, openrouter_time, "Transcription"

    # Normal mode: stream correction from OpenRouter and type as tokens arrive.
    context = build_context_from_history(state.transcription_history)
    app_context = get_application_context()

    if is_debug() and app_context:
        print(f"[DEBUG] Application context: {app_context}")

    openrouter_start = time.time()

    # Streamed correction: this will call type_fast(delta) for each incoming
    # chunk so the OS sees a natural, incremental typing pattern.
    best_transcription = correct_with_openrouter_streaming(
        transcriptions,
        context,
        app_context,
        on_delta=type_fast,
    )

    # In case streaming returned an empty result (e.g., error + fallback),
    # fall back to the best raw provider output for history/metrics.
    if not best_transcription or not best_transcription.strip():
        if is_debug():
            print(
                "[DEBUG] Streaming correction returned empty result; "
                "falling back to best raw provider output"
            )
        best_transcription = transcriptions[0][1]

    openrouter_time = time.time() - openrouter_start

    # We have already typed via streaming; just record the result in history.
    state.transcription_history.append((time.time(), best_transcription))
    if bool(ConfigManager().get_value("AUTO_COPY_RESULT_TO_CLIPBOARD")):
        # Convenience: let users paste instantly if an app blocks synthetic typing.
        copy_result_to_clipboard(best_transcription)

    return best_transcription, openrouter_time, "Transcription"


def _show_results(
    final_result: str,
    processing_start_time: float,
    preprocessing_time: float,
    provider_timings,
    openrouter_time: float,
    is_text_editing_mode: bool,
    state,
    mode_description: str,
) -> None:
    total_time = time.time() - state.recording_start_time
    recording_time = processing_start_time - state.recording_start_time
    processing_time = time.time() - processing_start_time

    word_count = len(final_result.split())
    speech_wpm_recording = int((word_count / recording_time) * 60) if recording_time > 0 else 0
    throughput_wpm_total = int((word_count / total_time) * 60) if total_time > 0 else 0

    if is_debug():
        _show_debug_timing(
            is_text_editing_mode,
            recording_time,
            preprocessing_time,
            provider_timings,
            openrouter_time,
            processing_time,
            total_time,
            speech_wpm_recording,
            throughput_wpm_total,
        )
    else:
        _show_clean_results(
            is_text_editing_mode,
            state,
            provider_timings,
            openrouter_time,
            recording_time,
            total_time,
            speech_wpm_recording,
            throughput_wpm_total,
            final_result,
            mode_description,
        )


def _show_debug_timing(
    is_text_editing_mode: bool,
    recording_time: float,
    preprocessing_time: float,
    provider_timings,
    openrouter_time: float,
    processing_time: float,
    total_time: float,
    speech_wpm_recording: int,
    throughput_wpm_total: int,
) -> None:
    mode_title = "TEXT EDITING" if is_text_editing_mode else "TRANSCRIPTION"
    print(f"\n╭───────────────────── {mode_title} TIMING ─────────────────────╮")
    print(f"│ Recording time:    {recording_time:.2f}s")
    print(f"│ Preprocessing:     {preprocessing_time:.2f}s")

    for provider, timing in provider_timings.items():
        print(f"│ {provider:<16} {timing:.2f}s")

    label = "Text editing:" if is_text_editing_mode else "Correction:"
    print(f"│ {label:<16} {openrouter_time:.2f}s")
    print(f"│ Processing time:   {processing_time:.2f}s")
    print(f"│ Total time:        {total_time:.2f}s")
    print(f"│ Speech rate (talk): {speech_wpm_recording} words per minute")
    print(f"│ Throughput (e2e):  {throughput_wpm_total} words per minute")
    print("╰───────────────────────────────────────────────────────────────╯")
    print("[DEBUG] _process_recording completed successfully")


def _show_clean_results(
    is_text_editing_mode: bool,
    state,
    provider_timings,
    openrouter_time: float,
    recording_time: float,
    total_time: float,
    speech_wpm_recording: int,
    throughput_wpm_total: int,
    final_result: str,
    mode_description: str,
) -> None:
    if is_text_editing_mode:
        print("✏️  Text Editing Results:")
    elif state.turbo_mode:
        print("⚡ TURBO MODE Results:")
    else:
        print("🎤 Transcription Results:")

    for provider, timing in sorted(provider_timings.items(), key=lambda item: item[1]):
        provider_name = provider.replace("providers.", "").replace("_", " ").title()
        print(f"   • {provider_name}: {timing:.2f}s")

    if is_text_editing_mode:
        print(f"   • Text editing: {openrouter_time:.2f}s")
    elif not state.turbo_mode:
        print(f"   • Correction: {openrouter_time:.2f}s")

    print(f"   • Recording time: {recording_time:.2f}s")
    print(f"   • Total time: {total_time:.2f}s")
    print(f"   • Speech rate (talk): {speech_wpm_recording} words/min")
    print(f"   • Throughput (e2e): {throughput_wpm_total} words/min")

    if is_text_editing_mode:
        print(f"   ✏️  {mode_description} → \"{final_result}\"")
    elif state.turbo_mode:
        print(f"   ⚡ RAW (no correction) → \"{final_result}\"")
    else:
        print(f"   → \"{final_result}\"")


def _cleanup_processing_thread() -> None:
    state = get_global_state()
    current_thread = threading.current_thread()
    if current_thread in state.active_threads:
        state.active_threads.remove(current_thread)

    state.selected_text = None

    if is_debug():
        print(
            f"[DEBUG] Removed current thread from active threads, remaining: "
            f"{len(state.active_threads)}"
        )
        print("[DEBUG] Recording state reset in _process_recording finally block")
