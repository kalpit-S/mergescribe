import io
import time
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf

from audio_processing import preprocess_audio
from config_manager import ConfigManager

from .state import get_global_state, is_debug


def start_recording() -> None:
    """Initialize the audio stream and begin buffering microphone input."""
    state = get_global_state()
    if is_debug():
        print(
            "[DEBUG] start_recording called, current state: "
            f"recording_in_progress={state.recording_in_progress}"
        )

    state.audio_hardware_busy = True

    if state.stream:
        stop_active_stream()
        time.sleep(0.02)

    state.audio_buffer = []
    state.recording_start_time = time.time()

    def audio_callback(indata, frames, callback_time, status):  # noqa: ARG001 - callback signature
        if status:
            print(f"Audio recording error: {status}")
        if indata is not None and len(indata) > 0:
            state.audio_buffer.append(indata.copy())

    if not hasattr(state, "mic_info_printed"):
        try:
            cfg = ConfigManager()
            selected_device = cfg.get_value("MIC_DEVICE_INDEX")
            if selected_device is not None:
                dev_info = sd.query_devices(selected_device)
            else:
                dev_info = sd.query_devices(kind="input")
            print(f"🎙️  Using microphone: {dev_info['name']}")
            state.mic_info_printed = True
        except Exception as exc:  # noqa: BLE001 - best effort logging
            print(f"Could not detect microphone: {exc}")

    max_retries = 3
    stream_created = False
    cfg = ConfigManager()
    selected_device = cfg.get_value("MIC_DEVICE_INDEX")

    # Validate configured device is available before attempting to use it
    if selected_device is not None:
        if not _is_device_available(selected_device):
            if is_debug():
                print(f"[DEBUG] Configured device {selected_device} not available, falling back to default")
            print(f"⚠️  Configured microphone (device {selected_device}) not found, using system default")
            selected_device = None

    for attempt in range(max_retries):
        try:
            stream, used_sr, used_ch = _open_stream(selected_device, audio_callback)
            state.stream = stream
            state.current_sample_rate = int(used_sr) if used_sr else None
            state.current_channels = int(used_ch) if used_ch else None
            if not is_debug():
                print("🔴 Recording...")
            else:
                print("Recording started...")
                print(f"[DEBUG] Using sample rate {used_sr} Hz, channels={used_ch}")
            stream_created = True
            break
        except Exception as exc:  # noqa: BLE001 - propagate on final failure
            print(f"Audio stream creation attempt {attempt + 1} failed: {exc}")
            if selected_device is not None:
                if is_debug():
                    print("[DEBUG] Retrying with system default microphone")
                selected_device = None
            if attempt < max_retries - 1:
                time.sleep(0.05)
            else:
                print("Failed to start recording after multiple attempts")
                state.stream = None
                state.audio_hardware_busy = False
                state.recording_in_progress = False
                raise

    if not stream_created:
        state.audio_hardware_busy = False
        state.recording_in_progress = False


def stop_active_stream() -> None:
    """Stop and dispose of the current audio stream if it exists."""
    state = get_global_state()
    if not state.stream:
        return

    try:
        if is_debug():
            print("[DEBUG] Stopping audio stream...")
        state.stream.stop()
        time.sleep(0.01)
        if is_debug():
            print("[DEBUG] Closing audio stream...")
        state.stream.close()
        time.sleep(0.01)
        if is_debug():
            print("[DEBUG] Audio stream closed successfully")
    except Exception as exc:  # noqa: BLE001 - logging only
        print(f"Error closing audio stream: {exc}")
    finally:
        state.stream = None
        state.audio_hardware_busy = False


def prepare_audio_data() -> Tuple[np.ndarray, float]:
    """Concatenate buffered chunks and run preprocessing."""
    state = get_global_state()
    audio_data = np.concatenate(state.audio_buffer, axis=0)

    if hasattr(state, "current_sample_rate") and state.current_sample_rate:
        sr_for_processing = state.current_sample_rate
    else:
        sr_for_processing = ConfigManager().get_value("SAMPLE_RATE") or 16000

    preprocessing_start = time.time()
    processed_audio = preprocess_audio(audio_data)
    preprocessing_time = time.time() - preprocessing_start

    return processed_audio, preprocessing_time


def get_audio_as_bytes(audio_data: np.ndarray, sample_rate: Optional[int] = None) -> bytes:
    """Encode numpy audio to WAV bytes."""
    sr = sample_rate or (ConfigManager().get_value("SAMPLE_RATE") or 16000)
    audio_data_int16 = (audio_data * 32767).astype(np.int16)
    buffer = io.BytesIO()
    sf.write(buffer, audio_data_int16, sr, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.getvalue()


def _is_device_available(device_index: int) -> bool:
    """Check if an audio input device is currently available."""
    try:
        devices = sd.query_devices()
        if isinstance(devices, dict):
            devices = [devices]
        if device_index < 0 or device_index >= len(devices):
            return False
        device = devices[device_index]
        return device.get('max_input_channels', 0) > 0
    except Exception:  # noqa: BLE001 - availability check should not crash
        return False


def _open_stream(device_index, audio_callback):
    cfg = ConfigManager()
    requested_sr = cfg.get_value("SAMPLE_RATE") or 16000
    requested_ch = cfg.get_value("CHANNELS") or 1

    default_sr = requested_sr
    max_in_channels = requested_ch
    try:
        dev_info = (
            sd.query_devices(device_index) if device_index is not None else sd.query_devices(kind="input")
        )
        if isinstance(dev_info.get("default_samplerate"), (int, float)):
            default_sr = int(dev_info["default_samplerate"])
        if isinstance(dev_info.get("max_input_channels"), int):
            max_in_channels = dev_info["max_input_channels"]
    except Exception:  # noqa: BLE001 - fall back to defaults silently
        pass

    channels_to_use = max(1, min(int(requested_ch), int(max_in_channels or 1)))

    candidate_srs = []
    for sr_candidate in [requested_sr, default_sr, 48000, 44100, 32000, 24000, 16000]:
        try:
            sr_int = int(sr_candidate)
            if sr_int not in candidate_srs:
                candidate_srs.append(sr_int)
        except Exception:  # noqa: BLE001 - skip invalid values
            continue

    last_exc = None
    for sr_candidate in candidate_srs:
        try:
            stream = sd.InputStream(
                samplerate=sr_candidate,
                channels=channels_to_use,
                callback=audio_callback,
                dtype="float32",
                device=device_index if device_index is not None else None,
            )
            stream.start()
            return stream, sr_candidate, channels_to_use
        except Exception as exc:  # noqa: BLE001 - keep last failure for reporting
            last_exc = exc
            continue

    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown error opening audio stream")
