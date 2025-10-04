import io
import time
from typing import List, Optional, Tuple

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
            (
                stream,
                used_sr,
                used_ch,
                used_device_index,
                used_device_name,
                used_fallback,
            ) = _open_stream(selected_device, audio_callback)
            state.stream = stream
            state.current_sample_rate = int(used_sr) if used_sr else None
            state.current_channels = int(used_ch) if used_ch else None
            state.current_device_index = used_device_index
            if used_device_name:
                previous_mic = state.last_mic_name
                if used_fallback:
                    print(
                        "⚠️  Primary microphone unavailable; using "
                        f"{used_device_name}"
                    )
                if used_fallback or used_device_name != previous_mic:
                    print(f"🎙️  Using microphone: {used_device_name}")
                state.last_mic_name = used_device_name
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

    def _list_input_devices() -> list[int]:
        try:
            devices = sd.query_devices()
        except Exception:  # noqa: BLE001 - best effort enumeration
            return []
        if isinstance(devices, dict):
            devices = [devices]
        indices: list[int] = []
        for idx, dev in enumerate(devices):
            try:
                if dev.get("max_input_channels", 0) > 0:
                    indices.append(idx)
            except Exception:  # noqa: BLE001 - skip malformed entries
                continue
        return indices

    def _query_device(dev_idx):
        try:
            if dev_idx is None:
                return sd.query_devices(kind="input")
            return sd.query_devices(dev_idx)
        except Exception:  # noqa: BLE001 - use empty info on failure
            return {}

    def _candidate_samplerates(dev_info):
        candidates = []
        for sr_candidate in [
            requested_sr,
            dev_info.get("default_samplerate"),
            48000,
            44100,
            32000,
            24000,
            16000,
        ]:
            try:
                sr_int = int(sr_candidate)
            except Exception:  # noqa: BLE001 - skip invalid entries
                continue
            if sr_int > 0 and sr_int not in candidates:
                candidates.append(sr_int)
        return candidates

    def _attempt_device(dev_idx):
        dev_info = _query_device(dev_idx)
        max_in_channels = dev_info.get("max_input_channels")
        try:
            max_in_channels = int(max_in_channels)
        except Exception:  # noqa: BLE001 - fall back to requested channel count
            max_in_channels = requested_ch
        channels_to_use = max(1, min(int(requested_ch), int(max_in_channels or 1)))
        candidate_srs = _candidate_samplerates(dev_info)
        last_exc_local = None

        for sr_candidate in candidate_srs:
            stream_kwargs = {
                "samplerate": sr_candidate,
                "channels": channels_to_use,
                "callback": audio_callback,
                "dtype": "float32",
            }
            if dev_idx is not None:
                stream_kwargs["device"] = dev_idx

            try:
                stream = sd.InputStream(**stream_kwargs)
                try:
                    stream.start()
                except Exception as exc:  # noqa: BLE001 - ensure stream is closed before retry
                    stream.close()
                    last_exc_local = exc
                    continue
                return stream, sr_candidate, channels_to_use, dev_info
            except Exception as exc:  # noqa: BLE001 - keep last error for diagnostics
                last_exc_local = exc
                continue

        if last_exc_local:
            raise last_exc_local
        raise RuntimeError("Unknown error opening audio stream")

    available_inputs = _list_input_devices()

    candidate_devices: List[Optional[int]] = []

    def _add_candidate(dev_idx):
        if dev_idx in candidate_devices:
            return
        candidate_devices.append(dev_idx)

    if device_index is not None and device_index in available_inputs:
        _add_candidate(device_index)
    else:
        device_index = None

    _add_candidate(None)

    try:
        default_input_idx = sd.default.device[0]
        if isinstance(default_input_idx, str):
            default_input_idx = int(default_input_idx)
    except Exception:  # noqa: BLE001 - ignore default lookup failure
        default_input_idx = None

    if (
        default_input_idx is not None
        and isinstance(default_input_idx, int)
        and default_input_idx in available_inputs
    ):
        _add_candidate(default_input_idx)

    for idx in available_inputs:
        _add_candidate(idx)

    if not candidate_devices:
        raise RuntimeError("No input devices available")

    last_exc = None

    for attempt_index, candidate in enumerate(candidate_devices):
        try:
            stream, used_sr, used_ch, info = _attempt_device(candidate)
            device_name = info.get("name") if isinstance(info, dict) else None
            was_fallback = attempt_index > 0
            return stream, used_sr, used_ch, candidate, device_name, was_fallback
        except Exception as exc:  # noqa: BLE001 - try next candidate
            last_exc = exc
            continue

    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown error opening audio stream")
