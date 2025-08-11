import concurrent.futures
import io
import subprocess
import threading
import time
from typing import Optional

import numpy as np
import pyautogui
import sounddevice as sd
import soundfile as sf


from config_manager import ConfigManager
from fast_text_input import type_fast
from text_editing import detect_selected_text, replace_selected_text


class GlobalState:
    def __init__(self):
        self.recording_in_progress = False
        self.audio_buffer = []
        self.transcription_history = []
        self.turbo_mode = False
        self.processing_lock = threading.Lock()
        self.active_threads = []
        self.stream = None
        self.selected_text = None
        self.audio_hardware_busy = False

_state = GlobalState()

def get_global_state():
    return _state

def is_debug() -> bool:
    return bool(ConfigManager().get_value("DEBUG_MODE"))

def configure_pyautogui():
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.01

def compare_transcriptions(transcriptions):
    """Print full transcriptions from all providers."""
    if len(transcriptions) < 2:
        return
    
    print("\n📊 Transcription Comparison:")
    print("─" * 60)

    for name, text in transcriptions:
        label = f"{name}"
        print(f"  {label}: {text}")
    
    print("─" * 60)

def detect_and_store_selected_text():
    """
    Detect if text is selected and store it in global state.
    Called when trigger key is pressed.
    """
    state = get_global_state()
    state.selected_text = detect_selected_text()
    
    if is_debug() and state.selected_text:
        print(f"[DEBUG] Selected text for editing: \"{state.selected_text[:50]}{'...' if len(state.selected_text) > 50 else ''}\")")

def start_recording():
    state = get_global_state()
    if is_debug():
        print(f"[DEBUG] start_recording called, current state: recording_in_progress={state.recording_in_progress}")
    
    state.audio_hardware_busy = True
    
    if state.stream:
        try:
            state.stream.stop()
            state.stream.close()
        except Exception:
            pass
        state.stream = None
        
        time.sleep(0.02)
        
    state.audio_buffer = []
    state.recording_start_time = time.time()

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Audio recording error: {status}")
        if indata is not None and len(indata) > 0:
            state.audio_buffer.append(indata.copy())

    if not hasattr(state, 'mic_info_printed'):
        try:
            cfg = ConfigManager()
            selected_device = cfg.get_value("MIC_DEVICE_INDEX")
            if selected_device is not None:
                dev_info = sd.query_devices(selected_device)
            else:
                dev_info = sd.query_devices(kind='input')
            print(f"🎙️  Using microphone: {dev_info['name']}")
            state.mic_info_printed = True
        except Exception as e:
            print(f"Could not detect microphone: {e}")
    
    max_retries = 3
    stream_created = False
    try:
        cfg = ConfigManager()
        selected_device = cfg.get_value("MIC_DEVICE_INDEX")
    except Exception:
        selected_device = None

    for attempt in range(max_retries):
        try:
            sr = ConfigManager().get_value("SAMPLE_RATE") or 16000
            ch = ConfigManager().get_value("CHANNELS") or 1
            state.stream = sd.InputStream(
                samplerate=sr,
                channels=ch,
                callback=audio_callback,
                dtype="float32",
                device=selected_device if selected_device is not None else None
            )
            state.stream.start()
            if not is_debug():
                print("🔴 Recording...")
            else:
                print("Recording started...")
            stream_created = True
            break
        except Exception as e:
            print(f"Audio stream creation attempt {attempt + 1} failed: {e}")
            if selected_device is not None:
                selected_device = None
            if attempt < max_retries - 1:
                time.sleep(0.05)
            else:
                print("Failed to start recording after multiple attempts")
                state.stream = None
                state.audio_hardware_busy = False  # Reset on failure
                raise
    
    if not stream_created:
        state.audio_hardware_busy = False

def process_in_thread():
    state = get_global_state()
    alive_threads = []
    for t in state.active_threads:
        if not t.is_alive():
            if is_debug():
                print(f"[DEBUG] Cleaning up completed thread: {t}")
        else:
            alive_threads.append(t)
    state.active_threads = alive_threads
    if is_debug():
        print(f"[DEBUG] Active threads before starting new one: {len(state.active_threads)}")
    
    thread = threading.Thread(target=_process_recording, daemon=True)
    state.active_threads.append(thread)
    thread.start()
    if is_debug():
        print(f"[DEBUG] Started new processing thread, total active: {len(state.active_threads)}")

def stop_recording_and_process():
    state = get_global_state()
    if is_debug():
        print("[DEBUG] stop_recording_and_process called")
    with state.processing_lock:
        if is_debug():
            print("[DEBUG] Acquired processing lock")
            if not state.recording_in_progress:
                print("[DEBUG] Warning: stop_recording_and_process called but recording_in_progress is already False")
        state.recording_in_progress = False
        if state.stream:
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
            except Exception as e:
                print(f"Error closing audio stream: {e}")
            finally:
                state.stream = None
                state.audio_hardware_busy = False
        if is_debug():
            print("[DEBUG] Released processing lock, starting background processing")
    process_in_thread()
    if is_debug():
        print("[DEBUG] stop_recording_and_process completed")

def _process_recording():
    """Process one recording."""
    state = get_global_state()
    if is_debug():
        print("[DEBUG] _process_recording started")
    
    try:
        processing_start_time = time.time()
        
        if not state.audio_buffer:
            print("No audio recorded.")
            return
        
        audio_data, preprocessing_time = _prepare_audio_data(state)
        is_text_editing_mode = state.selected_text is not None
        _print_mode_info(is_text_editing_mode, state)
        
        transcriptions, provider_timings = _get_transcriptions(audio_data, state)
        
        if not transcriptions:
            print("❌ No transcriptions received.")
            return
        
        if len(transcriptions) > 1:
            compare_transcriptions(transcriptions)
        
        final_result, openrouter_time, mode_description = _process_results(
            transcriptions, is_text_editing_mode, state
        )
        
        _show_results(final_result, processing_start_time, preprocessing_time, 
                     provider_timings, openrouter_time, is_text_editing_mode, 
                     state, mode_description)
        
        print(f"Result: {final_result}")
        
    except Exception as e:
        print(f"Error processing recording: {e}")
        if is_debug():
            import traceback
            traceback.print_exc()
    finally:
        _cleanup_processing_thread()

def _prepare_audio_data(state):
    """Prepare audio data."""
    audio_data = np.concatenate(state.audio_buffer, axis=0)
    
    from audio_processing import preprocess_audio
    preprocessing_start = time.time()
    audio_data = preprocess_audio(audio_data)
    preprocessing_time = time.time() - preprocessing_start
    
    return audio_data, preprocessing_time

def _print_mode_info(is_text_editing_mode, state):
    """Print mode info."""
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

def _get_transcriptions(audio_data, state):
    """Run transcription across providers."""
    from providers import get_providers
    providers = get_providers()
    
    print("🎤 Transcribing...")
    
    sr = ConfigManager().get_value("SAMPLE_RATE") or 16000
    audio_bytes = get_audio_as_bytes(audio_data, sample_rate=sr)
    transcriptions = []
    provider_timings = {}
    
    recording_length_sec = len(audio_data) / sr
    provider_timeout = max(3, min(120, recording_length_sec))
    
    if is_debug():
        print(f"[DEBUG] Provider timeout set to {provider_timeout}s for {recording_length_sec:.1f}s recording")
    
    # Limit workers to provider count; Parakeet MLX is serialized internally
    max_workers = max(1, len(providers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_provider = {
            executor.submit(provider.transcribe_sync, audio_bytes): provider.__name__ 
            for provider in providers
        }
        
        provider_start_times = {
            future_to_provider[future]: time.time() 
            for future in future_to_provider
        }
        
        if state.turbo_mode:
            transcriptions, provider_timings = _handle_turbo_mode(
                future_to_provider, provider_start_times, provider_timeout
            )
        else:
            transcriptions, provider_timings = _handle_normal_mode(
                future_to_provider, provider_start_times, provider_timeout
            )
    
    return transcriptions, provider_timings

def _handle_turbo_mode(future_to_provider, provider_start_times, timeout):
    """Use first provider result in turbo mode."""
    if is_debug():
        print("[DEBUG] Using turbo mode - waiting for first result")
    
    transcriptions = []
    provider_timings = {}
    
    done, not_done = concurrent.futures.wait(
        future_to_provider, 
        return_when=concurrent.futures.FIRST_COMPLETED,
        timeout=timeout
    )
    
    for future in done:
        try:
            provider_name = future_to_provider[future]
            provider_timings[provider_name] = time.time() - provider_start_times[provider_name]
            
            name, text = future.result()
            if text:
                if is_debug():
                    print(f"\n[Raw Transcription from {name}]\n{text}\n")
                transcriptions.append((name, text))
                break
        except Exception as e:
            print(f"Provider error: {e}")
    
    for future in not_done:
        future.cancel()
    
    return transcriptions, provider_timings

def _handle_normal_mode(future_to_provider, provider_start_times, timeout):
    """Collect results from all providers."""
    transcriptions = []
    provider_timings = {}
    
    try:
        for future in concurrent.futures.as_completed(future_to_provider, timeout=timeout):
            provider_name = future_to_provider[future]
            provider_timings[provider_name] = time.time() - provider_start_times[provider_name]
            
            try:
                name, text = future.result()
                if text:
                    if is_debug():
                        print(f"\n[Raw Transcription from {name}]\n{text}\n")
                    transcriptions.append((name, text))
            except Exception as e:
                print(f"{provider_name} error: {e}")
    except concurrent.futures.TimeoutError:
        print("[WARNING] Provider transcription timed out. Processing partial results if any.")
        transcriptions, provider_timings = _handle_timeout(
            future_to_provider, provider_start_times
        )
    
    return transcriptions, provider_timings

def _handle_timeout(future_to_provider, provider_start_times):
    """Process completed transcriptions after timeout."""
    transcriptions = []
    provider_timings = {}
    
    for future in future_to_provider:
        if future.done() and not future.cancelled():
            provider_name = future_to_provider[future]
            provider_timings[provider_name] = time.time() - provider_start_times[provider_name]
            try:
                name, text = future.result()
                if text:
                    if is_debug():
                        print(f"\n[Raw Transcription from {name}]\n{text}\n")
                    transcriptions.append((name, text))
            except Exception as e:
                print(f"{provider_name} error after timeout: {e}")
    
    for future in future_to_provider:
        if not future.done():
            future.cancel()
    
    if is_debug():
        print("[DEBUG] Remaining futures cancelled after timeout")
    
    return transcriptions, provider_timings

def _process_results(transcriptions, is_text_editing_mode, state):
    """Dispatch based on mode."""
    if is_text_editing_mode:
        return _handle_text_editing(transcriptions, state.selected_text)
    else:
        return _handle_normal_transcription(transcriptions, state)

def _handle_text_editing(transcriptions, selected_text):
    """Apply voice command to selected text."""
    voice_command = transcriptions[0][1] if transcriptions else ""
    
    if is_debug():
        print(f"[DEBUG] Voice command: \"{voice_command}\"")
    
    editing_start = time.time()
    edited_text = edit_text_with_openrouter(selected_text, voice_command)
    editing_time = time.time() - editing_start
    
    replace_selected_text(edited_text)
    
    return edited_text, editing_time, f"Text Editing: \"{voice_command}\""

def _handle_normal_transcription(transcriptions, state):
    """Process transcription and type the result."""
    if state.turbo_mode:
        best_transcription = transcriptions[0][1]
        openrouter_time = 0.0
        if is_debug():
            print(f"[DEBUG] Turbo mode: using raw result from {transcriptions[0][0]}")
    else:
        context = build_context_from_history(state.transcription_history)
        app_context = get_application_context()
        
        if is_debug() and app_context:
            print(f"[DEBUG] Application context: {app_context}")
        
        openrouter_start = time.time()
        best_transcription = correct_with_openrouter(transcriptions, context, app_context)
        # Guard against empty correction results
        if not best_transcription or not best_transcription.strip():
            if is_debug():
                print("[DEBUG] OpenRouter returned empty result; falling back to best raw provider output")
            best_transcription = transcriptions[0][1]
        openrouter_time = time.time() - openrouter_start
    
    state.transcription_history.append((time.time(), best_transcription))
    
    type_fast(best_transcription)
    
    return best_transcription, openrouter_time, "Transcription"

def _show_results(final_result, processing_start_time, preprocessing_time, 
                 provider_timings, openrouter_time, is_text_editing_mode, 
                 state, mode_description):
    """Print results and timing."""
    total_time = time.time() - state.recording_start_time
    recording_time = processing_start_time - state.recording_start_time
    processing_time = time.time() - processing_start_time
    
    word_count = len(final_result.split())
    # Recording-time WPM reflects speaking rate only
    speech_wpm_recording = int((word_count / recording_time) * 60) if recording_time > 0 else 0
    # End-to-end throughput WPM reflects overall time from trigger to paste
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

def _show_debug_timing(is_text_editing_mode, recording_time, preprocessing_time,
                      provider_timings, openrouter_time, processing_time,
                      total_time, speech_wpm_recording, throughput_wpm_total):
    """Print debug timing."""
    mode_title = "TEXT EDITING" if is_text_editing_mode else "TRANSCRIPTION"
    print(f"\n╭───────────────────── {mode_title} TIMING ─────────────────────╮")
    print(f"│ Recording time:    {recording_time:.2f}s")
    print(f"│ Preprocessing:     {preprocessing_time:.2f}s")
    
    for provider, timing in provider_timings.items():
        print(f"│ {provider:<16} {timing:.2f}s")
    
    ai_label = "Text editing:" if is_text_editing_mode else "Correction:"
    print(f"│ {ai_label:<16} {openrouter_time:.2f}s")
    print(f"│ Processing time:   {processing_time:.2f}s")
    print(f"│ Total time:        {total_time:.2f}s")
    print(f"│ Speech rate (talk): {speech_wpm_recording} words per minute")
    print(f"│ Throughput (e2e):  {throughput_wpm_total} words per minute")
    print("╰───────────────────────────────────────────────────────────────╯")
    print("[DEBUG] _process_recording completed successfully")

def _show_clean_results(is_text_editing_mode, state, provider_timings,
                       openrouter_time, recording_time, total_time,
                       speech_wpm_recording, throughput_wpm_total,
                       final_result, mode_description):
    """Print results summary."""
    if is_text_editing_mode:
        print("✏️  Text Editing Results:")
    elif state.turbo_mode:
        print("⚡ TURBO MODE Results:")
    else:
        print("🎤 Transcription Results:")
    
    for provider, timing in sorted(provider_timings.items(), key=lambda x: x[1]):
        provider_name = provider.replace('providers.', '').replace('_', ' ').title()
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

def _cleanup_processing_thread():
    """Cleanup processing thread."""
    state = get_global_state()
    current_thread = threading.current_thread()
    if current_thread in state.active_threads:
        state.active_threads.remove(current_thread)
    
    state.selected_text = None
    
    if is_debug():
        print(f"[DEBUG] Removed current thread from active threads, remaining: {len(state.active_threads)}")
        print("[DEBUG] Recording state reset in _process_recording finally block")

def build_context_from_history(transcription_history):
    """Build context string from transcription history with time filtering and dividers."""
    cfg = ConfigManager()
    if not bool(cfg.get_value("ENABLE_CONTEXT")) or not transcription_history:
        return ""
    
    current_time = time.time()
    recent_entries = []
    
    for timestamp, text in reversed(transcription_history):
        if current_time - timestamp <= (cfg.get_value("CONTEXT_MAX_AGE_SECONDS") or 300):
            recent_entries.append(text)
        if len(recent_entries) >= (cfg.get_value("CONTEXT_HISTORY_COUNT") or 3):
            break
    
    recent_entries.reverse()
    
    if not recent_entries:
        return ""
    
    return " | ".join(recent_entries)

def get_audio_as_bytes(audio_data, sample_rate: Optional[int] = None):
    """Encode numpy audio to WAV bytes."""
    sr = sample_rate or (ConfigManager().get_value("SAMPLE_RATE") or 16000)
    audio_data_int16 = (audio_data * 32767).astype(np.int16)
    buffer = io.BytesIO()
    sf.write(buffer, audio_data_int16, sr, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return buffer.getvalue()

def build_correction_prompt(transcriptions, context=None, app_context=None):
    """Builds a prompt for transcription correction with all provider results."""
    if not transcriptions:
        return "No transcription available"
    
    if len(transcriptions) == 1:
        provider_name, text = transcriptions[0]
        prompt = f"""Transcription to refine: {text}"""
    else:
        prompt = "Multiple transcriptions of the same recording:\n\n"
        for i, (provider_name, text) in enumerate(transcriptions, 1):
            prompt += f"{i}. {provider_name}: {text}\n"
        prompt += "\nProvide the most accurate transcription by comparing and choosing the best parts from each."
    
    context_parts = []
    
    if app_context and app_context.strip():
        context_parts.append(f"Current application: {app_context}")
    
    if context and context.strip():
        context_parts.append(f"Recent transcriptions: {context}")
    
    if context_parts:
        context_section = "\n".join(context_parts)
        prompt = f"""{context_section}

---

{prompt}"""
    
    return prompt

def correct_with_openrouter(transcriptions, context, app_context=None):
    cfg = ConfigManager()
    import requests
    import json
    
    prompt = build_correction_prompt(transcriptions, context, app_context)
    
    headers = {
        "Authorization": f"Bearer {cfg.get_value('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    site_url = cfg.get_value("OPENROUTER_SITE_URL") or ""
    site_name = cfg.get_value("OPENROUTER_SITE_NAME") or ""
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name
    
    data = {
        "model": cfg.get_value("OPENROUTER_MODEL") or "google/gemini-2.5-flash",
        "messages": [
            {"role": "system", "content": cfg.get_value("SYSTEM_CONTEXT") or ""},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 1000
    }
    
    for attempt in range(3):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"OpenRouter attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                return transcriptions[0][1] if transcriptions else ""
            time.sleep(1)

def edit_text_with_openrouter(selected_text, voice_command):
    """Edit selected text using OpenRouter based on voice command."""
    cfg = ConfigManager()
    import requests
    import json
    
    prompt = f"""TASK: {voice_command}

ORIGINAL TEXT:
{selected_text}

INSTRUCTIONS: Apply the task to the original text above. Return ONLY the edited text, nothing else. No explanations, no formatting, no extra content."""
    
    headers = {
        "Authorization": f"Bearer {cfg.get_value('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    site_url = cfg.get_value("OPENROUTER_SITE_URL") or ""
    site_name = cfg.get_value("OPENROUTER_SITE_NAME") or ""
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name
    
    data = {
        "model": cfg.get_value("TEXT_EDITING_MODEL") or "google/gemini-2.5-flash",
        "messages": [
            {"role": "system", "content": cfg.get_value("TEXT_EDITING_CONTEXT") or ""},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 4000
    }
    
    for attempt in range(3):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=15
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Text editing attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                print("❌ Text editing failed - returning original text")
                return selected_text
            time.sleep(1)


def get_application_context():
    """Get context about the current active application/window."""
    cfg = ConfigManager()
    if not bool(cfg.get_value("ENABLE_APPLICATION_CONTEXT")):
        return ""
    
    try:
        script = '''
        tell application "System Events"
            set frontApp to first application process whose frontmost is true
            set appName to name of frontApp
            try
                set windowTitle to name of front window of frontApp
                return appName & " | " & windowTitle
            on error
                return appName
            end try
        end tell
        '''
        
        result = subprocess.run(['osascript', '-e', script], 
                              capture_output=True, text=True, 
                              timeout=(cfg.get_value("APPLICATION_CONTEXT_TIMEOUT") or 2))
        
        if result.returncode == 0 and result.stdout.strip():
            app_context = result.stdout.strip()
            
            app_context = app_context.replace(' • ', ' | ')
            app_context = app_context.replace(' — ', ' | ')
            
            return app_context
        else:
            return ""
            
    except Exception as e:
        if is_debug():
            print(f"[DEBUG] Could not get application context: {e}")
        return ""

def reset_state():
    state = get_global_state()
    with state.processing_lock:
        state.recording_in_progress = False
        state.audio_hardware_busy = False
        state.audio_buffer = []
        state.selected_text = None
        if state.stream:
            try:
                state.stream.stop()
                state.stream.close()
            except Exception as e:
                print(f"Error closing audio stream: {e}")
            finally:
                state.stream = None
        for thread in state.active_threads:
            if thread.is_alive():
                print("Warning: Thread still running during reset")
        state.active_threads = []

def cleanup():
    get_global_state()
    reset_state()
    import time
    time.sleep(0.1)
