"""
Audio engine for multi-mic capture with pre-roll buffers and silence-based chunking.

Manages multiple sounddevice input streams, detects silence to emit chunks
during recording, and provides a thread-safe interface.
"""

import threading
import time
from collections import deque
from typing import Dict, List, Optional, Callable, Any

import numpy as np

from .types import AudioChunk
from .config import Config
from .ui.hud import normalize_level


# =============================================================================
# Audio Constants
# =============================================================================
# These constants control the audio capture and chunking behavior.
# Tune these values based on your microphone setup and environment.
#
# Key parameters:
#   - SILENCE_THRESHOLD_DB: Lower = more sensitive to quiet sounds
#   - MIN_CHUNK_SECONDS: Prevents tiny chunks from being emitted
#   - MAX_CHUNK_SECONDS: Forces emit even if speaker doesn't pause
#   - TRAILING_SILENCE: Keeps natural pause at end of phrases
#
# For noisy environments, raise SILENCE_THRESHOLD_DB to -30 or -25
# For quiet rooms, -40 works well
# =============================================================================

DEFAULT_BLOCKSIZE = 1024  # Number of frames per audio callback
SILENCE_THRESHOLD_DB = -35  # dB threshold for silence detection
MIN_CHUNK_SECONDS = 3.0  # Don't emit chunks shorter than this
TRAILING_SILENCE_SECONDS = 0.5  # Keep this much silence at end of chunk
MAX_CHUNK_SECONDS = 30.0  # Force chunk emit after this duration

# =============================================================================
# Pre-roll Settings
# =============================================================================
# Pre-roll captures audio BEFORE the user triggers recording, preventing
# the first word from being cut off due to reaction time latency.
# =============================================================================
# Pre-roll length comes from config.preroll_seconds.

# =============================================================================
# Silence Detection Tuning
# =============================================================================
# These control when the engine decides a "phrase" has ended.
# Shorter durations = faster response, but may cut off slow speakers.
# =============================================================================

# =============================================================================
# Adaptive Noise Floor Settings
# =============================================================================
# Instead of a fixed dB threshold, we track each mic's noise floor and
# consider it "speech" when audio is X dB above that floor.
# This automatically adapts to different mic gains and room noise.
# =============================================================================
ADAPTIVE_THRESHOLD_ENABLED = True  # Use adaptive per-mic thresholds
NOISE_FLOOR_WINDOW_SECONDS = 30.0  # Track noise floor over this window
SPEECH_HEADROOM_DB = 12.0  # Speech must be this many dB above noise floor
NOISE_FLOOR_PERCENTILE = 10  # Use this percentile as noise floor (filters outliers)
FALLBACK_NOISE_FLOOR_DB = -50.0  # Use this if not enough samples yet

# =============================================================================
# Audio Format Settings
# =============================================================================
# 16kHz mono is standard for speech-to-text models like Whisper.
# Higher sample rates waste bandwidth without improving transcription.
# =============================================================================
DEBUG_AUDIO_LEVELS = False  # Set True to log per-block audio levels (very chatty)
LOG_CHUNK_EVENTS = True  # Log when chunks are emitted
SAMPLE_RATE = 16000  # Default sample rate for audio capture


class AudioEngine:
    """
    Manages multiple mic streams with pre-roll buffers.
    Detects silence to emit chunks during recording.

    The engine maintains a pre-roll buffer for each microphone, allowing
    capture of audio slightly before recording officially starts. This
    prevents cutting off the beginning of speech.

    Thread-safe: all public methods can be called from any thread.

    Usage:
        engine = AudioEngine(config)
        mics = engine.initialize()

        engine.on_chunk_ready = session.on_chunk_ready
        engine.start_recording()
        # ... user speaks ...
        final_chunk = engine.stop_recording()
    """

    def __init__(self, config: Config):
        self.config = config

        # Streams and buffers (keyed by mic name)
        self.streams: Dict[str, Any] = {}  # sounddevice.InputStream (lazy import)
        self.preroll_buffers: Dict[str, deque] = {}
        self.current_chunk: Dict[str, List[np.ndarray]] = {}

        # State
        self.is_recording: bool = False
        self.last_speech_time: float = 0.0  # Time when ANY mic last detected speech
        self.current_level: float = 0.0  # 0..1 mic level for the HUD meter
        self._chunk_has_speech: bool = False  # Any speech in the chunk being built
        self._primary_mic: Optional[str] = None  # First mic drives silence timing
        self._lock = threading.Lock()

        # Adaptive noise floor tracking (per mic)
        # Stores recent dB values to compute each mic's noise floor
        self._noise_floor_samples: Dict[str, deque] = {}
        self._noise_floor_cache: Dict[str, float] = {}  # Cached floor values
        self._speech_active: Dict[str, bool] = {}  # Hysteresis state per mic

        # Disconnect/reconnect tracking
        self._stream_error_counts: Dict[str, int] = {}
        self._known_connected: set = set()  # mics with active streams
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop: Optional[threading.Event] = None

        # Callback for chunk emission
        self.on_chunk_ready: Optional[Callable[[AudioChunk], None]] = None


        # Computed values
        self._preroll_samples = int(config.preroll_seconds * config.sample_rate)
        self._silence_samples = int(config.silence_threshold * config.sample_rate)

    def initialize(self) -> List[str]:
        """
        Open streams for configured mics, then start device monitor.

        Returns:
            List of successfully initialized mic names
        """
        active_mics = self.sync_configured_mics()
        self._start_device_monitor()
        return active_mics

    def sync_configured_mics(self) -> List[str]:
        """
        Apply the current enabled_mics config to live streams.

        This is intentionally idle-only: changing the mic set mid-recording would
        make a session's audio bundle inconsistent, but the next recording should
        pick up settings changes without an app restart.
        """
        with self._lock:
            if self.is_recording:
                return list(self.streams.keys())
            configured_mics = list(dict.fromkeys(self.config.enabled_mics))
            active_mics = list(self.streams.keys())

        configured_set = set(configured_mics)
        for mic_name in active_mics:
            if mic_name not in configured_set:
                print(f"[Audio] Mic disabled in settings: {mic_name}")
                self._remove_mic(mic_name)

        for mic_name in configured_mics:
            self._try_open_stream(mic_name)

        with self._lock:
            if self._primary_mic not in self.streams:
                self._primary_mic = next(iter(self.streams), None)
            return list(self.streams.keys())

    def _try_open_stream(self, mic_name: str) -> bool:
        """Open a sounddevice stream for mic_name. Returns True on success."""
        import sounddevice as sd

        with self._lock:
            if mic_name in self.streams:
                return True

        device_index = self._find_device(mic_name)
        if device_index is None:
            return False

        try:
            preroll_chunks = int(self._preroll_samples / DEFAULT_BLOCKSIZE)
            noise_floor_samples = int(NOISE_FLOOR_WINDOW_SECONDS * self.config.sample_rate / DEFAULT_BLOCKSIZE)

            stream = sd.InputStream(
                device=device_index,
                samplerate=self.config.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=DEFAULT_BLOCKSIZE,
                callback=lambda indata, frames, time, status, mic=mic_name:
                    self._audio_callback(mic, indata, frames, time, status)
            )
            stream.start()

            with self._lock:
                self.streams[mic_name] = stream
                self.preroll_buffers[mic_name] = deque(maxlen=preroll_chunks)
                self.current_chunk[mic_name] = []
                self._noise_floor_samples[mic_name] = deque(maxlen=noise_floor_samples)
                self._noise_floor_cache[mic_name] = FALLBACK_NOISE_FLOOR_DB
                self._speech_active[mic_name] = False
                self._stream_error_counts[mic_name] = 0
                self._known_connected.add(mic_name)
                if self._primary_mic is None:
                    self._primary_mic = mic_name

            print(f"[Audio] Mic connected: {mic_name}")
            return True

        except Exception as e:
            print(f"[Audio] Failed to open stream for {mic_name}: {e}")
            return False

    def _remove_mic(self, mic_name: str) -> None:
        """Tear down a mic's stream and buffers, promote new primary if needed."""
        with self._lock:
            stream = self.streams.pop(mic_name, None)
            self.preroll_buffers.pop(mic_name, None)
            self.current_chunk.pop(mic_name, None)
            self._noise_floor_samples.pop(mic_name, None)
            self._noise_floor_cache.pop(mic_name, None)
            self._speech_active.pop(mic_name, None)
            self._stream_error_counts.pop(mic_name, None)
            self._known_connected.discard(mic_name)

            if self._primary_mic == mic_name:
                self._primary_mic = next(iter(self.streams), None)
                if self._primary_mic:
                    print(f"[Audio] Primary mic now: {self._primary_mic}")
                else:
                    print("[Audio] No active mics remaining")

        if stream:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def _start_device_monitor(self) -> None:
        """Start background thread that watches for connect/disconnect events."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._monitor_stop = threading.Event()
        self._monitor_thread = threading.Thread(
            target=self._device_monitor_loop,
            daemon=True,
            name="audio-device-monitor",
        )
        self._monitor_thread.start()

    def _device_monitor_loop(self) -> None:
        while not self._monitor_stop.wait(3.0):
            self._scan_devices()

    def _scan_devices(self) -> None:
        """Detect dead streams and reconnect any configured-but-missing mics."""
        with self._lock:
            is_recording = self.is_recording

        if not is_recording and hasattr(self.config, "reload"):
            self.config.reload()

        # Collect dead mics (stream gone inactive or too many callback errors)
        dead_mics = []
        with self._lock:
            configured = set(self.config.enabled_mics)
            for mic_name, stream in list(self.streams.items()):
                disabled = not is_recording and mic_name not in configured
                if disabled or not stream.active or self._stream_error_counts.get(mic_name, 0) >= 10:
                    dead_mics.append(mic_name)

        for mic_name in dead_mics:
            print(f"[Audio] Mic disconnected: {mic_name}")
            self._remove_mic(mic_name)

        # Try to open any configured mic that isn't currently streaming
        with self._lock:
            missing = [] if self.is_recording else [m for m in self.config.enabled_mics if m not in self.streams]

        for mic_name in missing:
            self._try_open_stream(mic_name)

    def _find_device(self, mic_name: str) -> Optional[int]:
        """Find device index by name (fuzzy matching)."""
        import sounddevice as sd

        devices = sd.query_devices()
        mic_lower = mic_name.lower()

        # Exact match first
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                if d["name"].lower() == mic_lower:
                    return i

        # Substring match
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                if mic_lower in d["name"].lower():
                    return i

        # Reverse substring
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                if d["name"].lower() in mic_lower:
                    return i

        return None

    def start_recording(self) -> None:
        """Begin capturing audio. Dumps preroll into current chunk."""
        import time
        with self._lock:
            self.is_recording = True
            self.last_speech_time = time.time()  # Assume speech at start
            self._chunk_has_speech = False

            # Dump preroll into current chunk, then clear to prevent stale reuse
            # This fixes a bug where Bluetooth mics (AirPods) that go idle between
            # sessions would have stale audio from previous sessions in their preroll
            for mic_name, preroll in self.preroll_buffers.items():
                self.current_chunk[mic_name] = list(preroll)
                preroll.clear()

    def stop_recording(self) -> AudioChunk:
        """
        Stop recording and return final chunk.

        IMPORTANT: Disconnects on_chunk_ready immediately to prevent race.

        Returns:
            Dict mapping mic names to audio arrays
        """
        with self._lock:
            self.is_recording = False
            self.on_chunk_ready = None  # Disconnect immediately
            self.last_speech_time = 0.0
            self._chunk_has_speech = False
            return self._flush_current_chunk()

    def shutdown(self) -> None:
        """Close all streams cleanly."""
        # Stop monitor thread first so it doesn't race with cleanup
        if self._monitor_stop:
            self._monitor_stop.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

        # Collect streams while holding lock, then close outside lock
        # to avoid deadlock with audio callback
        with self._lock:
            self.is_recording = False
            self.on_chunk_ready = None
            streams_to_close = list(self.streams.items())
            self.streams.clear()
            self.preroll_buffers.clear()
            self.current_chunk.clear()
            self._primary_mic = None
            self.last_speech_time = 0.0
            self._noise_floor_samples.clear()
            self._noise_floor_cache.clear()
            self._speech_active.clear()
            self._stream_error_counts.clear()
            self._known_connected.clear()

        # Close streams outside of lock
        for mic_name, stream in streams_to_close:
            try:
                stream.stop()
                stream.close()
            except Exception as e:
                print(f"Error closing stream {mic_name}: {e}")

    def _audio_callback(
        self,
        mic_name: str,
        indata: np.ndarray,
        frames: int,
        time_info,
        status
    ) -> None:
        """
        Called by sounddevice for each audio block.

        Handles:
        - Pre-roll buffer filling when not recording
        - Chunk building when recording
        - Silence detection and chunk emission
        """
        if status:
            print(f"Audio callback status ({mic_name}): {status}")
            with self._lock:
                if mic_name in self._stream_error_counts:
                    self._stream_error_counts[mic_name] += 1

        # One copy: ravel() is a view for C-contiguous input, so .copy() after
        # it allocates once. indata is reused by the driver, so the copy is
        # required.
        audio = indata.ravel().copy()

        with self._lock:
            # Check if shutdown has cleared our buffers
            if mic_name not in self.preroll_buffers:
                return

            # Always update preroll buffer (keeps it fresh for next session)
            self.preroll_buffers[mic_name].append(audio)

            if not self.is_recording:
                return

            # Append to current chunk
            self.current_chunk[mic_name].append(audio)

            if not self._get_bool_config("chunk_on_silence", True):
                return

            # Check if this mic detects speech - if so, update last_speech_time
            # This way ANY mic detecting speech prevents chunk emission
            is_silent = self._is_silence(audio, mic_name)
            if not is_silent:
                self.last_speech_time = time.time()
                self._chunk_has_speech = True

            # Only primary mic checks timing and emits chunks (to avoid duplicate processing)
            if mic_name == self._primary_mic:
                current_time = time.time()
                silence_duration = current_time - self.last_speech_time

                # Measured at 14us for a 30s chunk — not worth a second
                # source of truth that can drift from the buffers.
                chunk_samples = sum(len(b) for b in self.current_chunk[mic_name])
                chunk_duration = chunk_samples / self.config.sample_rate

                paused = (silence_duration >= self.config.silence_threshold
                          and chunk_duration >= MIN_CHUNK_SECONDS)

                # A chunk holding no speech is pure room tone, and STT models
                # hallucinate confidently on silence ("Yeah.", "Mm-hmm.", "嗯。").
                # Emitting also resets last_speech_time, so holding the key
                # without talking spawned a fresh silence chunk every
                # MIN_CHUNK_SECONDS: one 98s session produced 31 chunks, 22 of
                # them silence, whose hallucinations were concatenated into the
                # output. Never hand dead air to a provider.
                if not self._chunk_has_speech:
                    if chunk_duration >= MAX_CHUNK_SECONDS:
                        # Still clear the buffers, or a long silent hold grows
                        # them without bound.
                        self._discard_current_chunk()
                    return
                # Without this cap, a chunk only ever ends on a pause, so
                # talking continuously means nothing transcribes until the key
                # is released — 28% of sessions over 20s emitted a single
                # chunk. Force a cut so transcription keeps overlapping speech.
                too_long = chunk_duration >= MAX_CHUNK_SECONDS

                if paused or too_long:
                    if paused:
                        # Trim excess silence - keep only TRAILING_SILENCE_SECONDS
                        excess_silence = silence_duration - TRAILING_SILENCE_SECONDS
                        if excess_silence > 0:
                            samples_to_trim = int(excess_silence * self.config.sample_rate)
                            self._trim_trailing_samples(samples_to_trim)

                    if LOG_CHUNK_EVENTS:
                        reason = "pause" if paused else f"max {MAX_CHUNK_SECONDS:.0f}s"
                        print(f"[Audio] === CHUNK EMIT ({reason}) === duration={chunk_duration:.2f}s, silence={silence_duration:.2f}s")

                    # Emit chunk
                    chunk = self._flush_current_chunk()
                    callback = self.on_chunk_ready

                    # Reset speech time to now (start fresh for next chunk)
                    self.last_speech_time = current_time
                    self._chunk_has_speech = False

                    if callback and chunk:
                        # Release lock before callback to avoid deadlock
                        self._lock.release()
                        try:
                            callback(chunk)
                        except Exception as e:
                            # Don't let a callback failure kill the audio
                            # stream, but never swallow it silently either.
                            print(f"[Audio] Chunk callback error: {e}")
                        finally:
                            self._lock.acquire()
                        # stop_recording() may have run while unlocked
                        if not self.is_recording:
                            return

    def _is_silence(self, audio: np.ndarray, mic_name: str = "") -> bool:
        """
        Check if audio block is silence using adaptive per-mic threshold.

        Instead of a fixed dB threshold, we track each mic's noise floor
        and consider it "speech" when audio is SPEECH_HEADROOM_DB above that floor.
        """
        if len(audio) == 0:
            return True

        # Calculate RMS in dB
        rms = np.sqrt(np.mean(audio ** 2))
        if rms == 0:
            return True

        db = 20 * np.log10(rms)

        # Update noise floor samples for this mic (always, even when not recording)
        if mic_name in self._noise_floor_samples:
            self._noise_floor_samples[mic_name].append(db)

            # Recompute noise floor periodically (every ~1s = ~15 samples)
            if len(self._noise_floor_samples[mic_name]) % 15 == 0:
                self._update_noise_floor(mic_name)

        # Determine silence threshold
        adaptive_enabled = self._get_bool_config(
            "adaptive_threshold_enabled",
            ADAPTIVE_THRESHOLD_ENABLED,
        )
        if adaptive_enabled and mic_name in self._noise_floor_cache:
            noise_floor = self._noise_floor_cache[mic_name]
            speech_headroom_db = self._get_float_config("speech_headroom_db", SPEECH_HEADROOM_DB)
            speech_hysteresis_db = self._get_float_config("speech_hysteresis_db", 3.0)
            release_headroom_db = max(0.0, speech_headroom_db - max(0.0, speech_hysteresis_db))

            start_threshold = noise_floor + speech_headroom_db
            release_threshold = noise_floor + release_headroom_db

            active = self._speech_active.get(mic_name, False)
            if active:
                active = db >= release_threshold
            else:
                active = db >= start_threshold
            self._speech_active[mic_name] = active
            is_silent = not active
        else:
            threshold = SILENCE_THRESHOLD_DB
            noise_floor = None
            is_silent = db < threshold

        # Feed the HUD meter. The audio thread must never touch AppKit, so this
        # only stores a float; the HUD's redraw timer samples it on the main
        # thread. Primary mic only, so a second mic can't fight over the value.
        if mic_name == self._primary_mic:
            self.current_level = normalize_level(db, noise_floor)

        if DEBUG_AUDIO_LEVELS:
            # Log periodically (every ~0.5s worth of blocks)
            if not hasattr(self, '_debug_counter'):
                self._debug_counter = 0
            self._debug_counter += 1
            if self._debug_counter % 8 == 0:  # Log every ~0.5s at 16kHz/1024 blocksize
                import time
                silence_duration = time.time() - self.last_speech_time if self.last_speech_time > 0 else 0.0
                status = "SILENCE" if is_silent else "SPEECH"
                if noise_floor is not None:
                    start = self._get_float_config("speech_headroom_db", SPEECH_HEADROOM_DB)
                    hyst = self._get_float_config("speech_hysteresis_db", 3.0)
                    rel = max(0.0, start - max(0.0, hyst))
                    print(
                        f"[Audio] {mic_name}: {db:.1f}dB ({status}) | floor={noise_floor:.1f}dB start=+{start:.1f}dB rel=+{rel:.1f}dB | silence={silence_duration:.2f}s"
                    )
                else:
                    print(f"[Audio] {mic_name}: {db:.1f}dB ({status}) | silence={silence_duration:.2f}s")

        return is_silent

    def _update_noise_floor(self, mic_name: str) -> None:
        """
        Compute noise floor as the Nth percentile of recent dB samples.

        Using a percentile (e.g., 10th) instead of minimum filters out
        occasional quiet moments and gives a more stable floor estimate.
        """
        samples = self._noise_floor_samples.get(mic_name)
        if not samples or len(samples) < 10:
            return

        # Compute Nth percentile
        sorted_samples = sorted(samples)
        percentile = self._get_int_config("noise_floor_percentile", NOISE_FLOOR_PERCENTILE, min_value=1, max_value=50)
        percentile_idx = int(len(sorted_samples) * percentile / 100)
        noise_floor = sorted_samples[percentile_idx]

        self._noise_floor_cache[mic_name] = noise_floor

    def _get_bool_config(self, key: str, default: bool) -> bool:
        value = getattr(self.config, key, default)
        if isinstance(value, bool):
            return value
        # Handle strings/ints/mocks safely
        try:
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        except Exception:
            return default

    def _get_float_config(self, key: str, default: float) -> float:
        value = getattr(self.config, key, default)
        try:
            return float(value)
        except Exception:
            return default

    def _get_int_config(self, key: str, default: int, *, min_value: int, max_value: int) -> int:
        value = getattr(self.config, key, default)
        try:
            parsed = int(value)
        except Exception:
            return default
        return max(min_value, min(max_value, parsed))

    def _trim_trailing_samples(self, samples_to_trim: int) -> None:
        """
        Trim samples from the end of all mic buffers.

        Used to remove excess silence when emitting chunks.
        Must be called with lock held.
        """
        if samples_to_trim <= 0:
            return

        for mic_name in self.current_chunk:
            buffers = self.current_chunk[mic_name]
            if not buffers:
                continue

            # Work backwards through buffers, removing samples
            remaining_to_trim = samples_to_trim
            while remaining_to_trim > 0 and buffers:
                last_buffer = buffers[-1]
                if len(last_buffer) <= remaining_to_trim:
                    # Remove entire buffer
                    remaining_to_trim -= len(last_buffer)
                    buffers.pop()
                else:
                    # Trim partial buffer
                    buffers[-1] = last_buffer[:-remaining_to_trim]
                    remaining_to_trim = 0

    def _discard_current_chunk(self) -> None:
        """
        Drop the buffered audio without emitting it.

        Used for dead air: the buffers still have to be cleared or a long
        silent hold grows them forever, but the audio must not reach the
        providers. Must be called with lock held.
        """
        for mic_name in self.current_chunk:
            self.current_chunk[mic_name] = []

    def _flush_current_chunk(self) -> AudioChunk:
        """
        Flush current chunk buffers and return audio data.

        Must be called with lock held.
        """
        chunk: AudioChunk = {}

        for mic_name, buffers in self.current_chunk.items():
            if buffers:
                chunk[mic_name] = np.concatenate(buffers)
            else:
                chunk[mic_name] = np.array([], dtype=np.float32)

            # Reset buffer
            self.current_chunk[mic_name] = []

        return chunk
