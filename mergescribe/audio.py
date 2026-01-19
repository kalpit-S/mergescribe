"""
Audio engine for multi-mic capture with pre-roll buffers and silence-based chunking.

Manages multiple sounddevice input streams, detects silence to emit chunks
during recording, and provides a thread-safe interface.
"""

import threading
from collections import deque
from typing import Dict, List, Optional, Callable

import numpy as np

from .types import AudioChunk
from .config import Config


# Constants
DEFAULT_BLOCKSIZE = 1024
SILENCE_THRESHOLD_DB = -35  # dB threshold for silence detection
MIN_CHUNK_SECONDS = 5.0  # Don't emit chunks shorter than this
TRAILING_SILENCE_SECONDS = 0.5  # Keep this much silence at end of chunk


class AudioEngine:
    """
    Manages multiple mic streams with pre-roll buffers.
    Detects silence to emit chunks during recording.

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
        self.streams: Dict[str, "sd.InputStream"] = {}
        self.preroll_buffers: Dict[str, deque] = {}
        self.current_chunk: Dict[str, List[np.ndarray]] = {}

        # State
        self.is_recording: bool = False
        self.silence_duration: float = 0.0
        self._lock = threading.Lock()

        # Callback for chunk emission
        self.on_chunk_ready: Optional[Callable[[AudioChunk], None]] = None

        # Computed values
        self._preroll_samples = int(config.preroll_seconds * config.sample_rate)
        self._silence_samples = int(config.silence_threshold * config.sample_rate)

    def initialize(self) -> List[str]:
        """
        Open streams for configured mics.

        Returns:
            List of successfully initialized mic names
        """
        import sounddevice as sd

        active_mics = []

        for mic_name in self.config.enabled_mics:
            try:
                device_index = self._find_device(mic_name)
                if device_index is None:
                    print(f"Mic not found: {mic_name}")
                    continue

                # Initialize buffers
                preroll_chunks = int(
                    self._preroll_samples / DEFAULT_BLOCKSIZE
                )
                self.preroll_buffers[mic_name] = deque(maxlen=preroll_chunks)
                self.current_chunk[mic_name] = []

                # Create stream
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
                self.streams[mic_name] = stream
                active_mics.append(mic_name)
                print(f"Initialized mic: {mic_name}")

            except Exception as e:
                print(f"Failed to initialize mic {mic_name}: {e}")

        return active_mics

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
        with self._lock:
            self.is_recording = True
            self.silence_duration = 0.0

            # Dump preroll into current chunk
            for mic_name, preroll in self.preroll_buffers.items():
                self.current_chunk[mic_name] = list(preroll)

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
            self.silence_duration = 0.0
            return self._flush_current_chunk()

    def shutdown(self) -> None:
        """Close all streams cleanly."""
        # Collect streams while holding lock, then close outside lock
        # to avoid deadlock with audio callback
        with self._lock:
            self.is_recording = False
            self.on_chunk_ready = None
            streams_to_close = list(self.streams.items())
            self.streams.clear()
            self.preroll_buffers.clear()
            self.current_chunk.clear()

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

        # Make a copy of the audio data
        audio = indata.copy().flatten()

        with self._lock:
            # Check if shutdown has cleared our buffers
            if mic_name not in self.preroll_buffers:
                return

            if not self.is_recording:
                # Just fill preroll buffer
                self.preroll_buffers[mic_name].append(audio)
                return

            # Append to current chunk
            self.current_chunk[mic_name].append(audio)

            # Check for silence (use first mic as reference)
            if mic_name == list(self.current_chunk.keys())[0]:
                if self._is_silence(audio):
                    self.silence_duration += frames / self.config.sample_rate
                    if self.silence_duration >= self.config.silence_threshold:
                        # Check minimum chunk duration before emitting
                        chunk_samples = sum(len(b) for b in self.current_chunk[mic_name])
                        chunk_duration = chunk_samples / self.config.sample_rate

                        if chunk_duration >= MIN_CHUNK_SECONDS:
                            # Trim excess silence - keep only TRAILING_SILENCE_SECONDS
                            excess_silence = self.silence_duration - TRAILING_SILENCE_SECONDS
                            if excess_silence > 0:
                                samples_to_trim = int(excess_silence * self.config.sample_rate)
                                self._trim_trailing_samples(samples_to_trim)

                            # Emit chunk
                            chunk = self._flush_current_chunk()
                            callback = self.on_chunk_ready

                            if callback and chunk:
                                # Release lock before callback to avoid deadlock
                                self._lock.release()
                                try:
                                    callback(chunk)
                                finally:
                                    self._lock.acquire()
                                    # Check if stop_recording() was called while lock was released
                                    if not self.is_recording:
                                        return

                        self.silence_duration = 0.0
                else:
                    self.silence_duration = 0.0

    def _is_silence(self, audio: np.ndarray) -> bool:
        """Check if audio block is silence."""
        if len(audio) == 0:
            return True

        # Calculate RMS in dB
        rms = np.sqrt(np.mean(audio ** 2))
        if rms == 0:
            return True

        db = 20 * np.log10(rms)
        return db < SILENCE_THRESHOLD_DB

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
