#!/usr/bin/env python3
"""
Live transcription of system audio via BlackHole.

Usage:
    python transcribe_audio.py

Requires BlackHole to be installed and configured as a Multi-Output Device.
Run with --setup to create the Multi-Output Device automatically.
"""

import argparse
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

# Audio settings
SAMPLE_RATE = 16000
CHUNK_SECONDS = 3.0  # Transcribe every N seconds
BUFFER_SECONDS = 0.5  # Overlap for context
SILENCE_THRESHOLD_DB = -40  # More sensitive than voice recording


def find_blackhole_device() -> Optional[int]:
    """Find BlackHole 2ch device index."""
    import sounddevice as sd

    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            name = d["name"].lower()
            if "blackhole" in name:
                return i
    return None


def check_multi_output_exists() -> bool:
    """Check if a Multi-Output Device exists that includes BlackHole."""
    result = subprocess.run(
        ["system_profiler", "SPAudioDataType", "-json"],
        capture_output=True, text=True
    )
    return "multi-output" in result.stdout.lower() and "blackhole" in result.stdout.lower()


def setup_multi_output():
    """Guide user through Multi-Output Device setup."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                  BlackHole Multi-Output Setup                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  To hear audio AND capture it, you need a Multi-Output Device.   ║
║                                                                   ║
║  I'll open Audio MIDI Setup. Then:                               ║
║                                                                   ║
║  1. Click '+' in bottom-left → 'Create Multi-Output Device'      ║
║  2. Check BOTH:                                                   ║
║     ☑ BlackHole 2ch                                              ║
║     ☑ Your speakers/headphones (e.g., MacBook Pro Speakers)      ║
║  3. Right-click the new device → 'Use This Device For Sound Output' ║
║                                                                   ║
║  Press Enter to open Audio MIDI Setup...                         ║
╚══════════════════════════════════════════════════════════════════╝
""")
    input()
    subprocess.run(["open", "/System/Applications/Utilities/Audio MIDI Setup.app"])
    print("\nAfter setup, run: python transcribe_audio.py")
    sys.exit(0)


class TranscriptionWindow:
    """Tkinter window for displaying live transcriptions."""

    def __init__(self):
        import tkinter as tk
        from tkinter import scrolledtext

        self.root = tk.Tk()
        self.root.title("Live Transcription")

        # Window settings
        self.root.geometry("800x200+100+50")  # width x height + x + y
        self.root.attributes("-topmost", True)  # Always on top
        self.root.configure(bg="#1e1e1e")

        # Make window semi-transparent (macOS)
        try:
            self.root.attributes("-alpha", 0.95)
        except Exception:
            pass

        # Text widget with scrollbar
        self.text = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            font=("SF Mono", 14),
            bg="#1e1e1e",
            fg="#ffffff",
            insertbackground="#ffffff",
            selectbackground="#3d5a80",
            padx=15,
            pady=15,
            borderwidth=0,
            highlightthickness=0,
        )
        self.text.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status = tk.Label(
            self.root,
            text="Listening...",
            font=("SF Mono", 10),
            bg="#1e1e1e",
            fg="#666666",
            anchor="w",
            padx=10,
        )
        self.status.pack(fill=tk.X)

        # Allow text selection but not editing
        self.text.configure(state=tk.DISABLED)

        # Keyboard shortcuts
        self.root.bind("<Command-q>", lambda e: self.quit())
        self.root.bind("<Escape>", lambda e: self.quit())

        self._running = True

    def append_text(self, text: str):
        """Add text to the window (thread-safe)."""
        def _append():
            import tkinter as tk
            self.text.configure(state=tk.NORMAL)

            # Add space if needed
            current = self.text.get("1.0", tk.END).strip()
            if current and not current.endswith((" ", "\n")):
                self.text.insert(tk.END, " ")

            self.text.insert(tk.END, text)
            self.text.see(tk.END)  # Auto-scroll
            self.text.configure(state=tk.DISABLED)

        self.root.after(0, _append)

    def set_status(self, status: str):
        """Update status bar (thread-safe)."""
        try:
            self.root.after(0, lambda: self.status.configure(text=status))
        except RuntimeError:
            pass  # Main loop not started yet

    def quit(self):
        self._running = False
        self.root.quit()
        self.root.destroy()

    def run(self):
        """Start the main loop."""
        self.root.mainloop()

    @property
    def running(self) -> bool:
        return self._running


class AudioTranscriber:
    """Captures system audio and transcribes it."""

    def __init__(self, window: TranscriptionWindow, device_index: int):
        self.window = window
        self.device_index = device_index
        self.provider = None
        self._stop = False

        # Audio buffer
        self.buffer = deque(maxlen=int(SAMPLE_RATE * (CHUNK_SECONDS + BUFFER_SECONDS)))
        self._buffer_lock = threading.Lock()

    def initialize_provider(self):
        """Load Parakeet model."""
        self.window.set_status("Loading Parakeet model...")

        from mergescribe.providers.parakeet import ParakeetProvider
        self.provider = ParakeetProvider()
        self.provider.initialize()

        self.window.set_status("Listening...")

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        if status:
            print(f"Audio status: {status}")

        audio = indata.copy().flatten()
        with self._buffer_lock:
            self.buffer.extend(audio)

    def _is_silence(self, audio: np.ndarray) -> bool:
        """Check if audio chunk is silence."""
        if len(audio) == 0:
            return True
        rms = np.sqrt(np.mean(audio ** 2))
        if rms == 0:
            return True
        db = 20 * np.log10(rms)
        return db < SILENCE_THRESHOLD_DB

    def _transcription_loop(self):
        """Continuously transcribe audio buffer."""
        last_transcribe = time.time()

        while not self._stop and self.window.running:
            time.sleep(0.1)  # Check frequently

            now = time.time()
            if now - last_transcribe < CHUNK_SECONDS:
                continue

            # Get audio from buffer
            with self._buffer_lock:
                if len(self.buffer) < SAMPLE_RATE * 1.0:  # Need at least 1s
                    continue
                audio = np.array(list(self.buffer))
                self.buffer.clear()

            # Skip if silence
            if self._is_silence(audio):
                last_transcribe = now
                continue

            # Transcribe
            self.window.set_status("Transcribing...")
            try:
                result = self.provider.transcribe(audio, "system")
                if result.text.strip():
                    self.window.append_text(result.text.strip())
            except Exception as e:
                print(f"Transcription error: {e}")

            self.window.set_status("Listening...")
            last_transcribe = now

    def start(self):
        """Start audio capture and transcription."""
        import sounddevice as sd

        # Start audio stream
        self.stream = sd.InputStream(
            device=self.device_index,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=1024,
            callback=self._audio_callback,
        )
        self.stream.start()

        # Schedule initialization after UI loop starts
        self.window.root.after(100, self._delayed_init)

        # Run UI (blocks)
        try:
            self.window.run()
        finally:
            self._stop = True
            self.stream.stop()
            self.stream.close()

    def _delayed_init(self):
        """Initialize provider after UI is running."""
        def _init():
            self.initialize_provider()
            # Start transcription loop
            transcribe_thread = threading.Thread(target=self._transcription_loop, daemon=True)
            transcribe_thread.start()

        init_thread = threading.Thread(target=_init, daemon=True)
        init_thread.start()


def find_device_by_name(name: str) -> Optional[int]:
    """Find device index by name (fuzzy match)."""
    import sounddevice as sd

    devices = sd.query_devices()
    name_lower = name.lower()

    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            if name_lower in d["name"].lower():
                return i
    return None


def list_devices():
    """List all input devices."""
    import sounddevice as sd

    devices = sd.query_devices()
    print("Available input devices:")
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            print(f"  {i}: {d['name']}")
    print("\nUsage: python transcribe_audio.py --device 'AirPods'")


def main():
    parser = argparse.ArgumentParser(description="Live transcription of audio input")
    parser.add_argument("--setup", action="store_true", help="Set up Multi-Output Device for BlackHole")
    parser.add_argument("--device", type=str, help="Audio device name (e.g., 'AirPods', 'BlackHole')")
    parser.add_argument("--list", action="store_true", help="List available input devices")
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    if args.setup:
        setup_multi_output()
        return

    import sounddevice as sd

    # Find device
    device_index = None

    if args.device:
        device_index = find_device_by_name(args.device)
        if device_index is None:
            print(f"❌ Device '{args.device}' not found!")
            list_devices()
            sys.exit(1)
    else:
        # Default to BlackHole, fall back to first available
        device_index = find_blackhole_device()
        if device_index is None:
            print("No device specified and BlackHole not found.")
            print("Use --device to specify an input device.\n")
            list_devices()
            sys.exit(1)

        # Check Multi-Output setup for BlackHole
        if not check_multi_output_exists():
            print("\n⚠️  Multi-Output Device not detected.")
            print("   You may not hear audio while capturing.")
            print("   Run 'python transcribe_audio.py --setup' to configure.\n")

    device_info = sd.query_devices(device_index)
    print(f"✓ Using: {device_info['name']}")

    # Create window and start
    window = TranscriptionWindow()
    transcriber = AudioTranscriber(window, device_index)

    print("Starting live transcription... (Cmd+Q or Esc to quit)")
    transcriber.start()


if __name__ == "__main__":
    main()
