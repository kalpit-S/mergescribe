"""
Main entry point for MergeScribe.

Run with: python -m mergescribe
"""

import signal
import subprocess
import sys
import threading
from typing import Optional

from pynput import keyboard

from .config import Config
from .audio import AudioEngine
from .input import InputController
from .session import Session, SessionManager
from .metrics import get_metrics
from .training import TrainingDataWriter
from .providers import ProviderRegistry
from .providers.parakeet import ParakeetProvider
from .providers.groq import GroqProvider
from .providers.gemini import GeminiProvider
from .ui.menu_bar import MenuBarApp


# Global state
config: Config
audio_engine: AudioEngine
session_manager: SessionManager
input_controller: InputController
metrics: "MetricsWriter"
training_writer: Optional[TrainingDataWriter] = None
menu_bar: MenuBarApp
current_session: Optional[Session] = None
_keyboard_listener: Optional[keyboard.Listener] = None


def main():
    """Main entry point."""
    global config, audio_engine, session_manager, input_controller, metrics, training_writer, menu_bar, _keyboard_listener

    print("MergeScribe v2.0.0 starting...")

    # Load configuration
    config = Config.load()
    print(f"  Enabled mics: {config.enabled_mics}")
    print(f"  Enabled providers: {config.enabled_providers}")

    # Initialize metrics
    metrics = get_metrics(config.metrics_file)

    # Initialize training data writer (if enabled)
    if config.training_enabled:
        training_writer = TrainingDataWriter(
            config.training_data_dir,
            sample_rate=config.sample_rate
        )
        print(f"  Training data: {config.training_data_dir}")
    else:
        training_writer = None

    # Initialize providers
    providers = ProviderRegistry()
    _init_providers(providers, config)

    # Initialize audio engine
    audio_engine = AudioEngine(config)
    active_mics = audio_engine.initialize()
    print(f"  Active mics: {active_mics}")

    # Initialize session manager
    session_manager = SessionManager(
        config_snapshot_fn=config.snapshot,
        providers=providers,
        metrics=metrics,
        training_writer=training_writer,
    )

    # Initialize input controller
    input_controller = InputController(config)
    input_controller.on_start_recording = on_start
    input_controller.on_stop_recording = on_stop

    # Setup signal handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Start keyboard listener
    _keyboard_listener = keyboard.Listener(
        on_press=input_controller.on_key_press,
        on_release=input_controller.on_key_release
    )
    _keyboard_listener.start()
    print("  Keyboard listener started")

    # Initialize and run menu bar
    menu_bar = MenuBarApp()
    menu_bar.on_settings = on_settings
    menu_bar.set_status("idle")

    print(f"Ready! Press {config.trigger_key} to record.")
    print("Press Ctrl+C to quit.")

    # Show startup notification
    menu_bar.show_notification("Started", f"Press {config.trigger_key} to record")

    # Run menu bar (blocks)
    try:
        menu_bar.run()
    finally:
        shutdown()


def _init_providers(registry: ProviderRegistry, config: Config) -> None:
    """Initialize enabled providers."""
    for name in config.enabled_providers:
        try:
            if name == "parakeet":
                registry.register(ParakeetProvider())
            elif name == "groq" and config.groq_api_key:
                registry.register(GroqProvider(config.groq_api_key))
            elif name == "gemini" and config.gemini_api_key:
                registry.register(GeminiProvider(config.gemini_api_key))
            else:
                print(f"  Unknown or unconfigured provider: {name}")
        except Exception as e:
            print(f"  Failed to init provider {name}: {e}")


def on_start() -> None:
    """Called when recording should start."""
    global current_session

    session = session_manager.start_session()
    if session is None:
        # Busy - error sound already played
        return

    current_session = session

    # Override session completion callback to update menu bar
    original_on_complete = session.on_complete
    def on_session_complete(s):
        menu_bar.set_status("idle")
        original_on_complete(s)

    session.on_complete = on_session_complete
    session.start()

    # Wire up chunk callback
    audio_engine.on_chunk_ready = session.on_chunk_ready
    audio_engine.start_recording()

    menu_bar.set_status("recording")
    metrics.log("recording_started", session_id=str(session.id))


def on_stop() -> None:
    """Called when recording should stop."""
    global current_session

    if current_session is None:
        return

    # Get final chunk (disconnects callback)
    final_chunk = audio_engine.stop_recording()

    menu_bar.set_status("processing")

    # Finalize in background (will set status back to idle via callback)
    current_session.finalize(final_chunk)

    session_id = str(current_session.id)
    current_session = None

    metrics.log("recording_stopped", session_id=session_id)


def on_settings() -> None:
    """Open settings dialog."""
    # Run settings UI in subprocess (Flet runs its own event loop)
    subprocess.Popen(
        [sys.executable, "-m", "mergescribe.ui.settings"],
        start_new_session=True,
    )
    print("Settings dialog opened")


def shutdown() -> None:
    """Clean shutdown."""
    print("\nShutting down...")

    # Stop keyboard listener
    if _keyboard_listener:
        _keyboard_listener.stop()

    audio_engine.shutdown()
    session_manager.providers.shutdown()
    metrics.shutdown()

    if training_writer:
        training_writer.shutdown()

    print("Goodbye!")


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM."""
    shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
