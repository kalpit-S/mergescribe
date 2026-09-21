"""
Main entry point for MergeScribe.

Run with: python -m mergescribe
"""

import signal
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from pynput import keyboard

from .config import Config
from .audio import AudioEngine
from .input import InputController
from .session import Session, SessionManager
from .metrics import MetricsWriter, get_metrics
from .training import TrainingDataWriter
from .providers import ProviderRegistry
from .providers.parakeet import ParakeetProvider
from .providers.openrouter_stt import OpenRouterSTTProvider
from .ui.menu_bar import MenuBarApp
from .ui.hud import RecordingHUD


# Global state
config: Config
audio_engine: AudioEngine
session_manager: SessionManager
input_controller: InputController
metrics: MetricsWriter
training_writer: Optional[TrainingDataWriter] = None
menu_bar: MenuBarApp
hud: Optional[RecordingHUD] = None
current_session: Optional[Session] = None
_keyboard_listener: Optional[keyboard.Listener] = None


class _Tee:
    """Mirror stdout/stderr to a log file so sessions can be inspected later."""

    def __init__(self, stream, log_path: Path):
        self._stream = stream
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Simple rotation: start fresh when the log gets big
        if log_path.exists() and log_path.stat().st_size > 5_000_000:
            log_path.rename(log_path.with_suffix(".log.old"))
        self._file = open(log_path, "a", buffering=1)
        self._file.write(f"\n===== session start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")

    def write(self, data):
        self._stream.write(data)
        try:
            self._file.write(data)
        except Exception:
            pass

    def flush(self):
        self._stream.flush()
        try:
            self._file.flush()
        except Exception:
            pass


def _setup_logging() -> None:
    """Tee console output to ~/.mergescribe/logs/mergescribe.log."""
    log_path = Path.home() / ".mergescribe" / "logs" / "mergescribe.log"
    try:
        sys.stdout = _Tee(sys.stdout, log_path)
        sys.stderr = _Tee(sys.stderr, log_path)
    except Exception as e:
        print(f"Log tee setup failed: {e}")


def main():
    """Main entry point."""
    global config, audio_engine, session_manager, input_controller, metrics, training_writer, menu_bar, hud, _keyboard_listener

    _setup_logging()
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
    _sync_providers(providers, config)

    # Initialize audio engine
    audio_engine = AudioEngine(config)
    active_mics = audio_engine.initialize()
    print(f"  Active mics: {active_mics}")

    # Recording HUD (reads the live mic level straight off the audio engine)
    hud = RecordingHUD(
        level_source=lambda: audio_engine.current_level,
        enabled=config.hud_enabled,
    )

    # Initialize session manager
    session_manager = SessionManager(
        config_snapshot_fn=config.snapshot,
        providers=providers,
        metrics=metrics,
        training_writer=training_writer,
        observer=hud,
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
    set_status("idle")

    print(f"Ready! Press {config.trigger_key} to record.")
    print("Press Ctrl+C to quit.")

    # Show startup notification
    menu_bar.show_notification("Started", f"Press {config.trigger_key} to record")

    # Run menu bar (blocks)
    try:
        menu_bar.run()
    finally:
        shutdown()


def _provider_signature(provider) -> tuple:
    """Return the runtime identity that requires provider replacement when changed."""
    if isinstance(provider, ParakeetProvider):
        return ("parakeet",)
    if isinstance(provider, OpenRouterSTTProvider):
        return ("openrouter", provider.api_key, provider.model)
    return (provider.__class__.__name__, getattr(provider, "name", ""))


def _desired_provider_factories(config: Config) -> dict[str, tuple[Callable[[], object], tuple]]:
    """Build the desired transcription provider set from the current config."""
    desired: dict[str, tuple[Callable[[], object], tuple]] = {}

    for name in config.enabled_providers:
        if name == "parakeet":
            desired["parakeet"] = (lambda: ParakeetProvider(), ("parakeet",))
        else:
            print(f"  Unknown or unconfigured provider: {name}")

    if config.openrouter_api_key and config.openrouter_stt_models:
        for model in config.openrouter_stt_models:
            provider = OpenRouterSTTProvider(config.openrouter_api_key, model)
            desired[provider.name] = (
                lambda key=config.openrouter_api_key, model=model: OpenRouterSTTProvider(key, model),
                ("openrouter", config.openrouter_api_key, model),
            )

    return desired


def _sync_providers(registry: ProviderRegistry, config: Config) -> None:
    """Make registered transcription providers match current settings."""
    desired = _desired_provider_factories(config)

    for name in registry.names():
        provider = registry.get(name)
        if provider is None:
            continue
        if name not in desired or _provider_signature(provider) != desired[name][1]:
            print(f"  Removing provider: {name}")
            registry.unregister(name)

    for name, (factory, _) in desired.items():
        if registry.get(name) is not None:
            continue
        try:
            registry.register(factory())
        except Exception as e:
            print(f"  Failed to init provider {name}: {e}")


def set_status(status: str) -> None:
    """
    Show recording state in exactly one place.

    The HUD sits where you're actually looking, so when it's live the menu bar
    icon stays put instead of flashing red alongside it. If the HUD is turned
    off or has failed, the icon takes the job back rather than leaving no
    indication at all.
    """
    if hud is not None and hud.is_showing:
        menu_bar.set_status("idle")
    else:
        menu_bar.set_status(status)

    if hud is not None:
        hud.set_status(status)


def on_start() -> None:
    """Called when recording should start."""
    global current_session

    if session_manager.is_busy():
        session_manager.start_session()
        return

    config.reload()
    active_mics = audio_engine.sync_configured_mics()
    _sync_providers(session_manager.providers, config)

    if not active_mics:
        menu_bar.show_error("No active microphones")
        return

    if not session_manager.providers.values():
        menu_bar.show_error("No transcription providers")
        return

    session = session_manager.start_session()
    if session is None:
        # Busy - error sound already played
        return

    current_session = session

    # Override session completion callback to update menu bar
    original_on_complete = session.on_complete
    def on_session_complete(s):
        # A session still typing can finish after the next recording began;
        # only the newest one may put the status back to idle.
        if session_manager.active_session in (None, s):
            set_status("idle")
        original_on_complete(s)

    session.on_complete = on_session_complete

    # Capture audio first, then let the session gather context concurrently:
    # the press-to-record gap is what the user feels.
    audio_engine.on_chunk_ready = session.on_chunk_ready
    audio_engine.start_recording()
    set_status("recording")

    session.start(mics=active_mics)
    metrics.log("recording_started", session_id=str(session.id))


def on_stop() -> None:
    """Called when recording should stop."""
    global current_session

    if current_session is None:
        return

    # Get final chunk (disconnects callback)
    final_chunk = audio_engine.stop_recording()

    set_status("processing")

    # Finalize in background (will set status back to idle via callback)
    current_session.finalize(final_chunk)

    session_id = str(current_session.id)
    current_session = None

    metrics.log("recording_stopped", session_id=session_id)


def on_settings() -> None:
    """
    Open the settings window.

    Called from a rumps menu item, so this is already the main thread and the
    window can be built directly; AppKit is only ever touched from here.
    """
    try:
        from .ui.settings import open_settings
        open_settings()
    except Exception as e:
        print(f"Settings window failed to open: {e}")
        menu_bar.show_error("Could not open settings")


def shutdown() -> None:
    """Clean shutdown."""
    print("\nShutting down...")

    # Stop keyboard listener
    if _keyboard_listener:
        _keyboard_listener.stop()

    if hud is not None:
        hud.shutdown()

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
