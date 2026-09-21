"""
Configuration management with immutable snapshots.

Loads from: environment variables > settings.json > defaults
Provides immutable snapshots for session isolation.
"""

from pathlib import Path
from typing import List
import json
import os

from .types import ConfigSnapshot


# Defaults
DEFAULT_CONFIG = {
    # Audio
    "enabled_mics": [],
    "preroll_seconds": 1.0,  # 1 second is enough to catch speech start
    # 1.2s sits just above the p90 of measured pause length (1.09s), so chunks
    # break at sentence boundaries without cutting the frequent sub-second
    # thinking pauses. At the old 2.0s only 2.6% of pauses ever qualified.
    "silence_threshold": 1.2,
    "sample_rate": 16000,
    "chunk_on_silence": True,
    "adaptive_threshold_enabled": True,
    "speech_headroom_db": 12.0,
    "speech_hysteresis_db": 3.0,
    "noise_floor_percentile": 10,

    # Input
    "trigger_key": "alt_r",
    "double_tap_threshold": 0.45,
    "toggle_mode_timeout": 600,

    # Providers
    "enabled_providers": ["parakeet"],

    # Processing
    "consensus_threshold": 2,
    "consensus_max_words": 15,

    # A chunk waits for its slowest provider. Once the fastest has answered,
    # give the rest this much longer (1.0 = twice the fastest provider's time)
    # before giving up, but never cut before provider_deadline_min_ms.
    #
    # The floor is deliberately high. Cloud STT is slower than local parakeet on
    # 85% of chunks, so a low floor doesn't trim a tail — it systematically
    # drops the cloud provider, which is the more accurate one. At 900ms that
    # was 1054 cloud cuts to 160 local; at 2000ms it inverts to 114 local / 55
    # cloud and only fires on a genuinely stuck provider. That costs ~65ms of
    # the available saving and buys back the better transcript.
    # Lower it to 900 to reclaim 114ms mean / 313ms p90 at that cost.
    "provider_deadline_multiplier": 1.0,
    "provider_deadline_min_ms": 2000,

    # OpenRouter STT and correction
    "openrouter_stt_models": [],
    "openrouter_correction_model": "google/gemini-3.1-flash-lite",
    "openrouter_correction_provider_order": [],
    "openrouter_correction_allow_fallbacks": True,
    "openrouter_correction_reasoning_effort": "",

    # User customization
    "custom_instructions": "",

    # Voice-driven output routing (AX field inventory + TARGET prefix).
    # Experimental and off by default: a wrong destination is far more
    # disruptive than no routing at all, since the text has to be undone
    # and re-dictated somewhere else.
    "field_routing_enabled": False,
    "routing_allowed_apps": [],   # empty = all on-screen apps are eligible
    "routing_instructions": "",   # user rules for where dictation should go

    # Advanced settings
    "system_prompt": "",
    "editing_prompt": "",

    # Training data (local only, opt-in)
    "training_enabled": False,

    # Watch the destination field after typing and record whether the user
    # edited the text — implicit correction signal for prompt/model tuning.
    "edit_feedback_enabled": True,

    # Terms corrected the same way in two separate dictations are offered to
    # the correction model as vocabulary. Candidates, not substitution rules.
    "learn_vocabulary": True,

    # Floating recording HUD. The menu bar icon is at the top of the screen
    # while you're looking at the field you're dictating into, so on its own it
    # is not feedback at all.
    "hud_enabled": True,

    # Separate consecutive dictations with a space. Applied as a leading space
    # only when continuing a recent dictation into the same destination, so a
    # single recording leaves no trailing whitespace behind.
    "space_between_dictations": True,

}


class Config:
    """
    Single source of truth for all settings.

    Usage:
        config = Config.load()
        snapshot = config.snapshot()  # Immutable copy for session
    """

    def __init__(self):
        # Audio
        self.enabled_mics: List[str] = []
        self.preroll_seconds: float = 1.0
        self.silence_threshold: float = 1.2
        self.sample_rate: int = 16000
        self.chunk_on_silence: bool = True
        self.adaptive_threshold_enabled: bool = True
        self.speech_headroom_db: float = 12.0
        self.speech_hysteresis_db: float = 3.0
        self.noise_floor_percentile: int = 10

        # Input
        self.trigger_key: str = "alt_r"
        self.double_tap_threshold: float = 0.45
        self.toggle_mode_timeout: float = 600

        # Providers
        self.enabled_providers: List[str] = ["parakeet"]

        # Processing
        self.consensus_threshold: int = 2
        self.consensus_max_words: int = 15
        self.provider_deadline_multiplier: float = 1.0
        self.provider_deadline_min_ms: int = 2000

        # API Keys
        self.openrouter_api_key: str = ""

        # OpenRouter STT: list of models to run in parallel for transcription
        self.openrouter_stt_models: List[str] = []

        # OpenRouter correction: which model to use for LLM correction
        self.openrouter_correction_model: str = "google/gemini-3.1-flash-lite"
        self.openrouter_correction_provider_order: List[str] = []
        self.openrouter_correction_allow_fallbacks: bool = True
        self.openrouter_correction_reasoning_effort: str = ""

        # User customization
        self.custom_instructions: str = ""

        # Voice-driven output routing
        self.field_routing_enabled: bool = False
        self.routing_allowed_apps: List[str] = []
        self.routing_instructions: str = ""

        # Advanced settings
        self.system_prompt: str = ""
        self.editing_prompt: str = ""

        # Paths
        self.data_dir: Path = Path.home() / ".mergescribe"
        self.metrics_file: Path = self.data_dir / "metrics.jsonl"
        self.settings_file: Path = self.data_dir / "settings.json"
        self.env_file: Path = self.data_dir / ".env"

        # Training data collection (local only, opt-in)
        self.training_enabled: bool = False
        self.training_data_dir: Path = self.data_dir / "training"

        # Post-output edit detection
        self.edit_feedback_enabled: bool = True
        self.learn_vocabulary: bool = True

        # Floating recording HUD
        self.hud_enabled: bool = True

        # Separator between consecutive dictations
        self.space_between_dictations: bool = True


    @classmethod
    def load(cls) -> "Config":
        """Load configuration from all sources."""
        config = cls()
        config._ensure_data_dir()
        config._load_env()
        config._load_settings()
        return config

    def _ensure_data_dir(self) -> None:
        """Create data directory if it doesn't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _load_env(self) -> None:
        """Load API keys from .env file and environment."""
        # Check for .env in current directory (project root) for backward compat
        env_file = Path(".env")
        if env_file.exists():
            self._parse_env_file(env_file)

        # Also check ~/.mergescribe/.env
        if self.env_file.exists():
            self._parse_env_file(self.env_file)

        # Environment variables override file values
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", self.openrouter_api_key)

    def _parse_env_file(self, env_file: Path) -> None:
        """Parse a .env file and extract API keys."""
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("'\"")

                    if key == "OPENROUTER_API_KEY":
                        self.openrouter_api_key = value
        except Exception as e:
            print(f"Error loading {env_file}: {e}")

    def _load_settings(self) -> None:
        """Load settings from ~/.mergescribe/settings.json."""
        if self.settings_file.exists():
            self._apply_settings_file(self.settings_file)

    def reload(self) -> None:
        """Refresh environment and settings from disk."""
        for key, default in DEFAULT_CONFIG.items():
            if isinstance(default, list):
                value = list(default)
            else:
                value = default
            setattr(self, key, value)
        self.openrouter_api_key = ""
        self._load_env()
        self._load_settings()

    def _apply_settings_file(self, settings_file: Path) -> None:
        """Apply settings from a JSON file."""
        try:
            with open(settings_file) as f:
                data = json.load(f)

            # Map old config keys to new ones
            key_mapping = {
                "ENABLED_INPUT_DEVICES": "enabled_mics",
                "ENABLED_PROVIDERS": "enabled_providers",
                "TRIGGER_KEY": "trigger_key",
            }

            for old_key, new_key in key_mapping.items():
                if old_key in data:
                    value = data[old_key]
                    # Convert provider names if needed
                    if old_key == "ENABLED_PROVIDERS":
                        value = [p.replace("_mlx", "").replace("_whisper", "") for p in value]
                    setattr(self, new_key, value)

            # Apply settings with type validation
            for key, default in DEFAULT_CONFIG.items():
                if key in data:
                    setattr(self, key, type(default)(data[key]))

        except Exception as e:
            print(f"Error loading {settings_file}: {e}")

    def save_settings(self) -> None:
        """Save current settings to settings.json."""
        data = {
            "enabled_mics": self.enabled_mics,
            "enabled_providers": self.enabled_providers,
            "custom_instructions": self.custom_instructions,
            "trigger_key": self.trigger_key,
            "double_tap_threshold": self.double_tap_threshold,
            "silence_threshold": self.silence_threshold,
            "chunk_on_silence": self.chunk_on_silence,
            "adaptive_threshold_enabled": self.adaptive_threshold_enabled,
            "speech_headroom_db": self.speech_headroom_db,
            "speech_hysteresis_db": self.speech_hysteresis_db,
            "noise_floor_percentile": self.noise_floor_percentile,
            "openrouter_stt_models": self.openrouter_stt_models,
            "openrouter_correction_model": self.openrouter_correction_model,
            "openrouter_correction_provider_order": self.openrouter_correction_provider_order,
            "openrouter_correction_allow_fallbacks": self.openrouter_correction_allow_fallbacks,
            "openrouter_correction_reasoning_effort": self.openrouter_correction_reasoning_effort,
            "system_prompt": self.system_prompt,
            "editing_prompt": self.editing_prompt,
            "training_enabled": self.training_enabled,
            "field_routing_enabled": self.field_routing_enabled,
            "routing_allowed_apps": self.routing_allowed_apps,
            "routing_instructions": self.routing_instructions,
            "edit_feedback_enabled": self.edit_feedback_enabled,
            "learn_vocabulary": self.learn_vocabulary,
            "provider_deadline_multiplier": self.provider_deadline_multiplier,
            "provider_deadline_min_ms": self.provider_deadline_min_ms,
            "hud_enabled": self.hud_enabled,
            "space_between_dictations": self.space_between_dictations,
        }

        self._ensure_data_dir()
        with open(self.settings_file, "w") as f:
            json.dump(data, f, indent=2)

    def snapshot(self) -> ConfigSnapshot:
        """Return immutable copy for session isolation, reloading settings from disk first."""
        self.reload()
        return ConfigSnapshot(
            enabled_mics=list(self.enabled_mics),
            preroll_seconds=self.preroll_seconds,
            silence_threshold=self.silence_threshold,
            sample_rate=self.sample_rate,
            double_tap_threshold=self.double_tap_threshold,
            toggle_mode_timeout=self.toggle_mode_timeout,
            enabled_providers=list(self.enabled_providers),
            consensus_threshold=self.consensus_threshold,
            consensus_max_words=self.consensus_max_words,
            space_between_dictations=self.space_between_dictations,
            provider_deadline_multiplier=self.provider_deadline_multiplier,
            provider_deadline_min_ms=self.provider_deadline_min_ms,
            openrouter_api_key=self.openrouter_api_key,
            openrouter_stt_models=list(self.openrouter_stt_models),
            openrouter_correction_model=self.openrouter_correction_model,
            openrouter_correction_provider_order=list(self.openrouter_correction_provider_order),
            openrouter_correction_allow_fallbacks=self.openrouter_correction_allow_fallbacks,
            openrouter_correction_reasoning_effort=self.openrouter_correction_reasoning_effort,
            custom_instructions=self.custom_instructions,
            system_prompt=self.system_prompt,
            editing_prompt=self.editing_prompt,
            training_enabled=self.training_enabled,
            training_data_dir=str(self.training_data_dir),
            field_routing_enabled=self.field_routing_enabled,
            routing_allowed_apps=list(self.routing_allowed_apps),
            routing_instructions=self.routing_instructions,
            edit_feedback_enabled=self.edit_feedback_enabled,
            learn_vocabulary=self.learn_vocabulary,
        )
