"""
Configuration management with immutable snapshots.

Loads from: environment variables > settings.json > defaults
Provides immutable snapshots for session isolation.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import json
import os

from .types import ConfigSnapshot


# Defaults
DEFAULT_CONFIG = {
    # Audio
    "enabled_mics": [],
    "preroll_seconds": 1.0,  # 1 second is enough to catch speech start
    "silence_threshold": 2.0,
    "sample_rate": 16000,

    # Input
    "double_tap_threshold": 0.3,
    "toggle_mode_timeout": 600,

    # Providers
    "enabled_providers": ["parakeet"],

    # Processing
    "consensus_threshold": 2,
    "consensus_max_words": 15,

    # User customization
    "custom_instructions": "",

    # Advanced settings
    "system_prompt": "",
    "editing_prompt": "",

    # Training data (local only, opt-in)
    "training_enabled": False,
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
        self.silence_threshold: float = 2.0
        self.sample_rate: int = 16000

        # Input
        self.trigger_key: str = "alt_r"
        self.double_tap_threshold: float = 0.3
        self.toggle_mode_timeout: float = 600

        # Providers
        self.enabled_providers: List[str] = ["parakeet"]

        # Processing
        self.consensus_threshold: int = 2
        self.consensus_max_words: int = 15

        # API Keys
        self.openrouter_api_key: str = ""
        self.groq_api_key: str = ""
        self.gemini_api_key: str = ""

        # User customization
        self.custom_instructions: str = ""

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
        self.groq_api_key = os.getenv("GROQ_API_KEY", self.groq_api_key)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", self.gemini_api_key)

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
                    elif key == "GROQ_API_KEY":
                        self.groq_api_key = value
                    elif key == "GEMINI_API_KEY":
                        self.gemini_api_key = value
        except Exception as e:
            print(f"Error loading {env_file}: {e}")

    def _load_settings(self) -> None:
        """Load settings from settings.json."""
        # Check project root first for backwards compatibility
        project_settings = Path("settings.json")
        if project_settings.exists():
            self._apply_settings_file(project_settings)

        # Then check ~/.mergescribe/settings.json (overrides)
        if self.settings_file.exists():
            self._apply_settings_file(self.settings_file)

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
        }

        self._ensure_data_dir()
        with open(self.settings_file, "w") as f:
            json.dump(data, f, indent=2)

    def snapshot(self) -> ConfigSnapshot:
        """Return immutable copy for session isolation."""
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
            openrouter_api_key=self.openrouter_api_key,
            groq_api_key=self.groq_api_key,
            gemini_api_key=self.gemini_api_key,
            custom_instructions=self.custom_instructions,
            system_prompt=self.system_prompt,
            editing_prompt=self.editing_prompt,
            training_enabled=self.training_enabled,
            training_data_dir=str(self.training_data_dir),
        )
