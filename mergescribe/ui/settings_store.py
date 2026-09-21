"""
Reading and writing settings, independent of any UI toolkit.

Settings live in ~/.mergescribe/settings.json and API keys in
~/.mergescribe/.env. Keeping this separate from the window means the file
format can be tested without a display.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional



def get_available_mics() -> List[str]:
    """Query available input devices from sounddevice."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        mics = []
        for d in devices:
            if d["max_input_channels"] > 0:
                mics.append(d["name"])
        return mics
    except Exception:
        return []


def load_settings() -> Dict[str, Any]:
    """Load settings from ~/.mergescribe/settings.json."""
    settings = {}
    user_settings = Path.home() / ".mergescribe" / "settings.json"
    if user_settings.exists():
        try:
            with open(user_settings) as f:
                settings.update(json.load(f))
        except Exception:
            pass
    return settings


def load_env_keys() -> Dict[str, str]:
    """Load API keys from .env files."""
    keys = {"OPENROUTER_API_KEY": ""}

    for env_path in [Path(".env"), Path.home() / ".mergescribe" / ".env"]:
        if env_path.exists():
            try:
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip("'\"")
                        if key in keys:
                            keys[key] = value
            except Exception:
                pass

    return keys


def save_settings(settings: Dict[str, Any]) -> None:
    """Save settings to ~/.mergescribe/settings.json."""
    settings_dir = Path.home() / ".mergescribe"
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_file = settings_dir / "settings.json"

    # Merge with existing
    existing = {}
    if settings_file.exists():
        try:
            with open(settings_file) as f:
                existing = json.load(f)
        except Exception:
            pass

    existing.update(settings)

    with open(settings_file, "w") as f:
        json.dump(existing, f, indent=2)


def remove_settings(keys: List[str]) -> None:
    """
    Delete keys from settings.json.

    save_settings merges, so it can never remove anything. Without this, a
    prompt reset to its default would keep the old override on disk forever.
    """
    settings_file = Path.home() / ".mergescribe" / "settings.json"
    if not settings_file.exists():
        return
    try:
        with open(settings_file) as f:
            existing = json.load(f)
    except Exception:
        return
    changed = False
    for key in keys:
        if key in existing:
            del existing[key]
            changed = True
    if changed:
        with open(settings_file, "w") as f:
            json.dump(existing, f, indent=2)


def save_env_keys(keys: Dict[str, str]) -> None:
    """Save API keys to ~/.mergescribe/.env."""
    env_dir = Path.home() / ".mergescribe"
    env_dir.mkdir(parents=True, exist_ok=True)
    env_file = env_dir / ".env"

    # Read existing lines (preserve non-key lines)
    existing_lines = []
    if env_file.exists():
        try:
            with open(env_file) as f:
                for line in f:
                    key = line.split("=")[0].strip() if "=" in line else ""
                    if key not in keys:
                        existing_lines.append(line.rstrip())
        except Exception:
            pass

    # Write back with updated keys
    with open(env_file, "w") as f:
        for line in existing_lines:
            f.write(line + "\n")
        for key, value in keys.items():
            if value:
                f.write(f"{key}={value}\n")


KNOWN_STT_MODELS = [
    ("microsoft/mai-transcribe-2", "MAI-Transcribe 2"),
    ("microsoft/mai-transcribe-1.5", "MAI-Transcribe 1.5"),
    ("openai/gpt-4o-transcribe", "gpt-4o-transcribe"),
    ("google/gemini-3.5-flash", "Gemini 3.5 Flash (audio-in)"),
    ("google/gemini-3.7-flash", "Gemini 3.7 Flash (audio-in)"),
    ("openai/gpt-transcribe", "gpt-transcribe"),
    ("fish-audio/transcribe-1", "Fish Audio Transcribe 1"),
    ("x-ai/grok-stt-1.0", "Grok STT 1.0"),
]

KNOWN_CORRECTION_MODELS = [
    ("google/gemini-3.1-flash-lite", "Gemini 3.1 Flash Lite"),
    ("google/gemini-3.1-flash-lite-preview", "Gemini 3.1 Flash Lite Preview"),
    ("google/gemini-3.5-flash", "Gemini 3.5 Flash"),
    ("google/gemini-3.5-flash-lite", "Gemini 3.5 Flash Lite"),
    ("google/gemini-3.6-flash", "Gemini 3.6 Flash"),
    ("google/gemini-3.7-flash", "Gemini 3.7 Flash"),
    ("moonshotai/kimi-k2.6", "Kimi K2.6"),
    ("x-ai/grok-build-0.1", "Grok Build 0.1"),
    ("x-ai/grok-4.3", "Grok 4.3"),
    ("openai/gpt-5.4-mini", "GPT-5.4 Mini"),
    ("openai/gpt-5.6-luna", "GPT-5.6 Luna"),
    ("openai/gpt-5.6-luna:nitro", "GPT-5.6 Luna (Nitro)"),
    ("openai/gpt-5.6-terra", "GPT-5.6 Terra"),
    ("z-ai/glm-5.2", "GLM 5.2"),
    ("anthropic/claude-opus-4.8-fast", "Claude Opus 4.8 (Fast)"),
    ("minimax/minimax-m2.7:nitro", "MiniMax M2.7 Nitro"),
]

_DEFAULT_OR_CORRECTION_MODEL = "google/gemini-3.1-flash-lite"


def _parse_model_ids(value: str) -> List[str]:
    """Parse comma/newline separated model IDs."""
    model_ids: List[str] = []
    seen = set()
    for raw in value.replace(",", "\n").splitlines():
        model_id = raw.strip()
        if model_id and model_id not in seen:
            seen.add(model_id)
            model_ids.append(model_id)
    return model_ids


def get_routing_status(
    openrouter_key: str,
    correction_model: str = "",
    provider_order: Optional[List[str]] = None,
    reasoning_effort: str = "",
) -> str:
    """Generate correction routing status based on the configured model."""
    if not openrouter_key:
        return "No OpenRouter key — transcriptions won't be corrected"

    model = correction_model or _DEFAULT_OR_CORRECTION_MODEL
    label = next((name for slug, name in KNOWN_CORRECTION_MODELS if slug == model), model)
    suffix = ""
    if provider_order:
        suffix += f" via {', '.join(provider_order)}"
    if reasoning_effort:
        suffix += f" ({reasoning_effort} reasoning)"
    return f"Correction → OpenRouter: {label}{suffix}"
