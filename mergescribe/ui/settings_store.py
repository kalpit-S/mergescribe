"""
Reading and writing settings, independent of any UI toolkit.

Settings live in ~/.mergescribe/settings.json and API keys in
~/.mergescribe/.env. Keeping this separate from the window means the file
format can be tested without a display.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple



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
            existing = json.loads(settings_file.read_text())
        except (OSError, ValueError) as e:
            # Merging into nothing would write back only these keys and lose
            # every other setting; leave the file for a person to look at.
            print(f"[Settings] {settings_file} is unreadable ({e}); not saving")
            return

    existing.update(settings)
    _write_json(settings_file, existing)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    """Replace the file in one step, so a crash mid-write can't leave half of it."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


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
        _write_json(settings_file, existing)


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


# Current models only; anything else can still be typed in as a custom ID.
# Every entry was checked to work with this app's settings (Sep 2026).
KNOWN_STT_MODELS = [
    ("microsoft/mai-transcribe-2", "MAI-Transcribe 2"),
    ("assemblyai/universal-3-5-pro", "Universal 3.5 Pro"),
    ("fish-audio/transcribe-1-pro", "Fish Audio Transcribe 1 Pro"),
    ("openai/gpt-transcribe", "GPT Transcribe"),
]

# Fast enough for dictation: first token within about 1.5s.
KNOWN_CORRECTION_MODELS = [
    ("openai/gpt-6-luna", "GPT-6 Luna"),
    ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5"),
    ("moonshotai/kimi-k3", "Kimi K3"),
    ("google/gemini-3.5-flash-lite", "Gemini 3.5 Flash Lite"),
    ("google/gemini-3.8-flash", "Gemini 3.8 Flash"),
]

_DEFAULT_OR_CORRECTION_MODEL = "openai/gpt-6-luna"


def fetch_transcription_models(api_key: str = "", timeout: float = 5.0) -> List[Tuple[str, str]]:
    """
    OpenRouter's speech-to-text models as (id, name), newest first; [] when
    it can't be reached. Every one is served by /audio/transcriptions.
    """
    import requests
    try:
        response = requests.get("https://openrouter.ai/api/v1/models",
                                params={"output_modalities": "transcription"},
                                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                                timeout=timeout)
        models = response.json().get("data", [])
    except Exception:
        return []
    # "Microsoft AI: MAI-Transcribe 2" -> "MAI-Transcribe 2"
    return [(m["id"], str(m.get("name") or m["id"]).split(": ", 1)[-1])
            for m in models if isinstance(m, dict) and m.get("id")]


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
