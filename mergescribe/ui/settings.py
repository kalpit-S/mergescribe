"""
Settings dialog using Flet.

A clean, minimal settings UI for MergeScribe.
Three tabs: Setup, API Keys, Custom Instructions.
"""

import json
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

import flet as ft

from ..validate import KeyValidator, ValidationResult


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
    """Load settings from settings.json."""
    settings = {}

    # Load from project root first
    if Path("settings.json").exists():
        try:
            with open("settings.json") as f:
                settings.update(json.load(f))
        except Exception:
            pass

    # Override with user settings
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
    keys = {"GROQ_API_KEY": "", "GEMINI_API_KEY": "", "OPENROUTER_API_KEY": ""}

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


def get_routing_status(groq_key: str, gemini_key: str, openrouter_key: str) -> str:
    """Generate routing status text based on available keys."""
    has_groq = bool(groq_key)
    has_gemini = bool(gemini_key)
    has_openrouter = bool(openrouter_key)

    if not has_groq and not has_gemini and not has_openrouter:
        return "No API keys configured"

    # Determine routing
    fast = "Groq GPT-OSS" if has_groq else ("Gemini 3 Flash" if has_gemini else "OpenRouter")
    smart = "Gemini 3 Flash" if has_gemini else ("OpenRouter" if has_openrouter else fast)

    if fast == smart:
        return f"All inputs: {fast}"
    else:
        return f"Short inputs: {fast}\nLong inputs: {smart}"


def settings_app(page: ft.Page) -> None:
    """Main settings app."""
    page.title = "MergeScribe Settings"
    page.window.width = 620
    page.window.height = 620
    page.padding = 20
    page.bgcolor = "#111318"
    page.theme_mode = ft.ThemeMode.DARK

    # Colors
    ACCENT = "#3B82F6"
    BG_CARD = "#1a1d24"
    BORDER = "#2a2f3a"
    TEXT = "#e5e7eb"
    TEXT_DIM = "#9ca3af"
    SUCCESS = "#22c55e"

    settings = load_settings()
    env_keys = load_env_keys()
    available_mics = get_available_mics()

    # State
    mic_checkboxes: Dict[str, ft.Checkbox] = {}
    validator = KeyValidator()

    # Cleanup validator on window close
    def on_window_close(e):
        if e.data == "close":
            validator.shutdown()
            page.window.destroy()

    page.window.on_event = on_window_close

    def snack(msg: str, color: str = ACCENT):
        page.snack_bar = ft.SnackBar(ft.Text(msg), bgcolor=color)
        page.snack_bar.open = True
        page.update()

    def card(title: str, content: List[ft.Control], subtitle: str = "") -> ft.Container:
        """Create a settings card."""
        return ft.Container(
            content=ft.Column([
                ft.Text(title, size=14, weight=ft.FontWeight.W_600, color=TEXT),
                ft.Text(subtitle, size=12, color=TEXT_DIM) if subtitle else ft.Container(),
                ft.Container(height=8),
                *content,
            ], spacing=4),
            bgcolor=BG_CARD,
            border=ft.border.all(1, BORDER),
            border_radius=8,
            padding=16,
            margin=ft.margin.only(bottom=12),
        )

    # ========== Setup Tab ==========

    # Microphones
    enabled_mics = settings.get("enabled_mics", settings.get("ENABLED_INPUT_DEVICES", []))
    mic_column = ft.Column(spacing=4)

    for mic in available_mics:
        cb = ft.Checkbox(
            label=mic[:40] + "..." if len(mic) > 40 else mic,
            value=mic in enabled_mics,
            active_color=ACCENT,
            data=mic,
        )
        mic_checkboxes[mic] = cb
        mic_column.controls.append(cb)

    if not available_mics:
        mic_column.controls.append(
            ft.Text("No microphones found", color=TEXT_DIM, italic=True)
        )

    # Trigger key
    trigger_key = settings.get("trigger_key", settings.get("TRIGGER_KEY", "alt_r"))
    trigger_dropdown = ft.Dropdown(
        value=trigger_key,
        options=[
            ft.dropdown.Option("alt_r", "Right Option"),
            ft.dropdown.Option("alt_l", "Left Option"),
            ft.dropdown.Option("ctrl_r", "Right Control"),
            ft.dropdown.Option("f17", "F17"),
            ft.dropdown.Option("f18", "F18"),
        ],
        width=200,
        border_color=BORDER,
        bgcolor="#0d0f12",
        text_size=13,
    )

    # Transcription providers
    enabled_providers = settings.get("enabled_providers", settings.get("ENABLED_PROVIDERS", ["parakeet"]))
    # Normalize names
    enabled_normalized = {p.replace("_mlx", "").replace("_whisper", "") for p in enabled_providers}

    # Check which keys are available
    has_groq_key = bool(env_keys.get("GROQ_API_KEY", ""))
    has_gemini_key = bool(env_keys.get("GEMINI_API_KEY", ""))

    provider_parakeet = ft.Checkbox(
        label="Parakeet (local)",
        value="parakeet" in enabled_normalized,
        active_color=ACCENT,
    )
    provider_groq = ft.Checkbox(
        label="Groq Whisper" + ("" if has_groq_key else " (needs key)"),
        value="groq" in enabled_normalized and has_groq_key,
        active_color=ACCENT,
        disabled=not has_groq_key,
    )
    provider_gemini = ft.Checkbox(
        label="Gemini" + ("" if has_gemini_key else " (needs key)"),
        value="gemini" in enabled_normalized and has_gemini_key,
        active_color=ACCENT,
        disabled=not has_gemini_key,
    )

    setup_tab = ft.Column([
        card(
            "Microphones",
            [mic_column],
            "Select which mics to record from.",
        ),
        card(
            "Trigger Key",
            [trigger_dropdown],
            "Hold to record, double-tap to toggle.",
        ),
        card(
            "Transcription Providers",
            [ft.Column([provider_parakeet, provider_groq, provider_gemini], spacing=4)],
            "Multiple providers = redundancy.",
        ),
    ], scroll=ft.ScrollMode.AUTO)

    # ========== API Keys Tab ==========

    # Status indicators for each key
    groq_status = ft.Text("", size=11, color=TEXT_DIM)
    gemini_status = ft.Text("", size=11, color=TEXT_DIM)
    openrouter_status = ft.Text("", size=11, color=TEXT_DIM)

    def make_status_row(field: ft.TextField, status: ft.Text) -> ft.Row:
        """Create a row with text field and status indicator."""
        return ft.Column([
            field,
            ft.Container(status, padding=ft.padding.only(left=4, top=2)),
        ], spacing=0)

    def update_status(status_text: ft.Text, result: ValidationResult) -> None:
        """Update status text based on validation result."""
        if result.valid:
            status_text.value = f"✓ Valid ({result.latency_ms}ms)"
            status_text.color = SUCCESS
        elif result.error == "No key":
            status_text.value = ""
        else:
            status_text.value = f"✗ {result.error}"
            status_text.color = "#ef4444"
        page.update()

    def on_groq_key_change(e):
        groq_status.value = "Testing..."
        groq_status.color = TEXT_DIM
        page.update()
        validator.validate_groq(
            groq_key_field.value,
            lambda r: update_status(groq_status, r)
        )
        update_routing_status()
        update_provider_states()

    def on_gemini_key_change(e):
        gemini_status.value = "Testing..."
        gemini_status.color = TEXT_DIM
        page.update()
        validator.validate_gemini(
            gemini_key_field.value,
            lambda r: update_status(gemini_status, r)
        )
        update_routing_status()
        update_provider_states()

    def on_openrouter_key_change(e):
        openrouter_status.value = "Testing..."
        openrouter_status.color = TEXT_DIM
        page.update()
        validator.validate_openrouter(
            openrouter_key_field.value,
            lambda r: update_status(openrouter_status, r)
        )
        update_routing_status()

    groq_key_field = ft.TextField(
        label="Groq API Key",
        value=env_keys.get("GROQ_API_KEY", ""),
        password=True,
        can_reveal_password=True,
        border_color=BORDER,
        bgcolor="#0d0f12",
        text_size=13,
        on_blur=on_groq_key_change,
    )

    gemini_key_field = ft.TextField(
        label="Gemini API Key",
        value=env_keys.get("GEMINI_API_KEY", ""),
        password=True,
        can_reveal_password=True,
        border_color=BORDER,
        bgcolor="#0d0f12",
        text_size=13,
        on_blur=on_gemini_key_change,
    )

    openrouter_key_field = ft.TextField(
        label="OpenRouter API Key (fallback)",
        value=env_keys.get("OPENROUTER_API_KEY", ""),
        password=True,
        can_reveal_password=True,
        border_color=BORDER,
        bgcolor="#0d0f12",
        text_size=13,
        on_blur=on_openrouter_key_change,
    )

    # Routing status display
    routing_status = ft.Text(
        get_routing_status(
            env_keys.get("GROQ_API_KEY", ""),
            env_keys.get("GEMINI_API_KEY", ""),
            env_keys.get("OPENROUTER_API_KEY", ""),
        ),
        size=12,
        color=TEXT_DIM,
    )

    def update_routing_status(_=None):
        routing_status.value = get_routing_status(
            groq_key_field.value,
            gemini_key_field.value,
            openrouter_key_field.value,
        )
        page.update()

    def update_provider_states():
        """Update provider checkbox states based on available keys."""
        has_groq = bool(groq_key_field.value)
        has_gemini = bool(gemini_key_field.value)

        provider_groq.disabled = not has_groq
        provider_groq.label = "Groq Whisper" + ("" if has_groq else " (needs key)")
        if not has_groq:
            provider_groq.value = False

        provider_gemini.disabled = not has_gemini
        provider_gemini.label = "Gemini" + ("" if has_gemini else " (needs key)")
        if not has_gemini:
            provider_gemini.value = False

        page.update()

    # Validate existing keys on startup
    def validate_on_startup():
        if env_keys.get("GROQ_API_KEY"):
            groq_status.value = "Testing..."
            page.update()
            validator.validate_groq(
                env_keys["GROQ_API_KEY"],
                lambda r: update_status(groq_status, r)
            )
        if env_keys.get("GEMINI_API_KEY"):
            gemini_status.value = "Testing..."
            page.update()
            validator.validate_gemini(
                env_keys["GEMINI_API_KEY"],
                lambda r: update_status(gemini_status, r)
            )
        if env_keys.get("OPENROUTER_API_KEY"):
            openrouter_status.value = "Testing..."
            page.update()
            validator.validate_openrouter(
                env_keys["OPENROUTER_API_KEY"],
                lambda r: update_status(openrouter_status, r)
            )

    # Run validation after page is ready
    threading.Timer(0.5, validate_on_startup).start()

    api_tab = ft.Column([
        card(
            "API Keys",
            [
                make_status_row(groq_key_field, groq_status),
                make_status_row(gemini_key_field, gemini_status),
                make_status_row(openrouter_key_field, openrouter_status),
            ],
            "Keys are validated when you click away. More keys = faster routing.",
        ),
        card(
            "LLM Correction Routing",
            [routing_status],
            "Automatically routes based on available keys.",
        ),
    ], scroll=ft.ScrollMode.AUTO)

    # ========== Custom Instructions Tab ==========

    custom_instructions = ft.TextField(
        label="Your Preferences",
        value=settings.get("custom_instructions", ""),
        multiline=True,
        min_lines=8,
        max_lines=12,
        border_color=BORDER,
        bgcolor="#0d0f12",
        text_size=13,
        hint_text="When in Twitter, use all lowercase.\nWhen in Mail, use a professional tone.",
    )

    instructions_tab = ft.Column([
        card(
            "Custom Instructions",
            [
                custom_instructions,
                ft.Container(height=8),
                ft.Text(
                    "These rules are applied to every transcription. "
                    "Reference the active app (e.g., 'When in Twitter...').",
                    size=12,
                    color=TEXT_DIM,
                ),
            ],
            "Personalize how corrections are made.",
        ),
    ], scroll=ft.ScrollMode.AUTO)

    # ========== Advanced Tab ==========

    # Default system prompt (for reference)
    DEFAULT_PROMPT = """You are a transcription assistant that cleans up speech-to-text output while preserving the speaker's authentic voice and exact meaning.

Clean up:
- Remove pure filler sounds: "um", "uh", "er", "ah", "hmm"
- Fix obvious transcription errors and typos (e.g. "lead code" → "leetcode")
- Handle self-corrections: use the correction, not the mistake
- Fix grammar and add proper punctuation

BE CONSERVATIVE - when in doubt, preserve the original words.

When multiple transcriptions are provided, compare and choose the most accurate parts from each.

Preserve:
- The speaker's meaning and intent
- Natural speaking style, slang, and strong language
- All substantive content

Meta-commands (follow these, don't transcribe them):
- "scratch that", "never mind" → remove the previous content

Return only the cleaned transcription text."""

    system_prompt_field = ft.TextField(
        label="System Prompt",
        value=settings.get("system_prompt", DEFAULT_PROMPT),
        multiline=True,
        min_lines=10,
        max_lines=16,
        border_color=BORDER,
        bgcolor="#0d0f12",
        text_size=12,
    )

    def reset_prompt(_=None):
        system_prompt_field.value = DEFAULT_PROMPT
        page.update()

    reset_prompt_btn = ft.TextButton(
        "Reset to Default",
        icon=ft.Icons.RESTORE,
        on_click=reset_prompt,
    )

    # Editing prompt
    DEFAULT_EDITING_PROMPT = "You are a text editing assistant. Apply the user's requested change precisely and return only the edited text."

    editing_prompt_field = ft.TextField(
        label="Editing System Prompt",
        value=settings.get("editing_prompt", DEFAULT_EDITING_PROMPT),
        multiline=True,
        min_lines=2,
        max_lines=4,
        border_color=BORDER,
        bgcolor="#0d0f12",
        text_size=12,
    )

    def reset_editing_prompt(_=None):
        editing_prompt_field.value = DEFAULT_EDITING_PROMPT
        page.update()

    reset_editing_btn = ft.TextButton(
        "Reset",
        icon=ft.Icons.RESTORE,
        on_click=reset_editing_prompt,
    )

    advanced_tab = ft.Column([
        card(
            "Transcription Prompt",
            [
                system_prompt_field,
                ft.Container(height=4),
                ft.Row([
                    ft.Text(
                        "Instructions for cleaning up speech-to-text.",
                        size=11,
                        color=TEXT_DIM,
                    ),
                    ft.Container(expand=True),
                    reset_prompt_btn,
                ]),
            ],
            "Controls how transcriptions are corrected.",
        ),
        card(
            "Editing Prompt",
            [
                editing_prompt_field,
                ft.Container(height=4),
                ft.Row([
                    ft.Text(
                        "Instructions for voice-commanded text edits.",
                        size=11,
                        color=TEXT_DIM,
                    ),
                    ft.Container(expand=True),
                    reset_editing_btn,
                ]),
            ],
            "Controls how 'select + speak' edits work.",
        ),
    ], scroll=ft.ScrollMode.AUTO)

    # ========== Tabs ==========

    tabs = ft.Tabs(
        tabs=[
            ft.Tab(text="Setup", icon=ft.Icons.MIC, content=ft.Container(setup_tab, padding=12)),
            ft.Tab(text="API Keys", icon=ft.Icons.KEY, content=ft.Container(api_tab, padding=12)),
            ft.Tab(text="Instructions", icon=ft.Icons.EDIT_NOTE, content=ft.Container(instructions_tab, padding=12)),
            ft.Tab(text="Advanced", icon=ft.Icons.TUNE, content=ft.Container(advanced_tab, padding=12)),
        ],
        expand=True,
        indicator_color=ACCENT,
        label_color=TEXT,
        unselected_label_color=TEXT_DIM,
        divider_color=BORDER,
    )

    # ========== Actions ==========

    def save_all(_=None):
        try:
            new_settings = {}

            # Mics
            new_settings["enabled_mics"] = [
                mic for mic, cb in mic_checkboxes.items() if cb.value
            ]

            # Trigger
            new_settings["trigger_key"] = trigger_dropdown.value

            # Providers
            providers = []
            if provider_parakeet.value:
                providers.append("parakeet")
            if provider_groq.value:
                providers.append("groq")
            if provider_gemini.value:
                providers.append("gemini")
            new_settings["enabled_providers"] = providers

            # Custom instructions
            new_settings["custom_instructions"] = custom_instructions.value

            # Advanced settings - only save prompts if they differ from defaults
            if system_prompt_field.value.strip() != DEFAULT_PROMPT.strip():
                new_settings["system_prompt"] = system_prompt_field.value
            if editing_prompt_field.value.strip() != DEFAULT_EDITING_PROMPT.strip():
                new_settings["editing_prompt"] = editing_prompt_field.value

            save_settings(new_settings)

            # Save API keys
            save_env_keys({
                "GROQ_API_KEY": groq_key_field.value,
                "GEMINI_API_KEY": gemini_key_field.value,
                "OPENROUTER_API_KEY": openrouter_key_field.value,
            })

            snack("Settings saved!", SUCCESS)

        except Exception as e:
            snack(f"Error: {e}", "#ef4444")

    def close_window(_=None):
        page.window.close()

    save_btn = ft.ElevatedButton(
        "Save",
        icon=ft.Icons.SAVE,
        on_click=save_all,
        style=ft.ButtonStyle(bgcolor=ACCENT, color="white"),
    )

    close_btn = ft.TextButton(
        "Close",
        on_click=close_window,
    )

    # ========== Layout ==========

    page.add(
        ft.Row([
            ft.Icon(ft.Icons.SETTINGS, color=ACCENT, size=24),
            ft.Text("Settings", size=18, weight=ft.FontWeight.W_600),
        ], spacing=10),
        ft.Container(height=8),
        ft.Container(content=tabs, expand=True),
        ft.Divider(color=BORDER),
        ft.Row([close_btn, ft.Container(expand=True), save_btn]),
    )


def run_settings():
    """Run the settings app."""
    ft.app(target=settings_app)


if __name__ == "__main__":
    run_settings()
