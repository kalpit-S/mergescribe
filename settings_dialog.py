"""Settings UI (Flet)."""
from typing import Any

import flet as ft

from config_manager import ConfigManager
from mic_manager import (
    list_input_devices,
    find_device_by_name,
    get_device_display_label,
    save_preferred_device,
    MicDevice,
)


def settings_app(page: ft.Page) -> None:
    page.title = "MergeScribe Settings"
    page.window_width = 900
    page.window_height = 700
    page.padding = 16
    page.bgcolor = "#0F1115"  # Deep dark background
    page.theme_mode = ft.ThemeMode.DARK

    # Modern color palette
    ACCENT = "#3B82F6"        # Bright Blue
    BG_DARK = "#0F1115"
    BG_CARD = "#181B21"       # Slightly lighter for cards
    BORDER = "#272A32"
    TEXT_MAIN = "#F3F4F6"
    TEXT_MUTED = "#9CA3AF"
    SUCCESS = "#10B981"
    ERROR = "#EF4444"

    page.theme = ft.Theme(
        color_scheme_seed=ACCENT,
        font_family="Inter, system-ui, sans-serif",
    )

    config = ConfigManager()

    def show_snack(message: str, color: str = ACCENT) -> None:
        page.snack_bar = ft.SnackBar(ft.Text(message), bgcolor=color)
        page.snack_bar.open = True
        page.update()

    def section_header(title: str, icon: Any | None = None) -> ft.Control:
        items: list[ft.Control] = []
        if icon:
            items.append(ft.Icon(icon, size=16, color=ACCENT))
        items.append(ft.Text(title, size=14, weight=ft.FontWeight.W_600, color=TEXT_MAIN))
        return ft.Row(items, spacing=8, alignment=ft.MainAxisAlignment.START)

    def section(
        title: str,
        controls: list[ft.Control],
        description: str | None = None,
        icon: Any | None = None,
    ) -> ft.Container:
        header = section_header(title, icon)

        content_col = ft.Column([header], spacing=4)

        if description:
            content_col.controls.append(
                ft.Text(description, size=12, color=TEXT_MUTED)
            )
            content_col.controls.append(ft.Container(height=4))  # Spacer
        else:
            content_col.controls.append(ft.Container(height=2))

        content_col.controls.extend(controls)

        return ft.Container(
            content=content_col,
            bgcolor=BG_CARD,
            border=ft.border.all(1, BORDER),
            border_radius=10,
            padding=14,
        )

    # --- API Keys Tab ---
    groq_key = ft.TextField(
        label="GROQ API Key",
        password=True,
        can_reveal_password=True,
        width=600,
        text_size=13,
        border_color=BORDER,
        bgcolor=BG_DARK,
    )
    openrouter_key = ft.TextField(
        label="OpenRouter API Key",
        password=True,
        can_reveal_password=True,
        width=600,
        text_size=13,
        border_color=BORDER,
        bgcolor=BG_DARK,
    )

    api_tab = ft.Column(
        [
            section(
                "API Configuration",
                [
                    groq_key,
                    openrouter_key,
                    ft.Container(height=4),
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.INFO_OUTLINE, size=14, color=TEXT_MUTED),
                            ft.Text("Gemini runs via OpenRouter — no separate Gemini key needed.", color=TEXT_MUTED, size=12),
                        ],
                        spacing=6
                    ),
                ],
                icon=ft.Icons.VPN_KEY_OUTLINED,
                description="Manage your API keys for transcription and LLM services.",
            ),
        ],
        spacing=12,
        scroll=ft.ScrollMode.AUTO,
    )

    # --- General Tab ---
    KEY_LABELS = {
        "alt_r": "Option (Right)",
        "alt_l": "Option (Left)",
        "cmd_r": "Command (Right)",
        "cmd_l": "Command (Left)",
    }

    def on_change_trigger_key(e: ft.ControlEvent):
        try:
            ConfigManager().set_value("TRIGGER_KEY", e.control.value)
            ConfigManager().save_config()
        except Exception:
            pass

    trigger_key = ft.Dropdown(
        label="Trigger Key",
        options=[ft.dropdown.Option(key=k, text=v) for k, v in KEY_LABELS.items()],
        width=300,
        on_change=on_change_trigger_key,
        text_size=13,
        border_color=BORDER,
        bgcolor=BG_DARK,
    )

    provider_parakeet = ft.Checkbox(label="parakeet_mlx", active_color=ACCENT)
    provider_groq = ft.Checkbox(label="groq_whisper", active_color=ACCENT)
    provider_gemini = ft.Checkbox(label="gemini (via OpenRouter)", active_color=ACCENT)

    auto_copy_result = ft.Checkbox(
        label="Auto-copy final result to clipboard",
        active_color=ACCENT,
        tooltip="Helpful if target app blocks synthetic typing",
    )

    general_tab = ft.Column(
        [
            section(
                "Trigger",
                [trigger_key],
                description="Hold this key to record; release to transcribe. Double-tap to toggle recording.",
                icon=ft.Icons.KEYBOARD_OPTION_KEY,
            ),
            section(
                "Enabled Providers",
                [ft.Column([provider_parakeet, provider_groq, provider_gemini], spacing=4)],
                description="Select which transcription engines to run in parallel. Results are merged for accuracy.",
                icon=ft.Icons.TUNGSTEN_OUTLINED,
            ),
            section(
                "Workflow",
                [auto_copy_result],
                description="Configure how results are delivered to your system.",
                icon=ft.Icons.SETTINGS_SUGGEST_OUTLINED,
            ),
        ],
        spacing=12,
        scroll=ft.ScrollMode.AUTO,
    )

    # --- Audio Tab ---
    mic_status_icon = ft.Icon(ft.Icons.CHECK_CIRCLE_OUTLINE, size=14, color=SUCCESS)
    mic_status_text = ft.Text("", size=12, color=TEXT_MUTED)
    mic_status_container = ft.Container(
        content=ft.Row([mic_status_icon, mic_status_text], spacing=6),
        bgcolor="#0F3928",
        border=ft.border.all(1, "#10B98133"),
        border_radius=8,
        padding=10,
    )

    devices_cache: list[MicDevice] = []

    def update_mic_status(selected_name: str) -> None:
        """Update status display based on currently selected device."""
        if selected_name == "__default__" or not selected_name:
            mic_status_text.value = "Using system default microphone"
            mic_status_text.color = SUCCESS
            mic_status_icon.name = ft.Icons.CHECK_CIRCLE_OUTLINE
            mic_status_icon.color = SUCCESS
            mic_status_container.bgcolor = "#0F3928"
            mic_status_container.border = ft.border.all(1, "#10B98133")
        else:
            found = find_device_by_name(selected_name, devices_cache) if devices_cache else None
            if found:
                mic_status_text.value = f"✓ '{found.name}' ready"
                mic_status_text.color = SUCCESS
                mic_status_icon.name = ft.Icons.CHECK_CIRCLE_OUTLINE
                mic_status_icon.color = SUCCESS
                mic_status_container.bgcolor = "#0F3928"
                mic_status_container.border = ft.border.all(1, "#10B98133")
            else:
                mic_status_text.value = f"✗ '{selected_name}' not found"
                mic_status_text.color = ERROR
                mic_status_icon.name = ft.Icons.ERROR_OUTLINE
                mic_status_icon.color = ERROR
                mic_status_container.bgcolor = "#3F1515"
                mic_status_container.border = ft.border.all(1, "#EF444433")
        mic_status_text.update()
        mic_status_icon.update()
        mic_status_container.update()

    def on_mic_change(e):
        update_mic_status(e.control.value)

    mic_devices_dropdown = ft.Dropdown(
        label="Microphone Device",
        width=500,
        options=[],
        text_size=13,
        border_color=BORDER,
        bgcolor=BG_DARK,
        on_change=on_mic_change,
    )
    mic_refresh_btn = ft.IconButton(
        icon=ft.Icons.REFRESH,
        tooltip="Refresh devices",
        icon_color=ACCENT,
    )

    audio_tab = ft.Column(
        [
            section(
                "Input Device",
                [
                    ft.Row([mic_devices_dropdown, mic_refresh_btn], alignment=ft.MainAxisAlignment.START),
                    mic_status_container,
                ],
                description="Select your preferred microphone. We'll track it by name even if you unplug/replug it.",
                icon=ft.Icons.MIC_NONE_OUTLINED,
            ),
        ],
        spacing=12,
        scroll=ft.ScrollMode.AUTO,
    )

    # --- Models Tab ---
    whisper_model = ft.Dropdown(
        label="Whisper Model",
        options=[ft.dropdown.Option("whisper-large-v3"), ft.dropdown.Option("whisper-large-v3-turbo")],
        width=300,
        text_size=13,
        border_color=BORDER,
        bgcolor=BG_DARK,
    )
    whisper_language = ft.TextField(
        label="Language Code",
        hint_text="e.g. 'en' (empty = auto)",
        width=300,
        text_size=13,
        border_color=BORDER,
        bgcolor=BG_DARK,
    )
    openrouter_model = ft.TextField(
        label="OpenRouter Model (Correction)",
        width=400,
        text_size=13,
        border_color=BORDER,
        bgcolor=BG_DARK,
    )
    text_editing_model = ft.TextField(
        label="Text Editing Model",
        width=400,
        text_size=13,
        border_color=BORDER,
        bgcolor=BG_DARK,
    )

    models_tab = ft.Column(
        [
            section(
                "Speech Recognition",
                [
                    whisper_model,
                    whisper_language
                ],
                description="Cloud Whisper settings via Groq.",
                icon=ft.Icons.SPEED_OUTLINED,
            ),
            section(
                "Language Models",
                [
                    openrouter_model,
                    text_editing_model
                ],
                description="Models used for correction and text transformations.",
                icon=ft.Icons.SMART_TOY_OUTLINED,
            ),
        ],
        spacing=12,
        scroll=ft.ScrollMode.AUTO,
    )

    # --- Context Tab ---
    enable_context = ft.Checkbox(label="Enable context from previous transcriptions", active_color=ACCENT)
    context_history_count = ft.TextField(
        label="History Count",
        width=200,
        input_filter=ft.NumbersOnlyInputFilter(),
        text_size=13,
        border_color=BORDER,
        bgcolor=BG_DARK,
    )
    context_max_age_seconds = ft.TextField(
        label="Max Age",
        width=220,
        input_filter=ft.NumbersOnlyInputFilter(),
        suffix_text="sec",
        text_size=13,
        border_color=BORDER,
        bgcolor=BG_DARK,
    )
    enable_app_context = ft.Checkbox(label="Enable application context detection", active_color=ACCENT)
    app_context_timeout = ft.TextField(
        label="Detection Timeout",
        width=220,
        input_filter=ft.NumbersOnlyInputFilter(),
        suffix_text="sec",
        text_size=13,
        border_color=BORDER,
        bgcolor=BG_DARK,
    )

    context_tab = ft.Column(
        [
            section(
                "Conversation Memory",
                [
                    enable_context,
                    ft.Row([context_history_count, context_max_age_seconds]),
                ],
                description="Uses recent transcriptions to improve accuracy and maintain continuity.",
                icon=ft.Icons.HISTORY_TOGGLE_OFF,
            ),
            section(
                "App Awareness",
                [
                    enable_app_context,
                    app_context_timeout
                ],
                description="Uses the active application window to bias vocabulary (e.g. coding terms in VS Code).",
                icon=ft.Icons.APP_SHORTCUT_OUTLINED,
            ),
        ],
        spacing=12,
        scroll=ft.ScrollMode.AUTO,
    )

    # --- Prompts Tab ---
    system_context = ft.TextField(
        label="System Prompt (Transcription)",
        multiline=True,
        min_lines=6,
        width=800,
        text_size=13,
        border_color=BORDER,
        bgcolor=BG_DARK,
    )
    text_editing_context = ft.TextField(
        label="System Prompt (Text Editing)",
        multiline=True,
        min_lines=4,
        width=800,
        text_size=13,
        border_color=BORDER,
        bgcolor=BG_DARK,
    )

    prompts_tab = ft.Column(
        [
            section(
                "Transcription Guidelines",
                [system_context],
                description="Instructions for the AI on how to format, correct, and style transcriptions.",
                icon=ft.Icons.TIPS_AND_UPDATES_OUTLINED,
            ),
            section(
                "Editing Guidelines",
                [text_editing_context],
                description="Instructions for voice-commanded text editing.",
                icon=ft.Icons.EDIT_NOTE_OUTLINED,
            ),
        ],
        spacing=12,
        scroll=ft.ScrollMode.AUTO,
    )

    # --- Main Layout ---
    tabs = ft.Tabs(
        tabs=[
            ft.Tab(text="General", content=general_tab, icon=ft.Icons.TUNE),
            ft.Tab(text="Audio", content=audio_tab, icon=ft.Icons.MIC),
            ft.Tab(text="API Keys", content=api_tab, icon=ft.Icons.KEY),
            ft.Tab(text="Models", content=models_tab, icon=ft.Icons.PSYCHOLOGY),
            ft.Tab(text="Context", content=context_tab, icon=ft.Icons.MEMORY),
            ft.Tab(text="Prompts", content=prompts_tab, icon=ft.Icons.DESCRIPTION),
        ],
        expand=True,
        indicator_color=ACCENT,
        label_color=TEXT_MAIN,
        unselected_label_color=TEXT_MUTED,
        divider_color=BORDER,
        animation_duration=300,
        scrollable=True,
    )

    def load_all() -> None:
        groq_key.value = config.get_value("GROQ_API_KEY") or ""
        openrouter_key.value = config.get_value("OPENROUTER_API_KEY") or ""

        stored_key = config.get_value("TRIGGER_KEY") or "alt_r"
        if stored_key not in KEY_LABELS:
            stored_key = "alt_r"
            config.set_value("TRIGGER_KEY", stored_key)
            config.save_config()
        trigger_key.value = stored_key
        enabled = set(config.get_value("ENABLED_PROVIDERS") or [])
        provider_parakeet.value = "parakeet_mlx" in enabled
        provider_groq.value = "groq_whisper" in enabled
        provider_gemini.value = "gemini" in enabled
        auto_copy_result.value = bool(config.get_value("AUTO_COPY_RESULT_TO_CLIPBOARD"))

        # Load microphone devices using smart mic manager
        refresh_devices()

        whisper_model.value = config.get_value("WHISPER_MODEL") or "whisper-large-v3"
        whisper_language.value = config.get_value("WHISPER_LANGUAGE") or ""
        openrouter_model.value = config.get_value("OPENROUTER_MODEL") or ""
        text_editing_model.value = config.get_value("TEXT_EDITING_MODEL") or ""

        enable_context.value = bool(config.get_value("ENABLE_CONTEXT") or False)
        context_history_count.value = str(config.get_value("CONTEXT_HISTORY_COUNT") or 3)
        context_max_age_seconds.value = str(config.get_value("CONTEXT_MAX_AGE_SECONDS") or 300)
        enable_app_context.value = bool(config.get_value("ENABLE_APPLICATION_CONTEXT") or False)
        app_context_timeout.value = str(config.get_value("APPLICATION_CONTEXT_TIMEOUT") or 2)

        system_context.value = config.get_value("SYSTEM_CONTEXT") or ""
        text_editing_context.value = config.get_value("TEXT_EDITING_CONTEXT") or ""

        page.update()

    def save_all(_=None) -> None:
        try:
            config.set_value("GROQ_API_KEY", groq_key.value)
            config.set_value("OPENROUTER_API_KEY", openrouter_key.value)

            config.set_value("TRIGGER_KEY", trigger_key.value)
            enabled = []
            if provider_parakeet.value:
                enabled.append("parakeet_mlx")
            if provider_groq.value:
                enabled.append("groq_whisper")
            if provider_gemini.value:
                enabled.append("gemini")
            config.set_value("ENABLED_PROVIDERS", enabled)
            config.set_value("AUTO_COPY_RESULT_TO_CLIPBOARD", bool(auto_copy_result.value))

            # Save microphone using smart mic manager
            selected_name = mic_devices_dropdown.value or "__default__"
            if selected_name == "__default__":
                save_preferred_device(None)  # Clear preference - use system default
            else:
                # Find the device by name and save it
                devices = list_input_devices()
                found_device = find_device_by_name(selected_name, devices)
                save_preferred_device(found_device)

            config.set_value("WHISPER_MODEL", whisper_model.value)
            config.set_value("WHISPER_LANGUAGE", whisper_language.value)
            config.set_value("OPENROUTER_MODEL", openrouter_model.value)
            config.set_value("TEXT_EDITING_MODEL", text_editing_model.value)

            config.set_value("ENABLE_CONTEXT", bool(enable_context.value))
            config.set_value("CONTEXT_HISTORY_COUNT", int(context_history_count.value or 3))
            config.set_value("CONTEXT_MAX_AGE_SECONDS", int(context_max_age_seconds.value or 300))
            config.set_value("ENABLE_APPLICATION_CONTEXT", bool(enable_app_context.value))
            config.set_value("APPLICATION_CONTEXT_TIMEOUT", int(app_context_timeout.value or 2))

            config.set_value("SYSTEM_CONTEXT", system_context.value)
            config.set_value("TEXT_EDITING_CONTEXT", text_editing_context.value)

            config.save_config()
            show_snack("Settings saved successfully", SUCCESS)
        except Exception as ex:
            show_snack(f"Error saving settings: {ex}", ERROR)

    def reset_defaults(_=None) -> None:
        try:
            config.reset_to_defaults()
            load_all()
            show_snack("Reset to defaults", ACCENT)
        except Exception as ex:
            show_snack(f"Error resetting: {ex}", ERROR)

    def refresh_devices(_=None):
        try:
            nonlocal devices_cache
            devices = list_input_devices()
            devices_cache = devices
            # Add "System Default" option at the top
            mic_devices_dropdown.options = [
                ft.dropdown.Option(key="__default__", text="System Default")
            ] + [
                ft.dropdown.Option(key=dev.name, text=get_device_display_label(dev))
                for dev in devices
            ]

            # Find and select the configured device
            configured_name = config.get_value("MIC_DEVICE_NAME") or ""
            if configured_name:
                found_device = find_device_by_name(configured_name, devices)
                if found_device:
                    mic_devices_dropdown.value = found_device.name
                else:
                    mic_devices_dropdown.value = "__default__"
            else:
                mic_devices_dropdown.value = "__default__"

            # Update status based on current selection
            update_mic_status(mic_devices_dropdown.value)

            # Only show snackbar if triggered by button click (not initial load)
            if _ is not None:
                show_snack(f"Found {len(devices)} input device(s)", ACCENT)

            mic_devices_dropdown.update()
        except Exception as ex:
            if _ is not None:
                show_snack(f"Error listing devices: {ex}", ERROR)

    mic_refresh_btn.on_click = refresh_devices

    save_btn = ft.ElevatedButton(
        text="Save Changes",
        icon=ft.Icons.SAVE,
        style=ft.ButtonStyle(
            color=ft.Colors.WHITE,
            bgcolor=ACCENT,
            shape=ft.RoundedRectangleBorder(radius=8),
        ),
        on_click=save_all,
        height=40,
    )

    reset_btn = ft.TextButton(
        text="Reset Defaults",
        on_click=reset_defaults,
        style=ft.ButtonStyle(color=TEXT_MUTED)
    )

    close_btn = ft.TextButton(
        text="Close",
        on_click=lambda e: page.window.close(),
        style=ft.ButtonStyle(color=TEXT_MAIN)
    )

    header = ft.Row(
        [
            ft.Icon(ft.Icons.SETTINGS, size=20, color=ACCENT),
            ft.Text("MergeScribe Settings", size=16, weight=ft.FontWeight.W_600, color="#E5E7EB"),
        ],
        spacing=8,
    )

    page.add(
        header,
        ft.Container(content=tabs, expand=True),
        ft.Divider(color=BORDER, height=1),
        ft.Container(
            content=ft.Row(
                [reset_btn, ft.Container(expand=True), close_btn, save_btn],
                alignment=ft.MainAxisAlignment.END,
            ),
            padding=ft.padding.only(top=8),
        ),
    )

    load_all()


def main() -> None:
    ft.app(target=settings_app)


if __name__ == "__main__":
    main()
