"""Settings UI (Flet)."""
from typing import Any

import flet as ft

from config_manager import ConfigManager
from mic_manager import (
    list_input_devices,
    find_device_by_name,
    get_device_display_label,
    save_preferred_device,
    check_preferred_device_status,
    MicDevice,
)


def settings_app(page: ft.Page) -> None:
    page.title = "MergeScribe Settings"
    page.window_width = 900
    page.window_height = 700
    page.padding = 12
    page.bgcolor = "#0F1115"
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = ft.Theme(color_scheme_seed="#7AA2F7")

    config = ConfigManager()


    ACCENT = "#7AA2F7"
    CARD_BG = "#14171C"
    STROKE = "#22262E"
    MUTED = "#9BA3AF"

    GRADIENT_HEADER = ft.LinearGradient(
        begin=ft.alignment.top_left,
        end=ft.alignment.bottom_right,
        colors=["#0EA5EA", "#7AA2F7", "#A855F7"],
        stops=[0.0, 0.6, 1.0],
    )

    def section(title: str, controls: list[ft.Control], description: str | None = None, icon: Any | None = None) -> ft.Container:
        header_items: list[ft.Control] = []
        if icon:
            header_items.append(ft.Icon(icon, size=18, color=ACCENT))
        header_items.append(ft.Text(title, size=15, weight=ft.FontWeight.W_600))
        header = ft.Row(header_items, spacing=8)
        body_children: list[ft.Control] = [header]
        if description:
            body_children.append(ft.Text(description, size=12, color=MUTED))
        body_children.extend(controls)
        return ft.Container(
            content=ft.Column(body_children, spacing=10),
            bgcolor=CARD_BG,
            border=ft.border.all(1, STROKE),
            border_radius=12,
            padding=16,
        )
    groq_key = ft.TextField(label="GROQ API Key", password=True, can_reveal_password=True, width=600)
    openrouter_key = ft.TextField(label="OpenRouter API Key", password=True, can_reveal_password=True, width=600)

    api_tab = ft.Column(
        [
            section(
                "API Keys",
                [
                    groq_key,
                    openrouter_key,
                    ft.Text("Gemini runs via OpenRouter — no separate Gemini key needed.", color=MUTED, size=12),
                ],
                icon=ft.Icons.VPN_KEY_OUTLINED,
            ),
        ],
        spacing=16,
        scroll=ft.ScrollMode.AUTO,
    )

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
    )
    provider_parakeet = ft.Checkbox(label="parakeet_mlx")
    provider_groq = ft.Checkbox(label="groq_whisper")
    provider_gemini = ft.Checkbox(label="gemini (via OpenRouter)")
    auto_copy_result = ft.Checkbox(label="Auto-copy final result to clipboard")

    mic_devices_dropdown = ft.Dropdown(label="Microphone", width=500, options=[])
    mic_status_text = ft.Text("", size=12, color=MUTED)
    mic_refresh_btn = ft.TextButton(text="Refresh devices")

    general_tab = ft.Column(
        [
            section(
                "Trigger",
                [trigger_key],
                description="Hold this key to record; release to transcribe.",
                icon=ft.Icons.KEYBOARD_OPTION_KEY,
            ),
            section(
                "Enabled Providers",
                [ft.Row([provider_parakeet, provider_groq, provider_gemini])],
                description="Select which engines to run in parallel.",
                icon=ft.Icons.TUNGSTEN_OUTLINED,
            ),
            section(
                "Text Insertion",
                [auto_copy_result],
                description="If an app blocks synthetic typing (Warp/Cursor can sometimes do this after updates), you'll still be able to paste instantly.",
                icon=ft.Icons.CONTENT_PASTE_OUTLINED,
            ),
            section(
                "Audio Input",
                [
                    ft.Row([mic_devices_dropdown, mic_refresh_btn]),
                    mic_status_text,
                ],
                description="Choose your recording device. Your mic will be found by name even if the device index changes.",
                icon=ft.Icons.MIC_NONE_OUTLINED,
            ),
        ],
        spacing=16,
        scroll=ft.ScrollMode.AUTO,
    )

    whisper_model = ft.Dropdown(
        label="Whisper Model",
        options=[ft.dropdown.Option("whisper-large-v3"), ft.dropdown.Option("whisper-large-v3-turbo")],
        width=300,
    )
    whisper_language = ft.TextField(label="Language (e.g., 'en' or empty for auto)", width=300)
    openrouter_model = ft.TextField(label="OpenRouter Model (correction)", width=400)
    text_editing_model = ft.TextField(label="Text Editing Model", width=400)

    models_tab = ft.Column(
        [
            section(
                "ASR (Whisper)",
                [ft.Row([whisper_model, whisper_language])],
                description="Cloud Whisper via Groq; set model and optional language.",
                icon=ft.Icons.SPEED_OUTLINED,
            ),
            section(
                "LLMs",
                [openrouter_model, text_editing_model],
                description="Models used for correction and text editing (via OpenRouter).",
                icon=ft.Icons.TEXT_FIELDS_OUTLINED,
            ),
        ],
        spacing=16,
        scroll=ft.ScrollMode.AUTO,
    )

    enable_context = ft.Checkbox(label="Enable context from previous transcriptions")
    context_history_count = ft.TextField(label="History Count", width=200, input_filter=ft.NumbersOnlyInputFilter())
    context_max_age_seconds = ft.TextField(label="Max Age (seconds)", width=220, input_filter=ft.NumbersOnlyInputFilter())
    enable_app_context = ft.Checkbox(label="Enable application context detection")
    app_context_timeout = ft.TextField(label="Detection Timeout (seconds)", width=220, input_filter=ft.NumbersOnlyInputFilter())

    context_tab = ft.Column(
        [
            section(
                "Transcription Context",
                [enable_context, ft.Row([context_history_count, context_max_age_seconds])],
                description="Include previous results for continuity.",
                icon=ft.Icons.HISTORY_TOGGLE_OFF,
            ),
            section(
                "Application Context",
                [enable_app_context, app_context_timeout],
                description="Use current app/window to bias vocabulary.",
                icon=ft.Icons.APP_SHORTCUT_OUTLINED,
            ),
        ],
        spacing=16,
        scroll=ft.ScrollMode.AUTO,
    )

    system_context = ft.TextField(label="System Context (transcription)", multiline=True, min_lines=6, width=800)
    text_editing_context = ft.TextField(label="Text Editing Context", multiline=True, min_lines=4, width=800)

    prompts_tab = ft.Column(
        [
            section(
                "System Context",
                [system_context],
                description="Guides how transcription is corrected.",
                icon=ft.Icons.TIPS_AND_UPDATES_OUTLINED,
            ),
            section(
                "Text Editing Context",
                [text_editing_context],
                description="Controls how voice commands transform selected text.",
                icon=ft.Icons.EDIT_NOTE_OUTLINED,
            ),
        ],
        spacing=16,
        scroll=ft.ScrollMode.AUTO,
    )

    tabs = ft.Tabs(
        tabs=[
            ft.Tab(text="API Keys", content=api_tab),
            ft.Tab(text="General", content=general_tab),
            ft.Tab(text="Models", content=models_tab),
            ft.Tab(text="Context", content=context_tab),
            ft.Tab(text="Prompts", content=prompts_tab),
        ],
        expand=1,
        indicator_color=ACCENT,
        label_color="#E5E7EB",
        unselected_label_color="#9BA3AF",
        divider_color=STROKE,
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
        try:
            devices = list_input_devices()
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
                # Try to find device by name
                found_device = find_device_by_name(configured_name, devices)
                if found_device:
                    mic_devices_dropdown.value = found_device.name
                else:
                    # Device not currently available - still show the saved name
                    mic_devices_dropdown.value = "__default__"
            else:
                mic_devices_dropdown.value = "__default__"
            
            # Update status
            status_msg, is_available = check_preferred_device_status()
            mic_status_text.value = status_msg
            mic_status_text.color = "#4ADE80" if is_available else "#F87171"
        except Exception as _e:
            mic_devices_dropdown.options = [ft.dropdown.Option(key="__default__", text="System Default")]
            mic_status_text.value = f"Error listing devices: {_e}"
            mic_status_text.color = "#F87171"

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
            page.snack_bar = ft.SnackBar(ft.Text("Settings saved."))
            page.snack_bar.open = True
            page.update()
        except Exception as ex:
            page.snack_bar = ft.SnackBar(ft.Text(f"Error saving settings: {ex}"))
            page.snack_bar.open = True
            page.update()

    def reset_defaults(_=None) -> None:
        try:
            config.reset_to_defaults()
            load_all()
            page.snack_bar = ft.SnackBar(ft.Text("Reset to defaults."))
            page.snack_bar.open = True
            page.update()
        except Exception as ex:
            page.snack_bar = ft.SnackBar(ft.Text(f"Error resetting: {ex}"))
            page.snack_bar.open = True
            page.update()

    def refresh_devices(_=None):
        try:
            devices = list_input_devices()
            # Add "System Default" option at the top
            mic_devices_dropdown.options = [
                ft.dropdown.Option(key="__default__", text="System Default")
            ] + [
                ft.dropdown.Option(key=dev.name, text=get_device_display_label(dev))
                for dev in devices
            ]
            
            # Update status
            status_msg, is_available = check_preferred_device_status()
            mic_status_text.value = status_msg
            mic_status_text.color = "#4ADE80" if is_available else "#F87171"
            
            page.snack_bar = ft.SnackBar(ft.Text(f"Found {len(devices)} input device(s)"))
            page.snack_bar.open = True
            page.update()
        except Exception as ex:
            page.snack_bar = ft.SnackBar(ft.Text(f"Error listing devices: {ex}"))
            page.snack_bar.open = True
            page.update()

    mic_refresh_btn.on_click = refresh_devices

    save_btn = ft.FilledButton(text="Save", on_click=save_all)
    reset_btn = ft.OutlinedButton(text="Reset to Defaults", on_click=reset_defaults)
    close_btn = ft.TextButton(text="Close", on_click=lambda e: page.window.close())

    title = ft.Text(
        "MergeScribe – Settings",
        size=16,
        weight=ft.FontWeight.W_600,
        color="#E5E7EB",
    )
    header_row = ft.Row([title], alignment=ft.MainAxisAlignment.START)
    header_divider = ft.Container(height=2, gradient=GRADIENT_HEADER, border_radius=1)
    header = ft.Column([header_row, header_divider], spacing=6)

    page.add(
        header,
        ft.Container(height=4),
        tabs,
        ft.Container(height=8),
        ft.Row(
            [
                reset_btn,
                ft.Container(expand=True),
                close_btn,
                save_btn,
            ],
            alignment=ft.MainAxisAlignment.END,
        ),
    )

    load_all()


def main() -> None:
    ft.app(target=settings_app)


if __name__ == "__main__":
    main()


