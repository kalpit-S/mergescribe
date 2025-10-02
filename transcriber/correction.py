import subprocess
import time
from typing import Iterable, List, Optional, Tuple

import requests

from config_manager import ConfigManager

from .state import is_debug

Transcription = Tuple[str, str]


def build_correction_prompt(
    transcriptions: Iterable[Transcription],
    context: Optional[str] = None,
    app_context: Optional[str] = None,
) -> str:
    """Compose a prompt for LLM-based transcription correction."""
    transcriptions = list(transcriptions)
    if not transcriptions:
        return "No transcription available"

    if len(transcriptions) == 1:
        provider_name, text = transcriptions[0]
        prompt = f"Transcription to refine: {text}"
    else:
        prompt_lines: List[str] = ["Multiple transcriptions of the same recording:\n"]
        for idx, (provider_name, text) in enumerate(transcriptions, start=1):
            prompt_lines.append(f"{idx}. {provider_name}: {text}")
        prompt_lines.append("")
        prompt_lines.append(
            "Provide the most accurate transcription by comparing and choosing the best parts from each."
        )
        prompt = "\n".join(prompt_lines)

    context_parts = []
    if app_context and app_context.strip():
        context_parts.append(f"Current application: {app_context}")
    if context and context.strip():
        context_parts.append(f"Recent transcriptions: {context}")

    if context_parts:
        context_section = "\n".join(context_parts)
        prompt = f"{context_section}\n\n---\n\n{prompt}"

    return prompt


def _call_openrouter_api(messages, model_key, temperature=0.5, max_tokens=1000, timeout=10):
    cfg = ConfigManager()

    headers = {
        "Authorization": f"Bearer {cfg.get_value('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }

    site_url = cfg.get_value("OPENROUTER_SITE_URL") or ""
    site_name = cfg.get_value("OPENROUTER_SITE_NAME") or ""
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name

    data = {
        "model": cfg.get_value(model_key) or "google/gemini-2.5-flash",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(3):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 - return/fallback on final attempt
            print(f"OpenRouter attempt {attempt + 1} failed: {exc}")
            if attempt == 2:
                raise
            time.sleep(1)


def correct_with_openrouter(
    transcriptions: Iterable[Transcription],
    context: Optional[str],
    app_context: Optional[str] = None,
) -> str:
    cfg = ConfigManager()
    prompt = build_correction_prompt(transcriptions, context, app_context)

    messages = [
        {"role": "system", "content": cfg.get_value("SYSTEM_CONTEXT") or ""},
        {"role": "user", "content": prompt},
    ]

    try:
        return _call_openrouter_api(messages, "OPENROUTER_MODEL", timeout=10, max_tokens=1000)
    except Exception:
        transcriptions = list(transcriptions)
        return transcriptions[0][1] if transcriptions else ""


def edit_text_with_openrouter(selected_text: str, voice_command: str) -> str:
    cfg = ConfigManager()
    prompt = (
        f"TASK: {voice_command}\n\n"
        f"ORIGINAL TEXT:\n{selected_text}\n\n"
        "INSTRUCTIONS: Apply the task to the original text above. Return ONLY the edited text, "
        "nothing else. No explanations, no formatting, no extra content."
    )

    messages = [
        {"role": "system", "content": cfg.get_value("TEXT_EDITING_CONTEXT") or ""},
        {"role": "user", "content": prompt},
    ]

    try:
        return _call_openrouter_api(messages, "TEXT_EDITING_MODEL", timeout=15, max_tokens=4000)
    except Exception:
        print("❌ Text editing failed - returning original text")
        return selected_text


def get_application_context() -> str:
    cfg = ConfigManager()
    if not bool(cfg.get_value("ENABLE_APPLICATION_CONTEXT")):
        return ""

    try:
        script = """
        tell application "System Events"
            set frontApp to first application process whose frontmost is true
            set appName to name of frontApp
            try
                set windowTitle to name of front window of frontApp
                return appName & " | " & windowTitle
            on error
                return appName
            end try
        end tell
        """

        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=(cfg.get_value("APPLICATION_CONTEXT_TIMEOUT") or 2),
        )

        if result.returncode == 0 and result.stdout.strip():
            app_context = result.stdout.strip()
            app_context = app_context.replace(" • ", " | ")
            app_context = app_context.replace(" — ", " | ")
            return app_context
        return ""
    except Exception as exc:  # noqa: BLE001 - best effort logging, return empty context
        if is_debug():
            print(f"[DEBUG] Could not get application context: {exc}")
        return ""
