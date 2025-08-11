import base64
import requests
import json

from config_manager import ConfigManager


def _build_model_id(raw_model_id: str | None) -> str:
    """Normalize model id to OpenRouter format, defaulting to Gemini flash."""
    model_id = raw_model_id or "google/gemini-2.5-flash"
    if "/" not in model_id:
        model_id = f"google/{model_id}"
    return model_id


def transcribe_sync(audio_bytes: bytes):
    """Transcribe audio using Gemini via OpenRouter (synchronous)."""
    try:
        cfg = ConfigManager()
        api_key = cfg.get_value("OPENROUTER_API_KEY")
        if not api_key:
            print("Gemini via OpenRouter error: Missing or invalid OPENROUTER_API_KEY")
            return ("Gemini", "")

        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        site_url = cfg.get_value("OPENROUTER_SITE_URL") or ""
        site_name = cfg.get_value("OPENROUTER_SITE_NAME") or ""
        if site_url:
            headers["HTTP-Referer"] = site_url
        if site_name:
            headers["X-Title"] = site_name

        prompt_text = cfg.get_value("GEMINI_PROMPT") or "Transcribe this speech."
        model_id = _build_model_id(cfg.get_value("GEMINI_MODEL"))

        data = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": base64_audio, "format": "wav"},
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 4000
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=20
        )
        response.raise_for_status()
        result = response.json()

        text = (result["choices"][0]["message"]["content"] or "")
        return ("Gemini", text)
    except Exception as exc:
        print(f"Gemini via OpenRouter error: {exc}")
        return ("Gemini", "")
