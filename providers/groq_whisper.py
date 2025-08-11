from io import BytesIO

from groq import Groq

from config_manager import ConfigManager

_client = None

def _get_client() -> Groq | None:
    global _client
    api_key = ConfigManager().get_value("GROQ_API_KEY")
    if _client is None and api_key:
        try:
            _client = Groq(api_key=api_key)
        except Exception as e:
            print(f"Groq client init error: {e}")
            _client = None
    return _client

def transcribe_sync(audio_bytes, language=None, prompt=None, temperature=0.0):
    """Transcribe audio using Groq Whisper (synchronous)."""
    try:
        client = _get_client()
        if client is None:
            print("Groq Whisper error: Missing or invalid GROQ_API_KEY")
            return ("Groq Whisper", "")
        audio_file = BytesIO(audio_bytes)
        audio_file.name = "audio.wav"
        
        params = {
            "file": audio_file,
            "model": ConfigManager().get_value("WHISPER_MODEL") or "whisper-large-v3",
            "temperature": temperature
        }
        
        if language is None:
            language = ConfigManager().get_value("WHISPER_LANGUAGE")
        if language:
            params["language"] = language
        if prompt:
            params["prompt"] = prompt
        
        response = client.audio.transcriptions.create(**params)
        
        return ("Groq Whisper", response.text)
    except Exception as e:
        print(f"Groq Whisper error: {e}")
        return ("Groq Whisper", "")
