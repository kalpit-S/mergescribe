"""
OpenRouter STT provider for cloud transcription.

Routes to the appropriate API based on model type:
- Dedicated STT models (gpt-4o-transcribe, voxtral, etc): /audio/transcriptions
- Multimodal LLMs (gemini-3.1-flash-lite, etc): /chat/completions with audio content block
"""

import base64
import io
import time


import numpy as np
import requests
import soundfile as sf

from . import Provider
from ..types import TranscriptionResult

_BASE_URL = "https://openrouter.ai/api/v1"
_REQUEST_TIMEOUT_SECONDS = 30
_MAX_ATTEMPTS = 2
_RETRYABLE_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.SSLError,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ReadTimeout,
)

# Models that use the dedicated /audio/transcriptions endpoint (OpenAI-compatible STT)
_STT_ENDPOINT_MODELS = {
    "openai/gpt-4o-transcribe",
    "openai/gpt-4o-mini-transcribe",
    "openai/whisper-1",
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo",
    "google/chirp-3",
    "microsoft/mai-transcribe-1.5",
    "microsoft/mai-transcribe-2",
    "mistralai/voxtral-mini-transcribe",
    "qwen/qwen3-asr-flash-2026-02-10",
    "openai/gpt-transcribe",
    "fish-audio/transcribe-1",
    "x-ai/grok-stt-1.0",
}

# Multimodal models that support audio via chat completions.
# Value is the reasoning effort to request ("none" to fully disable; some models
# reject "none" outright and require a minimal non-zero effort instead).
_MULTIMODAL_REASONING_EFFORT = {
    "google/gemini-3.1-flash-lite": "none",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": "none",
    "google/gemini-3.5-flash": "minimal",  # rejects reasoning:none ("mandatory reasoning")
    "google/gemini-3.7-flash": "minimal",  # same: 400s on none, and 8x slower if omitted
}


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    audio_int16 = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, audio_int16, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.getvalue()


def _model_slug(model: str) -> str:
    return model.replace("/", "-").replace(".", "-").replace(":", "-")


class OpenRouterSTTProvider(Provider):
    """
    Cloud STT via OpenRouter — one instance per model.

    Auto-detects whether to use the dedicated /audio/transcriptions endpoint
    or a multimodal chat completion based on the model name.
    """

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.name = f"or-{_model_slug(model)}"
        self._use_stt_endpoint = model in _STT_ENDPOINT_MODELS
        self._reasoning_effort = _MULTIMODAL_REASONING_EFFORT.get(model)
        self._initialized = False

    def initialize(self) -> None:
        if not self.api_key:
            print(f"[{self.name}] No OpenRouter API key")
            return
        self._initialized = True
        mode = "stt-endpoint" if self._use_stt_endpoint else "multimodal-chat"
        print(f"[{self.name}] Initialized ({mode}, reasoning_effort={self._reasoning_effort})")

    def transcribe(self, audio: np.ndarray, mic_name: str = "") -> TranscriptionResult:
        start = time.time()
        text = ""

        if not self._initialized:
            return TranscriptionResult(text="", provider=self.name, mic=mic_name, latency_ms=0)

        try:
            wav = _audio_to_wav_bytes(audio)
            if self._use_stt_endpoint:
                text = self._call_stt_endpoint(wav)
            else:
                text = self._call_multimodal(wav)
        except Exception as e:
            print(f"[{self.name}] Error: {e}")

        latency_ms = int((time.time() - start) * 1000)
        return TranscriptionResult(text=text, provider=self.name, mic=mic_name, latency_ms=latency_ms)

    def _call_stt_endpoint(self, wav: bytes) -> str:
        # OpenRouter STT uses JSON + base64, not multipart form data
        b64 = base64.b64encode(wav).decode("utf-8")
        response = self._post_json(
            "/audio/transcriptions",
            {
                "model": self.model,
                "input_audio": {"data": b64, "format": "wav"},
            },
        )
        return response.json().get("text", "")

    def _call_multimodal(self, wav: bytes) -> str:
        b64 = base64.b64encode(wav).decode("utf-8")
        data: dict = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transcribe this audio exactly as spoken. Return only the transcription text, nothing else.",
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {"data": b64, "format": "wav"},
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 4000,
        }
        if self._reasoning_effort:
            data["reasoning"] = {"effort": self._reasoning_effort}

        response = self._post_json("/chat/completions", data)
        return response.json()["choices"][0]["message"]["content"] or ""

    def _post_json(self, path: str, payload: dict) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            session = self._new_session()
            try:
                response = session.post(
                    f"{_BASE_URL}{path}",
                    json=payload,
                    timeout=_REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                return response
            except _RETRYABLE_ERRORS as e:
                last_error = e
                if attempt < _MAX_ATTEMPTS:
                    print(f"[{self.name}] Transport error, retrying with fresh connection: {e}")
                    time.sleep(0.25)
                    continue
                raise
            finally:
                session.close()

        raise RuntimeError(f"OpenRouter request failed without response: {last_error}")

    def _new_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Connection": "close",
        })
        return session

    def shutdown(self) -> None:
        self._initialized = False
        print(f"[{self.name}] Shutdown")
