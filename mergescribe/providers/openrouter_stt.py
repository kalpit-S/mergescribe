"""
OpenRouter STT provider for cloud transcription.

Every OpenRouter transcription model is served by /audio/transcriptions
(the catalogue is /models?output_modalities=transcription), so any of them
works here without a table of which model takes which endpoint.
"""

import base64
import io
import re
import time


from typing import Callable, List, Optional

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

# Non-speech annotations: fish-audio's speaker labels (<|speaker:0|>) and sound
# events ([typing sound], [pause], [chuckle]), whisper's [BLANK_AUDIO]. A raw
# transcript can be typed as it stands - when the judge finds it clean, or the
# correction fails - and consensus compares transcripts word for word.
_ANNOTATION = re.compile(r"<\|[^|>]*\|>|\[[^\]\n]{1,40}\]")


def strip_annotations(text: str) -> str:
    return " ".join(_ANNOTATION.sub(" ", text).split())


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    audio_int16 = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, audio_int16, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.getvalue()


def _model_slug(model: str) -> str:
    return model.replace("/", "-").replace(".", "-").replace(":", "-")


class OpenRouterSTTProvider(Provider):
    """Cloud STT via OpenRouter — one instance per model."""

    def __init__(self, api_key: str, model: str,
                 keyterms: Optional[Callable[[], List[str]]] = None):
        self.api_key = api_key
        self.model = model
        # Words the speaker is known to use, for models that can be primed with
        # them (AssemblyAI's keyterms). Called per request, so it stays current.
        self._keyterms = keyterms
        self.name = f"or-{_model_slug(model)}"
        self._initialized = False

    def initialize(self) -> None:
        if not self.api_key:
            print(f"[{self.name}] No OpenRouter API key")
            return
        self._initialized = True
        print(f"[{self.name}] Initialized")

    def transcribe(self, audio: np.ndarray, mic_name: str = "") -> TranscriptionResult:
        start = time.time()
        text = ""

        if not self._initialized:
            return TranscriptionResult(text="", provider=self.name, mic=mic_name, latency_ms=0)

        try:
            text = self._call_stt_endpoint(_audio_to_wav_bytes(audio))
        except Exception as e:
            print(f"[{self.name}] Error: {e}")

        latency_ms = int((time.time() - start) * 1000)
        return TranscriptionResult(text=strip_annotations(text), provider=self.name, mic=mic_name,
                                   latency_ms=latency_ms)

    def _call_stt_endpoint(self, wav: bytes) -> str:
        # OpenRouter STT uses JSON + base64, not multipart form data
        b64 = base64.b64encode(wav).decode("utf-8")
        payload: dict = {"model": self.model, "input_audio": {"data": b64, "format": "wav"}}
        terms = self._terms() if self.model.startswith("assemblyai/") else []
        if terms:
            payload["provider"] = {"options": {"assemblyai": {"keyterms_prompt": terms}}}
        response = self._post_json("/audio/transcriptions", payload)
        return response.json().get("text", "")

    def _terms(self) -> List[str]:
        if self._keyterms is None:
            return []
        try:
            return list(self._keyterms())
        except Exception as e:
            print(f"[{self.name}] Couldn't read vocabulary: {e}")
            return []

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
