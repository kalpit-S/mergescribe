"""
API key validation with latency measurement.

Provides quick validation tests for each provider's API key.
"""

import time
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, Future

import requests


@dataclass
class ValidationResult:
    """Result of an API key validation test."""
    valid: bool
    latency_ms: Optional[int] = None
    error: Optional[str] = None


# Shared session for connection reuse
_session = requests.Session()


def validate_groq_key(api_key: str) -> ValidationResult:
    """Validate Groq API key with a minimal request."""
    if not api_key or len(api_key) < 10:
        return ValidationResult(valid=False, error="Key too short")

    try:
        start = time.perf_counter()
        response = _session.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        latency = int((time.perf_counter() - start) * 1000)

        if response.status_code == 200:
            return ValidationResult(valid=True, latency_ms=latency)
        elif response.status_code == 401:
            return ValidationResult(valid=False, error="Invalid key")
        else:
            return ValidationResult(valid=False, error=f"HTTP {response.status_code}")

    except requests.Timeout:
        return ValidationResult(valid=False, error="Timeout")
    except Exception as e:
        return ValidationResult(valid=False, error=str(e)[:50])


def validate_gemini_key(api_key: str) -> ValidationResult:
    """Validate Gemini API key with a minimal request."""
    if not api_key or len(api_key) < 10:
        return ValidationResult(valid=False, error="Key too short")

    try:
        start = time.perf_counter()
        response = _session.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            timeout=10,
        )
        latency = int((time.perf_counter() - start) * 1000)

        if response.status_code == 200:
            return ValidationResult(valid=True, latency_ms=latency)
        elif response.status_code == 400 or response.status_code == 403:
            return ValidationResult(valid=False, error="Invalid key")
        else:
            return ValidationResult(valid=False, error=f"HTTP {response.status_code}")

    except requests.Timeout:
        return ValidationResult(valid=False, error="Timeout")
    except Exception as e:
        return ValidationResult(valid=False, error=str(e)[:50])


def validate_openrouter_key(api_key: str) -> ValidationResult:
    """Validate OpenRouter API key with a minimal request."""
    if not api_key or len(api_key) < 10:
        return ValidationResult(valid=False, error="Key too short")

    try:
        start = time.perf_counter()
        response = _session.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        latency = int((time.perf_counter() - start) * 1000)

        if response.status_code == 200:
            return ValidationResult(valid=True, latency_ms=latency)
        elif response.status_code == 401:
            return ValidationResult(valid=False, error="Invalid key")
        else:
            return ValidationResult(valid=False, error=f"HTTP {response.status_code}")

    except requests.Timeout:
        return ValidationResult(valid=False, error="Timeout")
    except Exception as e:
        return ValidationResult(valid=False, error=str(e)[:50])


class KeyValidator:
    """
    Async key validator that runs tests in background threads.

    Usage:
        validator = KeyValidator()
        validator.validate_groq(key, on_result=lambda r: update_ui(r))
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=3)
        self._pending: dict[str, Future] = {}

    def validate_groq(self, api_key: str, on_result: callable) -> None:
        """Validate Groq key in background, call on_result when done."""
        self._cancel_pending("groq")
        if not api_key:
            on_result(ValidationResult(valid=False, error="No key"))
            return
        future = self._executor.submit(validate_groq_key, api_key)
        self._pending["groq"] = future
        future.add_done_callback(lambda f: self._handle_result("groq", f, on_result))

    def validate_gemini(self, api_key: str, on_result: callable) -> None:
        """Validate Gemini key in background, call on_result when done."""
        self._cancel_pending("gemini")
        if not api_key:
            on_result(ValidationResult(valid=False, error="No key"))
            return
        future = self._executor.submit(validate_gemini_key, api_key)
        self._pending["gemini"] = future
        future.add_done_callback(lambda f: self._handle_result("gemini", f, on_result))

    def validate_openrouter(self, api_key: str, on_result: callable) -> None:
        """Validate OpenRouter key in background, call on_result when done."""
        self._cancel_pending("openrouter")
        if not api_key:
            on_result(ValidationResult(valid=False, error="No key"))
            return
        future = self._executor.submit(validate_openrouter_key, api_key)
        self._pending["openrouter"] = future
        future.add_done_callback(lambda f: self._handle_result("openrouter", f, on_result))

    def _cancel_pending(self, provider: str) -> None:
        """Cancel any pending validation for this provider."""
        if provider in self._pending:
            self._pending[provider].cancel()
            del self._pending[provider]

    def _handle_result(self, provider: str, future: Future, on_result: callable) -> None:
        """Handle completed validation."""
        if provider in self._pending and self._pending[provider] == future:
            del self._pending[provider]

        if future.cancelled():
            return

        try:
            result = future.result()
            on_result(result)
        except Exception as e:
            on_result(ValidationResult(valid=False, error=str(e)[:50]))

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)
