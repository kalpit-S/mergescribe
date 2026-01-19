"""
Correction provider routing.

Automatically routes to the best available LLM provider based on:
- Which API keys are configured
- Input length (short → fast provider, long → smart provider)
- Provider health (tracks failures, backs off)
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict
from collections import defaultdict
import threading
import time

from .types import ConfigSnapshot


# Module-level state for failure tracking (persists across router instances)
_PROVIDER_FAILURES: Dict[str, int] = defaultdict(int)
_PROVIDER_BACKOFF_UNTIL: Dict[str, float] = defaultdict(float)
_ROUTER_LOCK = threading.Lock()


# Hardcoded best models for each provider
GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-3-flash-preview"
OPENROUTER_MODEL = "google/gemini-2.5-flash"

# Routing threshold
SHORT_INPUT_WORD_THRESHOLD = 20


@dataclass
class CorrectionProvider:
    """A correction provider with its characteristics."""
    name: str
    key_field: str          # Config field that holds the API key
    latency_ms: int         # Estimated latency (for routing decisions)
    priority: int           # Lower = preferred (for same-tier decisions)
    model: str              # The model to use

    def is_available(self, config: ConfigSnapshot) -> bool:
        """Check if this provider's API key is configured."""
        return bool(getattr(config, self.key_field, ""))


# Provider registry
# - latency_ms: for short inputs, pick fastest
# - priority: for long inputs, pick best quality (lower = better quality)
PROVIDERS = [
    CorrectionProvider("groq", "groq_api_key", latency_ms=400, priority=2, model=GROQ_MODEL),
    CorrectionProvider("gemini", "gemini_api_key", latency_ms=700, priority=1, model=GEMINI_MODEL),
    CorrectionProvider("openrouter", "openrouter_api_key", latency_ms=900, priority=3, model=OPENROUTER_MODEL),
]


class CorrectionRouter:
    """
    Routes correction requests to the best available provider.

    Routing logic:
    - Short inputs (<20 words): Pick fastest available provider
    - Long inputs (20+ words): Pick by priority (quality)
    - Tracks failures and backs off unhealthy providers
    """

    def __init__(self, config: ConfigSnapshot):
        self.config = config
        # Use module-level state for persistence across instances
        self._failures = _PROVIDER_FAILURES
        self._backoff_until = _PROVIDER_BACKOFF_UNTIL

    def get_available_providers(self) -> List[CorrectionProvider]:
        """Get all providers that have API keys configured."""
        now = time.time()
        available = []
        with _ROUTER_LOCK:
            for p in PROVIDERS:
                if p.is_available(self.config):
                    # Skip if in backoff period
                    if now < self._backoff_until.get(p.name, 0):
                        continue
                    available.append(p)
        return available

    def select_provider(self, word_count: int) -> Optional[CorrectionProvider]:
        """
        Select the best provider for the given input.

        Args:
            word_count: Number of words in the transcription

        Returns:
            Best available provider, or None if no providers configured
        """
        available = self.get_available_providers()
        if not available:
            return None

        if word_count < SHORT_INPUT_WORD_THRESHOLD:
            # Short input: pick fastest
            return min(available, key=lambda p: p.latency_ms)
        else:
            # Long input: pick by priority (quality)
            return min(available, key=lambda p: p.priority)

    def get_fallback(self, exclude: str) -> Optional[CorrectionProvider]:
        """Get next best provider, excluding the one that failed."""
        available = [p for p in self.get_available_providers() if p.name != exclude]
        if not available:
            return None
        return min(available, key=lambda p: p.priority)

    def record_failure(self, provider_name: str) -> None:
        """Record a provider failure for backoff logic."""
        with _ROUTER_LOCK:
            self._failures[provider_name] += 1
            failures = self._failures[provider_name]

            if failures >= 3:
                # Exponential backoff: 2^failures seconds, max 5 minutes
                backoff_seconds = min(2 ** failures, 300)
                self._backoff_until[provider_name] = time.time() + backoff_seconds
                print(f"[Router] {provider_name} backing off for {backoff_seconds}s after {failures} failures")

    def record_success(self, provider_name: str) -> None:
        """Record a provider success, reset failure count."""
        with _ROUTER_LOCK:
            self._failures[provider_name] = 0
            self._backoff_until[provider_name] = 0

    def get_routing_status(self) -> str:
        """Get human-readable routing status for UI display."""
        available = self.get_available_providers()
        if not available:
            return "No API keys configured"

        # Determine what would be used for short vs long
        short_provider = self.select_provider(word_count=5)
        long_provider = self.select_provider(word_count=50)

        if short_provider == long_provider:
            return f"All inputs → {short_provider.name} ({short_provider.model})"
        else:
            return (
                f"Short → {short_provider.name} (~{short_provider.latency_ms}ms)\n"
                f"Long → {long_provider.name} (~{long_provider.latency_ms}ms)"
            )


def get_provider_status(config: ConfigSnapshot) -> Dict[str, dict]:
    """
    Get status of all providers for UI display.

    Returns dict like:
    {
        "groq": {"configured": True, "model": "openai/gpt-oss-120b"},
        "gemini": {"configured": False, "model": "gemini-3-flash-preview"},
        ...
    }
    """
    status = {}
    for p in PROVIDERS:
        status[p.name] = {
            "configured": p.is_available(config),
            "model": p.model,
            "latency_ms": p.latency_ms,
        }
    return status
