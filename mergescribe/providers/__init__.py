"""
Transcription providers with lifecycle management.

Each provider holds its own state (model weights, HTTP clients)
and provides a consistent interface for transcription.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import threading

import numpy as np

from ..types import TranscriptionResult


class Provider(ABC):
    """
    Base class for transcription providers.

    Subclasses must implement:
    - initialize(): Load model weights / create HTTP client
    - transcribe(): Transcribe audio to text
    - shutdown(): Free resources
    """

    name: str = "base"

    # Providers wrapping a single non-thread-safe model serialize across mics:
    # the second mic waits on the first to release the lock, so running them on
    # every mic only queues work onto the critical path. Measured on parakeet,
    # a second mic cost 590ms median while network-bound fish-audio cost 20ms.
    # When True, the session runs this provider on the primary mic only.
    single_instance: bool = False

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the provider.

        For local models: Load weights into memory.
        For cloud APIs: Create HTTP client.
        """
        pass

    @abstractmethod
    def transcribe(self, audio: np.ndarray, mic_name: str = "") -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (16kHz, mono, float32)
            mic_name: Name of the microphone (for result metadata)

        Returns:
            TranscriptionResult with text and timing info
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        Shutdown the provider and free resources.

        For local models: Unload weights.
        For cloud APIs: Close HTTP client.
        """
        pass


class ProviderRegistry:
    """
    Manages provider lifecycle and parallel transcription.

    Usage:
        registry = ProviderRegistry()
        registry.register(ParakeetProvider())
        registry.register(OpenRouterSTTProvider(api_key, model))

        results = registry.transcribe_all(audio, mic_name)
    """

    def __init__(self, max_workers: int = 8):
        self.providers: Dict[str, Provider] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()

    def register(self, provider: Provider) -> None:
        """
        Register and initialize a provider.

        Args:
            provider: Provider instance to register
        """
        provider.initialize()
        old_provider: Optional[Provider] = None
        with self._lock:
            old_provider = self.providers.get(provider.name)
            self.providers[provider.name] = provider
        if old_provider and old_provider is not provider:
            try:
                old_provider.shutdown()
            except Exception as e:
                print(f"Error shutting down replaced provider {old_provider.name}: {e}")

    def unregister(self, name: str) -> None:
        """Remove and shutdown one provider by name."""
        with self._lock:
            provider = self.providers.pop(name, None)
        if provider:
            try:
                provider.shutdown()
            except Exception as e:
                print(f"Error shutting down {name}: {e}")

    def get(self, name: str) -> Optional[Provider]:
        """Get a provider by name."""
        with self._lock:
            return self.providers.get(name)

    def names(self) -> List[str]:
        """Get registered provider names."""
        with self._lock:
            return list(self.providers.keys())

    def values(self) -> List[Provider]:
        """Get all registered providers."""
        with self._lock:
            return list(self.providers.values())

    def transcribe_all(
        self,
        audio: np.ndarray,
        mic_name: str = "",
        timeout: float = 30.0
    ) -> List[TranscriptionResult]:
        """
        Run all providers in parallel on the same audio.

        Args:
            audio: Audio data
            mic_name: Microphone name for metadata
            timeout: Maximum time to wait for all providers

        Returns:
            List of transcription results
        """
        with self._lock:
            providers = list(self.providers.values())

        if not providers:
            return []

        futures = {
            self._executor.submit(p.transcribe, audio, mic_name): p.name
            for p in providers
        }

        results = []
        try:
            for future in as_completed(futures, timeout=timeout):
                provider_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Provider {provider_name} error: {e}")
        except TimeoutError:
            print(f"[ProviderRegistry] Timeout after {timeout}s waiting for providers")
            # Cancel any pending futures
            for f in futures:
                f.cancel()

        return results

    def shutdown(self) -> None:
        """Shutdown all providers and the executor."""
        with self._lock:
            for provider in self.providers.values():
                try:
                    provider.shutdown()
                except Exception as e:
                    print(f"Error shutting down {provider.name}: {e}")
            self.providers.clear()

        self._executor.shutdown(wait=True)
