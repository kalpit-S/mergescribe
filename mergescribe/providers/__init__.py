"""
Transcription providers with lifecycle management.

Each provider holds its own state (model weights, HTTP clients)
and provides a consistent interface for transcription.
"""

from abc import ABC, abstractmethod
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
    The recognizers currently configured, and their lifecycle. Sessions run
    them, chunk by chunk, with their own consensus and deadlines.
    """

    def __init__(self):
        self.providers: Dict[str, Provider] = {}
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

    def shutdown(self) -> None:
        """Shutdown all providers and the executor."""
        with self._lock:
            for provider in self.providers.values():
                try:
                    provider.shutdown()
                except Exception as e:
                    print(f"Error shutting down {provider.name}: {e}")
            self.providers.clear()
