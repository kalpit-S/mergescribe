"""
Shared type definitions for MergeScribe.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable
from uuid import UUID
import numpy as np


@dataclass
class TranscriptionResult:
    """Result from a single provider transcribing a single mic's audio."""
    text: str
    provider: str
    mic: str
    latency_ms: int
    confidence: Optional[float] = None


@dataclass
class AppContext:
    """Information about the active application when recording started."""
    app_name: str           # e.g., "Code"
    window_title: str       # e.g., "main.py - mergescribe"
    bundle_id: str          # e.g., "com.microsoft.VSCode"
    rigor_level: str        # "high" | "low" | "normal"


@dataclass
class ConfigSnapshot:
    """
    Immutable snapshot of configuration for a session.
    Ensures config changes mid-session don't cause inconsistency.
    """
    # Audio
    enabled_mics: List[str]
    preroll_seconds: float
    silence_threshold: float
    sample_rate: int

    # Input
    double_tap_threshold: float
    toggle_mode_timeout: float

    # Providers
    enabled_providers: List[str]

    # Processing
    consensus_threshold: int
    consensus_max_words: int

    # API Keys
    openrouter_api_key: str
    groq_api_key: str
    gemini_api_key: str

    # User customization
    custom_instructions: str = ""

    # Advanced settings
    system_prompt: str = ""  # Custom system prompt for LLM correction
    editing_prompt: str = ""  # Custom prompt for text editing mode

    # Training data collection (local only)
    training_enabled: bool = False
    training_data_dir: str = ""


@dataclass
class LLMCorrectionResult:
    """Result from LLM correction with metadata for logging."""
    text: str
    provider: str           # "groq", "gemini", "openrouter"
    model: str              # e.g., "moonshotai/kimi-k2-instruct-0905"
    input_tokens_est: int   # Estimated input tokens
    latency_ms: float
    streamed: bool = False


@dataclass
class TrainingMetadata:
    """Complete metadata for a training sample (saved as metadata.json)."""
    session_id: str
    timestamp: str                              # ISO format
    duration_ms: float
    sample_rate: int
    schema_version: int = 1                     # For future format evolution

    # Context
    app_context: Optional[Dict] = None          # Serialized AppContext

    # Transcription results
    transcriptions: List[Dict] = field(default_factory=list)

    # Consensus
    consensus: Optional[Dict] = None            # {reached, text, count}

    # LLM correction (if called)
    llm_correction: Optional[Dict] = None       # {provider, model, input_text, output_text, latency_ms}

    # Output
    final_output: str = ""
    output_method: str = ""                     # "typed" | "clipboard" | "streamed"


# Type aliases
AudioChunk = Dict[str, np.ndarray]  # {mic_name: audio_array}
ChunkResult = tuple[List[TranscriptionResult], Optional[str]]  # (results, consensus_if_found)
