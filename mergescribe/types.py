"""
Shared type definitions for MergeScribe.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
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

    # OpenRouter STT models to run in parallel (e.g. ["openai/gpt-4o-transcribe"])
    openrouter_stt_models: List[str] = field(default_factory=list)

    # Which model the OpenRouter correction call uses
    openrouter_correction_model: str = "google/gemini-3.1-flash-lite"
    openrouter_correction_provider_order: List[str] = field(default_factory=list)
    openrouter_correction_allow_fallbacks: bool = True
    openrouter_correction_reasoning_effort: str = ""

    # User customization
    custom_instructions: str = ""

    # Advanced settings
    system_prompt: str = ""  # Custom system prompt for LLM correction
    editing_prompt: str = ""  # Custom prompt for text editing mode

    # Training data collection (local only)
    training_enabled: bool = False
    training_data_dir: str = ""

    # Voice-driven output routing (AX field inventory + TARGET prefix).
    # Experimental, opt-in — see DEFAULT_CONFIG.
    field_routing_enabled: bool = False
    routing_allowed_apps: List[str] = field(default_factory=list)
    routing_instructions: str = ""

    # Post-output edit detection (implicit correction signal)
    edit_feedback_enabled: bool = True

    # Separate consecutive dictations with a space (leading, when continuing).
    space_between_dictations: bool = True

    # Offer terms the user has corrected twice to the correction model.
    learn_vocabulary: bool = True

    # Stop waiting for stragglers once the fastest provider has answered and
    # the slowest is taking disproportionately long. 0 disables the deadline.
    provider_deadline_multiplier: float = 1.0
    provider_deadline_min_ms: int = 2000


@dataclass
class LLMCorrectionResult:
    """Result from LLM correction with metadata for logging."""
    text: str
    provider: str           # "openrouter"
    model: str              # e.g., "moonshotai/kimi-k2-instruct-0905"
    input_tokens_est: int   # Estimated input tokens
    latency_ms: float
    streamed: bool = False
    generation_id: Optional[str] = None
    backend_provider: Optional[str] = None      # OpenRouter upstream provider, if known
    resolved_model: Optional[str] = None        # Provider/model revision returned by API, if known
    provider_order: List[str] = field(default_factory=list)
    allow_fallbacks: Optional[bool] = None
    reasoning_effort: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)


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
    config_snapshot: Optional[Dict] = None      # Safe config fields, no API keys

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
