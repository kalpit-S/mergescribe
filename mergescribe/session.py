"""
Session management for recording lifecycle.

A Session represents one recording from start to finish, including
chunk transcription, consensus checking, and LLM correction.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Callable, List, Tuple, Dict, TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np

from .types import (
    TranscriptionResult, AppContext, ConfigSnapshot, AudioChunk, ChunkResult,
    LLMCorrectionResult, TrainingMetadata,
)
from .providers import ProviderRegistry
from .consensus import check_consensus
from .context import get_app_context, detect_selected_text
from .output import type_text, copy_to_clipboard, notify, play_busy_sound

if TYPE_CHECKING:
    from .metrics import MetricsWriter
    from .training import TrainingDataWriter


@dataclass
class Session:
    """
    Represents one recording session.

    Manages chunk transcription with early consensus.
    Runs processing in background thread to avoid blocking UI.
    """
    id: UUID
    config_snapshot: ConfigSnapshot
    providers: ProviderRegistry
    output_lock: threading.Lock
    on_complete: Callable[["Session"], None]
    history: "TranscriptionHistory"
    metrics: Optional["MetricsWriter"] = None
    training_writer: Optional["TrainingDataWriter"] = None

    # Runtime state
    chunk_results: List[ChunkResult] = field(default_factory=list)
    pending_futures: List[Future] = field(default_factory=list)
    is_active: bool = False
    start_time: float = 0.0
    context: Optional[AppContext] = None
    selected_text: Optional[str] = None  # For text editing mode
    _executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=12))
    _chunk_lock: threading.Lock = field(default_factory=threading.Lock)
    _final_text: str = ""  # Store for adding to history

    # Data collection for metrics and training
    all_audio: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    all_transcription_results: List[TranscriptionResult] = field(default_factory=list)
    llm_result: Optional[LLMCorrectionResult] = None
    output_method: str = ""  # "typed" | "clipboard" | "streamed"
    finalize_start_time: float = 0.0  # When key was released (for processing WPM)

    def start(self) -> None:
        """Capture context at session start."""
        self.is_active = True
        self.start_time = time.time()
        self.context = get_app_context()
        self.selected_text = detect_selected_text()

        # Log session start
        if self.metrics:
            self.metrics.log(
                "session_start",
                session_id=str(self.id),
                context=asdict(self.context) if self.context else {},
                enabled_mics=self.config_snapshot.enabled_mics,
                providers=self.config_snapshot.enabled_providers,
            )

    def on_chunk_ready(self, chunk: AudioChunk) -> None:
        """
        Called by AudioEngine when silence detected.
        Starts transcription of chunk in background.
        """
        if not chunk or all(len(a) == 0 for a in chunk.values()):
            return  # Empty chunk, ignore

        # Accumulate audio for training data (protected by lock)
        with self._chunk_lock:
            for mic_name, audio in chunk.items():
                if len(audio) > 0:
                    if mic_name not in self.all_audio:
                        self.all_audio[mic_name] = []
                    self.all_audio[mic_name].append(audio.copy())

        # Log chunk received
        chunk_num = len(self.chunk_results) + 1
        if self.metrics:
            max_duration = max(len(a) / self.config_snapshot.sample_rate * 1000
                               for a in chunk.values() if len(a) > 0)
            self.metrics.log(
                "chunk_received",
                session_id=str(self.id),
                chunk_num=chunk_num,
                mic_names=list(chunk.keys()),
                audio_duration_ms=max_duration,
            )

        future = self._executor.submit(self._transcribe_chunk_with_consensus, chunk)
        with self._chunk_lock:
            self.pending_futures.append(future)

    def _transcribe_chunk_with_consensus(self, chunk: AudioChunk) -> None:
        """
        Transcribe one chunk, checking consensus as results arrive.

        Runs all mics × all providers in parallel.
        If early consensus is reached, cancels remaining futures.
        """
        # Create futures for all mic × provider combinations
        futures: Dict[Future, Tuple[str, str]] = {}

        for mic_name, audio in chunk.items():
            if len(audio) == 0:
                continue

            for provider in self.providers.values():
                future = self._executor.submit(
                    provider.transcribe, audio, mic_name
                )
                futures[future] = (mic_name, provider.name)

        if not futures:
            return

        results: List[TranscriptionResult] = []
        consensus: Optional[str] = None
        chunk_num = len(self.chunk_results) + 1

        matching_count = 0
        try:
            for future in as_completed(futures, timeout=30):
                try:
                    result = future.result()
                    results.append(result)
                    with self._chunk_lock:
                        self.all_transcription_results.append(result)

                    # Log each transcription result
                    text_preview = result.text[:50] + "..." if len(result.text) > 50 else result.text
                    print(f"[Chunk {chunk_num}] {result.provider}/{result.mic}: {result.latency_ms/1000:.2f}s -> \"{text_preview}\"")

                    # Log to metrics
                    if self.metrics:
                        self.metrics.log(
                            "transcription",
                            session_id=str(self.id),
                            chunk_num=chunk_num,
                            provider=result.provider,
                            mic=result.mic,
                            latency_ms=result.latency_ms,
                            text=result.text[:200],
                            confidence=result.confidence,
                        )

                    # Early consensus check
                    if len(results) >= self.config_snapshot.consensus_threshold:
                        consensus = check_consensus(results, self.config_snapshot)
                        if consensus:
                            # Count matching results for metrics
                            from .consensus import normalize_for_matching
                            norm_consensus = normalize_for_matching(consensus)
                            matching_count = sum(1 for r in results
                                                 if normalize_for_matching(r.text) == norm_consensus)

                            print(f"[Chunk {chunk_num}] ✓ Consensus reached: \"{consensus[:50]}...\"" if len(consensus) > 50 else f"[Chunk {chunk_num}] ✓ Consensus: \"{consensus}\"")
                            # Cancel remaining futures
                            for f in futures:
                                f.cancel()
                            break

                except Exception as e:
                    mic, provider = futures[future]
                    print(f"[Chunk {chunk_num}] Provider error ({provider}/{mic}): {e}")
        except TimeoutError:
            print(f"[Chunk {chunk_num}] Timeout waiting for transcriptions")
            # Cancel any pending futures to free resources
            for f in futures:
                f.cancel()

        # Log consensus result
        if self.metrics:
            self.metrics.log(
                "consensus",
                session_id=str(self.id),
                chunk_num=chunk_num,
                reached=consensus is not None,
                text=consensus[:200] if consensus else None,
                matching_count=matching_count,
            )

        # Store results
        with self._chunk_lock:
            self.chunk_results.append((results, consensus))

    def finalize(self, final_chunk: AudioChunk) -> None:
        """
        Called on key release. Runs finalization in background thread.

        Args:
            final_chunk: The last chunk of audio
        """
        threading.Thread(
            target=self._finalize_impl,
            args=(final_chunk,),
            daemon=True
        ).start()

    def _finalize_impl(self, final_chunk: AudioChunk) -> None:
        """
        Wait for chunks, aggregate, correct, output.

        This runs in a background thread.
        """
        try:
            finalize_start = time.time()
            self.finalize_start_time = finalize_start  # Track for WPM calculation
            key_held_duration = finalize_start - self.start_time

            # Accumulate audio from final_chunk for training data
            if final_chunk:
                for mic, audio in final_chunk.items():
                    if len(audio) > 0:
                        duration_ms = len(audio) / self.config_snapshot.sample_rate * 1000
                        print(f"[Audio] {mic}: {duration_ms/1000:.2f}s of audio")
                        # Accumulate for training
                        if mic not in self.all_audio:
                            self.all_audio[mic] = []
                        self.all_audio[mic].append(audio.copy())

            print(f"[Timing] Key held: {key_held_duration:.2f}s")

            # Transcribe final chunk (if not empty)
            transcribe_start = time.time()
            if final_chunk and any(len(a) > 0 for a in final_chunk.values()):
                self._transcribe_chunk_with_consensus(final_chunk)

            # Wait for any pending futures (copy list first to avoid deadlock)
            with self._chunk_lock:
                futures_to_wait = list(self.pending_futures)

            for future in futures_to_wait:
                try:
                    future.result(timeout=30)
                except Exception:
                    pass

            transcribe_elapsed = (time.time() - transcribe_start) * 1000
            print(f"[Timing] Transcription: {transcribe_elapsed/1000:.2f}s")

            # Aggregate results
            chunk_texts, all_results = self._aggregate_results()

            if not chunk_texts:
                print("No transcription results")
                return

            combined_text = " ".join(chunk_texts)
            print(f"[Session] {len(self.chunk_results)} chunks, {len(all_results)} transcriptions")

            # Text editing mode: transcription is the voice command
            if self.selected_text:
                print(f"[Session] Text edit mode: \"{combined_text[:50]}...\"")
                from .correct import edit_text_with_llm
                edited = edit_text_with_llm(
                    self.selected_text,
                    combined_text,
                    self.config_snapshot,
                )
                self._output(edited)
                return

            # Single chunk with consensus? Fast path
            if len(self.chunk_results) == 1 and self.chunk_results[0][1]:
                print(f"[Session] Fast path (consensus)")
                self.output_method = "typed"
                self._output(self.chunk_results[0][1])
                return

            # LLM correction with streaming output
            from .correct import correct_with_llm
            history_context = self.history.get_context()

            # Callback to capture LLM metadata
            def on_llm_metadata(result: LLMCorrectionResult) -> None:
                self.llm_result = result
                if self.metrics:
                    self.metrics.log(
                        "llm_correction",
                        session_id=str(self.id),
                        provider=result.provider,
                        model=result.model,
                        input_tokens_est=result.input_tokens_est,
                        latency_ms=result.latency_ms,
                    )

            # Check if we can stream (window hasn't changed)
            current_context = get_app_context()
            can_stream = (
                self.context and current_context and
                current_context.bundle_id == self.context.bundle_id
            )

            if can_stream:
                # Stream tokens directly as they arrive
                streamed_tokens: List[str] = []
                self.output_method = "streamed"

                def on_token(token: str) -> None:
                    streamed_tokens.append(token)
                    with self.output_lock:
                        type_text(token)

                correct_with_llm(
                    all_results,
                    self.context,
                    self.config_snapshot,
                    on_delta=on_token,
                    history_context=history_context,
                    on_metadata=on_llm_metadata,
                    custom_instructions=self.config_snapshot.custom_instructions,
                )

                corrected = "".join(streamed_tokens)
                self._final_text = corrected
                self.history.add(corrected)
            else:
                # Window changed - fall back to clipboard (no streaming)
                self.output_method = "clipboard"
                corrected = correct_with_llm(
                    all_results,
                    self.context,
                    self.config_snapshot,
                    history_context=history_context,
                    on_metadata=on_llm_metadata,
                    custom_instructions=self.config_snapshot.custom_instructions,
                )

                self._final_text = corrected
                copy_to_clipboard(corrected)
                notify("Window changed - copied to clipboard")
                self.history.add(corrected)

        except Exception as e:
            print(f"Session finalize error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Log session complete
            total_duration_ms = (time.time() - self.start_time) * 1000
            if self.metrics:
                self.metrics.log(
                    "session_complete",
                    session_id=str(self.id),
                    total_duration_ms=total_duration_ms,
                    chunks=len(self.chunk_results),
                    final_text=self._final_text[:500] if self._final_text else "",
                )

            # Save training data if enabled
            if (self.training_writer
                and self.config_snapshot.training_enabled
                and self._final_text
                and self.all_audio):
                self._save_training_data()

            self.is_active = False
            self._executor.shutdown(wait=False)
            self.on_complete(self)

    def _aggregate_results(self) -> Tuple[List[str], List[TranscriptionResult]]:
        """
        Aggregate chunk results.

        Per-chunk: use consensus if found, else longest result (most content).
        Returns: (list of chunk texts, flat list of all results)
        """
        chunk_texts: List[str] = []
        all_results: List[TranscriptionResult] = []

        for results, consensus in self.chunk_results:
            all_results.extend(results)

            if consensus:
                chunk_texts.append(consensus)
            elif results:
                # No consensus - use longest result (most likely has real content)
                best = max(results, key=lambda r: len(r.text.split()))
                chunk_texts.append(best.text)

        return chunk_texts, all_results

    def _output(self, text: str) -> None:
        """
        Thread-safe output. Verifies window hasn't changed.

        If the active window changed since recording started,
        copies to clipboard instead of typing.
        """
        if not text:
            return

        # Store for history
        self._final_text = text

        # Determine correction provider
        correction_provider = "consensus" if self.llm_result is None else self.llm_result.provider

        with self.output_lock:
            current_context = get_app_context()

            # Check if window changed (with null safety)
            if self.context and current_context:
                if current_context.bundle_id != self.context.bundle_id:
                    # Window changed! Copy to clipboard instead
                    copy_to_clipboard(text)
                    self.output_method = "clipboard"  # Track actual output method
                    print(f"[Timing] Output: clipboard (window changed)")
                    notify("Window changed - copied to clipboard")
                    self.history.add(text)
                    return

            type_start = time.time()
            type_text(text)
            self.output_method = "typed"  # Track actual output method
            type_end = time.time()
            type_elapsed = (type_end - type_start) * 1000

            # Calculate WPM metrics
            word_count = len(text.split())
            total_time = type_end - self.start_time  # Key press to typing done
            processing_time = type_end - self.finalize_start_time  # Key release to typing done

            total_wpm = (word_count / total_time) * 60 if total_time > 0 else 0
            processing_wpm = (word_count / processing_time) * 60 if processing_time > 0 else 0

            # Log output with provider and WPM
            print(f"[Output] {correction_provider} | {total_time:.2f}s total | {word_count} words")
            print(f"[Output] \"{text}\"")
            print(f"[WPM] Total: {total_wpm:.0f} wpm (from key press) | Processing: {processing_wpm:.0f} wpm (from key release)")

        # Add to history after successful output
        self.history.add(text)

    def _save_training_data(self) -> None:
        """Collect and save all session data for training."""
        # Concatenate all audio chunks per mic (copy under lock to avoid race)
        with self._chunk_lock:
            audio_data: Dict[str, np.ndarray] = {}
            for mic_name, chunks in self.all_audio.items():
                if chunks:
                    audio_data[mic_name] = np.concatenate(chunks)
            # Also copy transcription results
            transcription_results = list(self.all_transcription_results)

        if not audio_data:
            return

        # Build consensus info
        consensus_info: Optional[Dict] = None
        for results, consensus in self.chunk_results:
            if consensus:
                from .consensus import normalize_for_matching
                norm_consensus = normalize_for_matching(consensus)
                matching_count = sum(1 for r in results
                                     if normalize_for_matching(r.text) == norm_consensus)
                consensus_info = {
                    "reached": True,
                    "text": consensus,
                    "count": matching_count,
                }
                break
        if consensus_info is None:
            consensus_info = {"reached": False, "text": None, "count": 0}

        # Build LLM correction info
        llm_info: Optional[Dict] = None
        if self.llm_result:
            # Get input text (aggregated transcriptions)
            input_texts = [r.text for r in transcription_results[:10]]
            llm_info = {
                "provider": self.llm_result.provider,
                "model": self.llm_result.model,
                "input_text": " | ".join(input_texts),
                "output_text": self._final_text,
                "latency_ms": self.llm_result.latency_ms,
            }

        # Create metadata
        metadata = TrainingMetadata(
            session_id=str(self.id),
            timestamp=datetime.now().isoformat(),
            duration_ms=(time.time() - self.start_time) * 1000,
            sample_rate=self.config_snapshot.sample_rate,
            app_context=asdict(self.context) if self.context else None,
            transcriptions=[asdict(r) for r in transcription_results],
            consensus=consensus_info,
            llm_correction=llm_info,
            final_output=self._final_text,
            output_method=self.output_method or "typed",
        )

        self.training_writer.save_session(self.id, audio_data, metadata)


class TranscriptionHistory:
    """
    Stores recent transcriptions for context continuity.

    Keeps last N transcriptions within a time window.
    """

    def __init__(self, max_entries: int = 5, max_age_seconds: float = 300):
        self.max_entries = max_entries
        self.max_age_seconds = max_age_seconds
        self._entries: List[Tuple[float, str]] = []
        self._lock = threading.Lock()

    def add(self, text: str) -> None:
        """Add a transcription to history."""
        if not text or not text.strip():
            return

        with self._lock:
            self._entries.append((time.time(), text.strip()))
            # Prune old entries
            self._prune()

    def get_context(self) -> str:
        """
        Get recent transcriptions as context string.

        Returns pipe-separated recent entries, newest last.
        """
        with self._lock:
            self._prune()
            if not self._entries:
                return ""
            texts = [text for _, text in self._entries[-self.max_entries:]]
            return " | ".join(texts)

    def _prune(self) -> None:
        """Remove entries older than max_age_seconds and enforce max_entries."""
        cutoff = time.time() - self.max_age_seconds
        self._entries = [(t, text) for t, text in self._entries if t > cutoff]
        # Also enforce max_entries to prevent unbounded growth
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]


class SessionManager:
    """
    Manages active session, rejects if busy.

    Ensures only one session processes at a time.
    No queueing - if busy, plays error sound and rejects.
    """

    def __init__(
        self,
        config_snapshot_fn: Callable[[], ConfigSnapshot],
        providers: ProviderRegistry,
        metrics: Optional["MetricsWriter"] = None,
        training_writer: Optional["TrainingDataWriter"] = None,
    ):
        self.config_snapshot_fn = config_snapshot_fn
        self.providers = providers
        self.metrics = metrics
        self.training_writer = training_writer

        self.active_session: Optional[Session] = None
        self._lock = threading.Lock()
        self._output_lock = threading.Lock()
        self.history = TranscriptionHistory()

    def start_session(self) -> Optional[Session]:
        """
        Create and start a new session.

        Returns None if a session is already active (plays error sound).
        """
        with self._lock:
            if self.active_session and self.active_session.is_active:
                play_busy_sound()
                return None

            session = Session(
                id=uuid4(),
                config_snapshot=self.config_snapshot_fn(),
                providers=self.providers,
                output_lock=self._output_lock,
                on_complete=self._on_session_complete,
                history=self.history,
                metrics=self.metrics,
                training_writer=self.training_writer,
            )

            self.active_session = session
            return session

    def _on_session_complete(self, session: Session) -> None:
        """Called when a session finishes."""
        with self._lock:
            if self.active_session == session:
                self.active_session = None

    def is_busy(self) -> bool:
        """Check if a session is currently active."""
        with self._lock:
            return self.active_session is not None and self.active_session.is_active
