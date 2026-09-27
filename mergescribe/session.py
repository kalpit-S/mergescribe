"""
Session management for recording lifecycle.

A Session represents one recording from start to finish, including
chunk transcription, consensus checking, and LLM correction.
"""

import threading
import time
from concurrent.futures import (ThreadPoolExecutor, Future, wait,
                                FIRST_COMPLETED, ALL_COMPLETED)
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Callable, List, Protocol, Sequence, Tuple, Dict, TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np

from .types import (
    TranscriptionResult, AppContext, ConfigSnapshot, AudioChunk, ChunkResult,
    LLMCorrectionResult, TrainingMetadata,
)
from .providers import Provider, ProviderRegistry
from .consensus import check_consensus
from .context import Origin, at_origin, capture_origin, detect_selected_text, get_app_context, return_to
from .logging_tag import set_tag
from .output import (type_text, copy_to_clipboard, notify, play_busy_sound,
                     DictationFilter, NothingMarker, is_nothing)

if TYPE_CHECKING:
    from .metrics import MetricsWriter
    from .training import TrainingDataWriter


# Floor for sending audio to a transcriber at all. The preroll buffer means an
# instant tap still yields ~1s of room tone, so duration alone is not enough.
MIN_SESSION_SECONDS = 0.35
MIN_SESSION_DB = -55.0

# How recently the previous dictation must have landed for this one to be
# treated as continuing it rather than starting fresh.
_CONTINUATION_SECONDS = 8.0

# Absolute ceiling on waiting for a chunk's providers, regardless of deadline.
_CHUNK_HARD_TIMEOUT = 30.0


class SessionObserver(Protocol):
    """
    What a session reports as it runs, for display.

    A stream is one recognizer on one mic ("parakeet/MacBook Pro Microphone").
    Every call names the session it comes from, because a session still typing
    can overlap the next one's recording.
    """

    def streams_planned(self, session: str, streams: List[str]) -> None:
        """Recording started; these streams will transcribe it."""

    def stream_landed(self, session: str, stream: str, final: bool) -> None:
        """A stream returned a chunk. final: the chunk recorded after release."""

    def stream_dropped(self, session: str, stream: str) -> None:
        """A stream failed, or was left behind by the provider deadline."""

    def transcription_done(self, session: str) -> None:
        """Every chunk is in; correction starts now."""

    def correction_skipped(self, session: str) -> None:
        """The transcript needed no correcting and was typed as it stood."""

    def dictation_discarded(self, session: str) -> None:
        """The speaker called it off; nothing will be typed."""

    def token_typed(self, session: str) -> None:
        """A word just reached the screen."""

    def partial_text(self, session: str, text: str) -> None:
        """The raw transcript so far, while recording."""


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
    # True only while audio is being captured. Finalization (LLM + typing) runs
    # long after the mic is free, and output is kept in order by the manager's
    # OutputOrder, so a new recording may safely begin while the previous
    # session is still finalising. Gating new sessions on the whole session
    # instead made the app deaf for its ~2s tail.
    is_capturing: bool = False
    start_time: float = 0.0
    context: Optional[AppContext] = None
    selected_text: Optional[str] = None  # For text editing mode
    origin: Optional[Origin] = None      # the app and window the dictation started in
    output_order: Optional["OutputOrder"] = None
    _ticket: Optional[int] = None
    _turn_taken: bool = False
    _partial_output: bool = False        # only part of the text reached the field
    observer: Optional[SessionObserver] = None
    _executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=12))
    _chunk_lock: threading.Lock = field(default_factory=threading.Lock)
    # Monotonic chunk numbering. len(chunk_results)+1 raced: chunk_results is
    # only appended once transcription finishes, so a pause-emitted chunk and
    # the final chunk transcribing concurrently both read the same length and
    # logged themselves as the same chunk number.
    _chunk_counter: int = 0
    _final_text: str = ""  # Store for adding to history

    # Data collection for metrics and training
    all_audio: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    all_transcription_results: List[TranscriptionResult] = field(default_factory=list)
    _context_thread: Optional[threading.Thread] = None
    llm_result: Optional[LLMCorrectionResult] = None
    output_method: str = ""  # "typed" | "clipboard" | "streamed"
    finalize_start_time: float = 0.0  # When key was released (for processing WPM)

    @property
    def tag(self) -> str:
        """Short handle for the log, so overlapping sessions can be told apart."""
        return str(self.id)[:4]

    def start(self, mics: Sequence[str] = ()) -> None:
        """Mark the session live and capture context off the critical path.

        mics are the microphones recording, so observers can be told up front
        which streams will run.

        detect_selected_text() runs a synthetic Cmd+C plus three clipboard
        subprocesses (measured 353ms median, 1.4s worst case). Doing that
        before recording starts pushed the first word past the 1s preroll
        buffer, so it now overlaps the recording instead.
        """
        self.is_capturing = True
        self.start_time = time.time()
        set_tag(self.tag)
        self._notify("streams_planned", [f"{provider.name}/{mic}"
                                         for mic, provider in self.stream_plan(list(mics))])
        self._context_thread = threading.Thread(target=self._capture_context, daemon=True)
        self._context_thread.start()

        # Log session start
        if self.metrics:
            self.metrics.log(
                "session_start",
                session_id=str(self.id),
                context=asdict(self.context) if self.context else {},
                enabled_mics=self.config_snapshot.enabled_mics,
                providers=self.config_snapshot.enabled_providers,
            )

    def _capture_context(self) -> None:
        """Active app + any selected text. Runs while recording."""
        try:
            self.context = get_app_context()
            self.origin = capture_origin()
            self.selected_text = detect_selected_text()
            if getattr(self.config_snapshot, "edit_feedback_enabled", False):
                # Wake the destination app's accessibility tree while the user
                # is still talking, so its field is readable by the time the
                # edit watcher looks. Chromium/Electron apps keep an empty tree
                # until asked, and their fields otherwise read as nothing.
                from .feedback import wake_frontmost_app
                wake_frontmost_app()
        except Exception as e:
            print(f"[Session] Context capture failed: {e}")

    def _my_turn(self) -> None:
        """Wait for earlier dictations to finish typing, once, before this one types."""
        if self._turn_taken or self.output_order is None or self._ticket is None:
            return
        self._turn_taken = True
        self.output_order.wait(self._ticket)

    def _in_place(self) -> bool:
        """
        True when the words can go where the dictation started - bringing that
        window back first if the speaker moved on to read something while
        talking. False only when it can't be brought back (it was closed).
        """
        if self.origin is None:
            current = get_app_context()
            return bool(self.context and current and current.bundle_id == self.context.bundle_id)
        if at_origin(self.origin):
            return True
        if return_to(self.origin):
            name = self.context.app_name if self.context else "the starting window"
            print(f"[Output] Back to {name} to type")
            return True
        print("[Output] Couldn't bring back the window the dictation started in")
        return False

    def _await_context(self, timeout: float = 2.0) -> None:
        """Context is needed at finalize; by then it is almost always ready."""
        thread = self._context_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def _start_edit_watch(self, text: str) -> None:
        """Watch the destination field for user corrections to our output."""
        if not getattr(self.config_snapshot, "edit_feedback_enabled", False):
            return
        from .feedback import record_correction, watch_for_edits

        raw = " ".join(r.text for r in sorted(self.all_transcription_results, key=lambda r: r.chunk)
                       if r.text.strip())
        app = self.context.app_name if self.context else ""

        def on_result(outcome: str, after_s: float, original: str, corrected: str) -> None:
            if outcome == "edited":
                print(f"[Feedback] Edited within {after_s:.0f}s")
                print(f"[Feedback]   typed:     \"{original[:120]}\"")
                print(f"[Feedback]   corrected: \"{corrected[:120]}\"")
                # Durable corpus: session_id joins this to the saved audio, so
                # a confirmed transcription error becomes a training pair.
                record_correction({
                    "ts": time.time(),
                    "session_id": str(self.id),
                    "app": self.context.app_name if self.context else "",
                    "window": self.context.window_title if self.context else "",
                    "checked_after_s": after_s,
                    "raw_transcript": raw[:1000],
                    "typed": original[:1000],
                    "corrected": corrected[:1000],
                    # What produced the text the speaker fixed: a correction
                    # model's slip and a recognizer's are different lessons.
                    "output_method": self.output_method,
                    "correction_model": (self.llm_result.model
                                         if self.llm_result is not None
                                         and self.output_method in ("streamed", "clipboard")
                                         else ""),
                })
            if self.metrics:
                self.metrics.log(
                    "post_edit",
                    session_id=str(self.id),
                    outcome=outcome,
                    checked_after_s=after_s,
                    app=app,
                )

        watch_for_edits(text, on_result, app=app)

    def _output_destination(self) -> str:
        """Where this session's text ended up: the app and window it started in."""
        if self.context:
            title = self.context.window_title
            return f"{self.context.app_name}: {title}" if title else self.context.app_name
        return ""

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
        chunk_num = self._next_chunk_num()
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

        future = self._executor.submit(
            self._transcribe_chunk_with_consensus, chunk, chunk_num
        )
        with self._chunk_lock:
            self.pending_futures.append(future)

    def _next_chunk_num(self) -> int:
        """Hand out a unique chunk number. Safe against concurrent chunks."""
        with self._chunk_lock:
            self._chunk_counter += 1
            return self._chunk_counter

    def stream_plan(self, mics: List[str]) -> List[Tuple[str, Provider]]:
        """Which (mic, provider) pairs transcribe audio from these mics."""
        primary = next((m for m in self.config_snapshot.enabled_mics if m in mics),
                       mics[0] if mics else None)
        return [
            (mic, provider)
            for mic in mics
            for provider in self.providers.values()
            # Serialized providers run on the primary mic only: a second mic
            # just queues behind the first one's lock and lands on the
            # critical path, since the chunk waits for the slowest result.
            if not getattr(provider, "single_instance", False) or mic == primary
        ]

    def _transcribe_chunk_with_consensus(
        self, chunk: AudioChunk, chunk_num: Optional[int] = None, final: bool = False
    ) -> None:
        """
        Transcribe one chunk, checking consensus as results arrive.

        Runs every stream in stream_plan() in parallel. If early consensus is
        reached, cancels the rest. final marks the chunk recorded after release.
        """
        set_tag(self.tag)   # chunks transcribe on pool threads
        futures: Dict[Future, Tuple[str, str]] = {}
        heard = [mic for mic, audio in chunk.items() if len(audio) > 0]
        for mic_name, provider in self.stream_plan(heard):
            future = self._executor.submit(provider.transcribe, chunk[mic_name], mic_name)
            futures[future] = (mic_name, provider.name)

        if not futures:
            return

        results: List[TranscriptionResult] = []
        consensus: Optional[str] = None
        # The caller allocates when it already logged this chunk; the final
        # chunk comes straight from finalize() and needs its own number.
        if chunk_num is None:
            chunk_num = self._next_chunk_num()

        matching_count = 0
        # A chunk is only as fast as its slowest provider. Once the fastest has
        # answered, the rest get a proportional grace period and are then left
        # behind. At the default floor this is a hang guard rather than a
        # latency trim: cloud STT is slower than local parakeet on 85% of
        # chunks, so cutting early would just mean systematically discarding
        # the cloud transcript. See config.py for the measured trade.
        multiplier = max(0.0, float(self.config_snapshot.provider_deadline_multiplier))
        min_grace = max(0.0, self.config_snapshot.provider_deadline_min_ms / 1000.0)

        started = time.monotonic()
        pending = set(futures)
        deadline: Optional[float] = None
        abandoned: List[str] = []

        while pending:
            if deadline is None:
                timeout = _CHUNK_HARD_TIMEOUT - (time.monotonic() - started)
            else:
                timeout = deadline - time.monotonic()
            if timeout <= 0:
                break

            done, pending = wait(pending, timeout=timeout,
                                 return_when=FIRST_COMPLETED)
            if not done:
                break   # grace period expired, or nothing at all came back

            for future in done:
                try:
                    result = future.result()
                    result.chunk = chunk_num
                    results.append(result)
                    with self._chunk_lock:
                        self.all_transcription_results.append(result)

                    mic, provider = futures[future]
                    self._notify("stream_landed", f"{provider}/{mic}", final)

                    # Log each transcription result (full text for debugging)
                    print(f"[Chunk {chunk_num}] {result.provider}/{result.mic}: {result.latency_ms/1000:.2f}s")
                    print(f"    \"{result.text}\"")

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

                            print(f"[Chunk {chunk_num}] ✓ Consensus ({matching_count} agree):")
                            print(f"    \"{consensus}\"")
                            break

                except Exception as e:
                    mic, provider = futures[future]
                    print(f"[Chunk {chunk_num}] Provider error ({provider}/{mic}): {e}")
                    self._notify("stream_dropped", f"{provider}/{mic}")

            if consensus:
                break

            # Start the clock on the stragglers once something has landed - an
            # answer with words in it; one that heard nothing is no evidence the
            # rest are late. The grace never outlasts the hard limit.
            if deadline is None and any(r.text.strip() for r in results) and multiplier > 0 and pending:
                elapsed = time.monotonic() - started
                deadline = started + min(max(elapsed * (1.0 + multiplier), min_grace), _CHUNK_HARD_TIMEOUT)

        if pending:
            for future in pending:
                mic, provider = futures[future]
                if not consensus:
                    abandoned.append(f"{provider}/{mic}")
                    self._notify("stream_dropped", f"{provider}/{mic}")
                future.cancel()
            if abandoned:
                waited = (time.monotonic() - started) * 1000
                print(f"[Chunk {chunk_num}] Moved on after {waited:.0f}ms "
                      f"without {', '.join(abandoned)}")
                if self.metrics:
                    self.metrics.log(
                        "provider_deadline",
                        session_id=str(self.id),
                        chunk_num=chunk_num,
                        waited_ms=round(waited),
                        abandoned=abandoned,
                        kept=[f"{r.provider}/{r.mic}" for r in results],
                    )

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
            self.chunk_results.append((chunk_num, results, consensus))

        self._publish_partial_text()

    def _publish_partial_text(self) -> None:
        """
        Push the transcript so far to the HUD.

        Raw STT, not the corrected text — it is a progress signal, not a
        preview of what will be typed. Only reaches the screen during long
        multi-chunk dictations, which is exactly where there is otherwise no
        feedback at all for 40s or more.
        """
        if self.observer is None:
            return
        texts, _ = self._aggregate_results()     # _parts() takes the chunk lock itself
        self._notify("partial_text", " ".join(t.strip() for t in texts if t.strip()))

    def _notify(self, event: str, *args) -> None:
        """Tell the observer what happened. It is display only, so it never breaks a session."""
        if self.observer is None:
            return
        try:
            getattr(self.observer, event)(str(self.id), *args)
        except Exception as e:
            print(f"[Session] observer {event} failed: {e}")

    def finalize(self, final_chunk: AudioChunk) -> None:
        """
        Called on key release. Runs finalization in background thread.

        Args:
            final_chunk: The last chunk of audio
        """
        # The mic is free the moment the key is released; everything below this
        # point is post-processing.
        self.is_capturing = False
        if self.output_order is not None:
            self._ticket = self.output_order.take()
        threading.Thread(
            target=self._finalize_impl,
            args=(final_chunk,),
            daemon=True
        ).start()

    def _finalize_impl(self, final_chunk: AudioChunk) -> None:
        """
        Wait for chunks, aggregate, correct, output.

        This runs in a background thread. The steps below are ordered by how
        early they can bail out: a stray tap never reaches a provider, and a
        session with no text never reaches the correction model.
        """
        set_tag(self.tag)   # finalisation runs on its own thread
        try:
            self.finalize_start_time = time.time()
            self._collect_final_audio(final_chunk)
            print(f"[Timing] Key held: {self.finalize_start_time - self.start_time:.2f}s")

            # A stray tap produces a fraction of a second of room tone. STT
            # models hallucinate on that — "yeah", a stray CJK glyph — because
            # they always emit their best guess rather than nothing. Drop it
            # before it reaches a provider.
            #
            # Only the whole session is discarded, and only when nothing was
            # sent for transcription while recording. This used to test
            # chunk_results, which fills in only once a chunk *finishes*: a
            # release within a second of a 30s chunk being emitted found it
            # still in flight, saw a quiet tail, and threw away the 30 seconds
            # of speech along with it. Seen four times in the logs.
            final_has_speech = self._has_speech(final_chunk)
            if not self._any_chunk_sent() and not final_has_speech:
                print("[Session] No speech captured - ignoring")
                if self.metrics:
                    self.metrics.log("session_discarded",
                                     session_id=str(self.id), reason="no_speech")
                return
            if not final_has_speech:
                final_chunk = {}   # keep the speech already sent; skip the silent tail

            self._transcribe_remaining(final_chunk)

            chunk_texts, all_results = self._aggregate_results()
            if not chunk_texts:
                print("No transcription results")
                return

            combined_text = " ".join(chunk_texts)
            print(f"[Session] {len(self.chunk_results)} chunks, {len(all_results)} transcriptions")

            # Selection capture is a synthetic Cmd+C (up to 1.4s) and can outlast a
            # fast transcription; deciding before it lands types the command.
            self._await_context()
            if self.selected_text:
                self._run_edit_mode(combined_text)
                return

            self._correct_and_output(all_results, combined_text)

        except Exception as e:
            print(f"Session finalize error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self._teardown()

    def _any_chunk_sent(self) -> bool:
        """True once any chunk has gone to the recognizers, finished or not."""
        with self._chunk_lock:
            return bool(self.chunk_results or self.pending_futures)

    def _collect_final_audio(self, final_chunk: AudioChunk) -> None:
        """Keep the last chunk's audio for training data, and report its length."""
        if not final_chunk:
            return
        for mic, audio in final_chunk.items():
            if len(audio) == 0:
                continue
            seconds = len(audio) / self.config_snapshot.sample_rate
            print(f"[Audio] {mic}: {seconds:.2f}s of audio")
            self.all_audio.setdefault(mic, []).append(audio.copy())

    def _transcribe_remaining(self, final_chunk: AudioChunk) -> None:
        """Transcribe the final chunk and wait for any still in flight."""
        started = time.time()
        if final_chunk and any(len(a) > 0 for a in final_chunk.values()):
            self._transcribe_chunk_with_consensus(final_chunk, final=True)

        # Copy under the lock: waiting while holding it would deadlock the
        # chunk callbacks trying to append their own results.
        with self._chunk_lock:
            pending = list(self.pending_futures)
        if pending:
            wait(pending, timeout=_CHUNK_HARD_TIMEOUT, return_when=ALL_COMPLETED)

        print(f"[Timing] Transcription: {time.time() - started:.2f}s")
        self._notify("transcription_done")

    def _run_edit_mode(self, command: str) -> None:
        """Selection + speech: the transcript is an instruction, not content."""
        print(f"[Session] Text edit mode: \"{command[:50]}...\"")
        from .correct import edit_text_with_llm
        self._output(edit_text_with_llm(self.selected_text, command, self.config_snapshot))

    # -- correction --------------------------------------------------------

    def _correct_and_output(self, all_results: List[TranscriptionResult],
                            combined_text: str) -> None:
        """
        Run LLM correction and put the result on screen.

        Streams into the focused field when it is still the one recording
        started in; otherwise the text would land somewhere the user is no
        longer looking, so it goes to the clipboard instead.
        """
        self._await_context()
        same_window = self._in_place()

        corrected = (self._stream_correction(all_results) if same_window
                     else self._clipboard_correction(all_results))
        if corrected is None:
            # The model judged that the speaker called the whole dictation off.
            print("[Session] Speaker called the dictation off - typing nothing")
            self._notify("dictation_discarded")
            if self.metrics:
                self.metrics.log("session_discarded", session_id=str(self.id),
                                 reason="called_off")
            return

        if same_window:
            if not corrected:
                print("[Session] LLM correction failed - using raw transcript")
                self._output(combined_text)
                return
            self._final_text = corrected
            self._log_output_stats(corrected,
                                   "judge" if self.output_method == "judged" else "openrouter")
            if not self._partial_output:
                self._start_edit_watch(corrected)
        else:
            corrected = corrected or combined_text
            self._final_text = corrected
            self._my_turn()
            copy_to_clipboard(corrected)
            notify("Window changed - copied to clipboard")

        # History holds the raw transcription, never the LLM's output: feeding
        # corrected text back in makes the model echo its own phrasing.
        self.history.add(combined_text, destination=self._output_destination())

    def _continues_previous(self) -> bool:
        """
        True when this dictation is appending to the previous one.

        Consecutive recordings run their words together without a separator,
        but a space is only wanted when there is something to separate from.
        Measured over 675 recordings started within 8s of the previous, 673
        went to the same app, so recency plus destination is a reliable test
        and leaves a single dictation with clean, untrailed text.
        """
        if not self.config_snapshot.space_between_dictations:
            return False
        previous = self.history.last()
        if previous is None:
            return False
        when, _, destination = previous
        if time.time() - when > _CONTINUATION_SECONDS:
            return False
        return bool(destination) and destination == self._output_destination()

    def _correction_kwargs(self) -> dict:
        """Arguments shared by both correction paths."""
        return {
            "on_metadata": self._log_llm_metadata,
            "custom_instructions": self.config_snapshot.custom_instructions,
            "on_generation_metadata": self._log_generation_metadata,
        }

    def _stream_correction(self, all_results: List[TranscriptionResult]) -> Optional[str]:
        """
        Type tokens as they arrive, unless the judge says they need no correcting.

        The judge and the correction model are asked at the same moment and the
        first usable answer wins: a "clean" verdict lands in ~250ms against the
        correction model's ~750ms first token, so it types the transcript and
        the correction never reaches the screen. Everything else costs nothing,
        because the correction was already in flight.

        Returns the typed text, "" if the correction failed, or None when the
        model replied that the speaker called the whole dictation off.
        """
        from .correct import CorrectionInterrupted, correct_with_llm

        self.output_method = "streamed"
        continuing = self._continues_previous()
        separator_pending = [continuing]
        marker = NothingMarker()
        dictation = DictationFilter(continuing=continuing)
        typed: List[str] = []

        # Whoever types first owns the output; the loser's tokens are dropped.
        winner: List[str] = []
        claim_lock = threading.Lock()

        def claim(who: str) -> bool:
            with claim_lock:
                if not winner:
                    winner.append(who)
                return winner[0] == who

        def claimed() -> bool:
            with claim_lock:
                return bool(winner)

        # Checked between tokens, at most every 100ms: if the speaker moves to
        # another window mid-stream, the rest must not be typed into it. Once
        # gone it stays gone, so the text can't resume with a hole in it.
        held: List[str] = []
        last_check = [0.0]

        def in_place() -> bool:
            if held:
                return False
            if self.origin is None:
                return True
            now = time.monotonic()
            if now - last_check[0] < 0.1:
                return True
            last_check[0] = now
            return at_origin(self.origin)

        def emit(text: str) -> None:
            text = marker.feed(text)
            if not text:
                return
            if separator_pending[0]:
                # Only once there is something to type, so a dictation that
                # comes to nothing can't leave a stray space behind.
                text = " " + text
                separator_pending[0] = False
            text = dictation.feed(text)
            if not text:
                return
            self._my_turn()
            if not in_place():
                held.append(text)
                return
            typed.append(text)
            with self.output_lock:
                type_text(text)
            self._notify("token_typed")   # outside the lock: display only

        def on_delta(token: str) -> None:
            if claim("correction"):
                emit(token)

        interrupted: List[CorrectionInterrupted] = []

        def correct() -> None:
            try:
                correct_with_llm(all_results, self.context, self.config_snapshot,
                                 on_delta=on_delta, **self._correction_kwargs())
            except CorrectionInterrupted as e:
                interrupted.append(e)
            except Exception as e:
                # Nothing was typed by it, so the raw transcript goes in instead.
                print(f"[LLM] Correction failed: {e}")

        verdict = self._start_judge(all_results)
        correction = threading.Thread(target=correct, daemon=True)
        correction.start()

        if self._judge_wins(verdict, claim, claimed):
            typed_text = verdict["v"].text
            emit(typed_text)
            return "".join(typed)

        correction.join()
        emit(marker.flush())
        if marker.called_off:
            return None
        if interrupted:
            # Part of it is typed and the rest can't be joined on cleanly.
            whole = interrupted[0].complete or " ".join(r.text for r in all_results if r.text.strip())
            self._hand_over(whole, "The correction was cut off - the full text is on the clipboard")
            return "".join(typed)
        if held:
            self._hand_over("".join(held).lstrip(), "Window changed while typing - the rest is on the clipboard")
            return "".join(typed + held)
        return "".join(typed)

    def _hand_over(self, text: str, message: str) -> None:
        """Only part of a dictation reached the field; put the rest where the user can paste it."""
        self._partial_output = True
        copy_to_clipboard(text)
        notify(message)
        print(f"[Session] {message}")

    def _start_judge(self, all_results: List[TranscriptionResult]) -> dict:
        """
        Ask the judge in the background. The dict fills in with "v" when it answers.

        Single-chunk dictations only. The judge is handed transcripts as
        alternative readings of the same audio and answers with the best one,
        but across chunks they are *consecutive* pieces of one dictation - so
        on a multi-chunk session it would type one chunk and drop the rest.
        A long dictation is also nearly always worth correcting: over the
        logged sessions, every multi-chunk one scored far too much filler to
        skip anyway.
        """
        from .judge import judge_transcripts

        box: dict = {"done": threading.Event()}
        with self._chunk_lock:
            chunks = len(self.chunk_results)
        if chunks > 1:
            box["done"].set()
            return box

        def run() -> None:
            try:
                box["v"] = judge_transcripts(all_results, self.config_snapshot, self.context)
            except Exception as e:
                print(f"[Judge] failed ({e}); using the correction model")
            finally:
                box["done"].set()

        threading.Thread(target=run, daemon=True).start()
        return box

    def _judge_wins(self, box: dict, claim, claimed) -> bool:
        """
        Wait for the judge only while the correction has typed nothing.

        Returns True when it came back clean in time and this call owns the
        output. A slow or unavailable judge simply loses.
        """
        from .judge import _timeout_seconds

        deadline = time.monotonic() + _timeout_seconds(self.config_snapshot)
        while not box["done"].wait(0.02):
            if claimed() or time.monotonic() > deadline:
                break
        verdict = box.get("v")
        if self.metrics and verdict is not None:
            self.metrics.log("judge", session_id=str(self.id), clean=verdict.clean,
                             scores={k: round(v, 3) for k, v in verdict.scores.items()},
                             latency_ms=verdict.latency_ms, model=verdict.model)
        if verdict is not None:
            # Every verdict, not just the ones that win: a judge that only
            # speaks up when it acts looks like a judge that never ran.
            scores = " ".join(f"{k} {v:.2f}" for k, v in verdict.scores.items())
            print(f"[Judge] {'no correction needed' if verdict.clean else 'correcting'} "
                  f"({scores}) | {verdict.latency_ms}ms")
        if verdict is None or not verdict.clean:
            return False
        if not claim("judge"):
            return False   # the correction model was already typing
        print(f"[Judge] typing the transcript ({verdict.latency_ms}ms)")
        self.output_method = "judged"
        self._notify("correction_skipped")
        return True

    def _clipboard_correction(self, all_results: List[TranscriptionResult]) -> Optional[str]:
        """
        Correct without streaming, for when the target window has changed.

        Same return contract as _stream_correction.
        """
        from .correct import correct_with_llm

        self.output_method = "clipboard"
        corrected = correct_with_llm(
            all_results, self.context, self.config_snapshot, **self._correction_kwargs()
        )
        if not corrected:
            print("[Session] LLM correction failed - using raw transcript")
            return ""
        if is_nothing(corrected):
            return None
        # Clipboard output is pasted deliberately, so it never needs a
        # separator; just flatten any invented line breaks.
        return DictationFilter().feed(corrected)

    def _log_llm_metadata(self, result: LLMCorrectionResult) -> None:
        self.llm_result = result
        if not self.metrics:
            return
        self.metrics.log(
            "llm_correction",
            session_id=str(self.id),
            provider=result.provider,
            model=result.model,
            generation_id=result.generation_id,
            backend_provider=result.backend_provider,
            resolved_model=result.resolved_model,
            provider_order=result.provider_order,
            allow_fallbacks=result.allow_fallbacks,
            reasoning_effort=result.reasoning_effort,
            usage=result.usage,
            input_tokens_est=result.input_tokens_est,
            latency_ms=result.latency_ms,
        )

    def _log_generation_metadata(self, generation_id: str, usage: dict) -> None:
        """Cost details arrive after the session finishes, so they log separately."""
        if self.metrics:
            self.metrics.log("llm_generation", session_id=str(self.id),
                             generation_id=generation_id, **usage)

    # -- teardown ----------------------------------------------------------

    def _teardown(self) -> None:
        """
        Release the session, whatever happened above.

        Bookkeeping is wrapped separately because anything throwing here used
        to skip the release below, leaving the session held forever so every
        later key press was rejected as busy until the app was restarted.
        """
        try:
            if self.metrics:
                self.metrics.log(
                    "session_complete",
                    session_id=str(self.id),
                    total_duration_ms=(time.time() - self.start_time) * 1000,
                    chunks=len(self.chunk_results),
                    final_text=self._final_text[:500] if self._final_text else "",
                    words=len(self._final_text.split()) if self._final_text else 0,
                    app=self.context.app_name if self.context else "",
                )
            if (self.training_writer
                    and self.config_snapshot.training_enabled
                    and self._final_text
                    and self.all_audio):
                self._save_training_data()
        except Exception as e:
            print(f"[Session] Post-session bookkeeping failed: {e}")

        if self.output_order is not None and self._ticket is not None:
            self.output_order.done(self._ticket)
        self.is_capturing = False
        self._executor.shutdown(wait=False)
        try:
            self.on_complete(self)
        except Exception as e:
            print(f"[Session] on_complete handler failed: {e}")

    def _has_speech(self, chunk: AudioChunk) -> bool:
        """True if a chunk holds enough audio, loud enough, to be worth sending.

        Two gates, because either alone misfires: a long recording of silence
        passes on duration, and a short burst of noise passes on level.
        """
        if not chunk:
            return False
        for audio in chunk.values():
            if len(audio) == 0:
                continue
            duration = len(audio) / self.config_snapshot.sample_rate
            if duration < MIN_SESSION_SECONDS:
                continue
            rms = float(np.sqrt(np.mean(np.square(audio))))
            if rms <= 0:
                continue
            if 20 * np.log10(rms) >= MIN_SESSION_DB:
                return True
        return False

    def _parts(self) -> List[ChunkResult]:
        """The chunks in the order they were spoken, whatever order they finished in."""
        with self._chunk_lock:
            return sorted(self.chunk_results, key=lambda part: part[0])

    def _aggregate_results(self) -> Tuple[List[str], List[TranscriptionResult]]:
        """
        Aggregate chunk results.

        Per-chunk: use consensus if found, else longest result (most content).
        Returns: (list of chunk texts, flat list of all results)
        """
        chunk_texts: List[str] = []
        all_results: List[TranscriptionResult] = []

        for _, results, consensus in self._parts():
            all_results.extend(results)

            if consensus:
                chunk_texts.append(consensus)
            else:
                # No consensus - use longest non-empty result
                non_empty = [r for r in results if r.text.strip()]
                if non_empty:
                    best = max(non_empty, key=lambda r: len(r.text.split()))
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

        self._my_turn()
        with self.output_lock:
            if not self._in_place():
                copy_to_clipboard(text)
                self.output_method = "clipboard"  # Track actual output method
                print("[Timing] Output: clipboard (window changed)")
                notify("Window changed - copied to clipboard")
                self.history.add(text, destination=self._output_destination())
                return

            type_text(text)
            self.output_method = "typed"  # Track actual output method
            self._log_output_stats(text, correction_provider)
            self._start_edit_watch(text)

        # Add to history after successful output
        self.history.add(text, destination=self._output_destination())

    def _log_output_stats(self, text: str, correction_provider: str) -> None:
        """Print end-to-end timing/WPM. Used by both streamed and typed paths."""
        done = time.time()
        word_count = len(text.split())
        total_time = done - self.start_time            # Key press to output done
        processing_time = done - self.finalize_start_time  # Key release to output done

        total_wpm = (word_count / total_time) * 60 if total_time > 0 else 0
        processing_wpm = (word_count / processing_time) * 60 if processing_time > 0 else 0

        print(
            f"[Output] {correction_provider} via {self.output_method} | "
            f"{total_time:.2f}s total | {processing_time:.2f}s after release | {word_count} words"
        )
        print(f"[Output] \"{text}\"")
        print(
            f"[WPM] Total: {total_wpm:.0f} wpm (from key press) | "
            f"Processing: {processing_wpm:.0f} wpm (from key release)"
        )

    def _save_training_data(self) -> None:
        """Collect and save all session data for training."""
        # Concatenate all audio chunks per mic (copy under lock to avoid race)
        with self._chunk_lock:
            audio_data: Dict[str, np.ndarray] = {}
            for mic_name, chunks in self.all_audio.items():
                if chunks:
                    audio_data[mic_name] = np.concatenate(chunks)
            # Also copy transcription results
            transcription_results = sorted(self.all_transcription_results, key=lambda r: r.chunk)

        if not audio_data:
            return

        # Build consensus info
        consensus_info: Optional[Dict] = None
        for _, results, consensus in self._parts():
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
                "generation_id": self.llm_result.generation_id,
                "backend_provider": self.llm_result.backend_provider,
                "resolved_model": self.llm_result.resolved_model,
                "provider_order": self.llm_result.provider_order,
                "allow_fallbacks": self.llm_result.allow_fallbacks,
                "reasoning_effort": self.llm_result.reasoning_effort,
                "usage": self.llm_result.usage,
                "input_text": " | ".join(input_texts),
                "output_text": self._final_text,
                "latency_ms": self.llm_result.latency_ms,
            }

        config_info = {
            "enabled_mics": self.config_snapshot.enabled_mics,
            "enabled_providers": self.config_snapshot.enabled_providers,
            "consensus_threshold": self.config_snapshot.consensus_threshold,
            "openrouter_stt_models": self.config_snapshot.openrouter_stt_models,
            "openrouter_correction_model": self.config_snapshot.openrouter_correction_model,
            "openrouter_correction_provider_order": self.config_snapshot.openrouter_correction_provider_order,
            "openrouter_correction_allow_fallbacks": self.config_snapshot.openrouter_correction_allow_fallbacks,
            "openrouter_correction_reasoning_effort": self.config_snapshot.openrouter_correction_reasoning_effort,
            "custom_instructions_enabled": bool(self.config_snapshot.custom_instructions.strip()),
            "system_prompt_customized": bool(self.config_snapshot.system_prompt.strip()),
        }

        # Create metadata
        metadata = TrainingMetadata(
            session_id=str(self.id),
            timestamp=datetime.now().isoformat(),
            duration_ms=(time.time() - self.start_time) * 1000,
            sample_rate=self.config_snapshot.sample_rate,
            app_context=asdict(self.context) if self.context else None,
            config_snapshot=config_info,
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
        self._entries: List[Tuple[float, str, str]] = []  # (ts, text, destination)
        self._lock = threading.Lock()

    def last(self) -> Optional[Tuple[float, str, str]]:
        """Most recent (timestamp, text, destination), or None."""
        with self._lock:
            return self._entries[-1] if self._entries else None

    def add(self, text: str, destination: str = "") -> None:
        """Add a transcription, tagged with where it was typed (app/window)."""
        if not text or not text.strip():
            return

        with self._lock:
            self._entries.append((time.time(), text.strip(), destination.strip()))
            # Prune old entries
            self._prune()

    def _prune(self) -> None:
        """Remove entries older than max_age_seconds and enforce max_entries."""
        cutoff = time.time() - self.max_age_seconds
        self._entries = [e for e in self._entries if e[0] > cutoff]
        # Also enforce max_entries to prevent unbounded growth
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]


class OutputOrder:
    """
    Dictations reach the screen in the order they were released.

    A new recording can start while the last one is still being corrected, and
    a short second dictation can be ready first - typing it then would splice
    it into the middle of the first one's streamed text. Each session takes a
    ticket on release and waits for its turn just before its first keystroke,
    so its correction still runs in parallel.
    """

    def __init__(self, patience: float = 15.0):
        self._turn = threading.Condition()
        self._issued = 0
        self._serving = 0
        self._finished: set = set()
        self._patience = patience     # never wait forever on a session that hung

    def take(self) -> int:
        with self._turn:
            ticket = self._issued
            self._issued += 1
            return ticket

    def wait(self, ticket: int) -> None:
        deadline = time.monotonic() + self._patience
        with self._turn:
            while self._serving < ticket:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    print("[Session] The previous dictation is still typing; going ahead")
                    return
                self._turn.wait(remaining)

    def done(self, ticket: int) -> None:
        with self._turn:
            self._finished.add(ticket)
            while self._serving in self._finished:
                self._finished.discard(self._serving)
                self._serving += 1
            self._turn.notify_all()


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
        observer: Optional[SessionObserver] = None,
    ):
        self.config_snapshot_fn = config_snapshot_fn
        self.providers = providers
        self.metrics = metrics
        self.training_writer = training_writer
        self.observer = observer

        self.active_session: Optional[Session] = None
        self._lock = threading.Lock()
        self._output_lock = threading.Lock()
        self._output_order = OutputOrder()
        self.history = TranscriptionHistory()

    def start_session(self) -> Optional[Session]:
        """
        Create and start a new session.

        Returns None if a session is already active (plays error sound).
        """
        with self._lock:
            busy = bool(self.active_session and self.active_session.is_capturing)
            if not busy:
                session = Session(
                    id=uuid4(),
                    config_snapshot=self.config_snapshot_fn(),
                    providers=self.providers,
                    output_lock=self._output_lock,
                    output_order=self._output_order,
                    on_complete=self._on_session_complete,
                    history=self.history,
                    metrics=self.metrics,
                    training_writer=self.training_writer,
                    observer=self.observer,
                )
                self.active_session = session

        if busy:
            # Outside the lock: never hold it across an external process
            play_busy_sound()
            return None

        return session

    def _on_session_complete(self, session: Session) -> None:
        """Called when a session finishes."""
        with self._lock:
            if self.active_session == session:
                self.active_session = None

    def is_busy(self) -> bool:
        """True when a session still holds the microphone."""
        with self._lock:
            return self.active_session is not None and self.active_session.is_capturing
