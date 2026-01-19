"""
Thread-safe metrics logging with batched writes.

Usage:
    metrics = MetricsWriter(config.metrics_file)
    metrics.log("transcription", provider="parakeet", latency_ms=234)
"""

import json
import time
import threading
from queue import Queue, Empty
from pathlib import Path
from typing import Any


class MetricsWriter:
    """
    Thread-safe metrics writer with atomic appends.
    Uses a queue to batch writes from multiple threads.
    """

    def __init__(self, metrics_file: Path):
        self.metrics_file = metrics_file
        self._queue: Queue[dict] = Queue()
        self._shutdown = threading.Event()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def log(self, event: str, **kwargs: Any) -> None:
        """
        Queue a metric for writing. Non-blocking.

        Args:
            event: Event name (e.g., "transcription", "consensus", "llm_correction")
            **kwargs: Additional fields to log
        """
        entry = {
            "ts": time.time(),
            "event": event,
            **kwargs
        }
        self._queue.put(entry)

    def _writer_loop(self) -> None:
        """Background thread that batches and writes metrics."""
        while not self._shutdown.is_set():
            try:
                # Wait for first entry
                entries = [self._queue.get(timeout=1.0)]

                # Drain queue (batch writes)
                while True:
                    try:
                        entries.append(self._queue.get_nowait())
                    except Empty:
                        break

                # Write batch
                self._write_entries(entries)

            except Empty:
                # Timeout, check shutdown flag and continue
                continue
            except Exception as e:
                print(f"MetricsWriter error: {e}")

    def _write_entries(self, entries: list[dict]) -> None:
        """Write entries to file."""
        try:
            # Ensure parent directory exists
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.metrics_file, "a") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"Failed to write metrics: {e}")

    def flush(self) -> None:
        """Flush any pending metrics to disk."""
        # Drain queue
        entries = []
        while True:
            try:
                entries.append(self._queue.get_nowait())
            except Empty:
                break

        if entries:
            self._write_entries(entries)

    def shutdown(self) -> None:
        """Shutdown the writer thread gracefully."""
        self._shutdown.set()
        self.flush()
        self._writer_thread.join(timeout=2.0)


# Global instance (initialized lazily)
_metrics: MetricsWriter | None = None


def get_metrics(metrics_file: Path) -> MetricsWriter:
    """Get or create the global metrics writer."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsWriter(metrics_file)
    return _metrics


# Typed helper functions for consistent event logging

def log_session_start(
    metrics: MetricsWriter,
    session_id: str,
    context: dict,
    enabled_mics: list,
    providers: list,
) -> None:
    """Log session_start event."""
    metrics.log(
        "session_start",
        session_id=session_id,
        context=context,
        enabled_mics=enabled_mics,
        providers=providers,
    )


def log_chunk_received(
    metrics: MetricsWriter,
    session_id: str,
    chunk_num: int,
    mic_names: list,
    audio_duration_ms: float,
) -> None:
    """Log chunk_received event."""
    metrics.log(
        "chunk_received",
        session_id=session_id,
        chunk_num=chunk_num,
        mic_names=mic_names,
        audio_duration_ms=audio_duration_ms,
    )


def log_transcription(
    metrics: MetricsWriter,
    session_id: str,
    chunk_num: int,
    provider: str,
    mic: str,
    latency_ms: float,
    text: str,
    confidence: float | None = None,
) -> None:
    """Log transcription event."""
    metrics.log(
        "transcription",
        session_id=session_id,
        chunk_num=chunk_num,
        provider=provider,
        mic=mic,
        latency_ms=latency_ms,
        text=text[:200],  # Truncate for metrics
        confidence=confidence,
    )


def log_consensus(
    metrics: MetricsWriter,
    session_id: str,
    chunk_num: int,
    reached: bool,
    text: str | None,
    matching_count: int,
) -> None:
    """Log consensus event."""
    metrics.log(
        "consensus",
        session_id=session_id,
        chunk_num=chunk_num,
        reached=reached,
        text=text[:200] if text else None,
        matching_count=matching_count,
    )


def log_llm_correction(
    metrics: MetricsWriter,
    session_id: str,
    provider: str,
    model: str,
    input_tokens_est: int,
    latency_ms: float,
) -> None:
    """Log llm_correction event."""
    metrics.log(
        "llm_correction",
        session_id=session_id,
        provider=provider,
        model=model,
        input_tokens_est=input_tokens_est,
        latency_ms=latency_ms,
    )


def log_cache_event(
    metrics: MetricsWriter,
    session_id: str,
    hit: bool,
    raw_text_hash: str,
) -> None:
    """Log cache_hit or cache_miss event."""
    event = "cache_hit" if hit else "cache_miss"
    metrics.log(event, session_id=session_id, raw_text_hash=raw_text_hash)


def log_output(
    metrics: MetricsWriter,
    session_id: str,
    method: str,  # "typed" | "clipboard" | "streamed"
    latency_ms: float,
) -> None:
    """Log output event."""
    metrics.log(
        "output",
        session_id=session_id,
        method=method,
        latency_ms=latency_ms,
    )


def log_session_complete(
    metrics: MetricsWriter,
    session_id: str,
    total_duration_ms: float,
    chunks: int,
    final_text: str,
) -> None:
    """Log session_complete event."""
    metrics.log(
        "session_complete",
        session_id=session_id,
        total_duration_ms=total_duration_ms,
        chunks=chunks,
        final_text=final_text[:500],  # Truncate for metrics
    )
