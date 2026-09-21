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
