"""
Training data collection for local model fine-tuning.

Saves audio files and metadata in a structured format.
All data stays on the user's machine - this is opt-in local storage only.

Non-blocking: all I/O happens in a background thread via a bounded queue.
"""

import json
import os
import tempfile
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty, Full
from typing import Dict, Optional
from uuid import UUID

import numpy as np
import soundfile as sf

from .types import TrainingMetadata


# Queue limits to prevent memory blowup
MAX_QUEUE_SIZE = 10
MIN_AUDIO_DURATION_MS = 500  # Don't save very short recordings


class TrainingDataWriter:
    """
    Async writer for training data (audio + metadata).

    Saves to: {training_dir}/{date}/{session_id}/
      - audio_{mic_name}.wav (16kHz, mono, PCM_16)
      - metadata.json

    Usage:
        writer = TrainingDataWriter(training_dir, sample_rate=16000)
        writer.save_session(session_id, audio_chunks, metadata)
        # ...
        writer.shutdown()
    """

    def __init__(self, training_dir: Path, sample_rate: int = 16000):
        self.training_dir = Path(training_dir)
        self.sample_rate = sample_rate

        self._queue: Queue[tuple] = Queue(maxsize=MAX_QUEUE_SIZE)
        self._shutdown = threading.Event()
        self._dropped_count = 0
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def save_session(
        self,
        session_id: UUID,
        audio_chunks: Dict[str, np.ndarray],  # {mic_name: audio_array}
        metadata: TrainingMetadata,
    ) -> bool:
        """
        Queue session data for async saving.

        Args:
            session_id: Unique session identifier
            audio_chunks: Dict mapping mic names to concatenated audio arrays
            metadata: Complete session metadata

        Returns:
            True if queued, False if dropped (queue full or invalid)
        """
        # Validate audio duration
        if not audio_chunks:
            return False

        total_samples = max(len(a) for a in audio_chunks.values()) if audio_chunks else 0
        duration_ms = (total_samples / self.sample_rate) * 1000

        if duration_ms < MIN_AUDIO_DURATION_MS:
            return False

        try:
            self._queue.put_nowait((session_id, audio_chunks, metadata))
            return True
        except Full:
            self._dropped_count += 1
            print(f"[Training] Queue full, dropped session {session_id} (total dropped: {self._dropped_count})")
            return False

    def _writer_loop(self) -> None:
        """Background thread for saving training data."""
        while not self._shutdown.is_set():
            try:
                item = self._queue.get(timeout=1.0)
                session_id, audio_chunks, metadata = item
                self._save_session_sync(session_id, audio_chunks, metadata)
            except Empty:
                continue
            except Exception as e:
                print(f"[Training] Writer error: {e}")

    def _save_session_sync(
        self,
        session_id: UUID,
        audio_chunks: Dict[str, np.ndarray],
        metadata: TrainingMetadata,
    ) -> None:
        """Synchronously save session data to disk."""
        try:
            # Create directory: training/{date}/{session_id}/
            date_str = datetime.now().strftime("%Y-%m-%d")
            session_dir = self.training_dir / date_str / str(session_id)
            session_dir.mkdir(parents=True, exist_ok=True)

            # Set restrictive permissions on directory (user only)
            try:
                os.chmod(session_dir, 0o700)
            except OSError:
                pass  # Best effort

            # Save audio files (one per mic)
            for mic_name, audio in audio_chunks.items():
                if len(audio) == 0:
                    continue

                # Sanitize mic name for filename
                safe_name = self._sanitize_filename(mic_name)
                wav_path = session_dir / f"audio_{safe_name}.wav"

                # Convert float32 to int16 with proper scaling and clipping
                audio_clipped = np.clip(audio, -1.0, 1.0)
                audio_int16 = (audio_clipped * 32767).astype(np.int16)

                # Atomic write: write to temp, then replace (works on Windows too)
                fd, temp_path = tempfile.mkstemp(dir=session_dir, suffix=".wav.tmp")
                try:
                    os.close(fd)
                    sf.write(temp_path, audio_int16, self.sample_rate,
                             format="WAV", subtype="PCM_16")
                    os.replace(temp_path, wav_path)  # Atomic on both POSIX and Windows
                    # Set restrictive permissions on file
                    try:
                        os.chmod(wav_path, 0o600)
                    except OSError:
                        pass
                except Exception:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    raise

            # Save metadata.json (atomic write)
            metadata_path = session_dir / "metadata.json"
            fd, temp_path = tempfile.mkstemp(dir=session_dir, suffix=".json.tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(asdict(metadata), f, indent=2, default=str)
                os.replace(temp_path, metadata_path)  # Atomic on both POSIX and Windows
                try:
                    os.chmod(metadata_path, 0o600)
                except OSError:
                    pass
            except Exception:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

            print(f"[Training] Saved session {session_id} ({len(audio_chunks)} mics)")

        except Exception as e:
            print(f"[Training] Failed to save session {session_id}: {e}")

    def _sanitize_filename(self, name: str) -> str:
        """Convert mic name to safe filename."""
        safe = name.lower()
        for char in " /\\:*?\"<>|()":
            safe = safe.replace(char, "_")
        while "__" in safe:
            safe = safe.replace("__", "_")
        safe = safe.strip("_")
        # Fallback if name becomes empty or too short
        if not safe or len(safe) < 2:
            safe = f"mic_{hash(name) & 0xFFFF:04x}"
        return safe

    @property
    def dropped_count(self) -> int:
        """Number of sessions dropped due to full queue."""
        return self._dropped_count

    def flush(self) -> None:
        """Process remaining items in queue."""
        while True:
            try:
                item = self._queue.get_nowait()
                session_id, audio_chunks, metadata = item
                self._save_session_sync(session_id, audio_chunks, metadata)
            except Empty:
                break

    def shutdown(self) -> None:
        """Gracefully shutdown the writer thread."""
        self._shutdown.set()
        self.flush()
        self._writer_thread.join(timeout=5.0)
        if self._dropped_count > 0:
            print(f"[Training] Total sessions dropped: {self._dropped_count}")


# Global instance (lazy-loaded)
_training_writer: Optional[TrainingDataWriter] = None


def get_training_writer(training_dir: Path, sample_rate: int = 16000) -> TrainingDataWriter:
    """Get or create the global training data writer."""
    global _training_writer
    if _training_writer is None:
        _training_writer = TrainingDataWriter(training_dir, sample_rate)
    return _training_writer
