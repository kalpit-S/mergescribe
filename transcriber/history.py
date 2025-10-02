import time
from typing import Iterable, Tuple

from config_manager import ConfigManager

HistoryEntry = Tuple[float, str]


def build_context_from_history(transcription_history: Iterable[HistoryEntry]) -> str:
    """Build a compact context string from recent transcription history."""
    cfg = ConfigManager()
    history = list(transcription_history)
    if not bool(cfg.get_value("ENABLE_CONTEXT")) or not history:
        return ""

    current_time = time.time()
    recent_entries = []

    for timestamp, text in reversed(history):
        if current_time - timestamp <= (cfg.get_value("CONTEXT_MAX_AGE_SECONDS") or 300):
            recent_entries.append(text)
        if len(recent_entries) >= (cfg.get_value("CONTEXT_HISTORY_COUNT") or 3):
            break

    recent_entries.reverse()

    if not recent_entries:
        return ""

    return " | ".join(recent_entries)
