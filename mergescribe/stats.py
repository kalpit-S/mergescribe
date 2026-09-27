"""
Usage stats for the Settings window, read from the metrics log.

A dictation counts once it completes: how many words it produced, how long
the key was held, and how long after release the text was done.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

METRICS_PATH = Path.home() / ".mergescribe" / "metrics.jsonl"
TRAINING_DIR = Path.home() / ".mergescribe" / "training"
_OLD_TEXT_LIMIT = 500   # session_complete kept this much text before it logged a word count
_MIN_WORDS = 5          # shorter dictations make a words-per-minute figure meaningless
_DAYS = 14


@dataclass
class Usage:
    dictations: Dict[str, int] = field(default_factory=dict)   # "today", "week", "all"
    words: Dict[str, int] = field(default_factory=dict)
    speaking_wpm: Optional[float] = None     # while the key is held, median over the past week
    overall_wpm: Optional[float] = None      # key press to text typed, median over the past week
    after_release: Optional[Tuple[float, float]] = None   # seconds, (median, slowest tenth)
    daily_words: List[int] = field(default_factory=list)  # the last _DAYS days, oldest first
    apps: List[Tuple[str, float]] = field(default_factory=list)   # share of the past week


def _full_word_count(session_id: str, ts: float, training: Path) -> Optional[int]:
    """The word count from the training capture, for records that kept only part of the text."""
    for day in (datetime.fromtimestamp(ts), datetime.fromtimestamp(ts) - timedelta(days=1)):
        path = training / day.strftime("%Y-%m-%d") / session_id / "metadata.json"
        try:
            return len((json.loads(path.read_text()).get("final_output") or "").split())
        except (OSError, ValueError):
            continue
    return None


def usage(path: Path = METRICS_PATH, training: Path = TRAINING_DIR,
          now: Optional[float] = None) -> Usage:
    now = time.time() if now is None else now
    today = datetime.fromtimestamp(now).date()
    week_ago = now - 7 * 86400
    started: Dict[str, float] = {}
    stopped: Dict[str, float] = {}
    completed: List[dict] = []
    try:
        with open(path, errors="ignore") as lines:
            for line in lines:
                # Cheap filter first: the log is mostly per-chunk events.
                if '"recording_st' not in line and '"session_complete"' not in line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                event, session = record.get("event"), record.get("session_id")
                if event == "recording_started":
                    started[session] = record["ts"]
                elif event == "recording_stopped":
                    stopped[session] = record["ts"]
                elif event == "session_complete":
                    completed.append(record)
    except OSError:
        return Usage()

    result = Usage(dictations={"today": 0, "week": 0, "all": 0}, words={"today": 0, "week": 0, "all": 0})
    daily = [0] * _DAYS
    speaking, overall, waits = [], [], []
    apps: Dict[str, int] = {}
    for record in completed:
        ts, session = record["ts"], record.get("session_id", "")
        words = record.get("words")
        if words is None:
            text = record.get("final_text") or ""
            words = len(text.split())
            if len(text) >= _OLD_TEXT_LIMIT:
                words = _full_word_count(session, ts, training) or words
        if not words:
            continue
        day = datetime.fromtimestamp(ts).date()
        periods = ["all"] + (["week"] if ts >= week_ago else []) + (["today"] if day == today else [])
        for period in periods:
            result.dictations[period] += 1
            result.words[period] += words
        age = (today - day).days
        if 0 <= age < _DAYS:
            daily[_DAYS - 1 - age] += words
        if ts < week_ago:
            continue
        if record.get("app"):
            apps[record["app"]] = apps.get(record["app"], 0) + 1
        if session in stopped:
            waits.append(ts - stopped[session])
        if words < _MIN_WORDS:
            continue
        if session in started and session in stopped and stopped[session] > started[session]:
            speaking.append(words / ((stopped[session] - started[session]) / 60.0))
        if record.get("total_duration_ms", 0) > 1000:
            overall.append(words / (record["total_duration_ms"] / 60000.0))

    result.daily_words = daily
    if speaking:
        result.speaking_wpm = statistics.median(speaking)
    if overall:
        result.overall_wpm = statistics.median(overall)
    if waits:
        waits.sort()
        result.after_release = (statistics.median(waits), waits[int(len(waits) * 0.9)])
    total = sum(apps.values())
    result.apps = [(app, count / total) for app, count in
                   sorted(apps.items(), key=lambda kv: -kv[1])[:4]] if total else []
    return result
