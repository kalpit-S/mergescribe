"""Tests for the usage stats shown in Settings, read from the metrics log."""

import json
from datetime import datetime

NOW = datetime(2026, 9, 27, 15, 0).timestamp()
DAY = 86400.0


def _log(path, dictations):
    """One dictation: key down at start, up after held seconds, text done wait seconds later."""
    lines = []
    for i, d in enumerate(dictations):
        sid, start = f"s{i}", NOW - d.get("ago", 0.0)
        stop, done = start + d.get("held", 10.0), start + d.get("held", 10.0) + d.get("wait", 1.5)
        lines += [{"ts": start, "event": "recording_started", "session_id": sid},
                  {"ts": stop, "event": "recording_stopped", "session_id": sid},
                  {"ts": done, "event": "transcription", "session_id": sid, "provider": "p"},
                  {"ts": done, "event": "session_complete", "session_id": sid,
                   "total_duration_ms": (done - start) * 1000, **d.get("record", {})}]
    path.write_text("\n".join(json.dumps(line) for line in lines))


def test_words_and_dictations_by_period(tmp_path):
    from mergescribe.stats import usage

    log = tmp_path / "metrics.jsonl"
    _log(log, [{"record": {"words": 30}},                       # today
               {"ago": 3 * DAY, "record": {"words": 20}},       # this week
               {"ago": 30 * DAY, "record": {"words": 10}}])     # before that
    result = usage(log, tmp_path, now=NOW)
    assert result.words == {"today": 30, "week": 50, "all": 60}
    assert result.dictations == {"today": 1, "week": 2, "all": 3}
    assert result.daily_words[-1] == 30 and result.daily_words[-4] == 20


def test_old_records_count_their_text_and_long_ones_read_the_training_capture(tmp_path):
    """Before words were logged, the text was kept only to 500 characters."""
    from mergescribe.stats import usage

    log = tmp_path / "metrics.jsonl"
    _log(log, [{"record": {"final_text": "ship it friday"}},
               {"record": {"final_text": "word " * 100}}])   # 500 characters: cut off
    capture = tmp_path / datetime.fromtimestamp(NOW).strftime("%Y-%m-%d") / "s1"
    capture.mkdir(parents=True)
    (capture / "metadata.json").write_text(json.dumps({"final_output": "word " * 180}))
    assert usage(log, tmp_path, now=NOW).words["all"] == 3 + 180


def test_speed_is_the_median_over_the_week_and_skips_tiny_dictations(tmp_path):
    from mergescribe.stats import usage

    log = tmp_path / "metrics.jsonl"
    _log(log, [{"held": 12.0, "wait": 1.0, "record": {"words": 30}},    # 150 wpm speaking
               {"held": 20.0, "wait": 2.0, "record": {"words": 40}},    # 120
               {"held": 60.0, "wait": 1.5, "record": {"words": 150}},   # 150
               {"held": 1.0, "wait": 1.0, "record": {"words": 2}}])     # too short to count
    result = usage(log, tmp_path, now=NOW)
    assert round(result.speaking_wpm) == 150
    assert result.overall_wpm < result.speaking_wpm        # processing time counts end to end
    assert result.after_release[0] == 1.25                  # every dictation's wait counts


def test_apps_are_shares_of_the_week(tmp_path):
    from mergescribe.stats import usage

    log = tmp_path / "metrics.jsonl"
    _log(log, [{"record": {"words": 9, "app": "Claude"}}] * 3 + [{"record": {"words": 9, "app": "Slack"}}])
    assert usage(log, tmp_path, now=NOW).apps == [("Claude", 0.75), ("Slack", 0.25)]


def test_no_log_is_no_stats(tmp_path):
    from mergescribe.stats import usage

    result = usage(tmp_path / "missing.jsonl", tmp_path, now=NOW)
    assert result.words == {} and result.speaking_wpm is None
