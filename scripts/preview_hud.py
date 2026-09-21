"""Dev helper: play a scripted dictation through the recording HUD.

    ./venv/bin/python scripts/preview_hud.py            # once
    ./venv/bin/python scripts/preview_hud.py --loop     # until Ctrl-C

Three streams (Parakeet on the built-in mic, MAI on two mics), a chunk coming
back while you talk, release, Parakeet answering first, MAI next, the second
mic left behind by the deadline, then correction: every state the HUD has,
without a microphone or an API key.
"""

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from AppKit import NSApplication, NSApplicationActivationPolicyAccessory  # noqa: E402
from PyObjCTools import AppHelper  # noqa: E402

from mergescribe.ui.hud import RecordingHUD  # noqa: E402

_started = time.monotonic()


def level_at(t: float) -> float:
    """Syllables at about 7Hz, phrases that swell and fade, a pause that ends a chunk."""
    if 0.85 < t < 1.2 or t > 3.8:
        return 0.03
    syllables = abs(math.sin(t * 7.3)) * (0.55 + 0.45 * math.sin(t * 1.7 + 1.0))
    return min(1.0, 0.1 + 0.85 * syllables)


def fake_level() -> float:
    return level_at(time.monotonic() - _started)


S = "preview"
STREAMS = ["parakeet/MacBook Pro Microphone", "or-mai-transcribe-2/MacBook Pro Microphone",
           "or-mai-transcribe-2/SoloCast"]
SCRIPT = [
    (0.0, "set_status", ("recording",)),
    (0.0, "streams_planned", (S, STREAMS)),
    (1.25, "stream_landed", (S, STREAMS[0], False)),
    (1.4, "partial_text", (S, "so I was thinking we should")),
    (1.6, "stream_landed", (S, STREAMS[1], False)),
    (1.75, "stream_landed", (S, STREAMS[2], False)),
    (2.8, "partial_text", (S, "so I was thinking we should merge all the strands into one")),
    (3.8, "set_status", ("processing",)),
    (4.1, "stream_landed", (S, STREAMS[0], True)),
    (4.7, "stream_landed", (S, STREAMS[1], True)),
    (5.5, "stream_dropped", (S, STREAMS[2])),
    (5.5, "transcription_done", (S,)),
    (7.2, "set_status", ("idle",)),
]
LENGTH = 8.0


def play(hud: RecordingHUD, loop: bool) -> None:
    global _started
    _started = time.monotonic()
    for at, method, args in SCRIPT:
        AppHelper.callLater(at, getattr(hud, method), *args)
    if loop:
        AppHelper.callLater(LENGTH, play, hud, loop)
    else:
        AppHelper.callLater(LENGTH, AppHelper.stopEventLoop)


if __name__ == "__main__":
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    hud = RecordingHUD(level_source=fake_level)
    play(hud, loop="--loop" in sys.argv)
    AppHelper.runEventLoop(installInterrupt=True)
