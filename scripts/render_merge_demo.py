"""Render a logged dictation as the merge it was.

    ./venv/bin/python scripts/render_merge_demo.py <session-id> [--out demos/merge_demo.gif]

Each recognizer's transcript arrives at its real time after the key was
released, the words no transcript agreed on are struck out, and the corrected
text assembles from the words it took, each flying in from where it came from.
Reads ~/.mergescribe/metrics.jsonl; needs ffmpeg to encode the GIF. Playback is
slowed down so it can be read; every time printed on it is the logged one.
"""

import argparse
import difflib
import json
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from AppKit import (  # noqa: E402
    NSAttributedString, NSBezierPath, NSBitmapImageFileTypePNG, NSBitmapImageRep, NSColor,
    NSDeviceRGBColorSpace, NSFont, NSFontAttributeName, NSFontWeightMedium,
    NSFontWeightRegular, NSFontWeightSemibold, NSForegroundColorAttributeName,
    NSGraphicsContext, NSMakeRect, NSStrikethroughColorAttributeName,
    NSStrikethroughStyleAttributeName,
)

from mergescribe.ui.hud import _PALETTE  # noqa: E402  the HUD's stream colours

METRICS = Path.home() / ".mergescribe" / "metrics.jsonl"

SCALE, FPS = 2, 30
WIDTH, HEIGHT = 600, 300   # resized to fit the dictation in render()
SLOWDOWN = 3.0          # playback seconds per real second
LEAD_IN = 0.6           # before the key is released
HOLD = 2.8              # on the finished text
FADE = 0.4              # back to empty before the loop restarts
STREAMED_OVER = 0.45    # real seconds the corrected words stream in before done
FLIGHT = 0.55           # playback seconds for a word to fly into place
MARGIN = 36

_NAMES = {
    "fish-audio-transcribe-1": "Fish Audio",
    "microsoft-mai-transcribe-2": "MAI-Transcribe 2",
    "microsoft-mai-transcribe-1-5": "MAI-Transcribe 1.5",
    "openai-gpt-4o-transcribe": "GPT-4o Transcribe",
}
_MODELS = {"openai/gpt-5.6-luna": "GPT-5.6 Luna", "openai/gpt-5.6-terra": "GPT-5.6 Terra",
           "google/gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite"}


@dataclass
class Stream:
    name: str
    where: str
    arrived: float      # seconds after release
    words: List[str]


@dataclass
class Dictation:
    streams: List[Stream]
    final: List[str]
    model: str
    typed: float        # seconds after release


def display_name(provider: str) -> Tuple[str, str]:
    if provider == "parakeet":
        return "Parakeet", "on-device"
    slug = provider.removeprefix("or-")
    return _NAMES.get(slug, slug.replace("-", " ").title()), "cloud"


def load(session_id: str, path: Path = METRICS) -> Dictation:
    released = finished = None
    final, model, rows = "", "", []
    with path.open() as f:
        for line in f:
            if session_id not in line:
                continue
            event = json.loads(line)
            kind = event.get("event")
            if kind == "recording_stopped":
                released = event["ts"]
            elif kind == "transcription" and event.get("text", "").strip():
                rows.append(event)
            elif kind == "llm_correction":
                model = event.get("model", "")
            elif kind == "session_complete":
                final, finished = event.get("final_text", ""), event["ts"]
    if released is None or finished is None or not rows:
        sys.exit(f"{session_id}: not a complete session in {path}")
    if len({r["chunk_num"] for r in rows}) != 1:
        sys.exit(f"{session_id}: spans several chunks; pick a single-chunk dictation")
    streams = []
    for row in sorted(rows, key=lambda r: r["ts"]):
        name, where = display_name(row["provider"])
        streams.append(Stream(name, where, row["ts"] - released, row["text"].split()))
    return Dictation(streams, final.split(), _MODELS.get(model, model), finished - released)


def _norm(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower())


def provenance(d: Dictation) -> List[List[Tuple[int, int]]]:
    """For each final word, the (stream, word index) pairs it was aligned with."""
    sources = [[] for _ in d.final]
    target = [_norm(w) for w in d.final]
    for s, stream in enumerate(d.streams):
        matcher = difflib.SequenceMatcher(None, [_norm(w) for w in stream.words], target,
                                          autojunk=False)
        for a, b, size in matcher.get_matching_blocks():
            for k in range(size):
                sources[b + k].append((s, a + k))
    return sources


# -- drawing ----------------------------------------------------------------

def _color(rgb, alpha=1.0):
    return NSColor.colorWithSRGBRed_green_blue_alpha_(rgb[0], rgb[1], rgb[2], alpha)


WHITE = (1.0, 1.0, 1.0)
DIM = (0.55, 0.57, 0.62)


def _text(string, size, weight, rgb, alpha=1.0, strike=0.0):
    attrs = {NSFontAttributeName: NSFont.systemFontOfSize_weight_(size, weight),
             NSForegroundColorAttributeName: _color(rgb, alpha)}
    if strike > 0:
        attrs[NSStrikethroughStyleAttributeName] = 1
        attrs[NSStrikethroughColorAttributeName] = _color((1.0, 0.42, 0.42), strike)
    return NSAttributedString.alloc().initWithString_attributes_(string, attrs)


def _draw(string, x, top, size, weight, rgb, alpha=1.0, strike=0.0):
    """Draw with the baseline box's top edge at `top`, measured from the top."""
    text = _text(string, size, weight, rgb, alpha, strike)
    text.drawAtPoint_((x, HEIGHT - top - text.size().height))
    return text.size().width


def _layout(words, x, size, weight):
    """x position of each word when set as one line."""
    space = _text(" ", size, weight, WHITE).size().width
    positions = []
    for word in words:
        positions.append(x)
        x += _text(word, size, weight, WHITE).size().width + space
    return positions


def _ease(u):
    u = max(0.0, min(1.0, u))
    return 1.0 - (1.0 - u) ** 3


def render(d: Dictation, out: Path) -> None:
    global WIDTH, HEIGHT
    sources = provenance(d)
    used = {pair for pairs in sources for pair in pairs}
    row_top = [MARGIN + 30 + i * 64 for i in range(len(d.streams))]
    final_top = row_top[-1] + 104
    stream_x = [_layout(s.words, MARGIN, 17, NSFontWeightRegular) for s in d.streams]
    final_x = _layout(d.final, MARGIN, 21, NSFontWeightSemibold)
    widest = max([_text(" ".join(s.words), 17, NSFontWeightRegular, WHITE).size().width
                  for s in d.streams]
                 + [_text(" ".join(d.final), 21, NSFontWeightSemibold, WHITE).size().width])
    WIDTH, HEIGHT = max(520, int(widest) + 2 * MARGIN), final_top + 56

    released = LEAD_IN
    at = lambda real: released + real * SLOWDOWN   # noqa: E731
    judged = at(d.streams[-1].arrived) + 0.35
    # Words stream in over the last moments, the last one landing as it was typed.
    word_start = [at(d.typed - STREAMED_OVER * (1 - k / max(1, len(d.final) - 1))) - FLIGHT
                  for k in range(len(d.final))]
    total = at(d.typed) + HOLD + FADE

    frames = Path(tempfile.mkdtemp(prefix="merge_demo_"))
    for n in range(int(total * FPS)):
        t = n / FPS
        rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
            None, WIDTH * SCALE, HEIGHT * SCALE, 8, 4, True, False, NSDeviceRGBColorSpace, 0, 0)
        rep.setSize_((WIDTH, HEIGHT))
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.setCurrentContext_(NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep))
        _color((0.067, 0.071, 0.086)).setFill()
        NSBezierPath.fillRect_(NSMakeRect(0, 0, WIDTH, HEIGHT))
        _frame(d, t, at, judged, word_start, sources, used, row_top, final_top,
               stream_x, final_x)
        fade = (t - (total - FADE)) / FADE
        if fade > 0:
            _color((0.067, 0.071, 0.086), min(1.0, fade)).setFill()
            NSBezierPath.fillRect_(NSMakeRect(0, 0, WIDTH, HEIGHT))
        NSGraphicsContext.restoreGraphicsState()
        png = rep.representationUsingType_properties_(NSBitmapImageFileTypePNG, {})
        png.writeToFile_atomically_(str(frames / f"{n:04d}.png"), False)

    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", str(frames / "%04d.png"),
        "-vf", "split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
               "[b][p]paletteuse=dither=none:diff_mode=rectangle",
        str(out)], check=True)
    print(f"{out}: {out.stat().st_size / 1e6:.2f} MB, {total:.1f}s")


def _frame(d, t, at, judged, word_start, sources, used, row_top, final_top, stream_x, final_x):
    released = at(0.0)
    clock = max(0.0, min((t - released) / SLOWDOWN, d.typed))
    _draw("key released" if t >= released else "recording", WIDTH - MARGIN - 190, MARGIN - 8,
          12, NSFontWeightMedium, DIM)
    _draw(f"+{clock:.2f} s", WIDTH - MARGIN - 70, MARGIN - 8, 12, NSFontWeightMedium,
          WHITE, 0.9 if t >= released else 0.4)
    _draw(f"slowed {SLOWDOWN:g}×", MARGIN, MARGIN - 8, 12, NSFontWeightMedium, DIM, 0.8)

    judging = _ease((t - judged) / 0.4)
    for s, stream in enumerate(d.streams):
        color = _PALETTE[s % len(_PALETTE)][0]
        top = row_top[s]
        _color(color).setFill()
        NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(MARGIN, HEIGHT - top - 12, 8, 8)).fill()
        width = _draw(stream.name, MARGIN + 16, top, 13, NSFontWeightSemibold, WHITE, 0.9)
        arrived = at(stream.arrived)
        detail = (f"{stream.where} · {stream.arrived:.2f} s" if t >= arrived
                  else f"{stream.where} · transcribing…" if t >= released
                  else stream.where)
        _draw(detail, MARGIN + 24 + width, top + 1, 12, NSFontWeightRegular, DIM)
        if t < arrived:
            continue
        shown = _ease((t - arrived) / 0.3)
        for k, word in enumerate(stream.words):
            kept = (s, k) in used
            alpha = shown * (1.0 if kept else 1.0 - 0.6 * judging)
            _draw(word, stream_x[s][k], top + 20 + 6 * (1 - shown), 17, NSFontWeightRegular,
                  WHITE, 0.92 * alpha, strike=0.0 if kept else judging)

    # The corrected line: each word flies in from the transcript it came from.
    top = final_top
    if t >= word_start[0]:
        _draw(d.model or "correction model", MARGIN + 16, top - 30, 13, NSFontWeightSemibold,
              WHITE, 0.9 * _ease((t - word_start[0]) / 0.3))
        typed = t >= at(d.typed)
        _draw(f"merged · typed at {d.typed:.2f} s" if typed else "merging…",
              MARGIN + 16 + _text(d.model or "correction model", 13, NSFontWeightSemibold,
                                  WHITE).size().width + 8,
              top - 29, 12, NSFontWeightRegular, DIM)
        _color(WHITE, 0.85).setFill()
        NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(MARGIN, HEIGHT - top + 18, 8, 8)).fill()
    for k, word in enumerate(d.final):
        u = (t - word_start[k]) / FLIGHT
        if u < 0:
            continue
        pairs = sources[k]
        if pairs:
            s, i = pairs[0]
            origin = (stream_x[s][i], row_top[s] + 20)
            tint = _PALETTE[s % len(_PALETTE)][0] if len({p[0] for p in pairs}) == 1 \
                and len(d.streams) > 1 else WHITE
        else:
            origin, tint = (final_x[k], top + 12), WHITE    # the model's own word
        e = _ease(u)
        x = origin[0] + (final_x[k] - origin[0]) * e
        y = origin[1] + (top - origin[1]) * e - 14 * math.sin(math.pi * min(1.0, u))
        size = 17 + 4 * e
        _draw(word, x, y, size, NSFontWeightSemibold, tint, min(1.0, u * 3.0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("session_id")
    parser.add_argument("--out", type=Path, default=Path("demos/merge_demo.gif"))
    args = parser.parse_args()
    render(load(args.session_id), args.out)
