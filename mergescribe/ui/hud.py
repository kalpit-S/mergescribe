"""
Floating recording HUD.

The menu bar already showed state, but it sits at the top of the screen while
your eyes are on the field you're dictating into, so in practice there was no
feedback at all: press, talk, release, then ~1.2s of nothing before text
appears. This puts the same state where you are actually looking.

The animation follows what the app does. While you talk, each recognizer is a
curtain of aurora rising and flickering with your voice. When you let go the
curtains settle, one by one as their results land, into a single glowing hem,
and each word the correction model types sends light running along it and on
through your words.

Everything that moves is a damped spring in HUDAnimation, a pure-Python model:
motion keeps its momentum when interrupted instead of restarting a tween, and
the model can be stepped and tested without a screen. The AppKit view only
reads it.

Two hard constraints shape the implementation:

1. It must never take focus. Dictation types into whatever field is focused, so
   a panel that activates would break the entire app. Hence a borderless
   NSPanel with NSWindowStyleMaskNonactivatingPanel, shown with
   orderFrontRegardless(), ignoring mouse events, and with hidesOnDeactivate
   off so it survives the owning app never being frontmost.

2. It must never block or crash recording. Every entry point is wrapped and
   degrades to a no-op, and the audio thread never touches AppKit: it only
   writes a float, which the redraw timer samples on the main thread.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple
import math
import threading
import time

import numpy as np

# Geometry. The pill stays compact until there is transcript to show: 65% of
# sessions are a single chunk, so nothing is ever available before release and
# they should keep the small unobtrusive form.
_COMPACT_WIDTH = 150.0
_WIDE_WIDTH = 420.0
_HEIGHT = 40.0
_DOT = 40.0             # the capsule is born from, and returns to, a circle
_BOTTOM_MARGIN = 130.0
_PADDING = 14.0
_TEXT_STRIP = 112.0     # how much of the capsule the aurora keeps once words arrive
_TEXT_GAP = 14.0
# The capsule is ink, not glass: a blurred desktop tints it muddy (olive over a
# green wallpaper, slate over blue) and greys the light inside. Near-black with
# a faint cool falloff keeps depth and lets the aurora's colours carry.
_INK_TOP = (0.075, 0.082, 0.11)
_INK_BOTTOM = (0.018, 0.02, 0.03)
_AURORA_PAD = 12.0      # clear margin round the aurora image for its glow to spread into
_AURORA_GLOW = 5.0      # pt of blur on the glow copy
# The panel is a fixed transparent canvas larger than the pill, so the pill can
# grow, overshoot and glow without resizing the window every frame.
_CANVAS_MARGIN = 48.0

_FPS = 60.0
_MAX_STEP = 1.0 / 20.0   # a stalled main thread shouldn't fling the springs

# Springs as (stiffness, damping). Damping ratio is damping / (2 * sqrt(stiffness)):
# below 1 overshoots a little, which is what makes an entrance feel physical.
_PRESENCE_SPRING = (190.0, 17.0)   # 0.62: pops in with a small overshoot
_WIDTH_SPRING = (210.0, 25.0)      # 0.86: widens for text without wobbling
_MERGE_SPRING = (80.0, 18.0)       # 1.0: strands converge without bouncing

# Asymmetric ballistics, borrowed from analogue VU meters: jump to a peak
# almost immediately, fall away slowly. A meter that tracks raw RMS both ways
# reads as noise; this reads as a voice.
_LEVEL_ATTACK = 0.65
_LEVEL_RELEASE = 0.16

# Level mapping: how many dB above the mic's noise floor counts as "full scale".
# Speech typically sits 12-30dB above the floor, so 30 keeps normal talking in
# the upper half of the meter without clipping every syllable to max.
_LEVEL_SPAN_DB = 30.0

# One strand per stream (a recognizer on a mic), so what you see is what is
# running. Each drifts at its own speed, in radians a second, and alternate
# ones the other way, so no two curtains move in step.
_STRAND_SPEEDS = (5.3, -4.1, 6.4, -5.6)
_DEFAULT_STREAMS = 3       # until a session says how many it runs
_STRAND_DIMMED = 0.35      # missed a chunk while recording; may yet catch up
_FLASH_DECAY = 0.3         # seconds for a landing's flash to fade

# Joined: streams that have returned settle into one hem. Once correction
# starts, light travels along it and on through the words.
_MERGED_SPEED = 11.0
_PULSE_PERIOD = 1.5        # seconds per pass, when nothing reports tokens
_PULSE_LEAD_IN = 0.25      # lets the last strand arrive before the pulse sets off
_PULSE_SPEED = 0.9         # how fast one token's ripple crosses the line
_MAX_RIPPLES = 8           # a fast stream would otherwise draw hundreds
# Correction models stream in bursts - five tokens in a packet, then a pause -
# so ripples are spaced out to move at the rate words land, not packets.
_RIPPLE_SPACING = 0.06

# Only the tail of the transcript is shown. Multi-chunk sessions hold 81 words
# at release (p90: 378), which no pill can display, and a wall of your own
# words appearing while you are still speaking is a good way to lose your
# thread. The last few words confirm it is listening without inviting reading.
_TAIL_CHARS = 58

_VISIBLE = ("recording", "processing")


def normalize_level(db: float, noise_floor: Optional[float]) -> float:
    """
    Map a dB reading to 0..1 for the meter.

    Relative to the mic's own noise floor when we have one, so the meter reads
    the same on a quiet built-in mic and a hot condenser. Falls back to a fixed
    -60..0 dB range before the floor has been established.
    """
    if not math.isfinite(db):
        return 0.0
    if noise_floor is not None and math.isfinite(noise_floor):
        level = (db - noise_floor) / _LEVEL_SPAN_DB
    else:
        level = (db + 60.0) / 60.0
    return max(0.0, min(1.0, level))


def smooth_level(current: float, previous: float,
                 attack: float = _LEVEL_ATTACK,
                 release: float = _LEVEL_RELEASE) -> float:
    """
    Ballistics for the meter: fast attack, slow release.

    Rising levels are followed almost immediately so consonants register;
    falling ones decay gently so the meter doesn't flicker between syllables.
    """
    rate = attack if current > previous else release
    return previous + (current - previous) * rate


def spring_step(value: float, velocity: float, target: float, dt: float,
                stiffness: float, damping: float) -> Tuple[float, float]:
    """
    Advance a damped spring by dt. Returns the new (value, velocity).

    Semi-implicit Euler, which stays stable at frame-rate steps. Retargeting
    mid-flight keeps the velocity, so an interrupted animation bends toward
    its new goal instead of starting over.
    """
    velocity += (stiffness * (target - value) - damping * velocity) * dt
    return value + velocity * dt, velocity


class Spring:
    """A value that chases its target with momentum."""

    __slots__ = ("value", "velocity", "target", "stiffness", "damping")

    def __init__(self, value: float, params: Tuple[float, float]):
        self.value = self.target = value
        self.velocity = 0.0
        self.stiffness, self.damping = params

    def step(self, dt: float) -> None:
        self.value, self.velocity = spring_step(
            self.value, self.velocity, self.target, dt, self.stiffness, self.damping)

    def snap(self, value: float) -> None:
        self.value = self.target = value
        self.velocity = 0.0


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _wrap(angle: float) -> float:
    """An angle folded into -pi..pi, so phases converge the short way round."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def tail_text(text: str, budget: int = _TAIL_CHARS) -> str:
    """
    Last few words of the transcript, trimmed at a word boundary.

    Returns "" for empty input so the HUD knows to stay in its compact form.
    """
    text = " ".join((text or "").split())
    if len(text) <= budget:
        return text
    clipped = text[-budget:]
    space = clipped.find(" ")
    if space != -1 and space < budget // 2:
        clipped = clipped[space + 1:]
    return "… " + clipped


class _Strand:
    """One stream's curtain of light."""

    __slots__ = ("key", "speed", "phase", "join", "presence", "flash")

    def __init__(self, key: str, index: int):
        self.key = key
        self.speed = _STRAND_SPEEDS[index % len(_STRAND_SPEEDS)]
        self.phase = index * 2.1
        self.join = Spring(0.0, _MERGE_SPRING)       # 0 its own curtain, 1 part of the hem
        self.presence = Spring(1.0, _MERGE_SPRING)   # 0 dropped, 1 present
        self.flash = 0.0                             # brightens when a chunk lands


class HUDAnimation:
    """
    Everything on the HUD that moves, advanced by step().

    presence: 0 hidden, 1 shown. Drives opacity, scale and a small rise.
    width:    0 compact, 1 wide enough for transcript.
    work:     0 listening to your voice, 1 released and transcribing.
    pulse:    0 still transcribing, 1 correcting.

    Each strand is a stream the session reported. While you talk they move
    with your voice and flash when a chunk comes back. After release each
    joins the line as its final result lands, or fades if it is dropped;
    when every stream is in, the pulse starts.
    """

    def __init__(self):
        self.status = "idle"
        self.transcript = ""
        self.clock = 0.0
        self.level = 0.0
        self.presence = Spring(0.0, _PRESENCE_SPRING)
        self.width = Spring(0.0, _WIDTH_SPRING)
        self.work = Spring(0.0, _MERGE_SPRING)
        self.pulse = Spring(0.0, _MERGE_SPRING)
        self.strands = [_Strand("", i) for i in range(_DEFAULT_STREAMS)]
        self._session: Optional[str] = None
        self._merged_phase = 0.0
        self._pulse_clock = 0.0
        self._correcting = False
        self.skipped = False    # the transcript needed no correcting
        self.flash = 0.0        # the moment that was decided
        self.discarded = False  # the speaker called the dictation off
        self.edge_phase = 0.0     # where the light running round the edge has got to
        self.collapse = Spring(0.0, _MERGE_SPRING)
        self._ripples: List[float] = []   # position of each token's ripple
        self._tokens_seen = False         # has anything reported tokens at all?

    # -- what the session reports -------------------------------------------

    @property
    def visible(self) -> bool:
        return self.status in _VISIBLE

    def set_status(self, status: str) -> None:
        if status == "recording":
            # A new recording owns the pill, even if the last one is still
            # fading out: its words and its streams no longer apply.
            self._session = None
            self.set_transcript("")
            self.work.target = 0.0
            self.pulse.snap(0.0)
            self._correcting = False
            self.skipped = False
            self.discarded = False
            self.flash = 0.0
            self._ripples = []
            self._tokens_seen = False
            self.collapse.snap(0.0)
            for strand in self.strands:
                strand.join.target = 0.0
                strand.presence.target = 1.0
        elif status == "processing":
            self.work.target = 1.0
            if self._session is None:
                # Nobody is reporting streams, so nothing will say when
                # they're done. Merge now rather than wait forever.
                self._start_correcting()
        self.status = status

    def streams_planned(self, session: str, streams: List[str]) -> None:
        self._session = session
        if [s.key for s in self.strands] != list(streams):
            self.strands = [_Strand(key, i) for i, key in enumerate(streams)]

    def stream_landed(self, session: str, stream: str, final: bool) -> None:
        strand = self._strand(session, stream)
        if strand is None:
            return
        strand.flash = 1.0
        strand.presence.target = 1.0
        if final:
            strand.join.target = 1.0

    def stream_dropped(self, session: str, stream: str) -> None:
        strand = self._strand(session, stream)
        if strand is not None:
            strand.presence.target = 0.0 if self.status == "processing" else _STRAND_DIMMED

    def transcription_done(self, session: str) -> None:
        if session == self._session:
            self._start_correcting()

    def token_typed(self, session: str) -> None:
        """A word just landed on screen. Send a ripple down the line."""
        if session != self._session:
            return
        self._tokens_seen = True
        if len(self._ripples) >= _MAX_RIPPLES:
            return
        # The newest ripple is the one furthest left, so queue behind the
        # smallest position - otherwise a burst stacks into a single lump.
        newest = min(self._ripples, default=1.0)
        self._ripples.append(min(0.0, newest - _RIPPLE_SPACING))

    def dictation_discarded(self, session: str) -> None:
        """Nothing will be typed: draw the line inward rather than fade it wide."""
        if session == self._session:
            self.discarded = True
            self.collapse.target = 1.0

    def correction_skipped(self, session: str) -> None:
        """The transcript was typed as it stood: no pulse, one bright flash."""
        if session != self._session:
            return
        self.skipped = True
        self.flash = 1.0
        self.pulse.target = 0.0

    def partial_text(self, session: str, text: str) -> None:
        if session == self._session:
            self.set_transcript(text)

    def set_transcript(self, text: str) -> None:
        self.transcript = text or ""
        self.width.target = 1.0 if self.transcript else 0.0

    def _strand(self, session: str, stream: str) -> Optional[_Strand]:
        if session != self._session:
            return None     # a finished session reporting late
        return next((s for s in self.strands if s.key == stream), None)

    def _start_correcting(self) -> None:
        # Streams that neither landed nor dropped were cancelled by consensus:
        # the others agreed for them, so they join too.
        for strand in self.strands:
            if strand.presence.target > 0.0:
                strand.presence.target = 1.0
                strand.join.target = 1.0
        if not self._correcting:
            self._correcting = True
            self.pulse.target = 1.0
            self._pulse_clock = -_PULSE_LEAD_IN

    # -- time ----------------------------------------------------------------

    def step(self, dt: float, level: float = 0.0) -> None:
        dt = max(0.0, min(dt, _MAX_STEP))
        self.clock += dt
        self.presence.target = 1.0 if self.visible else 0.0
        for spring in (self.presence, self.width, self.work, self.pulse):
            spring.step(dt)
        self.level = smooth_level(level if self.status == "recording" else 0.0, self.level)

        # Each strand drifts at its own speed, faster while it transcribes; as
        # it joins, its speed converges and its phase is pulled onto the line.
        busy = _lerp(1.0, 1.5, _clamp01(self.work.value))
        self._merged_phase += _MERGED_SPEED * dt
        fade = math.exp(-dt / _FLASH_DECAY)
        for strand in self.strands:
            strand.join.step(dt)
            strand.presence.step(dt)
            strand.flash *= fade
            m = _clamp01(strand.join.value)
            strand.phase += _lerp(strand.speed * busy, _MERGED_SPEED, m) * dt
            strand.phase += _wrap(self._merged_phase - strand.phase) * min(1.0, 8.0 * dt) * m
        self.flash *= fade
        self.collapse.step(dt)
        # The edge light drifts while listening and runs while the pipeline
        # works; it stops when there is nothing left to wait for.
        if self.skipped or self.discarded:
            laps_per_second = 0.0
        else:
            laps_per_second = _lerp(0.08, 0.55, _clamp01(self.work.value))
        self.edge_phase = (self.edge_phase + laps_per_second * dt) % 1.0
        # Ripples travel on their own; nothing moves while the stream is quiet.
        self._ripples = [p + _PULSE_SPEED * dt for p in self._ripples if p < 1.25]
        if self._correcting and not self.skipped:
            self._pulse_clock += dt

    def reset(self) -> None:
        """Back to a blank slate once fully hidden."""
        status = self.status
        self.__init__()
        self.status = status

    # -- what the view draws ------------------------------------------------

    @property
    def hidden(self) -> bool:
        """Faded out with nothing left to show: the panel can be ordered out."""
        return not self.visible and self.presence.value < 0.01

    @property
    def alpha(self) -> float:
        return _clamp01(self.presence.value)

    @property
    def morph(self) -> float:
        """How far the dot has opened into a capsule.

        Unclamped on purpose: the presence spring's small overshoot is what
        makes the capsule land with a little give, the way a physical thing does.
        """
        return self.presence.value

    @property
    def content_alpha(self) -> float:
        """The strands and words arrive once the capsule has mostly opened."""
        opened = _clamp01((self.presence.value - 0.35) / 0.55)
        return opened * (1.0 - _clamp01(self.collapse.value))

    @property
    def rise(self) -> float:
        """How far below its resting place the capsule is, in points."""
        return (1.0 - min(self.presence.value, 1.0)) * 8.0

    def edge_light(self) -> Tuple[float, float, float]:
        """
        The light along the capsule's edge, as (steady, head, sweep).

        steady lights the whole edge: it follows your voice while listening and
        flares when a dictation is typed without correction. head is where a
        brighter sweep has got to going round (0..1), and sweep how bright it
        is - slow and faint while listening, running while the work happens.
        """
        working = _clamp01(self.work.value)
        steady = _lerp(0.02 + 0.5 * self.level ** 1.5, 0.05, working) + 0.9 * self.flash
        sweep = 0.0 if self.skipped else _lerp(0.4, 0.9, working)
        fade = 1.0 - _clamp01(self.collapse.value)
        return steady * fade, self.edge_phase, sweep * fade

    @property
    def correcting(self) -> float:
        """0..1, how far into the correcting look the pill is."""
        return _clamp01(self.pulse.value)

    @property
    def pulse_position(self) -> float:
        """Where the correcting pulse is, as a fraction of the pill's content.

        Starts off the left edge and exits off the right, so each pass fades
        in and out rather than appearing at a point.
        """
        progress = (max(self._pulse_clock, 0.0) / _PULSE_PERIOD) % 1.0
        return -0.2 + 1.4 * progress

    @property
    def pulse_track(self) -> float:
        """Where the shimmer over the words sits, so it moves with the line.

        Following the furthest-along ripple keeps the two halves of the pill
        telling one story: the wave reaches the strip, the shimmer carries it
        on through the words.
        """
        if self._tokens_seen:
            return max(self._ripples, default=-1.0)
        return self.pulse_position


# -- the aurora -----------------------------------------------------------------
#
# Each stream is a curtain of northern lights: a bright hem low in the capsule
# with rays climbing from it, flickering with your voice and taller when you are
# louder. The curtains settle into a single glowing hem as the results merge,
# and each word the correction types sends light running along it.
#
# It is a field of soft light, so it is computed here with numpy at 1x over the
# aurora's strip only (about 5,000 pixels) and handed to the GPU as an image;
# the glow is a blurred copy composited by Core Image, costing no CPU at all.

_CURTAINS = np.array(((0.30, 1.00, 0.62),    # green
                      (0.22, 0.78, 1.00),    # cyan
                      (0.70, 0.46, 1.00),    # violet
                      (0.95, 0.62, 0.35)),   # amber, for a fourth stream
                     dtype=np.float32)
_CROWN = np.array((0.95, 0.42, 0.85), np.float32)     # tall rays go magenta at the top
_SETTLED = np.array((0.78, 1.00, 0.93), np.float32)   # the hem once the streams agree
_HEM_BELOW_CENTRE = 7.0     # pt: the hem sits low so the rays have room to climb
_RAY_GRAIN = 0.16           # rays per pt, roughly
_RIPPLE_REACH = 11.0        # pt: how far along the hem one word's light spreads


def _noise1(x: np.ndarray, seed: float) -> np.ndarray:
    """Smooth 1-D value noise in [-1, 1]."""
    i = np.floor(x)
    f = x - i

    def hash_(n):
        return np.mod(np.sin((n + seed * 17.13) * 127.1 + 311.7) * 43758.5453, 1.0)

    a, b = hash_(i), hash_(i + 1.0)
    return (a + (b - a) * f * f * (3 - 2 * f)) * 2 - 1


def _rays(x: np.ndarray, t: float, seed: float) -> np.ndarray:
    """The vertical striation of a curtain along x, drifting over time."""
    v = np.zeros_like(x)
    amp, freq = 0.5, 1.0
    for octave in range(3):
        v = v + amp * _noise1(x * freq + t * (0.6 + 0.3 * octave), seed + octave)
        amp, freq = amp * 0.5, freq * 2.1
    return v


def aurora_rgba(anim: "HUDAnimation", width_pt: float, height_pt: float,
                pad_pt: float = 0.0, scale: float = 1.0,
                track_pt: Optional[float] = None) -> np.ndarray:
    """
    The aurora for this frame as premultiplied RGBA bytes, rows top to bottom.

    width_pt x height_pt is the strip it lives in; pad_pt of clear margin is
    added on every side so a blurred copy has room to glow. Alpha is the
    light's brightness, so drawn normally over the dark capsule it adds light.

    track_pt is the length of the whole content, strip and words together,
    that travelling light crosses: the same track the shimmer over the words
    follows, so a word's light runs along the hem and carries on into them.
    """
    w = max(1, int(round(width_pt * scale)))
    h = max(1, int(round(height_pt * scale)))
    pad = int(round(pad_pt * scale))
    x = (np.arange(w, dtype=np.float32) + 0.5) / scale            # pt along the strip
    y = ((np.arange(h, dtype=np.float32) + 0.5) / scale)[:, None]  # pt down from the top
    u = x / max(1.0, width_pt)
    ends = (np.clip(u / 0.18, 0, 1) ** 2 * (3 - 2 * np.clip(u / 0.18, 0, 1))
            * np.clip((1 - u) / 0.18, 0, 1) ** 2 * (3 - 2 * np.clip((1 - u) / 0.18, 0, 1)))
    rest = height_pt / 2.0 + _HEM_BELOW_CENTRE
    work = _clamp01(anim.work.value)
    # Once anything reports tokens the light follows them, and goes still when
    # they stop - a stalled stream should look stalled. Only a correction
    # nobody reports falls back to a steady sweep.
    track = width_pt if track_pt is None else track_pt
    if anim._tokens_seen:
        travelling, gain = anim._ripples, 1.0
    else:
        travelling, gain = [anim.pulse_position], anim.correcting
    ripple = np.zeros_like(u)
    for p in travelling:
        ripple += np.exp(-((x - p * track) / _RIPPLE_REACH) ** 2)
    ripple = np.minimum(ripple, 1.0) * gain
    present = [_clamp01(s.presence.value) for s in anim.strands]
    shown = max(1, sum(1 for p in present if p > 0.05))
    joined = sum(_clamp01(s.join.value) * p for s, p in zip(anim.strands, present)) / max(1e-3, sum(present))
    share = 1.0 / (1.0 + (shown - 1) * joined)     # merged curtains overlap: split the light
    light = np.zeros((h, w, 3), np.float32)
    for i, strand in enumerate(anim.strands):
        join, here = _clamp01(strand.join.value), present[i]
        if here < 0.01:
            continue
        wander = 0.6 + 0.4 * math.sin(anim.clock * (1.1 + 0.5 * i) + 2 * i)
        amp = (2.5 + 7.5 * anim.level * wander) * (1 - join) + 0.8 * join
        freq = (1.1, 1.6, 0.8, 1.3)[i % 4]
        hem = rest + amp * np.sin(2 * math.pi * freq * u + strand.phase * 0.35
                                  + 0.9 * np.sin(2 * math.pi * 0.6 * u + anim.clock * 0.8 + i))
        hem = hem * (1 - join) + rest * join
        rays = (0.55 + 0.45 * _rays(x * _RAY_GRAIN + i * 9.7, anim.clock * 0.9, i)) * (1 - join) + 0.9 * join
        reach = (4.0 + 16.0 * anim.level + 9.0 * strand.flash) * (1 - 0.8 * join) + 3.0 * ripple * join
        d = hem[None, :] - y                                           # > 0 above the hem
        above = np.exp(-np.maximum(d, 0) / np.maximum(reach, 0.8)[None, :]) * rays[None, :]
        below = np.exp(-(np.minimum(d, 0) / 1.6) ** 2)
        shape = np.where(d > 0, above, below) + np.exp(-(d / 1.1) ** 2) * (0.55 + 0.45 * join)
        bright = ((0.5 + 0.7 * anim.level + 1.1 * strand.flash) * (1 - work) + 0.8 * work) * here * share
        bright = bright * (1 + 1.1 * ripple * join)
        tall = np.clip(d / (reach[None, :] * 2.2 + 1), 0, 1)[..., None]
        colour = _CURTAINS[i % 4] * (1 - 0.45 * tall) + _CROWN * 0.45 * tall
        colour = colour * (1 - 0.8 * join) + _SETTLED * 0.8 * join
        light += colour * (shape * (ends * bright)[None, :])[..., None] * 0.55
    light += (_SETTLED * 0.3 * anim.flash)[None, None, :] * ends[None, :, None]
    light = 1.0 - np.exp(-light * 1.6)                      # saturate gently, never clip
    alpha = light.max(axis=-1, keepdims=True)
    rgba = np.concatenate([light, alpha], axis=-1) * _clamp01(anim.content_alpha * 1.2)
    if pad:
        rgba = np.pad(rgba, ((pad, pad), (pad, pad), (0, 0)))
    return (np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8)


class RecordingHUD:
    """
    Facade over the AppKit panel.

    Safe to construct before NSApplication exists; the window is created lazily
    on first show, which is always after the menu bar's event loop is running.
    """

    def __init__(self, level_source: Optional[Callable[[], float]] = None,
                 enabled: bool = True):
        self._level_source = level_source
        self._enabled = enabled
        self._controller = None
        self._unavailable = False
        self._last_token = 0.0
        self._lock = threading.Lock()

    @property
    def is_showing(self) -> bool:
        """True when the HUD is the one carrying recording state.

        The menu bar defers to it when this is True, and takes the job back
        when the HUD is switched off or has disabled itself after a failure.
        """
        return self._enabled and not self._unavailable

    def set_status(self, status: str) -> None:
        """Show 'recording' or 'processing'; anything else hides the HUD."""
        if not self._enabled or self._unavailable:
            return
        try:
            self._call_on_main(lambda: self._apply_status(status))
        except Exception as e:
            print(f"[HUD] set_status failed, disabling: {e}")
            self._unavailable = True

    # SessionObserver: what the running session reports, drawn as strands.
    # Called from session threads; each hops to the main thread like set_status.

    def streams_planned(self, session: str, streams: List[str]) -> None:
        self._forward("streams_planned", session, list(streams))

    def stream_landed(self, session: str, stream: str, final: bool) -> None:
        self._forward("stream_landed", session, stream, final)

    def stream_dropped(self, session: str, stream: str) -> None:
        self._forward("stream_dropped", session, stream)

    def transcription_done(self, session: str) -> None:
        self._forward("transcription_done", session)

    def correction_skipped(self, session: str) -> None:
        self._forward("correction_skipped", session)

    def dictation_discarded(self, session: str) -> None:
        self._forward("dictation_discarded", session)

    def token_typed(self, session: str) -> None:
        """Throttled: a burst of tokens cannot outpace the screen."""
        now = time.monotonic()
        if now - self._last_token < 1.0 / _FPS:
            return
        self._last_token = now
        self._forward("token_typed", session)

    def partial_text(self, session: str, text: str) -> None:
        self._forward("partial_text", session, tail_text(text))

    def _forward(self, event: str, *args) -> None:
        if not self._enabled or self._unavailable:
            return
        try:
            self._call_on_main(lambda: self._apply_event(event, args))
        except Exception as e:
            print(f"[HUD] {event} failed, disabling: {e}")
            self._unavailable = True

    def _apply_event(self, event: str, args: tuple) -> None:
        try:
            if self._controller is None:
                return   # nothing visible yet; status will bring it up
            getattr(self._controller.animation(), event)(*args)
        except Exception as e:
            print(f"[HUD] {event} update failed, disabling: {e}")
            self._unavailable = True

    def shutdown(self) -> None:
        if self._controller is None:
            return
        try:
            self._call_on_main(self._teardown)
        except Exception:
            pass

    # -- internals -------------------------------------------------------

    def _call_on_main(self, fn) -> None:
        """Run fn on the main thread; AppKit objects must not be touched off it."""
        from PyObjCTools import AppHelper
        AppHelper.callAfter(fn)

    def _apply_status(self, status: str) -> None:
        try:
            # Don't build a panel just to keep it hidden. main() sets "idle"
            # before the event loop starts, and most statuses never show.
            if status not in _VISIBLE and self._controller is None:
                return
            controller = self._ensure_controller()
            if controller is None:
                return
            controller.setStatus_(status)
        except Exception as e:
            print(f"[HUD] update failed, disabling: {e}")
            self._unavailable = True

    def _ensure_controller(self):
        with self._lock:
            if self._controller is None:
                self._controller = _HUDController.alloc().initWithLevelSource_(
                    self._level_source
                )
                if self._controller is not None:
                    self._controller.on_failure = self._render_failed
            return self._controller

    def _render_failed(self) -> None:
        """The panel stopped drawing: from here on the menu bar carries the status."""
        print("[HUD] disabled after a render error; the menu bar shows status")
        self._unavailable = True

    def _teardown(self) -> None:
        try:
            self._controller.tearDown()
        except Exception:
            pass
        self._controller = None


try:  # pragma: no cover - requires a macOS GUI session
    from AppKit import (
        NSApplication, NSAttributedString, NSBackingStoreBuffered, NSColor,
        NSFont, NSFontAttributeName, NSFontWeightMedium, NSForegroundColorAttributeName,
        NSGraphicsContext, NSMakeRect, NSPanel, NSScreen, NSTimer, NSView,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowCollectionBehaviorStationary,
        NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel,
    )
    from Foundation import NSData, NSObject
    from Quartz import (
        CAShapeLayer, CATransaction, CGBitmapContextCreate, CGBitmapContextCreateImage,
        CGColorSpaceCreateWithName, CGDataProviderCreateWithCFData, CGImageCreate,
        CGContextAddPath, 
        CGContextBeginTransparencyLayer, CGContextClearRect, CGContextClip,
        CGContextDrawLinearGradient, CGContextDrawRadialGradient,
        CGContextEndTransparencyLayer, CGContextFillPath, CGContextFillRect,
        CGContextReplacePathWithStrokedPath, CGContextRestoreGState,
        CGContextSaveGState, CGContextSetAlpha, CGContextSetBlendMode,
        
        CGContextSetLineWidth, CGContextSetRGBFillColor,
        CGContextSetRGBStrokeColor, CGContextStrokePath, CGColorCreateSRGB, CGGradientCreateWithColors,
        CGPathCreateWithRoundedRect, CGRectMake, CIFilter, kCGBitmapByteOrder32Little,
        kCGBlendModeDestinationIn, kCGColorSpaceSRGB, kCGImageAlphaPremultipliedFirst,
        kCGImageAlphaPremultipliedLast, kCGRenderingIntentDefault,
    )
    import objc

    # Above normal windows and full-screen apps, below the screen saver.
    _PANEL_LEVEL = 25
    _SRGB = CGColorSpaceCreateWithName(kCGColorSpaceSRGB)
    _EVERYWHERE = CGRectMake(-4000, -4000, 8000, 8000)

    # How soft the light beneath the capsule is. It is drawn at 1x and blurred
    # on the GPU, so the softness costs nothing per frame.
    UNDER_BLUR = 7.0    # the shadow and the edge light's halo, beneath the capsule

    _EDGE = tuple(float(c) for c in _SETTLED)

    def _light(toward_white: float):
        """The aurora's settled mint lifted toward white: light should glow, not paint."""
        return tuple(_lerp(c, 1.0, toward_white) for c in _EDGE)

    def canvas_size():
        return (_WIDE_WIDTH + 2.0 * _CANVAS_MARGIN, _HEIGHT + 2.0 * _CANVAS_MARGIN)

    def capsule_rect(bounds, anim):
        """Where the capsule is this frame: a dot that opens into it, and closes back."""
        target = _lerp(_COMPACT_WIDTH, _WIDE_WIDTH, anim.width.value)
        width = max(_DOT, _DOT + (target - _DOT) * anim.morph)
        width = _lerp(width, _DOT, 0.85 * _clamp01(anim.collapse.value))
        cx = bounds.size.width / 2.0
        cy = _CANVAS_MARGIN + _HEIGHT / 2.0 - anim.rise
        return CGRectMake(cx - width / 2.0, cy - _HEIGHT / 2.0, width, _HEIGHT)

    def _capsule(rect, inset=0.0):
        r = rect.size.height / 2.0 - inset
        return CGPathCreateWithRoundedRect(
            CGRectMake(rect.origin.x + inset, rect.origin.y + inset,
                       rect.size.width - 2 * inset, rect.size.height - 2 * inset), r, r, None)

    def _point_on_edge(rect, t):
        """The point a fraction t of the way round the capsule, clockwise from top-left."""
        r = rect.size.height / 2.0
        left, right = rect.origin.x + r, rect.origin.x + rect.size.width - r
        cy = rect.origin.y + r
        straight, arc = max(0.0, right - left), math.pi * r
        s = (t % 1.0) * (2.0 * straight + 2.0 * arc)
        if s < straight:
            return left + s, cy + r
        s -= straight
        if s < arc:
            a = math.pi / 2 - s / r
            return right + r * math.cos(a), cy + r * math.sin(a)
        s -= arc
        if s < straight:
            return right - s, cy - r
        a = -math.pi / 2 - (s - straight) / r
        return left + r * math.cos(a), cy + r * math.sin(a)

    def _gradient(stops):
        """
        A CGGradient from [(location, (r, g, b, a)), ...].

        Built from CGColors on purpose: through PyObjC,
        CGGradientCreateWithColorComponents misreads its arrays - a uniform
        0.86 came back as 0.86, 0, 0.31, 1.0 across the width.
        """
        return CGGradientCreateWithColors(
            _SRGB, [CGColorCreateSRGB(*rgba) for _, rgba in stops], [loc for loc, _ in stops])

    _gradients: dict = {}

    def _falloff(rgb):
        """A radial fade from rgb to nothing, cached per colour."""
        key = tuple(round(c, 3) for c in rgb)
        if key not in _gradients:
            _gradients[key] = _gradient([(0.0, (*rgb, 1.0)), (0.35, (*rgb, 0.35)), (1.0, (*rgb, 0.0))])
        return _gradients[key]

    def _clip_to_stroke(cg, path, width):
        CGContextAddPath(cg, path)
        CGContextSetLineWidth(cg, width)
        CGContextReplacePathWithStrokedPath(cg)
        CGContextClip(cg)

    def _edge_light(cg, rect, anim, width, reach, strength):
        """A band of light along the edge: steady, plus a sweep with a short tail."""
        steady, head, sweep = anim.edge_light()
        rgb = _light(0.35)
        CGContextSaveGState(cg)
        _clip_to_stroke(cg, _capsule(rect, 0.5), width)
        CGContextSetRGBFillColor(cg, rgb[0], rgb[1], rgb[2], _clamp01(steady * strength))
        CGContextFillRect(cg, _EVERYWHERE)
        if sweep > 0.01:
            for lag, weight in ((0.0, 1.0), (0.035, 0.55), (0.07, 0.25)):
                x, y = _point_on_edge(rect, head - lag)
                CGContextSetAlpha(cg, _clamp01(sweep * strength * weight))
                CGContextDrawRadialGradient(cg, _falloff(rgb), (x, y), 0.0, (x, y), reach, 0)
        CGContextRestoreGState(cg)

    def _layout(rect, anim):
        """The aurora's span and centre line, where the words begin, and the pulse."""
        mid_y = rect.origin.y + rect.size.height / 2.0
        left = rect.origin.x + _PADDING
        right = rect.origin.x + rect.size.width - _PADDING
        strip_right = _lerp(right, left + _TEXT_STRIP, anim.width.value)
        # A discarded dictation draws itself in to a point instead of fading
        # away at full width: nothing was typed, and it should look like it.
        collapse = _clamp01(anim.collapse.value)
        if collapse:
            centre = (left + strip_right) / 2.0
            left, strip_right = _lerp(left, centre, collapse), _lerp(strip_right, centre, collapse)
        pulse_x = left + anim.pulse_track * (right - left)
        return mid_y, left, strip_right, right, pulse_x

    # -- the layers, back to front -------------------------------------------------

    def draw_under(bounds, anim: HUDAnimation) -> None:
        """Beneath the capsule: a shadow for depth, and the halo of the edge light."""
        cg = NSGraphicsContext.currentContext().CGContext()
        rect = capsule_rect(bounds, anim)
        shadow = CGRectMake(rect.origin.x - 1.0, rect.origin.y - 7.0,
                            rect.size.width + 2.0, rect.size.height)
        CGContextAddPath(cg, _capsule(shadow))
        CGContextSetRGBFillColor(cg, 0.0, 0.0, 0.0, 0.34)
        CGContextFillPath(cg)
        _edge_light(cg, rect, anim, width=6.0, reach=72.0, strength=0.8)

    def draw_base(bounds, anim: HUDAnimation) -> None:
        """Crisp, beneath the aurora: the ink capsule, its rim, and the edge light."""
        cg = NSGraphicsContext.currentContext().CGContext()
        rect = capsule_rect(bounds, anim)
        top, bottom = rect.origin.y + rect.size.height, rect.origin.y

        # A dark hairline just outside, so the capsule keeps its edge on a pale wallpaper.
        CGContextAddPath(cg, _capsule(rect, -0.35))
        CGContextSetLineWidth(cg, 0.7)
        CGContextSetRGBStrokeColor(cg, 0.0, 0.0, 0.0, 0.18)
        CGContextStrokePath(cg)

        CGContextSaveGState(cg)
        CGContextAddPath(cg, _capsule(rect, 0.3))
        CGContextClip(cg)
        ink = _gradient([(0.0, (*_INK_TOP, 1.0)), (1.0, (*_INK_BOTTOM, 1.0))])
        CGContextDrawLinearGradient(cg, ink, (0, top), (0, bottom), 0)
        CGContextRestoreGState(cg)

        # The lit rim: bright along the top, a faint return along the bottom.
        CGContextSaveGState(cg)
        _clip_to_stroke(cg, _capsule(rect, 0.6), 1.0)
        rim = _gradient([(0.0, (1, 1, 1, 0.3)), (0.45, (1, 1, 1, 0.06)),
                         (0.7, (1, 1, 1, 0.04)), (1.0, (1, 1, 1, 0.11))])
        CGContextDrawLinearGradient(cg, rim, (0, top), (0, bottom), 0)
        CGContextRestoreGState(cg)

        _edge_light(cg, rect, anim, width=1.0, reach=34.0, strength=0.55)

    def draw_words(bounds, anim: HUDAnimation) -> None:
        """Crisp, above the aurora: the newest words."""
        if not anim.transcript or anim.content_alpha <= 0.01:
            return
        cg = NSGraphicsContext.currentContext().CGContext()
        rect = capsule_rect(bounds, anim)
        CGContextSaveGState(cg)
        CGContextAddPath(cg, _capsule(rect, 1.5))
        CGContextClip(cg)
        mid_y, _, strip_right, right, pulse_x = _layout(rect, anim)
        _draw_words(cg, anim, strip_right + _TEXT_GAP, right, mid_y, pulse_x)
        CGContextRestoreGState(cg)

    def aurora_frame(bounds, anim: HUDAnimation):
        """
        Where this frame's aurora image goes, padded for its glow, with the
        strip's length and the track its light travels (None when empty).
        """
        rect = capsule_rect(bounds, anim)
        _, left, strip_right, right, _ = _layout(rect, anim)
        span = strip_right - left
        if span < 2.0 or anim.content_alpha <= 0.005:
            return None, None, None
        return (NSMakeRect(left - _AURORA_PAD, rect.origin.y - _AURORA_PAD,
                           span + 2 * _AURORA_PAD, rect.size.height + 2 * _AURORA_PAD),
                span, right - left)

    def _cgimage(rgba):
        """Premultiplied RGBA bytes, rows top to bottom, as a CGImage."""
        h, w = rgba.shape[:2]
        data = NSData.dataWithBytes_length_(rgba.tobytes(), rgba.nbytes)
        return CGImageCreate(w, h, 8, 32, w * 4, _SRGB, kCGImageAlphaPremultipliedLast,
                             CGDataProviderCreateWithCFData(data), None, False,
                             kCGRenderingIntentDefault)

    _fitted: dict = {}

    def _fit(text: str, room: float):
        """The newest words that fit in room, and whether any had to go.

        Trimming whole words from the front, rather than letting the line run
        off the left edge, means a fade never cuts through the middle of a letter.
        """
        key = (text, int(room))
        if key in _fitted:
            return _fitted[key]
        attrs = {NSFontAttributeName: NSFont.systemFontOfSize_weight_(13.0, NSFontWeightMedium),
                 NSForegroundColorAttributeName: NSColor.whiteColor()}
        words, trimmed = text.split(), False
        while words:
            line = NSAttributedString.alloc().initWithString_attributes_(" ".join(words), attrs)
            if line.size().width <= room or len(words) == 1:
                break
            words, trimmed = words[1:], True
        if len(_fitted) > 64:
            _fitted.clear()
        _fitted[key] = (line, trimmed)
        return _fitted[key]

    def _draw_words(cg, anim, left, right, mid_y, pulse_x):
        """
        The newest words. Older ones fade at the left; while the correction
        runs, a shimmer travels through them with the ripples on the line.
        """
        room = right - left
        if room < 20.0:
            return
        line, trimmed = _fit(anim.transcript, room)
        size = line.size()
        m = anim.correcting
        base = _lerp(0.86, 0.5, m) * _clamp01(anim.width.value) * anim.content_alpha
        stops = []
        for k in range(25):
            u = k / 24
            px = left + u * room
            a = base + (1.0 - base) * m * math.exp(-((px - pulse_x) / 30.0) ** 2) * anim.content_alpha
            if trimmed:
                a *= _clamp01((px - left) / 30.0) ** 1.5
            stops.append((u, (1, 1, 1, _clamp01(a))))
        mask = _gradient(stops)
        CGContextSaveGState(cg)
        CGContextBeginTransparencyLayer(cg, None)
        line.drawAtPoint_((left, mid_y - size.height / 2.0 + 0.5))
        CGContextSetBlendMode(cg, kCGBlendModeDestinationIn)
        CGContextDrawLinearGradient(cg, mask, (left, 0), (right, 0), 0)
        CGContextEndTransparencyLayer(cg)
        CGContextRestoreGState(cg)

    # -- views ------------------------------------------------------------------

    class _CrispView(NSView):
        """A layer drawn at full Retina resolution by one of the draw functions."""

        def initWithFrame_draw_(self, frame, draw):
            self = objc.super(_CrispView, self).initWithFrame_(frame)
            if self is None:
                return None
            self.animation = None
            self._draw = draw
            self.setWantsLayer_(True)
            return self

        def isOpaque(self):
            return False

        def drawRect_(self, rect):
            if self.animation is not None:
                self._draw(self.bounds(), self.animation)

    class _AuroraView(NSView):
        """
        One copy of the aurora image. The glow copy is the same image, blurred
        by Core Image on the GPU and laid underneath, so the bloom is free.
        """

        def initWithBlur_opacity_(self, blur, opacity):
            self = objc.super(_AuroraView, self).initWithFrame_(NSMakeRect(0, 0, 1, 1))
            if self is None:
                return None
            self.setWantsLayer_(True)
            if blur:
                self.setLayerUsesCoreImageFilters_(True)
                softness = CIFilter.filterWithName_("CIGaussianBlur")
                softness.setDefaults()
                softness.setValue_forKey_(blur, "inputRadius")
                self.setContentFilters_([softness])
            self.layer().setOpacity_(opacity)
            return self

    class _SoftView(NSView):
        """
        A light layer drawn at 1x and blurred by Core Image on the GPU.

        Blur is the expensive part of glow, and a blur has no fine detail for
        Retina pixels to show, so these layers are drawn at a quarter of the
        pixels and softened by the compositor rather than by us.
        """

        def initWithFrame_draw_blur_(self, frame, draw, blur):
            self = objc.super(_SoftView, self).initWithFrame_(frame)
            if self is None:
                return None
            self.animation = None
            self._draw = draw
            self.setWantsLayer_(True)
            self.setLayerUsesCoreImageFilters_(True)
            softness = CIFilter.filterWithName_("CIGaussianBlur")
            softness.setDefaults()
            softness.setValue_forKey_(blur, "inputRadius")
            self.setContentFilters_([softness])
            width, height = int(frame.size.width), int(frame.size.height)
            self._bitmap = CGBitmapContextCreate(
                None, width, height, 8, 0, _SRGB,
                kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little)
            self._context = NSGraphicsContext.graphicsContextWithCGContext_flipped_(
                self._bitmap, False)
            return self

        def wantsUpdateLayer(self):
            return True

        def updateLayer(self):
            if self.animation is None:
                return
            bounds = self.bounds()
            CGContextClearRect(self._bitmap, CGRectMake(0, 0, bounds.size.width, bounds.size.height))
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.setCurrentContext_(self._context)
            self._draw(bounds, self.animation)
            NSGraphicsContext.restoreGraphicsState()
            self.layer().setContents_(CGBitmapContextCreateImage(self._bitmap))

    class _HUDController(NSObject):
        """Owns the panel and the redraw timer. Main thread only."""

        def initWithLevelSource_(self, level_source):
            self = objc.super(_HUDController, self).init()
            if self is None:
                return None
            self._level_source = level_source
            self._animation = HUDAnimation()
            self.on_failure = None     # tells the owner to hand status back to the menu bar
            self._timer = None
            self._last_tick = None
            self._panel = self._makePanel()
            return self

        @objc.python_method
        def _makePanel(self):
            # Ensure NSApp exists; rumps normally has already created it.
            NSApplication.sharedApplication()
            frame = self._canvasFrame()
            panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                frame,
                NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
                NSBackingStoreBuffered,
                False,
            )
            panel.setLevel_(_PANEL_LEVEL)
            panel.setOpaque_(False)
            panel.setBackgroundColor_(NSColor.clearColor())
            # The layers draw their own shadow and light; a window shadow would
            # outline the whole transparent canvas.
            panel.setHasShadow_(False)
            # Without this the panel disappears whenever MergeScribe isn't the
            # frontmost app - which is always, since it never takes focus.
            panel.setHidesOnDeactivate_(False)
            panel.setIgnoresMouseEvents_(True)
            panel.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorStationary
                | NSWindowCollectionBehaviorFullScreenAuxiliary
            )
            panel.setAlphaValue_(0.0)
            canvas = NSMakeRect(0, 0, frame.size.width, frame.size.height)
            container = NSView.alloc().initWithFrame_(canvas)
            container.setWantsLayer_(True)
            under = _SoftView.alloc().initWithFrame_draw_blur_(canvas, draw_under, UNDER_BLUR)
            base = _CrispView.alloc().initWithFrame_draw_(canvas, draw_base)
            words = _CrispView.alloc().initWithFrame_draw_(canvas, draw_words)
            self._canvas = canvas
            # The aurora is kept inside the capsule by a mask that follows it.
            self._inside = NSView.alloc().initWithFrame_(canvas)
            self._inside.setWantsLayer_(True)
            self._mask = CAShapeLayer.layer()
            self._mask.setFrame_(canvas)
            self._inside.layer().setMask_(self._mask)
            self._aurora_glow = _AuroraView.alloc().initWithBlur_opacity_(_AURORA_GLOW, 0.9)
            self._aurora = _AuroraView.alloc().initWithBlur_opacity_(0.0, 1.0)
            for view in (self._aurora_glow, self._aurora):
                self._inside.addSubview_(view)
            self._views = [under, base, words]
            for view in self._views:
                view.animation = self._animation
            for view in (under, base, self._inside, words):   # back to front
                container.addSubview_(view)
            panel.setContentView_(container)
            return panel

        @objc.python_method
        def _canvasFrame(self):
            screen = NSScreen.mainScreen() or NSScreen.screens()[0]
            vf = screen.visibleFrame()
            width, height = canvas_size()
            x = vf.origin.x + (vf.size.width - width) / 2.0
            y = vf.origin.y + _BOTTOM_MARGIN - _CANVAS_MARGIN
            return NSMakeRect(x, y, width, height)

        @objc.python_method
        def animation(self):
            return self._animation

        def setStatus_(self, status):
            self._animation.set_status(status)
            if self._animation.visible:
                if not self._panel.isVisible():
                    # Re-home on show: the active screen may have changed.
                    self._panel.setFrame_display_(self._canvasFrame(), False)
                self._panel.orderFrontRegardless()
                self._startTimer()
            # Hiding is left to tick_: the pill springs away, then the panel
            # is ordered out, so the HUD doesn't blink off.

        @objc.python_method
        def _startTimer(self):
            if self._timer is not None:
                return
            self._last_tick = None
            self._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                1.0 / _FPS, self, b"tick:", None, True
            )

        def tick_(self, timer):
            try:
                self._tickBody()
            except Exception as e:
                print(f"[HUD] render error, stopping: {e}")
                self._stopTimer()
                # A frozen panel must not stay up claiming to show the status.
                self._panel.orderOut_(None)
                if self.on_failure is not None:
                    self.on_failure()

        @objc.python_method
        def _tickBody(self):
            now = time.monotonic()
            dt = 1.0 / _FPS if self._last_tick is None else now - self._last_tick
            self._last_tick = now

            level = 0.0
            if self._animation.status == "recording" and self._level_source is not None:
                try:
                    level = float(self._level_source())
                except Exception:
                    level = 0.0
            self._animation.step(dt, level)

            if self._animation.hidden:
                self._panel.orderOut_(None)
                self._stopTimer()
                self._animation.reset()
                return
            self._panel.setAlphaValue_(self._animation.alpha)
            canvas = self._canvas
            capsule = capsule_rect(canvas, self._animation)
            CATransaction.begin()
            CATransaction.setDisableActions_(True)      # follow every frame, don't tween
            self._mask.setPath_(_capsule(capsule, 0.6))
            frame, span, track = aurora_frame(canvas, self._animation)
            if frame is None:
                self._aurora.layer().setContents_(None)
                self._aurora_glow.layer().setContents_(None)
            else:
                image = _cgimage(aurora_rgba(self._animation, span, _HEIGHT, _AURORA_PAD,
                                             track_pt=track))
                for view in (self._aurora_glow, self._aurora):
                    view.setFrame_(frame)
                    view.layer().setContents_(image)
            CATransaction.commit()
            for view in self._views:
                view.setNeedsDisplay_(True)

        @objc.python_method
        def _stopTimer(self):
            if self._timer is not None:
                self._timer.invalidate()
                self._timer = None

        @objc.python_method
        def tearDown(self):
            self._stopTimer()
            self._panel.orderOut_(None)

except ImportError:  # pragma: no cover - non-macOS / headless test runs
    _HUDController = None
