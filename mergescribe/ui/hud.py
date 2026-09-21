"""
Floating recording HUD.

The menu bar already showed state, but it sits at the top of the screen while
your eyes are on the field you're dictating into, so in practice there was no
feedback at all: press, talk, release, then ~1.2s of nothing before text
appears. This puts the same state where you are actually looking.

The animation follows what the app does. While you talk, three strands of
light move with your voice, the way several recognizers listen at once. When
you let go they merge into a single line, and a pulse runs along it and on
through your words while the correction model works.

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

# Geometry. The pill stays compact until there is transcript to show: 65% of
# sessions are a single chunk, so nothing is ever available before release and
# they should keep the small unobtrusive form.
_COMPACT_WIDTH = 176.0
_WIDE_WIDTH = 460.0
_HEIGHT = 44.0
_BOTTOM_MARGIN = 130.0
_PADDING = 18.0
_TEXT_STRIP = 88.0      # how much of the pill the strands keep once words arrive
_TEXT_GAP = 12.0
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
# running. Each gets a colour, a spatial frequency (cycles across the meter)
# and a speed (radians per second); different speeds keep them out of step.
_PALETTE = (
    ((0.30, 0.78, 1.00), 1.5, 5.3),
    ((0.60, 0.46, 1.00), 2.0, -4.1),
    ((1.00, 0.43, 0.66), 1.2, 6.4),
    ((1.00, 0.70, 0.32), 1.8, -5.6),
)
_DEFAULT_STREAMS = 3       # until a session says how many it runs
_STRAND_SAMPLES = 44
_STRAND_AMPLITUDE = 15.0   # px at full voice; the pill is 44 tall
_STRAND_REST = 0.08        # share left at silence, so it breathes rather than dies
_STRAND_ALPHA = 0.55       # strands add up where they overlap
_STRAND_DIMMED = 0.35      # missed a chunk while recording; may yet catch up
_WORK_AMPLITUDE = 7.0      # after release, a stream still transcribing
_FLASH_DECAY = 0.3         # seconds for a landing's flash to fade

# Joined: streams that have returned fall into one line. Once correction
# starts, a pulse travels along it and on through the words.
_MERGED_COLOR = (0.88, 0.93, 1.00)
_MERGED_FREQUENCY = 3.0
_MERGED_SPEED = 11.0
_JOINED_AMPLITUDE = 3.0
_PULSE_AMPLITUDE = 6.0
_PULSE_WIDTH = 0.16        # fraction of the strands' length
_PULSE_PERIOD = 1.5        # seconds per pass across the pill
_PULSE_LEAD_IN = 0.25      # lets the last strand arrive before the pulse sets off

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
    """One stream's line of light."""

    __slots__ = ("key", "color", "frequency", "speed", "phase", "join", "presence", "flash")

    def __init__(self, key: str, index: int):
        self.key = key
        self.color, self.frequency, self.speed = _PALETTE[index % len(_PALETTE)]
        self.phase = index * 2.1
        self.join = Spring(0.0, _MERGE_SPRING)       # 0 its own wave, 1 part of the line
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
        if self._correcting:
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
    def scale(self) -> float:
        # Unclamped on purpose: the spring's overshoot is the "pop".
        return 0.84 + 0.16 * self.presence.value

    @property
    def rise(self) -> float:
        """How far below its resting place the pill is, in points."""
        return (1.0 - min(self.presence.value, 1.0)) * 10.0

    @property
    def correcting(self) -> float:
        """0..1, how far into the correcting look the pill is."""
        return _clamp01(self.pulse.value)

    def glow_layers(self) -> List[Tuple[Tuple[float, float, float], float]]:
        """(colour, strength) of the light under the pill, one per present strand."""
        strength = _lerp(_lerp(0.14 + 0.55 * self.level, 0.22, _clamp01(self.work.value)),
                         0.30, self.correcting)
        return [(s.color, strength * _clamp01(s.presence.value)) for s in self.strands]

    @property
    def pulse_position(self) -> float:
        """Where the correcting pulse is, as a fraction of the pill's content.

        Starts off the left edge and exits off the right, so each pass fades
        in and out rather than appearing at a point.
        """
        progress = (max(self._pulse_clock, 0.0) / _PULSE_PERIOD) % 1.0
        return -0.2 + 1.4 * progress

    def strand_shapes(self, pulse_u: float = 0.5
                      ) -> List[Tuple[Tuple[float, float, float], float, List[float]]]:
        """
        Each strand as (colour, opacity, vertical offsets at evenly spaced samples).

        pulse_u is the pulse position in the strands' own 0..1 coordinates.
        The view mirrors each offset about the centre line, so a strand is
        a chain of lobes that pinch where its wave crosses zero.
        """
        work, pulse = _clamp01(self.work.value), self.correcting
        voice = _STRAND_REST + (1.0 - _STRAND_REST) * self.level
        joined = _lerp(_JOINED_AMPLITUDE, _PULSE_AMPLITUDE, pulse)
        shapes = []
        for i, strand in enumerate(self.strands):
            m = _clamp01(strand.join.value)
            present = _clamp01(strand.presence.value)
            # Slow independent wander, so no two strands swell together.
            wander = 0.72 + 0.28 * math.sin(self.clock * (1.1 + 0.6 * i) + 2.0 * i)
            apart = _lerp(_STRAND_AMPLITUDE * voice, _WORK_AMPLITUDE, work) * wander
            amplitude = (_lerp(apart, joined, m) * (0.4 + 0.6 * present)
                         * (1.0 + 0.5 * strand.flash))
            freq = _lerp(strand.frequency, _MERGED_FREQUENCY, m)
            offsets = []
            for k in range(_STRAND_SAMPLES + 1):
                u = k / _STRAND_SAMPLES
                taper = (4.0 * u * (1.0 - u)) ** 2
                travelling = math.exp(-((u - pulse_u) / _PULSE_WIDTH) ** 2) * taper ** 0.25
                envelope = _lerp(taper, _lerp(taper, travelling, pulse), m)
                offsets.append(amplitude * envelope
                               * math.sin(2.0 * math.pi * freq * u + strand.phase))
            tint = tuple(_lerp(c, w, 0.55 * m) for c, w in zip(strand.color, _MERGED_COLOR))
            opacity = min(1.0, _STRAND_ALPHA * present * (1.0 + 0.8 * strand.flash))
            shapes.append((tint, opacity, offsets))
        return shapes


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
            return self._controller

    def _teardown(self) -> None:
        try:
            self._controller.tearDown()
        except Exception:
            pass
        self._controller = None


try:  # pragma: no cover - requires a macOS GUI session
    from AppKit import (
        NSApplication, NSBackingStoreBuffered, NSBezierPath, NSColor, NSColorSpace,
        NSCompositingOperationPlusLighter, NSGradient, NSGraphicsContext, NSMakeRect,
        NSPanel, NSScreen, NSShadow, NSTimer, NSView,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowCollectionBehaviorStationary,
        NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel,
        NSAttributedString, NSFont, NSFontAttributeName,
        NSForegroundColorAttributeName,
    )
    from Foundation import NSObject
    from Quartz import (
        CGBitmapContextCreate, CGBitmapContextCreateImage, CGColorSpaceCreateWithName,
        CGContextBeginTransparencyLayer, CGContextClearRect, CGContextEndTransparencyLayer,
        CGContextSetBlendMode, CGRectMake, kCGBitmapByteOrder32Little, kCGBlendModeDestinationIn,
        kCGColorSpaceSRGB, kCGImageAlphaPremultipliedFirst,
    )
    import objc

    # Above normal windows and full-screen apps, below the screen saver.
    _PANEL_LEVEL = 25

    def _rgba(rgb, alpha):
        return NSColor.colorWithSRGBRed_green_blue_alpha_(rgb[0], rgb[1], rgb[2], alpha)

    def _white(alpha):
        return NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, alpha)

    _BODY = (0.055, 0.058, 0.07)

    def canvas_size():
        return (_WIDE_WIDTH + 2.0 * _CANVAS_MARGIN, _HEIGHT + 2.0 * _CANVAS_MARGIN)

    def _pill(bounds, anim):
        """The pill's rect this frame, and its outline."""
        scale = anim.scale
        width = _lerp(_COMPACT_WIDTH, _WIDE_WIDTH, anim.width.value) * scale
        height = _HEIGHT * scale
        cx = bounds.size.width / 2.0
        cy = _CANVAS_MARGIN + _HEIGHT / 2.0 - anim.rise
        rect = NSMakeRect(cx - width / 2.0, cy - height / 2.0, width, height)
        return rect, NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            rect, height / 2.0, height / 2.0)

    def draw_glow(bounds, anim: HUDAnimation) -> None:
        """
        What sits under the pill: a drop shadow for depth, then light in the
        strands' colours drifting beneath it, brighter as you speak.
        """
        rect, path = _pill(bounds, anim)
        drift = min(20.0, rect.size.width * 0.08)
        layers = [((0.0, 0.0, 0.0), 0.45, 14.0, (0.0, -4.0))]
        layers += [(color, strength, 16.0, (drift * math.sin(anim.clock * 0.9 + i * 2.1), 0.0))
                   for i, (color, strength) in enumerate(anim.glow_layers())]
        for rgb, strength, blur, offset in layers:
            NSGraphicsContext.saveGraphicsState()
            shadow = NSShadow.alloc().init()
            shadow.setShadowColor_(_rgba(rgb, _clamp01(strength)))
            shadow.setShadowBlurRadius_(blur)
            shadow.setShadowOffset_(offset)
            shadow.set()
            _rgba(_BODY, 1.0).setFill()
            path.fill()
            NSGraphicsContext.restoreGraphicsState()

    def draw_hud(bounds, anim: HUDAnimation) -> None:
        """The pill itself: body, strands and words."""
        rect, path = _pill(bounds, anim)
        _rgba(_BODY, 0.94).setFill()
        path.fill()
        _white(0.13).setStroke()
        path.setLineWidth_(0.8)
        path.stroke()

        NSGraphicsContext.saveGraphicsState()
        path.addClip()
        mid_y = rect.origin.y + rect.size.height / 2.0
        left = rect.origin.x + _PADDING
        right = rect.origin.x + rect.size.width - _PADDING
        strip_right = _lerp(right, left + _TEXT_STRIP, anim.width.value)
        pulse_x = left + anim.pulse_position * (right - left)
        pulse_u = (pulse_x - left) / max(1.0, strip_right - left)
        _draw_strands(anim, left, strip_right, mid_y, pulse_u)
        if anim.transcript:
            _draw_text(anim, strip_right + _TEXT_GAP, right, mid_y, pulse_x)
        NSGraphicsContext.restoreGraphicsState()

    def _draw_strands(anim, left, right, mid_y, pulse_u):
        span = right - left
        _white(0.14).setFill()
        NSBezierPath.fillRect_(NSMakeRect(left, mid_y - 0.5, span, 1.0))

        context = NSGraphicsContext.currentContext()
        context.saveGraphicsState()
        # Additive: where strands cross, their colours sum toward white.
        context.setCompositingOperation_(NSCompositingOperationPlusLighter)
        for color, opacity, offsets in anim.strand_shapes(pulse_u):
            last = len(offsets) - 1
            shape = NSBezierPath.bezierPath()
            shape.moveToPoint_((left, mid_y))
            for s, y in enumerate(offsets):
                shape.lineToPoint_((left + span * s / last, mid_y + y))
            for s in range(last, -1, -1):
                shape.lineToPoint_((left + span * s / last, mid_y - offsets[s]))
            shape.closePath()
            _rgba(color, opacity).setFill()
            shape.fill()
        context.restoreGraphicsState()

    def _draw_text(anim, left, right, mid_y, pulse_x):
        """
        The newest words, right-aligned when they overflow so the oldest are
        the ones that fade out. While correcting, a shimmer runs through them.
        """
        attrs = {
            NSFontAttributeName: NSFont.systemFontOfSize_(12.0),
            NSForegroundColorAttributeName: _white(1.0),
        }
        text = NSAttributedString.alloc().initWithString_attributes_(anim.transcript, attrs)
        size = text.size()
        room = right - left
        x = left if size.width <= room else right - size.width
        box = NSMakeRect(left, mid_y - size.height / 2.0, room, size.height)

        m = anim.correcting
        base = _lerp(0.78, 0.45, m) * _clamp01(anim.width.value)
        overflow = size.width > room
        stops = 24
        colors, locations = [], []
        for k in range(stops + 1):
            u = k / stops
            px = left + u * room
            alpha = base + (1.0 - base) * m * math.exp(-((px - pulse_x) / 34.0) ** 2)
            if overflow:
                alpha *= _clamp01((px - left) / 28.0)
            colors.append(_white(_clamp01(alpha)))
            locations.append(u)
        mask = NSGradient.alloc().initWithColors_atLocations_colorSpace_(
            colors, locations, NSColorSpace.sRGBColorSpace())

        context = NSGraphicsContext.currentContext()
        cg = context.CGContext()
        context.saveGraphicsState()
        NSBezierPath.clipRect_(box)
        CGContextBeginTransparencyLayer(cg, None)
        text.drawAtPoint_((x, box.origin.y))
        CGContextSetBlendMode(cg, kCGBlendModeDestinationIn)
        mask.drawInRect_angle_(box, 0.0)
        CGContextEndTransparencyLayer(cg)
        context.restoreGraphicsState()

    class _HUDView(NSView):
        """Draws whatever state its HUDAnimation is in."""

        def initWithFrame_(self, frame):
            self = objc.super(_HUDView, self).initWithFrame_(frame)
            if self is None:
                return None
            self.animation = None
            return self

        def isOpaque(self):
            return False

        def drawRect_(self, rect):
            if self.animation is not None:
                draw_hud(self.bounds(), self.animation)

    class _GlowView(NSView):
        """
        The glow, rendered at 1x and handed to the layer as an image.

        Blur is most of the cost of a frame, and a blur has no fine detail for
        Retina pixels to show, so it is drawn at a quarter of the pixels and
        the GPU scales it up.
        """

        def initWithFrame_(self, frame):
            self = objc.super(_GlowView, self).initWithFrame_(frame)
            if self is None:
                return None
            self.animation = None
            self.setWantsLayer_(True)
            width, height = int(frame.size.width), int(frame.size.height)
            self._bitmap = CGBitmapContextCreate(
                None, width, height, 8, 0, CGColorSpaceCreateWithName(kCGColorSpaceSRGB),
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
            draw_glow(bounds, self.animation)
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
            # The view draws its own shadow and glow; a window shadow would
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
            self._views = []
            for cls in (_GlowView, _HUDView):   # back to front
                view = cls.alloc().initWithFrame_(canvas)
                view.setWantsLayer_(True)
                view.animation = self._animation
                container.addSubview_(view)
                self._views.append(view)
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
