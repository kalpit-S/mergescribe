"""
Output functions for typing text, clipboard, and notifications.

Uses macOS accessibility APIs and system commands.
"""

import re
import subprocess
import time

try:
    import Quartz
    _QUARTZ_AVAILABLE = True
except ImportError:
    _QUARTZ_AVAILABLE = False


# CGEventKeyboardSetUnicodeString drops characters when handed too much at
# once; 20 chars per event stays comfortably inside the limit.
_TYPE_CHUNK_CHARS = 20
_TYPE_CHUNK_DELAY = 0.002

_LINE_BREAKS = re.compile(r"[\r\n]+")


NOTHING_MARKER = "[nothing]"


def is_nothing(text: str) -> bool:
    """True when a finished correction is the model's "type nothing" marker."""
    return (text or "").strip().lower().startswith(NOTHING_MARKER)


class NothingMarker:
    """
    Recognizes the correction model saying "type nothing", without delaying text.

    The model is told to reply with exactly NOTHING_MARKER when the speaker
    called off the whole dictation. An empty reply can't carry that meaning:
    empty is how a failed call looks, and a failure falls back to typing the
    raw transcript, so a flaky empty answer must not silently lose a dictation.

    Output streams, so the marker would be typed before it was complete. Text
    is held back only while it could still be the marker; anything that does
    not start with "[" passes through on the first token.
    """

    def __init__(self) -> None:
        self._held = ""
        self._decided = False
        self.called_off = False

    def feed(self, text: str) -> str:
        if self._decided:
            return "" if self.called_off else text
        self._held += text
        probe = self._held.lstrip().lower()
        if probe.startswith(NOTHING_MARKER):
            self._decided = self.called_off = True
            return ""
        if NOTHING_MARKER.startswith(probe):
            return ""   # still could be the marker
        self._decided = True
        released, self._held = self._held, ""
        return released

    def flush(self) -> str:
        """Release anything held back by a reply that ended mid-way through a prefix."""
        if self._decided:
            return ""
        self._decided = True
        released, self._held = self._held, ""
        return released


class DictationFilter:
    """
    Flattens the line breaks a correction model invents in dictated speech.

    Typing a newline through CGEvent delivers a Return keystroke, and in Slack,
    ChatGPT and most chat composers Return *sends*. A paragraph break the model
    added would submit half the dictation and type the remainder into an empty
    box. Speech has no paragraphs, and 14% of measured outputs contained one.

    Stateful because correction streams token by token: a break can straddle a
    token boundary, and the space standing in for it must not double up with a
    space the next token already starts with.
    """

    def __init__(self, continuing: bool = False) -> None:
        # At the start of a field a leading space is noise, so it is swallowed.
        # When appending to a dictation already in the field it is the
        # separator, so it must survive.
        self._ends_with_space = not continuing

    def feed(self, text: str) -> str:
        """Return the text to actually type for this token."""
        if not text:
            return ""
        flattened = _LINE_BREAKS.sub(" ", text)
        if self._ends_with_space:
            flattened = flattened.lstrip(" ")
        if not flattened:
            return ""
        # Collapse runs created by joining a break to neighbouring whitespace.
        flattened = re.sub(r"  +", " ", flattened)
        self._ends_with_space = flattened.endswith(" ")
        return flattened



def _escape_for_applescript(text: str) -> str:
    """Escape special characters for AppleScript string."""
    # Order matters: backslash first
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    text = text.replace("\r", "\\r")
    text = text.replace("\n", "\\n")
    text = text.replace("\t", "\\t")
    return text


def type_text(text: str) -> None:
    """
    Type text at the current cursor position.

    Uses CGEvent unicode injection (no subprocess). This is called once per
    streamed token, so the osascript path's ~35ms process spawn was pure
    latency on the critical path; CGEvent costs microseconds.

    Args:
        text: Text to type
    """
    if not text:
        return
    if not _QUARTZ_AVAILABLE:
        _hand_over(text, "type_text: Quartz is unavailable")
        return
    _type_text_quartz(text)


def _hand_over(rest: str, reason: str) -> None:
    """
    What couldn't be typed goes to the clipboard. Retyping the whole text by
    another route (this used to fall back to osascript) would repeat the part
    that already went in.
    """
    print(f"{reason}; the rest is on the clipboard")
    copy_to_clipboard(rest)
    notify("Couldn't type everything - the rest is on the clipboard")


def _type_text_quartz(text: str) -> None:
    """Type via synthetic unicode key events."""
    for start in range(0, len(text), _TYPE_CHUNK_CHARS):
        chunk = text[start:start + _TYPE_CHUNK_CHARS]
        try:
            for is_key_down in (True, False):
                event = Quartz.CGEventCreateKeyboardEvent(None, 0, is_key_down)
                # UTF-16 units, not characters: an emoji is two, and a count
                # of one types half of it.
                Quartz.CGEventKeyboardSetUnicodeString(event, len(chunk.encode("utf-16-le")) // 2, chunk)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        except Exception as e:
            _hand_over(text[start:], f"type_text: key events failed ({e})")
            return
        if start + _TYPE_CHUNK_CHARS < len(text):
            time.sleep(_TYPE_CHUNK_DELAY)


def copy_to_clipboard(text: str) -> None:
    """
    Copy text to the system clipboard.

    Args:
        text: Text to copy
    """
    if not text:
        return

    try:
        subprocess.run(
            ["pbcopy"],
            input=text.encode("utf-8"),
            timeout=2.0
        )
    except Exception as e:
        print(f"copy_to_clipboard error: {e}")


def notify(message: str, title: str = "MergeScribe") -> None:
    """
    Show a macOS notification.

    Args:
        message: Notification body
        title: Notification title
    """
    try:
        escaped_message = _escape_for_applescript(message)
        escaped_title = _escape_for_applescript(title)
        script = f'''
        display notification "{escaped_message}" with title "{escaped_title}"
        '''
        subprocess.run(
            ["osascript"],
            input=script.encode("utf-8"),
            capture_output=True,
            timeout=2.0
        )
    except Exception as e:
        print(f"notify error: {e}")


def play_sound(sound_name: str = "Tink") -> None:
    """
    Play a system sound. Fire-and-forget: callers are on latency-sensitive
    paths and must not block for the length of the sound.

    Args:
        sound_name: Name of sound in /System/Library/Sounds/
    """
    try:
        subprocess.Popen(
            ["afplay", f"/System/Library/Sounds/{sound_name}.aiff"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"play_sound error: {e}")


def play_busy_sound() -> None:
    """Play a sound indicating the system is busy."""
    play_sound("Basso")
