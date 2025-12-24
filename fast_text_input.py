"""Fast text input using macOS Core Graphics Events."""
import Quartz
import time
import subprocess
from typing import Optional

MAX_CHUNK = 20  # CGEvent truncates beyond ~20 UTF-16 units

def get_clipboard() -> str:
    """Return current clipboard content (best-effort)."""
    try:
        result = subprocess.run(["pbpaste"], capture_output=True, text=True, check=True)
        return result.stdout
    except Exception:
        return ""

def set_clipboard(text: str) -> bool:
    """Set clipboard content (best-effort). Returns True if it likely worked."""
    try:
        subprocess.run("pbcopy", input=text.encode("utf-8"), check=True)
        return True
    except Exception:
        return False

def copy_result_to_clipboard(text: str) -> None:
    """Convenience wrapper for copying transcription results to clipboard."""
    if not text:
        return
    ok = set_clipboard(text)
    if not ok:
        # Don't raise: clipboard is a convenience.
        print("[WARN] Could not copy result to clipboard (pbcopy failed).")

def _best_event_tap() -> int:
    """Pick the most appropriate event tap for posting synthetic key events."""
    # Prefer session-level taps for app delivery; fall back to HID if needed.
    return (
        getattr(Quartz, "kCGAnnotatedSessionEventTap", None)
        or getattr(Quartz, "kCGSessionEventTap", None)
        or Quartz.kCGHIDEventTap
    )

def _send_text_chunk(chunk: str) -> None:
    """Post key-down and key-up CGEvents carrying `chunk` as Unicode payload."""
    # Use an explicit event source; helps with consistency across apps.
    source: Optional[object]
    try:
        source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    except Exception:
        source = None

    tap = _best_event_tap()

    # Key down
    evt_down = Quartz.CGEventCreateKeyboardEvent(source, 0, True)
    # Ensure we don't inherit any modifier state.
    try:
        Quartz.CGEventSetFlags(evt_down, 0)
    except Exception:
        pass
    Quartz.CGEventKeyboardSetUnicodeString(evt_down, _count_utf16_units(chunk), chunk)
    Quartz.CGEventPost(tap, evt_down)
    
    # Small delay between down and up
    time.sleep(0.001)  # 1ms
    
    # Key up
    evt_up = Quartz.CGEventCreateKeyboardEvent(source, 0, False)
    try:
        Quartz.CGEventSetFlags(evt_up, 0)
    except Exception:
        pass
    # Key-up event doesn't need the Unicode payload; keep it empty to avoid
    # confusing apps that interpret Unicode on key-up oddly.
    Quartz.CGEventKeyboardSetUnicodeString(evt_up, 0, "")
    Quartz.CGEventPost(tap, evt_up)

def _count_utf16_units(text: str) -> int:
    """Count the number of UTF-16 code units in a string."""
    # In Python 3, strings are Unicode. Emojis and other characters
    # outside the BMP (Basic Multilingual Plane) use surrogate pairs
    # in UTF-16, taking up 2 units instead of 1.
    return len(text.encode('utf-16-le')) // 2

def _chunk_by_utf16(text: str, max_units: int):
    """Yield chunks of text, each with at most max_units UTF-16 code units."""
    chunk = ""
    chunk_units = 0
    
    for char in text:
        char_units = _count_utf16_units(char)
        
        # If adding this character would exceed the limit, yield current chunk
        if chunk_units + char_units > max_units:
            if chunk:  # Don't yield empty chunks
                yield chunk
            chunk = char
            chunk_units = char_units
        else:
            chunk += char
            chunk_units += char_units
    
    # Yield remaining chunk
    if chunk:
        yield chunk

def type_fast(text: str) -> None:
    """Type text using CGEvent (up to 20 UTF-16 units per event)."""
    try:
        for chunk in _chunk_by_utf16(text, MAX_CHUNK):
            _send_text_chunk(chunk)
            time.sleep(0.010)  # 10ms delay between chunks to prevent overlap
    except Exception as e:
        print(f"Fast text input error: {e}")
        # Fallback: pyautogui typing (does not touch clipboard).
        try:
            import pyautogui

            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0.01
            pyautogui.write(text)
        except Exception as e2:
            print(f"Fast text input fallback error: {e2}")