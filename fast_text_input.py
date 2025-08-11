"""Fast text input using macOS Core Graphics Events."""
import Quartz
import time

MAX_CHUNK = 20  # CGEvent truncates beyond ~20 UTF-16 units

def _key_down(chunk: str) -> None:
    """Post a key-down CGEvent carrying `chunk` as its Unicode payload."""
    evt = Quartz.CGEventCreateKeyboardEvent(None, 0, True)
    Quartz.CGEventKeyboardSetUnicodeString(evt, len(chunk), chunk)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)

def type_fast(text: str) -> None:
    """Type text using CGEvent (up to 20 UTF-16 units per event)."""
    try:
        for i in range(0, len(text), MAX_CHUNK):
            chunk = text[i : i + MAX_CHUNK]
            _key_down(chunk)
            time.sleep(0.005)  # 5ms delay to prevent overlap
    except Exception as e:
        print(f"Fast text input error: {e}")
        # Fallback to pyautogui if CGEvent fails
        import pyautogui
        pyautogui.write(text)