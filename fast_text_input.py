"""Fast text input using macOS Core Graphics Events."""
import Quartz
import time

MAX_CHUNK = 20  # CGEvent truncates beyond ~20 UTF-16 units

def _send_text_chunk(chunk: str) -> None:
    """Post key-down and key-up CGEvents carrying `chunk` as Unicode payload."""
    # Key down
    evt_down = Quartz.CGEventCreateKeyboardEvent(None, 0, True)
    Quartz.CGEventKeyboardSetUnicodeString(evt_down, len(chunk), chunk)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt_down)
    
    # Small delay between down and up
    time.sleep(0.001)  # 1ms
    
    # Key up
    evt_up = Quartz.CGEventCreateKeyboardEvent(None, 0, False)
    Quartz.CGEventKeyboardSetUnicodeString(evt_up, len(chunk), chunk)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt_up)

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
        # Fallback to pyautogui if CGEvent fails
        import pyautogui
        pyautogui.write(text)