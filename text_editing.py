"""Text selection and replacement helpers."""

import subprocess
import time

import Quartz

from fast_text_input import type_fast


def get_clipboard():
    """Return current clipboard content."""
    try:
        result = subprocess.run(["pbpaste"], capture_output=True, text=True, check=True)
        return result.stdout
    except Exception as e:
        print(f"Error getting clipboard: {e}")
        return ""

def set_clipboard(text):
    """Set clipboard content."""
    try:
        subprocess.run("pbcopy", input=text.encode(), check=True)
    except Exception as e:
        print(f"Error setting clipboard: {e}")

def detect_selected_text():
    """Copy currently selected text and return it."""
    try:
        original_clipboard = get_clipboard()
        try:
            c_key_code = 8
            key_down_event = Quartz.CGEventCreateKeyboardEvent(None, c_key_code, True)
            Quartz.CGEventSetFlags(key_down_event, Quartz.kCGEventFlagMaskCommand)
            key_up_event = Quartz.CGEventCreateKeyboardEvent(None, c_key_code, False)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, key_down_event)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, key_up_event)
            time.sleep(0.05)
        except Exception as e:
            print(f"Error copying selected text: {e}")
            return None
        new_clipboard = get_clipboard()
        if new_clipboard != original_clipboard and new_clipboard.strip():
            return new_clipboard
        else:
            return None
    except Exception as e:
        print(f"Error in text selection: {e}")
        return None

def replace_selected_text(new_text):
    """Replace currently selected text with `new_text`."""
    try:
        type_fast(new_text)
        print(f"✅ Replaced text with: \"{new_text[:50]}{'...' if len(new_text) > 50 else ''}\"")
        return True
    except Exception as e:
        print(f"Error replacing text: {e}")
        return False