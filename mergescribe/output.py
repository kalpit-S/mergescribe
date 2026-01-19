"""
Output functions for typing text, clipboard, and notifications.

Uses macOS accessibility APIs and system commands.
"""

import subprocess
import time
from typing import Optional


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

    Uses macOS accessibility API via osascript.
    Should be called from a thread that can access the main run loop,
    or use dispatch to main thread.

    Args:
        text: Text to type
    """
    if not text:
        return

    try:
        escaped = _escape_for_applescript(text)

        script = f'''
        tell application "System Events"
            keystroke "{escaped}"
        end tell
        '''

        # Use stdin instead of -e to avoid ARG_MAX limits for long text
        subprocess.run(
            ["osascript"],
            input=script.encode("utf-8"),
            capture_output=True,
            timeout=10.0
        )
    except subprocess.TimeoutExpired:
        print("type_text: osascript timed out")
    except Exception as e:
        print(f"type_text error: {e}")


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


def get_clipboard() -> str:
    """
    Get text from the system clipboard.

    Returns:
        Clipboard contents as string
    """
    try:
        result = subprocess.run(
            ["pbpaste"],
            capture_output=True,
            timeout=2.0
        )
        return result.stdout.decode("utf-8")
    except Exception as e:
        print(f"get_clipboard error: {e}")
        return ""


def replace_selection(text: str) -> None:
    """
    Replace the currently selected text.

    Saves clipboard, pastes new text, restores clipboard.

    Args:
        text: Text to replace selection with
    """
    if not text:
        return

    try:
        # Save current clipboard
        old_clipboard = get_clipboard()

        # Copy new text to clipboard
        copy_to_clipboard(text)

        # Paste (Cmd+V)
        script = '''
        tell application "System Events"
            keystroke "v" using command down
        end tell
        '''
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=2.0)

        # Wait a bit for paste to complete
        time.sleep(0.1)

        # Restore old clipboard
        copy_to_clipboard(old_clipboard)

    except Exception as e:
        print(f"replace_selection error: {e}")


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
    Play a system sound.

    Args:
        sound_name: Name of sound in /System/Library/Sounds/
    """
    try:
        subprocess.run(
            ["afplay", f"/System/Library/Sounds/{sound_name}.aiff"],
            capture_output=True,
            timeout=2.0
        )
    except Exception as e:
        print(f"play_sound error: {e}")


def play_busy_sound() -> None:
    """Play a sound indicating the system is busy."""
    play_sound("Basso")
