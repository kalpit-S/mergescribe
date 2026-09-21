"""
Application context detection via macOS APIs.

Detects the active application and window title to provide
context-aware transcription correction.
"""

import subprocess
import time
from typing import Optional, Tuple

from .types import AppContext


# Cache for get_app_context() - avoids repeated osascript calls
_context_cache: Tuple[float, Optional[AppContext]] = (0.0, None)
_CONTEXT_CACHE_TTL = 0.3  # 300ms TTL - long enough to avoid repeated calls, short enough to detect window changes


def _native_context():
    """Frontmost app via NSWorkspace/AX — ~0.4ms vs ~700ms for osascript.

    osascript's "first application process whose frontmost is true" walks every
    process through Apple Events and was measured at 713ms median / 1.3s worst
    case, all of it blocking the start of recording.
    """
    try:
        from AppKit import NSWorkspace
        from ApplicationServices import (
            AXUIElementCreateApplication, AXUIElementCopyAttributeValue,
        )
    except ImportError:
        return None

    try:
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return None
        name = str(app.localizedName() or "")
        bundle = str(app.bundleIdentifier() or "")

        def ax(el, attr):
            try:
                err, value = AXUIElementCopyAttributeValue(el, attr, None)
                return value if err == 0 else None
            except Exception:
                return None

        title = ""
        app_el = AXUIElementCreateApplication(app.processIdentifier())
        for attr in ("AXFocusedWindow", "AXMainWindow"):
            window = ax(app_el, attr)
            if window is not None:
                title = str(ax(window, "AXTitle") or "")
                if title:
                    break
        return (name, bundle, title)
    except Exception:
        return None


def get_app_context() -> AppContext:
    """
    Get active application info via macOS APIs.

    Uses osascript to query the frontmost application.
    Results are cached for 300ms to avoid repeated calls.

    Returns:
        AppContext with app name, window title, and bundle ID
    """
    global _context_cache

    # Check cache first
    cache_time, cached_context = _context_cache
    if cached_context is not None and (time.time() - cache_time) < _CONTEXT_CACHE_TTL:
        return cached_context

    app_name = ""
    window_title = ""
    bundle_id = ""

    native = _native_context()
    if native is not None:
        app_name, bundle_id, window_title = native
        context = AppContext(app_name=app_name, window_title=window_title, bundle_id=bundle_id)
        _context_cache = (time.time(), context)
        return context

    # Fallback: osascript (slow, but works if the AX/AppKit path is unavailable)
    try:
        # Get frontmost app info via AppleScript
        script = '''
        tell application "System Events"
            set frontApp to first application process whose frontmost is true
            set appName to name of frontApp
            set bundleId to bundle identifier of frontApp

            try
                set windowTitle to name of front window of frontApp
            on error
                set windowTitle to ""
            end try

            return appName & "|||" & bundleId & "|||" & windowTitle
        end tell
        '''

        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=2.0
        )

        if result.returncode == 0:
            parts = result.stdout.strip().split("|||")
            if len(parts) >= 3:
                app_name = parts[0]
                bundle_id = parts[1]
                window_title = parts[2]

    except subprocess.TimeoutExpired:
        print("get_app_context: osascript timed out")
    except Exception as e:
        print(f"get_app_context error: {e}")

    context = AppContext(
        app_name=app_name,
        window_title=window_title,
        bundle_id=bundle_id,
    )

    # Cache result
    _context_cache = (time.time(), context)

    return context


def detect_selected_text() -> Optional[str]:
    """
    Detect and return currently selected text.

    Works by simulating Cmd+C, reading clipboard, then restoring original.
    Returns None if no text is selected.
    """
    try:
        import Quartz
    except ImportError:
        print("Quartz not available - text editing disabled")
        return None

    original = None
    try:
        # Save original clipboard
        original = _get_clipboard()

        # Simulate Cmd+C to copy selection
        c_key_code = 8  # 'c' key
        key_down = Quartz.CGEventCreateKeyboardEvent(None, c_key_code, True)
        Quartz.CGEventSetFlags(key_down, Quartz.kCGEventFlagMaskCommand)
        key_up = Quartz.CGEventCreateKeyboardEvent(None, c_key_code, False)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, key_down)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, key_up)
        time.sleep(0.05)

        # Read new clipboard
        new_clipboard = _get_clipboard()

        # Check if we got a selection
        if new_clipboard != original and new_clipboard and new_clipboard.strip():
            return new_clipboard

        return None

    except Exception as e:
        print(f"detect_selected_text error: {e}")
        return None
    finally:
        # Always restore original clipboard
        if original is not None:
            _set_clipboard(original)


def _get_clipboard() -> str:
    """Get current clipboard content."""
    try:
        result = subprocess.run(["pbpaste"], capture_output=True, text=True, check=True)
        return result.stdout
    except Exception:
        return ""


def _set_clipboard(text: str) -> None:
    """Set clipboard content."""
    try:
        subprocess.run("pbcopy", input=text.encode(), check=True)
    except Exception:
        pass
