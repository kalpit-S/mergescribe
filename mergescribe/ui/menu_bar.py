"""
macOS menu bar application using rumps.

Provides status icon and menu for MergeScribe.
"""

from typing import Optional, Callable

import rumps

# SF Symbols, drawn as template images so they take the menu bar's own colour
# in light and dark mode. With the HUD on, the bar only ever shows "idle";
# the others are for anyone who switches the HUD off.
_SYMBOLS = {
    "idle": "waveform",
    "recording": "waveform.circle.fill",
    "processing": "ellipsis.circle",
    "error": "exclamationmark.triangle",
}
# Where SF Symbols aren't available (before macOS 11), plain text still works.
_FALLBACK = {"idle": "🎤", "recording": "🔴", "processing": "⚡", "error": "❌"}


def _symbol(name: str):
    """A template NSImage for an SF Symbol, sized for the menu bar, or None."""
    try:
        from AppKit import NSFontWeightRegular, NSImage, NSImageSymbolConfiguration
        image = NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, "MergeScribe")
        if image is None:
            return None
        image = image.imageWithSymbolConfiguration_(
            NSImageSymbolConfiguration.configurationWithPointSize_weight_(14.0, NSFontWeightRegular))
        image.setTemplate_(True)
        return image
    except Exception:
        return None


class MenuBarApp:
    """
    Menu bar application for MergeScribe.

    Shows status icon and provides menu with:
    - Status indicator (idle/recording/processing)
    - Settings
    - Quit
    """

    def __init__(self):
        self.on_settings: Optional[Callable[[], None]] = None
        self.on_quit: Optional[Callable[[], None]] = None

        self._app: Optional[rumps.App] = None
        self._current_status = "idle"


    def run(self) -> None:
        """Run the menu bar app (blocks)."""
        self._app = _MergeScribeRumpsApp(self)
        self._app.run()

    def set_status(self, status: str) -> None:
        """
        Update status indicator.

        Args:
            status: One of "idle", "recording", "processing", "error"
        """
        self._current_status = status
        if self._app:
            self._app.show_status(status)

    def show_notification(self, title: str, message: str) -> None:
        """Show macOS notification."""
        try:
            rumps.notification("MergeScribe", title, message)
        except Exception as e:
            print(f"Notification failed: {e}")

    def show_error(self, message: str) -> None:
        """Show error message via notification."""
        self.set_status("error")
        self.show_notification("Error", message)


class _MergeScribeRumpsApp(rumps.App):
    """Internal rumps app implementation."""

    def __init__(self, parent: MenuBarApp):
        super().__init__("MergeScribe", title=None, quit_button="Quit MergeScribe")
        self.parent = parent
        self.show_status("idle")

        # Build menu
        self.menu = [
            rumps.MenuItem("Status: Idle", callback=None),
            None,  # Separator
            rumps.MenuItem("Settings...", callback=self._settings_clicked),
            None,  # Separator
        ]

        # Store reference to status item for updates
        self._status_item = self.menu["Status: Idle"]

    def show_status(self, status: str) -> None:
        """Put the glyph for this status in the menu bar."""
        image = _symbol(_SYMBOLS.get(status, "waveform"))
        if image is None:
            self.title = _FALLBACK.get(status, "🎤")
            return
        # rumps draws whatever NSImage it holds here, and redraws on request.
        self._icon_nsimage = image
        self._title = None
        try:
            self._nsapp.setStatusBarIcon()
            self._nsapp.setStatusBarTitle()
        except AttributeError:
            pass   # not running yet; it picks the image up when it starts

    def _settings_clicked(self, _) -> None:
        """Handle settings menu click."""
        if self.parent.on_settings:
            self.parent.on_settings()
        else:
            print("Settings clicked (no handler)")
