"""
macOS menu bar application using rumps.

Provides status icon and menu for MergeScribe.
"""

from typing import Optional, Callable

import rumps


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

        # Status icons
        self._icons = {
            "idle": "🎤",
            "recording": "🔴",
            "processing": "⚡",
            "error": "❌",
        }

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
        icon = self._icons.get(status, "🎤")

        if self._app:
            self._app.title = icon

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
        super().__init__("🎤")
        self.parent = parent

        # Build menu
        self.menu = [
            rumps.MenuItem("Status: Idle", callback=None),
            None,  # Separator
            rumps.MenuItem("Settings...", callback=self._settings_clicked),
            None,  # Separator
        ]

        # Store reference to status item for updates
        self._status_item = self.menu["Status: Idle"]

    def _settings_clicked(self, _) -> None:
        """Handle settings menu click."""
        if self.parent.on_settings:
            self.parent.on_settings()
        else:
            print("Settings clicked (no handler)")
