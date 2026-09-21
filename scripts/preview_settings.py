"""Dev helper: open the settings window on its own, without the menu bar app.

    ./venv/bin/python scripts/preview_settings.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from AppKit import (  # noqa: E402
    NSApplication, NSApplicationActivationPolicyRegular,
)
from PyObjCTools import AppHelper  # noqa: E402

from mergescribe.ui.settings import open_settings  # noqa: E402

if __name__ == "__main__":
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    open_settings()
    AppHelper.runEventLoop()
