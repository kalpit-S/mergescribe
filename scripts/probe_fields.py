"""
Diagnostic: dump the on-screen text field inventory the routing feature sees.

Run from the same terminal you launch MergeScribe from (so it shares the
Accessibility permission):

    ./venv/bin/python scripts/probe_fields.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ApplicationServices import (  # noqa: E402
    AXIsProcessTrusted,
    AXUIElementCreateApplication,
    AXUIElementSetAttributeValue,
)
from AppKit import NSWorkspace  # noqa: E402

from mergescribe.fields import (  # noqa: E402
    _TEXT_ROLES,
    _ax_get,
    describe_fields_for_prompt,
    snapshot_fields,
)


def survey_apps() -> None:
    """Show every regular app and what its AX tree exposes."""
    workspace = NSWorkspace.sharedWorkspace()
    front = workspace.frontmostApplication()
    print(f"frontmost: {front.localizedName() if front else None}\n")
    print(f"{'app':<28}{'pid':>7}  {'#win':>4}  {'#elems':>7}  {'#fields':>7}  {'walk_s':>7}")

    for app in workspace.runningApplications():
        if app.activationPolicy() != 0:
            continue
        pid = app.processIdentifier()
        name = str(app.localizedName() or "?")
        app_el = AXUIElementCreateApplication(pid)
        for wake_attr in ("AXManualAccessibility", "AXEnhancedUserInterface"):
            try:
                AXUIElementSetAttributeValue(app_el, wake_attr, True)
            except Exception:
                pass
        windows = _ax_get(app_el, "AXWindows") or []

        t0 = time.time()
        elems = 0
        found = 0
        stack = [(w, 0) for w in windows]
        while stack and elems < 2000 and time.time() - t0 < 3.0:
            el, depth = stack.pop()
            elems += 1
            role = _ax_get(el, "AXRole")
            if role in _TEXT_ROLES:
                found += 1
            if depth < 14:
                children = _ax_get(el, "AXChildren")
                if children:
                    stack.extend((c, depth + 1) for c in children)

        print(f"{name:<28}{pid:>7}  {len(windows):>4}  {elems:>7}  {found:>7}  {time.time()-t0:>7.2f}")


def main() -> None:
    print("AX process trusted:", AXIsProcessTrusted())
    if not AXIsProcessTrusted():
        print("\nGrant your terminal Accessibility in System Settings > Privacy &")
        print("Security > Accessibility, then re-run.")
        return

    print("\n=== App survey (all regular apps, generous budgets) ===")
    survey_apps()

    print("\n=== snapshot_fields() as the app runs it ===")
    start = time.time()
    fields = snapshot_fields()
    elapsed = time.time() - start
    print(f"{len(fields)} fields found in {elapsed:.2f}s")
    by_app = {}
    for f in fields:
        by_app.setdefault(f.app_name, []).append(f)
    print("per app:", {app: len(fs) for app, fs in by_app.items()} or "(none)")
    print()
    print(describe_fields_for_prompt(fields) or "(none)")


if __name__ == "__main__":
    main()
