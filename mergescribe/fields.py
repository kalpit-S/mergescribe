"""
On-screen text field inventory and voice-driven output routing.

Uses the macOS Accessibility API to enumerate text input fields across
running apps, so the correction LLM can decide which field dictation
belongs in (prefixing its reply with "TARGET: <field-id>").

Requires the Accessibility permission the app already holds for typing.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .feedback import wake_accessibility

# AX APIs are macOS-only; guard so pure-logic functions stay importable anywhere.
try:
    from ApplicationServices import (
        AXIsProcessTrusted,
        AXUIElementCreateApplication,
        AXUIElementCopyAttributeValue,
        AXUIElementSetAttributeValue,
        AXUIElementPerformAction,
        AXUIElementIsAttributeSettable,
    )
    from AppKit import NSRunningApplication, NSWorkspace
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGWindowListOptionOnScreenOnly,
        kCGWindowListExcludeDesktopElements,
        kCGNullWindowID,
    )
    _AX_AVAILABLE = True
except ImportError:
    _AX_AVAILABLE = False


# Roles that accept typed text
_TEXT_ROLES = {"AXTextField", "AXTextArea", "AXSearchField", "AXComboBox"}

# Address bars are typable but are never a dictation destination. Leaving them
# in cost a real misroute: the model correctly picked a ChatGPT window, whose
# only exposed field was the omnibox, so the text navigated instead of sending.
_JUNK_LABEL_SUBSTRINGS = (
    "search google or type a url",
    "ask google or type a url",
    "address and search bar",
    "search or enter website",   # Safari
)

# Roles worth sampling when summarising what a window is about. Kept narrow on
# purpose: pulling text off every group/button floods the sample with UI chrome.
_TEXT_BEARING_ROLES = {
    "AXStaticText",
    "AXHeading",
    "AXLink",
    "AXCell",
    "AXTextArea",
    "AXTextField",
}

# Traversal budgets — AX trees can be huge (especially Electron apps)
_MAX_APPS = 6
# Web inputs (a ChatGPT composer in Chrome) sit far deeper than native
# controls; at depth 14 the walk stopped above them and only the omnibox
# was ever found, so dictation aimed at a chat landed in the URL bar.
_MAX_DEPTH = 25
_MAX_ELEMENTS_PER_WINDOW = 1500
_MAX_WINDOWS_PER_APP = 6
# Per-app slice of the global budget: one browser with many windows must not
# consume the whole snapshot and starve the other allowlisted apps.
_APP_TIME_BUDGET = 0.8
_MAX_FIELDS_PER_APP = 10          # e.g. Preview exposes 50 PDF annotation boxes
_TIME_BUDGET_SECONDS = 3.0
_VALUE_PREVIEW_CHARS = 120

# Window-content sampling. Electron/web trees are deep and wide, so these are
# far more generous than the field walk; the per-window time budget is what
# actually bounds the cost.
_WINDOW_CONTEXT_CHARS = 400
_WINDOW_TITLE_CHARS = 90
_HARVEST_MAX_DEPTH = 25
_HARVEST_MAX_ELEMENTS = 2500
_HARVEST_TIME_BUDGET = 0.25
# Context for a field is sampled from an ancestor of that field rather than
# from the window root: a root walk returns whatever sits at the end of the
# DOM (footers, sidebars), while the subtree around a composer is the
# conversation the dictation would be joining.
_HARVEST_ANCESTOR_LEVELS = 5
_HARVEST_ENOUGH_CHARS = 120   # stop climbing once an ancestor yields this much
_FIELD_HARVEST_BUDGET = 0.15  # per field, for the whole climb
# Electron trees populate asynchronously after the wake poke; if a walk looks
# like it hit an asleep tree (windows but almost no elements), wait and re-walk.
_WAKE_GRACE_SECONDS = 0.35
_ASLEEP_ELEMENT_THRESHOLD = 60

DEBUG_WALK = True  # Log which apps were walked, what they yielded, and timing

TARGET_FOCUSED = "focused"

_TARGET_RE = re.compile(r"^\s*TARGET:\s*(\S+)\s*(?:\n|$)")
# Buffer at most this much before concluding there is no TARGET prefix
_TARGET_DECISION_CHARS = 64


@dataclass
class FieldTarget:
    """One text input field discovered on screen."""
    id: str                      # Short id used in the prompt, e.g. "f3"
    app_name: str
    pid: int
    window_title: str
    role: str
    label: str                   # Placeholder/title/description, best effort
    value_preview: str           # Truncated current content
    window_context: str = ""     # Sample of the window's visible text
    window_index: int = 0        # Z-order within its app (0 = that app's front window)
    app_is_frontmost: bool = False
    is_focused: bool = False     # The window the user is actually looking at
    element: object = field(default=None, repr=False)   # AXUIElement ref
    window: object = field(default=None, repr=False)    # AXUIElement ref


def _ax_get(element, attribute: str):
    """Read an AX attribute, returning None on any error."""
    try:
        err, value = AXUIElementCopyAttributeValue(element, attribute, None)
        return value if err == 0 else None
    except Exception:
        return None


def _is_editable(element) -> bool:
    """
    Filter out read-only text regions (e.g. Preview's OCR overlay on images):
    a real input accepts focus or value changes.
    """
    for attr in ("AXFocused", "AXValue"):
        try:
            err, settable = AXUIElementIsAttributeSettable(element, attr, None)
            if err == 0 and settable:
                return True
        except Exception:
            pass
    return False


def _field_label(element) -> str:
    for attr in ("AXPlaceholderValue", "AXTitle", "AXDescription", "AXRoleDescription"):
        value = _ax_get(element, attr)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _collect_fields(window, app_name: str, pid: int,
                    window_title: str, out: List[FieldTarget],
                    deadline: float, budget: List[int],
                    fields_remaining: List[int], window_index: int = 0) -> None:
    """Depth-first walk of one window collecting text-input elements."""
    stack = [(window, 0)]
    while stack:
        if time.time() > deadline or budget[0] <= 0 or fields_remaining[0] <= 0:
            return
        element, depth = stack.pop()
        budget[0] -= 1

        role = _ax_get(element, "AXRole")
        if role in _TEXT_ROLES and _is_editable(element):
            label = _field_label(element)
            if any(junk in label.lower() for junk in _JUNK_LABEL_SUBSTRINGS):
                continue
            value = _ax_get(element, "AXValue")
            preview = (value or "")[:_VALUE_PREVIEW_CHARS] if isinstance(value, str) else ""
            fields_remaining[0] -= 1
            out.append(FieldTarget(
                id=f"f{len(out) + 1}",
                app_name=app_name,
                pid=pid,
                window_title=window_title,
                role=role,
                label=label,
                value_preview=preview,
                window_index=window_index,
                element=element,
                window=window,
            ))

        if depth < _MAX_DEPTH:
            children = _ax_get(element, "AXChildren")
            if children:
                for child in children:
                    stack.append((child, depth + 1))


def _element_text(element, role: str) -> str:
    """Readable text from one element, or "" if it carries none."""
    if role in _TEXT_BEARING_ROLES:
        for attr in ("AXValue", "AXTitle", "AXDescription"):
            value = _ax_get(element, attr)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _harvest_window_text(window, max_chars: int = _WINDOW_CONTEXT_CHARS,
                         deadline: Optional[float] = None) -> str:
    """
    Sample a window's visible text so the routing model knows what it's about
    without a full dump.

    Electron/web content nests far deeper than native UI (React trees run 15-25
    levels), so the depth and element budgets have to be generous or the walk
    terminates above the content entirely. Traversal is depth-first from the
    last child, which reaches the bottom of a scrollable view first — for a
    chat window that's the most recent messages.
    """
    stop_at = time.time() + _HARVEST_TIME_BUDGET
    if deadline is not None:
        stop_at = min(stop_at, deadline)

    parts: List[str] = []
    seen: set = set()
    total = 0
    stack = [(window, 0)]
    visited = 0

    while stack and total < max_chars and visited < _HARVEST_MAX_ELEMENTS:
        if visited % 64 == 0 and time.time() > stop_at:
            break
        element, depth = stack.pop()
        visited += 1

        role = _ax_get(element, "AXRole")
        text = _element_text(element, role)
        if len(text) > 3 and text not in seen:
            seen.add(text)
            parts.append(text)
            total += len(text)
            if role == "AXStaticText":
                continue  # leaf text: no useful children

        if depth < _HARVEST_MAX_DEPTH:
            children = _ax_get(element, "AXChildren")
            if children:
                for child in children:
                    stack.append((child, depth + 1))

    return " · ".join(parts)[:max_chars]


def _harvest_near_element(element, window, deadline: Optional[float] = None) -> str:
    """
    Sample the text surrounding a specific input.

    Climbs AXParent from the input and harvests each ancestor's subtree,
    stopping as soon as one yields a useful amount of text. Depth-first from
    the last child means the composer's own subtree is visited first (usually
    empty) and then earlier siblings — for a chat, the most recent messages.
    Falls back to the window root if the climb finds nothing.
    """
    stop_at = time.time() + _FIELD_HARVEST_BUDGET
    if deadline is not None:
        stop_at = min(stop_at, deadline)

    node = element
    best = ""
    for _ in range(_HARVEST_ANCESTOR_LEVELS):
        if time.time() > stop_at:
            break
        parent = _ax_get(node, "AXParent")
        if parent is None:
            break
        node = parent
        text = _harvest_window_text(node, deadline=stop_at)
        if len(text) > len(best):
            best = text
        if len(best) >= _HARVEST_ENOUGH_CHARS:
            break

    if not best and window is not None:
        best = _harvest_window_text(window, deadline=stop_at)
    return best


def _on_screen_pids() -> List[int]:
    """PIDs of apps with on-screen windows, front-to-back z-order."""
    info = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
        kCGNullWindowID,
    ) or []
    pids: List[int] = []
    for w in info:
        # layer 0 == normal windows (skips menu bar items, overlays, etc.)
        if w.get("kCGWindowLayer") == 0:
            pid = w.get("kCGWindowOwnerPID")
            if pid and pid not in pids:
                pids.append(pid)
    return pids


def _walk_app(pid: int, fields: List[FieldTarget], deadline: float) -> int:
    """Walk one app's windows; returns elements visited."""
    app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
    if app is None:
        return 0
    deadline = min(deadline, time.time() + _APP_TIME_BUDGET)
    app_name = str(app.localizedName() or "")
    app_element = AXUIElementCreateApplication(pid)

    # Chromium/Electron apps expose an empty tree until an assistive client
    # asks for one; see feedback.wake_accessibility.
    wake_accessibility(pid)

    def walk_once() -> int:
        windows = (_ax_get(app_element, "AXWindows") or [])[:_MAX_WINDOWS_PER_APP]
        visited_total = 0
        fields_remaining = [_MAX_FIELDS_PER_APP]
        for window_index, window in enumerate(windows):
            # Per-WINDOW budget: one heavy window (e.g. a dense dashboard tab)
            # must not starve the app's other windows of enumeration.
            budget = [_MAX_ELEMENTS_PER_WINDOW]
            title = _ax_get(window, "AXTitle")
            _collect_fields(
                window, app_name, pid, str(title or ""),
                fields, deadline, budget, fields_remaining, window_index,
            )
            visited_total += _MAX_ELEMENTS_PER_WINDOW - budget[0]
        return visited_total

    def harvest_window_context(start_index: int) -> None:
        """Attach text surrounding each field found in this app."""
        for f in fields[start_index:]:
            if time.time() > deadline:
                return
            try:
                f.window_context = _harvest_near_element(f.element, f.window, deadline)
            except Exception:
                f.window_context = ""

    before = len(fields)
    visited = walk_once()

    # Asleep Electron tree. Two shapes: a window shell with almost nothing
    # inside, or — when the wake poke hasn't taken effect yet — no windows
    # reported at all (visited == 0). Both need a beat, then a retry.
    if (len(fields) == before and visited < _ASLEEP_ELEMENT_THRESHOLD
            and time.time() + _WAKE_GRACE_SECONDS < deadline):
        time.sleep(_WAKE_GRACE_SECONDS)
        visited = walk_once()

    # Some apps (Electron with a dormant tree, sparse native apps) never
    # expose their inputs. Fall back to the window itself as a target:
    # focusing it types into whatever field that app focuses by default.
    # Only for asleep-looking trees — a fully exposed tree with no text
    # fields (e.g. Finder) is genuinely not a typing destination.
    if len(fields) == before and visited < _ASLEEP_ELEMENT_THRESHOLD:
        # One target per window (multiple sessions of the same app are
        # distinct destinations), each with its own content sample.
        windows = _ax_get(app_element, "AXWindows") or []
        for window_index, window in enumerate(windows[:4]):
            title = str(_ax_get(window, "AXTitle") or "")
            fields.append(FieldTarget(
                id=f"f{len(fields) + 1}",
                app_name=app_name,
                pid=pid,
                window_title=title,
                role="AXWindow",
                label="window (types into its focused input)",
                value_preview=_harvest_window_text(window, deadline=deadline),
                window_index=window_index,
                element=window,
                window=window,
            ))
    else:
        harvest_window_context(before)

    return visited


def _matches(text: str, entries: List[str]) -> bool:
    """True if any allowlist entry appears in text (case-insensitive).

    One-directional on purpose: "Chrome" matches window/app "Google Chrome",
    but an entry like "Google Chrome - Kalpit" must NOT match the bare app
    name — that's what scopes routing to a single Chrome profile.
    """
    text = text.lower()
    return any(e.lower().strip() in text for e in entries if e.strip())


def _app_allowed(app_identity: str, allowed_apps: Optional[List[str]]) -> bool:
    """Empty/None allowlist = everything is eligible. Match is case-insensitive substring.

    app_identity should include both display name and bundle id — some apps'
    display names differ from what users call them (e.g. the Codex app's
    display name is "ChatGPT" but its bundle id is com.openai.codex).
    """
    if not allowed_apps:
        return True
    return _matches(app_identity, allowed_apps)


def _window_titles(pid: int) -> List[str]:
    """Cheap AX read of an app's window titles (no tree walk)."""
    app_element = AXUIElementCreateApplication(pid)
    windows = _ax_get(app_element, "AXWindows") or []
    return [str(_ax_get(w, "AXTitle") or "") for w in windows]


def snapshot_fields(allowed_apps: Optional[List[str]] = None) -> List[FieldTarget]:
    """
    Enumerate text input fields across apps you've allowed as destinations.

    Apps matching the allowlist by name or bundle id are walked first, in
    z-order, before any title-scoped probing. Z-order alone starved the
    declared destinations: a browser with several windows consumed the whole
    budget (plus an AX round trip per app just to read window titles) and
    Codex/Claude were never reached.

    Returns [] if AX is unavailable or the process lacks the
    Accessibility permission.
    """
    if not _AX_AVAILABLE or not AXIsProcessTrusted():
        return []

    fields: List[FieldTarget] = []
    deadline = time.time() + _TIME_BUDGET_SECONDS
    walked = 0
    walked_pids: set = set()
    trace: List[str] = []

    onscreen = _on_screen_pids()
    rank = {pid: i for i, pid in enumerate(onscreen)}

    def walk(pid: int, name: str, title_scoped: bool) -> None:
        nonlocal walked
        walked += 1
        walked_pids.add(pid)
        started = time.time()
        before = len(fields)
        try:
            _walk_app(pid, fields, deadline)
        except Exception as e:
            trace.append(f"{name}: error {e}")
            return
        if title_scoped:
            kept = [f for f in fields[before:] if _matches(f.window_title, allowed_apps)]
            del fields[before:]
            fields.extend(kept)
        found = len(fields) - before
        scope = " (title-scoped)" if title_scoped else ""
        trace.append(f"{name}: {found} fields{scope} in {time.time() - started:.2f}s")

    # Pass 1 — apps allowed by name/bundle id, frontmost first.
    candidates = []
    for app in NSWorkspace.sharedWorkspace().runningApplications():
        if app.activationPolicy() != 0:
            continue
        pid = app.processIdentifier()
        identity = f"{app.localizedName() or ''} {app.bundleIdentifier() or ''}"
        if _app_allowed(identity, allowed_apps):
            candidates.append((rank.get(pid, 10_000), pid, str(app.localizedName() or "")))
    candidates.sort()

    for _, pid, name in candidates:
        if walked >= _MAX_APPS or time.time() > deadline:
            trace.append(f"{name}: skipped (budget spent)")
            continue
        walk(pid, name, title_scoped=False)

    # Pass 2 — allowlist entries that scope by window title (e.g. one Chrome
    # profile). Each candidate costs an AX round trip to read its titles, so
    # this runs only on whatever budget pass 1 left behind.
    if allowed_apps:
        for pid in onscreen:
            if walked >= _MAX_APPS or time.time() > deadline:
                break
            if pid in walked_pids:
                continue
            app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
            if app is None:
                continue
            name = str(app.localizedName() or "")
            try:
                if not any(_matches(t, allowed_apps) for t in _window_titles(pid)):
                    continue
            except Exception:
                continue
            walk(pid, name, title_scoped=True)

    _mark_focus(fields)

    # Focused target first, then the frontmost app's other windows, then the
    # rest — so what the user is looking at leads the inventory.
    fields.sort(key=lambda f: (not f.is_focused, not f.app_is_frontmost, f.window_index))

    # Renumber after sorting so ids stay dense and in presentation order
    for i, f in enumerate(fields):
        f.id = f"f{i + 1}"

    if DEBUG_WALK and trace:
        print("[Routing] walk: " + "; ".join(trace))

    return fields


def _mark_focus(fields: List[FieldTarget]) -> None:
    """
    Flag the target the user is actually looking at.

    Without this the inventory is flat, so a Chrome window buried behind three
    others reads exactly like the one on screen — and text routed there lands
    somewhere the user can't see.
    """
    try:
        front = NSWorkspace.sharedWorkspace().frontmostApplication()
    except Exception:
        return
    if front is None:
        return

    front_pid = front.processIdentifier()
    focused_title = ""
    try:
        focused_window = _ax_get(AXUIElementCreateApplication(front_pid), "AXFocusedWindow")
        if focused_window is not None:
            focused_title = str(_ax_get(focused_window, "AXTitle") or "")
    except Exception:
        pass

    for f in fields:
        f.app_is_frontmost = (f.pid == front_pid)
        f.is_focused = (
            f.app_is_frontmost
            and (f.window_title == focused_title if focused_title else f.window_index == 0)
        )


def describe_fields_for_prompt(fields: List[FieldTarget]) -> str:
    """Render the field inventory as prompt text."""
    lines = []
    seen_context: set = set()
    for f in fields:
        label = f.label or f.role
        content = f" | contains: \"{f.value_preview}\"" if f.value_preview.strip() else " | empty"
        title = f.window_title
        if len(title) > _WINDOW_TITLE_CHARS:
            title = title[:_WINDOW_TITLE_CHARS] + "..."
        if f.is_focused:
            where = " <- FOCUSED: the window and field the user is looking at right now"
        elif f.window_index > 0:
            where = f" (background window of {f.app_name}; likely hidden from view)"
        else:
            where = " (this app is not in front)"
        lines.append(f"[{f.id}] {f.app_name} — window \"{title}\" — {label}{content}{where}")
        # Context is now sampled per field, so dedupe on the text itself:
        # neighbouring fields in one window often resolve to the same sample.
        context = f.window_context.strip()
        if context and context not in seen_context:
            seen_context.add(context)
            lines.append(f"    nearby text: \"{context}\"")
    return "\n".join(lines)


def focus_field(target: FieldTarget) -> bool:
    """
    Bring the target's app forward, raise its window, and focus the field.

    Returns True only if the field focus was confirmed — callers should
    fall back to the currently focused field otherwise.
    """
    if not _AX_AVAILABLE:
        return False
    try:
        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(target.pid)
        if app is None:
            return False
        app.activateWithOptions_(2)  # NSApplicationActivateIgnoringOtherApps
        if target.window is not None:
            AXUIElementPerformAction(target.window, "AXRaise")

        if target.role == "AXWindow":
            # Whole-window target: activate + raise is the whole job; the app
            # decides which of its inputs receives the keystrokes.
            time.sleep(0.25)
            return True

        err = AXUIElementSetAttributeValue(target.element, "AXFocused", True)
        time.sleep(0.25)  # let focus settle before synthetic keystrokes
        return err == 0
    except Exception as e:
        print(f"[Routing] Failed to focus {target.id} ({target.app_name}): {e}")
        return False


def parse_target_prefix(text: str) -> Tuple[Optional[str], str]:
    """
    Split a leading "TARGET: <id>" line off LLM output.

    Returns (target_id or None, remaining_text).
    """
    match = _TARGET_RE.match(text)
    if match:
        return match.group(1), text[match.end():]
    return None, text


class TargetStreamParser:
    """
    Incremental parser for streamed LLM output that may start with a
    "TARGET: <id>" line.

    Buffers tokens until the routing decision can be made (first newline,
    or enough text to rule a prefix out), invokes on_target exactly once,
    then passes all remaining text through.
    """

    def __init__(self, on_target: Callable[[Optional[str]], None]):
        self._on_target = on_target
        self._buffer = ""
        self._decided = False

    def feed(self, token: str) -> str:
        """Feed one token; returns text safe to emit now (may be empty)."""
        if self._decided:
            return token

        self._buffer += token
        stripped = self._buffer.lstrip()

        # Still possibly accumulating a TARGET prefix?
        if "\n" not in self._buffer and len(self._buffer) < _TARGET_DECISION_CHARS:
            if not stripped or "TARGET:"[:len(stripped)].startswith(stripped) or stripped.startswith("TARGET:"):
                return ""

        return self._decide()

    def flush(self) -> str:
        """Call after the stream ends; returns any withheld text."""
        if self._decided:
            return ""
        return self._decide()

    def _decide(self) -> str:
        self._decided = True
        target_id, remainder = parse_target_prefix(self._buffer)
        self._on_target(target_id)
        return remainder
