"""
Post-output edit detection.

After text is typed into a field, watch that field and capture what the user
changed. Edits are the strongest correction signal available: human-verified
ground truth for what the output should have been, produced at zero cost.

How it works: right after typing, snapshot the field's ENTIRE value as a
baseline and locate our typed span inside it. Later checks diff baseline
against the current value and keep only changes that land *inside* that span.

The baseline matters. Comparing the typed text against the whole field (the
first implementation) could not tell "you rewrote my words" from "you typed
more text nearby", and since dictating twice into one field is normal usage,
almost every detection was a false positive.

Two kinds of edit end up in the corpus and they are NOT interchangeable:
  - transcription errors  ("post grass" -> "Postgres")  -> training data
  - changes of intent     ("three months" -> "four months")       -> not an error
Separating them needs semantics, so it happens later in a batch LLM pass
(scripts/classify_corrections.py) rather than here.
"""

import difflib
import json
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

try:
    from ApplicationServices import (
        AXUIElementCreateApplication,
        AXUIElementCreateSystemWide,
        AXUIElementCopyAttributeValue,
        AXUIElementGetPid,
        AXUIElementSetAttributeValue,
    )
    from CoreFoundation import kCFBooleanTrue
    _AX_AVAILABLE = True
except ImportError:
    _AX_AVAILABLE = False


_CHECK_DELAYS_SECONDS = (8.0, 25.0, 60.0)
_BASELINE_SETTLE_SECONDS = 0.4   # let the app finish processing our keystrokes
_MIN_EDIT_CHARS = 2              # ignore trailing-whitespace noise
_SETTLE_CONFIRM_SECONDS = 2.5    # a changed field must hold still before it counts
# A correction keeps most of the original. Anything that replaces the text
# wholesale is a send/clear, not an edit: composers show a placeholder
# ("Ask ChatGPT", "Message #general") once submitted, and diffing a paragraph
# against that placeholder produced 83% of the first corpus as garbage.
_MIN_EDIT_SIMILARITY = 0.5
_MIN_EDIT_LENGTH_RATIO = 0.5
# A field can hold a whole document or chat thread. A character diff is
# quadratic pure Python: two 40k-character snapshots took tens of seconds per
# check, holding the GIL the whole time, which starved the main thread and
# dropped the HUD to one frame a second. Edits are local, so the unchanged
# ends are stripped first and only what is left is diffed - up to this much.
_MAX_DIFF_CHARS = 6000
_MAX_FUZZY_FIELD_CHARS = 20000

CORRECTIONS_PATH = Path.home() / ".mergescribe" / "corrections.jsonl"

# A tree that was only just woken can hand back a placeholder element first.
_UNREADABLE_RETRIES = 3
_UNREADABLE_RETRY_SECONDS = 0.3


def _ax_get(element, attribute: str):
    try:
        err, value = AXUIElementCopyAttributeValue(element, attribute, None)
        return value if err == 0 else None
    except Exception:
        return None


def wake_accessibility(pid: int) -> None:
    """
    Ask a Chromium or Electron app to build its accessibility tree.

    Chrome, Slack, Claude, ChatGPT and VS Code expose an empty tree until an
    assistive client asks for one, so their text fields read as nothing.
    Harmless for native apps. kCFBooleanTrue rather than Python True, because
    Electron ignores a wrongly bridged value.
    """
    if not _AX_AVAILABLE or not pid:
        return
    app = AXUIElementCreateApplication(pid)
    for attribute in ("AXManualAccessibility", "AXEnhancedUserInterface"):
        try:
            AXUIElementSetAttributeValue(app, attribute, kCFBooleanTrue)
        except Exception:
            pass


def wake_frontmost_app() -> None:
    """Wake whichever app currently has focus: the one being dictated into."""
    try:
        from AppKit import NSWorkspace
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is not None:
            wake_accessibility(int(app.processIdentifier()))
    except Exception:
        pass


def _pid(element) -> Optional[int]:
    try:
        err, pid = AXUIElementGetPid(element, None)
        return pid if err == 0 else None
    except Exception:
        return None


def read_field(target, pid: Optional[int]):
    """
    Read the watched field. Returns (element, value); value is None if it's gone.

    Some apps destroy the composer and build a new one when a message is sent:
    ChatGPT does, Claude clears its composer in place. The saved element then
    goes stale and every read fails - which was being filed as "unreadable",
    165 times in three days, when it was the same event Claude reports as
    "cleared". When that happens, whatever now has focus in the *same* app is
    the field's successor: after a send it is the fresh empty composer, and if
    the app merely re-rendered mid-edit it holds the edit, so either way it
    classifies correctly. Focus in another app means the field is truly gone.
    """
    value = _ax_get(target, "AXValue")
    if isinstance(value, str) or pid is None:
        return target, value if isinstance(value, str) else None
    successor = focused_element()
    if successor is None or _pid(successor) != pid:
        return target, None
    value = _ax_get(successor, "AXValue")
    return (successor, value) if isinstance(value, str) else (target, None)


def focused_element():
    """The UI element that currently has keyboard focus, or None."""
    if not _AX_AVAILABLE:
        return None
    try:
        return _ax_get(AXUIElementCreateSystemWide(), "AXFocusedUIElement")
    except Exception:
        return None


def _anchor(baseline: str, typed: str) -> Optional[Tuple[int, int]]:
    """
    Locate the text we typed inside the field's value.

    Exact match usually works, but apps normalise on insert (collapsing
    whitespace, converting newlines, trimming) and a streamed insert can be
    reflowed. Falls back to matching a distinctive tail of the typed text,
    then to a fuzzy block match, before giving up.
    """
    # rfind: if the same phrase appears twice, ours is the later one
    start = baseline.rfind(typed)
    if start >= 0:
        return (start, start + len(typed))

    # Tail match: the head is likelier to be reflowed than the end
    tail = typed[-60:] if len(typed) > 60 else typed
    if len(tail) >= 12:
        start = baseline.rfind(tail)
        if start >= 0:
            return (max(0, start + len(tail) - len(typed)), start + len(tail))

    # Fuzzy: longest common block, accepted only if it covers most of our text.
    # Pure Python and linear in the field: past this size it would hold the
    # GIL long enough to stutter the HUD, for a match that rarely lands anyway.
    if len(baseline) > _MAX_FUZZY_FIELD_CHARS:
        return None
    match = difflib.SequenceMatcher(None, baseline, typed, autojunk=False)\
        .find_longest_match(0, len(baseline), 0, len(typed))
    if match.size >= max(20, int(len(typed) * 0.5)):
        start = max(0, match.a - match.b)
        return (start, min(len(baseline), start + len(typed)))

    return None


def _map_index(opcodes, index: int, prefer_end: bool = False) -> int:
    """Translate a position in the baseline string to one in the current string.

    prefer_end asks for an EXCLUSIVE end offset, so a position landing inside
    an unchanged block has to map one past itself — without the +1 the last
    character of every captured span was silently dropped.
    """
    for tag, i1, i2, j1, j2 in opcodes:
        if i1 <= index < i2:
            if tag == "equal":
                return j1 + (index - i1) + (1 if prefer_end else 0)
            return j2 if prefer_end else j1
    return opcodes[-1][4] if opcodes else 0


def _common_ends(a: str, b: str) -> Tuple[int, int]:
    """Lengths of the common prefix and suffix, bisecting on slice equality (C speed)."""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    prefix = lo
    lo, hi = 0, min(len(a), len(b)) - prefix
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[len(a) - mid:] == b[len(b) - mid:]:
            lo = mid
        else:
            hi = mid - 1
    return prefix, lo


def _opcodes(a: str, b: str) -> Optional[list]:
    """
    SequenceMatcher opcodes turning a into b, diffing only what lies between
    the unchanged ends. None if even that is too large to diff.
    """
    prefix, suffix = _common_ends(a, b)
    a_mid, b_mid = a[prefix:len(a) - suffix], b[prefix:len(b) - suffix]
    if len(a_mid) + len(b_mid) > _MAX_DIFF_CHARS:
        return None
    opcodes = [("equal", 0, prefix, 0, prefix)] if prefix else []
    if a_mid or b_mid:
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                None, a_mid, b_mid, autojunk=False).get_opcodes():
            opcodes.append((tag, i1 + prefix, i2 + prefix, j1 + prefix, j2 + prefix))
    if suffix:
        opcodes.append(("equal", len(a) - suffix, len(a), len(b) - suffix, len(b)))
    return opcodes


def diff_span(baseline: str, current: str, span: Tuple[int, int]):
    """
    Compare baseline vs current, restricted to the span we typed.

    Returns (changed, corrected_text). `changed` is True only when an edit
    lands inside the span — text added elsewhere in the field is ignored.
    None when the field changed too much, in too many places, to diff.
    """
    start, end = span
    opcodes = _opcodes(baseline, current)
    if opcodes is None:
        return None

    changed = any(
        tag != "equal" and i1 < end and i2 > start
        for tag, i1, i2, _, _ in opcodes
    )
    new_start = _map_index(opcodes, start)
    new_end = _map_index(opcodes, max(start, end - 1), prefer_end=True)
    return changed, current[new_start:max(new_start, new_end)]


def classify(baseline: str, current: Optional[str], span: Tuple[int, int]):
    """
    Returns (outcome, corrected_text).

    unchanged   - our span survived as typed (text elsewhere may have changed)
    edited      - our span was modified; corrected_text is the new version
    cleared     - the field emptied (message sent, draft discarded)
    replaced    - the text was swapped wholesale (a send leaves a placeholder)
    reformatted - only case/punctuation changed, i.e. the app normalised it
    gone        - the field no longer exists and nothing in its app replaced it
    unmeasured  - our text is no longer there verbatim, and the field changed
                  too much elsewhere to diff it cheaply
    """
    if not isinstance(current, str):
        return ("gone", "")
    if not current.strip():
        return ("cleared", "")

    diff = diff_span(baseline, current, span)
    if diff is None:
        intact = baseline[span[0]:span[1]] in current
        return ("unchanged", "") if intact else ("unmeasured", "")
    changed, corrected = diff
    if not changed:
        return ("unchanged", "")

    corrected = corrected.strip()
    typed = baseline[span[0]:span[1]].strip()
    if len(corrected) < _MIN_EDIT_CHARS:
        return ("cleared", "")

    # Guard against the field being emptied and refilled with something
    # unrelated. Both checks are needed: a placeholder can be similar in
    # length to a short dictation, and a long unrelated string can score
    # a deceptively high ratio on shared common words.
    if len(corrected) < len(typed) * _MIN_EDIT_LENGTH_RATIO:
        return ("replaced", "")
    if difflib.SequenceMatcher(None, typed, corrected).ratio() < _MIN_EDIT_SIMILARITY:
        return ("replaced", "")
    if is_app_transform(typed, corrected):
        return ("reformatted", "")

    return ("edited", corrected)


def is_app_transform(typed: str, corrected: str) -> bool:
    """True when the app rewrote the text rather than the user correcting it.

    Observed in the wild: a browser omnibox URL-encodes a query
    ("Quiet hiking trails" -> "Quiet+hiking+trails"), and some fields lowercase or
    strip punctuation on commit. None of those say anything about what was
    misheard, so they must not enter the training corpus.

    The test is whether any letters or digits actually changed. If only case,
    punctuation, and separators differ, nothing was corrected.
    """
    from .vocabulary import _letters
    return _letters(typed) == _letters(corrected)


def record_correction(entry: dict, path: Path = CORRECTIONS_PATH) -> None:
    """Append a correction to the durable corpus (joins to audio by session_id)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[Feedback] Could not record correction: {e}")


def load_corrections(path: Path = CORRECTIONS_PATH) -> List[dict]:
    """Read the correction corpus."""
    if not path.exists():
        return []
    out = []
    for line in open(path):
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def watch_for_edits(
    typed_text: str,
    on_result: Callable[[str, float, str, str], None],
    element=None,
    app: str = "",
) -> bool:
    """
    Watch the destination field and report what happened to typed_text.

    on_result(outcome, checked_after_s, original, corrected) fires once from a
    background thread. Returns False if watching isn't possible.
    """
    if not _AX_AVAILABLE or not typed_text.strip():
        return False
    target = element if element is not None else focused_element()
    if target is None:
        return False

    pid = _pid(target)

    def work() -> None:
        nonlocal target
        time.sleep(_BASELINE_SETTLE_SECONDS)
        baseline = _ax_get(target, "AXValue")
        for _ in range(_UNREADABLE_RETRIES):
            if isinstance(baseline, str):
                break
            time.sleep(_UNREADABLE_RETRY_SECONDS)
            target = focused_element() or target
            baseline = _ax_get(target, "AXValue")
        if not isinstance(baseline, str):
            # Most of these were silent: 61% of dictations over three days
            # could not be watched at all, so corrections never reached the
            # corpus. Say which app, so the gap can be closed.
            print(f"[Feedback] Can't read the field in {app or 'the focused app'} "
                  f"(role {_ax_get(target, 'AXRole')}, value {type(baseline).__name__})")
            on_result("unreadable", 0.0, typed_text, "")
            return

        span = _anchor(baseline, typed_text)
        if span is None:
            # Could not find our text in the field: the app transformed it on
            # insert, or focus moved before the snapshot. Report what the field
            # actually held so the mismatch is diagnosable.
            preview = baseline.strip().replace("\n", " ")[:90]
            print(f"[Feedback] Could not anchor output in field "
                  f"(field has {len(baseline)} chars: \"{preview}\")")
            on_result("unanchored", 0.0, typed_text, "")
            return

        elapsed = _BASELINE_SETTLE_SECONDS
        for delay in _CHECK_DELAYS_SECONDS:
            time.sleep(max(0.0, delay - elapsed))
            elapsed = delay
            target, current = read_field(target, pid)
            outcome, corrected = classify(baseline, current, span)
            if outcome == "unchanged":
                continue

            # Re-read before recording: a field that is still changing is
            # someone mid-keystroke, not a finished edit. Without this, the
            # 8s check caught partial words ("basically" -> "bically",
            # "here's how" -> "here'how") and 37 of the first 68 rows were
            # snapshots of text shorter than what had been typed.
            time.sleep(_SETTLE_CONFIRM_SECONDS)
            elapsed += _SETTLE_CONFIRM_SECONDS
            target, current = read_field(target, pid)
            outcome2, corrected2 = classify(baseline, current, span)
            if outcome2 != outcome or corrected2 != corrected:
                continue

            on_result(outcome, elapsed, typed_text, corrected)
            return

        on_result("unchanged", _CHECK_DELAYS_SECONDS[-1], typed_text, "")

    threading.Thread(target=work, daemon=True).start()
    return True
