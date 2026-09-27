"""
Which dictation a log line belongs to.

Sessions overlap by design - one finalises while the next records - so their
lines interleave in the log and chunk numbers appear to jump backwards. Each
session stamps its threads with a short tag, and the console tee prefixes
whatever they print, including code that knows nothing about sessions.
"""

from __future__ import annotations

import threading

_local = threading.local()


def set_tag(tag: str) -> None:
    """Tag everything this thread prints from here on."""
    _local.tag = tag


def current_tag() -> str:
    return getattr(_local, "tag", "")


def tagged(line: str) -> str:
    """Prefix one already-formatted line, leaving blank lines alone."""
    tag = current_tag()
    if not tag or not line.strip():
        return line
    return f"[{tag}] {line}"
