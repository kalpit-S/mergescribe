"""
Corrections the speaker has made more than once, handed to the model as evidence.

When the same edit shows up in two separate dictations (the recognizers wrote
"cloud", the speaker changed it to "Claude"), that is strong evidence about the
speaker's vocabulary. The corrections go to the correction model as they are,
with how the recognizers wrote the word and how often it was fixed.

Deciding what a correction means is the model's job: whether it is a name the
recognizers keep missing, a stylistic preference, or noise, and whether it
applies to the sentence at hand. Code decides only what counts as repeated,
and trims what would waste the prompt.
"""

from __future__ import annotations

import difflib
import json
import re
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from .feedback import CORRECTIONS_PATH

MIN_SESSIONS = 2        # one correction is an anecdote; the same one twice is a pattern
MAX_CORRECTIONS = 30    # bounds how much the prompt can grow
MAX_SPAN_WORDS = 4      # long rewrites cost prompt space and say little about vocabulary
_EDGE_PUNCTUATION = ".,!?;:\"'()[]"

# (what the recognizers wrote, what the speaker corrected it to, dictations seen in)
Correction = Tuple[List[str], str, int]

_cache_lock = threading.Lock()
_cache: Tuple[float, List[Correction]] = (-1.0, [])


def _letters(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _replacements(typed: str, corrected: str) -> List[Tuple[str, str]]:
    """Word spans the user replaced, as (what was typed, what it became)."""
    before, after = typed.split(), corrected.split()
    return [
        (" ".join(before[i1:i2]).strip(_EDGE_PUNCTUATION),
         " ".join(after[j1:j2]).strip(_EDGE_PUNCTUATION))
        for tag, i1, i2, j1, j2
        in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
        if tag == "replace"
    ]


def _most_common(items: List[str]) -> str:
    """The most frequent form; on a tie, the most recent."""
    counts = Counter(items)
    top = max(counts.values())
    return next(item for item in reversed(items) if counts[item] == top)


def repeated_corrections(rows: Iterable[dict], *, min_sessions: int = MIN_SESSIONS,
                         limit: int = MAX_CORRECTIONS) -> List[Correction]:
    """
    Words the speaker corrected to the same thing in at least `min_sessions` dictations.

    Grouped by what the text was corrected *to*, so "cloud" and "Claud" both
    fixed to "Claude" count as evidence for one word. Ordered by how many
    dictations each appeared in, then by recency.
    """
    sessions: Dict[str, Set[str]] = defaultdict(set)
    afters: Dict[str, List[str]] = defaultdict(list)
    befores: Dict[str, List[str]] = defaultdict(list)
    last_seen: Dict[str, float] = {}
    for index, row in enumerate(rows):
        session = str(row.get("session_id") or f"row-{index}")
        for before, after in _replacements(row.get("typed", ""), row.get("corrected", "")):
            if not before or not after:
                continue
            if max(len(before.split()), len(after.split())) > MAX_SPAN_WORDS:
                continue
            if _letters(before) == _letters(after):
                continue   # only case or punctuation changed: nothing about words to learn
            key = _letters(after)
            sessions[key].add(session)
            afters[key].append(after)
            befores[key].append(before)
            last_seen[key] = max(last_seen.get(key, 0.0), float(row.get("ts") or 0.0))
    kept = [key for key, seen in sessions.items() if len(seen) >= min_sessions]
    kept.sort(key=lambda key: (len(sessions[key]), last_seen[key]), reverse=True)
    return [(list(dict.fromkeys(befores[key])), _most_common(afters[key]), len(sessions[key]))
            for key in kept[:limit]]


def learned_corrections(path: Path = CORRECTIONS_PATH) -> List[Correction]:
    """Repeated corrections from the corpus, re-read only when the file changes."""
    global _cache
    try:
        modified = path.stat().st_mtime
    except OSError:
        return []
    with _cache_lock:
        if _cache[0] == modified:
            return list(_cache[1])
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    corrections = repeated_corrections(rows)
    with _cache_lock:
        _cache = (modified, corrections)
    return list(corrections)


def vocabulary_prompt(corrections: List[Correction]) -> str:
    """The corrections as the model sees them: evidence, not replacements."""
    if not corrections:
        return ""
    described = "; ".join(
        " or ".join(f'"{b}"' for b in before) + f' corrected to "{after}" ({count} dictations)'
        for before, after, count in corrections
    )
    return ("Across separate earlier dictations, this speaker corrected the typed text in these "
            f"ways: {described}. This is evidence about the words they use, not a list of "
            "replacements: apply one only where the audio and the surrounding words support it.")
