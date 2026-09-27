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
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from .feedback import CORRECTIONS_PATH

# Words handed in from outside, one per line: anything that knows the speaker's
# vocabulary before they say it (names in their calendar, jargon in what they
# read) can write here. Most important first - the prompt has room for the head.
SUPPLIED_PATH = Path.home() / ".mergescribe" / "vocabulary.txt"
# Learned corrections the speaker removed in Settings, by what they corrected to; kept
# beside the corpus.
FORGOTTEN_NAME = "vocabulary_forgotten.txt"
MAX_PROMPT_TERMS = 100  # supplied terms the correction prompt can afford
MAX_KEYTERMS = 1000     # AssemblyAI's limit per request
MAX_KEYTERM_WORDS = 6   # and per term

MIN_SESSIONS = 2        # one correction is an anecdote; the same one twice is a pattern
MAX_CORRECTIONS = 30    # bounds how much the prompt can grow
MAX_SPAN_WORDS = 4      # long rewrites cost prompt space and say little about vocabulary
_EDGE_PUNCTUATION = ".,!?;:\"'()[]"

# Words every recognizer already knows. A learned correction made only of these came from
# rewording ("anything" -> "it"), and priming a recognizer with them does nothing but skew
# it; the correction model still sees them as evidence and judges them itself.
EVERYDAY = frozenset("""
a about above across actually after again against all almost along already also although always
am among an and any anyone anything are arent around as at away back bad basically be because
been before behind being below best better between big bit both but by came can cant case come
could couldnt day days definitely did didnt do does doesnt doing done dont down during each else
even ever every everyone everything fact feel few first for from gave get getting give go goes
going gone gonna good got gotta great had hadnt has hasnt have havent having he her here hers
hes high him his how i id if ill im in instead into is isnt it its ive just keep kind kinda knew
know last let lets like liked literally little long look lot low made make many may maybe me
mean meant might mine more most much must my myself near need never new next no nope nor not
nothing now of off often ok okay old on one ones only onto or other our ours out over own part
people per perhaps person place point probably put quite rather really right said same saw say
says see seem seen shall she shes short should shouldnt since small so some someone something
sometimes sort sorta still such sure take tell than that thats the their theirs them then there
theres these they theyre theyve thing things think this those though thought through time times
to today together told tomorrow too took toward towards try under unless until up upon us use
used very via wanna want was wasnt way we well went were werent weve what whatever whats when
where whether which while who whom whose why will with within without wont work would wouldnt
wrong yeah year years yep yes yesterday yet you youd youll your youre yours youve
""".split())

# (what the recognizers wrote, what the speaker corrected it to, dictations seen in)
Correction = Tuple[List[str], str, int]

_cache_lock = threading.Lock()
_cache: Tuple[float, List[Correction]] = (-1.0, [])
_supplied_cache: Tuple[float, List[str]] = (-1.0, [])
_forgotten_cache: Tuple[float, Set[str]] = (-1.0, set())


def _letters(text: str) -> str:
    """The letters and digits of any script, case-folded: what a word is, not how it is typed."""
    return "".join(ch for ch in text.casefold() if ch.isalnum())


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


def _is_truncation(before: str, after: str) -> bool:
    """
    True when the "correction" is just a fragment of what was there.

    Word-level diffs cut inside words when an edit lands mid-word, so the
    corpus fills up with pairs like "them" -> "the" and "not" -> "no" that
    teach the model nothing and, in the case of negations, teach it something
    harmful. Only a *shorter* fragment is an artifact: a correction that spells
    a word out more fully ("Claud" -> "Claude") is exactly what this is for.
    """
    a, b = _letters(before), _letters(after)
    return bool(a) and bool(b) and len(b) < len(a) and b in a


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
            if _is_truncation(before, after):
                continue   # a diff artifact, not a word
            key = _letters(after)
            sessions[key].add(session)
            afters[key].append(after)
            befores[key].append(before)
            last_seen[key] = max(last_seen.get(key, 0.0), float(row.get("ts") or 0.0))
    kept = [key for key, seen in sessions.items() if len(seen) >= min_sessions]
    kept.sort(key=lambda key: (len(sessions[key]), last_seen[key]), reverse=True)
    return [(list(dict.fromkeys(befores[key])), _most_common(afters[key]), len(sessions[key]))
            for key in kept[:limit]]


def worth_priming(term: str) -> bool:
    """False for a term made only of everyday words: a recognizer gains nothing from it."""
    keys = [_letters(word) for word in term.split()]
    return any(key and key not in EVERYDAY for key in keys)


def forgotten_terms(path: Path) -> Set[str]:
    """What removed corrections corrected to, as letters; re-read only when the file changes."""
    global _forgotten_cache
    try:
        modified = path.stat().st_mtime
    except OSError:
        return set()
    with _cache_lock:
        if _forgotten_cache[0] == modified:
            return set(_forgotten_cache[1])
    keys = {_letters(line.split("#", 1)[0]) for line in path.read_text(errors="ignore").splitlines()} - {""}
    with _cache_lock:
        _forgotten_cache = (modified, keys)
    return set(keys)


def forget(after: str, path: Path = CORRECTIONS_PATH.with_name(FORGOTTEN_NAME)) -> None:
    """Stop using a learned correction; the corpus itself is kept."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(after.strip() + "\n")


def learned_corrections(path: Path = CORRECTIONS_PATH) -> List[Correction]:
    """Repeated corrections from the corpus, less any the speaker removed."""
    forgotten = forgotten_terms(path.with_name(FORGOTTEN_NAME))
    return [c for c in _repeated_from(path) if _letters(c[1]) not in forgotten]


def _repeated_from(path: Path) -> List[Correction]:
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


def supplied_terms(path: Path = SUPPLIED_PATH) -> List[str]:
    """The terms in the vocabulary file ('#' starts a comment), re-read only when it changes."""
    global _supplied_cache
    try:
        modified = path.stat().st_mtime
    except OSError:
        return []
    with _cache_lock:
        if _supplied_cache[0] == modified:
            return list(_supplied_cache[1])
    terms = []
    for line in path.read_text(errors="ignore").splitlines():
        term = " ".join(line.split("#", 1)[0].split())
        if term:
            terms.append(term)
    with _cache_lock:
        _supplied_cache = (modified, terms)
    return list(terms)


def known_terms(corrections_path: Path = CORRECTIONS_PATH,
                supplied_path: Path = SUPPLIED_PATH) -> List[str]:
    """
    Every term the speaker is known to use, sized for priming a recognizer:
    repeated corrections first (proven in their own dictations, everyday words
    left out), then the supplied ones, each once.
    """
    seen, out = set(), []
    learned = [after for _, after, _ in learned_corrections(corrections_path) if worth_priming(after)]
    for term in learned + supplied_terms(supplied_path):
        key = _letters(term)
        if key and key not in seen and len(term.split()) <= MAX_KEYTERM_WORDS:
            seen.add(key)
            out.append(term)
    return out[:MAX_KEYTERMS]


def supplied_prompt(terms: List[str]) -> str:
    """The supplied terms as the correction model sees them."""
    if not terms:
        return ""
    return ("Words and names this speaker uses: " + ", ".join(terms[:MAX_PROMPT_TERMS]) +
            ". Where the audio fits one of these, spell it this way.")


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
