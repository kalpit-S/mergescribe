#!/usr/bin/env python3
"""
Golden fixtures for the Swift port: what the Python code returns for a set of inputs.

The Swift tests load these and assert the same answers, so the port is checked against
the behaviour it replaces rather than against a re-reading of it. Every input is invented
or random (the repository is public): no logged dictation goes in here.

    ./venv/bin/python scripts/swift_goldens.py
"""

import difflib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "swift" / "Tests" / "MergeScribeCoreTests" / "Goldens"

ALPHABETS = ["ab c", "abcde ", "aé ü🚀字", "the quick brown fox ", "éé x"]


def opcodes(sm):
    return [[tag, i1, i2, j1, j2] for tag, i1, i2, j1, j2 in sm.get_opcodes()]


def sequence_matcher():
    rng = random.Random(11)
    cases = []
    for n in range(400):
        alphabet = ALPHABETS[n % len(ALPHABETS)]
        long = n % 10 == 0                        # past 200 elements, where autojunk kicks in
        size = (lambda: rng.randint(180, 320)) if long else (lambda: rng.randint(0, 40))
        a = "".join(rng.choice(alphabet) for _ in range(size()))
        b = "".join(rng.choice(alphabet) for _ in range(size()))
        autojunk = n % 3 != 0
        sm = difflib.SequenceMatcher(None, a, b, autojunk=autojunk)
        m = sm.find_longest_match(0, len(a), 0, len(b))
        cases.append({"a": a, "b": b, "autojunk": autojunk, "opcodes": opcodes(sm),
                      "ratio": sm.ratio(), "longest": [m.a, m.b, m.size]})
    words = "we should ship the migration on friday not monday it is fine".split()
    word_cases = []
    for n in range(150):
        a = [rng.choice(words) for _ in range(rng.randint(0, 14))]
        b = [rng.choice(words) for _ in range(rng.randint(0, 14))]
        word_cases.append({"a": a, "b": b,
                           "opcodes": opcodes(difflib.SequenceMatcher(None, a, b, autojunk=False))})
    return {"strings": cases, "words": word_cases}


SENTENCES = [
    "Can you check the retry logic for post grass and rerun the migrations?",
    "Quiet hiking trails near the coast in early spring.",
    "We should move the standup to Thursday, not Friday.",
    "Ça marche, on se voit demain à dix heures 🚀",
    "The build uses 16 kHz mono audio, which is plenty for speech.",
]
EDITS = [
    ("post grass", "Postgres"), ("Thursday", "Wednesday"), ("rerun", "re-run"), ("not", "and"),
    ("demain", "ce soir"), ("16", "sixteen"), ("plenty", "more than enough"), ("coast", "Coast"),
]


def mutate(rng, base, typed_at, typed):
    """One plausible thing that happens to a field after dictation."""
    kind = rng.randrange(10)
    if kind == 0:
        return base
    if kind == 1:
        return base + " And another sentence typed after."
    if kind == 2:
        return "Something new at the top.\n" + base
    if kind == 3:
        for old, new in rng.sample(EDITS, len(EDITS)):
            if old in typed:
                return base[:typed_at] + typed.replace(old, new, 1) + base[typed_at + len(typed):]
        return base.replace(" ", "  ", 1)
    if kind == 4:
        return base.lower()
    if kind == 5:
        return base.replace(",", "").replace(".", "").replace("?", "")
    if kind == 6:
        return rng.choice(["", "   ", "Ask ChatGPT", "Message #general", "Follow up"])
    if kind == 7:
        cut = rng.randint(0, len(base))
        return base[:cut]
    if kind == 8:
        i = rng.randint(0, len(base))
        return base[:i] + rng.choice(["x", "🚀", "é", " word "]) + base[i:]
    return base.replace(typed, typed.replace(" ", "+"))


def feedback():
    from mergescribe.feedback import _anchor, classify
    rng = random.Random(5)
    cases = []
    for n in range(400):
        typed = rng.choice(SENTENCES)
        before = rng.choice(["", "Earlier text. ", "intro 字 ", "One\ntwo. "])
        after = rng.choice(["", " outro.", " 🚀 done", "\nPS"])
        base = before + typed + after
        span = _anchor(base, typed)
        current = mutate(rng, base, len(before), typed)
        outcome, corrected = classify(base, current, span)
        cases.append({"baseline": base, "typed": typed, "current": current, "span": list(span),
                      "outcome": outcome, "corrected": corrected})
    gone = classify(SENTENCES[0], None, (0, len(SENTENCES[0])))
    anchors = []
    for n in range(120):
        typed = rng.choice(SENTENCES)
        variant = rng.randrange(4)
        field = typed if variant == 0 else (
            typed.replace("  ", " ") if variant == 1 else
            ("x" * rng.randint(0, 30)) + typed[:-3] + "…" if variant == 2 else
            typed[5:])
        field = rng.choice(["", "pre ", "a" * 70]) + field + rng.choice(["", " post"])
        span = _anchor(field, typed)
        anchors.append({"field": field, "typed": typed, "span": list(span) if span else None})
    return {"classify": cases, "gone": list(gone), "anchor": anchors}


def write(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"wrote {name}.json")


if __name__ == "__main__":
    write("sequence_matcher", sequence_matcher())
    write("feedback", feedback())
