#!/usr/bin/env python3
"""
How much each recognizer adds to what gets typed.

Two different kinds of value, reported separately:

  Coverage: how often a recognizer answered in time at all, and how often it
  was the only one that did (the others failed or missed the deadline).

  Words: on single-chunk dictations where two or more answered - so all of
  them heard the same audio - a word in the final text counts toward a
  recognizer when it was the only one to hear it. Those are the words the
  ensemble would have lost without it. Words no recognizer heard came from
  the correction model. Spelling variants ("e-mail"/"email", "okay"/"OK")
  are not counted as disagreements.

A word only one recognizer heard is not necessarily right - the correction
model adopted it, which is evidence, not proof. Read the examples.

    ./venv/bin/python scripts/provider_value.py               # the last 14 days
    ./venv/bin/python scripts/provider_value.py --days 3 --examples 15

Reads ~/.mergescribe/training, so training capture must be on.
"""

import argparse
import glob
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime

TRAINING = os.path.expanduser("~/.mergescribe/training")
FILLERS = {"um", "uh", "uhm", "umm", "er", "erm", "ah", "hmm", "mm", "mhm"}
ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def spell(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    return str(n)


CANONICAL = {"okay": "ok", "alright": "allright"}


def words(text: str) -> list:
    """The words of a transcript, as comparable across recognizers as cheaply possible."""
    text = (text or "").lower().replace("’", "'")
    text = re.sub(r"\d+", lambda m: " " + spell(int(m.group())) + " ", text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    out = [w.strip("'") for w in text.split()]
    return [CANONICAL.get(w, w) for w in out if w and w not in FILLERS]


def forms(seq: list) -> set:
    """Every word, and every adjacent pair run together ("set up" -> "setup")."""
    return set(seq) | {a + b for a, b in zip(seq, seq[1:])}


def heard_alone(mine: list, others: list) -> set:
    """Words in mine that none of the others heard, even spelled differently."""
    theirs = set().union(*(forms(o) for o in others))
    alone = set(mine) - theirs
    for a, b in zip(mine, mine[1:]):           # I wrote "set up", they wrote "setup"
        if a + b in theirs:
            alone -= {a, b}
    return alone


def stream_name(model: str) -> str:
    """The name an OpenRouter model's results carry (see OpenRouterSTTProvider)."""
    return "or-" + model.replace("/", "-").replace(".", "-").replace(":", "-")


def ensemble(meta: dict) -> tuple:
    config = meta.get("config_snapshot") or {}
    names = list(config.get("enabled_providers") or [])
    names += [stream_name(m) for m in config.get("openrouter_stt_models") or []]
    return tuple(sorted(names))


def load(since: float):
    for path in glob.glob(os.path.join(TRAINING, "*", "*", "metadata.json")):
        if os.path.getmtime(path) < since:
            continue
        try:
            meta = json.load(open(path))
        except (OSError, ValueError):
            continue
        if meta.get("final_output") and len(ensemble(meta)) >= 2:
            yield meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--days", type=float, default=14.0)
    parser.add_argument("--examples", type=int, default=0, help="most common words each one alone contributed")
    parser.add_argument("--min-dictations", type=int, default=20)
    args = parser.parse_args()

    groups = defaultdict(list)
    for meta in load(time.time() - args.days * 86400):
        groups[ensemble(meta)].append(meta)

    for names, sessions in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(sessions) < args.min_dictations:
            continue
        answered, sole, alone, kept, changed = Counter(), Counter(), Counter(), Counter(), Counter()
        examples = defaultdict(Counter)
        typed_words = nobody = compared = 0
        for meta in sessions:
            results = [r for r in meta.get("transcriptions") or [] if (r.get("text") or "").strip()]
            heard = defaultdict(list)
            for r in results:
                heard[r["provider"]] += words(r["text"])
            for name in heard:
                answered[name] += 1
            if len(heard) == 1:
                sole[next(iter(heard))] += 1
            # One result per recognizer per mic means one chunk: everyone heard the same audio.
            single = max(Counter((r["provider"], r.get("mic")) for r in results).values(), default=0) == 1
            if not single or len(heard) < 2:
                continue
            compared += 1
            final = set(words(meta["final_output"]))
            typed_words += len(final)
            nobody += len(final - set().union(*(forms(w) for w in heard.values())))
            for name, mine in heard.items():
                only = heard_alone(mine, [w for n, w in heard.items() if n != name])
                used = only & final
                alone[name] += len(only)
                kept[name] += len(used)
                changed[name] += bool(used)
                examples[name].update(used)

        first = min(datetime.fromisoformat(m["timestamp"]) for m in sessions if m.get("timestamp"))
        n = len(sessions)
        print(f"\n{' + '.join(names)}")
        print(f"{n} dictations since {first:%b %d}; words compared on the {compared} single-chunk "
              f"ones where two or more answered\n")
        print(f"  {'recognizer':38s} {'answered':>8s} {'only answer':>11s} {'only it heard':>14s} "
              f"{'kept':>6s} {'dictations changed':>19s} {'kept per 1k words':>18s}")
        for name in names:
            print(f"  {name:38s} {answered[name] / n:8.0%} {sole[name]:11d} {alone[name]:14d} {kept[name]:6d} "
                  f"{changed[name]:9d} ({changed[name] / max(1, compared):5.1%}) "
                  f"{1000 * kept[name] / max(1, typed_words):12.1f}")
        print(f"  {'(no recognizer: the correction model)':38s} {'':8s} {'':11s} {'':14s} {nobody:6d} "
              f"{'':19s} {1000 * nobody / max(1, typed_words):12.1f}")
        if args.examples:
            for name in names:
                common = ", ".join(w for w, _ in examples[name].most_common(args.examples))
                print(f"    {name}: {common}")


if __name__ == "__main__":
    main()
