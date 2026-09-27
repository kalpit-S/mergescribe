#!/usr/bin/env python3
"""
How well the judge decides "type it as it is" vs "correct it", on your own dictations.

Label: what the correction model actually produced for logged single-chunk
dictations - only ones corrected by the model configured now, since the label
means "what today's correction step would do". Corrections longer than every
transcript are dropped: the step never adds words, so those are correction bugs
(an older model leaked earlier dictations into its output), not labels. A
dictation is clean when one of its transcripts already says those
words (punctuation, case, spacing and number formatting aside - the punctuation
question covers those); a skip is right only when the transcript the judge
picks is one of them. Uses the judge's own questions() and decide(), with your
written preferences and each dictation's window, so this measures what ships.
The label ignores case, so it can't tell whether a preference was kept. The app
follows the latest Jev; run this when it moves.

    ./venv/bin/python scripts/judge_eval.py                        # 600 dictations, shipped model
    ./venv/bin/python scripts/judge_eval.py --model ~typesafe/jev-latest

Costs a few cents; answers are cached per model in ~/.mergescribe/judge_eval/.
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mergescribe.config import Config  # noqa: E402
from mergescribe.judge import (DECISIONS_URL, JUDGE_MODEL_DEFAULT, LIMIT,  # noqa: E402
                               _candidates, decide, preferences, questions)
from mergescribe.types import AppContext, TranscriptionResult  # noqa: E402

TRAINING = os.path.expanduser("~/.mergescribe/training")
ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        "sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
CONTRACTED = {"wanna": "want to", "gonna": "going to", "gotta": "got to", "kinda": "kind of", "okay": "ok"}


def spell(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + spell(n % 100))
    if n < 1000000:
        return spell(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + spell(n % 1000))
    return str(n)


def words(text: str) -> str:
    """What the words are, not how they are written. Fillers are kept: dropping them is correction's job."""
    text = (text or "").lower().replace("’", "'").replace("-", " ")
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)
    text = re.sub(r"\d+", lambda m: " " + spell(int(m.group())) + " ", text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    out = [CONTRACTED.get(w, w) for w in (w.strip("'") for w in text.split()) if w]
    return "".join(w for w in out if w != "and").replace(" ", "")


def dictations(count: int, model: str = "", seed: int = 7):
    """Logged single-chunk dictations the correction model handled, 2-60 words."""
    paths = sorted(glob.glob(os.path.join(TRAINING, "*", "*", "metadata.json")))
    random.Random(seed).shuffle(paths)
    for path in paths:
        try:
            meta = json.load(open(path))
        except (OSError, ValueError):
            continue
        if meta.get("output_method") != "streamed" or not (meta.get("final_output") or "").strip():
            continue
        results = [r for r in meta.get("transcriptions") or [] if (r.get("text") or "").strip()]
        streams = [(r["provider"], r.get("mic")) for r in results]
        if not results or len(streams) != len(set(streams)):
            continue            # more than one chunk: the judge never sees these
        if not 2 <= len(meta["final_output"].split()) <= 60:
            continue
        used = ((meta.get("llm_correction") or {}).get("model") or "").split(":")[0]
        if model and used != model.split(":")[0]:
            continue
        if len(meta["final_output"].split()) > max(len(r["text"].split()) for r in results) * 1.25 + 2:
            continue            # the correction added words: a correction bug, not a label
        candidates = _candidates([TranscriptionResult(text=r["text"], provider=r["provider"],
                                                      mic=r.get("mic") or "", latency_ms=0) for r in results])
        window = meta.get("app_context") or {}
        context = AppContext(window.get("app_name") or "", window.get("window_title") or "",
                             window.get("bundle_id") or "")
        yield {"id": meta["session_id"], "candidates": candidates, "corrected": meta["final_output"],
               "context": context}
        count -= 1
        if count == 0:
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-n", type=int, default=600)
    parser.add_argument("--model", default=JUDGE_MODEL_DEFAULT)
    parser.add_argument("--any-correction-model", action="store_true",
                        help="label with dictations from every correction model, not just the current one")
    args = parser.parse_args()

    config = Config.load()
    key = config.openrouter_api_key
    correction_model = "" if args.any_correction_model else config.openrouter_correction_model
    if not key:
        sys.exit("No OpenRouter key: set one in Settings or ~/.mergescribe/.env")
    cache_path = Path.home() / ".mergescribe" / "judge_eval" / f"{re.sub(r'[^a-z0-9.]+', '-', args.model)}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    lock = threading.Lock()

    def ask(d):
        asking = questions(len(d["candidates"]), preferences(config), d["context"])
        asked = json.dumps({"q": asking, "t": d["candidates"]}, sort_keys=True)
        request_key = f"{d['id']}|{hashlib.sha1(asked.encode()).hexdigest()[:12]}"
        if request_key in cache:
            return d, cache[request_key]
        body = json.dumps({"model": args.model, "state": {"transcripts": d["candidates"]},
                           "questions": asking}).encode()
        request = urllib.request.Request(DECISIONS_URL, body, {"Authorization": f"Bearer {key}",
                                                                "Content-Type": "application/json"})
        for attempt in range(3):
            try:
                started = time.monotonic()
                with urllib.request.urlopen(request, timeout=30) as response:
                    reply = json.load(response)
                reply["ms"] = (time.monotonic() - started) * 1000
                break
            except Exception as e:
                reply = {"error": str(e)}
                time.sleep(1 + attempt)
        if "answers" in reply:                       # never cache a failure
            with lock:
                cache[request_key] = reply
                cache_path.write_text(json.dumps(cache))
        return d, reply

    with ThreadPoolExecutor(8) as pool:
        replies = list(pool.map(ask, dictations(args.n, correction_model)))
    answered = [(d, r) for d, r in replies if "answers" in r]
    if not answered:
        sys.exit(f"No answers: {replies[0][1].get('error') if replies else 'no logged dictations found'}")

    rows = []
    for d, reply in answered:
        _, pick, scores = decide(reply["answers"], len(d["candidates"]))
        right = {i for i, c in enumerate(d["candidates"]) if words(c) == words(d["corrected"])}
        rows.append((max(scores.values()), pick in right, bool(right), reply.get("ms", 0.0)))
    clean = sum(r[2] for r in rows)
    print(f"{len(rows)} single-chunk dictations corrected by {correction_model or 'any model'}, "
          f"judged by {args.model}; {clean} ({clean / len(rows):.0%}) "
          f"had a transcript that already said what the correction produced\n")
    for limit in sorted({0.2, 0.25, LIMIT, 0.35, 0.4, 0.5}):
        skipped = [r for r in rows if r[0] <= limit]
        wrong = sum(not r[1] for r in skipped)
        print(f"  limit {limit:<5}{' (shipped)' if limit == LIMIT else '          '} skips {len(skipped) / len(rows):5.1%}"
              f" of dictations | catches {(len(skipped) - wrong) / max(1, clean):5.1%} of the clean ones"
              f" | wrong skips {wrong:3d} ({wrong / max(1, len(skipped)):5.1%} of skips)")
    ms = sorted(r[3] for r in rows)
    print(f"\n  latency p50 {ms[len(ms) // 2]:.0f}ms, p90 {ms[int(len(ms) * 0.9)]:.0f}ms")


if __name__ == "__main__":
    main()
