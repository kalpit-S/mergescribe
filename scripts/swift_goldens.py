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


def result(text, provider, mic="m", chunk=0):
    from mergescribe.types import TranscriptionResult
    return TranscriptionResult(text=text, provider=provider, mic=mic, latency_ms=0, chunk=chunk)


def prompts():
    from unittest.mock import Mock, patch
    from mergescribe.correct import DEFAULT_EDITING_PROMPT, DEFAULT_SYSTEM_CONTEXT, _build_prompt, build_system_prompt
    from mergescribe.types import AppContext, ConfigSnapshot

    slack = AppContext(app_name="Slack", window_title="general - Acme - Slack", bundle_id="com.tinyspeck.slackmacgap")
    notitle = AppContext(app_name="Notes", window_title="", bundle_id="com.apple.Notes")
    inputs = [
        ([result("ship it friday", "a")], None),
        ([result("ship it friday", "a"), result("Ship it Friday.", "b"), result("ship it  friday", "c", "m2")], slack),
        ([result("part two", "a", chunk=2), result("ship it", "a", chunk=1), result("ship it", "b", chunk=1),
          result("ship it", "a", chunk=3), result("", "b", chunk=3)], notitle),
        ([result("Ça marche 🚀", "a"), result("ça marche", "b")], slack),
    ]
    built = [{"results": [{"text": r.text, "provider": r.provider, "mic": r.mic, "chunk": r.chunk} for r in rs],
              "context": None if c is None else {"app_name": c.app_name, "window_title": c.window_title, "bundle_id": c.bundle_id},
              "prompt": _build_prompt([r for r in rs if r.text.strip()], c)} for rs, c in inputs]

    systems = []
    for custom_prompt, instructions, learn in [("", "", False), ("", "In Slack, write in lowercase.", True),
                                               ("Translate to Spanish.", "", True)]:
        config = Mock(spec=ConfigSnapshot)
        config.system_prompt = custom_prompt
        config.learn_vocabulary = learn
        with patch("mergescribe.vocabulary.learned_corrections", return_value=[(["cloud", "Claud"], "Claude", 2)]), \
             patch("mergescribe.vocabulary.supplied_terms", return_value=["Postgres", "Kubernetes operator"]):
            systems.append({"system_prompt": custom_prompt, "custom_instructions": instructions, "learn": learn,
                            "learned": [[["cloud", "Claud"], "Claude", 2]], "supplied": ["Postgres", "Kubernetes operator"],
                            "built": build_system_prompt(config, instructions)})
    return {"default_system": DEFAULT_SYSTEM_CONTEXT, "default_editing": DEFAULT_EDITING_PROMPT,
            "prompts": built, "systems": systems}


def judge():
    from mergescribe.judge import LIMIT, decide, destination, questions
    from mergescribe.types import AppContext

    slack = AppContext(app_name="Slack", window_title="general - Acme - Slack", bundle_id="x")
    asked = []
    for count, prefs, context in [(1, "", None), (3, "", None), (2, "In Slack, lowercase.", slack),
                                  (1, "Be formal.", AppContext(app_name="Notes", window_title="", bundle_id="y")),
                                  (1, "Be formal.", None)]:
        asked.append({"count": count, "preferences": prefs,
                      "context": None if context is None else {"app_name": context.app_name,
                                                               "window_title": context.window_title,
                                                               "bundle_id": context.bundle_id},
                      "questions": questions(count, prefs, context)})
    rng = random.Random(3)
    decisions = []
    for n in range(200):
        count = rng.randint(1, 3)
        with_prefs = rng.random() < 0.5
        probabilities = {str(i): rng.random() for i in range(count)}
        probabilities["none"] = rng.random() * 0.6
        if rng.random() < 0.15:
            probabilities = {}
        answers = {"best": {"choice": rng.choice([str(i) for i in range(count)] + ["none", "7"]),
                            "probabilities": probabilities}}
        for i in range(count):
            for job in ["filler", "restart", "misheard", "punct"] + (["prefs"] if with_prefs else []):
                answers[f"{job}_{i}"] = {"noul": round(rng.random() ** 3, 3)}
        clean, pick, scores = decide(answers, count)
        decisions.append({"count": count, "answers": answers, "clean": clean, "pick": pick, "scores": scores})
    return {"limit": LIMIT, "questions": asked, "decisions": decisions}


def chunker():
    """Scripted audio through the Python chunker, with a fake clock: what it emits, and when."""
    from collections import deque
    from unittest.mock import patch

    import numpy as np

    import mergescribe.audio as audio_module
    from mergescribe.audio import FALLBACK_NOISE_FLOOR_DB, AudioEngine
    from mergescribe.config import Config

    BLOCK = 1024

    def block(amp):
        # A square wave: its RMS is exactly amp, in float32 and in Double alike.
        out = np.full(BLOCK, amp, dtype=np.float32)
        out[1::2] *= -1
        return out

    def run(mics, script):
        config = Config()
        config.enabled_mics = list(mics)
        engine = AudioEngine(config)
        preroll_chunks = int(engine._preroll_samples / BLOCK)
        floor_samples = int(audio_module.NOISE_FLOOR_WINDOW_SECONDS * config.sample_rate / BLOCK)
        for mic in mics:
            engine.preroll_buffers[mic] = deque(maxlen=preroll_chunks)
            engine.current_chunk[mic] = []
            engine._noise_floor_samples[mic] = deque(maxlen=floor_samples)
            engine._noise_floor_cache[mic] = FALLBACK_NOISE_FLOOR_DB
            engine._speech_active[mic] = False
        engine._primary_mic = mics[0]
        clock = [1000.0]
        events = []

        def emitted(chunk):
            events.append({"at": step[0], "lengths": [len(chunk[m]) for m in mics]})
        engine.on_chunk_ready = emitted
        step = [0]
        with patch.object(audio_module.time, "time", lambda: clock[0]), \
             patch.object(audio_module.time, "monotonic", lambda: clock[0]):
            for index, entry in enumerate(script):
                step[0] = index
                if entry == "start":
                    engine.start_recording()
                    continue
                if entry == "stop":
                    final = engine.stop_recording()
                    events.append({"at": index, "final": [len(final[m]) for m in mics]})
                    continue
                for mic, amp in zip(mics, entry):
                    engine._audio_callback(mic, block(amp).reshape(-1, 1), BLOCK, None, None)
                clock[0] += BLOCK / config.sample_rate
        floors = [engine._noise_floor_cache[m] for m in mics]
        return {"mics": list(mics), "script": script, "events": events, "floors": floors}

    room, speech, quiet = 0.001, 0.1, 0.0002
    one = lambda amp, n: [[amp]] * n
    two = lambda a, b, n: [[a, b]] * n
    scenarios = [
        # A pause ends a chunk; the rest comes on release.
        one(room, 100) + ["start"] + one(speech, 60) + one(room, 25) + one(speech, 30) + ["stop"],
        # Short pauses don't cut; past 30s the next breath does; past 45s it is cut anyway.
        two(room, room, 80) + ["start"] + (two(speech, 0.05, 40) + two(room, room, 8)) * 12
            + two(speech, speech, 720) + ["stop"],
        # Holding the key in silence emits nothing.
        one(room, 50) + ["start"] + one(room, 600) + ["stop"],
        # Speech just before the press is kept; stale pre-roll isn't possible with a steady clock.
        one(room, 60) + one(speech, 5) + ["start"] + one(speech, 20) + ["stop"],
        # A noisy room raises the floor; speech above it still counts.
        one(0.02, 120) + ["start"] + one(0.3, 70) + one(0.02, 30) + one(0.3, 10) + ["stop"],
        # A very quiet room: the floor drops well below the fallback.
        one(quiet, 200) + ["start"] + one(0.004, 50) + one(quiet, 30) + ["stop"],
    ]
    return {"block": BLOCK, "scenarios": [run(["mic1"] if len(sc[0]) == 1 else ["mic1", "mic2"], sc)
                                          for sc in scenarios]}


def write(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"wrote {name}.json")


if __name__ == "__main__":
    write("sequence_matcher", sequence_matcher())
    write("feedback", feedback())
    write("prompts", prompts())
    write("judge", judge())
    write("chunker", chunker())
    from mergescribe.vocabulary import EVERYDAY
    write("everyday", sorted(EVERYDAY))
