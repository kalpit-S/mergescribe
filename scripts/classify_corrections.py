"""
Classify captured corrections, then emit Parakeet training pairs.

Every edit you make to dictated text falls into one of two buckets, and only
one of them is training data:

  transcription  the words were heard wrong      ("post grass" -> "Postgres")
  intent         the words were right, you       ("three months" -> "four months")
                 changed your mind
  formatting     punctuation/casing/whitespace only
  unrelated      the span was replaced by something else entirely

Telling these apart needs semantics, so a cheap LLM does it in batch rather
than in the hot path. Only `transcription` rows become fine-tuning pairs:
training Parakeet on an intent change would teach it to hear words that were
never said.

    ./venv/bin/python scripts/classify_corrections.py            # classify new rows
    ./venv/bin/python scripts/classify_corrections.py --export   # write training pairs
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mergescribe.config import Config  # noqa: E402
from mergescribe.correct import _call_openrouter  # noqa: E402
from mergescribe.feedback import CORRECTIONS_PATH, load_corrections  # noqa: E402

CLASSIFIED_PATH = Path.home() / ".mergescribe" / "corrections_classified.jsonl"
TRAINING_PAIRS_PATH = Path.home() / ".mergescribe" / "training_pairs.jsonl"

CLASSIFIER_MODEL = "google/gemini-3.1-flash-lite"

SYSTEM = """You label edits a user made to text produced by a speech-to-text system.

For each edit, decide why the text changed:
- "transcription": the system misheard. The user's edit restores what was actually said
  (wrong word, wrong name, wrong number that reads like a mishearing, garbled phrase).
- "intent": the system heard correctly but the user changed their mind about content
  (different number because they decided differently, reworded an idea, added a thought).
- "formatting": only punctuation, capitalisation, whitespace or line breaks changed.
- "unrelated": the span was replaced with entirely different content.

Judge by whether the ORIGINAL is a plausible mishearing of the CORRECTED text.
"three months" -> "four months" is intent: those do not sound alike.
"cruise tickets" -> "Cruz tickets" is transcription: they sound identical.

Reply with one line of JSON and nothing else:
{"label": "...", "confidence": 0.0-1.0, "note": "<8 words"}"""


def classify_one(config, entry: dict) -> dict:
    prompt = (
        f"ORIGINAL (what the system typed):\n{entry.get('typed', '')}\n\n"
        f"CORRECTED (what the user changed it to):\n{entry.get('corrected', '')}"
    )
    raw = _call_openrouter(prompt, SYSTEM, config, timeout=20)
    try:
        start, end = raw.find("{"), raw.rfind("}")
        return json.loads(raw[start:end + 1])
    except Exception:
        return {"label": "unparsed", "confidence": 0.0, "note": raw[:60]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true",
                    help="write training pairs from already-classified rows")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    if args.export:
        rows = [json.loads(x) for x in CLASSIFIED_PATH.open()] if CLASSIFIED_PATH.exists() else []
        pairs = [
            {
                "session_id": r["session_id"],
                "audio_hint": f"~/.mergescribe/training/*/{r['session_id']}/",
                "raw_transcript": r.get("raw_transcript", ""),
                "corrected_text": r.get("corrected", ""),
                "confidence": r.get("confidence", 0),
            }
            for r in rows
            if r.get("label") == "transcription" and r.get("confidence", 0) >= 0.6
        ]
        with TRAINING_PAIRS_PATH.open("w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"Wrote {len(pairs)} training pairs -> {TRAINING_PAIRS_PATH}")
        print("Each joins to its audio by session_id; see PARAKEET_FINETUNE_PLAN.md")
        return

    corrections = load_corrections()
    if not corrections:
        print(f"No corrections yet at {CORRECTIONS_PATH}")
        return

    done = set()
    if CLASSIFIED_PATH.exists():
        for line in CLASSIFIED_PATH.open():
            try:
                done.add(json.loads(line)["session_id"])
            except Exception:
                pass

    todo = [c for c in corrections if c.get("session_id") not in done][:args.limit]
    print(f"{len(corrections)} corrections, {len(done)} already classified, {len(todo)} to do\n")
    if not todo:
        return

    config = Config.load().snapshot()
    config.openrouter_correction_model = CLASSIFIER_MODEL
    config.openrouter_correction_reasoning_effort = ""

    counts = {}
    with CLASSIFIED_PATH.open("a") as out:
        for i, entry in enumerate(todo, 1):
            result = classify_one(config, entry)
            label = result.get("label", "unparsed")
            counts[label] = counts.get(label, 0) + 1
            out.write(json.dumps({**entry, **result}) + "\n")
            print(f"[{i}/{len(todo)}] {label:<14} {entry.get('typed','')[:60]!r}")
            if label == "transcription":
                print(f"{'':18} -> {entry.get('corrected','')[:60]!r}")

    print("\nlabels:", counts)
    print("Run with --export to write Parakeet training pairs.")


if __name__ == "__main__":
    main()
