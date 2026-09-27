"""
Print the EXACT prompt the correction model receives, for your current screen.

Uses the same code paths as a real dictation, so what you see here is what the
model sees. Run it from the terminal you launch MergeScribe from (it needs the
same Accessibility permission):

    ./venv/bin/python scripts/dump_prompt.py
    ./venv/bin/python scripts/dump_prompt.py "pretend I said this"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mergescribe.config import Config  # noqa: E402
from mergescribe.context import get_app_context  # noqa: E402
from mergescribe.correct import _build_prompt, build_system_prompt  # noqa: E402
from mergescribe.types import TranscriptionResult  # noqa: E402


def main() -> None:
    spoken = sys.argv[1] if len(sys.argv) > 1 else "okay so what do you think about this approach"

    config = Config.load().snapshot()
    context = get_app_context()

    results = [TranscriptionResult(
        text=spoken, provider="parakeet", mic="MacBook Pro Microphone", latency_ms=0,
    )]

    # History is per-process, so a real session's prompt would also include a
    # "Recent dictations..." line; shown here as a placeholder for shape.
    history = "[to Claude: Claude] example previous dictation"

    system_prompt = build_system_prompt(config, config.custom_instructions)
    user_prompt = _build_prompt(results, context, history)

    print("=" * 72)
    print("SYSTEM PROMPT")
    print("=" * 72)
    print(system_prompt)
    print()
    print("=" * 72)
    print("USER PROMPT")
    print("=" * 72)
    print(user_prompt)
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    total_chars = len(system_prompt) + len(user_prompt)
    print(f"prompt size       : {total_chars} chars, ~{total_chars // 4} tokens")


if __name__ == "__main__":
    main()
