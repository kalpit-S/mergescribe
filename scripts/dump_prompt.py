"""
Print the EXACT prompt the correction model receives, for your current screen.

Uses the same code paths as a real dictation, so what you see here is what the
model sees. Run it from the terminal you launch MergeScribe from (it needs the
same Accessibility permission):

    ./venv/bin/python scripts/dump_prompt.py
    ./venv/bin/python scripts/dump_prompt.py "pretend I said this"
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mergescribe.config import Config  # noqa: E402
from mergescribe.context import get_app_context  # noqa: E402
from mergescribe.correct import _build_prompt, build_system_prompt  # noqa: E402
from mergescribe.fields import snapshot_fields  # noqa: E402
from mergescribe.types import TranscriptionResult  # noqa: E402


def main() -> None:
    spoken = sys.argv[1] if len(sys.argv) > 1 else "okay so what do you think about this approach"

    config = Config.load().snapshot()
    context = get_app_context()

    targets = []
    walk_seconds = 0.0
    if config.field_routing_enabled:
        # Electron accessibility trees populate lazily after the first poke.
        # The running app has usually woken them already; a fresh process has
        # not, so warm up once and measure the second pass.
        snapshot_fields(allowed_apps=config.routing_allowed_apps)
        start = time.time()
        targets = snapshot_fields(allowed_apps=config.routing_allowed_apps)
        walk_seconds = time.time() - start

    results = [TranscriptionResult(
        text=spoken, provider="parakeet", mic="MacBook Pro Microphone", latency_ms=0,
    )]

    # History is per-process, so a real session's prompt would also include a
    # "Recent dictations..." line; shown here as a placeholder for shape.
    history = "[to Claude: Claude] example previous dictation"

    system_prompt = build_system_prompt(config, targets, config.custom_instructions)
    user_prompt = _build_prompt(results, context, history, targets)

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
    print(f"routing enabled   : {config.field_routing_enabled}")
    print(f"allowed apps      : {config.routing_allowed_apps or '(all)'}")
    print(f"targets found     : {len(targets)}  (AX walk took {walk_seconds:.2f}s)")
    by_app: dict = {}
    for t in targets:
        by_app.setdefault(t.app_name, set()).add(t.window_title)
    for app, windows in by_app.items():
        print(f"  {app}: {len(windows)} window(s)")

    # Harvest quality: how much window text did we actually get per target?
    # 0 chars means the model is routing on the window title alone.
    if targets:
        print()
        print("window content harvested per target:")
        for t in targets:
            sample = t.window_context or t.value_preview
            n = len(sample)
            flag = "  <-- EMPTY, title-only routing" if n == 0 else ""
            label = (t.window_title or t.label)[:44]
            print(f"  [{t.id}] {t.app_name:<16} {label:<46} {n:>4} chars{flag}")
    total_chars = len(system_prompt) + len(user_prompt)
    print(f"prompt size       : {total_chars} chars, ~{total_chars // 4} tokens")


if __name__ == "__main__":
    main()
