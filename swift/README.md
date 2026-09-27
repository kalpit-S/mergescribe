# MergeScribe for macOS, in Swift

The native port of the Python app in `../mergescribe/`. Same behaviour, same files under
`~/.mergescribe/` (settings.json, .env, metrics.jsonl, corrections.jsonl, vocabulary.txt,
training captures, logs), so the Python scripts in `../scripts/` keep working on its output.

## Build and run

```bash
swift/scripts/build-app.sh          # release build -> swift/build/MergeScribe.app
open swift/build/MergeScribe.app
```

Quit the Python app first: both listen for the same trigger key.

The build uses the Command Line Tools toolchain (Swift 6.2 and the macOS 26 SDK, which
SpeechAnalyzer needs), with no third-party dependencies.

### Permissions

On first use macOS asks for three: **Microphone**, **Input Monitoring** (to hear the trigger
key) and **Accessibility** (to type, and to bring the dictation's window back). They are tied
to the app's signature. An ad-hoc signed build changes signature with every build and is asked
again each time; signing with a stable identity avoids that. Create a self-signed code-signing
certificate named `MergeScribe Dev` once (Keychain Access > Certificate Assistant > Create a
Certificate, type Code Signing) and the build script uses it.

## Developing

```bash
cd swift
scripts/test.sh                                   # the tests, a second or two
swift run --scratch-path .build-clt MergeScribe replay ~/.mergescribe/training/<date>/<session>   # a saved dictation, end to end
swift run --scratch-path .build-clt MergeScribe stats                       # the Stats page's numbers
swift run --scratch-path .build-clt MergeScribe devices                     # input devices Core Audio sees
swift run --scratch-path .build-clt MergeScribe hud-preview [--loop]        # the HUD playing a scripted dictation
swift run --scratch-path .build-clt MergeScribe hud-frames <dir>            # the same, rendered to PNGs
swift run --scratch-path .build-clt MergeScribe settings-preview            # the Settings window on its own
swift run --scratch-path .build-clt MergeScribe apple-eval out.jsonl          # Apple's recognizer over saved dictations
```

`replay` needs no microphone or accessibility permission: it runs a saved recording through
the chunker, the real recognizers, Jev and the correction model, and prints what would have
been typed. The saved audio has its pauses trimmed, so its chunks differ from the original's.

Parity with the Python code is checked against goldens: `../scripts/swift_goldens.py` runs the
Python implementation on invented inputs (diffs, edit classification, prompts, Jev questions and
decisions, chunking of scripted audio) and the Swift tests assert the same answers.

## Layout

| Target | What |
|--------|------|
| `MergeScribeCore` | Foundation only, all of it testable without a screen: settings, the session pipeline (chunk transcription, deadlines, consensus, the Jev/correction race, streaming output), the chunker, the hotkey state machine, OpenRouter clients, difflib's SequenceMatcher, edit classification, vocabulary, metrics, training capture, stats, the HUD's animation model and aurora. Platform work sits behind protocols (`Recognizer`, `Focus`, `TextOutput`, `EditWatching`, `Correcting`, `Judging`). |
| `MergeScribeMac` | The Mac behind those protocols: capture with one AVAudioEngine per microphone, a listen-only event tap for the trigger key, CGEvent typing, accessibility focus and return-to-window, the edit watcher, Core Audio ducking, the HUD panel. |
| `MergeScribe` | The menu bar app, the SwiftUI Settings window, the log file, and the commands above. |

## What differs from the Python app

- **One speech detector.** The chunker's adaptive detector (each mic's own noise floor, sampled
  all the time) also decides whether the last chunk holds speech, and pre-roll counts. The
  Python app had a second, fixed gate (0.35s at -55 dBFS) that let room tone through in loud
  rooms. Until a mic's floor is measured the old gate still decides, and the log notes any
  final chunk where the two would disagree.
- **Recognizers per recording.** Each dictation builds its recognizer set from the settings of
  the moment, so a change in Settings never swaps them under a session that is finishing.
- **Typing from a private event source**, so a modifier you are holding can't turn text into
  shortcuts.
- **Parakeet** (local transcription) is not in this build yet; its place is FluidAudio (Core ML).
