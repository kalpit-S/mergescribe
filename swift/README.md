# MergeScribe for macOS, in Swift

The native port of the Python app in `../mergescribe/`. Same behaviour, same files under
`~/.mergescribe/` (settings.json, .env, metrics.jsonl, corrections.jsonl, vocabulary.txt,
training captures, logs), so the Python scripts in `../scripts/` keep working on its output.

## Build and run

```bash
swift/scripts/build-app.sh          # release build -> swift/build/MergeScribe.app
open swift/build/MergeScribe.app
```

Quit the Python app first: both listen for the same trigger key. Opening MergeScribe again while
it runs (Spotlight, `open`, or the binary from a terminal) opens its Settings instead of a second
copy, and `MergeScribe settings` does the same: the way in when a full menu bar hides the icon.
Settings > Recording can also bind a second key that opens Settings directly from anywhere, same
mechanism as the trigger key; off by default, since a good choice depends on what else is bound.

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
build/MergeScribe.app/Contents/MacOS/MergeScribe mic-test [seconds] [times]   # open and close the enabled mics as dictations do
swift run --scratch-path .build-clt MergeScribe stream-test <audio.wav>     # a saved recording streamed to AssemblyAI at speaking pace
swift run --scratch-path .build-clt MergeScribe hud-preview [--loop]        # the HUD playing a scripted dictation
swift run --scratch-path .build-clt MergeScribe hud-frames <dir>            # the same, rendered to PNGs
swift run --scratch-path .build-clt MergeScribe settings-preview [Stats]    # the Settings window on its own, at a section
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
- **Each mic through its own input-only HAL unit**, pointed at that device before it starts.
  AVAudioEngine's input shares an I/O unit with the default output: with a Bluetooth headset
  connected (16 kHz in its call mode) it failed to start or ran silent, and it could open the
  headset's mic. The mics open on the key press (about 35-50 ms until audio arrives) unless
  "Keep the microphone on" is set.
- **Streaming (optional).** With an AssemblyAI key and "Stream while you talk" on, the dictation
  streams to Universal 3.6 Pro as it is spoken and isn't cut into chunks; the transcript is final
  about 150 ms after the release (vs about 0.84 s waiting on the batch recognizers). The session
  opens on the key press (~0.35 s); audio heard before it opens is held and sent then. A stream that
  fails or stalls past 2 s falls back to batch Universal on the whole recording. On 114 of the
  speaker's recordings the typed text matched the batch ensemble's.
- **Recognizers per recording.** Each dictation builds its recognizer set from the settings of
  the moment, so a change in Settings never swaps them under a session that is finishing.
- **Typing from a private event source**, so a modifier you are holding can't turn text into
  shortcuts.
- **The whole dictation in the log.** `session_complete` keeps the full text (the Python app kept
  500 characters) and says how it was typed (`output_method`), for History in Settings.
- **Parakeet** (local transcription) is not in this build yet; its place is FluidAudio (Core ML).
