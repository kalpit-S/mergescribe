import AppKit
import Foundation
import MergeScribeCore
import MergeScribeMac

let arguments = Array(CommandLine.arguments.dropFirst())

switch arguments.first {
case "replay":
    guard arguments.count >= 2 else {
        print("usage: MergeScribe replay <~/.mergescribe/training/<date>/<session>>")
        exit(2)
    }
    let url = URL(fileURLWithPath: (arguments[1] as NSString).expandingTildeInPath)
    let status = Locked<Int32>(1)
    let finished = DispatchSemaphore(value: 0)
    Task.detached {
        let code = await Replay.run(url)
        status.update { $0 = code }
        finished.signal()
    }
    finished.wait()
    exit(status.current)

case "apple-eval":
    let status = Locked<Int32>(1)
    let finished = DispatchSemaphore(value: 0)
    Task.detached {
        let code = await AppleEval.run(arguments)
        status.update { $0 = code }
        finished.signal()
    }
    finished.wait()
    exit(status.current)

case "live-test":
    // A saved recording fed at real-time pace to the live transcriber, printing the words as the HUD would get them.
    guard arguments.count >= 2, let data = try? Data(contentsOf: URL(fileURLWithPath: arguments[1])),
          let (samples, rate) = WAV.decode(data) else { print("usage: MergeScribe live-test <audio.wav>"); exit(2) }
    let words = LiveWords()
    let started = HostClock.now
    let last = Locked("")
    words.start(sampleRate: rate, vocabulary: VocabularyStore().knownTerms()) { text in
        guard text != last.current else { return }
        last.update { $0 = text }
        print(String(format: "%5.2fs  ", HostClock.now - started) + text)
    }
    var offset = 0
    while offset < samples.count {
        let end = min(offset + Chunker.blockSize, samples.count)
        words.append(Array(samples[offset..<end]))
        offset = end
        usleep(useconds_t(Double(Chunker.blockSize) / Double(rate) * 1_000_000))
    }
    print(String(format: "%5.2fs  (audio ends)", HostClock.now - started))
    usleep(1_500_000)
    words.stop()
    exit(0)

case "stats":
    let usage = Stats.usage()
    print("words:", usage.words, "dictations:", usage.dictations)
    print("speaking wpm:", usage.speakingWpm.map { String(format: "%.1f", $0) } ?? "-",
          "overall wpm:", usage.overallWpm.map { String(format: "%.1f", $0) } ?? "-")
    if let after = usage.afterRelease { print(String(format: "after release: %.2fs median, %.2fs slowest tenth", after.median, after.slow)) }
    print("daily words:", usage.dailyWords)
    print("apps:", usage.apps.map { "\($0.app) \(Int(($0.share * 100).rounded()))%" }.joined(separator: ", "))

case "devices":
    for device in CoreAudioDevices.inputDevices() {
        print("\(device.name)  [\(device.inputChannels) ch, id \(device.id)]")
    }

case "hud-frames":
    let directory = URL(fileURLWithPath: arguments.count > 1 ? arguments[1] : "hud-frames")
    HUDScript.renderFrames(to: directory, at: [0.3, 1.0, 2.0, 3.2, 4.4, 5.2, 6.4, 7.2, 8.3])
    print("Wrote frames to \(directory.path)")

case "hud-preview":
    // The scripted dictation, live on screen, until Ctrl-C with --loop.
    let app = NSApplication.shared
    app.setActivationPolicy(.accessory)
    let loop = arguments.contains("--loop")
    let started = Locked(HostClock.now)
    let hud = HUD(level: { HUDScript.level(at: HostClock.now - started.current) })
    func play() {
        started.update { $0 = HostClock.now }
        for (at, event) in HUDScript.previewEvents {
            DispatchQueue.main.asyncAfter(deadline: .now() + at) { event(hud) }
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + HUDScript.length) {
            if loop { play() } else { exit(0) }
        }
    }
    play()
    if let index = arguments.firstIndex(of: "--capture"), index + 1 < arguments.count {
        let url = URL(fileURLWithPath: arguments[index + 1])
        for at in [2.9, 5.2, 6.4] {
            DispatchQueue.main.asyncAfter(deadline: .now() + at) {
                if let image = MainActor.assumeIsolated({ hud.captureLayers() }) {
                    HUDScript.write(image, to: url.appendingPathComponent(String(format: "live_%.1f.png", at)))
                }
            }
        }
    }
    app.run()

case "settings-preview":
    // The Settings window on its own, optionally opened at a section: settings-preview Stats [--background]
    let app = NSApplication.shared
    app.setActivationPolicy(.accessory)
    let section = arguments.dropFirst().first.flatMap { name in
        SettingsSection.allCases.first { $0.rawValue.lowercased() == name.lowercased() }
    }
    let controller = MainActor.assumeIsolated { SettingsWindowController(http: URLSessionHTTPClient()) }
    let background = arguments.contains("--background")
    MainActor.assumeIsolated { controller.present(section: section, activate: !background) }
    app.run()

case nil, "run":
    LogFile.start()
    let app = NSApplication.shared
    app.setActivationPolicy(.accessory)
    let delegate = MainActor.assumeIsolated { App() }
    app.delegate = delegate
    app.run()

default:
    print("usage: MergeScribe [run | replay <session dir> | stats | devices | hud-frames <dir>]")
    exit(2)
}
