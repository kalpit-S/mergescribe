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

case nil, "run":
    LogFile.start()
    let app = NSApplication.shared
    app.setActivationPolicy(.accessory)
    let delegate = MainActor.assumeIsolated { App() }
    app.delegate = delegate
    app.run()

default:
    print("usage: MergeScribe [run | replay <session dir> | stats | devices]")
    exit(2)
}
