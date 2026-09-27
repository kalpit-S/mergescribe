import Foundation
import MergeScribeCore
import MergeScribeMac

/// `MergeScribe replay <training session dir>`: a saved dictation through the real
/// recognizers, judge and correction, printing what would have been typed. Needs no
/// microphone or accessibility permission. The saved audio has its pauses trimmed, so the
/// chunks it makes aren't the original ones; everything downstream is the real thing.
enum Replay {
    struct PrintOutput: TextOutput {
        let typed = Locked("")
        func type(_ text: String) { typed.update { $0 += text } }
        func copyToClipboard(_ text: String) { Log.line("[Replay] clipboard: \"\(text)\"") }
        func notify(_ message: String) { Log.line("[Replay] notify: \(message)") }
        func playBusySound() {}
    }

    struct StillFocus: Focus {
        let context: AppContext?
        func appContext() -> AppContext? { context }
        func captureOrigin() -> Origin? { nil }
        func atOrigin(_ origin: Origin) -> Bool { true }
        func returnTo(_ origin: Origin) -> Bool { true }
        func selectedText() -> String? { nil }
        func wakeFrontmostApp() {}
    }

    static func run(_ directory: URL) async -> Int32 {
        guard let metadata = (try? Data(contentsOf: directory.appendingPathComponent("metadata.json")))
                .flatMap({ try? JSONDecoder().decode(JSON.self, from: $0) }) else {
            print("No metadata.json in \(directory.path)")
            return 1
        }
        let wavs = ((try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.lastPathComponent.hasPrefix("audio_") && $0.pathExtension == "wav" }.sorted { $0.path < $1.path }
        let audio = wavs.compactMap { url -> (String, [Float])? in
            guard let data = try? Data(contentsOf: url), let (samples, rate) = WAV.decode(data), rate == 16000 else { return nil }
            return (String(url.deletingPathExtension().lastPathComponent.dropFirst(6)), samples)
        }
        guard !audio.isEmpty else {
            print("No 16 kHz audio_*.wav in \(directory.path)")
            return 1
        }
        let settings = Settings.load()
        let context = metadata["app_context"].flatMap { ctx -> AppContext? in
            guard let app = ctx["app_name"]?.string else { return nil }
            return AppContext(appName: app, windowTitle: ctx["window_title"]?.string ?? "", bundleId: ctx["bundle_id"]?.string ?? "")
        }
        print("Replaying \(directory.lastPathComponent): \(audio.map { "\($0.0) \(String(format: "%.1f", Double($0.1.count) / 16000))s" }.joined(separator: ", "))")
        print("Logged output (\(metadata["output_method"]?.string ?? "?")): \"\(metadata["final_output"]?.string ?? "")\"")

        let http = URLSessionHTTPClient()
        let vocabulary = VocabularyStore()
        let output = PrintOutput()
        let services = SessionServices(focus: StillFocus(context: context), output: output,
                                       corrector: CorrectionClient(http: http, vocabulary: vocabulary), judge: JudgeClient(http: http))
        let manager = SessionManager(services: services)
        let done = Locked(false)
        manager.onSessionComplete = { _, _ in done.update { $0 = true } }
        var runSettings = settings
        runSettings.enabledMics = audio.map(\.0)
        runSettings.editFeedbackEnabled = false
        runSettings.trainingEnabled = false
        let recognizers = App.recognizers(for: runSettings, http: http, vocabulary: vocabulary)
        guard let session = manager.startSession(settings: runSettings, recognizers: recognizers) else { return 1 }

        // Feed the audio through a chunker as if it were being heard, one block at a time.
        var chunker = Chunker(config: Chunker.Config(runSettings))
        for (mic, _) in audio { chunker.addMic(mic) }
        let blockSeconds = Double(Chunker.blockSize) / 16000
        for i in 0..<30 {
            for (mic, _) in audio { _ = chunker.feed(mic, [Float](repeating: 0.0005, count: Chunker.blockSize), now: Double(i) * blockSeconds) }
        }
        var clock = 30 * blockSeconds
        chunker.startRecording(now: clock)
        session.start(mics: audio.map(\.0))
        let longest = audio.map(\.1.count).max()!
        var offset = 0
        while offset < longest {
            for (mic, samples) in audio {
                let end = min(offset + Chunker.blockSize, samples.count)
                let block = offset < samples.count ? Array(samples[offset..<end]) : []
                if let chunk = chunker.feed(mic, block + [Float](repeating: 0, count: Chunker.blockSize - block.count), now: clock) {
                    session.chunkReady(chunk)
                }
            }
            offset += Chunker.blockSize
            clock += blockSeconds
        }
        for message in chunker.messages { Log.line(message) }
        session.finalize(chunker.stopRecording())
        while !done.current { try? await Task.sleep(for: .milliseconds(50)) }
        print("Swift typed: \"\(output.typed.current)\"")
        return 0
    }
}
