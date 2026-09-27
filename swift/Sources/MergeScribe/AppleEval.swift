import Foundation
import MergeScribeCore
import MergeScribeMac

/// `MergeScribe apple-eval <out.jsonl> [--days N] [--limit N]`: Apple's on-device recognizer
/// over saved single-chunk dictations, one JSON line each (session, text, latency), for
/// scoring against the logged cloud recognizers the way scripts/provider_value.py does.
enum AppleEval {
    static func run(_ arguments: [String]) async -> Int32 {
        guard #available(macOS 26.0, *) else { print("Needs macOS 26"); return 1 }
        guard arguments.count >= 2 else { print("usage: MergeScribe apple-eval <out.jsonl> [--days N] [--limit N]"); return 2 }
        let out = URL(fileURLWithPath: arguments[1])
        func option(_ name: String, _ fallback: Int) -> Int {
            arguments.firstIndex(of: name).flatMap { Int(arguments[$0 + 1]) } ?? fallback
        }
        let days = option("--days", 3), limit = option("--limit", 300)
        guard await AppleRecognizer.available() else { print("The English speech model isn't installed"); return 1 }

        let training = AppPaths().trainingDir
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        var sessions: [URL] = []
        for back in 0..<days {
            let day = training.appendingPathComponent(formatter.string(from: Date().addingTimeInterval(-86400 * Double(back))))
            let entries = (try? FileManager.default.contentsOfDirectory(at: day, includingPropertiesForKeys: nil)) ?? []
            sessions += entries.sorted { $0.path > $1.path }
        }
        let vocabulary = Array(VocabularyStore().knownTerms().prefix(AppleRecognizer.maxContextualStrings))
        FileManager.default.createFile(atPath: out.path, contents: nil)
        let handle = try! FileHandle(forWritingTo: out)
        var done = 0, totalAudio = 0.0, totalTime = 0.0
        for dir in sessions where done < limit {
            guard let data = try? Data(contentsOf: dir.appendingPathComponent("metadata.json")),
                  let meta = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                  let transcriptions = meta["transcriptions"] as? [[String: Any]], !transcriptions.isEmpty,
                  Set(transcriptions.compactMap { $0["chunk"] as? Int }).count <= 1,
                  let wav = try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
                    .first(where: { $0.lastPathComponent.hasPrefix("audio_") }),
                  let audio = try? Data(contentsOf: wav), let (samples, rate) = WAV.decode(audio) else { continue }
            let clock = ContinuousClock(), started = clock.now
            let text: String
            do {
                text = try await AppleRecognizer.transcribe(samples, sampleRate: rate, locale: Locale(identifier: "en_US"),
                                                            vocabulary: vocabulary, dictation: arguments.contains("--dictation"))
            } catch {
                print("\(dir.lastPathComponent): \(error)")
                continue
            }
            let elapsed = (clock.now - started).seconds
            let seconds = Double(samples.count) / Double(rate)
            totalAudio += seconds; totalTime += elapsed; done += 1
            let line = JSON.object(["session_id": .string(dir.lastPathComponent), "text": .string(text),
                                    "latency_ms": .number((elapsed * 1000).rounded()), "audio_s": .number(seconds)]).line()
            handle.write(Data((line + "\n").utf8))
            if done % 25 == 0 { print("\(done) done, \(String(format: "%.0f", totalAudio))s of audio in \(String(format: "%.1f", totalTime))s") }
        }
        try? handle.close()
        print("\(done) dictations, \(String(format: "%.0f", totalAudio))s of audio transcribed in \(String(format: "%.1f", totalTime))s "
              + "(\(String(format: "%.0f", totalAudio / max(totalTime, 0.001)))x real time)")
        return 0
    }
}
