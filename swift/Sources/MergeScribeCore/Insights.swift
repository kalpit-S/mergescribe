import Foundation

/// What the metrics log says about each part of the pipeline, for Settings: how fast each
/// recognizer answers and how often it misses the deadline, how fast correction is, how often
/// Jev skips it, and the recent dictations themselves.
public enum Insights {
    public struct Recognizer: Identifiable, Equatable, Sendable {
        public var id: String { name }
        public let name: String                 // as results carry it: "or-microsoft-mai-transcribe-2"
        public var answers = 0
        public var medianMs: Double?
        public var p90Ms: Double?
        /// Times the deadline moved on without it.
        public var leftBehind = 0
        /// Answers that heard nothing.
        public var empty = 0
    }

    public struct Correction: Identifiable, Equatable, Sendable {
        public var id: String { model }
        public let model: String
        public var count = 0
        public var medianMs: Double?
        public var p90Ms: Double?
    }

    public struct JudgeSummary: Equatable, Sendable {
        public var verdicts = 0
        /// Verdicts that found nothing to correct. Not all of them skipped correction: one that
        /// came back after the correction had started typing did nothing.
        public var clean = 0
        public var medianMs: Double?
    }

    public struct Entry: Identifiable, Equatable, Sendable {
        public var id: String { sessionId }
        public let sessionId: String
        public let date: Date
        public let text: String
        public let words: Int
        public let app: String
        public let seconds: Double
        /// Only the start was logged (the Python app kept 500 characters).
        public var clipped = false
    }

    public struct Report: Equatable, Sendable {
        public var recognizers: [Recognizer] = []
        public var corrections: [Correction] = []
        public var judge = JudgeSummary()
        public var history: [Entry] = []
        public var days = 7
    }

    /// The last `days` of the log, and the newest `historyLimit` dictations.
    public static func report(metrics: URL = AppPaths().metricsFile, days: Int = 7, historyLimit: Int = 300,
                              now: Date = Date()) -> Report {
        guard let data = try? Data(contentsOf: metrics) else { return Report(days: days) }
        let since = now.timeIntervalSince1970 - Double(days) * 86400
        var latencies: [String: [Double]] = [:], empty: [String: Int] = [:], behind: [String: Int] = [:]
        var correction: [String: [Double]] = [:]
        var judgeLatency: [Double] = [], verdicts = 0, clean = 0
        var history: [Entry] = []
        MetricsScan.lines(data, containing: ["\"transcription\"", "\"provider_deadline\"", "\"llm_correction\"",
                                             "\"judge\"", "\"session_complete\""]) { record in
            guard let ts = (record["ts"] as? NSNumber)?.doubleValue else { return }
            let event = record["event"] as? String
            if event == "session_complete" {
                let text = record["final_text"] as? String ?? ""
                guard !text.isEmpty else { return }
                let clipped = record["output_method"] == nil && text.unicodeScalars.count >= Stats.oldTextLimit
                history.append(Entry(sessionId: record["session_id"] as? String ?? "", date: Date(timeIntervalSince1970: ts),
                                     text: text, words: (record["words"] as? NSNumber)?.intValue ?? Stats.wordsAndLength(text).words,
                                     app: record["app"] as? String ?? "",
                                     seconds: ((record["total_duration_ms"] as? NSNumber)?.doubleValue ?? 0) / 1000,
                                     clipped: clipped))
                return
            }
            guard ts >= since else { return }
            switch event {
            case "transcription":
                guard let provider = record["provider"] as? String else { return }
                latencies[provider, default: []].append((record["latency_ms"] as? NSNumber)?.doubleValue ?? 0)
                if ((record["text"] as? String) ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    empty[provider, default: 0] += 1
                }
            case "provider_deadline":
                for stream in record["abandoned"] as? [String] ?? [] {
                    behind[String(stream.split(separator: "/", maxSplits: 1).first ?? ""), default: 0] += 1
                }
            case "llm_correction":
                guard let model = record["model"] as? String else { return }
                correction[model, default: []].append((record["latency_ms"] as? NSNumber)?.doubleValue ?? 0)
            case "judge":
                verdicts += 1
                if record["clean"] as? Bool == true { clean += 1 }
                if let ms = (record["latency_ms"] as? NSNumber)?.doubleValue { judgeLatency.append(ms) }
            default: break
            }
        }
        var report = Report(days: days)
        report.recognizers = Set(latencies.keys).union(behind.keys).map { name in
            let ms = latencies[name] ?? []
            return Recognizer(name: name, answers: ms.count, medianMs: percentile(ms, 0.5), p90Ms: percentile(ms, 0.9),
                              leftBehind: behind[name] ?? 0, empty: empty[name] ?? 0)
        }.sorted { $0.answers > $1.answers }
        report.corrections = correction.map { model, ms in
            Correction(model: model, count: ms.count, medianMs: percentile(ms, 0.5), p90Ms: percentile(ms, 0.9))
        }.sorted { $0.count > $1.count }
        report.judge = JudgeSummary(verdicts: verdicts, clean: clean, medianMs: percentile(judgeLatency, 0.5))
        report.history = Array(history.sorted { $0.date > $1.date }.prefix(historyLimit))
        return report
    }

    /// The median, or the nearest-rank percentile: the smallest value with at least p of them at
    /// or below it (p90 of ten is the ninth).
    static func percentile(_ values: [Double], _ p: Double) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        if p == 0.5 { return Stats.median(sorted) }
        let rank = Int((Double(sorted.count) * p).rounded(.up))
        return sorted[min(max(rank, 1), sorted.count) - 1]
    }

    /// One dictation in full, from its training capture: what each recognizer heard, and
    /// what was typed and how. nil when capture was off.
    public struct Detail: Equatable, Sendable {
        public var transcripts: [(provider: String, text: String, latencyMs: Int)]
        public var typed: String
        public var method: String
        public var correctionModel: String?
        public var audio: URL?

        public static func == (l: Detail, r: Detail) -> Bool {
            l.typed == r.typed && l.method == r.method && l.transcripts.map(\.text) == r.transcripts.map(\.text)
        }
    }

    public static func detail(_ entry: Entry, training: URL = AppPaths().trainingDir) -> Detail? {
        let formatter = Stats.dayFormatter(.current)
        for day in [entry.date, entry.date.addingTimeInterval(-86400)] {
            let dir = training.appendingPathComponent(formatter.string(from: day)).appendingPathComponent(entry.sessionId)
            guard let data = try? Data(contentsOf: dir.appendingPathComponent("metadata.json")),
                  let meta = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { continue }
            let transcripts = (meta["transcriptions"] as? [[String: Any]] ?? []).map {
                (provider: $0["provider"] as? String ?? "", text: $0["text"] as? String ?? "",
                 latencyMs: ($0["latency_ms"] as? NSNumber)?.intValue ?? 0)
            }
            let audio = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil))?
                .first { $0.lastPathComponent.hasPrefix("audio_") }
            return Detail(transcripts: transcripts, typed: meta["final_output"] as? String ?? entry.text,
                          method: meta["output_method"] as? String ?? "",
                          correctionModel: (meta["llm_correction"] as? [String: Any])?["model"] as? String, audio: audio)
        }
        return nil
    }
}

/// The lines of the metrics log that contain one of a few markers, parsed. The log is tens of
/// megabytes of mostly per-chunk events: lines are found by their bytes, and only the few that
/// matter are parsed.
public enum MetricsScan {
    public static func lines(_ data: Data, containing markers: [String], _ handle: ([String: Any]) -> Void) {
        let needles = markers.map { Array($0.utf8) }
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Void in
            guard let base = raw.baseAddress else { return }
            var start = 0
            while start < raw.count {
                var end = raw.count
                if let found = memchr(base + start, 0x0A, raw.count - start) { end = UnsafeRawPointer(found) - base }
                let line = UnsafeRawBufferPointer(rebasing: raw[start..<end])
                start = end + 1
                var matches = false
                for needle in needles where !matches {
                    matches = needle.withUnsafeBytes { memmem(line.baseAddress, line.count, $0.baseAddress, $0.count) != nil }
                }
                guard matches, let object = (try? JSONSerialization.jsonObject(with: Data(line))) as? [String: Any] else { continue }
                handle(object)
            }
        }
    }
}
