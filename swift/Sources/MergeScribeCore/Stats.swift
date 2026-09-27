import Foundation

/// Usage stats for the Settings window, read from the metrics log.
///
/// A dictation counts once it completes: how many words it produced, how long the key was
/// held, and how long after release the text was done.
public enum Stats {
    /// session_complete kept this much text before it logged a word count.
    static let oldTextLimit = 500
    /// Shorter dictations make a words-per-minute figure meaningless.
    static let minWords = 5
    public static let days = 14

    public struct Usage: Equatable, Sendable {
        public var dictations: [String: Int] = [:]   // "today", "week", "all"
        public var words: [String: Int] = [:]
        /// While the key is held, median over the past week.
        public var speakingWpm: Double?
        /// Key press to text typed, median over the past week.
        public var overallWpm: Double?
        /// Seconds from release to done: (median, slowest tenth).
        public var afterRelease: (median: Double, slow: Double)?
        /// Words per day for the last `days` days, oldest first.
        public var dailyWords: [Int] = []
        /// Share of the past week's dictations, top four.
        public var apps: [(app: String, share: Double)] = []

        public static func == (l: Usage, r: Usage) -> Bool {
            l.dictations == r.dictations && l.words == r.words && l.speakingWpm == r.speakingWpm
                && l.overallWpm == r.overallWpm && l.afterRelease?.median == r.afterRelease?.median
                && l.afterRelease?.slow == r.afterRelease?.slow && l.dailyWords == r.dailyWords
                && l.apps.map(\.app) == r.apps.map(\.app) && l.apps.map(\.share) == r.apps.map(\.share)
        }
    }

    public static func usage(metrics: URL = AppPaths().metricsFile, training: URL = AppPaths().trainingDir,
                             now: Date = Date(), calendar: Calendar = .current) -> Usage {
        guard let text = try? String(contentsOf: metrics, encoding: .utf8) else { return Usage() }
        var started: [String: Double] = [:], stopped: [String: Double] = [:]
        var completed: [[String: JSON]] = []
        for line in text.split(whereSeparator: \.isNewline) {
            // Cheap filter first: the log is mostly per-chunk events.
            guard line.contains("\"recording_st") || line.contains("\"session_complete\"") else { continue }
            guard let record = (try? JSONDecoder().decode(JSON.self, from: Data(line.utf8)))?.object,
                  let ts = record["ts"]?.double else { continue }
            let session = record["session_id"]?.string ?? ""
            switch record["event"]?.string {
            case "recording_started": started[session] = ts
            case "recording_stopped": stopped[session] = ts
            case "session_complete": completed.append(record)
            default: break
            }
        }

        var result = Usage(dictations: ["today": 0, "week": 0, "all": 0], words: ["today": 0, "week": 0, "all": 0])
        let nowTs = now.timeIntervalSince1970, weekAgo = nowTs - 7 * 86400
        let today = calendar.startOfDay(for: now)
        var daily = Array(repeating: 0, count: days)
        var speaking: [Double] = [], overall: [Double] = [], waits: [Double] = []
        var apps: [String: Int] = [:], appOrder: [String] = []

        for record in completed {
            let ts = record["ts"]!.double!, session = record["session_id"]?.string ?? ""
            var words = record["words"]?.int
            if words == nil {
                let finalText = record["final_text"]?.string ?? ""
                words = finalText.split(whereSeparator: \.isWhitespace).count
                if finalText.count >= oldTextLimit, let full = fullWordCount(session, ts, training, calendar) { words = full }
            }
            guard let words, words > 0 else { continue }
            let date = Date(timeIntervalSince1970: ts)
            let day = calendar.startOfDay(for: date)
            var periods = ["all"]
            if ts >= weekAgo { periods.append("week") }
            if day == today { periods.append("today") }
            for period in periods {
                result.dictations[period, default: 0] += 1
                result.words[period, default: 0] += words
            }
            if let age = calendar.dateComponents([.day], from: day, to: today).day, age >= 0, age < days {
                daily[days - 1 - age] += words
            }
            guard ts >= weekAgo else { continue }
            if let app = record["app"]?.string, !app.isEmpty {
                if apps[app] == nil { appOrder.append(app) }
                apps[app, default: 0] += 1
            }
            if let stop = stopped[session] { waits.append(ts - stop) }
            guard words >= minWords else { continue }
            if let start = started[session], let stop = stopped[session], stop > start {
                speaking.append(Double(words) / ((stop - start) / 60))
            }
            if let total = record["total_duration_ms"]?.double, total > 1000 {
                overall.append(Double(words) / (total / 60000))
            }
        }

        result.dailyWords = daily
        result.speakingWpm = median(speaking)
        result.overallWpm = median(overall)
        if !waits.isEmpty {
            waits.sort()
            result.afterRelease = (median(waits)!, waits[Int(Double(waits.count) * 0.9)])
        }
        let total = apps.values.reduce(0, +)
        if total > 0 {
            let ranked = appOrder.enumerated().sorted { l, r in
                apps[l.element]! != apps[r.element]! ? apps[l.element]! > apps[r.element]! : l.offset < r.offset
            }
            result.apps = ranked.prefix(4).map { ($0.element, Double(apps[$0.element]!) / Double(total)) }
        }
        return result
    }

    /// The word count from the training capture, for records that kept only part of the text.
    static func fullWordCount(_ session: String, _ ts: Double, _ training: URL, _ calendar: Calendar) -> Int? {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = calendar
        formatter.timeZone = calendar.timeZone
        formatter.dateFormat = "yyyy-MM-dd"
        let date = Date(timeIntervalSince1970: ts)
        for day in [date, date.addingTimeInterval(-86400)] {
            let url = training.appendingPathComponent(formatter.string(from: day)).appendingPathComponent(session)
                .appendingPathComponent("metadata.json")
            if let data = try? Data(contentsOf: url), let meta = try? JSONDecoder().decode(JSON.self, from: data) {
                return (meta["final_output"]?.string ?? "").split(whereSeparator: \.isWhitespace).count
            }
        }
        return nil
    }

    static func median(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted(), mid = sorted.count / 2
        return sorted.count % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
    }
}
