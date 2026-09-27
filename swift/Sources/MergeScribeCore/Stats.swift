import Foundation

/// Usage stats for the Settings window, read from the metrics log.
///
/// A dictation counts once it completes: how many words it produced, how long the key was
/// held, and how long after release the text was done.
public enum Stats {
    /// session_complete kept this much text before it logged a word count (and the Python app
    /// still does); the Swift app keeps up to textLimit, for History.
    static let oldTextLimit = 500
    static let textLimit = 20_000
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
        guard let data = try? Data(contentsOf: metrics) else { return Usage() }
        let (started, stopped, completed) = scan(data)

        var result = Usage(dictations: ["today": 0, "week": 0, "all": 0], words: ["today": 0, "week": 0, "all": 0])
        let nowTs = now.timeIntervalSince1970, weekAgo = nowTs - 7 * 86400
        let today = calendar.startOfDay(for: now)
        var daily = Array(repeating: 0, count: days)
        var speaking: [Double] = [], overall: [Double] = [], waits: [Double] = []
        var apps: [String: Int] = [:], appOrder: [String] = []
        // Calendar arithmetic is slow next to everything else here; days repeat, so remember them.
        var dayStarts: [Int: Date] = [:]
        func dayStart(_ ts: Double) -> Date {
            let hour = Int((ts / 3600).rounded(.down))
            if let known = dayStarts[hour] { return known }
            let start = calendar.startOfDay(for: Date(timeIntervalSince1970: ts))
            dayStarts[hour] = start
            return start
        }
        let formatter = dayFormatter(calendar)

        for record in completed {
            let ts = record["ts"]!.double!, session = record["session_id"]?.string ?? ""
            var words = record["words"]?.int
            if words == nil {
                let finalText = record["final_text"]?.string ?? ""
                let (count, length) = wordsAndLength(finalText)
                words = count
                if length >= oldTextLimit, let full = fullWordCount(session, ts, training, formatter) { words = full }
            }
            guard let words, words > 0 else { continue }
            let day = dayStart(ts)
            var periods = ["all"]
            if ts >= weekAgo { periods.append("week") }
            if day == today { periods.append("today") }
            for period in periods {
                result.dictations[period, default: 0] += 1
                result.words[period, default: 0] += words
            }
            // Whole days between the two midnights (a DST day is 23 or 25 hours, hence rounding).
            let age = Int(((today.timeIntervalSince1970 - day.timeIntervalSince1970) / 86400).rounded())
            if age >= 0, age < days { daily[days - 1 - age] += words }
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

    /// The events stats need.
    public static func scan(_ data: Data) -> (started: [String: Double], stopped: [String: Double], completed: [[String: JSON]]) {
        var started: [String: Double] = [:], stopped: [String: Double] = [:]
        var completed: [[String: JSON]] = []
        MetricsScan.lines(data, containing: ["\"recording_st", "\"session_complete\""]) { object in
            guard let ts = (object["ts"] as? NSNumber)?.doubleValue else { return }
            let session = object["session_id"] as? String ?? ""
            switch object["event"] as? String {
            case "recording_started": started[session] = ts
            case "recording_stopped": stopped[session] = ts
            case "session_complete": completed.append(object.mapValues { JSON(any: $0) })
            default: break
            }
        }
        return (started, stopped, completed)
    }

    /// Words (runs between whitespace) and length in code points, in one pass over the scalars:
    /// counting Characters over twenty thousand records took seconds.
    static func wordsAndLength(_ text: String) -> (words: Int, length: Int) {
        var words = 0, length = 0, inWord = false
        for scalar in text.unicodeScalars {
            length += 1
            // ASCII first: the Unicode property lookup is the slow part, and nearly all of it is ASCII.
            let space = scalar.isASCII ? (scalar.value == 32 || (9...13).contains(scalar.value)) : scalar.properties.isWhitespace
            if space {
                inWord = false
            } else if !inWord {
                inWord = true
                words += 1
            }
        }
        return (words, length)
    }

    static func dayFormatter(_ calendar: Calendar) -> DateFormatter {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = calendar
        formatter.timeZone = calendar.timeZone
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter
    }

    static func fullWordCount(_ session: String, _ ts: Double, _ training: URL, _ formatter: DateFormatter) -> Int? {
        let date = Date(timeIntervalSince1970: ts)
        for day in [date, date.addingTimeInterval(-86400)] {
            let url = training.appendingPathComponent(formatter.string(from: day)).appendingPathComponent(session)
                .appendingPathComponent("metadata.json")
            if let data = try? Data(contentsOf: url),
               let meta = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
                return wordsAndLength(meta["final_output"] as? String ?? "").words
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
