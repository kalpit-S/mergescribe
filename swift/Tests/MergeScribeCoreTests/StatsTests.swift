import Foundation
import Testing
@testable import MergeScribeCore

@Suite struct StatsTests {
    static let now: Date = {
        var c = DateComponents(); c.year = 2026; c.month = 9; c.day = 27; c.hour = 15
        return Calendar.current.date(from: c)!
    }()
    static let day = 86400.0

    struct Dictation {
        var ago = 0.0, held = 10.0, wait = 1.5
        var record: [String: JSON] = [:]
    }

    /// One dictation: key down, up after `held` seconds, text done `wait` seconds later.
    func log(_ home: TempHome, _ dictations: [Dictation]) {
        var lines: [JSON] = []
        for (i, d) in dictations.enumerated() {
            let sid = JSON.string("s\(i)"), start = Self.now.timeIntervalSince1970 - d.ago
            let stop = start + d.held, done = stop + d.wait
            var complete: [String: JSON] = ["ts": .number(done), "event": "session_complete", "session_id": sid,
                                            "total_duration_ms": .number((done - start) * 1000)]
            complete.merge(d.record) { _, new in new }
            lines += [["ts": .number(start), "event": "recording_started", "session_id": sid],
                      ["ts": .number(stop), "event": "recording_stopped", "session_id": sid],
                      ["ts": .number(done), "event": "transcription", "session_id": sid, "provider": "p"],
                      .object(complete)]
        }
        home.write("metrics.jsonl", lines.map { $0.line() }.joined(separator: "\n"))
    }

    func usage(_ home: TempHome) -> Stats.Usage {
        Stats.usage(metrics: home.paths.metricsFile, training: home.paths.trainingDir, now: Self.now)
    }

    @Test func wordsAndDictationsByPeriod() {
        let home = TempHome()
        log(home, [Dictation(record: ["words": 30]), Dictation(ago: 3 * Self.day, record: ["words": 20]),
                   Dictation(ago: 30 * Self.day, record: ["words": 10])])
        let result = usage(home)
        #expect(result.words == ["today": 30, "week": 50, "all": 60])
        #expect(result.dictations == ["today": 1, "week": 2, "all": 3])
        #expect(result.dailyWords.last == 30 && result.dailyWords[result.dailyWords.count - 4] == 20)
    }

    @Test func oldRecordsCountTheirTextAndLongOnesReadTheTrainingCapture() throws {
        // Before words were logged, the text was kept only to 500 characters.
        let home = TempHome()
        log(home, [Dictation(record: ["final_text": "ship it friday"]),
                   Dictation(record: ["final_text": .string(String(repeating: "word ", count: 100))])])
        let formatter = DateFormatter(); formatter.dateFormat = "yyyy-MM-dd"
        let capture = home.paths.trainingDir.appendingPathComponent(formatter.string(from: Self.now)).appendingPathComponent("s1")
        try FileManager.default.createDirectory(at: capture, withIntermediateDirectories: true)
        try JSON.object(["final_output": .string(String(repeating: "word ", count: 180))]).line()
            .write(to: capture.appendingPathComponent("metadata.json"), atomically: true, encoding: .utf8)
        #expect(usage(home).words["all"] == 3 + 180)
    }

    @Test func speedIsTheMedianOverTheWeekAndSkipsTinyDictations() {
        let home = TempHome()
        log(home, [Dictation(held: 12, wait: 1, record: ["words": 30]),     // 150 wpm speaking
                   Dictation(held: 20, wait: 2, record: ["words": 40]),     // 120
                   Dictation(held: 60, wait: 1.5, record: ["words": 150]),  // 150
                   Dictation(held: 1, wait: 1, record: ["words": 2])])      // too short to count
        let result = usage(home)
        #expect(result.speakingWpm.map { $0.rounded() } == 150)
        #expect(result.overallWpm! < result.speakingWpm!)
        #expect(abs(result.afterRelease!.median - 1.25) < 1e-9)            // every dictation's wait counts
    }

    @Test func appsAreSharesOfTheWeek() {
        let home = TempHome()
        log(home, Array(repeating: Dictation(record: ["words": 9, "app": "Claude"]), count: 3)
                  + [Dictation(record: ["words": 9, "app": "Slack"])])
        let apps = usage(home).apps
        #expect(apps.map(\.app) == ["Claude", "Slack"] && apps.map(\.share) == [0.75, 0.25])
    }

    @Test func noLogIsNoStats() {
        let result = usage(TempHome())
        #expect(result.words.isEmpty && result.speakingWpm == nil)
    }
}
