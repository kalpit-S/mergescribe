import Foundation
import Testing
@testable import MergeScribeCore

@Suite struct InsightsTests {
    @Test func recognizersCorrectionJudgeAndHistory() {
        let home = TempHome()
        let now = Date(timeIntervalSince1970: 1_800_000_000)
        let t = now.timeIntervalSince1970 - 3600
        var lines: [JSON] = []
        for (i, ms) in [300, 500, 700].enumerated() {
            lines.append(["ts": .number(t + Double(i)), "event": "transcription", "session_id": "s", "provider": "or-mai", "latency_ms": .number(Double(ms)), "text": "hi"])
        }
        lines.append(["ts": .number(t), "event": "transcription", "session_id": "s", "provider": "or-fish", "latency_ms": 900, "text": ""])
        lines.append(["ts": .number(t), "event": "provider_deadline", "session_id": "s", "abandoned": ["or-fish/MacBook Pro Microphone"], "kept": []])
        lines.append(["ts": .number(t), "event": "llm_correction", "session_id": "s", "model": "openai/gpt-6-luna", "latency_ms": 800])
        lines.append(["ts": .number(t), "event": "judge", "session_id": "s", "clean": true, "latency_ms": 220])
        lines.append(["ts": .number(t), "event": "judge", "session_id": "s2", "clean": false, "latency_ms": 240])
        lines.append(["ts": .number(t), "event": "session_complete", "session_id": "s", "final_text": "Ship it Friday.", "words": 3, "app": "Slack", "total_duration_ms": 2500])
        lines.append(["ts": .number(t - 30 * 86400), "event": "transcription", "session_id": "old", "provider": "parakeet", "latency_ms": 100, "text": "x"])
        home.write("metrics.jsonl", lines.map { $0.line() }.joined(separator: "\n"))

        let report = Insights.report(metrics: home.paths.metricsFile, days: 7, now: now)
        #expect(report.recognizers.map(\.name) == ["or-mai", "or-fish"], "a month-old recognizer is out of the window")
        #expect(report.recognizers[0].medianMs == 500 && report.recognizers[0].answers == 3)
        #expect(report.recognizers[1].leftBehind == 1 && report.recognizers[1].empty == 1)
        #expect(report.corrections.first?.model == "openai/gpt-6-luna" && report.corrections.first?.medianMs == 800)
        #expect(report.judge.verdicts == 2 && report.judge.clean == 1)
        #expect(report.history.first?.text == "Ship it Friday." && report.history.first?.app == "Slack")
    }

    @Test func nearestRankPercentile() {
        let ten = (1...10).map(Double.init)
        #expect(Insights.percentile(ten, 0.9) == 9)
        #expect(Insights.percentile(ten + [11], 0.9) == 10)
        #expect(Insights.percentile([4], 0.9) == 4)
        #expect(Insights.percentile([], 0.9) == nil)
    }

    @Test func pythonsClippedTextIsMarked() {
        let home = TempHome()
        let long = String(repeating: "a", count: 500)
        let lines: [JSON] = [
            ["ts": 1, "event": "session_complete", "session_id": "py", "final_text": .string(long), "words": 1],
            ["ts": 2, "event": "session_complete", "session_id": "swift", "final_text": .string(long), "words": 1, "output_method": "streamed"],
        ]
        home.write("metrics.jsonl", lines.map { $0.line() }.joined(separator: "\n"))
        let history = Insights.report(metrics: home.paths.metricsFile, now: Date(timeIntervalSince1970: 3)).history
        #expect(history.map(\.clipped) == [false, true], "newest first; only the one without output_method was cut")
    }

    @Test func noLogIsAnEmptyReport() {
        #expect(Insights.report(metrics: TempHome().paths.metricsFile).recognizers.isEmpty)
    }
}
