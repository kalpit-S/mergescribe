import Foundation

/// ~/.mergescribe/metrics.jsonl: one JSON object per event, appended in batches off the
/// calling thread. Every line carries "ts" (Unix seconds) and "event", as the Python app
/// wrote them, so scripts/ and the Stats page read both apps' logs alike.
public final class MetricsWriter: MetricsLogging, @unchecked Sendable {
    let url: URL
    private let queue = DispatchQueue(label: "mergescribe.metrics")
    private let clock: @Sendable () -> Double

    public init(url: URL, clock: @escaping @Sendable () -> Double = { Date().timeIntervalSince1970 }) {
        self.url = url
        self.clock = clock
    }

    public func log(_ event: String, _ fields: [String: JSON]) {
        var entry = fields
        entry["ts"] = .number(clock())
        entry["event"] = .string(event)
        let line = JSON.object(entry).line() + "\n"
        queue.async { [url] in Self.append(line, to: url) }
    }

    /// Wait for everything logged so far to reach the file.
    public func flush() { queue.sync {} }

    static func append(_ text: String, to url: URL) {
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            if !FileManager.default.fileExists(atPath: url.path) {
                FileManager.default.createFile(atPath: url.path, contents: nil)
            }
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: Data(text.utf8))
        } catch {
            Log.line("Failed to write \(url.lastPathComponent): \(error.localizedDescription)")
        }
    }
}

/// ~/.mergescribe/corrections.jsonl: edits the speaker made to typed text. session_id
/// joins a row to its saved audio, so a confirmed transcription error becomes a training pair.
public struct CorrectionsCorpus: Sendable {
    public let url: URL
    public init(url: URL = AppPaths().correctionsFile) { self.url = url }

    public func record(_ entry: [String: JSON]) {
        MetricsWriter.append(JSON.object(entry).line() + "\n", to: url)
    }
}
