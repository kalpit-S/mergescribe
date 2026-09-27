import Foundation

/// Opt-in local capture of each dictation's audio and what became of it, for fine-tuning.
///
/// Saves to training/<date>/<session id>/: audio_<mic>.wav (16 kHz mono PCM16) and
/// metadata.json, readable by this user only. Writes happen on a background queue; a backlog
/// past a few sessions is dropped rather than held in memory.
public final class TrainingWriter: @unchecked Sendable {
    static let maxQueued = 10
    /// Don't keep very short recordings.
    static let minDurationMs = 500.0

    let directory: URL
    let sampleRate: Int
    private let queue = DispatchQueue(label: "mergescribe.training")
    private let lock = NSLock()
    private var queued = 0
    public private(set) var dropped = 0

    public init(directory: URL, sampleRate: Int = 16000) {
        self.directory = directory
        self.sampleRate = sampleRate
    }

    /// Queue a session. False when it was too short or the queue was full.
    @discardableResult
    public func save(sessionId: String, audio: [(mic: String, samples: [Float])], metadata: JSON, date: Date = Date()) -> Bool {
        let longest = audio.map(\.samples.count).max() ?? 0
        guard Double(longest) / Double(sampleRate) * 1000 >= Self.minDurationMs else { return false }
        let accepted: Bool = lock.withLock {
            guard queued < Self.maxQueued else { dropped += 1; return false }
            queued += 1
            return true
        }
        guard accepted else {
            Log.line("[Training] Queue full, dropped session \(sessionId) (total dropped: \(dropped))")
            return false
        }
        queue.async { [self] in
            write(sessionId: sessionId, audio: audio, metadata: metadata, date: date)
            lock.withLock { queued -= 1 }
        }
        return true
    }

    public func flush() { queue.sync {} }

    func write(sessionId: String, audio: [(mic: String, samples: [Float])], metadata: JSON, date: Date) {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd"
        let dir = directory.appendingPathComponent(formatter.string(from: date)).appendingPathComponent(sessionId)
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
            for (mic, samples) in audio where !samples.isEmpty {
                let file = dir.appendingPathComponent("audio_\(Self.sanitize(mic)).wav")
                try AtomicFile.write(WAV.encode(samples, sampleRate: sampleRate), to: file, permissions: 0o600)
            }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            try AtomicFile.write(try encoder.encode(metadata), to: dir.appendingPathComponent("metadata.json"), permissions: 0o600)
            Log.line("[Training] Saved session \(sessionId) (\(audio.count) mics)")
        } catch {
            Log.line("[Training] Failed to save session \(sessionId): \(error.localizedDescription)")
        }
    }

    /// A mic name as a file name, the way the Python app wrote it.
    static func sanitize(_ name: String) -> String {
        var safe = name.lowercased()
        for ch in " /\\:*?\"<>|()" { safe = safe.replacingOccurrences(of: String(ch), with: "_") }
        while safe.contains("__") { safe = safe.replacingOccurrences(of: "__", with: "_") }
        safe = safe.trimmingCharacters(in: CharacterSet(charactersIn: "_"))
        if safe.count < 2 {
            var hash: UInt32 = 2166136261
            for byte in name.utf8 { hash = (hash ^ UInt32(byte)) &* 16777619 }
            safe = String(format: "mic_%04x", hash & 0xFFFF)
        }
        return safe
    }
}
