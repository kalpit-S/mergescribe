import Foundation

/// The console log, with each line tagged by the dictation it belongs to.
///
/// Dictations overlap by design (one finishes while the next records), so their lines
/// interleave and chunk numbers seem to jump backwards. Code running for a session sets
/// `Log.tag` for its task, and child tasks inherit it, so a line says which dictation it is
/// from even when it comes from code that knows nothing about sessions.
public enum Log {
    @TaskLocal public static var tag: String = ""

    private static let lock = NSLock()
    nonisolated(unsafe) private static var sink: @Sendable (String) -> Void = { print($0) }

    /// Where lines go; the app tees them to ~/.mergescribe/logs/mergescribe.log.
    public static func setSink(_ newSink: @escaping @Sendable (String) -> Void) {
        lock.lock(); sink = newSink; lock.unlock()
    }

    public static func line(_ text: String) {
        let tag = Log.tag
        let tagged = text.split(separator: "\n", omittingEmptySubsequences: false).map { part in
            tag.isEmpty || part.trimmingCharacters(in: .whitespaces).isEmpty ? String(part) : "[\(tag)] \(part)"
        }.joined(separator: "\n")
        lock.lock(); let current = sink; lock.unlock()
        current(tagged)
    }
}
