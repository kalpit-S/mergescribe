import Foundation

// MARK: - What a session needs from the Mac

/// A speech recognizer.
public protocol Recognizer: Sendable {
    /// The name results and metrics carry, e.g. "or-assemblyai-universal-3-5-pro".
    var name: String { get }
    /// A recognizer that serializes its work (one local model behind a lock) runs on the
    /// primary mic only: on a second mic it would just queue onto the critical path.
    var singleInstance: Bool { get }
    /// What it heard. A failure it can explain comes back as an empty transcript; a throw
    /// drops the stream.
    func transcribe(_ samples: [Float], mic: String) async throws -> TranscriptionResult
}

/// Where a dictation started: an app and its focused window. Opaque to Core.
public final class Origin: @unchecked Sendable {
    public let pid: Int32
    public let handle: Any?
    public init(pid: Int32, handle: Any?) { self.pid = pid; self.handle = handle }
}

/// The frontmost app, the window a dictation started in, and the selection.
public protocol Focus: Sendable {
    func appContext() -> AppContext?
    func captureOrigin() -> Origin?
    /// True when that app is frontmost with that window focused.
    func atOrigin(_ origin: Origin) -> Bool
    /// Bring that app and window back to the front. True once they are.
    func returnTo(_ origin: Origin) -> Bool
    /// The selected text, if any (a synthetic copy: slow, so it runs while recording).
    func selectedText() -> String?
    /// Ask the frontmost app to build its accessibility tree, so the edit watcher can read
    /// the field later (Chromium and Electron keep an empty one until asked).
    func wakeFrontmostApp()
}

/// Keystrokes, the clipboard, notifications and sounds.
public protocol TextOutput: Sendable {
    /// Type at the cursor. Whatever can't be typed goes to the clipboard.
    func type(_ text: String)
    func copyToClipboard(_ text: String)
    func notify(_ message: String)
    func playBusySound()
}

/// Watches the field typed into and reports what the user did to the text.
public protocol EditWatching: Sendable {
    func watch(typed: String, app: String,
               onResult: @escaping @Sendable (_ outcome: EditDiff.Outcome, _ after: Double, _ original: String, _ corrected: String) -> Void)
}

/// Correction and editing, as the session uses them.
public protocol Correcting: Sendable {
    func correct(_ results: [TranscriptionResult], context: AppContext?, settings: Settings,
                 onDelta: CorrectionClient.Delta?,
                 onMetadata: (@Sendable (LLMCorrectionResult) -> Void)?,
                 onGenerationMetadata: (@Sendable (String, JSON) -> Void)?) async throws -> String
    func edit(selected: String, command: String, settings: Settings) async -> String
}

extension CorrectionClient: Correcting {}

public protocol Judging: Sendable {
    func judge(_ results: [TranscriptionResult], settings: Settings, context: AppContext?) async -> Judge.Verdict?
}

extension JudgeClient: Judging {}

public protocol MetricsLogging: Sendable {
    func log(_ event: String, _ fields: [String: JSON])
}

/// What a session reports as it runs, for display. A stream is one recognizer on one mic
/// ("or-mai/MacBook Pro Microphone"). Every call names the session, because a session
/// still typing can overlap the next one's recording.
public protocol SessionObserver: Sendable {
    func streamsPlanned(session: String, streams: [String])
    /// A stream returned a chunk. final: the chunk recorded after release.
    func streamLanded(session: String, stream: String, final: Bool)
    /// A stream failed, or was left behind by the deadline.
    func streamDropped(session: String, stream: String)
    /// Every chunk is in; correction starts now.
    func transcriptionDone(session: String)
    /// The transcript needed no correcting and was typed as it stood.
    func correctionSkipped(session: String)
    /// The speaker called it off; nothing will be typed.
    func dictationDiscarded(session: String)
    /// A word just reached the screen.
    func tokenTyped(session: String)
    /// The raw transcript so far, while recording.
    func partialText(session: String, text: String)
}

public extension SessionObserver {
    func streamsPlanned(session: String, streams: [String]) {}
    func streamLanded(session: String, stream: String, final: Bool) {}
    func streamDropped(session: String, stream: String) {}
    func transcriptionDone(session: String) {}
    func correctionSkipped(session: String) {}
    func dictationDiscarded(session: String) {}
    func tokenTyped(session: String) {}
    func partialText(session: String, text: String) {}
}

// MARK: - Shared between sessions

/// Recent dictations: when, what (the raw transcript), and where they went.
public final class TranscriptionHistory: @unchecked Sendable {
    public struct Entry: Sendable { public var time: Double; public var text: String; public var destination: String }

    let maxEntries: Int
    let maxAge: Double
    private let lock = NSLock()
    var entries: [Entry] = []
    let clock: @Sendable () -> Double

    public init(maxEntries: Int = 5, maxAge: Double = 300, clock: @escaping @Sendable () -> Double = { Date().timeIntervalSince1970 }) {
        self.maxEntries = maxEntries
        self.maxAge = maxAge
        self.clock = clock
    }

    public func last() -> Entry? { lock.withLock { entries.last } }

    public func add(_ text: String, destination: String = "") {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        lock.withLock {
            let now = clock()
            entries.append(Entry(time: now, text: text, destination: destination.trimmingCharacters(in: .whitespaces)))
            entries.removeAll { $0.time <= now - maxAge }
            if entries.count > maxEntries { entries.removeFirst(entries.count - maxEntries) }
        }
    }

    /// For tests: move the newest entry back in time.
    func backdateLast(by seconds: Double) {
        lock.withLock { if !entries.isEmpty { entries[entries.count - 1].time -= seconds } }
    }
}

/// Dictations reach the screen in the order they were released.
///
/// A new recording can start while the last one is still being corrected, and a short
/// second dictation can be ready first: typing it then would splice it into the middle of
/// the first one's streamed text. Each session takes a ticket on release and waits for its
/// turn just before its first keystroke, so its correction still runs in parallel.
public final class OutputOrder: @unchecked Sendable {
    private let condition = NSCondition()
    private var issued = 0
    private var serving = 0
    private var finished = Set<Int>()
    /// Never wait forever on a session that hung.
    let patience: Double

    public init(patience: Double = 15) { self.patience = patience }

    public func take() -> Int {
        condition.lock(); defer { condition.unlock() }
        defer { issued += 1 }
        return issued
    }

    /// Block until it is this ticket's turn, or patience runs out.
    public func waitBlocking(_ ticket: Int) {
        let deadline = Date().addingTimeInterval(patience)
        condition.lock(); defer { condition.unlock() }
        while serving < ticket {
            if !condition.wait(until: deadline) {
                Log.line("[Session] The previous dictation is still typing; going ahead")
                return
            }
        }
    }

    public func wait(_ ticket: Int) async {
        await blocking { [self] in waitBlocking(ticket) }
    }

    public func done(_ ticket: Int) {
        condition.lock(); defer { condition.unlock() }
        finished.insert(ticket)
        while finished.contains(serving) {
            finished.remove(serving)
            serving += 1
        }
        condition.broadcast()
    }
}

// MARK: - Plumbing

/// Results arriving from tasks, taken one at a time with a timeout: waiting on a slow
/// recognizer must be something the session can walk away from, which a task group's
/// scope (it waits for every child, cancelled or not) can't do.
final class Inbox<T: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [T] = []
    private var waiter: (id: Int, continuation: CheckedContinuation<T?, Never>)?
    private var nextId = 0

    func put(_ item: T) {
        lock.lock()
        if let waiter {
            self.waiter = nil
            lock.unlock()
            waiter.continuation.resume(returning: item)
        } else {
            items.append(item)
            lock.unlock()
        }
    }

    /// The next item, or nil once `timeout` seconds pass without one.
    func take(timeout: Double) async -> T? {
        await withCheckedContinuation { continuation in
            lock.lock()
            if !items.isEmpty {
                let item = items.removeFirst()
                lock.unlock()
                continuation.resume(returning: item)
                return
            }
            nextId += 1
            let id = nextId
            waiter = (id, continuation)
            lock.unlock()
            DispatchQueue.global().asyncAfter(deadline: .now() + max(0, timeout)) { [self] in
                lock.lock()
                guard let waiting = waiter, waiting.id == id else { lock.unlock(); return }
                waiter = nil
                lock.unlock()
                waiting.continuation.resume(returning: nil)
            }
        }
    }
}

/// Run blocking platform work (accessibility calls, a synthetic copy) off the cooperative
/// thread pool, and wait for it.
public func blocking<T: Sendable>(_ work: @escaping @Sendable () -> T) async -> T {
    await withCheckedContinuation { continuation in
        DispatchQueue.global(qos: .userInitiated).async { continuation.resume(returning: work()) }
    }
}

/// Mutable state shared between a session's tasks.
public final class Locked<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Value
    public init(_ value: Value) { self.value = value }
    public func read<R>(_ body: (Value) -> R) -> R { lock.withLock { body(value) } }
    @discardableResult public func update<R>(_ body: (inout Value) -> R) -> R { lock.withLock { body(&value) } }
    public var current: Value { lock.withLock { value } }
}
