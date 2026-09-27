import Foundation
@testable import MergeScribeCore

/// A recognizer that answers after a delay (or never, or with an error).
struct FakeRecognizer: Recognizer {
    let name: String
    var singleInstance = false
    var delay: Double = 0
    var text: (_ samples: [Float]) -> String = { _ in "" }
    var fails = false
    var calls: Recorder<(String, Int)>? = nil

    init(_ name: String, delay: Double = 0, text: String = "", singleInstance: Bool = false, fails: Bool = false,
         calls: Recorder<(String, Int)>? = nil) {
        self.name = name
        self.delay = delay
        self.text = { _ in text }
        self.singleInstance = singleInstance
        self.fails = fails
        self.calls = calls
    }

    init(_ name: String, delay: @escaping @Sendable ([Float]) -> Double, text: @escaping @Sendable ([Float]) -> String) {
        self.name = name
        self.delayFor = delay
        self.text = text
    }

    var delayFor: (@Sendable ([Float]) -> Double)?

    func transcribe(_ samples: [Float], mic: String) async throws -> TranscriptionResult {
        calls?.add((mic, samples.count))
        let wait = delayFor?(samples) ?? delay
        // Deliberately ignores cancellation, like a request already on the wire.
        if wait > 0 { await blocking { Thread.sleep(forTimeInterval: wait) } }
        if fails { throw URLError(.badServerResponse) }
        return TranscriptionResult(text: text(samples), provider: name, mic: mic, latencyMs: Int(wait * 1000))
    }
}

extension FakeRecognizer: @unchecked Sendable {}

/// Everything appended, in order, from any thread.
final class Recorder<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [T] = []
    func add(_ item: T) { lock.withLock { items.append(item) } }
    var all: [T] { lock.withLock { items } }
    var count: Int { lock.withLock { items.count } }
}

final class FakeFocus: Focus, @unchecked Sendable {
    var context: AppContext? = AppContext(appName: "Editor", windowTitle: "t", bundleId: "com.x")
    var current: AppContext?          // what appContext() reports after the start; nil = same as context
    var origin: Origin? = nil
    var here: [Bool] = [true]         // successive answers to atOrigin (the last repeats)
    var comesBack = true
    var selection: String? = nil
    var captureDelay: Double = 0
    let returned = Recorder<Bool>()
    private var reads = 0
    private let lock = NSLock()

    func appContext() -> AppContext? {
        lock.withLock {
            reads += 1
            return reads == 1 ? context : (current ?? context)
        }
    }
    func captureOrigin() -> Origin? { origin }
    func atOrigin(_ origin: Origin) -> Bool {
        lock.withLock {
            let answer = here.first ?? true
            if here.count > 1 { here.removeFirst() }
            return answer
        }
    }
    func returnTo(_ origin: Origin) -> Bool { returned.add(comesBack); return comesBack }
    func selectedText() -> String? {
        if captureDelay > 0 { Thread.sleep(forTimeInterval: captureDelay) }
        return selection
    }
    func wakeFrontmostApp() {}
}

final class FakeOutput: TextOutput, @unchecked Sendable {
    let typedPieces = Recorder<String>()
    let clipboard = Recorder<String>()
    let notifications = Recorder<String>()
    let busy = Recorder<Bool>()
    func type(_ text: String) { typedPieces.add(text) }
    func copyToClipboard(_ text: String) { clipboard.add(text) }
    func notify(_ message: String) { notifications.add(message) }
    func playBusySound() { busy.add(true) }
    var typed: String { typedPieces.all.joined() }
}

/// A correction that streams scripted tokens, then returns, fails or is cut off.
final class FakeCorrector: Correcting, @unchecked Sendable {
    var tokens: [String] = []
    var result: String? = nil               // nil: the tokens joined
    var delay: Double = 0
    var tokenGap: Double = 0
    var interrupted: CorrectionInterrupted? = nil
    var fails = false
    var metadataModel: String? = nil
    var editResult = "edited"
    let calls = Recorder<[TranscriptionResult]>()
    let edits = Recorder<(String, String)>()

    func correct(_ results: [TranscriptionResult], context: AppContext?, settings: Settings,
                 onDelta: CorrectionClient.Delta?, onMetadata: (@Sendable (LLMCorrectionResult) -> Void)?,
                 onGenerationMetadata: (@Sendable (String, JSON) -> Void)?) async throws -> String {
        calls.add(results)
        if delay > 0 { try? await Task.sleep(for: .milliseconds(Int(delay * 1000))) }
        if fails { throw URLError(.timedOut) }
        for token in tokens {
            onDelta?(token)
            if tokenGap > 0 { usleep(UInt32(tokenGap * 1_000_000)) }
        }
        if let interrupted { throw interrupted }
        let text = result ?? tokens.joined()
        if let model = metadataModel {
            onMetadata?(LLMCorrectionResult(text: text, provider: "openrouter", model: model, inputTokensEst: 1, latencyMs: 1))
        }
        return text
    }

    func edit(selected: String, command: String, settings: Settings) async -> String {
        edits.add((selected, command))
        return editResult
    }
}

final class FakeJudge: Judging, @unchecked Sendable {
    var verdict: Judge.Verdict?
    var delay: Double = 0
    let asked = Recorder<[TranscriptionResult]>()
    init(_ verdict: Judge.Verdict? = nil, delay: Double = 0) { self.verdict = verdict; self.delay = delay }

    func judge(_ results: [TranscriptionResult], settings: Settings, context: AppContext?) async -> Judge.Verdict? {
        asked.add(results)
        if delay > 0 { try? await Task.sleep(for: .milliseconds(Int(delay * 1000))) }
        return verdict
    }

    static func clean(_ text: String, latency: Int = 240) -> Judge.Verdict {
        Judge.Verdict(clean: true, text: text, scores: [.init(name: "filler", value: 0.1)], model: "jev", latencyMs: latency)
    }
}

final class FakeMetrics: MetricsLogging, @unchecked Sendable {
    let events = Recorder<(String, [String: JSON])>()
    func log(_ event: String, _ fields: [String: JSON]) { events.add((event, fields)) }
    func named(_ event: String) -> [[String: JSON]] { events.all.filter { $0.0 == event }.map(\.1) }
}

final class FakeObserver: SessionObserver, @unchecked Sendable {
    let events = Recorder<String>()
    func streamsPlanned(session: String, streams: [String]) { events.add("planned \(streams.joined(separator: ","))") }
    func streamLanded(session: String, stream: String, final: Bool) { events.add("landed \(stream)\(final ? " final" : "")") }
    func streamDropped(session: String, stream: String) { events.add("dropped \(stream)") }
    func transcriptionDone(session: String) { events.add("done") }
    func correctionSkipped(session: String) { events.add("skipped") }
    func dictationDiscarded(session: String) { events.add("discarded") }
    func tokenTyped(session: String) {}
    func partialText(session: String, text: String) {}
}

final class FakeEditWatcher: EditWatching, @unchecked Sendable {
    let callbacks = Recorder<@Sendable (EditDiff.Outcome, Double, String, String) -> Void>()
    func watch(typed: String, app: String,
               onResult: @escaping @Sendable (EditDiff.Outcome, Double, String, String) -> Void) {
        callbacks.add(onResult)
    }
}

/// A session wired to fakes.
struct Rig {
    var settings = Settings()
    var recognizers: [Recognizer] = []
    let focus = FakeFocus()
    let output = FakeOutput()
    let corrector = FakeCorrector()
    var judge = FakeJudge()
    let metrics = FakeMetrics()
    let observer = FakeObserver()
    let watcher = FakeEditWatcher()
    var corpusURL: URL? = nil
    let history = TranscriptionHistory()
    let order = OutputOrder()

    init() {
        settings.enabledMics = ["mic1"]
        settings.spaceBetweenDictations = false
        settings.editFeedbackEnabled = false
        settings.providerDeadlineMultiplier = 1.0
        settings.providerDeadlineMinMs = 900
        settings.judgeEnabled = true
        settings.sampleRate = 16000
    }

    var services: SessionServices {
        SessionServices(focus: focus, output: output, corrector: corrector, judge: judge, editWatcher: watcher,
                        metrics: metrics, corpus: corpusURL.map { CorrectionsCorpus(url: $0) }, observer: observer)
    }

    func session() -> Session {
        let s = Session(settings: settings, recognizers: recognizers, services: services, history: history,
                        outputOrder: order, outputLock: NSLock())
        s.state.update {
            $0.startTime = Date().timeIntervalSince1970
            $0.context = focus.context
        }
        return s
    }
}

func speech(_ seconds: Double = 1.0, value: Float? = nil) -> AudioChunk {
    let n = Int(16000 * seconds)
    return AudioChunk([("mic1", value.map { Array(repeating: $0, count: n) } ?? (0..<n).map { i in Float(sin(Double(i) * 0.3) * 0.3) })])
}

func result(_ text: String, _ provider: String = "p", chunk: Int = 0) -> TranscriptionResult {
    TranscriptionResult(text: text, provider: provider, mic: "m", latencyMs: 1, chunk: chunk)
}

/// Wait for a condition that another task will make true.
func eventually(_ timeout: Double = 3, _ condition: () -> Bool) async -> Bool {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
        if condition() { return true }
        try? await Task.sleep(for: .milliseconds(10))
    }
    return condition()
}
