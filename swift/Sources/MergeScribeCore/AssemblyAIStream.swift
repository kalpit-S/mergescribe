import Foundation

/// A dictation streamed to AssemblyAI's real-time API (Universal 3.6 Pro) while it is spoken,
/// so the transcript is final about 150 ms after the speaker stops: measured on their own
/// recordings, against ~0.5 s for a batch request on the last chunk plus the wait on the others.
/// In what got typed it matched the three batch recognizers (Oct 2026, 114 recordings).
///
/// Opened on the key press. Audio heard before the session opens (~0.4 s) is held and sent the
/// moment it does; the service catches up faster than real time while the speaker talks.
public final class AssemblyAIStream: @unchecked Sendable {
    public static let endpoint = URL(string: "wss://streaming.assemblyai.com/v3/ws")!
    /// The API's limit on keyterms.
    public static let maxKeyterms = 100

    public let name: String
    private let request: URLRequest
    private let sampleRate: Int
    private let session: URLSession
    private let queue = DispatchQueue(label: "mergescribe.stream")   // sends and state, in order
    private var task: URLSessionWebSocketTask?
    private var open = false
    private var held: [Data] = []
    private var turns = StreamTurns()
    private var audioEnded = false
    private var endpointSent = false
    private var finalSinceEndpoint = false   // the service answered the end of turn
    private var graceOver = false            // or had its chance to
    private var failure: String?
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private let started = DispatchTime.now()
    private var openedAt: DispatchTime?
    private var endedAt: DispatchTime?
    private var finalAt: DispatchTime?
    private var samplesSent = 0
    private var lastMessage: (kind: String, at: DispatchTime)?

    public init(key: String, model: String, sampleRate: Int, keyterms: [String], session: URLSession = .shared) {
        name = "assemblyai-" + model.replacingOccurrences(of: ".", with: "-")
        self.sampleRate = sampleRate
        self.session = session
        var components = URLComponents(url: Self.endpoint, resolvingAgainstBaseURL: false)!
        var query = [URLQueryItem(name: "speech_model", value: model),
                     URLQueryItem(name: "sample_rate", value: String(sampleRate)),
                     URLQueryItem(name: "encoding", value: "pcm_s16le"),
                     URLQueryItem(name: "format_turns", value: "true"),
                     URLQueryItem(name: "mode", value: "balanced")]
        let terms = Array(keyterms.prefix(Self.maxKeyterms))
        if !terms.isEmpty, let json = try? JSONSerialization.data(withJSONObject: terms) {
            query.append(URLQueryItem(name: "keyterms_prompt", value: String(decoding: json, as: UTF8.self)))
        }
        components.queryItems = query
        var request = URLRequest(url: components.url!)
        request.setValue(key, forHTTPHeaderField: "Authorization")
        self.request = request
    }

    /// Connect; audio appended meanwhile is held until the session opens.
    public func start() {
        queue.async { [self] in
            let task = session.webSocketTask(with: request)
            self.task = task
            task.resume()
            receive(task)
        }
    }

    /// 16 kHz float samples, as the microphones deliver them.
    public func append(_ samples: [Float]) {
        // The service refuses a frame under 50 ms; only the last part-block can be one, and
        // silence after it changes nothing.
        let data = Self.frame(samples, sampleRate: sampleRate)
        queue.async { [self] in
            guard failure == nil, !audioEnded else { return }
            samplesSent += data.count / 2
            if open { send(.data(data)) } else { held.append(data) }
        }
    }

    /// The speaker released the key: finish the turn now rather than waiting for silence.
    public func endOfAudio() {
        queue.async { [self] in
            audioEnded = true
            endedAt = .now()
            if open { sendEndpoint() }
        }
        // A dictation with no speech never asks for its transcript (the session asks at once
        // otherwise): don't hold one of the account's few sessions open.
        queue.asyncAfter(deadline: .now() + 3) { [self] in
            if waiters.isEmpty { close() }
        }
    }

    /// The whole transcript, once every turn with words is final; nil when the stream failed or
    /// didn't finish within `timeout` of being asked, so the caller can fall back.
    public func transcript(timeout: Double) async -> String? {
        let deadline = DispatchTime.now() + timeout
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            queue.async { [self] in
                if isSettled { done.resume(); return }
                waiters.append(done)
                queue.asyncAfter(deadline: deadline) { [self] in
                    guard !isSettled else { return }
                    failure = failure ?? "no final transcript within \(timeout)s (\(state))"
                    settle()
                }
            }
        }
        return queue.sync { () -> String? in
            if let failure {
                Log.line("[Stream] \(failure)")
                close()
                return nil
            }
            close()
            return turns.text
        }
    }

    /// Milliseconds from the press to the open session, and from the end of audio to the final
    /// transcript: what streaming costs, for the log and metrics.
    public var timing: (connectMs: Int?, finalMs: Int?) {
        queue.sync {
            let ms = { (from: DispatchTime, to: DispatchTime?) in
                to.map { Int((Double($0.uptimeNanoseconds) - Double(from.uptimeNanoseconds)) / 1e6) }
            }
            return (ms(started, openedAt), endedAt.flatMap { ms($0, finalAt) }.map { max(0, $0) })
        }
    }

    public func cancel() {
        queue.async { [self] in
            failure = failure ?? "cancelled"
            settle()
            close()
        }
    }

    // MARK: - On the queue

    /// Done once the release has been answered. The service replies to ForceEndpoint with a
    /// final turn when speech is pending, and with nothing when the turn had already ended, so:
    /// a final turn after it settles at once; if every turn with words was final already, a
    /// short grace for one still on its way; with no words heard yet, wait (the timeout, then
    /// the batch fallback, cover a stream that never answers). Settling on "no turns yet" once
    /// lost a whole one-second dictation.
    private var isSettled: Bool {
        Self.settled(failed: failure != nil, endpointSent: endpointSent, turns: turns,
                     finalSinceEndpoint: finalSinceEndpoint, graceOver: graceOver)
    }

    static func settled(failed: Bool, endpointSent: Bool, turns: StreamTurns, finalSinceEndpoint: Bool, graceOver: Bool) -> Bool {
        if failed { return true }
        guard endpointSent, turns.hasWords, turns.isComplete else { return false }
        return finalSinceEndpoint || graceOver
    }

    private func settle() {
        guard isSettled else { return }
        if finalAt == nil, failure == nil { finalAt = .now() }
        let waiting = waiters
        waiters.removeAll()
        for waiter in waiting { waiter.resume() }
    }

    private func sendEndpoint() {
        guard !endpointSent else { return }
        endpointSent = true
        send(.string(#"{"type":"ForceEndpoint"}"#))
        queue.asyncAfter(deadline: .now() + Self.grace) { [self] in
            graceOver = true
            settle()
        }
    }

    /// How long a turn that had already ended gets for another to appear: the service's final
    /// for pending speech came 0.15 s (median) to 0.38 s (worst) after the audio.
    static let grace = 0.3

    private func send(_ message: URLSessionWebSocketTask.Message) {
        task?.send(message) { [weak self] error in
            guard let self, let error else { return }
            self.queue.async {
                self.failure = self.failure ?? "send failed: \(error.localizedDescription)"
                self.settle()
            }
        }
    }

    private func receive(_ task: URLSessionWebSocketTask) {
        task.receive { [weak self] result in
            guard let self else { return }
            self.queue.async {
                switch result {
                case .failure(let error):
                    if !self.isSettled { self.failure = self.failure ?? "connection: \(error.localizedDescription)" }
                    self.settle()
                case .success(let message):
                    let text: String? = switch message {
                    case .string(let text): text
                    case .data(let data): String(data: data, encoding: .utf8)
                    @unknown default: nil
                    }
                    if let text { self.handle(text) }
                    if self.task === task { self.receive(task) }   // until closed
                }
            }
        }
    }

    /// What the stream had when it gave up, so a stall can be told apart from silence.
    private var state: String {
        let ms = { (t: DispatchTime) in Int((Double(DispatchTime.now().uptimeNanoseconds) - Double(t.uptimeNanoseconds)) / 1e6) }
        let spoken = turns.turns.values.filter { !$0.text.trimmingCharacters(in: .whitespaces).isEmpty }
        return "opened \(openedAt.map { "\(Int((Double($0.uptimeNanoseconds) - Double(started.uptimeNanoseconds)) / 1e6))ms after the press" } ?? "never"), "
            + String(format: "%.1fs of audio sent, ", Double(samplesSent) / Double(sampleRate))
            + "endpoint \(endpointSent ? "sent" : "not sent"), \(turns.turns.count) turns (\(spoken.count) with words, "
            + "\(spoken.filter(\.final).count) final), last message \(lastMessage.map { "\($0.kind) \(ms($0.at))ms ago" } ?? "none")"
    }

    private func handle(_ text: String) {
        guard let json = try? JSONDecoder().decode(JSON.self, from: Data(text.utf8)) else {
            Log.line("[Stream] Unreadable message: \(text.prefix(200))")
            return
        }
        let kind = json["type"]?.string ?? "?"
        lastMessage = (kind == "Turn" ? "Turn \(json["turn_order"]?.int ?? -1)\(json["end_of_turn"]?.bool == true ? " end" : "")\(json["turn_is_formatted"]?.bool == true ? " formatted" : "")" : kind, .now())
        switch json["type"]?.string {
        case "Begin":
            open = true
            openedAt = .now()
            for data in held { send(.data(data)) }
            held.removeAll()
            if audioEnded { sendEndpoint() }
        case "Turn":
            turns.update(json)
            if endpointSent, json["end_of_turn"]?.bool == true, json["turn_is_formatted"]?.bool == true {
                finalSinceEndpoint = true
            }
            settle()
        case "Termination":
            break
        default:
            if let error = json["error"]?.string {
                failure = "AssemblyAI: \(error)"
                settle()
            }
        }
    }

    private func close() {
        guard let task else { return }
        self.task = nil
        if open { task.send(.string(#"{"type":"Terminate"}"#)) { _ in } }
        task.cancel(with: .normalClosure, reason: nil)
    }

    /// A frame of at least 50 ms, padded with silence.
    static func frame(_ samples: [Float], sampleRate: Int) -> Data {
        let minimum = sampleRate / 20
        return pcm16(samples.count < minimum ? samples + [Float](repeating: 0, count: minimum - samples.count) : samples)
    }

    /// 16-bit little-endian PCM, as the API takes it.
    static func pcm16(_ samples: [Float]) -> Data {
        var data = Data(capacity: samples.count * 2)
        for sample in samples {
            let value = Int16((max(-1, min(1, sample)) * 32767).rounded(.towardZero))
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        return data
    }
}

/// The turns of a streamed transcript, as the service revises them.
struct StreamTurns: Sendable {
    /// turn_order -> (transcript, final)
    private(set) var turns: [Int: (text: String, final: Bool)] = [:]

    mutating func update(_ message: JSON) {
        guard let order = message["turn_order"]?.int else { return }
        let final = message["end_of_turn"]?.bool == true && message["turn_is_formatted"]?.bool == true
        turns[order] = (message["transcript"]?.string ?? "", final)
    }

    /// Every turn with words is final. A turn that ended opens an empty next one, which never
    /// becomes final, so empty turns don't count; with no words at all there is nothing to wait for.
    var isComplete: Bool { turns.values.filter { !$0.text.trimmingCharacters(in: .whitespaces).isEmpty }.allSatisfy(\.final) }

    var hasWords: Bool { turns.values.contains { !$0.text.trimmingCharacters(in: .whitespaces).isEmpty } }

    var text: String {
        turns.sorted { $0.key < $1.key }.map(\.value.text)
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }.joined(separator: " ")
    }
}

/// The stream as a recognizer: the session asks for the final chunk's transcript and gets the
/// streamed one; if the stream failed or stalls, the batch recognizer transcribes the audio.
public struct StreamingRecognizer: Recognizer {
    public let stream: AssemblyAIStream
    public let fallback: Recognizer?
    /// How long to wait for the final transcript once asked; measured p90 is 0.25 s.
    public var timeout = 2.0

    public init(stream: AssemblyAIStream, fallback: Recognizer?) {
        self.stream = stream
        self.fallback = fallback
    }

    public var name: String { stream.name }
    public var singleInstance: Bool { true }

    public func transcribe(_ samples: [Float], mic: String) async throws -> TranscriptionResult {
        let started = DispatchTime.now()
        if let text = await stream.transcript(timeout: timeout) {
            let ms = Int((Double(DispatchTime.now().uptimeNanoseconds) - Double(started.uptimeNanoseconds)) / 1e6)
            let timing = stream.timing
            Log.line("[Stream] Opened \(timing.connectMs.map { "\($0)ms" } ?? "?") after the press; "
                     + "final \(timing.finalMs.map { "\($0)ms" } ?? "?") after the release")
            return TranscriptionResult(text: text, provider: name, mic: mic, latencyMs: ms)
        }
        guard let fallback else { return TranscriptionResult(text: "", provider: name, mic: mic, latencyMs: 0) }
        Log.line("[Stream] Falling back to \(fallback.name)")
        return try await fallback.transcribe(samples, mic: mic)
    }
}
