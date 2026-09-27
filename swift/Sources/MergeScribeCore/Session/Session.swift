import Foundation

/// Everything a session talks to, shared by every session the manager starts.
public struct SessionServices: Sendable {
    public var focus: Focus
    public var output: TextOutput
    public var corrector: Correcting
    public var judge: Judging
    public var editWatcher: EditWatching?
    public var metrics: MetricsLogging?
    public var training: TrainingWriter?
    public var corpus: CorrectionsCorpus?
    public var observer: SessionObserver?
    public var clock: @Sendable () -> Double

    public init(focus: Focus, output: TextOutput, corrector: Correcting, judge: Judging,
                editWatcher: EditWatching? = nil, metrics: MetricsLogging? = nil, training: TrainingWriter? = nil,
                corpus: CorrectionsCorpus? = nil, observer: SessionObserver? = nil,
                clock: @escaping @Sendable () -> Double = { Date().timeIntervalSince1970 }) {
        self.focus = focus
        self.output = output
        self.corrector = corrector
        self.judge = judge
        self.editWatcher = editWatcher
        self.metrics = metrics
        self.training = training
        self.corpus = corpus
        self.observer = observer
        self.clock = clock
    }
}

/// One dictation, from key press to typed text.
///
/// Chunks transcribe on every recognizer in parallel while the speaker is still talking.
/// On release the last chunk joins them; then Jev and the correction model race, and the
/// winner is typed into the window the dictation started in.
public final class Session: @unchecked Sendable {
    /// How recently the previous dictation must have landed for this one to continue it.
    static let continuationSeconds = 8.0
    /// Absolute ceiling on waiting for a chunk's recognizers, whatever the deadline.
    var chunkHardTimeout = 30.0

    public let id: UUID
    public let settings: Settings
    let recognizers: [Recognizer]
    let services: SessionServices
    let history: TranscriptionHistory
    let outputOrder: OutputOrder
    let outputLock: NSLock
    var onComplete: (@Sendable (Session) -> Void)?

    /// Short handle for the log, so overlapping sessions can be told apart.
    public var tag: String { String(id.uuidString.lowercased().prefix(4)) }

    struct State {
        var isCapturing = false
        var startTime = 0.0
        var finalizeStart = 0.0
        var context: AppContext?
        var origin: Origin?
        var selectedText: String?
        var chunkResults: [(num: Int, results: [TranscriptionResult], consensus: String?)] = []
        var chunkCounter = 0
        var chunksSubmitted = 0
        var pending: [Task<Void, Never>] = []
        var audio: [(mic: String, blocks: [[Float]])] = []
        var allResults: [TranscriptionResult] = []
        var llmResult: LLMCorrectionResult?
        var outputMethod = ""
        var finalText = ""
        var ticket: Int?
        var turnTaken = false
        var partialOutput = false
    }
    let state = Locked(State())
    private var contextTask: Task<Void, Never>?

    init(id: UUID = UUID(), settings: Settings, recognizers: [Recognizer], services: SessionServices,
         history: TranscriptionHistory, outputOrder: OutputOrder, outputLock: NSLock) {
        self.id = id
        self.settings = settings
        self.recognizers = recognizers
        self.services = services
        self.history = history
        self.outputOrder = outputOrder
        self.outputLock = outputLock
    }

    public var isCapturing: Bool { state.read(\.isCapturing) }
    /// What the dictation became, and how it got out: "streamed", "judged" or "typed" went
    /// into the window it started in; "clipboard" didn't.
    public var outcome: (text: String, method: String, partial: Bool) {
        state.read { ($0.finalText, $0.outputMethod, $0.partialOutput) }
    }
    public var context: AppContext? { state.read(\.context) }
    var sid: String { id.uuidString.lowercased() }

    // MARK: - Starting

    /// Mark the session live and capture context off the critical path. Selection capture
    /// is a synthetic copy (median 353ms, worst 1.4s): doing it before recording started
    /// pushed the first word past the pre-roll, so it overlaps the recording instead.
    public func start(mics: [String]) {
        state.update {
            $0.isCapturing = true
            $0.startTime = services.clock()
        }
        notify { $0.streamsPlanned(session: sid, streams: streamPlan(mics).map { "\($0.recognizer.name)/\($0.mic)" }) }
        let focus = services.focus, feedback = settings.editFeedbackEnabled
        contextTask = Task { [state] in
            await Log.$tag.withValue(tag) {
                let (context, origin, selected) = await blocking {
                    let context = focus.appContext()
                    let origin = focus.captureOrigin()
                    let selected = focus.selectedText()
                    // Wake the destination app's accessibility tree while the user is still
                    // talking, so its field is readable when the edit watcher looks.
                    if feedback { focus.wakeFrontmostApp() }
                    return (context, origin, selected)
                }
                state.update {
                    $0.context = context
                    $0.origin = origin
                    $0.selectedText = selected
                }
            }
        }
        log("session_start", ["context": .object([:]), "enabled_mics": JSON(settings.enabledMics),
                              "providers": JSON(settings.enabledProviders)])
    }

    /// Context is needed at finalize; by then it is almost always ready.
    func awaitContext(timeout: Double = 2.0) async {
        guard let task = contextTask else { return }
        let inbox = Inbox<Bool>()
        Task { await task.value; inbox.put(true) }
        _ = await inbox.take(timeout: timeout)
    }

    /// Which (mic, recognizer) pairs transcribe audio from these mics. A serialized
    /// recognizer runs on the primary mic only: the first enabled mic that produced audio.
    func streamPlan(_ mics: [String]) -> [(mic: String, recognizer: Recognizer)] {
        let primary = settings.enabledMics.first { mics.contains($0) } ?? mics.first
        return mics.flatMap { mic in
            recognizers.filter { !$0.singleInstance || mic == primary }.map { (mic, $0) }
        }
    }

    // MARK: - Chunks while recording

    /// A chunk emitted on a pause while the key is still held.
    public func chunkReady(_ chunk: AudioChunk) {
        guard !chunk.isEmpty else { return }
        let num = state.update { s -> Int in
            Self.keepAudio(chunk, in: &s)
            s.chunkCounter += 1
            s.chunksSubmitted += 1
            return s.chunkCounter
        }
        let longest = chunk.mics.map { Double($0.samples.count) / Double(settings.sampleRate) * 1000 }.max() ?? 0
        log("chunk_received", ["chunk_num": JSON(num), "mic_names": JSON(chunk.mics.map(\.name)),
                               "audio_duration_ms": .number(longest)])
        let task = Task { await Log.$tag.withValue(tag) { await self.transcribe(chunk, num: num) } }
        state.update { $0.pending.append(task) }
    }

    static func keepAudio(_ chunk: AudioChunk, in s: inout State) {
        for (mic, samples) in chunk.mics where !samples.isEmpty {
            if let i = s.audio.firstIndex(where: { $0.mic == mic }) {
                s.audio[i].blocks.append(samples)
            } else {
                s.audio.append((mic, [samples]))
            }
        }
    }

    /// Transcribe one chunk on every stream, checking consensus as results arrive.
    ///
    /// A chunk is only as fast as its slowest recognizer. Once one has answered with words,
    /// the rest get a proportional grace period and are then left behind; an answer that
    /// heard nothing is no evidence the others are late.
    func transcribe(_ chunk: AudioChunk, num: Int? = nil, final: Bool = false) async {
        let plan = streamPlan(chunk.heard.map(\.name))
        guard !plan.isEmpty else { return }
        let num = num ?? state.update { s -> Int in s.chunkCounter += 1; return s.chunkCounter }

        let inbox = Inbox<(Int, Result<TranscriptionResult, Error>)>()
        let tasks = plan.enumerated().map { index, stream in
            let samples = chunk.samples(stream.mic) ?? []
            return Task {
                do { inbox.put((index, .success(try await stream.recognizer.transcribe(samples, mic: stream.mic)))) }
                catch { inbox.put((index, .failure(error))) }
            }
        }

        let multiplier = max(0, settings.providerDeadlineMultiplier)
        let minGrace = max(0, Double(settings.providerDeadlineMinMs) / 1000)
        let clock = ContinuousClock(), started = clock.now
        var results: [TranscriptionResult] = []
        var finished = Set<Int>()
        var consensus: String?
        var matching = 0
        var deadline: Double?     // seconds after `started`

        while finished.count < plan.count {
            let elapsed = (clock.now - started).seconds
            let timeout = deadline.map { $0 - elapsed } ?? (chunkHardTimeout - elapsed)
            guard timeout > 0, let (index, outcome) = await inbox.take(timeout: timeout) else { break }
            finished.insert(index)
            let stream = "\(plan[index].recognizer.name)/\(plan[index].mic)"
            switch outcome {
            case .success(var result):
                result.chunk = num
                results.append(result)
                state.update { $0.allResults.append(result) }
                notify { $0.streamLanded(session: sid, stream: stream, final: final) }
                Log.line("[Chunk \(num)] \(result.provider)/\(result.mic): \(String(format: "%.2f", Double(result.latencyMs) / 1000))s")
                Log.line("    \"\(result.text)\"")
                log("transcription", ["chunk_num": JSON(num), "provider": .string(result.provider), "mic": .string(result.mic),
                                      "latency_ms": JSON(result.latencyMs), "text": .string(String(result.text.prefix(200))),
                                      "confidence": JSON(result.confidence)])
                if results.count >= settings.consensusThreshold,
                   let agreed = Consensus.check(results, threshold: settings.consensusThreshold) {
                    consensus = agreed
                    matching = Consensus.matching(results, agreed)
                    Log.line("[Chunk \(num)] ✓ Consensus (\(matching) agree):")
                    Log.line("    \"\(agreed)\"")
                }
            case .failure(let error):
                Log.line("[Chunk \(num)] Provider error (\(stream)): \(error)")
                notify { $0.streamDropped(session: sid, stream: stream) }
            }
            if consensus != nil { break }
            if deadline == nil && multiplier > 0 && finished.count < plan.count
                && results.contains(where: { !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) {
                let elapsed = (clock.now - started).seconds
                deadline = min(max(elapsed * (1 + multiplier), minGrace), chunkHardTimeout)
            }
        }

        let unfinished = plan.indices.filter { !finished.contains($0) }
        var abandoned: [String] = []
        for index in unfinished {
            tasks[index].cancel()
            if consensus == nil {
                let stream = "\(plan[index].recognizer.name)/\(plan[index].mic)"
                abandoned.append(stream)
                notify { $0.streamDropped(session: sid, stream: stream) }
            }
        }
        if !abandoned.isEmpty {
            let waited = (clock.now - started).seconds * 1000
            Log.line("[Chunk \(num)] Moved on after \(Int(waited.rounded()))ms without \(abandoned.joined(separator: ", "))")
            log("provider_deadline", ["chunk_num": JSON(num), "waited_ms": JSON(Int(waited.rounded())),
                                      "abandoned": JSON(abandoned), "kept": JSON(results.map { "\($0.provider)/\($0.mic)" })])
        }
        log("consensus", ["chunk_num": JSON(num), "reached": .bool(consensus != nil),
                          "text": consensus.map { .string(String($0.prefix(200))) } ?? .null, "matching_count": JSON(matching)])
        state.update { $0.chunkResults.append((num, results, consensus)) }
        publishPartialText()
    }

    /// The raw transcript so far, for the HUD: a progress signal during long dictations,
    /// where there is otherwise no feedback for 40s or more.
    func publishPartialText() {
        guard services.observer != nil else { return }
        let text = aggregate().texts.map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }.joined(separator: " ")
        notify { $0.partialText(session: sid, text: text) }
    }

    /// The chunks in the order they were spoken: each one's consensus, or else its longest
    /// transcript; and every result.
    func aggregate() -> (texts: [String], results: [TranscriptionResult]) {
        let parts = state.read { $0.chunkResults }.sorted { $0.num < $1.num }
        var texts: [String] = [], all: [TranscriptionResult] = []
        for part in parts {
            all += part.results
            if let consensus = part.consensus {
                texts.append(consensus)
            } else {
                let nonEmpty = part.results.filter { !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
                var best: TranscriptionResult?
                for r in nonEmpty where best == nil
                    || r.text.split(whereSeparator: \.isWhitespace).count > best!.text.split(whereSeparator: \.isWhitespace).count {
                    best = r
                }
                if let best { texts.append(best.text) }
            }
        }
        return (texts, all)
    }

    // MARK: - Release

    /// The key came up. The mic is free now; everything after is post-processing, and a new
    /// recording may start while this one finishes.
    public func finalize(_ final: FinalChunk) {
        let ticket = outputOrder.take()
        state.update {
            $0.isCapturing = false
            $0.ticket = ticket
        }
        Task { await Log.$tag.withValue(tag) { await self.finalizeInBackground(final) } }
    }

    func finalizeInBackground(_ final: FinalChunk) async {
        defer { teardown() }
        let finalizeStart = services.clock()
        state.update { $0.finalizeStart = finalizeStart }
        collectFinalAudio(final.chunk)
        Log.line("[Timing] Key held: \(String(format: "%.2f", finalizeStart - state.read(\.startTime)))s")

        // A stray tap is a fraction of a second of room tone, and recognizers hallucinate
        // on that. Only the whole session is dropped, and only when nothing was sent while
        // recording: a release just after a long chunk left must keep that chunk's speech.
        let anySent = state.read { !$0.chunkResults.isEmpty || $0.chunksSubmitted > 0 }
        if !anySent && !final.hasSpeech {
            Log.line("[Session] No speech captured - ignoring")
            log("session_discarded", ["reason": "no_speech"])
            return
        }
        await transcribeRemaining(final.hasSpeech ? final.chunk : AudioChunk())

        let (texts, results) = aggregate()
        guard !texts.isEmpty else {
            Log.line("No transcription results")
            return
        }
        let combined = texts.joined(separator: " ")
        Log.line("[Session] \(state.read(\.chunkResults.count)) chunks, \(results.count) transcriptions")

        // Selection capture can outlast a fast transcription; deciding before it lands
        // would type the instruction instead of applying it.
        await awaitContext()
        if let selected = state.read(\.selectedText) {
            await runEditMode(command: combined, selected: selected)
            return
        }
        await correctAndOutput(results, combined: combined)
    }

    func collectFinalAudio(_ chunk: AudioChunk) {
        for (mic, samples) in chunk.heard {
            Log.line("[Audio] \(mic): \(String(format: "%.2f", Double(samples.count) / Double(settings.sampleRate)))s of audio")
        }
        state.update { Self.keepAudio(chunk, in: &$0) }
    }

    /// Transcribe the final chunk, and wait for any still in flight.
    func transcribeRemaining(_ chunk: AudioChunk) async {
        let started = services.clock()
        if !chunk.isEmpty { await transcribe(chunk, final: true) }
        let pending = state.read(\.pending)
        let inbox = Inbox<Bool>()
        Task {
            for task in pending { await task.value }
            inbox.put(true)
        }
        _ = await inbox.take(timeout: chunkHardTimeout)
        Log.line("[Timing] Transcription: \(String(format: "%.2f", services.clock() - started))s")
        notify { $0.transcriptionDone(session: sid) }
    }

    /// Selection + speech: the transcript is an instruction, not content.
    func runEditMode(command: String, selected: String) async {
        Log.line("[Session] Text edit mode: \"\(command.prefix(50))...\"")
        await outputAsync(await services.corrector.edit(selected: selected, command: command, settings: settings))
    }

    // MARK: - Correction

    /// Correct and put the result on screen: streamed into the window the dictation started
    /// in, brought back if the speaker moved on; the clipboard if it can't come back.
    func correctAndOutput(_ results: [TranscriptionResult], combined: String) async {
        await awaitContext()
        let sameWindow = await inPlace()
        let corrected = sameWindow ? await streamCorrection(results) : await clipboardCorrection(results)
        guard var corrected else {
            // The model judged that the speaker called the whole dictation off.
            Log.line("[Session] Speaker called the dictation off - typing nothing")
            notify { $0.dictationDiscarded(session: sid) }
            log("session_discarded", ["reason": "called_off"])
            return
        }
        if sameWindow {
            if corrected.isEmpty {
                Log.line("[Session] LLM correction failed - using raw transcript")
                await outputAsync(combined)
                return
            }
            state.update { $0.finalText = corrected }
            logOutputStats(corrected, provider: state.read(\.outputMethod) == "judged" ? "judge" : "openrouter")
            if !state.read(\.partialOutput) { startEditWatch(corrected) }
        } else {
            if corrected.isEmpty { corrected = combined }
            let text = corrected
            state.update { $0.finalText = text }
            await myTurn()
            services.output.copyToClipboard(text)
            services.output.notify("Window changed - copied to clipboard")
        }
        // History holds the raw transcript, never the correction: feeding corrected text
        // back makes the model echo its own phrasing.
        history.add(combined, destination: outputDestination())
    }

    /// Where this session's text is going: the app and window it started in.
    func outputDestination() -> String {
        guard let context = state.read(\.context) else { return "" }
        return context.windowTitle.isEmpty ? context.appName : "\(context.appName): \(context.windowTitle)"
    }

    /// True when this dictation appends to the previous one, so it leads with a space.
    /// Over 675 recordings started within 8s of the previous, 673 went to the same app, so
    /// recency plus destination is a reliable test.
    func continuesPrevious() -> Bool {
        guard settings.spaceBetweenDictations, let previous = history.last() else { return false }
        guard services.clock() - previous.time <= Self.continuationSeconds else { return false }
        return !previous.destination.isEmpty && previous.destination == outputDestination()
    }

    /// Type tokens as they arrive, unless the judge says they need no correcting.
    ///
    /// The judge and the correction are asked at the same moment and the first usable
    /// answer wins: a clean verdict (~250ms) beats the correction's first token (~750ms)
    /// and types the transcript. Returns the typed text, "" if the correction failed, or
    /// nil when the model said the speaker called the whole dictation off.
    func streamCorrection(_ results: [TranscriptionResult]) async -> String? {
        state.update { $0.outputMethod = "streamed" }
        let stream = StreamState(continuing: continuesPrevious())

        let onDelta: CorrectionClient.Delta = { [self] token in
            if stream.claim("correction") { emit(token, stream) }
        }
        let interrupted = Locked<CorrectionInterrupted?>(nil)
        let correction = Task { [self] in
            await Log.$tag.withValue(tag) {
                do {
                    _ = try await services.corrector.correct(results, context: state.read(\.context), settings: settings,
                                                             onDelta: onDelta, onMetadata: { [self] in logLLM($0) },
                                                             onGenerationMetadata: { [self] in logGeneration($0, $1) })
                } catch let error as CorrectionInterrupted {
                    interrupted.update { $0 = error }
                } catch {
                    // Nothing was typed by it, so the raw transcript goes in instead.
                    Log.line("[LLM] Correction failed: \(error)")
                }
            }
        }
        let verdict = startJudge(results)
        if let clean = await judgeWins(verdict, stream) {
            emit(clean.text, stream)
            return stream.typed
        }
        await correction.value
        emit(stream.flushMarker(), stream)
        if stream.calledOff { return nil }
        if let cut = interrupted.current {
            // Part of it is typed and the rest can't be joined on cleanly.
            let whole = cut.complete.isEmpty
                ? results.map(\.text).filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }.joined(separator: " ")
                : cut.complete
            handOver(whole, "The correction was cut off - the full text is on the clipboard")
            return stream.typed
        }
        let held = stream.heldText
        if !held.isEmpty {
            handOver(String(held.drop(while: \.isWhitespace)), "Window changed while typing - the rest is on the clipboard")
            return stream.typed + held
        }
        return stream.typed
    }

    /// One piece of output on its way to the keyboard.
    func emit(_ text: String, _ stream: StreamState) {
        guard let text = stream.prepare(text) else { return }
        waitForTurnBlocking()
        // Held once focus is lost, and from then on: the text can't resume with a hole in it.
        if !stream.stillInPlace({ [self] in inPlaceNow(midStream: true) }) {
            stream.hold(text)
            return
        }
        stream.typedPiece(text)
        outputLock.withLock { services.output.type(text) }
        notify { $0.tokenTyped(session: sid) }
    }

    /// Only part of a dictation reached the field; put the rest where the user can paste it.
    func handOver(_ text: String, _ message: String) {
        state.update { $0.partialOutput = true }
        services.output.copyToClipboard(text)
        services.output.notify(message)
        Log.line("[Session] \(message)")
    }

    /// Ask the judge in the background.
    ///
    /// Single-chunk dictations only: across chunks the transcripts are consecutive pieces of
    /// one dictation, not alternative readings, so picking one would drop the rest.
    func startJudge(_ results: [TranscriptionResult]) -> Task<Judge.Verdict?, Never> {
        let chunks = state.read(\.chunkResults.count)
        let judge = services.judge, settings = settings, context = state.read(\.context)
        return Task {
            if chunks > 1 { return nil }
            return await Log.$tag.withValue(tag) { await judge.judge(results, settings: settings, context: context) }
        }
    }

    /// Wait for the judge only while the correction has typed nothing. The verdict when it
    /// came back clean in time and this call owns the output; nil otherwise.
    func judgeWins(_ task: Task<Judge.Verdict?, Never>, _ stream: StreamState) async -> Judge.Verdict? {
        let done = Locked<Judge.Verdict??>(nil)
        Task { let v = await task.value; done.update { $0 = .some(v) } }
        let deadline = ContinuousClock.now + .milliseconds(settings.judgeTimeoutMs)
        while done.current == nil && !stream.claimed && ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(20))
        }
        guard let verdict = done.current ?? nil else { return nil }
        log("judge", ["clean": .bool(verdict.clean),
                      "scores": .object(Dictionary(uniqueKeysWithValues: verdict.scores.map { ($0.name, JSON(($0.value * 1000).rounded() / 1000)) })),
                      "latency_ms": JSON(verdict.latencyMs), "model": .string(verdict.model)])
        // Every verdict, not just the ones that win: a judge that only speaks up when it
        // acts looks like a judge that never ran.
        let scores = verdict.scores.map { "\($0.name) \(String(format: "%.2f", $0.value))" }.joined(separator: " ")
        Log.line("[Judge] \(verdict.clean ? "no correction needed" : "correcting") (\(scores)) | \(verdict.latencyMs)ms")
        guard verdict.clean, stream.claim("judge") else { return nil }
        Log.line("[Judge] typing the transcript (\(verdict.latencyMs)ms)")
        state.update { $0.outputMethod = "judged" }
        notify { $0.correctionSkipped(session: sid) }
        return verdict
    }

    /// Correct without streaming, for when the target window is gone. Same contract as
    /// streamCorrection.
    func clipboardCorrection(_ results: [TranscriptionResult]) async -> String? {
        state.update { $0.outputMethod = "clipboard" }
        let corrected = (try? await services.corrector.correct(results, context: state.read(\.context), settings: settings,
                                                               onDelta: nil, onMetadata: { [self] in logLLM($0) },
                                                               onGenerationMetadata: { [self] in logGeneration($0, $1) })) ?? ""
        if corrected.isEmpty {
            Log.line("[Session] LLM correction failed - using raw transcript")
            return ""
        }
        if isNothing(corrected) { return nil }
        // Pasted deliberately, so never a separator; just flatten invented line breaks.
        var filter = DictationFilter()
        return filter.feed(corrected)
    }

    func logLLM(_ result: LLMCorrectionResult) {
        state.update { $0.llmResult = result }
        log("llm_correction", ["provider": .string(result.provider), "model": .string(result.model),
                               "generation_id": JSON(result.generationId), "backend_provider": JSON(result.backendProvider),
                               "resolved_model": JSON(result.resolvedModel), "provider_order": JSON(result.providerOrder),
                               "allow_fallbacks": JSON(result.allowFallbacks), "reasoning_effort": .string(result.reasoningEffort),
                               "usage": result.usage, "input_tokens_est": JSON(result.inputTokensEst),
                               "latency_ms": .number(result.latencyMs)])
    }

    /// Cost details arrive after the session finishes, so they log separately.
    func logGeneration(_ id: String, _ data: JSON) {
        var fields = data.object ?? [:]
        fields["generation_id"] = .string(id)
        log("llm_generation", fields)
    }

    // MARK: - Output

    /// Wait for earlier dictations to finish typing, once, before this one types.
    func myTurn() async {
        let ticket: Int? = state.update { s in
            guard !s.turnTaken, let t = s.ticket else { return nil }
            s.turnTaken = true
            return t
        }
        if let ticket { await outputOrder.wait(ticket) }
    }

    /// myTurn() from synchronous code: the correction's delta callback runs inside its
    /// stream reader, which is waiting on us anyway.
    func waitForTurnBlocking() {
        let ticket: Int? = state.update { s in
            guard !s.turnTaken, let t = s.ticket else { return nil }
            s.turnTaken = true
            return t
        }
        if let ticket { outputOrder.waitBlocking(ticket) }
    }

    /// True when the words can go where the dictation started, bringing that window back
    /// first if the speaker moved on to read something. False only when it can't come back.
    func inPlace() async -> Bool {
        if await blocking({ [self] in inPlaceNow() }) { return true }
        // Bringing a window back moves focus, so it waits for this dictation's turn: an
        // earlier one may still be typing into the window that has focus now.
        await myTurn()
        return await blocking { [self] in inPlaceNow(bringBack: true) }
    }

    /// Checked between streamed tokens (at most every 100ms by StreamState): here or not,
    /// without bringing anything back mid-stream.
    func inPlaceNow(bringBack: Bool = false, midStream: Bool = false) -> Bool {
        let (origin, context) = state.read { ($0.origin, $0.context) }
        let focus = services.focus
        guard let origin else {
            // Mid-stream there is nothing to compare against; before output, the same app
            // is as close as it gets.
            if midStream { return true }
            guard let context, let current = focus.appContext() else { return false }
            return current.bundleId == context.bundleId
        }
        if focus.atOrigin(origin) { return true }
        guard bringBack else { return false }
        if focus.returnTo(origin) {
            Log.line("[Output] Back to \(context?.appName ?? "the starting window") to type")
            return true
        }
        Log.line("[Output] Couldn't bring back the window the dictation started in")
        return false
    }

    /// Type a finished text (the raw transcript, or an edit), or copy it when the window
    /// it belongs in is gone.
    func outputAsync(_ text: String) async {
        guard !text.isEmpty else { return }
        state.update { $0.finalText = text }
        let provider = state.read(\.llmResult)?.provider ?? "consensus"
        await myTurn()
        let here = await inPlace()
        let typed: Bool = outputLock.withLock {
            guard here else {
                services.output.copyToClipboard(text)
                state.update { $0.outputMethod = "clipboard" }
                return false
            }
            services.output.type(text)
            state.update { $0.outputMethod = "typed" }
            return true
        }
        if typed {
            logOutputStats(text, provider: provider)
            startEditWatch(text)
        } else {
            Log.line("[Timing] Output: clipboard (window changed)")
            services.output.notify("Window changed - copied to clipboard")
        }
        history.add(text, destination: outputDestination())
    }

    func logOutputStats(_ text: String, provider: String) {
        let done = services.clock()
        let (start, finalizeStart, method) = state.read { ($0.startTime, $0.finalizeStart, $0.outputMethod) }
        let words = text.split(whereSeparator: \.isWhitespace).count
        let total = done - start, processing = done - finalizeStart
        let totalWpm = total > 0 ? Double(words) / total * 60 : 0
        let processingWpm = processing > 0 ? Double(words) / processing * 60 : 0
        Log.line("[Output] \(provider) via \(method) | \(String(format: "%.2f", total))s total | "
                 + "\(String(format: "%.2f", processing))s after release | \(words) words")
        Log.line("[Output] \"\(text)\"")
        Log.line("[WPM] Total: \(Int(totalWpm.rounded())) wpm (from key press) | Processing: \(Int(processingWpm.rounded())) wpm (from key release)")
    }

    /// Watch the destination field for the user's corrections to our output.
    func startEditWatch(_ text: String) {
        guard settings.editFeedbackEnabled, let watcher = services.editWatcher else { return }
        let (results, context) = state.read { ($0.allResults, $0.context) }
        let raw = results.sorted { $0.chunk < $1.chunk }.map(\.text)
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }.joined(separator: " ")
        let app = context?.appName ?? ""
        watcher.watch(typed: text, app: app) { [self] outcome, after, original, corrected in
            Log.$tag.withValue(tag) {
                if outcome == .edited {
                    Log.line("[Feedback] Edited within \(Int(after.rounded()))s")
                    Log.line("[Feedback]   typed:     \"\(original.prefix(120))\"")
                    Log.line("[Feedback]   corrected: \"\(corrected.prefix(120))\"")
                    let (method, llm) = state.read { ($0.outputMethod, $0.llmResult) }
                    // What produced the text the speaker fixed: a correction model's slip and
                    // a recognizer's are different lessons.
                    let model = llm != nil && ["streamed", "clipboard"].contains(method) ? llm!.model : ""
                    services.corpus?.record([
                        "ts": .number(services.clock()), "session_id": .string(sid),
                        "app": .string(context?.appName ?? ""), "window": .string(context?.windowTitle ?? ""),
                        "checked_after_s": .number(after), "raw_transcript": .string(String(raw.prefix(1000))),
                        "typed": .string(String(original.prefix(1000))), "corrected": .string(String(corrected.prefix(1000))),
                        "output_method": .string(method), "correction_model": .string(model),
                    ])
                }
                log("post_edit", ["outcome": .string(outcome.rawValue), "checked_after_s": .number(after), "app": .string(app)])
            }
        }
    }

    // MARK: - Teardown

    /// Release the session, whatever happened. Bookkeeping can't stop the release: a throw
    /// there once left the session held forever, and every later key press was rejected.
    func teardown() {
        let s = state.current
        log("session_complete", ["total_duration_ms": .number((services.clock() - s.startTime) * 1000),
                                 "chunks": JSON(s.chunkResults.count), "final_text": .string(String(s.finalText.prefix(500))),
                                 "words": JSON(s.finalText.split(whereSeparator: \.isWhitespace).count),
                                 "app": .string(s.context?.appName ?? "")])
        if settings.trainingEnabled, !s.finalText.isEmpty, !s.audio.isEmpty, let training = services.training {
            saveTrainingData(s, to: training)
        }
        if let ticket = s.ticket { outputOrder.done(ticket) }
        state.update { $0.isCapturing = false }
        onComplete?(self)
    }

    func saveTrainingData(_ s: State, to writer: TrainingWriter) {
        let results = s.allResults.sorted { $0.chunk < $1.chunk }
        var consensus: JSON = ["reached": false, "text": .null, "count": 0]
        for part in s.chunkResults.sorted(by: { $0.num < $1.num }) {
            if let agreed = part.consensus {
                consensus = ["reached": true, "text": .string(agreed), "count": JSON(Consensus.matching(part.results, agreed))]
                break
            }
        }
        var llm: JSON = .null
        if let r = s.llmResult {
            llm = ["provider": .string(r.provider), "model": .string(r.model), "generation_id": JSON(r.generationId),
                   "backend_provider": JSON(r.backendProvider), "resolved_model": JSON(r.resolvedModel),
                   "provider_order": JSON(r.providerOrder), "allow_fallbacks": JSON(r.allowFallbacks),
                   "reasoning_effort": .string(r.reasoningEffort), "usage": r.usage,
                   "input_text": .string(results.prefix(10).map(\.text).joined(separator: " | ")),
                   "output_text": .string(s.finalText), "latency_ms": .number(r.latencyMs)]
        }
        let encoder = JSONEncoder()
        let transcriptions = results.map { r -> JSON in
            (try? JSONDecoder().decode(JSON.self, from: encoder.encode(r))) ?? .null
        }
        let context: JSON = s.context.map { ["app_name": .string($0.appName), "window_title": .string($0.windowTitle),
                                             "bundle_id": .string($0.bundleId)] } ?? .null
        let metadata: JSON = [
            "session_id": .string(sid),
            "timestamp": .string(Self.isoTimestamp(Date())),
            "duration_ms": .number((services.clock() - s.startTime) * 1000),
            "sample_rate": JSON(settings.sampleRate),
            "schema_version": 1,
            "app_context": context,
            "config_snapshot": [
                "enabled_mics": JSON(settings.enabledMics), "enabled_providers": JSON(settings.enabledProviders),
                "consensus_threshold": JSON(settings.consensusThreshold),
                "openrouter_stt_models": JSON(settings.openrouterSttModels),
                "openrouter_correction_model": .string(settings.openrouterCorrectionModel),
                "openrouter_correction_provider_order": JSON(settings.openrouterCorrectionProviderOrder),
                "openrouter_correction_allow_fallbacks": .bool(settings.openrouterCorrectionAllowFallbacks),
                "openrouter_correction_reasoning_effort": .string(settings.openrouterCorrectionReasoningEffort),
                "custom_instructions_enabled": .bool(!settings.customInstructions.trimmingCharacters(in: .whitespaces).isEmpty),
                "system_prompt_customized": .bool(!settings.systemPrompt.trimmingCharacters(in: .whitespaces).isEmpty),
            ],
            "transcriptions": .array(transcriptions),
            "consensus": consensus,
            "llm_correction": llm,
            "final_output": .string(s.finalText),
            "output_method": .string(s.outputMethod.isEmpty ? "typed" : s.outputMethod),
        ]
        writer.save(sessionId: sid, audio: s.audio.map { ($0.mic, Array($0.blocks.joined())) }, metadata: metadata)
    }

    /// Local time, microseconds, no zone: what Python's datetime.now().isoformat() wrote.
    static func isoTimestamp(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        let micros = Int((date.timeIntervalSince1970 * 1_000_000).truncatingRemainder(dividingBy: 1_000_000))
        return formatter.string(from: date) + String(format: ".%06d", micros)
    }

    // MARK: - Helpers

    func log(_ event: String, _ fields: [String: JSON]) {
        var fields = fields
        fields["session_id"] = .string(sid)
        services.metrics?.log(event, fields)
    }

    /// Tell the observer what happened. It is display only.
    func notify(_ event: (SessionObserver) -> Void) {
        if let observer = services.observer { event(observer) }
    }
}

/// The streamed output of one correction: who owns it, what has been typed, what is held.
final class StreamState: @unchecked Sendable {
    private let lock = NSLock()
    private var winner: String?
    private var marker = NothingMarker()
    private var filter: DictationFilter
    private var separatorPending: Bool
    private var pieces: [String] = []
    private var held: [String] = []
    private var lastFocusCheck = ContinuousClock.now - .seconds(1)

    init(continuing: Bool) {
        filter = DictationFilter(continuing: continuing)
        separatorPending = continuing
    }

    /// Whoever types first owns the output; the loser's tokens are dropped.
    func claim(_ who: String) -> Bool {
        lock.withLock {
            if winner == nil { winner = who }
            return winner == who
        }
    }
    var claimed: Bool { lock.withLock { winner != nil } }

    /// The text to type for this token, or nil when there is nothing (yet).
    func prepare(_ token: String) -> String? {
        lock.withLock {
            var text = marker.feed(token)
            guard !text.isEmpty else { return nil }
            if separatorPending {
                // Only once there is something to type, so a dictation that comes to
                // nothing can't leave a stray space behind.
                text = " " + text
                separatorPending = false
            }
            let filtered = filter.feed(text)
            return filtered.isEmpty ? nil : filtered
        }
    }

    func flushMarker() -> String { lock.withLock { marker.flush() } }
    var calledOff: Bool { lock.withLock { marker.calledOff } }

    /// Once focus is lost it stays lost. Checked at most every 100ms.
    func stillInPlace(_ check: () -> Bool) -> Bool {
        let due: Bool? = lock.withLock {
            guard held.isEmpty else { return nil }
            let now = ContinuousClock.now
            guard now - lastFocusCheck >= .milliseconds(100) else { return false }
            lastFocusCheck = now
            return true
        }
        guard let due else { return false }
        return due ? check() : true
    }

    func hold(_ text: String) { lock.withLock { held.append(text) } }
    func typedPiece(_ text: String) { lock.withLock { pieces.append(text) } }
    var typed: String { lock.withLock { pieces.joined() } }
    var heldText: String { lock.withLock { held.joined() } }
}
