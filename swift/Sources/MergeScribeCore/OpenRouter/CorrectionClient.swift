import Foundation

/// The correction's stream broke after some of it had already been typed.
///
/// `shown` is what reached the screen; `complete` is the whole correction if a second
/// attempt produced one that doesn't continue what was shown, else "".
public struct CorrectionInterrupted: Error, Equatable {
    public let shown: String
    public let complete: String
}

/// What OpenRouter said about how it served a correction.
public struct CorrectionMetadata: Sendable, Equatable {
    public var providerOrder: [String] = []
    public var allowFallbacks: Bool?
    public var reasoningEffort = ""
    public var resolvedModel: String?
    public var backendProvider: String?
    public var usage: JSON?
    public var generationId: String?
}

/// Models that refuse to switch reasoning off, learned from their refusal rather than kept
/// in a list that goes stale with every release.
public final class ReasoningMemory: @unchecked Sendable {
    public static let shared = ReasoningMemory()
    private let lock = NSLock()
    private var needsSome: Set<String> = []

    public init() {}

    func contains(_ model: String) -> Bool { lock.withLock { needsSome.contains(model) } }
    func insert(_ model: String) { lock.withLock { _ = needsSome.insert(model) } }
}

/// Correction and editing through OpenRouter's chat completions.
public final class CorrectionClient: Sendable {
    public static let defaultModel = "openai/gpt-6-luna"
    /// Abort a stream that stops producing content while the connection stays open. Seen:
    /// a first token at 0.71s, then a 19s stall dribbling text into the user's document.
    let stallSeconds: Double
    static let requestTimeout: TimeInterval = 15

    let http: HTTPClient
    let reasoning: ReasoningMemory
    let vocabulary: VocabularyStore?

    public init(http: HTTPClient, reasoning: ReasoningMemory = .shared, vocabulary: VocabularyStore? = nil,
                stallSeconds: Double = 8) {
        self.http = http
        self.reasoning = reasoning
        self.vocabulary = vocabulary
        self.stallSeconds = stallSeconds
    }

    // MARK: - Correcting a dictation

    public typealias Delta = @Sendable (String) -> Void

    /// Correct the transcripts. Returns "" when the call fails after a retry; the caller
    /// falls back to the raw transcript. Throws CorrectionInterrupted when a stream broke
    /// after typing began and the rest can't be joined on cleanly.
    public func correct(_ results: [TranscriptionResult], context: AppContext?, settings: Settings,
                        onDelta: Delta? = nil,
                        onMetadata: (@Sendable (LLMCorrectionResult) -> Void)? = nil,
                        onGenerationMetadata: (@Sendable (String, JSON) -> Void)? = nil) async throws -> String {
        let results = results.filter { !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        guard !results.isEmpty else { return "" }
        guard !settings.openrouterApiKey.isEmpty else {
            Log.line("[LLM] No OpenRouter API key configured")
            return ""
        }
        let prompt = CorrectionPrompt.user(results: results, context: context)
        let system = systemPrompt(settings)
        let totalWords = results.map { $0.text.split(whereSeparator: \.isWhitespace).count }.max() ?? 0
        let estTokens = (prompt.unicodeScalars.count + system.unicodeScalars.count) / 4
        let model = settings.openrouterCorrectionModel

        let clock = ContinuousClock(), start = clock.now
        let progress = Progress()

        let timingDelta: Delta = { token in
            progress.firstToken(at: clock.now)
            if let onDelta {
                progress.show(token)
                onDelta(token)
            }
        }

        var result = ""
        var metadata = CorrectionMetadata()
        for attempt in 0..<2 {
            let outcome = await stream(prompt: prompt, system: system, settings: settings, onDelta: timingDelta,
                                       onGenerationMetadata: onGenerationMetadata)
            metadata = outcome.metadata
            if outcome.complete && !outcome.text.isEmpty {
                result = outcome.text
                break
            }
            let shown = progress.shown
            if !shown.isEmpty {
                // Part of it is already typed: streaming again would type it twice.
                result = try await finishWithoutStreaming(shown: shown, prompt: prompt, system: system,
                                                          settings: settings, onDelta: onDelta)
                break
            }
            result = ""
            if attempt == 0 { Log.line("[LLM] Retrying OpenRouter") }
        }
        guard !result.isEmpty else {
            Log.line("[LLM] Correction failed")
            return ""
        }

        let elapsed = (clock.now - start).seconds
        let ttft = progress.firstTokenAt.map { ($0 - start).seconds } ?? elapsed
        // Say what actually served it, not what was asked for: ":nitro" and the provider
        // preferences mean the two can differ.
        let served = metadata.resolvedModel ?? model
        let provider = metadata.backendProvider.map { " @\($0)" } ?? ""
        let asked = served != model ? " (asked \(model))" : ""
        Log.line("[LLM] \(served)\(provider)\(asked) | in: \(totalWords) words, ~\(estTokens) tok prompt | "
                 + "TTFT \(String(format: "%.2f", ttft))s | total \(String(format: "%.2f", elapsed))s")

        var llm = LLMCorrectionResult(text: result, provider: "openrouter", model: model,
                                      inputTokensEst: estTokens, latencyMs: elapsed * 1000)
        llm.streamed = onDelta != nil
        llm.generationId = metadata.generationId
        llm.backendProvider = metadata.backendProvider
        llm.resolvedModel = metadata.resolvedModel
        llm.providerOrder = metadata.providerOrder
        llm.allowFallbacks = metadata.allowFallbacks
        llm.reasoningEffort = metadata.reasoningEffort
        llm.usage = metadata.usage ?? .object([:])
        onMetadata?(llm)
        return result
    }

    /// The system prompt, with what vocabulary learning knows when it is on.
    public func systemPrompt(_ settings: Settings) -> String {
        let learn = settings.learnVocabulary && vocabulary != nil
        return CorrectionPrompt.system(settings: settings,
                                       learned: learn ? vocabulary!.learnedCorrections() : [],
                                       supplied: learn ? vocabulary!.suppliedTerms() : [])
    }

    /// Recover a correction whose stream broke after typing began: ask again without
    /// streaming, and type only the rest if the new answer continues what is on screen.
    func finishWithoutStreaming(shown: String, prompt: String, system: String, settings: Settings,
                                onDelta: Delta?) async throws -> String {
        Log.line("[LLM] Stream broke after typing began; finishing it without streaming")
        let outcome = await stream(prompt: prompt, system: system, settings: settings)
        if outcome.complete && outcome.text.hasPrefix(shown) {
            let rest = String(outcome.text.dropFirst(shown.count))
            if let onDelta, !rest.isEmpty { onDelta(rest) }
            return outcome.text
        }
        throw CorrectionInterrupted(shown: shown, complete: outcome.complete ? outcome.text : "")
    }

    // MARK: - Editing selected text

    /// Apply a spoken instruction to the selected text. The original, unchanged, if the call fails.
    public func edit(selected: String, command: String, settings: Settings) async -> String {
        let prompt = CorrectionPrompt.editing(selected: selected, command: command)
        let system = settings.editingPrompt.isEmpty ? CorrectionPrompt.defaultEditing : settings.editingPrompt
        for _ in 0..<2 {
            let outcome = await stream(prompt: prompt, system: system, settings: settings)
            if outcome.complete && !outcome.text.isEmpty { return outcome.text }
        }
        return selected
    }

    // MARK: - The request

    public struct Outcome: Sendable {
        public var text: String
        /// False when the stream broke, stalled or reported an error part way: text is then
        /// only what arrived.
        public var complete: Bool
        public var metadata: CorrectionMetadata
    }

    /// One streamed chat completion.
    public func stream(prompt: String, system: String, settings: Settings, onDelta: Delta? = nil,
                       onGenerationMetadata: (@Sendable (String, JSON) -> Void)? = nil) async -> Outcome {
        let key = settings.openrouterApiKey
        var metadata = CorrectionMetadata()
        guard !key.isEmpty else { return Outcome(text: "", complete: false, metadata: metadata) }

        let model = settings.openrouterCorrectionModel
        let order = settings.openrouterCorrectionProviderOrder.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        let allowFallbacks = settings.openrouterCorrectionAllowFallbacks
        // ":nitro" and friends are routing hints on the same underlying model.
        let baseModel = String(model.split(separator: ":", maxSplits: 1).first ?? "")
        // Correction is a rewrite, and thinking only adds latency (a Gemini that thinks by
        // default measured 11.9s against 1.6s with it off), so reasoning is off unless chosen.
        var effort = settings.openrouterCorrectionReasoningEffort.trimmingCharacters(in: .whitespaces).lowercased()
        if effort.isEmpty { effort = "none" }
        if effort == "none" && reasoning.contains(baseModel) { effort = "minimal" }
        metadata.providerOrder = order
        metadata.allowFallbacks = order.isEmpty ? nil : allowFallbacks
        metadata.reasoningEffort = effort

        var body: [String: JSON] = [
            "model": .string(model),
            "messages": [["role": "system", "content": .string(system)], ["role": "user", "content": .string(prompt)]],
            "temperature": 0.5,
            "max_tokens": 2000,
            "stream": true,
            "reasoning": ["effort": .string(effort)],
        ]
        if !order.isEmpty {
            body["provider"] = ["order": JSON(order), "allow_fallbacks": .bool(allowFallbacks)]
        }
        let request = OpenRouterAPI.request(OpenRouterAPI.base.appendingPathComponent("chat/completions"),
                                            key: key, json: .object(body), timeout: Self.requestTimeout)

        let collected = Collected()
        do {
            let (response, lines) = try await http.lines(for: request)
            if response.statusCode != 200 {
                let detail = String(await lines.collect().prefix(300))
                if response.statusCode == 400 && effort == "none" && detail.lowercased().contains("reason") {
                    reasoning.insert(baseModel)
                    Log.line("[LLM] \(baseModel) won't switch reasoning off; using minimal")
                    return await stream(prompt: prompt, system: system, settings: settings, onDelta: onDelta,
                                        onGenerationMetadata: onGenerationMetadata)
                }
                Log.line("[LLM] OpenRouter API error: \(response.statusCode) \(detail.prefix(120))")
                return Outcome(text: "", complete: false, metadata: metadata)
            }

            collected.touch()      // the stall clock starts when the reply does
            let complete = try await readStream(lines, collected: collected, metadata: &metadata, onDelta: onDelta)
            if let id = collected.generationId {
                metadata.generationId = id
                if metadata.backendProvider == nil, !order.isEmpty, !allowFallbacks {
                    metadata.backendProvider = order[0]
                }
                if metadata.resolvedModel == nil { metadata.resolvedModel = model }
                // Usage and cost live behind a second request. Fetching it inline delayed the
                // end of the session by up to ~4.5s after the text was typed.
                if let onGenerationMetadata {
                    let http = self.http
                    Task.detached {
                        if let data = await Self.generationMetadata(http: http, key: key, id: id) {
                            onGenerationMetadata(id, data)
                        }
                    }
                }
            }
            return Outcome(text: collected.text, complete: complete, metadata: metadata)
        } catch {
            if !(error is CancellationError) { Log.line("[LLM] OpenRouter error: \(error.localizedDescription)") }
            return Outcome(text: collected.text, complete: false, metadata: metadata)
        }
    }

    /// Read the server-sent events until [DONE], an error, a stall, or the end. True when
    /// the reply arrived whole.
    private func readStream(_ lines: AsyncThrowingStream<String, Error>, collected: Collected,
                            metadata: inout CorrectionMetadata, onDelta: Delta?) async throws -> Bool {
        // The watchdog cancels the read when content stops arriving; the stream then ends
        // with a cancellation, which is reported as incomplete.
        let reader = Task { () -> (Bool, CorrectionMetadata) in
            var meta = CorrectionMetadata()
            for try await raw in lines {
                let line = raw.trimmingCharacters(in: .whitespaces)
                guard line.hasPrefix("data: ") else { continue }     // blank, ":" keep-alive, other fields
                let payload = String(line.dropFirst(6))
                if payload == "[DONE]" { return (true, meta) }
                guard let parsed = try? JSONDecoder().decode(JSON.self, from: Data(payload.utf8)) else { continue }
                if let id = parsed["id"]?.string, !id.isEmpty { collected.generationId = id }
                // OpenRouter names the model and provider it routed to on each chunk.
                if let model = parsed["model"]?.string, !model.isEmpty { meta.resolvedModel = model }
                if let provider = parsed["provider"]?.string, !provider.isEmpty { meta.backendProvider = provider }
                if let error = parsed["error"] {
                    Log.line("[LLM] OpenRouter stream error: \(error.line())")
                    return (false, meta)
                }
                if let usage = parsed["usage"], usage.object?.isEmpty == false { meta.usage = usage }
                if let content = parsed["choices"]?.array?.first?["delta"]?["content"]?.string, !content.isEmpty {
                    collected.append(content)
                    onDelta?(content)
                }
            }
            return (true, meta)
        }
        let stallSeconds = self.stallSeconds
        let watchdog = Task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(min(250, Int(stallSeconds * 250))))
                if collected.secondsSinceContent > stallSeconds {
                    Log.line("[LLM] Stream stalled >\(Int(stallSeconds))s")
                    collected.stalled = true
                    reader.cancel()
                    return
                }
            }
        }
        defer { watchdog.cancel() }
        do {
            let (complete, meta) = try await reader.value
            metadata.resolvedModel = meta.resolvedModel ?? metadata.resolvedModel
            metadata.backendProvider = meta.backendProvider ?? metadata.backendProvider
            metadata.usage = meta.usage ?? metadata.usage
            return complete && !collected.stalled
        } catch {
            if collected.stalled || error is CancellationError { return false }
            throw error
        }
    }

    /// Usage and provider details for a finished generation, or nil.
    static func generationMetadata(http: HTTPClient, key: String, id: String) async -> JSON? {
        var components = URLComponents(url: OpenRouterAPI.base.appendingPathComponent("generation"),
                                       resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "id", value: id)]
        let request = OpenRouterAPI.request(components.url!, key: key, timeout: 2)
        for attempt in 0..<2 {
            if let (data, response) = try? await http.data(for: request), response.statusCode == 200,
               let payload = (try? JSONDecoder().decode(JSON.self, from: data))?["data"] {
                var usage: [String: JSON] = [:]
                for field in ["tokens_prompt", "tokens_completion", "native_tokens_prompt", "native_tokens_completion",
                              "native_tokens_cached", "native_tokens_reasoning", "num_media_prompt",
                              "num_media_completion", "total_cost", "upstream_inference_cost", "latency",
                              "generation_time", "finish_reason", "native_finish_reason", "router", "is_byok",
                              "request_id"] {
                    if let value = payload[field], !value.isNull { usage[field] = value }
                }
                return ["backend_provider": payload["provider_name"] ?? .null,
                        "resolved_model": payload["model"] ?? .null,
                        "usage": .object(usage)]
            }
            if attempt == 0 { try? await Task.sleep(for: .milliseconds(250)) }
        }
        return nil
    }
}

/// What a correction has shown so far, shared by its delta callback and the retry logic.
private final class Progress: @unchecked Sendable {
    private let lock = NSLock()
    private var tokens: [String] = []
    private(set) var firstTokenAt: ContinuousClock.Instant?

    func firstToken(at instant: ContinuousClock.Instant) {
        lock.withLock { if firstTokenAt == nil { firstTokenAt = instant } }
    }
    func show(_ token: String) { lock.withLock { tokens.append(token) } }
    var shown: String { lock.withLock { tokens.joined() } }
}

/// The text a stream has produced, and when content last arrived.
private final class Collected: @unchecked Sendable {
    private let lock = NSLock()
    private var chunks: [String] = []
    private var lastContent = ContinuousClock.now
    private var _generationId: String?
    private var _stalled = false

    func append(_ content: String) {
        lock.withLock { chunks.append(content); lastContent = .now }
    }
    func touch() { lock.withLock { lastContent = .now } }
    var text: String { lock.withLock { chunks.joined() } }
    var secondsSinceContent: Double { lock.withLock { (ContinuousClock.now - lastContent).seconds } }
    var generationId: String? {
        get { lock.withLock { _generationId } }
        set { lock.withLock { _generationId = newValue } }
    }
    var stalled: Bool {
        get { lock.withLock { _stalled } }
        set { lock.withLock { _stalled = newValue } }
    }
}

extension Duration {
    public var seconds: Double {
        Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}
