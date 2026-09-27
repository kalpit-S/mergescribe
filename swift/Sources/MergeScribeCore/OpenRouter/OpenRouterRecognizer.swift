import Foundation

/// One OpenRouter transcription model as a recognizer.
public struct OpenRouterRecognizer: Recognizer {
    public let model: String
    let key: String
    let client: TranscriptionClient
    let sampleRate: Int
    /// Words the speaker is known to use, read per request so it stays current.
    let keyterms: @Sendable () -> [String]

    public init(model: String, key: String, client: TranscriptionClient, sampleRate: Int = 16000,
                keyterms: @escaping @Sendable () -> [String] = { [] }) {
        self.model = model
        self.key = key
        self.client = client
        self.sampleRate = sampleRate
        self.keyterms = keyterms
    }

    public var name: String { TranscriptionClient.providerName(model) }
    public var singleInstance: Bool { false }

    /// What the model heard; an empty transcript when the request failed, as the rest of
    /// the pipeline expects of a recognizer that answered but heard nothing usable.
    public func transcribe(_ samples: [Float], mic: String) async throws -> TranscriptionResult {
        guard !key.isEmpty else { return TranscriptionResult(text: "", provider: name, mic: mic, latencyMs: 0) }
        let clock = ContinuousClock(), started = clock.now
        var text = ""
        do {
            let terms = TranscriptionClient.acceptsVocabulary(model) ? keyterms() : []
            text = try await client.transcribe(wav: WAV.encode(samples, sampleRate: sampleRate), model: model,
                                               key: key, keyterms: terms)
        } catch {
            // Left behind by the deadline: nobody is waiting for this answer.
            if !Task.isCancelled { Log.line("[\(name)] Error: \(error)") }
        }
        let latency = Int(((clock.now - started).seconds * 1000).rounded())
        return TranscriptionResult(text: text, provider: name, mic: mic, latencyMs: latency)
    }
}
