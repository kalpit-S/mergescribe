import Foundation

public enum MergeScribe {
    public static let version = "3.0.0-dev"
}

/// What one recognizer heard on one microphone.
public struct TranscriptionResult: Sendable, Equatable, Codable {
    public var text: String
    public var provider: String
    public var mic: String
    public var latencyMs: Int
    public var confidence: Double?
    /// Which part of the dictation, from 1; 0 when there is only the one.
    public var chunk: Int

    public init(text: String, provider: String, mic: String, latencyMs: Int,
                confidence: Double? = nil, chunk: Int = 0) {
        self.text = text
        self.provider = provider
        self.mic = mic
        self.latencyMs = latencyMs
        self.confidence = confidence
        self.chunk = chunk
    }

    enum CodingKeys: String, CodingKey {
        case text, provider, mic, confidence, chunk
        case latencyMs = "latency_ms"
    }

    // Written with every key present, null included, as the Python app's asdict() did.
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(text, forKey: .text)
        try c.encode(provider, forKey: .provider)
        try c.encode(mic, forKey: .mic)
        try c.encode(latencyMs, forKey: .latencyMs)
        try c.encode(confidence, forKey: .confidence)
        try c.encode(chunk, forKey: .chunk)
    }
}

/// The app and window a dictation is going to.
public struct AppContext: Sendable, Equatable, Codable {
    public var appName: String
    public var windowTitle: String
    public var bundleId: String

    public init(appName: String, windowTitle: String, bundleId: String) {
        self.appName = appName
        self.windowTitle = windowTitle
        self.bundleId = bundleId
    }

    enum CodingKeys: String, CodingKey {
        case appName = "app_name"
        case windowTitle = "window_title"
        case bundleId = "bundle_id"
    }
}

/// What the correction call did, for the log and the metrics.
public struct LLMCorrectionResult: Sendable, Equatable {
    public var text: String
    public var provider: String            // "openrouter"
    public var model: String
    public var inputTokensEst: Int
    public var latencyMs: Double
    public var streamed: Bool = false
    public var generationId: String?
    public var backendProvider: String?
    public var resolvedModel: String?
    public var providerOrder: [String] = []
    public var allowFallbacks: Bool?
    public var reasoningEffort: String = ""
    public var usage: JSON = .object([:])

    public init(text: String, provider: String, model: String, inputTokensEst: Int, latencyMs: Double) {
        self.text = text
        self.provider = provider
        self.model = model
        self.inputTokensEst = inputTokensEst
        self.latencyMs = latencyMs
    }
}
