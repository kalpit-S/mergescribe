import Foundation

/// The models Settings offers, and the OpenRouter calls it makes.
public enum Catalogue {
    /// Current models only; anything else can be added from OpenRouter's catalogue or typed
    /// in as a custom id. Every entry was checked to work with this app (Sep 2026).
    public static let transcriptionModels: [(id: String, name: String)] = [
        ("microsoft/mai-transcribe-2", "MAI-Transcribe 2"),
        ("assemblyai/universal-3-5-pro", "Universal 3.5 Pro"),
        ("fish-audio/transcribe-1-pro", "Fish Audio Transcribe 1 Pro"),
        ("openai/gpt-transcribe", "GPT Transcribe"),
    ]

    /// Fast enough for dictation: first token within about 1.5s.
    public static let correctionModels: [(id: String, name: String)] = [
        ("deepseek/deepseek-v4.1-flash", "DeepSeek V4.1 Flash"),
        ("openai/gpt-6-luna", "GPT-6 Luna"),
        ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5"),
        ("moonshotai/kimi-k3", "Kimi K3"),
        ("google/gemini-3.5-flash-lite", "Gemini 3.5 Flash Lite"),
        ("google/gemini-3.8-flash", "Gemini 3.8 Flash"),
    ]

    /// Model ids from text separated by commas, spaces or new lines; blanks and repeats dropped.
    public static func parseModelIds(_ text: String) -> [String] {
        var seen = Set<String>()
        return text.split(whereSeparator: { $0 == "," || $0.isWhitespace })
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && seen.insert($0).inserted }
    }

    /// One line saying where corrections go.
    public static func routingStatus(key: String, model: String, providerOrder: [String], effort: String) -> String {
        guard !key.isEmpty else { return "No OpenRouter key — transcriptions won't be corrected" }
        let model = model.isEmpty ? CorrectionClient.defaultModel : model
        let label = correctionModels.first { $0.id == model }?.name ?? model
        var suffix = ""
        if !providerOrder.isEmpty { suffix += " via \(providerOrder.joined(separator: ", "))" }
        if !effort.isEmpty { suffix += " (\(effort) reasoning)" }
        return "Correction → OpenRouter: \(label)\(suffix)"
    }

    /// OpenRouter's speech-to-text models as (id, name), newest first; [] when unreachable.
    public static func fetchTranscriptionModels(http: HTTPClient, key: String) async -> [(id: String, name: String)] {
        var components = URLComponents(url: OpenRouterAPI.base.appendingPathComponent("models"), resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "output_modalities", value: "transcription")]
        var request = URLRequest(url: components.url!, timeoutInterval: 5)
        if !key.isEmpty { request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization") }
        guard let (data, _) = try? await http.data(for: request),
              let models = (try? JSONDecoder().decode(JSON.self, from: data))?["data"]?.array else { return [] }
        return models.compactMap { model in
            guard let id = model["id"]?.string, !id.isEmpty else { return nil }
            // "Microsoft AI: MAI-Transcribe 2" -> "MAI-Transcribe 2"
            let name = model["name"]?.string.flatMap { $0.isEmpty ? nil : $0 } ?? id
            return (id, name.components(separatedBy: ": ").dropFirst().joined(separator: ": ").nilIfEmpty ?? name)
        }
    }

    public struct KeyCheck: Sendable, Equatable {
        public var valid: Bool
        public var latencyMs: Int?
        public var error: String?
    }

    /// Whether the key works, with how long OpenRouter took to say so.
    public static func validateKey(http: HTTPClient, key: String) async -> KeyCheck {
        guard key.count >= 10 else { return KeyCheck(valid: false, error: "Key too short") }
        var request = URLRequest(url: OpenRouterAPI.base.appendingPathComponent("models"), timeoutInterval: 10)
        request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        let clock = ContinuousClock(), started = clock.now
        do {
            let (_, response) = try await http.data(for: request)
            let latency = Int(((clock.now - started).seconds * 1000).rounded())
            switch response.statusCode {
            case 200: return KeyCheck(valid: true, latencyMs: latency)
            case 401: return KeyCheck(valid: false, error: "Invalid key")
            default: return KeyCheck(valid: false, error: "HTTP \(response.statusCode)")
            }
        } catch let error as URLError where error.code == .timedOut {
            return KeyCheck(valid: false, error: "Timeout")
        } catch {
            return KeyCheck(valid: false, error: String(error.localizedDescription.prefix(50)))
        }
    }
}

extension String {
    var nilIfEmpty: String? { isEmpty ? nil : self }
}
