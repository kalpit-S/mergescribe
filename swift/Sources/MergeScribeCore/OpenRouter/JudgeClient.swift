import Foundation

/// Jev, asked through OpenRouter's decisions API.
public final class JudgeClient: Sendable {
    let http: HTTPClient

    public init(http: HTTPClient) { self.http = http }

    /// Whether these transcripts can be typed as they are. nil when the judge is off,
    /// unreachable, or too slow to matter: each just means the correction model's answer
    /// is used, so the caller has nothing to handle.
    public func judge(_ results: [TranscriptionResult], settings: Settings, context: AppContext?) async -> Judge.Verdict? {
        guard settings.judgeEnabled, !settings.openrouterApiKey.isEmpty else { return nil }
        let candidates = Judge.candidates(results)
        guard !candidates.isEmpty else { return nil }
        let body: JSON = [
            "model": .string(settings.judgeModel.isEmpty ? Judge.defaultModel : settings.judgeModel),
            "state": ["transcripts": JSON(candidates)],
            "questions": Judge.questions(count: candidates.count, preferences: Judge.preferences(settings), context: context),
        ]
        let timeout = max(0.1, Double(settings.judgeTimeoutMs) / 1000)
        let request = OpenRouterAPI.request(Judge.decisionsURL, key: settings.openrouterApiKey, json: body, timeout: timeout)
        let clock = ContinuousClock(), started = clock.now
        let payload: JSON
        do {
            let (data, response) = try await http.data(for: request)
            guard response.statusCode == 200 else { throw HTTPError(status: response.statusCode, body: "") }
            payload = try JSONDecoder().decode(JSON.self, from: data)
        } catch {
            Log.line("[Judge] unavailable (\(error)); using the correction model")
            return nil
        }
        do {
            guard let answers = payload["answers"] else { throw Judge.DecideError.missingAnswer("answers") }
            let (clean, pick, scores) = try Judge.decide(answers, count: candidates.count)
            return Judge.Verdict(clean: clean, text: candidates[pick], scores: scores,
                                 model: payload["model"]?.string ?? "",
                                 latencyMs: Int(((clock.now - started).seconds * 1000).rounded()))
        } catch {
            Log.line("[Judge] unexpected reply (\(error)); using the correction model")
            return nil
        }
    }
}
