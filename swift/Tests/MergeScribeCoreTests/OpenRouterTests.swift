import Foundation
import Testing
@testable import MergeScribeCore

private func settings(model: String = "google/gemini-3.8-flash", effort: String = "") -> Settings {
    var s = Settings()
    s.openrouterApiKey = "sk-or-test"
    s.openrouterCorrectionModel = model
    s.openrouterCorrectionReasoningEffort = effort
    s.learnVocabulary = false
    return s
}

private let done = "data: [DONE]"
private func results(_ texts: String...) -> [TranscriptionResult] {
    texts.enumerated().map { TranscriptionResult(text: $1, provider: "p\($0)", mic: "m", latencyMs: 1) }
}

/// Everything a correction typed, in order.
private final class Typed: @unchecked Sendable {
    private let lock = NSLock()
    private var tokens: [String] = []
    func append(_ t: String) { lock.withLock { tokens.append(t) } }
    var text: String { lock.withLock { tokens.joined() } }
}

@Suite struct ReasoningTests {
    // Correction asks for reasoning off, or as little as the model allows.
    @Test func reasoningIsOffUnlessChosen() async {
        let http = FakeHTTP([.lines(200, [sse("Ship it."), done])])
        let outcome = await CorrectionClient(http: http, reasoning: ReasoningMemory()).stream(prompt: "p", system: "s", settings: settings())
        #expect(outcome.text == "Ship it." && outcome.complete)
        #expect(http.bodies.map { $0["reasoning"]!["effort"]!.string! } == ["none"])
    }

    @Test func aModelThatRefusesOffGetsMinimalAndItIsRemembered() async {
        let refused = FakeHTTP.Reply.data(400, #"{"error":{"message":"Reasoning is mandatory for this endpoint"}}"#)
        let http = FakeHTTP([refused, .lines(200, [sse("Ship it."), done]), .lines(200, [sse("Ship it."), done])])
        let client = CorrectionClient(http: http, reasoning: ReasoningMemory())
        #expect(await client.stream(prompt: "p", system: "s", settings: settings()).text == "Ship it.")
        #expect(await client.stream(prompt: "p", system: "s", settings: settings(model: "google/gemini-3.8-flash:nitro")).text == "Ship it.")
        #expect(http.bodies.map { $0["reasoning"]!["effort"]!.string! } == ["none", "minimal", "minimal"])
    }

    @Test func aChosenEffortIsSentAsChosen() async {
        let http = FakeHTTP([.lines(200, [sse("Ship it."), done])])
        _ = await CorrectionClient(http: http, reasoning: ReasoningMemory()).stream(prompt: "p", system: "s", settings: settings(effort: "low"))
        #expect(http.bodies.map { $0["reasoning"]!["effort"]!.string! } == ["low"])
    }

    @Test func otherErrorsAreNotMistakenForARefusal() async {
        let http = FakeHTTP([.data(400, #"{"error":{"message":"invalid model"}}"#)])
        let outcome = await CorrectionClient(http: http, reasoning: ReasoningMemory()).stream(prompt: "p", system: "s", settings: settings())
        #expect(outcome.text == "" && !outcome.complete)
        #expect(http.sent.count == 1)
    }

    @Test func theRequestCarriesTheRoutingAndTheMessages() async {
        var s = settings(model: "openai/gpt-6-luna")
        s.openrouterCorrectionProviderOrder = ["openai/fast"]
        s.openrouterCorrectionAllowFallbacks = false
        let http = FakeHTTP([.lines(200, [sse("ok"), done])])
        let outcome = await CorrectionClient(http: http, reasoning: ReasoningMemory()).stream(prompt: "user text", system: "system text", settings: s)
        let body = http.bodies[0]
        #expect(body["provider"] == ["order": ["openai/fast"], "allow_fallbacks": false])
        #expect(body["messages"]!.array!.map { $0["content"]!.string! } == ["system text", "user text"])
        #expect(body["stream"] == true && body["temperature"] == 0.5 && body["max_tokens"] == 2000)
        #expect(http.sent[0].value(forHTTPHeaderField: "Authorization") == "Bearer sk-or-test")
        #expect(outcome.metadata.resolvedModel == "openai/gpt-6-luna" && outcome.metadata.backendProvider == "OpenAI")
        #expect(outcome.metadata.generationId == "gen-1")
    }
}

@Suite struct BrokenStreamTests {
    // Tokens are typed as they arrive, so a retry must never type them again.
    @Test func theRestIsTypedWhenASecondAnswerContinuesTheFirst() async throws {
        let http = FakeHTTP([.lines(200, [sse("Ship "), sse("it")], fail: URLError(.networkConnectionLost)),
                             .lines(200, [sse("Ship it Friday."), done])])
        let typed = Typed()
        let result = try await CorrectionClient(http: http, reasoning: ReasoningMemory())
            .correct(results("ship it friday"), context: nil, settings: settings(), onDelta: typed.append)
        #expect(typed.text == "Ship it Friday." && result == "Ship it Friday.")
    }

    @Test func anAnswerThatDoesNotContinueItIsHandedOverWhole() async {
        let http = FakeHTTP([.lines(200, [sse("Ship "), sse("it")], fail: URLError(.networkConnectionLost)),
                             .lines(200, [sse("We ship Friday."), done])])
        let typed = Typed()
        await #expect(throws: CorrectionInterrupted(shown: "Ship it", complete: "We ship Friday.")) {
            try await CorrectionClient(http: http, reasoning: ReasoningMemory())
                .correct(results("ship it friday"), context: nil, settings: settings(), onDelta: typed.append)
        }
        #expect(typed.text == "Ship it")
    }

    @Test func aFailureBeforeAnythingWasTypedJustRetries() async throws {
        let http = FakeHTTP([.error(URLError(.timedOut)), .lines(200, [sse("Ship it Friday."), done])])
        let typed = Typed()
        let result = try await CorrectionClient(http: http, reasoning: ReasoningMemory())
            .correct(results("ship it friday"), context: nil, settings: settings(), onDelta: typed.append)
        #expect(typed.text == "Ship it Friday." && result == "Ship it Friday.")
        #expect(http.bodies.allSatisfy { $0["stream"] == true })
    }

    @Test func twoFailuresMeanTheRawTranscript() async throws {
        let http = FakeHTTP([.error(URLError(.timedOut)), .error(URLError(.timedOut))])
        #expect(try await CorrectionClient(http: http, reasoning: ReasoningMemory())
            .correct(results("ship it"), context: nil, settings: settings()) == "")
    }

    @Test func aStalledStreamIsCutOffAsIncomplete() async {
        let http = FakeHTTP([.lines(200, [sse("Ship ")], hang: true)])
        let started = ContinuousClock.now
        let outcome = await CorrectionClient(http: http, reasoning: ReasoningMemory(), stallSeconds: 0.3)
            .stream(prompt: "p", system: "s", settings: settings())
        #expect(!outcome.complete && outcome.text == "Ship ")
        #expect((ContinuousClock.now - started).seconds < 3)
    }

    @Test func anErrorEventInTheStreamIsIncomplete() async {
        let http = FakeHTTP([.lines(200, [sse("Ship "), #"data: {"error": {"message": "overloaded"}}"#, done])])
        let outcome = await CorrectionClient(http: http, reasoning: ReasoningMemory()).stream(prompt: "p", system: "s", settings: settings())
        #expect(!outcome.complete)
    }

    @Test func theMetadataReportsWhatServedIt() async throws {
        let http = FakeHTTP([.lines(200, [sse("Ship it."), done])])
        let box = Box<LLMCorrectionResult>()
        _ = try await CorrectionClient(http: http, reasoning: ReasoningMemory())
            .correct(results("ship it"), context: nil, settings: settings(), onDelta: { _ in }, onMetadata: { box.value = $0 })
        #expect(box.value?.model == "google/gemini-3.8-flash")
        #expect(box.value?.resolvedModel == "openai/gpt-6-luna")
        #expect(box.value?.streamed == true && box.value?.reasoningEffort == "none")
    }

    @Test func editingFallsBackToTheOriginal() async {
        let http = FakeHTTP([.error(URLError(.timedOut)), .error(URLError(.timedOut))])
        #expect(await CorrectionClient(http: http, reasoning: ReasoningMemory())
            .edit(selected: "Original text", command: "make it formal", settings: settings()) == "Original text")
    }
}

@Suite struct TranscriptionClientTests {
    @Test func assemblyAIIsPrimedWithTheKnownTerms() async throws {
        let http = FakeHTTP([.data(200, #"{"text": "ok"}"#)])
        _ = try await TranscriptionClient(http: http).transcribe(wav: Data([1, 2]), model: "assemblyai/universal-3-5-pro",
                                                                 key: "k", keyterms: ["Claude", "Postgres"])
        #expect(http.bodies[0]["provider"] == ["options": ["assemblyai": ["keyterms_prompt": ["Claude", "Postgres"]]]])
        #expect(http.bodies[0]["input_audio"]!["format"] == "wav")
        #expect(http.bodies[0]["input_audio"]!["data"]!.string == Data([1, 2]).base64EncodedString())
    }

    @Test func otherModelsAndNoTermsGetNoProviderOptions() async throws {
        let http = FakeHTTP([.data(200, #"{"text": "ok"}"#), .data(200, #"{"text": "ok"}"#)])
        let client = TranscriptionClient(http: http)
        _ = try await client.transcribe(wav: Data(), model: "microsoft/mai-transcribe-2", key: "k", keyterms: ["Claude"])
        _ = try await client.transcribe(wav: Data(), model: "assemblyai/universal-3-5-pro", key: "k", keyterms: [])
        #expect(http.bodies.allSatisfy { $0["provider"] == nil })
    }

    @Test func aTransportErrorIsRetriedOnAFreshConnection() async throws {
        let first = FakeHTTP([.error(URLError(.secureConnectionFailed))])
        let fresh = FakeHTTP([.data(200, #"{"text": "<|speaker:0|> ok [pause] then"}"#)])
        let text = try await TranscriptionClient(http: first, freshHTTP: { fresh })
            .transcribe(wav: Data(), model: "microsoft/mai-transcribe-2", key: "k")
        #expect(text == "ok then")
        #expect(first.sent.count == 1 && fresh.sent.count == 1)
    }

    @Test func anHTTPErrorIsNotRetried() async {
        let http = FakeHTTP([.data(402, "payment required"), .data(200, #"{"text": "ok"}"#)])
        await #expect(throws: HTTPError.self) {
            try await TranscriptionClient(http: http).transcribe(wav: Data(), model: "x/y", key: "k")
        }
        #expect(http.sent.count == 1)
    }

    @Test func providerNamesMatchThePythonOnes() {
        #expect(TranscriptionClient.providerName("assemblyai/universal-3-5-pro") == "or-assemblyai-universal-3-5-pro")
        #expect(TranscriptionClient.providerName("openai/gpt-4o-transcribe:free") == "or-openai-gpt-4o-transcribe-free")
    }

    @Test func wavRoundTrips() {
        let samples: [Float] = (0..<1600).map { sin(Float($0) * 0.05) * 0.5 } + [1.5, -1.5]
        let data = WAV.encode(samples, sampleRate: 16000)
        #expect(data.prefix(4) == Data("RIFF".utf8) && data.count == 44 + samples.count * 2)
        let (back, rate) = WAV.decode(data)!
        #expect(rate == 16000 && back.count == samples.count)
        for (a, b) in zip(samples.prefix(1600), back) { #expect(abs(a - b) < 1e-4) }
        #expect(back.suffix(2).map { $0 > 0.99 || $0 < -0.99 } == [true, true], "clipped, not wrapped")
    }
}

@Suite struct JudgeClientTests {
    private func judgeSettings(_ change: (inout Settings) -> Void = { _ in }) -> Settings {
        var s = settings()
        s.judgeEnabled = true
        change(&s)
        return s
    }

    private func reply(filler: Double = 0.05, prefs: Double? = nil, pick: String = "0", count: Int = 1) -> FakeHTTP.Reply {
        var probabilities: [String: JSON] = ["none": 0.05]
        for i in 0..<count { probabilities[String(i)] = .number(String(i) == pick ? 0.9 : 0.02) }
        var answers: [String: JSON] = ["best": ["choice": .string(pick), "probabilities": .object(probabilities)]]
        for i in 0..<count {
            for job in ["filler", "restart", "misheard", "punct"] { answers["\(job)_\(i)"] = ["noul": .number(job == "filler" ? filler : 0.05)] }
            if let prefs { answers["prefs_\(i)"] = ["noul": .number(prefs)] }
        }
        return .data(200, JSON.object(["model": "typesafe/jev-1.13", "answers": .object(answers)]).line())
    }

    @Test func offOrKeylessMeansNoCall() async {
        let http = FakeHTTP([])
        #expect(await JudgeClient(http: http).judge(results("hi"), settings: settings(), context: nil) == nil)
        #expect(await JudgeClient(http: http).judge(results("hi"), settings: judgeSettings { $0.openrouterApiKey = "" }, context: nil) == nil)
        #expect(http.sent.isEmpty)
        #expect(Settings().judgeEnabled == false, "off by default")
    }

    @Test func failuresAreSilent() async {
        #expect(await JudgeClient(http: FakeHTTP([.error(URLError(.timedOut))])).judge(results("hi"), settings: judgeSettings(), context: nil) == nil)
        #expect(await JudgeClient(http: FakeHTTP([.data(200, #"{"answers": {"filler": {}}}"#)])).judge(results("hi"), settings: judgeSettings(), context: nil) == nil)
    }

    @Test func aCleanTranscriptIsClearedForTyping() async {
        let verdict = await JudgeClient(http: FakeHTTP([reply()])).judge(results("Ship it Friday."), settings: judgeSettings(), context: nil)
        #expect(verdict?.clean == true && verdict?.text == "Ship it Friday." && verdict?.model == "typesafe/jev-1.13")
    }

    @Test func fillerOrABrokenPreferenceSendsItToTheCorrectionModel() async {
        #expect(await JudgeClient(http: FakeHTTP([reply(filler: 0.9)])).judge(results("um, ship it"), settings: judgeSettings(), context: nil)?.clean == false)
        let prefs = judgeSettings { $0.customInstructions = "In Slack, lowercase." }
        #expect(await JudgeClient(http: FakeHTTP([reply(prefs: 0.9)])).judge(results("Ship it."), settings: prefs, context: nil)?.clean == false)
    }

    @Test func itTypesTheTranscriptItPicked() async {
        let verdict = await JudgeClient(http: FakeHTTP([reply(pick: "1", count: 2)]))
            .judge(results("the new series", "the new Siri"), settings: judgeSettings(), context: nil)
        #expect(verdict?.text == "the new Siri")
    }

    @Test func preferencesStayOutOfTheSharedState() async {
        let http = FakeHTTP([reply(prefs: 0.1)])
        let slack = AppContext(appName: "Slack", windowTitle: "general", bundleId: "x")
        _ = await JudgeClient(http: http).judge(results("Ship it."), settings: judgeSettings { $0.customInstructions = "lowercase" }, context: slack)
        #expect(http.bodies[0]["state"] == ["transcripts": ["Ship it."]])
        #expect(http.bodies[0]["questions"]!["prefs_0"] != nil)
        #expect(http.sent[0].timeoutInterval == 0.9)
    }
}

@Suite struct CatalogueTests {
    @Test func theTranscriptionCatalogueIsParsedAndUnreachableIsEmpty() async {
        let http = FakeHTTP([.data(200, #"{"data": [{"id": "microsoft/mai-transcribe-2", "name": "Microsoft AI: MAI-Transcribe 2"}, {"id": "x/plain"}]}"#)])
        let models = await Catalogue.fetchTranscriptionModels(http: http, key: "")
        #expect(models.map(\.id) == ["microsoft/mai-transcribe-2", "x/plain"])
        #expect(models.map(\.name) == ["MAI-Transcribe 2", "x/plain"])
        #expect(await Catalogue.fetchTranscriptionModels(http: FakeHTTP([]), key: "").isEmpty)
    }

    @Test func modelIdsAndRoutingStatus() {
        #expect(Catalogue.parseModelIds("a/b, c/d\ne/f") == ["a/b", "c/d", "e/f"])
        #expect(Catalogue.parseModelIds("a/b\n\n a/b \n c/d") == ["a/b", "c/d"])
        #expect(Catalogue.routingStatus(key: "", model: "", providerOrder: [], effort: "").contains("No OpenRouter key"))
        let status = Catalogue.routingStatus(key: "sk", model: "openai/gpt-6-luna", providerOrder: ["together"], effort: "none")
        #expect(status.contains("GPT-6 Luna") && status.contains("together") && status.contains("none reasoning"))
        #expect(Catalogue.correctionModels.contains { $0.id == CorrectionClient.defaultModel })
        #expect(Catalogue.correctionModels.contains { $0.id == Settings().openrouterCorrectionModel })
        for list in [Catalogue.correctionModels, Catalogue.transcriptionModels] {
            #expect(Set(list.map(\.id)).count == list.count)
        }
    }

    @Test func keyValidation() async {
        #expect(await Catalogue.validateKey(http: FakeHTTP([]), key: "short").error == "Key too short")
        #expect(await Catalogue.validateKey(http: FakeHTTP([.data(200, "{}")]), key: "sk-or-long-enough").valid)
        #expect(await Catalogue.validateKey(http: FakeHTTP([.data(401, "")]), key: "sk-or-long-enough").error == "Invalid key")
    }
}

/// A value set from a callback.
final class Box<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: T?
    var value: T? {
        get { lock.withLock { _value } }
        set { lock.withLock { _value = newValue } }
    }
}
