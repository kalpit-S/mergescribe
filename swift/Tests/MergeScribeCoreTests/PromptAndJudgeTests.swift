import Foundation
import Testing
@testable import MergeScribeCore

private func context(_ json: JSON?) -> AppContext? {
    guard let json, !json.isNull else { return nil }
    return AppContext(appName: json["app_name"]!.string!, windowTitle: json["window_title"]!.string!,
                      bundleId: json["bundle_id"]!.string!)
}

@Suite struct CorrectionPromptGoldenTests {
    static let golden = Golden.load("prompts")

    @Test func theDefaultPromptsAreThePythonOnes() {
        #expect(CorrectionPrompt.defaultSystem == Self.golden["default_system"]!.string!)
        #expect(CorrectionPrompt.defaultEditing == Self.golden["default_editing"]!.string!)
        #expect(CorrectionPrompt.defaultSystem.contains("exactly \(nothingMarker)"),
                "the marker the NothingMarker gate expects")
    }

    @Test func userPromptsMatchPython() {
        for case let .object(c) in Self.golden["prompts"]!.array! {
            let results = c["results"]!.array!.map {
                TranscriptionResult(text: $0["text"]!.string!, provider: $0["provider"]!.string!,
                                    mic: $0["mic"]!.string!, latencyMs: 0, chunk: $0["chunk"]!.int!)
            }.filter { !$0.text.trimmingCharacters(in: .whitespaces).isEmpty }
            #expect(CorrectionPrompt.user(results: results, context: context(c["context"])) == c["prompt"]!.string!)
        }
    }

    @Test func systemPromptsMatchPython() {
        for case let .object(c) in Self.golden["systems"]!.array! {
            var settings = Settings()
            settings.systemPrompt = c["system_prompt"]!.string!
            settings.customInstructions = c["custom_instructions"]!.string!
            settings.learnVocabulary = c["learn"]!.bool!
            let learned = c["learned"]!.array!.map {
                Vocabulary.Correction($0.array![0].array!.map { $0.string! }, $0.array![1].string!, $0.array![2].int!)
            }
            let supplied = c["supplied"]!.array!.map { $0.string! }
            #expect(CorrectionPrompt.system(settings: settings, learned: learned, supplied: supplied) == c["built"]!.string!)
        }
    }

    @Test func aPhraseSaidInTwoPartsIsKeptTwiceInOrder() {
        let r = { (text: String, provider: String, chunk: Int) in
            TranscriptionResult(text: text, provider: provider, mic: "m", latencyMs: 0, chunk: chunk)
        }
        let prompt = CorrectionPrompt.user(results: [r("part two", "a", 2), r("ship it", "a", 1),
                                                     r("ship it", "b", 1), r("ship it", "a", 3)], context: nil)
        let body = String(prompt[prompt.range(of: "Transcriptions")!.upperBound...])
        #expect(body.components(separatedBy: "ship it").count - 1 == 2)
        let p1 = body.range(of: "Part 1")!.lowerBound, p2 = body.range(of: "Part 2")!.lowerBound
        let p3 = body.range(of: "Part 3")!.lowerBound
        #expect(p1 < p2 && p2 < p3)
        #expect(body.range(of: "part two")!.lowerBound > p2)
    }
}

@Suite struct JudgeTests {
    static let golden = Golden.load("judge")

    @Test func theLimitIsPythons() {
        #expect(Judge.limit == Self.golden["limit"]!.double!)
    }

    @Test func questionsMatchPython() {
        for case let .object(c) in Self.golden["questions"]!.array! {
            let asked = Judge.questions(count: c["count"]!.int!, preferences: c["preferences"]!.string!,
                                        context: context(c["context"]))
            #expect(asked == c["questions"]!)
        }
    }

    @Test func decisionsMatchPython() {
        for case let .object(c) in Self.golden["decisions"]!.array! {
            let (clean, pick, scores) = try! Judge.decide(c["answers"]!, count: c["count"]!.int!)
            #expect(clean == c["clean"]!.bool!)
            #expect(pick == c["pick"]!.int!)
            let expected = c["scores"]!.object!.mapValues { $0.double! }
            #expect(Dictionary(uniqueKeysWithValues: scores.map { ($0.name, $0.value) }) == expected)
        }
    }

    @Test func candidatesAreDistinctAndLongestFirst() {
        let r = { (t: String) in TranscriptionResult(text: t, provider: "p", mic: "m", latencyMs: 0) }
        #expect(Judge.candidates([r("Ship it."), r("ship it."), r("")]) == ["Ship it."])
        #expect(Judge.candidates([r("the new Siri"), r("the new series")]) == ["the new series", "the new Siri"])
    }

    @Test func aReplacedCorrectionPromptCountsAsAPreference() {
        // The four jobs are the default prompt's; a prompt of your own can ask for more.
        let rule = "In Slack, write everything in lowercase."
        var s = Settings()
        s.customInstructions = rule
        s.systemPrompt = "Translate what I say into Spanish."
        #expect(Judge.preferences(s) == "Translate what I say into Spanish.\n\n\(rule)")
        for prompt in [CorrectionPrompt.defaultSystem, ""] {
            s.systemPrompt = prompt
            #expect(Judge.preferences(s) == rule)
        }
        s.customInstructions = "  "
        #expect(Judge.preferences(s) == "")
    }

    @Test func preferencesAreAskedAboutWhereTheTextIsGoing() {
        let slack = AppContext(appName: "Slack", windowTitle: "general - Acme - Slack", bundleId: "x")
        let asked = Judge.questions(count: 1, preferences: "lowercase in Slack", context: slack)
        let prefs = asked["prefs_0"]!["instructions"]!.string!
        #expect(prefs.contains("lowercase in Slack") && prefs.contains("Slack (window: general - Acme - Slack)"))
        #expect(Judge.questions(count: 1)["prefs_0"] == nil)
        #expect(Judge.destination(nil) == "an unknown app")
    }

    @Test func aMissingJobAnswerIsAnError() {
        #expect(throws: Judge.DecideError.self) {
            try Judge.decide(["best": ["choice": "0", "probabilities": ["0": 0.9]]], count: 1)
        }
    }
}
