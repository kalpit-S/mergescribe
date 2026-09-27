import Foundation
import Testing
@testable import MergeScribeCore

private func row(_ session: String, _ typed: String, _ corrected: String, ts: Double = 0) -> Vocabulary.Row {
    Vocabulary.Row(sessionId: session, typed: typed, corrected: corrected, ts: ts)
}

private func afters(_ corrections: [Vocabulary.Correction]) -> [String] { corrections.map(\.after) }

@Suite struct RepeatedCorrectionTests {
    @Test func aCorrectionMadeInTwoDictationsIsKept() {
        // The Claude case: heard as "cloud", fixed twice.
        let rows = [row("s1", "thoughts about cloud today", "thoughts about Claude today"),
                    row("s2", "as good as cloud is", "as good as Claude is")]
        #expect(Vocabulary.repeatedCorrections(rows) == [.init(["cloud"], "Claude", 2)])
    }

    @Test func differentMishearingsOfOneWordAddUp() {
        let rows = [row("s1", "about cloud", "about Claude"), row("s2", "the Claud model", "the Claude model")]
        #expect(Vocabulary.repeatedCorrections(rows) == [.init(["cloud", "Claud"], "Claude", 2)])
    }

    @Test func oneCorrectionOrOneDictationIsNotEnough() {
        #expect(Vocabulary.repeatedCorrections([row("s1", "about cloud", "about Claude")]).isEmpty)
        let sameSession = [row("s1", "cloud and cloud", "Claude and Claude"), row("s1", "cloud", "Claude")]
        #expect(Vocabulary.repeatedCorrections(sameSession).isEmpty)
    }

    @Test func caseAndPunctuationChangesCarryNothing() {
        #expect(Vocabulary.repeatedCorrections([row("s1", "okay sure", "Okay, sure."), row("s2", "okay sure", "Okay, sure.")]).isEmpty)
    }

    @Test func longRewritesAreLeftOut() {
        let before = "we should probably go and look at it", after = "I will review the deployment logs later tonight"
        #expect(Vocabulary.repeatedCorrections([row("s1", before, after), row("s2", before, after)]).isEmpty)
    }

    @Test func codeDoesNotJudgeWhatACorrectionMeans() {
        // Ordinary words used to be filtered by a dictionary; that call is the model's now.
        let rows = [row("s1", "over their", "over there"), row("s2", "over their", "over there")]
        #expect(Vocabulary.repeatedCorrections(rows) == [.init(["their"], "there", 2)])
    }

    @Test func mostEstablishedFirstAndCapped() {
        let rows = (0..<3).map { row("a\($0)", "for post grass", "for Postgres") }
            + (0..<2).map { row("b\($0)", "about cloud", "about Claude") }
        #expect(afters(Vocabulary.repeatedCorrections(rows)) == ["Postgres", "Claude"])
        #expect(afters(Vocabulary.repeatedCorrections(rows, limit: 1)) == ["Postgres"])
    }

    // Word diffs cut inside words; the leftovers are not vocabulary.
    @Test func truncationArtifactsAreNotCorrections() {
        #expect(Vocabulary.repeatedCorrections([row("s1", "look at them okay", "look at the okay"),
                                                row("s2", "send them now", "send the now")]).isEmpty)
        // "not" -> "no" would teach the model to weaken negations.
        #expect(Vocabulary.repeatedCorrections([row("s1", "it is not ready", "it is no ready"),
                                                row("s2", "we are not sure", "we are no sure")]).isEmpty)
    }

    @Test func realRespellingsStillCount() {
        #expect(afters(Vocabulary.repeatedCorrections([row("s1", "deploy it to cooper netties", "deploy it to Kubernetes"),
                                                       row("s2", "the cube or netties cluster", "the Kubernetes cluster")])) == ["Kubernetes"])
        #expect(afters(Vocabulary.repeatedCorrections([row("s1", "parse the Jason file", "parse the JSON file"),
                                                       row("s2", "return Jason here", "return JSON here")])) == ["JSON"])
        // "Claud" -> "Claude" adds a letter; that is the whole point.
        #expect(afters(Vocabulary.repeatedCorrections([row("s1", "ask Claud about it", "ask Claude about it"),
                                                       row("s2", "Claud said so", "Claude said so")])) == ["Claude"])
    }

    @Test func lettersInAnyScript() {
        let learned = Vocabulary.repeatedCorrections([row("s1", "去北经", "去北京"), row("s2", "在上还", "在上海")], minSessions: 1)
        #expect(afters(learned).sorted() == ["去北京", "在上海"])
        #expect(EditDiff.isAppTransform("Café Ümlaut", "café ümlaut"))
        #expect(!EditDiff.isAppTransform("去北经", "去北京"))
    }
}

@Suite struct VocabularyPromptTests {
    @Test func nothingLearnedAddsNothing() {
        #expect(Vocabulary.correctionsPrompt([]) == "")
        #expect(Vocabulary.suppliedPrompt([]) == "")
    }

    @Test func correctionsAreEvidenceNotReplacements() {
        let text = Vocabulary.correctionsPrompt([.init(["cloud", "Claud"], "Claude", 2)])
        #expect(text.contains(#""cloud" or "Claud" corrected to "Claude" (2 dictations)"#))
        #expect(text.contains("not a list of replacements"))
        #expect(text.contains("only where the audio"))
    }

    @Test func thePromptTakesOnlyTheHeadOfALongList() {
        let prompt = Vocabulary.suppliedPrompt((0..<(Vocabulary.maxPromptTerms + 50)).map { "term\($0)" })
        #expect(prompt.contains("term\(Vocabulary.maxPromptTerms - 1)."))
        #expect(!prompt.contains("term\(Vocabulary.maxPromptTerms)"))
    }
}

@Suite struct VocabularyFileTests {
    private func jsonLine(_ session: String, _ typed: String, _ corrected: String) -> String {
        JSON.object(["session_id": .string(session), "typed": .string(typed), "corrected": .string(corrected), "ts": 0]).line()
    }

    @Test func readsTheCorpusAndNoticesNewCorrections() throws {
        let home = TempHome()
        let store = VocabularyStore(paths: home.paths)
        home.write("corrections.jsonl", jsonLine("s1", "about cloud", "about Claude") + "\n")
        #expect(store.learnedCorrections().isEmpty)
        home.write("corrections.jsonl", jsonLine("s1", "about cloud", "about Claude") + "\n"
                   + jsonLine("s2", "the Claud model", "the Claude model") + "\n")
        try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(5)],
                                              ofItemAtPath: home.paths.correctionsFile.path)
        #expect(store.learnedCorrections() == [.init(["cloud", "Claud"], "Claude", 2)])
    }

    @Test func missingFilesAreHarmless() {
        let store = VocabularyStore(paths: TempHome().paths)
        #expect(store.learnedCorrections().isEmpty)
        #expect(store.suppliedTerms().isEmpty)
    }

    @Test func suppliedTermsOnePerLineWithComments() {
        let home = TempHome()
        home.write("vocabulary.txt", "# from reading, most important first\nPostgres\n\n  Kubernetes   operator  # a phrase\n")
        #expect(VocabularyStore(paths: home.paths).suppliedTerms() == ["Postgres", "Kubernetes operator"])
    }

    @Test func knownTermsPutLearnedFirstAndRepeatNothing() {
        let home = TempHome()
        home.write("corrections.jsonl", jsonLine("s1", "thoughts about cloud today", "thoughts about Claude today") + "\n"
                   + jsonLine("s2", "as good as cloud is", "as good as Claude is"))
        home.write("vocabulary.txt", "claude\nPostgres\none two three four five six seven\n")
        #expect(VocabularyStore(paths: home.paths).knownTerms() == ["Claude", "Postgres"])   // 7 words is too long
    }
}
