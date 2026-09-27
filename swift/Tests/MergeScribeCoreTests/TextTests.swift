import Testing
@testable import MergeScribeCore

private func result(_ text: String, _ provider: String, _ mic: String = "m") -> TranscriptionResult {
    TranscriptionResult(text: text, provider: provider, mic: mic, latencyMs: 100)
}

@Suite struct ConsensusTests {
    @Test func normalizeStripsPunctuationAndWhitespace() {
        #expect(Consensus.normalize("Hello, world!") == "hello world")
        #expect(Consensus.normalize("Hello.") == "hello")
        #expect(Consensus.normalize("hello   world") == "hello world")
        #expect(Consensus.normalize("  hello  ") == "hello")
    }

    @Test func punctuationThatSplitsAWordIsNotIgnored() {
        // "re-sign" and "resign" are different words; commas and full stops are not.
        #expect(Consensus.normalize("re-sign the lease") != Consensus.normalize("resign the lease"))
        #expect(Consensus.normalize("Hello, world.") == Consensus.normalize("hello world"))
        #expect(Consensus.normalize("Don't ship it") == Consensus.normalize("dont ship it"))
        #expect(Consensus.normalize("Don’t ship it") == Consensus.normalize("dont ship it"))
    }

    @Test func lettersInAnyScriptAreWords() {
        #expect(Consensus.normalize("Ça marche, merci.") == "ça marche merci")
        #expect(Consensus.normalize("東京に行く。") == "東京に行く")
    }

    @Test func exactMatch() {
        let results = [result("Hello world", "p1", "m1"), result("Hello world", "p2", "m1"),
                       result("Hello world", "p1", "m2")]
        #expect(Consensus.check(results, threshold: 2) == "Hello world")
    }

    @Test func sameProviderOnTwoMicsIsNotConsensus() {
        // One model agreeing with itself across mics proves nothing about model bias.
        let results = [result("Ableton routing", "parakeet", "mbp"), result("Ableton routing", "parakeet", "solocast")]
        #expect(Consensus.check(results, threshold: 2) == nil)
    }

    @Test func twoProvidersAgreeing() {
        let results = [result("Ship it today", "parakeet"), result("Ship it today", "fish-audio")]
        #expect(Consensus.check(results, threshold: 2) == "Ship it today")
    }

    @Test func punctuationDifferencesAgreeAndTheFirstWordingIsKept() {
        let results = [result("Hello world.", "p1"), result("Hello world", "p2")]
        #expect(Consensus.check(results, threshold: 2) == "Hello world.")
    }

    @Test func fillerAndLengthDoNotStopAgreement() {
        // Whether to drop the "um" is the correction's job, not a reason to keep waiting.
        #expect(Consensus.check([result("Um, ship it today", "a"), result("Um, ship it today", "b")], threshold: 2)
                == "Um, ship it today")
        let long = Array(repeating: "word", count: 40).joined(separator: " ")
        #expect(Consensus.check([result(long, "a"), result(long, "b")], threshold: 2) == long)
    }

    @Test func noMatch() {
        #expect(Consensus.check([result("Hello world", "p1"), result("Hi there", "p2")], threshold: 2) == nil)
        #expect(Consensus.check([result("", "p1"), result(" ", "p2")], threshold: 2) == nil)
        #expect(Consensus.check([], threshold: 2) == nil)
    }
}

@Suite struct DictationFilterTests {
    // Line breaks must never reach the keyboard: Return sends in chat apps.
    @Test func flattensBreaksAcrossTokenBoundaries() {
        var f = DictationFilter()
        #expect(["Okay cool.", "\n\n", "Also, it'd be"].map { f.feed($0) }.joined() == "Okay cool. Also, it'd be")
    }

    @Test func collapsesABreakNextToExistingSpaces() {
        var f = DictationFilter()
        #expect(["one ", "\n\n", " two"].map { f.feed($0) }.joined() == "one two")
    }

    @Test func swallowsLeadingWhitespaceUnlessContinuing() {
        var fresh = DictationFilter()
        #expect(fresh.feed("   hello") == "hello")
        var continuing = DictationFilter(continuing: true)
        #expect(continuing.feed(" hello") == " hello")
    }

    @Test func handlesCarriageReturns() {
        var f = DictationFilter()
        #expect(f.feed("a\r\nb") == "a b")
    }

    @Test func ordinaryTextPassesThrough() {
        var f = DictationFilter()
        let text = "Nothing special here, just a sentence."
        #expect(f.feed(text) == text)
    }

    @Test func emptyTokensAreSafe() {
        var f = DictationFilter()
        #expect(f.feed("") == "")
        #expect(f.feed("\n") == "")
    }
}

@Suite struct NothingMarkerTests {
    // The model's "type nothing" reply must be recognized without delaying real text.
    @Test func ordinaryTextPassesOnTheFirstToken() {
        var gate = NothingMarker()
        #expect(gate.feed("Ship ") == "Ship ")
        #expect(gate.feed("it.") == "it.")
        #expect(!gate.calledOff)
    }

    @Test func theMarkerSplitAcrossTokensIsCaught() {
        var gate = NothingMarker()
        #expect(["[no", "th", "ing]"].map { gate.feed($0) } == ["", "", ""])
        #expect(gate.calledOff)
        #expect(gate.flush() == "")
    }

    @Test func caseAndSurroundingSpaceDoNotMatter() {
        var gate = NothingMarker()
        _ = gate.feed("  [Nothing]")
        #expect(gate.calledOff)
        #expect(isNothing(" [NOTHING] "))
        #expect(!isNothing("nothing to see here"))
    }

    @Test func bracketedTextIsReleasedOnceItDiverges() {
        var gate = NothingMarker()
        #expect(gate.feed("[no") == "")
        #expect(gate.feed("te] fix") == "[note] fix")
        #expect(!gate.calledOff)
    }

    @Test func aReplyThatStopsMidPrefixIsReleased() {
        var gate = NothingMarker()
        _ = gate.feed("[not")
        #expect(gate.flush() == "[not")
    }
}

@Suite struct AnnotationTests {
    @Test func nonSpeechAnnotationsAreStripped() {
        // A raw transcript can be typed as it stands, so sound tags must not reach it.
        let fish = "<|speaker:0|> So I was [pause] thinking [typing sound] we should ship it."
        #expect(stripAnnotations(fish) == "So I was thinking we should ship it.")
        #expect(stripAnnotations("[BLANK_AUDIO]") == "")
        #expect(stripAnnotations("Testing, 1, 2, 3.") == "Testing, 1, 2, 3.")
    }
}
