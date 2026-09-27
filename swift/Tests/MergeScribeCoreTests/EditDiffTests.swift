import Foundation
import Testing
@testable import MergeScribeCore

private let typed = "Can you check the retry logic for post grass and rerun the migrations?"

private func spanOf(_ baseline: String, _ text: String) -> EditDiff.Span {
    let start = EditDiff.lastIndex(of: Array(text.unicodeScalars), in: Array(baseline.unicodeScalars))!
    return (start, start + text.unicodeScalars.count)
}

private func classify(_ base: String, _ current: String?, _ text: String = typed) -> (EditDiff.Outcome, String) {
    EditDiff.classify(base, current, spanOf(base, text))
}

@Suite struct EditClassifyTests {
    @Test func unchanged() {
        let base = "draft: \(typed)"
        #expect(classify(base, base).0 == .unchanged)
    }

    @Test func clearedAndGone() {
        #expect(classify(typed, "").0 == .cleared)
        #expect(classify(typed, nil).0 == .gone)
    }

    @Test func wordCorrected() {
        let (outcome, corrected) = classify(typed, typed.replacingOccurrences(of: "post grass", with: "Postgres"))
        #expect(outcome == .edited)
        #expect(corrected.contains("Postgres"))
    }

    @Test func typingAfterOrBeforeIsNotAnEdit() {
        // The bug that made the first implementation useless.
        #expect(classify(typed, typed + " Also check the after-hours flow.").0 == .unchanged)
        #expect(classify(typed, "Earlier note.\n" + typed).0 == .unchanged)
    }

    @Test func aSecondDictationIntoTheSameFieldIsNotAnEdit() {
        let base = "previous sentence. \(typed)"
        #expect(classify(base, base + " A whole new dictated sentence appended here.").0 == .unchanged)
    }

    @Test func anEditSurvivesSurroundingChanges() {
        let base = "intro. \(typed) outro."
        let current = "CHANGED INTRO. \(typed.replacingOccurrences(of: "post grass", with: "Postgres")) different outro."
        let (outcome, corrected) = classify(base, current)
        #expect(outcome == .edited && corrected.contains("Postgres"))
    }

    @Test func diffSpanReportsOnlyChangesInsideTheSpan() {
        let base = "hello " + typed
        #expect(EditDiff.diffSpan(base, "goodbye " + typed, spanOf(base, typed))?.changed == false)
        let inside = EditDiff.diffSpan(typed, typed.replacingOccurrences(of: "rerun", with: "re-run"), spanOf(typed, typed))
        #expect(inside?.changed == true && inside?.corrected.contains("re-run") == true)
    }
}

@Suite struct EditLargeFieldTests {
    // A field can hold a whole document; diffing it must stay cheap.
    static func document(words: Int = 9000) -> String {
        var rng = SeededRandom(seed: 1)
        let vocab = "the of and to in is you that it was for on are as with they at be this have".split(separator: " ")
        return (0..<words).map { _ in String(vocab[rng.next(vocab.count)]) }.joined(separator: " ")
    }

    static func withTyped() -> String {
        let doc = Array(document())
        return String(doc[..<20000]) + " " + typed + " " + String(doc[20000...])
    }

    @Test func anEditInsideALongDocumentIsFoundQuickly() {
        let base = Self.withTyped()
        let started = Date()
        let (outcome, corrected) = classify(base, base.replacingOccurrences(of: "post grass", with: "Postgres"))
        #expect(outcome == .edited && corrected.contains("Postgres"))
        #expect(Date().timeIntervalSince(started) < 0.5, "a quadratic diff here froze the HUD")
    }

    @Test func editsElsewhereInALongDocumentAreNotOurs() {
        let base = Self.withTyped()
        let result = classify(base, "A new first line. " + base + " And a new last one.")
        #expect(result.0 == .unchanged && result.1 == "")
    }

    @Test func aDocumentRewrittenEverywhereGivesUpRatherThanStalling() {
        let base = Self.withTyped()
        let current = base.replacingOccurrences(of: " the ", with: " a ").replacingOccurrences(of: "post grass", with: "Postgres")
        let started = Date()
        #expect(classify(base, current).0 == .unmeasured)
        #expect(Date().timeIntervalSince(started) < 0.5)
    }

    @Test func trimmedOpcodesRebuildTheNewText() {
        var rng = SeededRandom(seed: 7)
        let alphabet = Array("ab c".unicodeScalars)
        for _ in 0..<200 {
            let a = (0..<rng.next(31)).map { _ in alphabet[rng.next(4)] }
            let b = (0..<rng.next(31)).map { _ in alphabet[rng.next(4)] }
            let rebuilt = EditDiff.opcodes(a, b)!.flatMap { op in op.tag == .equal ? Array(a[op.i1..<op.i2]) : Array(b[op.j1..<op.j2]) }
            #expect(rebuilt == b)
        }
    }
}

@Suite struct EditRejectionTests {
    // A sent message leaves the composer showing its placeholder, not an edit.
    @Test func aPlaceholderAfterSendIsNotAnEdit() {
        let text = "And as far as the rollout goes, we already have the staging environment set up, so verifying the change should not take very long at all."
        let result = classify(text, "Ask ChatGPT", text)
        #expect(result.0 == .replaced && result.1 == "")
        let short = "Yeah, I just set up the new workstation and noticed a few things were broken."
        for placeholder in ["Ask Gemini", "Follow up", "GPT", "Message #general"] {
            #expect(classify(short, placeholder, short).0 == .replaced, "\(placeholder) should not count as an edit")
        }
    }

    @Test func trimmingHalfTheTextIsAReplacement() {
        let text = "First sentence that is fairly long. Second sentence also long enough."
        #expect(classify(text, "First sentence", text).0 == .replaced)
    }

    // The app rewriting text is not the user correcting it.
    @Test func appTransformsAreRejected() {
        let hike = "Quiet hiking trails near the coast in early spring"
        #expect(classify(hike, hike.replacingOccurrences(of: " ", with: "+"), hike).0 == .reformatted)
        let trace = "The trace shows the uploads land on U.K. storage"
        #expect(classify(trace, trace.lowercased(), trace).0 == .reformatted)
        let chart = "Could you move the chart onto the second slide?"
        #expect(classify(chart, "Could you move the chart onto the second slide", chart).0 == .reformatted)
    }

    @Test func realWordChangesAreStillCaptured() {
        let text = "From the trace the backups are going to EU storage"
        let (outcome, corrected) = classify(text, "From the trace the backups go to EU storage", text)
        #expect(outcome == .edited && corrected.contains("go to"))
        let vocab = "check the retry logic for post grass today"
        let fixed = classify(vocab, "check the retry logic for Postgres today", vocab)
        #expect(fixed.0 == .edited && fixed.1.contains("Postgres"))
    }
}

@Suite struct EditDiffGoldenTests {
    static let golden = Golden.load("feedback")

    @Test func classifyMatchesPython() {
        var count = 0
        for case let .object(c) in Self.golden["classify"]!.array! {
            let span = c["span"]!.array!.map { $0.int! }
            let (outcome, corrected) = EditDiff.classify(c["baseline"]!.string!, c["current"]!.string!, (span[0], span[1]))
            #expect(outcome.rawValue == c["outcome"]!.string!, "\(c["current"]!.string!.debugDescription)")
            #expect(corrected == c["corrected"]!.string!)
            count += 1
        }
        #expect(count == 400)
    }

    @Test func anchorMatchesPython() {
        for case let .object(c) in Self.golden["anchor"]!.array! {
            let span = EditDiff.anchor(c["field"]!.string!, c["typed"]!.string!)
            let expected = c["span"]!.array?.map { $0.int! }
            #expect(span.map { [$0.start, $0.end] } == expected, "\(c["field"]!.string!.debugDescription)")
        }
    }
}

/// A small deterministic generator, so a test's random input is the same every run.
struct SeededRandom {
    private var state: UInt64
    init(seed: UInt64) { state = seed &+ 0x9E3779B97F4A7C15 }
    mutating func next(_ bound: Int) -> Int {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        z ^= z >> 31
        return Int(z % UInt64(max(1, bound)))
    }
}
