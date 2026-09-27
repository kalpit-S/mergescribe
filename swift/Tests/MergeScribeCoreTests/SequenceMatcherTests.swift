import Testing
@testable import MergeScribeCore

@Suite struct SequenceMatcherTests {
    static let golden = Golden.load("sequence_matcher")

    private func opcodes<T>(_ sm: SequenceMatcher<T>) -> [[JSON]] {
        sm.opcodes().map { [.string($0.tag.rawValue), JSON($0.i1), JSON($0.i2), JSON($0.j1), JSON($0.j2)] }
    }

    @Test func stringsMatchPythonExactly() {
        var checked = 0
        for case let .object(c) in Self.golden["strings"]!.array! {
            let a = c["a"]!.string!, b = c["b"]!.string!
            let sm = SequenceMatcher(a, b, autojunk: c["autojunk"]!.bool!)
            #expect(opcodes(sm) == c["opcodes"]!.array!.map { $0.array! }, "opcodes for \(a.debugDescription) -> \(b.debugDescription)")
            #expect(abs(sm.ratio() - c["ratio"]!.double!) < 1e-12)
            let m = sm.findLongestMatch(0, sm.a.count, 0, sm.b.count)
            #expect([m.a, m.b, m.size] == c["longest"]!.array!.map { $0.int! })
            checked += 1
        }
        #expect(checked == 400)
    }

    @Test func wordListsMatchPythonExactly() {
        for case let .object(c) in Self.golden["words"]!.array! {
            let a = c["a"]!.array!.map { $0.string! }, b = c["b"]!.array!.map { $0.string! }
            #expect(opcodes(SequenceMatcher(a, b, autojunk: false)) == c["opcodes"]!.array!.map { $0.array! })
        }
    }

    @Test func emptySequences() {
        #expect(SequenceMatcher("", "").ratio() == 1.0)
        #expect(SequenceMatcher("", "").opcodes().isEmpty)
        #expect(SequenceMatcher("abc", "").opcodes() == [.init(.delete, 0, 3, 0, 0)])
    }
}
