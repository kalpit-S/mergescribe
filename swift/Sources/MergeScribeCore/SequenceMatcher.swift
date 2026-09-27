/// A faithful port of Python's difflib.SequenceMatcher (no junk function).
///
/// The edit watcher and vocabulary learning were tuned against its exact output: which
/// block it calls the longest match, how it splits a change into opcodes, what ratio it
/// gives. A different diff algorithm would move those thresholds, so this follows
/// CPython's Lib/difflib.py step for step, including the "autojunk" heuristic.
public struct SequenceMatcher<Element: Hashable> {
    public enum Tag: String, Sendable { case equal, replace, delete, insert }

    public struct Opcode: Equatable, Sendable {
        public let tag: Tag
        public let i1: Int, i2: Int, j1: Int, j2: Int
        public init(_ tag: Tag, _ i1: Int, _ i2: Int, _ j1: Int, _ j2: Int) {
            self.tag = tag; self.i1 = i1; self.i2 = i2; self.j1 = j1; self.j2 = j2
        }
    }

    public struct Match: Equatable, Sendable {
        public let a: Int, b: Int, size: Int
    }

    public let a: [Element]
    public let b: [Element]
    private var b2j: [Element: [Int]] = [:]

    public init(_ a: [Element], _ b: [Element], autojunk: Bool = true) {
        self.a = a
        self.b = b
        for (j, element) in b.enumerated() { b2j[element, default: []].append(j) }
        // Elements that make up more than 1% of a long b are treated as junk.
        if autojunk && b.count >= 200 {
            let ntest = b.count / 100 + 1
            let popular = b2j.filter { $0.value.count > ntest }.keys
            for element in popular { b2j.removeValue(forKey: element) }
        }
    }

    /// The longest matching block in a[alo..<ahi] and b[blo..<bhi]; the earliest in a, then
    /// in b, among equally long ones.
    public func findLongestMatch(_ alo: Int, _ ahi: Int, _ blo: Int, _ bhi: Int) -> Match {
        var besti = alo, bestj = blo, bestsize = 0
        var j2len: [Int: Int] = [:]
        for i in alo..<max(alo, ahi) {
            var newj2len: [Int: Int] = [:]
            for j in b2j[a[i]] ?? [] {
                if j < blo { continue }
                if j >= bhi { break }
                let k = (j2len[j - 1] ?? 0) + 1
                newj2len[j] = k
                if k > bestsize {
                    besti = i - k + 1
                    bestj = j - k + 1
                    bestsize = k
                }
            }
            j2len = newj2len
        }
        // Extend the match by equal elements on each side. CPython's junk set comes only from
        // a junk function, which isn't used here, so elements purged as popular extend a
        // match like any other; the second pair of loops CPython has for junk never runs.
        while besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1] {
            besti -= 1; bestj -= 1; bestsize += 1
        }
        while besti + bestsize < ahi && bestj + bestsize < bhi && a[besti + bestsize] == b[bestj + bestsize] {
            bestsize += 1
        }
        return Match(a: besti, b: bestj, size: bestsize)
    }

    public func matchingBlocks() -> [Match] {
        var queue = [(0, a.count, 0, b.count)]
        var blocks: [Match] = []
        while let (alo, ahi, blo, bhi) = queue.popLast() {
            let m = findLongestMatch(alo, ahi, blo, bhi)
            if m.size > 0 {
                blocks.append(m)
                if alo < m.a && blo < m.b { queue.append((alo, m.a, blo, m.b)) }
                if m.a + m.size < ahi && m.b + m.size < bhi {
                    queue.append((m.a + m.size, ahi, m.b + m.size, bhi))
                }
            }
        }
        blocks.sort { ($0.a, $0.b, $0.size) < ($1.a, $1.b, $1.size) }
        // Collapse adjacent blocks.
        var i1 = 0, j1 = 0, k1 = 0
        var merged: [Match] = []
        for block in blocks {
            if i1 + k1 == block.a && j1 + k1 == block.b {
                k1 += block.size
            } else {
                if k1 > 0 { merged.append(Match(a: i1, b: j1, size: k1)) }
                i1 = block.a; j1 = block.b; k1 = block.size
            }
        }
        if k1 > 0 { merged.append(Match(a: i1, b: j1, size: k1)) }
        merged.append(Match(a: a.count, b: b.count, size: 0))
        return merged
    }

    public func opcodes() -> [Opcode] {
        var i = 0, j = 0
        var answer: [Opcode] = []
        for block in matchingBlocks() {
            let tag: Tag? = i < block.a && j < block.b ? .replace
                : i < block.a ? .delete
                : j < block.b ? .insert : nil
            if let tag { answer.append(Opcode(tag, i, block.a, j, block.b)) }
            i = block.a + block.size
            j = block.b + block.size
            if block.size > 0 { answer.append(Opcode(.equal, block.a, i, block.b, j)) }
        }
        return answer
    }

    /// 2 * matches / total length; 1.0 for two empty sequences.
    public func ratio() -> Double {
        let matches = matchingBlocks().reduce(0) { $0 + $1.size }
        let length = a.count + b.count
        return length > 0 ? 2.0 * Double(matches) / Double(length) : 1.0
    }
}

extension SequenceMatcher where Element == Unicode.Scalar {
    /// Python strings index by code point, so the diffs here do too.
    public init(_ a: String, _ b: String, autojunk: Bool = true) {
        self.init(Array(a.unicodeScalars), Array(b.unicodeScalars), autojunk: autojunk)
    }
}
