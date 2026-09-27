import Foundation

/// Post-output edit detection: the pure part.
///
/// After text is typed, the destination field's whole value is snapshotted as a baseline
/// and the typed span located inside it. Later reads are diffed against the baseline, and
/// only changes landing inside that span count. Comparing the typed text against the whole
/// field (the first implementation) couldn't tell "you rewrote my words" from "you typed
/// more nearby", and dictating twice into one field is normal usage.
///
/// Everything here indexes by Unicode code point, as the Python it replaces did, so the
/// thresholds it was tuned with mean the same thing.
public enum EditDiff {
    /// Ignore trailing-whitespace noise.
    static let minEditChars = 2
    /// A correction keeps most of the original. Anything that replaces the text wholesale
    /// is a send or a clear: composers show a placeholder ("Ask ChatGPT", "Message #general")
    /// once submitted, and diffing a paragraph against that produced 83% of the first corpus
    /// as garbage.
    static let minEditSimilarity = 0.5
    static let minEditLengthRatio = 0.5
    /// A field can hold a whole document. Edits are local, so the unchanged ends are
    /// stripped first and only what is left is diffed, up to this much.
    static let maxDiffChars = 6000
    static let maxFuzzyFieldChars = 20000

    public typealias Text = [Unicode.Scalar]
    public typealias Span = (start: Int, end: Int)

    public enum Outcome: String, Sendable {
        /// Our span survived as typed (text elsewhere may have changed).
        case unchanged
        /// Our span was modified; the corrected text is the new version.
        case edited
        /// The field emptied: message sent, draft discarded.
        case cleared
        /// Swapped wholesale: a send leaves a placeholder.
        case replaced
        /// Only case or punctuation changed: the app normalised it.
        case reformatted
        /// The field no longer exists and nothing in its app replaced it.
        case gone
        /// Our text is no longer there verbatim, and the field changed too much to diff.
        case unmeasured
        // Reported by the watcher, not by classify().
        case unreadable, unanchored
    }

    // MARK: - Locating what we typed

    /// Where the typed text sits in the field's value.
    ///
    /// Exact match usually works, but apps normalise on insert (collapsing whitespace,
    /// converting newlines, trimming), so this falls back to a distinctive tail of the typed
    /// text, then to a fuzzy block match, before giving up.
    public static func anchor(_ baseline: String, _ typed: String) -> Span? {
        let base = Array(baseline.unicodeScalars), text = Array(typed.unicodeScalars)
        // The last occurrence: if the same phrase appears twice, ours is the later one.
        if let start = lastIndex(of: text, in: base) { return (start, start + text.count) }

        let tail = text.count > 60 ? Array(text.suffix(60)) : text
        if tail.count >= 12, let start = lastIndex(of: tail, in: base) {
            return (max(0, start + tail.count - text.count), start + tail.count)
        }

        guard base.count <= maxFuzzyFieldChars else { return nil }
        let match = SequenceMatcher(base, text, autojunk: false).findLongestMatch(0, base.count, 0, text.count)
        if match.size >= max(20, Int(Double(text.count) * 0.5)) {
            let start = max(0, match.a - match.b)
            return (start, min(base.count, start + text.count))
        }
        return nil
    }

    static func lastIndex(of needle: Text, in haystack: Text) -> Int? {
        if needle.isEmpty { return haystack.count }
        guard needle.count <= haystack.count else { return nil }
        var start = haystack.count - needle.count
        while start >= 0 {
            if haystack[start] == needle[0] && haystack[start..<(start + needle.count)].elementsEqual(needle) {
                return start
            }
            start -= 1
        }
        return nil
    }

    // MARK: - Diffing

    typealias Opcode = SequenceMatcher<Unicode.Scalar>.Opcode

    /// Opcodes turning a into b, diffing only what lies between the unchanged ends. nil when
    /// even that is too large to diff.
    static func opcodes(_ a: Text, _ b: Text) -> [Opcode]? {
        var prefix = 0
        while prefix < min(a.count, b.count) && a[prefix] == b[prefix] { prefix += 1 }
        var suffix = 0
        while suffix < min(a.count, b.count) - prefix && a[a.count - 1 - suffix] == b[b.count - 1 - suffix] {
            suffix += 1
        }
        let aMid = Array(a[prefix..<(a.count - suffix)]), bMid = Array(b[prefix..<(b.count - suffix)])
        guard aMid.count + bMid.count <= maxDiffChars else { return nil }
        var out: [Opcode] = prefix > 0 ? [Opcode(.equal, 0, prefix, 0, prefix)] : []
        if !aMid.isEmpty || !bMid.isEmpty {
            for op in SequenceMatcher(aMid, bMid, autojunk: false).opcodes() {
                out.append(Opcode(op.tag, op.i1 + prefix, op.i2 + prefix, op.j1 + prefix, op.j2 + prefix))
            }
        }
        if suffix > 0 { out.append(Opcode(.equal, a.count - suffix, a.count, b.count - suffix, b.count)) }
        return out
    }

    /// A position in the baseline translated to one in the current text. preferEnd asks for
    /// an exclusive end offset, so a position inside an unchanged block maps one past itself.
    static func mapIndex(_ opcodes: [Opcode], _ index: Int, preferEnd: Bool = false) -> Int {
        for op in opcodes where op.i1 <= index && index < op.i2 {
            if op.tag == .equal { return op.j1 + (index - op.i1) + (preferEnd ? 1 : 0) }
            return preferEnd ? op.j2 : op.j1
        }
        return opcodes.last?.j2 ?? 0
    }

    /// Whether an edit landed inside the span, and what the span says now. nil when the
    /// field changed too much, in too many places, to diff.
    static func diffSpan(_ baseline: Text, _ current: Text, _ span: Span) -> (changed: Bool, corrected: Text)? {
        guard let ops = opcodes(baseline, current) else { return nil }
        let changed = ops.contains { $0.tag != .equal && $0.i1 < span.end && $0.i2 > span.start }
        let newStart = mapIndex(ops, span.start)
        let newEnd = mapIndex(ops, max(span.start, span.end - 1), preferEnd: true)
        return (changed, Array(current[newStart..<max(newStart, newEnd)]))
    }

    public static func diffSpan(_ baseline: String, _ current: String, _ span: Span) -> (changed: Bool, corrected: String)? {
        diffSpan(Array(baseline.unicodeScalars), Array(current.unicodeScalars), span)
            .map { ($0.changed, String(String.UnicodeScalarView($0.corrected))) }
    }

    // MARK: - Classifying

    /// What happened to the typed span, and the corrected text when it was edited.
    public static func classify(_ baseline: String, _ current: String?, _ span: Span) -> (Outcome, String) {
        guard let current else { return (.gone, "") }
        if current.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return (.cleared, "") }

        let base = Array(baseline.unicodeScalars), now = Array(current.unicodeScalars)
        let typedScalars = Array(base[span.start..<span.end])
        guard let diff = diffSpan(base, now, span) else {
            let intact = lastIndex(of: typedScalars, in: now) != nil
            return (intact ? .unchanged : .unmeasured, "")
        }
        guard diff.changed else { return (.unchanged, "") }

        let corrected = scalarString(diff.corrected).trimmingCharacters(in: .whitespacesAndNewlines)
        let typed = scalarString(typedScalars).trimmingCharacters(in: .whitespacesAndNewlines)
        let correctedCount = corrected.unicodeScalars.count, typedCount = typed.unicodeScalars.count
        if correctedCount < minEditChars { return (.cleared, "") }
        // Both checks are needed: a placeholder can be similar in length to a short
        // dictation, and a long unrelated string can score a high ratio on common words.
        if Double(correctedCount) < Double(typedCount) * minEditLengthRatio { return (.replaced, "") }
        if SequenceMatcher(typed, corrected).ratio() < minEditSimilarity { return (.replaced, "") }
        if isAppTransform(typed, corrected) { return (.reformatted, "") }
        return (.edited, corrected)
    }

    /// True when the app rewrote the text rather than the user correcting it: a browser
    /// omnibox URL-encodes a query ("Quiet hiking trails" -> "Quiet+hiking+trails"), some
    /// fields lowercase or strip punctuation on commit. If only case, punctuation and
    /// separators differ, nothing was corrected.
    public static func isAppTransform(_ typed: String, _ corrected: String) -> Bool {
        letters(typed) == letters(corrected)
    }

    static func scalarString(_ scalars: Text) -> String { String(String.UnicodeScalarView(scalars)) }
}

/// The letters and digits of any script, case-folded: what a word is, not how it is typed.
public func letters(_ text: String) -> String {
    let folded = text.folding(options: .caseInsensitive, locale: nil)
    return String(String.UnicodeScalarView(folded.unicodeScalars.filter(isAlphanumeric)))
}

/// Python's str.isalnum() for one code point: a letter, or any kind of number.
func isAlphanumeric(_ scalar: Unicode.Scalar) -> Bool {
    switch scalar.properties.generalCategory {
    case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter:
        return true
    default:
        return scalar.properties.numericType != nil
    }
}
