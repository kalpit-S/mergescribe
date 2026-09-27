import Foundation

/// Corrections the speaker has made more than once, handed to the model as evidence.
///
/// When the same edit shows up in two separate dictations (the recognizers wrote "cloud",
/// the speaker changed it to "Claude"), that is strong evidence about the speaker's
/// vocabulary. Deciding what a correction means (a name the recognizers keep missing, a
/// stylistic preference, noise) is the model's job; code decides only what counts as
/// repeated, and trims what would waste the prompt.
public enum Vocabulary {
    /// Supplied terms the correction prompt can afford.
    public static let maxPromptTerms = 100
    /// AssemblyAI's limits per request: terms, and words per term.
    public static let maxKeyterms = 1000
    public static let maxKeytermWords = 6
    /// One correction is an anecdote; the same one twice is a pattern.
    public static let minSessions = 2
    /// Bounds how much the prompt can grow.
    public static let maxCorrections = 30
    /// Long rewrites cost prompt space and say little about vocabulary.
    static let maxSpanWords = 4
    static let edgePunctuation = CharacterSet(charactersIn: ".,!?;:\"'()[]")

    /// What the recognizers wrote, what the speaker corrected it to, and in how many dictations.
    public struct Correction: Equatable, Sendable {
        public var before: [String]
        public var after: String
        public var count: Int
        public init(_ before: [String], _ after: String, _ count: Int) {
            self.before = before; self.after = after; self.count = count
        }
    }

    /// Words every recognizer already knows. A learned correction made only of these came from
    /// rewording ("anything" -> "it"), and priming a recognizer with them does nothing but skew
    /// it; the correction model still sees them as evidence and judges them itself.
    public static let everyday: Set<String> = Set("""
        a about above across actually after again against all almost along already also although
        always am among an and any anyone anything are arent around as at away back bad basically be
        because been before behind being below best better between big bit both but by came can cant
        case come could couldnt day days definitely did didnt do does doesnt doing done dont down
        during each else even ever every everyone everything fact feel few first for from gave get
        getting give go goes going gone gonna good got gotta great had hadnt has hasnt have havent
        having he her here hers hes high him his how i id if ill im in instead into is isnt it its
        ive just keep kind kinda knew know last let lets like liked literally little long look lot
        low made make many may maybe me mean meant might mine more most much must my myself near
        need never new next no nope nor not nothing now of off often ok okay old on one ones only
        onto or other our ours out over own part people per perhaps person place point probably put
        quite rather really right said same saw say says see seem seen shall she shes short should
        shouldnt since small so some someone something sometimes sort sorta still such sure take
        tell than that thats the their theirs them then there theres these they theyre theyve thing
        things think this those though thought through time times to today together told tomorrow
        too took toward towards try under unless until up upon us use used very via wanna want was
        wasnt way we well went were werent weve what whatever whats when where whether which while
        who whom whose why will with within without wont work would wouldnt wrong yeah year years
        yep yes yesterday yet you youd youll your youre yours youve
        """.split(whereSeparator: \.isWhitespace).map(String.init))

    /// False for a term made only of everyday words: a recognizer gains nothing from it.
    public static func worthPriming(_ term: String) -> Bool {
        term.split(whereSeparator: \.isWhitespace).contains { word in
            let key = letters(String(word))
            return !key.isEmpty && !everyday.contains(key)
        }
    }

    /// Word spans the user replaced, as (what was typed, what it became).
    static func replacements(_ typed: String, _ corrected: String) -> [(String, String)] {
        let before = typed.split(whereSeparator: \.isWhitespace).map(String.init)
        let after = corrected.split(whereSeparator: \.isWhitespace).map(String.init)
        return SequenceMatcher(before, after, autojunk: false).opcodes().filter { $0.tag == .replace }.map {
            (before[$0.i1..<$0.i2].joined(separator: " ").trimmingCharacters(in: edgePunctuation),
             after[$0.j1..<$0.j2].joined(separator: " ").trimmingCharacters(in: edgePunctuation))
        }
    }

    /// True when the "correction" is just a fragment of what was there.
    ///
    /// Word diffs cut inside words when an edit lands mid-word, so the corpus fills with
    /// pairs like "them" -> "the" and "not" -> "no" that teach nothing and, for negations,
    /// teach harm. Only a shorter fragment is an artifact: spelling a word out more fully
    /// ("Claud" -> "Claude") is exactly what this is for.
    static func isTruncation(_ before: String, _ after: String) -> Bool {
        let a = letters(before), b = letters(after)
        return !a.isEmpty && !b.isEmpty && b.unicodeScalars.count < a.unicodeScalars.count && a.contains(b)
    }

    /// A row of corrections.jsonl, as far as vocabulary is concerned.
    public struct Row: Sendable {
        public var sessionId: String?
        public var typed: String
        public var corrected: String
        public var ts: Double
        public init(sessionId: String?, typed: String, corrected: String, ts: Double = 0) {
            self.sessionId = sessionId; self.typed = typed; self.corrected = corrected; self.ts = ts
        }

        init(json: JSON) {
            sessionId = json["session_id"]?.string.flatMap { $0.isEmpty ? nil : $0 }
            typed = json["typed"]?.string ?? ""
            corrected = json["corrected"]?.string ?? ""
            ts = json["ts"]?.double ?? 0
        }
    }

    /// Words the speaker corrected to the same thing in at least `minSessions` dictations.
    ///
    /// Grouped by what the text was corrected *to*, so "cloud" and "Claud" both fixed to
    /// "Claude" count as evidence for one word. Most dictations first, then most recent.
    public static func repeatedCorrections(_ rows: [Row], minSessions: Int = minSessions,
                                           limit: Int = maxCorrections) -> [Correction] {
        var sessions: [String: Set<String>] = [:]
        var afters: [String: [String]] = [:]
        var befores: [String: [String]] = [:]
        var lastSeen: [String: Double] = [:]
        var order: [String] = []
        for (index, row) in rows.enumerated() {
            let session = row.sessionId ?? "row-\(index)"
            for (before, after) in replacements(row.typed, row.corrected) {
                if before.isEmpty || after.isEmpty { continue }
                if max(before.split(whereSeparator: \.isWhitespace).count,
                       after.split(whereSeparator: \.isWhitespace).count) > maxSpanWords { continue }
                if letters(before) == letters(after) { continue }   // only case or punctuation changed
                if isTruncation(before, after) { continue }         // a diff artifact, not a word
                let key = letters(after)
                if sessions[key] == nil { order.append(key) }
                sessions[key, default: []].insert(session)
                afters[key, default: []].append(after)
                befores[key, default: []].append(before)
                lastSeen[key] = max(lastSeen[key] ?? 0, row.ts)
            }
        }
        let kept = order.enumerated().filter { sessions[$0.element]!.count >= minSessions }
            .sorted { lhs, rhs in
                let l = (sessions[lhs.element]!.count, lastSeen[lhs.element]!)
                let r = (sessions[rhs.element]!.count, lastSeen[rhs.element]!)
                return l != r ? l > r : lhs.offset < rhs.offset
            }
            .map(\.element)
        return kept.prefix(limit).map { key in
            var seen = Set<String>()
            let uniqueBefores = befores[key]!.filter { seen.insert($0).inserted }
            return Correction(uniqueBefores, mostCommon(afters[key]!), sessions[key]!.count)
        }
    }

    /// The most frequent form; on a tie, the most recent.
    static func mostCommon(_ items: [String]) -> String {
        var counts: [String: Int] = [:]
        for item in items { counts[item, default: 0] += 1 }
        let top = counts.values.max()!
        return items.last { counts[$0] == top }!
    }

    /// The supplied terms as the correction model sees them.
    public static func suppliedPrompt(_ terms: [String]) -> String {
        guard !terms.isEmpty else { return "" }
        return "Words and names this speaker uses: " + terms.prefix(maxPromptTerms).joined(separator: ", ")
            + ". Where the audio fits one of these, spell it this way."
    }

    /// The corrections as the model sees them: evidence, not replacements.
    public static func correctionsPrompt(_ corrections: [Correction]) -> String {
        guard !corrections.isEmpty else { return "" }
        let described = corrections.map { c in
            c.before.map { "\"\($0)\"" }.joined(separator: " or ") + " corrected to \"\(c.after)\" (\(c.count) dictations)"
        }.joined(separator: "; ")
        return "Across separate earlier dictations, this speaker corrected the typed text in these "
            + "ways: \(described). This is evidence about the words they use, not a list of "
            + "replacements: apply one only where the audio and the surrounding words support it."
    }
}

/// The vocabulary files, re-read only when they change.
public final class VocabularyStore: @unchecked Sendable {
    public let correctionsFile: URL
    /// Words handed in from outside, one per line, most important first: anything that
    /// knows the speaker's vocabulary before they say it can write here.
    public let suppliedFile: URL
    /// Learned corrections the speaker removed in Settings, by what they corrected to; kept
    /// beside the corpus.
    public var forgottenFile: URL { correctionsFile.deletingLastPathComponent().appendingPathComponent("vocabulary_forgotten.txt") }
    private let lock = NSLock()
    private var learnedCache: (Date?, [Vocabulary.Correction]) = (nil, [])
    private var suppliedCache: (Date?, [String]) = (nil, [])
    private var forgottenCache: (Date?, Set<String>) = (nil, [])

    public init(paths: AppPaths = AppPaths()) {
        correctionsFile = paths.correctionsFile
        suppliedFile = paths.vocabularyFile
    }

    public init(correctionsFile: URL, suppliedFile: URL) {
        self.correctionsFile = correctionsFile
        self.suppliedFile = suppliedFile
    }

    private static func modified(_ url: URL) -> Date? {
        (try? FileManager.default.attributesOfItem(atPath: url.path))?[.modificationDate] as? Date
    }

    /// Repeated corrections from the corpus, less any the speaker removed.
    public func learnedCorrections() -> [Vocabulary.Correction] {
        let forgotten = forgottenKeys()
        return repeated().filter { !forgotten.contains(letters($0.after)) }
    }

    /// Stop using a learned correction; the corpus itself is kept.
    public func forget(_ after: String) throws {
        let line = Data((after.trimmingCharacters(in: .whitespacesAndNewlines) + "\n").utf8)
        try FileManager.default.createDirectory(at: forgottenFile.deletingLastPathComponent(), withIntermediateDirectories: true)
        if let handle = try? FileHandle(forWritingTo: forgottenFile) {
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: line)
        } else {
            try line.write(to: forgottenFile)
        }
    }

    private func forgottenKeys() -> Set<String> {
        guard let modified = Self.modified(forgottenFile) else { return [] }
        lock.lock()
        if forgottenCache.0 == modified { defer { lock.unlock() }; return forgottenCache.1 }
        lock.unlock()
        let text = (try? String(contentsOf: forgottenFile, encoding: .utf8)) ?? ""
        let keys = Set(text.split(whereSeparator: \.isNewline).map {
            letters(String($0.split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false).first ?? ""))
        }).subtracting([""])
        lock.lock(); forgottenCache = (modified, keys); lock.unlock()
        return keys
    }

    private func repeated() -> [Vocabulary.Correction] {
        guard let modified = Self.modified(correctionsFile) else { return [] }
        lock.lock()
        if learnedCache.0 == modified { defer { lock.unlock() }; return learnedCache.1 }
        lock.unlock()
        let text = (try? String(contentsOf: correctionsFile, encoding: .utf8)) ?? ""
        let rows = text.split(whereSeparator: \.isNewline).compactMap { line -> Vocabulary.Row? in
            guard let json = try? JSONDecoder().decode(JSON.self, from: Data(line.utf8)) else { return nil }
            return Vocabulary.Row(json: json)
        }
        let corrections = Vocabulary.repeatedCorrections(rows)
        lock.lock(); learnedCache = (modified, corrections); lock.unlock()
        return corrections
    }

    /// The terms in the vocabulary file ('#' starts a comment).
    public func suppliedTerms() -> [String] {
        guard let modified = Self.modified(suppliedFile) else { return [] }
        lock.lock()
        if suppliedCache.0 == modified { defer { lock.unlock() }; return suppliedCache.1 }
        lock.unlock()
        let text = (try? String(contentsOf: suppliedFile, encoding: .utf8)) ?? ""
        let terms = text.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline).compactMap { line -> String? in
            let term = String(line.split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false).first ?? "")
                .collapsingWhitespace()
            return term.isEmpty ? nil : term
        }
        lock.lock(); suppliedCache = (modified, terms); lock.unlock()
        return terms
    }

    /// Every term the speaker is known to use, sized for priming a recognizer: repeated
    /// corrections first (proven in their own dictations, everyday words left out), then the
    /// supplied ones, each once.
    public func knownTerms() -> [String] {
        var seen = Set<String>(), out: [String] = []
        for term in learnedCorrections().map(\.after).filter(Vocabulary.worthPriming) + suppliedTerms() {
            let key = letters(term)
            if !key.isEmpty && !seen.contains(key)
                && term.split(whereSeparator: \.isWhitespace).count <= Vocabulary.maxKeytermWords {
                seen.insert(key)
                out.append(term)
            }
        }
        return Array(out.prefix(Vocabulary.maxKeyterms))
    }
}
