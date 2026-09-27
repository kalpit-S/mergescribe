import Foundation

/// Deciding whether a transcript needs correcting at all.
///
/// Roughly a quarter of dictations come back already saying exactly what the speaker meant,
/// and waiting ~1.1s for a correction model to hand them back unchanged is pure latency.
/// TypeSafe's Jev answers typed questions about a state with calibrated probabilities, so the
/// questions are narrow and factual, and the decision to skip correction is made here, in
/// code, from the numbers.
///
/// The questions are the correction step's own jobs (see CorrectionPrompt.defaultSystem):
/// drop filler, drop stutters and false starts, fix mishearings, fix punctuation and
/// grammar. A transcript can be typed as it stands exactly when none of those jobs applies,
/// and none of the speaker's own preferences would be broken.
///
/// It runs alongside the correction request, never before it: a clean verdict wins the race
/// and types immediately, and anything else costs nothing because the correction was
/// already in flight. The measurements behind every choice here are in the Python
/// mergescribe/judge.py this was ported from, and scripts/judge_eval.py re-runs them.
public enum Judge {
    public static let decisionsURL = URL(string: "https://openrouter.ai/api/alpha/decisions")!
    /// Follows the latest Jev, to improve as it does. `limit` was fitted on
    /// jev-1.13-20260917; scripts/judge_eval.py shows whether a new version still sits well
    /// under it.
    public static let defaultModel = "~typesafe/jev-latest"

    /// One threshold for every question, including "none of them" on the best pick. It stays
    /// well below 0.5 on purpose: a borderline dictation repeated came back 0.58 and then
    /// 0.41, while confident answers repeat exactly.
    public static let limit = 0.3

    typealias Job = (name: String, claim: String, yes: String, no: String)

    /// The correction step's jobs, one snap judgment each, asked of every transcript. A
    /// question that bundles them ("could this be sent as is?") measured worse.
    static let jobs: [Job] = [
        ("filler", "contains filler: sounds like um or uh, or words that carry no meaning such as "
            + "like, you know, I mean, kind of, basically.",
         "at least one filler", "every word carries meaning"),
        ("restart", "contains a stutter, a repeated word or phrase, a false start, a self-correction, "
            + "or something the speaker called off (scratch that, never mind).",
         "present", "the speech runs straight through"),
        ("misheard", "has a word the speech recognizer probably got wrong: one that does not fit the "
            + "sentence, or that the other transcripts hear differently in a way that changes the meaning.",
         "a likely recognition error", "every word is probably right"),
        ("punct", "needs punctuation, capitalization or grammar fixes that a careful writer would "
            + "make before sending it.",
         "needs fixes", "reads correctly as written"),
    ]

    /// Asked only when the speaker has preferences, which typing a transcript as it stands
    /// would otherwise skip ("all lowercase in Slack"). They go in this question rather than
    /// the shared state: in the state they made every other question more cautious (skips
    /// fell from 8.0% to 4.2% of 400 dictations, with no fewer wrong ones), while here the
    /// other scores move by about 0.01.
    static func preferenceClaim(destination: String, preferences: String) -> String {
        "typed as it stands into \(destination), would break one of the speaker's preferences below "
            + "that applies there.\n\nSpeaker's preferences:\n\(preferences)"
    }
    static let preferenceAnswers = (yes: "breaks a preference", no: "follows every preference that applies here")

    /// Distinct non-empty transcripts, longest first so ties favour detail.
    public static func candidates(_ results: [TranscriptionResult]) -> [String] {
        var seen = Set<String>(), out: [String] = []
        let texts = results.map { $0.text.trimmingCharacters(in: .whitespacesAndNewlines) }
        // A stable sort by length, longest first (Python's sorted(..., reverse=True) keeps
        // equal lengths in their original order).
        let ordered = texts.enumerated().sorted { l, r in
            let a = l.element.unicodeScalars.count, b = r.element.unicodeScalars.count
            return a != b ? a > b : l.offset < r.offset
        }.map(\.element)
        for text in ordered where !text.isEmpty && seen.insert(text.lowercased()).inserted {
            out.append(text)
        }
        return out
    }

    /// What the speaker asked of the correction step beyond `jobs`: their written
    /// instructions, and their own prompt when it replaces the default. A replaced prompt
    /// all but stops skips (Jev reads a whole prompt as work still to do), which is safe.
    public static func preferences(_ settings: Settings) -> String {
        let prompt = settings.systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let replaced = prompt == CorrectionPrompt.defaultSystem.trimmingCharacters(in: .whitespacesAndNewlines) ? "" : prompt
        let instructions = settings.customInstructions.trimmingCharacters(in: .whitespacesAndNewlines)
        return [replaced, instructions].filter { !$0.isEmpty }.joined(separator: "\n\n")
    }

    /// Where the text is going: the app, and its window title when there is one.
    public static func destination(_ context: AppContext?) -> String {
        guard let context, !context.appName.isEmpty else { return "an unknown app" }
        return context.windowTitle.isEmpty ? context.appName : "\(context.appName) (window: \(context.windowTitle))"
    }

    /// Every question for `count` candidate transcripts, all answered in one call.
    public static func questions(count: Int, preferences: String = "", context: AppContext? = nil) -> JSON {
        var criteria: [String: JSON] = [:]
        for i in 0..<count { criteria[String(i)] = .string("transcript at index \(i)") }
        criteria["none"] = "none of them: each has at least one word wrong or missing"
        var out: [String: JSON] = ["best": [
            "type": "choice",
            "instructions": "Which entry of `transcripts` is most likely word for word what the speaker said?",
            "criteria": .object(criteria),
        ]]
        var asked = jobs
        if !preferences.isEmpty {
            asked.append(("prefs", preferenceClaim(destination: destination(context), preferences: preferences),
                          preferenceAnswers.yes, preferenceAnswers.no))
        }
        for i in 0..<count {
            for job in asked {
                out["\(job.name)_\(i)"] = [
                    "type": "noul",
                    "instructions": .string("Transcript at index \(i) \(job.claim)"),
                    "criteria": ["true": .string(job.yes), "false": .string(job.no)],
                ]
            }
        }
        return .object(out)
    }

    public struct Score: Equatable, Sendable {
        public let name: String
        public let value: Double
    }

    public enum DecideError: Error { case missingAnswer(String) }

    /// (clean, the index of the transcript to type, that transcript's scores).
    ///
    /// The pick is the likeliest transcript, not "none"; it is clean only when "none" is
    /// unlikely and none of the correction step's jobs applies to it.
    public static func decide(_ answers: JSON, count: Int) throws -> (clean: Bool, pick: Int, scores: [Score]) {
        guard let best = answers["best"] else { throw DecideError.missingAnswer("best") }
        let probabilities = best["probabilities"]?.object ?? [:]
        let indices = (0..<count).map(String.init)
        var pick = 0
        if indices.contains(where: { probabilities[$0] != nil }) {
            var top = -Double.infinity
            for (i, key) in indices.enumerated() {
                let p = probabilities[key]?.double ?? 0
                if p > top { top = p; pick = i }      // the first of equal maxima, as max() picks
            }
        } else if let choice = best["choice"].map(Self.pythonString), let index = indices.firstIndex(of: choice) {
            pick = index
        }
        var scores = [Score(name: "none", value: probabilities["none"]?.double ?? 0)]
        for job in jobs.map(\.name) + ["prefs"] {
            guard let noul = answers["\(job)_\(pick)"]?["noul"]?.double else {
                if job == "prefs" { continue }
                throw DecideError.missingAnswer("\(job)_\(pick)")
            }
            scores.append(Score(name: job, value: noul))
        }
        return (scores.allSatisfy { $0.value <= limit }, pick, scores)
    }

    /// str() of a JSON value, for comparing a choice with an index.
    static func pythonString(_ value: JSON) -> String {
        switch value {
        case .string(let s): return s
        case .number(let n): return n.rounded() == n ? String(Int(n)) : String(n)
        default: return ""
        }
    }

    /// What the judge concluded, and everything worth logging about it.
    public struct Verdict: Equatable, Sendable {
        public var clean: Bool
        public var text: String
        public var scores: [Score]
        public var model: String
        public var latencyMs: Int
    }
}
