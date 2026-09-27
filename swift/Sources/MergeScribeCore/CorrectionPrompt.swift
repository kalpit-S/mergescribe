import Foundation

/// What the correction model is told, and the transcripts it is shown.
public enum CorrectionPrompt {
    /// The default system prompt. Settings can replace it; `Judge.JOBS` mirror its jobs.
    public static let defaultSystem = """
Turn this speech-to-text transcript into the message the speaker would have typed if they had written it themselves.

Typed text drops the scaffolding of speech: filler sounds, filler words that carry no meaning ("like", "you know", "I mean", "kind of", "basically"), stutters, repeats, and false starts. When the speaker corrects themselves, keep only the version they settled on. When the speaker calls something off with "scratch that" or "never mind", drop it along with what it cancels; anything they say after still stands. Reply with exactly [nothing] only if nothing at all is left.

Everything else is theirs. Keep their words, slang, tone, and phrasing, and never add words they didn't say. Fix punctuation, grammar, and clear mishearings; when several transcriptions of the same audio are given, use them together to work out what was said. Recognizers often drop or swap short words like "not", "never" and "no" ("no reason" heard as "a reason"), so if any transcription has a negation the others lack, keep it: losing one reverses the meaning. Unfamiliar names are usually real products, people, or jargon, so keep them. When unsure, keep what was said.

Never use an em dash or en dash. Where the speaker breaks off, restarts or changes course, end the sentence there or use a comma, as a person typing would.

Return only the cleaned text, as one continuous line with no line breaks, markdown, or commentary.
"""

    public static let defaultEditing = "You are a text editing assistant. Apply the user's requested change precisely and return only the edited text."

    /// The system prompt: the base (the user's own if they replaced it), their preferences,
    /// and what vocabulary learning knows.
    public static func system(settings: Settings, learned: [Vocabulary.Correction], supplied: [String]) -> String {
        var prompt = settings.systemPrompt.isEmpty ? defaultSystem : settings.systemPrompt
        if !settings.customInstructions.isEmpty {
            prompt += "\n\nUser preferences:\n\(settings.customInstructions)"
        }
        if settings.learnVocabulary {
            for evidence in [Vocabulary.correctionsPrompt(learned), Vocabulary.suppliedPrompt(supplied)] where !evidence.isEmpty {
                prompt += "\n\n\(evidence)"
            }
        }
        return prompt
    }

    /// The user message: where the text is going, and what each recognizer heard.
    ///
    /// Earlier dictations are deliberately not included: the model regurgitated them instead
    /// of correcting the current one (14 of 42 captured corrections leaked, worst case 187
    /// words of a previous dictation for a 4-word utterance).
    public static func user(results: [TranscriptionResult], context: AppContext?) -> String {
        // One line per distinct reading; identical ones only cost tokens.
        func listing(_ group: [TranscriptionResult]) -> [String] {
            var seen = Set<String>(), lines: [String] = []
            for r in group {
                let normalized = r.text.lowercased().collapsingWhitespace()
                if !normalized.isEmpty && seen.insert(normalized).inserted {
                    lines.append("[\(r.provider)/\(r.mic)]: \(r.text)")
                }
            }
            return lines
        }
        // A long dictation is transcribed in consecutive parts. Listing them flat would read
        // as alternatives of one piece of audio, and deduplicating across them would drop a
        // sentence the speaker really said twice.
        let parts = Set(results.map(\.chunk)).sorted()
        let transcriptions: String
        if parts.count > 1 {
            let blocks = parts.enumerated().map { n, part in
                "Part \(n + 1):\n" + listing(results.filter { $0.chunk == part }).joined(separator: "\n")
            }
            transcriptions = "The audio came in consecutive parts; each lists what the recognizers heard of that part.\n"
                + blocks.joined(separator: "\n")
        } else {
            transcriptions = listing(results).joined(separator: "\n")
        }
        var sections: [String] = []
        if let context {
            var lines = ["Active application: \(context.appName)"]
            if !context.windowTitle.isEmpty { lines.append("Window: \(context.windowTitle)") }
            sections.append(lines.joined(separator: "\n"))
        }
        sections.append("Transcriptions:\n\(transcriptions)")
        return sections.joined(separator: "\n\n")
    }

    /// Selection + speech: the transcript is an instruction for the selected text.
    public static func editing(selected: String, command: String) -> String {
        "TASK: \(command)\n\nORIGINAL TEXT:\n\(selected)\n\n"
            + "INSTRUCTIONS: Apply the task to the original text above. Return ONLY the edited text, "
            + "nothing else. No explanations, no formatting, no extra content."
    }
}
