import Foundation

/// The correction model's reply when the speaker called the whole dictation off.
public let nothingMarker = "[nothing]"

/// True when a finished correction is the "type nothing" marker.
public func isNothing(_ text: String) -> Bool {
    text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased().hasPrefix(nothingMarker)
}

/// Recognizes the correction model saying "type nothing", without delaying text.
///
/// An empty reply can't carry that meaning: empty is how a failed call looks, and a failure
/// falls back to typing the raw transcript. Output streams, so the marker would be typed
/// before it was complete: text is held back only while it could still be the marker, and
/// anything that does not start with "[" passes through on the first token.
public struct NothingMarker: Sendable {
    private var held = ""
    private var decided = false
    public private(set) var calledOff = false

    public init() {}

    public mutating func feed(_ text: String) -> String {
        if decided { return calledOff ? "" : text }
        held += text
        let probe = String(held.drop(while: \.isWhitespace)).lowercased()
        if probe.hasPrefix(nothingMarker) {
            decided = true
            calledOff = true
            return ""
        }
        if nothingMarker.hasPrefix(probe) { return "" }     // could still be the marker
        decided = true
        defer { held = "" }
        return held
    }

    /// Release anything held back by a reply that ended part way through a prefix.
    public mutating func flush() -> String {
        if decided { return "" }
        decided = true
        defer { held = "" }
        return held
    }
}

/// Flattens the line breaks a correction model invents in dictated speech.
///
/// A newline typed through CGEvent is a Return keystroke, and in most chat composers Return
/// sends: a paragraph break would submit half the dictation. Speech has no paragraphs, and
/// 14% of measured outputs contained one. Stateful because a break can straddle a token
/// boundary, and the space standing in for it must not double up with the next token's.
public struct DictationFilter: Sendable {
    private var endsWithSpace: Bool

    /// continuing: appending to a dictation already in the field, where a leading space is
    /// the separator rather than noise.
    public init(continuing: Bool = false) {
        endsWithSpace = !continuing
    }

    public mutating func feed(_ text: String) -> String {
        guard !text.isEmpty else { return "" }
        // Scalars, not Characters: "\r\n" is one Character and would slip past a character class.
        var flattened = text.replacing(/[\r\n]+/.matchingSemantics(.unicodeScalar), with: " ")
        if endsWithSpace { flattened = String(flattened.drop { $0 == " " }) }
        guard !flattened.isEmpty else { return "" }
        flattened = flattened.replacing(/\ {2,}/, with: " ")
        endsWithSpace = flattened.hasSuffix(" ")
        return flattened
    }
}

/// Recognizer output without its non-speech annotations: speaker labels (<|speaker:0|>),
/// sound events ([typing sound], [pause]), whisper's [BLANK_AUDIO]. A raw transcript can be
/// typed as it stands, and consensus compares transcripts word for word.
public func stripAnnotations(_ text: String) -> String {
    text.replacing(/<\|[^|>]*\|>|\[[^\]\n]{1,40}\]/, with: " ").collapsingWhitespace()
}
