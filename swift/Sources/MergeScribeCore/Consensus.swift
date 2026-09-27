import Foundation

/// Consensus: enough distinct recognizers heard the same words.
///
/// It settles what was said, so a chunk stops waiting for slower recognizers. It does not
/// settle whether the text needs tidying ("um, ship it" can be agreed on too), so agreed
/// text still goes to the judge and the correction model like any other.
public enum Consensus {
    /// Words only: case and punctuation aside. Apostrophes join a word ("don't"); any
    /// other mark separates words, so "re-sign" stays two words and doesn't match "resign".
    public static func normalize(_ text: String) -> String {
        var out = ""
        for ch in text where ch != "'" && ch != "\u{2019}" {
            out.append(ch.isLetter || ch.isNumber || ch == "_" || ch.isWhitespace ? ch : " ")
        }
        return out.lowercased().collapsingWhitespace()
    }

    /// The agreed text (as the first agreeing recognizer wrote it), or nil.
    ///
    /// Counts distinct providers, not matching results: the same model on two mics agrees
    /// with itself on its own systematic errors. Two mics rule out acoustic noise, not
    /// model bias.
    public static func check(_ results: [TranscriptionResult], threshold: Int) -> String? {
        let normalized = results.map { ($0, normalize($0.text)) }.filter { !$0.1.isEmpty }
        guard !normalized.isEmpty else { return nil }
        // The most common reading; on a tie, the one seen first (Counter.most_common's order).
        var counts: [String: Int] = [:]
        var order: [String] = []
        for (_, norm) in normalized {
            if counts[norm] == nil { order.append(norm) }
            counts[norm, default: 0] += 1
        }
        let top = counts.values.max()!
        let winner = order.first { counts[$0] == top }!
        let providers = Set(normalized.filter { $0.1 == winner }.map { $0.0.provider })
        guard providers.count >= threshold else { return nil }
        return normalized.first { $0.1 == winner }!.0.text
    }

    /// How many results say the agreed text.
    public static func matching(_ results: [TranscriptionResult], _ consensus: String) -> Int {
        let target = normalize(consensus)
        return results.filter { normalize($0.text) == target }.count
    }
}

extension String {
    /// Runs of whitespace as single spaces, none at the ends: Python's " ".join(s.split()).
    public func collapsingWhitespace() -> String {
        split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }
}
