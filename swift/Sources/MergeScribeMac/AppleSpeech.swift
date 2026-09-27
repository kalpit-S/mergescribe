@preconcurrency import AVFoundation
import Foundation
import MergeScribeCore
import Speech

/// Apple's on-device recognizer (SpeechAnalyzer, macOS 26) as a member of the ensemble.
///
/// Free, offline and quick: the model runs on this Mac, stays loaded for the life of the
/// app, and is primed with the speaker's vocabulary as contextual strings, the same terms
/// AssemblyAI is given.
@available(macOS 26.0, *)
public final class AppleRecognizer: Recognizer, @unchecked Sendable {
    public static let providerName = "apple-speech"
    public var name: String { Self.providerName }
    public var singleInstance: Bool { false }

    let locale: Locale
    let sampleRate: Int
    let vocabulary: @Sendable () -> [String]
    /// Contextual strings the analyzer takes; more mostly dilutes them.
    public static let maxContextualStrings = 100

    public init(locale: Locale = Locale(identifier: "en_US"), sampleRate: Int = 16000,
                vocabulary: @escaping @Sendable () -> [String] = { [] }) {
        self.locale = locale
        self.sampleRate = sampleRate
        self.vocabulary = vocabulary
    }

    /// True when the model for this locale is on this Mac, so nothing needs downloading.
    public static func available(locale: Locale = Locale(identifier: "en_US")) async -> Bool {
        let wanted = locale.identifier(.bcp47)
        return await SpeechTranscriber.installedLocales.contains { $0.identifier(.bcp47) == wanted }
    }

    public func transcribe(_ samples: [Float], mic: String) async throws -> TranscriptionResult {
        let clock = ContinuousClock(), started = clock.now
        let text = try await Self.transcribe(samples, sampleRate: sampleRate, locale: locale,
                                             vocabulary: Array(vocabulary().prefix(Self.maxContextualStrings)))
        return TranscriptionResult(text: text, provider: name, mic: mic,
                                   latencyMs: Int(((clock.now - started).seconds * 1000).rounded()))
    }

    /// One stretch of audio, transcribed whole.
    public static func transcribe(_ samples: [Float], sampleRate: Int, locale: Locale, vocabulary: [String],
                                  dictation: Bool = false) async throws -> String {
        if dictation {
            let transcriber = DictationTranscriber(locale: locale, contentHints: [], transcriptionOptions: [.punctuation],
                                                   reportingOptions: [], attributeOptions: [])
            return try await run(transcriber, transcriber.results, samples, sampleRate, vocabulary)
        }
        let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)
        return try await run(transcriber, transcriber.results, samples, sampleRate, vocabulary)
    }

    static func run<Results: AsyncSequence & Sendable>(_ transcriber: any SpeechModule, _ results: Results, _ samples: [Float],
                                                       _ sampleRate: Int, _ vocabulary: [String]) async throws -> String
        where Results.Element: SpeechModuleResult {
        guard let input = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false),
              let buffer = AVAudioPCMBuffer(pcmFormat: input, frameCapacity: AVAudioFrameCount(max(1, samples.count))) else {
            throw AppleSpeechError.audio
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
        let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber], considering: input) ?? input
        let converted = try convert(buffer, to: format)

        let context = AnalysisContext()
        if !vocabulary.isEmpty { context.contextualStrings[.general] = vocabulary }
        let (inputs, feed) = AsyncStream<AnalyzerInput>.makeStream()
        let analyzer = SpeechAnalyzer(inputSequence: inputs, modules: [transcriber],
                                      options: .init(priority: .userInitiated, modelRetention: .processLifetime),
                                      analysisContext: context)
        let collected = Task { () -> String in
            var pieces: [String] = []
            for try await result in results where result.isFinal {
                if let text = (result as? SpeechTranscriber.Result)?.text ?? (result as? DictationTranscriber.Result)?.text {
                    pieces.append(String(text.characters))
                }
            }
            return pieces.joined()
        }
        feed.yield(AnalyzerInput(buffer: converted))
        feed.finish()
        do {
            try await analyzer.finalizeAndFinishThroughEndOfInput()
        } catch {
            collected.cancel()
            throw error
        }
        return try await collected.value.collapsingWhitespace()
    }

    static func convert(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat) throws -> AVAudioPCMBuffer {
        if buffer.format == format { return buffer }
        guard let converter = AVAudioConverter(from: buffer.format, to: format),
              let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(
                Double(buffer.frameLength) * format.sampleRate / buffer.format.sampleRate) + 64) else {
            throw AppleSpeechError.audio
        }
        let fed = Locked(false)
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if fed.current { status.pointee = .endOfStream; return nil }
            fed.update { $0 = true }
            status.pointee = .haveData
            return buffer
        }
        if let error { throw error }
        return out
    }

    enum AppleSpeechError: Error { case audio }
}
