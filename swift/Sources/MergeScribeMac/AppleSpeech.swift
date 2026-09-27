@preconcurrency import AVFoundation
import Foundation
import MergeScribeCore
import Speech

/// How settings.json names Apple's recognizer in enabled_providers.
public enum AppleProvider {
    public static let name = "apple-speech"
}

/// Apple's on-device recognizer (SpeechAnalyzer, macOS 26) as a member of the ensemble.
///
/// Free, offline and quick: the model runs on this Mac, stays loaded for the life of the
/// app, and is primed with the speaker's vocabulary as contextual strings, the same terms
/// AssemblyAI is given.
@available(macOS 26.0, *)
public final class AppleRecognizer: Recognizer, @unchecked Sendable {
    public static let providerName = AppleProvider.name
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

/// Apple's recognizer listening along while you talk, for the words on the HUD.
///
/// Fed the primary microphone's blocks as they are captured; reports the text so far
/// (finished phrases plus the one still being heard) each time it changes. Display only:
/// what gets typed still comes from the ensemble and the correction.
@available(macOS 26.0, *)
public final class LiveTranscriber: @unchecked Sendable {
    private let transcriber: SpeechTranscriber
    private let analyzer: SpeechAnalyzer
    private let feed: AsyncStream<AnalyzerInput>.Continuation
    private let format: AVAudioFormat
    private let converter: AVAudioConverter?
    private let input: AVAudioFormat
    private let reader: Task<Void, Never>
    private let lock = NSLock()
    private var finished = false

    /// Starts listening; nil when the model isn't available.
    public static func start(sampleRate: Int, locale: Locale = Locale(identifier: "en_US"), vocabulary: [String],
                             onText: @escaping @Sendable (String) -> Void) async -> LiveTranscriber? {
        guard await AppleRecognizer.available(locale: locale),
              let input = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false)
        else { return nil }
        let transcriber = SpeechTranscriber(locale: locale, transcriptionOptions: [], reportingOptions: [.volatileResults, .fastResults],
                                            attributeOptions: [])
        let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber], considering: input) ?? input
        return LiveTranscriber(transcriber: transcriber, input: input, format: format, vocabulary: vocabulary, onText: onText)
    }

    private init(transcriber: SpeechTranscriber, input: AVAudioFormat, format: AVAudioFormat, vocabulary: [String],
                 onText: @escaping @Sendable (String) -> Void) {
        self.transcriber = transcriber
        self.input = input
        self.format = format
        converter = input == format ? nil : AVAudioConverter(from: input, to: format)
        let context = AnalysisContext()
        if !vocabulary.isEmpty { context.contextualStrings[.general] = Array(vocabulary.prefix(AppleRecognizer.maxContextualStrings)) }
        let (inputs, feed) = AsyncStream<AnalyzerInput>.makeStream()
        self.feed = feed
        analyzer = SpeechAnalyzer(inputSequence: inputs, modules: [transcriber],
                                  options: .init(priority: .userInitiated, modelRetention: .processLifetime),
                                  analysisContext: context)
        let results = transcriber.results
        reader = Task {
            var settled = "", volatile = ""
            do {
                for try await result in results {
                    let text = String(result.text.characters)
                    if result.isFinal { settled += text; volatile = "" } else { volatile = text }
                    onText((settled + volatile).collapsingWhitespace())
                }
            } catch {}
        }
    }

    /// One block of the primary microphone's audio.
    public func append(_ samples: [Float]) {
        guard !lock.withLock({ finished }), !samples.isEmpty,
              let buffer = AVAudioPCMBuffer(pcmFormat: input, frameCapacity: AVAudioFrameCount(samples.count)) else { return }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
        guard let converter else { feed.yield(AnalyzerInput(buffer: buffer)); return }
        let capacity = AVAudioFrameCount(Double(samples.count) * format.sampleRate / input.sampleRate) + 32
        guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else { return }
        let fed = Locked(false)
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if fed.current { status.pointee = .noDataNow; return nil }
            fed.update { $0 = true }
            status.pointee = .haveData
            return buffer
        }
        if error == nil, out.frameLength > 0 { feed.yield(AnalyzerInput(buffer: out)) }
    }

    /// Stop listening. What it heard doesn't matter any more: the ensemble has the audio.
    public func stop() {
        let first: Bool = lock.withLock { defer { finished = true }; return !finished }
        guard first else { return }
        feed.finish()
        let analyzer = analyzer, reader = reader
        Task {
            await analyzer.cancelAndFinishNow()
            reader.cancel()
        }
    }

    /// Load the model now, so the first dictation doesn't wait for it.
    public static func warmUp(sampleRate: Int) {
        Task.detached(priority: .utility) {
            _ = try? await AppleRecognizer.transcribe([Float](repeating: 0, count: sampleRate / 2), sampleRate: sampleRate,
                                                      locale: Locale(identifier: "en_US"), vocabulary: [])
        }
    }
}

/// The live transcriber for one recording: blocks that arrive before it is ready are held,
/// and nothing is fed once it has stopped.
public final class LiveWords: @unchecked Sendable {
    private let lock = NSLock()
    private var pending: [[Float]] = []
    private var transcriber: AnyObject?
    private var stopped = false

    public init() {}

    public func append(_ block: [Float]) {
        lock.lock()
        if stopped { lock.unlock(); return }
        guard let transcriber else {
            if pending.count < 400 { pending.append(block) }
            lock.unlock()
            return
        }
        lock.unlock()
        if #available(macOS 26.0, *) { (transcriber as? LiveTranscriber)?.append(block) }
    }

    public func start(sampleRate: Int, vocabulary: [String], onText: @escaping @Sendable (String) -> Void) {
        guard #available(macOS 26.0, *) else { return }
        Task {
            guard let live = await LiveTranscriber.start(sampleRate: sampleRate, vocabulary: vocabulary, onText: onText) else { return }
            let held: [[Float]]? = lock.withLock {
                if stopped { return nil }
                defer { pending = []; transcriber = live }
                return pending
            }
            guard let held else { live.stop(); return }
            for block in held { live.append(block) }
        }
    }

    public func stop() {
        let live: AnyObject? = lock.withLock {
            stopped = true
            pending = []
            return transcriber
        }
        if #available(macOS 26.0, *) { (live as? LiveTranscriber)?.stop() }
    }
}
