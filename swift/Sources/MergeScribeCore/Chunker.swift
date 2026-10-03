import Foundation

/// Audio from every recording microphone for one stretch of a dictation, in the order the
/// microphones were opened.
public struct AudioChunk: Sendable, Equatable {
    public var mics: [(name: String, samples: [Float])]

    public init(_ mics: [(name: String, samples: [Float])] = []) { self.mics = mics }

    public var isEmpty: Bool { mics.allSatisfy { $0.samples.isEmpty } }
    public var heard: [(name: String, samples: [Float])] { mics.filter { !$0.samples.isEmpty } }
    public func samples(_ mic: String) -> [Float]? { mics.first { $0.name == mic }?.samples }

    public static func == (lhs: AudioChunk, rhs: AudioChunk) -> Bool {
        lhs.mics.map(\.name) == rhs.mics.map(\.name) && lhs.mics.map(\.samples) == rhs.mics.map(\.samples)
    }
}

/// The final chunk and whether the speech detector heard anything in it.
public struct FinalChunk: Sendable {
    public var chunk: AudioChunk
    public var hasSpeech: Bool
}

/// Silence-based chunking and speech detection, fed blocks of audio.
///
/// Pure: the Mac audio engine hands it copied 16 kHz mono blocks with the time each was
/// captured, and delivers whatever chunks it emits. Chunks transcribe while the speaker is
/// still talking, so a long dictation is mostly done by the time the key comes up.
///
/// Speech is judged against each microphone's own noise floor, measured all the time (not
/// only while dictating), so the same code works on a quiet built-in mic and a hot condenser,
/// in a quiet room and a café. The same detector decides whether the last chunk holds any
/// speech at all: one judgment of "is anyone talking", not two that can disagree.
public struct Chunker: Sendable {
    // Chunk boundaries
    /// Don't emit chunks shorter than this.
    static let minChunkSeconds = 3.0
    /// Keep this much silence at the end of a chunk.
    static let trailingSilenceSeconds = 0.5
    /// Start looking for somewhere to cut after this. Cutting the moment the cap is reached
    /// lands mid-word: 35% of chunks were cut that way, and the correction model then spent
    /// 18% of its edits within two words of a seam. Once overdue, cut at the next gap.
    static let maxChunkSeconds = 30.0
    /// A breath between words is enough to hide a cut.
    static let seamSilenceSeconds = 0.15
    /// Nobody talks this long without a gap; cut anyway.
    static let hardMaxChunkSeconds = 45.0

    // Speech detection
    /// Used until a microphone's own floor is known.
    public static let fallbackNoiseFloorDb = -50.0
    static let fixedSilenceThresholdDb = -35.0
    static let noiseFloorWindowSeconds = 30.0
    public static let blockSize = 1024
    /// The floor is known once this many blocks have been measured (about 0.6s).
    static let floorSamplesNeeded = 10

    // The old whole-chunk gate (0.35s, -55 dBFS), kept to report where it and the detector
    // disagree about a final chunk, and used on a microphone whose floor isn't known yet.
    static let minSessionSeconds = 0.35
    static let minSessionDb = -55.0

    public struct Config: Sendable, Equatable {
        public var sampleRate = 16000
        public var prerollSeconds = 1.0
        public var silenceThreshold = 1.2
        public var chunkOnSilence = true
        public var adaptiveThreshold = true
        public var speechHeadroomDb = 12.0
        public var speechHysteresisDb = 3.0
        public var noiseFloorPercentile = 10

        public init() {}

        public init(_ settings: Settings) {
            sampleRate = settings.sampleRate
            prerollSeconds = settings.prerollSeconds
            silenceThreshold = settings.silenceThreshold
            // A streamed dictation is transcribed as it goes: cutting it into chunks would
            // only send the batch fallback pieces it doesn't need.
            chunkOnSilence = settings.chunkOnSilence && !settings.streaming
            adaptiveThreshold = settings.adaptiveThresholdEnabled
            speechHeadroomDb = settings.speechHeadroomDb
            speechHysteresisDb = settings.speechHysteresisDb
            noiseFloorPercentile = settings.noiseFloorPercentile
        }
    }

    struct Mic: Sendable {
        var name: String
        var preroll: [(heard: Double, block: [Float])] = []
        var chunk: [[Float]] = []
        var floorSamples: [Double] = []   // recent block levels, dB
        var floorCount = 0                // blocks measured, to pace the recompute
        var noiseFloor: Double = Chunker.fallbackNoiseFloorDb
        var floorMeasured = false         // the floor above is this mic's own, not the guess
        var speaking = false              // hysteresis state
        var errors = 0
    }

    public var config: Config
    var mics: [Mic] = []
    public private(set) var isRecording = false
    public private(set) var primaryMic: String?
    var lastSpeech: Double = 0
    var chunkHasSpeech = false
    /// 0..1, the primary mic's level for the HUD meter.
    public private(set) var level: Double = 0
    /// Lines worth logging, drained by the engine.
    public var messages: [String] = []

    public init(config: Config = Config()) { self.config = config }

    var prerollBlocks: Int { max(1, Int(config.prerollSeconds * Double(config.sampleRate)) / Self.blockSize) }
    var floorWindow: Int { Int(Self.noiseFloorWindowSeconds * Double(config.sampleRate)) / Self.blockSize }

    // MARK: - Microphones

    public var micNames: [String] { mics.map(\.name) }

    public mutating func addMic(_ name: String) {
        guard !mics.contains(where: { $0.name == name }) else { return }
        mics.append(Mic(name: name))
        if primaryMic == nil { primaryMic = name }
    }

    public mutating func removeMic(_ name: String) {
        mics.removeAll { $0.name == name }
        if primaryMic == name {
            primaryMic = mics.first?.name
            messages.append(primaryMic.map { "[Audio] Primary mic now: \($0)" } ?? "[Audio] No active mics remaining")
        }
    }

    public func noiseFloor(_ mic: String) -> Double? { mics.first { $0.name == mic }?.noiseFloor }

    // MARK: - Recording

    /// Begin a dictation: what the microphones heard just before the press opens it.
    ///
    /// preroll false: the microphones were closed until now, so anything held is the tail of
    /// the last dictation, and a dictation started just after it would say that again.
    public mutating func startRecording(now: Double, preroll: Bool = true) {
        isRecording = true
        lastSpeech = now          // assume speech at the start
        chunkHasSpeech = false
        // Only the moment before the press: a mic that stopped delivering (AirPods idling)
        // would otherwise replay what it last heard, maybe minutes ago.
        let horizon = preroll ? config.prerollSeconds + 0.25 : -1
        for i in mics.indices {
            let fresh = mics[i].preroll.filter { now - $0.heard <= horizon }.map(\.block)
            mics[i].chunk = fresh
            mics[i].preroll.removeAll()
            // Speech in the pre-roll counts: a word said as the key went down is still a word.
            for block in fresh where !isSilence(block, micIndex: i, updateLevel: false) {
                chunkHasSpeech = true
            }
        }
    }

    /// What a mic's chunk holds so far: at the start of a dictation, its pre-roll.
    public func held(_ mic: String) -> [[Float]] {
        mics.first { $0.name == mic }?.chunk ?? []
    }

    /// End the dictation: the audio since the last chunk, and whether it holds speech.
    public mutating func stopRecording() -> FinalChunk {
        isRecording = false
        let heardSpeech = chunkHasSpeech
        lastSpeech = 0
        chunkHasSpeech = false
        let chunk = flush()
        let longEnough = chunk.mics.contains { Double($0.samples.count) / Double(config.sampleRate) >= Self.minSessionSeconds }
        let floorsKnown = mics.allSatisfy(\.floorMeasured)
        let loudEnough = levelGate(chunk)
        // Before the room has been measured the detector compares against a fixed guess, and
        // could discard quiet speech; the level gate is the better judge until then.
        let hasSpeech = longEnough && (floorsKnown && config.adaptiveThreshold ? heardSpeech : loudEnough)
        if longEnough && floorsKnown && config.adaptiveThreshold && heardSpeech != loudEnough {
            messages.append("[Audio] Final chunk: the speech detector says \(heardSpeech ? "speech" : "silence"), "
                            + "the old level gate would have said \(loudEnough ? "speech" : "silence")")
        }
        return FinalChunk(chunk: chunk, hasSpeech: hasSpeech)
    }

    /// Any microphone at least 0.35s long and -55 dBFS or louder over the whole chunk.
    func levelGate(_ chunk: AudioChunk) -> Bool {
        chunk.mics.contains { mic in
            guard Double(mic.samples.count) / Double(config.sampleRate) >= Self.minSessionSeconds else { return false }
            let rms = Self.rms(mic.samples)
            return rms > 0 && 20 * log10(rms) >= Self.minSessionDb
        }
    }

    /// One block from one microphone. Returns a chunk when this block ends one.
    public mutating func feed(_ mic: String, _ block: [Float], now: Double) -> AudioChunk? {
        guard let i = mics.firstIndex(where: { $0.name == mic }) else { return nil }
        mics[i].preroll.append((now, block))
        if mics[i].preroll.count > prerollBlocks { mics[i].preroll.removeFirst(mics[i].preroll.count - prerollBlocks) }
        // The room is measured all the time, not only while dictating: otherwise the first
        // dictation after launch, or on a new mic, judges speech against a fixed guess.
        sampleNoiseFloor(block, micIndex: i)

        guard isRecording else { return nil }
        mics[i].chunk.append(block)

        // Any mic hearing speech holds the chunk open. Detection runs even when chunks aren't
        // cut on silence: it also decides whether the final chunk holds speech at all.
        if !isSilence(block, micIndex: i, updateLevel: true) {
            lastSpeech = now
            chunkHasSpeech = true
        }
        guard config.chunkOnSilence else { return nil }
        // The primary mic alone decides when to cut, so chunks aren't emitted twice.
        guard mic == primaryMic else { return nil }
        let silence = now - lastSpeech
        let duration = Double(mics[i].chunk.reduce(0) { $0 + $1.count }) / Double(config.sampleRate)
        let paused = silence >= config.silenceThreshold && duration >= Self.minChunkSeconds

        // A chunk holding no speech is pure room tone, and recognizers hallucinate
        // confidently on silence ("Yeah.", "Mm-hmm."). Never hand dead air to a provider;
        // just keep a long silent hold from growing the buffers without bound.
        guard chunkHasSpeech else {
            if duration >= Self.maxChunkSeconds {
                for j in mics.indices { mics[j].chunk.removeAll() }
            }
            return nil
        }
        // Without a cap, a chunk only ends on a pause, so continuous talking would transcribe
        // nothing until the key comes up.
        let overdue = duration >= Self.maxChunkSeconds
        let atSeam = overdue && silence >= Self.seamSilenceSeconds
        let forced = duration >= Self.hardMaxChunkSeconds
        guard paused || atSeam || forced else { return nil }

        if paused {
            let excess = silence - Self.trailingSilenceSeconds
            if excess > 0 { trimTrailing(Int(excess * Double(config.sampleRate))) }
        }
        let reason = paused ? "pause" : atSeam ? "seam past \(Int(Self.maxChunkSeconds))s" : "hard cap \(Int(Self.hardMaxChunkSeconds))s"
        messages.append("[Audio] === CHUNK EMIT (\(reason)) === duration=\(String(format: "%.2f", duration))s, "
                        + "silence=\(String(format: "%.2f", silence))s")
        lastSpeech = now
        chunkHasSpeech = false
        return flush()
    }

    // MARK: - Speech detection

    static func rms(_ samples: [Float]) -> Double {
        guard !samples.isEmpty else { return 0 }
        var sum = 0.0
        for s in samples { sum += Double(s) * Double(s) }
        return (sum / Double(samples.count)).squareRoot()
    }

    /// Whether a block is silence, against the mic's floor with hysteresis: speech starts
    /// `speechHeadroomDb` above the floor and continues until it drops `speechHysteresisDb`
    /// below that, so it doesn't flicker in noise.
    mutating func isSilence(_ block: [Float], micIndex i: Int, updateLevel: Bool) -> Bool {
        let rms = Self.rms(block)
        guard rms > 0 else { return true }
        let db = 20 * log10(rms)
        let silent: Bool
        var floor: Double?
        if config.adaptiveThreshold {
            floor = mics[i].noiseFloor
            let headroom = config.speechHeadroomDb
            let release = max(0, headroom - max(0, config.speechHysteresisDb))
            let active = mics[i].speaking ? db >= floor! + release : db >= floor! + headroom
            mics[i].speaking = active
            silent = !active
        } else {
            silent = db < Self.fixedSilenceThresholdDb
        }
        if updateLevel && mics[i].name == primaryMic {
            level = HUDModel.normalizeLevel(db: db, noiseFloor: floor)
        }
        return silent
    }

    mutating func sampleNoiseFloor(_ block: [Float], micIndex i: Int) {
        let rms = Self.rms(block)
        guard rms > 0 else { return }
        mics[i].floorSamples.append(20 * log10(rms))
        if mics[i].floorSamples.count > floorWindow { mics[i].floorSamples.removeFirst(mics[i].floorSamples.count - floorWindow) }
        // A counter, not the window's length: once the window is full its length never
        // changes, and pacing on it stopped the floor adapting for good.
        mics[i].floorCount += 1
        if mics[i].floorCount % 15 == 0 { updateNoiseFloor(i) }
    }

    /// The floor is a low percentile of recent levels: steadier than the minimum, which one
    /// quiet moment would drag down.
    mutating func updateNoiseFloor(_ i: Int) {
        let samples = mics[i].floorSamples
        guard samples.count >= Self.floorSamplesNeeded else { return }
        let sorted = samples.sorted()
        let percentile = max(1, min(50, config.noiseFloorPercentile))
        mics[i].noiseFloor = sorted[sorted.count * percentile / 100]
        mics[i].floorMeasured = true
    }

    // MARK: - Buffers

    /// Trim samples from the end of every mic's buffers.
    mutating func trimTrailing(_ count: Int) {
        guard count > 0 else { return }
        for i in mics.indices {
            var remaining = count
            while remaining > 0, let last = mics[i].chunk.last {
                if last.count <= remaining {
                    remaining -= last.count
                    mics[i].chunk.removeLast()
                } else {
                    mics[i].chunk[mics[i].chunk.count - 1] = Array(last.dropLast(remaining))
                    remaining = 0
                }
            }
        }
    }

    mutating func flush() -> AudioChunk {
        let chunk = AudioChunk(mics.map { ($0.name, Array($0.chunk.joined())) })
        for i in mics.indices { mics[i].chunk.removeAll() }
        return chunk
    }
}
