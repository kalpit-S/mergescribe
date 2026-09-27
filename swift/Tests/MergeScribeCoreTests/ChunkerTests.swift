import Foundation
import Testing
@testable import MergeScribeCore

private func square(_ amp: Float, _ n: Int = Chunker.blockSize) -> [Float] {
    (0..<n).map { $0 % 2 == 0 ? amp : -amp }
}

private func chunker(_ mics: [String] = ["mic1"], _ change: (inout Chunker.Config) -> Void = { _ in }) -> Chunker {
    var config = Chunker.Config()
    change(&config)
    var c = Chunker(config: config)
    for mic in mics { c.addMic(mic) }
    return c
}

@Suite struct ChunkerGoldenTests {
    static let golden = Golden.load("chunker")

    @Test func emissionsMatchPython() {
        let blockSeconds = Double(Chunker.blockSize) / 16000
        for case let .object(scenario) in Self.golden["scenarios"]!.array! {
            let mics = scenario["mics"]!.array!.map { $0.string! }
            var c = chunker(mics)
            var clock = 1000.0
            var events: [JSON] = []
            for (index, entry) in scenario["script"]!.array!.enumerated() {
                switch entry {
                case "start": c.startRecording(now: clock)
                case "stop":
                    let final = c.stopRecording().chunk
                    events.append(["at": JSON(index), "final": .array(mics.map { JSON(final.samples($0)!.count) })])
                default:
                    for (mic, amp) in zip(mics, entry.array!) {
                        if let chunk = c.feed(mic, square(Float(amp.double!)), now: clock) {
                            events.append(["at": JSON(index), "lengths": .array(mics.map { JSON(chunk.samples($0)!.count) })])
                        }
                    }
                    clock += blockSeconds
                }
            }
            #expect(JSON.array(events) == scenario["events"]!, "scenario with \(mics.count) mic(s)")
            let floors = mics.map { c.noiseFloor($0)! }
            for (swift, python) in zip(floors, scenario["floors"]!.array!.map { $0.double! }) {
                #expect(abs(swift - python) < 1e-3)
            }
        }
    }
}

@Suite struct ChunkerTests {
    let block = Double(Chunker.blockSize) / 16000

    @Test func adaptiveSilenceUsesHysteresis() {
        var c = chunker()
        c.mics[0].noiseFloor = -30
        let at = { (db: Double) in square(Float(pow(10, db / 20))) }
        let loud = c.isSilence(at(-17), micIndex: 0, updateLevel: false)     // above -30 + 12: speech
        let between = c.isSilence(at(-20), micIndex: 0, updateLevel: false)  // below start, above release (-21)
        let below = c.isSilence(at(-23), micIndex: 0, updateLevel: false)    // below release: silence
        #expect(!loud && !between && below)
    }

    @Test func fixedThresholdWhenAdaptiveIsOff() {
        var c = chunker { $0.adaptiveThreshold = false }
        let zero = c.isSilence([Float](repeating: 0, count: 1024), micIndex: 0, updateLevel: false)
        let quiet = c.isSilence(square(0.0001), micIndex: 0, updateLevel: false)
        let speech = c.isSilence(square(0.1), micIndex: 0, updateLevel: false)
        #expect(zero && quiet && !speech)
    }

    @Test func theRoomIsMeasuredBeforeTheFirstDictation() {
        // Otherwise the floor is a fixed guess, and quiet speech under it is discarded as silence.
        var c = chunker()
        for i in 0..<60 { _ = c.feed("mic1", square(0.0005), now: Double(i) * block) }   // about -66 dBFS, key up
        #expect(c.noiseFloor("mic1")! < Chunker.fallbackNoiseFloorDb - 10)
    }

    @Test func preRollIsOnlyTheMomentBeforeThePress() {
        // A mic that went quiet (AirPods idling) must not replay what it heard minutes ago.
        var c = chunker()
        for i in 0..<4 { _ = c.feed("mic1", square(0.1), now: 1000 + Double(i) * block) }
        c.startRecording(now: 1300)
        #expect(c.mics[0].chunk.isEmpty)
    }

    @Test func preRollIsKeptAndItsSpeechCounts() {
        var c = chunker()
        for i in 0..<30 { _ = c.feed("mic1", square(0.001), now: Double(i) * block) }
        for i in 30..<33 { _ = c.feed("mic1", square(0.1), now: Double(i) * block) }
        c.startRecording(now: 33 * block)
        let final = c.stopRecording()
        #expect(final.chunk.samples("mic1")!.count == 15 * Chunker.blockSize)   // one second of pre-roll
        #expect(final.hasSpeech, "a word said as the key went down is still a word")
    }

    @Test func silenceIsNeverEmittedAndTheFinalChunkHasNoSpeech() {
        // Dead air must not reach the providers: recognizers hallucinate on silence.
        var c = chunker()
        for i in 0..<50 { _ = c.feed("mic1", square(0.001), now: Double(i) * block) }
        c.startRecording(now: 50 * block)
        var emitted = 0
        for i in 50..<200 { if c.feed("mic1", square(0.001), now: Double(i) * block) != nil { emitted += 1 } }
        #expect(emitted == 0)
        let final = c.stopRecording()
        #expect(!final.hasSpeech)
    }

    @Test func aTapTooShortToHoldAWordHasNoSpeech() {
        var c = chunker { $0.prerollSeconds = 0.1 }
        for i in 0..<30 { _ = c.feed("mic1", square(0.001), now: Double(i) * block) }
        c.startRecording(now: 30 * block)
        _ = c.feed("mic1", square(0.2), now: 30 * block)
        let final = c.stopRecording()
        #expect(!final.hasSpeech, "under 0.35s")
    }

    @Test func beforeTheFloorIsKnownTheLevelGateDecides() {
        // Quiet speech at -45 dBFS would be silence against the -50 guess plus 12 dB.
        var c = chunker()
        c.startRecording(now: 0)
        for i in 0..<10 { _ = c.feed("mic1", square(0.0056), now: Double(i) * block) }
        let final = c.stopRecording()
        #expect(final.hasSpeech)
    }

    @Test func withoutSplittingOnSilenceSpeechIsStillHeard() {
        // The detector also decides whether the last chunk holds speech; turning off
        // chunking must not make every dictation look silent.
        var c = chunker { $0.chunkOnSilence = false }
        for i in 0..<30 { _ = c.feed("mic1", square(0.001), now: Double(i) * block) }
        c.startRecording(now: 30 * block)
        var emitted = 0
        for i in 30..<130 { if c.feed("mic1", square(i < 60 ? 0.1 : 0.001), now: Double(i) * block) != nil { emitted += 1 } }
        let final = c.stopRecording()
        #expect(emitted == 0, "no chunks are cut")
        #expect(final.hasSpeech)
    }

    @Test func removingThePrimaryPromotesTheNext() {
        var c = chunker(["a", "b"])
        #expect(c.primaryMic == "a")
        c.removeMic("a")
        #expect(c.primaryMic == "b" && c.micNames == ["b"])
        c.removeMic("b")
        #expect(c.primaryMic == nil)
    }

    @Test func levelMapping() {
        #expect(HUDModel.normalizeLevel(db: -30, noiseFloor: -60) == 1)
        #expect(HUDModel.normalizeLevel(db: -45, noiseFloor: -60) == 0.5)
        #expect(HUDModel.normalizeLevel(db: -70, noiseFloor: -60) == 0)
        #expect(HUDModel.normalizeLevel(db: -30, noiseFloor: nil) == 0.5)
        #expect(HUDModel.normalizeLevel(db: -.infinity, noiseFloor: nil) == 0)
    }
}
