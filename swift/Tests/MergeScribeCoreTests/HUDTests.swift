import Foundation
import Testing
@testable import MergeScribeCore

/// This frame's aurora as 0...1 floats: image[row][column] = (r, g, b, a), 1pt per pixel.
private func aurora(_ anim: HUDAnimation, width: Double = 120, track: Double? = nil) -> [[SIMD4<Double>]] {
    let (pixels, w, h) = HUDModel.aurora(anim, width: width, height: 40, track: track)
    return (0..<h).map { row in
        (0..<w).map { col in
            let o = (row * w + col) * 4
            return SIMD4(Double(pixels[o]), Double(pixels[o + 1]), Double(pixels[o + 2]), Double(pixels[o + 3])) / 255
        }
    }
}

private func alpha(_ image: [[SIMD4<Double>]]) -> [[Double]] { image.map { $0.map(\.w) } }
private func columnSums(_ a: [[Double]]) -> [Double] { (0..<a[0].count).map { c in a.reduce(0) { $0 + $1[c] } } }

/// How much brighter the brightest column in the middle of the strip is than the dimmest.
private func lift(_ anim: HUDAnimation, track: Double? = nil) -> Double {
    let columns = columnSums(alpha(aurora(anim, track: track)))[30..<90]
    return columns.max()! / columns.min()!
}

private func run(_ anim: HUDAnimation, _ seconds: Double, level: Double = 0) {
    for _ in 0..<Int(seconds * 60) { anim.step(1.0 / 60, level: level) }
}

@Suite struct HUDLevelTests {
    @Test func mapsRelativeToTheNoiseFloorAndClamps() {
        // A hot mic and a quiet one should read the same on the meter.
        #expect(HUDModel.normalizeLevel(db: -35, noiseFloor: -50) == HUDModel.normalizeLevel(db: -5, noiseFloor: -20))
        #expect(HUDModel.normalizeLevel(db: -200, noiseFloor: -50) == 0)
        #expect(HUDModel.normalizeLevel(db: 50, noiseFloor: -50) == 1)
        #expect(HUDModel.normalizeLevel(db: .nan, noiseFloor: nil) == 0)
        #expect(HUDModel.normalizeLevel(db: -60, noiseFloor: nil) == 0 && HUDModel.normalizeLevel(db: -30, noiseFloor: nil) == 0.5)
    }

    @Test func attackIsFastAndReleaseIsSlow() {
        let rise = HUDModel.smoothLevel(1, 0), fall = HUDModel.smoothLevel(0, 1)
        #expect(rise > 0.5 && fall > 0.5 && rise > 1 - fall)
        var v = 0.0
        for _ in 0..<40 { v = HUDModel.smoothLevel(0.7, v) }
        #expect(abs(v - 0.7) < 0.01)
    }
}

@Suite struct SpringTests {
    @Test func settlesOnItsTarget() {
        var s = Spring(0, (190, 17)); s.target = 1
        for _ in 0..<120 { s.step(1.0 / 60) }
        #expect(abs(s.value - 1) < 0.01 && abs(s.velocity) < 0.05)
    }

    @Test func anUnderdampedEntranceOvershootsALittle() {
        var s = Spring(0, HUDModel.presenceSpring); s.target = 1
        var peak = 0.0
        for _ in 0..<120 { s.step(1.0 / 60); peak = max(peak, s.value) }
        #expect(peak > 1 && peak < 1.12)
    }

    @Test func retargetingKeepsMomentum() {
        var s = Spring(0, (190, 17)); s.target = 1
        for _ in 0..<6 { s.step(1.0 / 60) }
        let rising = s.value
        s.target = 0
        s.step(1.0 / 60)
        #expect(s.value > rising)
    }
}

@Suite struct HUDAnimationTests {
    @Test func appearsThenSpringsAway() {
        let anim = HUDAnimation()
        anim.setStatus("recording"); run(anim, 0.8)
        #expect(anim.alpha > 0.99 && !anim.hidden)
        anim.setStatus("idle"); run(anim, 0.8)
        #expect(anim.hidden)
    }

    @Test func silenceBreathesAndSpeechSwells() {
        func light(_ level: Double) -> Double {
            let anim = HUDAnimation(); anim.setStatus("recording"); run(anim, 0.5, level: level)
            return alpha(aurora(anim))[..<20].reduce(0) { $0 + $1.reduce(0, +) }
        }
        let quiet = light(0), loud = light(1)
        let anim = HUDAnimation(); anim.setStatus("recording"); run(anim, 0.5)
        #expect(alpha(aurora(anim)).reduce(0) { $0 + $1.reduce(0, +) } > 0, "silence should breathe, not go dark")
        #expect(loud > 8 * quiet, "a voice sends the rays up the capsule")
    }

    @Test func theAuroraFadesToNothingAtBothEnds() {
        let anim = HUDAnimation(); anim.setStatus("recording"); run(anim, 0.5, level: 1)
        let a = alpha(aurora(anim))
        #expect(a.map { $0.first! }.max()! < 0.02 && a.map { $0.last! }.max()! < 0.02)
        #expect(a.map { $0[60] }.max()! > 0.5)
    }

    @Test func releasingSettlesTheCurtainsIntoOneHem() {
        // While listening each curtain's hem wanders; once merged they lie on one line.
        func wander(_ anim: HUDAnimation) -> Double {
            let a = alpha(aurora(anim))
            let rows = (30..<90).map { c in Double((0..<a.count).max { a[$0][c] < a[$1][c] }!) }
            let mean = rows.reduce(0, +) / Double(rows.count)
            return (rows.map { pow($0 - mean, 2) }.reduce(0, +) / Double(rows.count)).squareRoot()
        }
        let anim = HUDAnimation(); anim.setStatus("recording"); run(anim, 1, level: 0.7)
        #expect(wander(anim) > 1)
        anim.setStatus("processing"); run(anim, 1.5)
        #expect(wander(anim) < 0.3)
    }

    @Test func thePulseStartsOffTheLeftEdge() {
        let anim = HUDAnimation(); anim.setStatus("recording"); anim.setStatus("processing"); anim.step(1.0 / 60)
        #expect(anim.pulsePosition < 0)
    }

    @Test func widensForWordsAndANewRecordingStartsCompact() {
        let anim = HUDAnimation(); anim.setStatus("recording"); anim.setTranscript("so I was thinking"); run(anim, 0.6)
        #expect(anim.width.value > 0.95)
        anim.setStatus("idle"); anim.setStatus("recording")      // pressed again before the fade finished
        #expect(anim.transcript == "")
        run(anim, 0.6)
        #expect(anim.width.value < 0.05)
    }

    @Test func aStalledFrameDoesNotFlingAnything() {
        let anim = HUDAnimation(); anim.setStatus("recording"); anim.step(5, level: 1)
        #expect(anim.presence.value >= 0 && anim.presence.value <= 1)
        #expect(alpha(aurora(anim))[0].max()! < 0.3, "the rays should not be flung to the top")
    }
}

@Suite struct HonestStrandTests {
    static let streams = ["parakeet/built-in", "mai/built-in", "mai/solocast"]

    func recording(_ streams: [String] = streams) -> HUDAnimation {
        let anim = HUDAnimation(); anim.setStatus("recording")
        anim.streamsPlanned(session: "s1", streams: streams)
        run(anim, 0.5, level: 0.6)
        return anim
    }

    func strand(_ anim: HUDAnimation, _ key: String) -> HUDAnimation.Strand { anim.strands.first { $0.key == key }! }

    @Test func oneStrandPerStream() {
        #expect(recording(Array(Self.streams.prefix(2))).strands.count == 2)
        #expect(recording().strands.count == 3)
    }

    @Test func aChunkLandingMidRecordingFlashesWithoutJoining() {
        let anim = recording()
        anim.streamLanded(session: "s1", stream: "mai/built-in", final: false)
        #expect(strand(anim, "mai/built-in").flash == 1)
        run(anim, 0.5)
        #expect(strand(anim, "mai/built-in").join.value < 0.05)
    }

    @Test func finalResultsJoinTheLineOneAtATime() {
        let anim = recording(); anim.setStatus("processing")
        anim.streamLanded(session: "s1", stream: "parakeet/built-in", final: true)
        run(anim, 0.6)
        #expect(strand(anim, "parakeet/built-in").join.value > 0.95)
        #expect(strand(anim, "mai/built-in").join.value < 0.05)
        #expect(anim.correcting < 0.05)
    }

    @Test func aStreamLeftBehindFadesAndOneMissedWhileRecordingDims() {
        let processing = recording(); processing.setStatus("processing")
        processing.streamDropped(session: "s1", stream: "mai/solocast"); run(processing, 0.6)
        #expect(strand(processing, "mai/solocast").presence.value < 0.05)
        let listening = recording()
        listening.streamDropped(session: "s1", stream: "mai/solocast"); run(listening, 0.6)
        let dimmed = strand(listening, "mai/solocast").presence.value
        #expect(dimmed > 0.2 && dimmed < 0.5)
    }

    @Test func transcriptionDoneJoinsTheRestAndStartsThePulse() {
        let anim = recording(); anim.setStatus("processing")
        anim.streamLanded(session: "s1", stream: "parakeet/built-in", final: true)
        anim.streamDropped(session: "s1", stream: "mai/solocast")
        anim.transcriptionDone(session: "s1")
        run(anim, 0.8)
        #expect(strand(anim, "mai/built-in").join.value > 0.95)
        #expect(strand(anim, "mai/solocast").presence.value < 0.05)
        #expect(anim.correcting > 0.95)
    }

    @Test func aFinishedSessionReportingLateIsIgnored() {
        let anim = recording()
        anim.setStatus("recording")
        anim.streamsPlanned(session: "s2", streams: Self.streams)
        anim.streamLanded(session: "s1", stream: "parakeet/built-in", final: true)
        anim.transcriptionDone(session: "s1")
        anim.partialText(session: "s1", text: "old words")
        run(anim, 0.5)
        #expect(strand(anim, "parakeet/built-in").join.value < 0.05)
        #expect(anim.correcting == 0 && anim.transcript == "")
    }

    @Test func withoutReportsReleasingMergesEverything() {
        let anim = HUDAnimation(); anim.setStatus("recording"); anim.setStatus("processing"); run(anim, 0.8)
        #expect(anim.strands.allSatisfy { $0.join.value > 0.95 } && anim.correcting > 0.95)
    }
}

@Suite struct TailTextTests {
    @Test func shortAndEmpty() {
        #expect(HUDModel.tailText("hello there") == "hello there")
        #expect(HUDModel.tailText("") == "" && HUDModel.tailText(nil) == "" && HUDModel.tailText("   \n  ") == "")
        #expect(HUDModel.tailText("  one   two \n three ") == "one two three")
    }

    @Test func keepsTheEndWithoutCuttingAWord() {
        let tail = HUDModel.tailText((0..<100).map { "word\($0)" }.joined(separator: " "))
        #expect(tail.contains("word99") && !tail.contains("word0 ") && tail.hasPrefix("…"))
        let body = HUDModel.tailText(String(repeating: "supercalifragilistic ", count: 20), budget: 30)
            .trimmingCharacters(in: CharacterSet(charactersIn: "… "))
        #expect(["supercalifragilistic", ""].contains(body.split(separator: " ").first.map(String.init) ?? ""))
        for n in [10, 40, 200] { #expect(HUDModel.tailText(String(repeating: "a bb ccc dddd ", count: 50), budget: n).count <= n + 2) }
    }
}

@Suite struct SkippedAndDiscardedTests {
    func processing() -> HUDAnimation {
        let anim = HUDAnimation(); anim.setStatus("recording")
        anim.streamsPlanned(session: "s1", streams: ["parakeet/built-in", "mai/built-in"])
        run(anim, 0.5, level: 0.6)
        anim.setStatus("processing"); anim.transcriptionDone(session: "s1")
        return anim
    }

    private func total(_ anim: HUDAnimation) -> Double { alpha(aurora(anim)).reduce(0) { $0 + $1.reduce(0, +) } }

    @Test func aSkippedCorrectionNeverPulsesAndFlashesThenSettles() {
        let anim = processing(); anim.correctionSkipped(session: "s1")
        let bright = total(anim)
        run(anim, 1)
        #expect(anim.pulse.value < 0.2 && anim.skipped)
        #expect(bright > total(anim) * 1.5, "the flash should be visible, then fade")
    }

    @Test func aStaleSessionCannotFlashAndANewRecordingClearsIt() {
        let anim = processing(); anim.correctionSkipped(session: "other")
        #expect(!anim.skipped)
        anim.correctionSkipped(session: "s1"); anim.setStatus("idle"); run(anim, 1); anim.reset()
        #expect(!anim.skipped && anim.flash == 0)
    }

    @Test func aCalledOffDictationDrawsItselfInAndANewRecordingReopensIt() {
        let anim = HUDAnimation(); anim.setStatus("recording")
        anim.streamsPlanned(session: "s1", streams: ["parakeet/built-in"])
        anim.setStatus("processing"); anim.dictationDiscarded(session: "s1"); run(anim, 0.75)
        #expect(anim.discarded && anim.collapse.value > 0.9)
        anim.setStatus("recording")
        #expect(anim.collapse.value == 0 && !anim.discarded)
    }
}

@Suite struct TokenRippleTests {
    func correcting() -> HUDAnimation {
        let anim = HUDAnimation(); anim.setStatus("recording")
        anim.streamsPlanned(session: "s1", streams: ["parakeet/built-in"])
        anim.setStatus("processing"); anim.transcriptionDone(session: "s1")
        for _ in 0..<20 { anim.step(1.0 / 60) }
        return anim
    }

    @Test func aTokenSendsARippleThatTravels() {
        let anim = correcting(); anim.tokenTyped(session: "s1")
        let start = anim.ripples[0]
        for _ in 0..<20 { anim.step(1.0 / 60) }
        #expect(anim.ripples[0] > start)
    }

    @Test func aStalledStreamGoesQuiet() {
        // No tokens, no movement: a stalled correction must look stalled.
        let anim = correcting(); anim.tokenTyped(session: "s1")
        run(anim, 3)
        #expect(anim.ripples.isEmpty)
        #expect(lift(anim) < 1.05, "the hem should be evenly lit, with nothing travelling")
        #expect(anim.pulseTrack < 0, "nothing to shimmer once the ripples are gone")
    }

    @Test func aBurstIsSpreadOutAndCapped() {
        let anim = correcting()
        for _ in 0..<5 { anim.tokenTyped(session: "s1") }
        #expect(Set(anim.ripples).count == 5 && anim.ripples.max()! - anim.ripples.min()! > 0.1)
        for _ in 0..<50 { anim.tokenTyped(session: "s1") }
        #expect(anim.ripples.count <= 8)
    }

    @Test func anUnreportedStreamStillSweeps() {
        let anim = correcting()
        for _ in 0..<30 { anim.step(1.0 / 60) }
        #expect(lift(anim) > 1.3)
    }

    @Test func aWordsLightRunsTheTrackTheShimmerDoes() {
        let anim = correcting(); anim.tokenTyped(session: "s1")
        for _ in 0..<7 { anim.step(1.0 / 60) }
        let p = anim.ripples[0]
        for track in [480.0, 800.0] {
            let columns = columnSums(alpha(aurora(anim, track: track)))
            let brightest = Double(columns.indices.max { columns[$0] < columns[$1] }!)
            #expect(abs(brightest - p * track) < 3)
        }
        let first = anim.pulseTrack
        for _ in 0..<20 { anim.step(1.0 / 60) }
        #expect(anim.pulseTrack > first)
    }

    @Test func aStaleSessionCannotRipple() {
        let anim = correcting(); anim.tokenTyped(session: "other")
        #expect(anim.ripples.isEmpty)
    }
}

@Suite struct CapsuleAndLightTests {
    func listening(level: Double = 0, frames: Int = 40) -> HUDAnimation {
        let anim = HUDAnimation(); anim.setStatus("recording")
        anim.streamsPlanned(session: "s1", streams: ["parakeet/built-in"])
        for _ in 0..<frames { anim.step(1.0 / 60, level: level) }
        return anim
    }

    @Test func itOpensFromADotAndTheContentsWait() {
        let anim = HUDAnimation(); anim.setStatus("recording")
        #expect(anim.morph == 0)
        anim.step(1.0 / 60)
        #expect(anim.morph > 0 && anim.morph < 0.5 && anim.contentAlpha == 0)
        for _ in 0..<60 { anim.step(1.0 / 60) }
        #expect(abs(anim.morph - 1) < 0.05 && anim.contentAlpha > 0.95)
    }

    @Test func theEdgeIsNearlyDarkInSilenceAndLightsWithAVoice() {
        let quiet = listening(level: 0).edgeLight().steady, loud = listening(level: 1).edgeLight().steady
        #expect(quiet < 0.05 && loud > 5 * quiet)
    }

    @Test func theSweepRunsFasterOnceReleased() {
        let a = listening(frames: 60)
        var start = a.edgePhase
        for _ in 0..<60 { a.step(1.0 / 60) }
        let drift = (a.edgePhase - start + 1).truncatingRemainder(dividingBy: 1)
        let b = listening(frames: 60); b.setStatus("processing")
        for _ in 0..<40 { b.step(1.0 / 60) }
        start = b.edgePhase
        for _ in 0..<60 { b.step(1.0 / 60) }
        #expect((b.edgePhase - start + 1).truncatingRemainder(dividingBy: 1) > 3 * drift)
    }

    @Test func aSkippedCorrectionFlaresAndADiscardedOnePutsTheLightOut() {
        let skipped = listening(frames: 60); skipped.setStatus("processing")
        skipped.transcriptionDone(session: "s1"); skipped.correctionSkipped(session: "s1")
        let light = skipped.edgeLight()
        #expect(light.steady > 0.8 && light.sweep == 0)
        let discarded = listening(level: 0.8, frames: 60); discarded.dictationDiscarded(session: "s1")
        for _ in 0..<60 { discarded.step(1.0 / 60) }
        let out = discarded.edgeLight()
        #expect(out.steady < 0.05 && out.sweep < 0.05 && discarded.contentAlpha < 0.05)
    }

    @Test func theAuroraClimbsWithYourVoice() {
        func highestLitRow(_ anim: HUDAnimation) -> Int { alpha(aurora(anim)).firstIndex { $0.contains { $0 > 0.25 } }! }
        #expect(highestLitRow(listening(level: 1)) + 10 < highestLitRow(listening(level: 0)))
    }
}

@Suite struct AuroraImageTests {
    func merged() -> HUDAnimation {
        let anim = HUDAnimation(); anim.setStatus("recording"); anim.setStatus("processing"); run(anim, 1.5)
        return anim
    }

    @Test func paddingAndScale() {
        let anim = HUDAnimation(); anim.setStatus("recording"); run(anim, 0.5, level: 0.5)
        let (pixels, w, h) = HUDModel.aurora(anim, width: 112, height: 40, pad: 12)
        #expect(w == 136 && h == 64 && pixels.count == 136 * 64 * 4)
        #expect(pixels[..<(12 * w * 4)].allSatisfy { $0 == 0 }, "the pad is clear, for the glow")
        #expect((0..<h).allSatisfy { row in pixels[(row * w * 4)..<(row * w * 4 + 48)].allSatisfy { $0 == 0 } })
        let scaled = HUDModel.aurora(anim, width: 112, height: 40, scale: 2)
        #expect(scaled.width == 224 && scaled.height == 80)
    }

    @Test func itIsPremultipliedAndTheSettledHemIsMint() {
        let image = aurora(merged())
        #expect(image.allSatisfy { $0.allSatisfy { $0.x <= $0.w + 1.0 / 255 && $0.y <= $0.w + 1.0 / 255 && $0.z <= $0.w + 1.0 / 255 } })
        let column = image.map { $0[60] }
        let brightest = column.max { $0.w < $1.w }!
        #expect(brightest.y > brightest.z && brightest.z > brightest.x)
    }

    @Test func nothingIsDrawnBeforeTheCapsuleOpens() {
        let anim = HUDAnimation(); anim.setStatus("recording"); anim.step(1.0 / 60)
        #expect(alpha(aurora(anim)).allSatisfy { $0.allSatisfy { $0 == 0 } })
    }
}
