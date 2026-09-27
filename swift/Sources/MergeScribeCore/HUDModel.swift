import Foundation

/// A value that chases its target with momentum.
///
/// Semi-implicit Euler, stable at frame-rate steps. Retargeting mid-flight keeps the
/// velocity, so an interrupted animation bends toward its new goal instead of restarting.
public struct Spring: Sendable {
    public var value: Double
    public var velocity = 0.0
    public var target: Double
    let stiffness: Double
    let damping: Double

    public init(_ value: Double, _ params: (stiffness: Double, damping: Double)) {
        self.value = value
        target = value
        stiffness = params.stiffness
        damping = params.damping
    }

    public mutating func step(_ dt: Double) {
        velocity += (stiffness * (target - value) - damping * velocity) * dt
        value += velocity * dt
    }

    public mutating func snap(_ value: Double) {
        self.value = value
        target = value
        velocity = 0
    }
}

/// The HUD's pure parts: level mapping, springs, the animation state, and the aurora.
public enum HUDModel {
    /// dB above the mic's noise floor that counts as full scale. Speech typically sits
    /// 12-30 dB above the floor, so 30 keeps normal talking in the upper half of the meter.
    static let levelSpanDb = 30.0

    // Springs as (stiffness, damping). Damping ratio is damping / (2 * sqrt(stiffness)):
    // below 1 overshoots a little, which is what makes an entrance feel physical.
    public static let presenceSpring = (stiffness: 190.0, damping: 17.0)   // 0.62: pops in
    public static let widthSpring = (stiffness: 210.0, damping: 25.0)      // 0.86: widens without wobble
    public static let mergeSpring = (stiffness: 80.0, damping: 18.0)       // 1.0: converges without bounce

    /// Asymmetric ballistics, borrowed from analogue VU meters: jump to a peak almost at
    /// once, fall away slowly. A meter that tracks RMS both ways reads as noise, not a voice.
    static let levelAttack = 0.65
    static let levelRelease = 0.16

    /// Only the tail of the transcript is shown: a wall of your own words appearing while
    /// you are still speaking is a good way to lose your thread.
    public static let tailChars = 58

    /// A dB reading as 0...1 for the meter: relative to the mic's own floor when it is known,
    /// so the meter reads the same on a quiet built-in mic and a hot condenser; a fixed
    /// -60...0 dB range before then.
    public static func normalizeLevel(db: Double, noiseFloor: Double?) -> Double {
        guard db.isFinite else { return 0 }
        let level: Double
        if let noiseFloor, noiseFloor.isFinite {
            level = (db - noiseFloor) / levelSpanDb
        } else {
            level = (db + 60) / 60
        }
        return max(0, min(1, level))
    }

    /// Fast attack, slow release: consonants register, and the meter doesn't flicker
    /// between syllables.
    public static func smoothLevel(_ current: Double, _ previous: Double) -> Double {
        let rate = current > previous ? levelAttack : levelRelease
        return previous + (current - previous) * rate
    }

    /// The last few words of the transcript, trimmed at a word boundary; "" keeps the HUD
    /// in its compact form.
    public static func tailText(_ text: String?, budget: Int = tailChars) -> String {
        let text = (text ?? "").collapsingWhitespace()
        guard text.count > budget else { return text }
        var clipped = String(text.suffix(budget))
        if let space = clipped.firstIndex(of: " "), clipped.distance(from: clipped.startIndex, to: space) < budget / 2 {
            clipped = String(clipped[clipped.index(after: space)...])
        }
        return "… " + clipped
    }
}

func clamp01(_ x: Double) -> Double { max(0, min(1, x)) }
func lerp(_ a: Double, _ b: Double, _ t: Double) -> Double { a + (b - a) * t }

/// An angle folded into -pi..pi, so phases converge the short way round.
func wrapAngle(_ angle: Double) -> Double {
    let twoPi = 2 * Double.pi
    var a = (angle + .pi).truncatingRemainder(dividingBy: twoPi)
    if a < 0 { a += twoPi }
    return a - .pi
}

/// Everything on the HUD that moves, advanced by step().
///
/// presence: 0 hidden, 1 shown. width: 0 compact, 1 wide enough for the transcript.
/// work: 0 listening, 1 released and transcribing. pulse: 0 still transcribing, 1 correcting.
///
/// Each strand is a stream the session reported (a recognizer on a mic). While you talk they
/// move with your voice and flash when a chunk comes back. After release each joins the line
/// as its final result lands, or fades if it is dropped; when every stream is in, the pulse
/// starts, and each word the correction types sends light along the line and through the words.
public final class HUDAnimation {
    static let visibleStatuses: Set<String> = ["recording", "processing"]
    static let maxStep = 1.0 / 20           // a stalled main thread shouldn't fling the springs
    /// Each strand drifts at its own speed, alternate ones the other way.
    static let strandSpeeds = [5.3, -4.1, 6.4, -5.6]
    static let defaultStreams = 3
    static let strandDimmed = 0.35          // missed a chunk while recording; may yet catch up
    static let flashDecay = 0.3             // seconds for a landing's flash to fade
    static let mergedSpeed = 11.0
    static let pulsePeriod = 1.5            // seconds per pass, when nothing reports tokens
    static let pulseLeadIn = 0.25           // lets the last strand arrive before the pulse sets off
    static let pulseSpeed = 0.9             // how fast one token's ripple crosses the line
    static let maxRipples = 8
    /// Correction models stream in bursts, so ripples are spaced to move at the rate words
    /// land, not packets.
    static let rippleSpacing = 0.06

    public struct Strand {
        public let key: String
        let speed: Double
        public var phase: Double
        public var join = Spring(0, HUDModel.mergeSpring)       // 0 its own curtain, 1 part of the hem
        public var presence = Spring(1, HUDModel.mergeSpring)   // 0 dropped, 1 present
        public var flash = 0.0                                  // brightens when a chunk lands

        init(_ key: String, _ index: Int) {
            self.key = key
            speed = HUDAnimation.strandSpeeds[index % HUDAnimation.strandSpeeds.count]
            phase = Double(index) * 2.1
        }
    }

    public private(set) var status = "idle"
    public private(set) var transcript = ""
    public private(set) var clock = 0.0
    public private(set) var level = 0.0
    public var presence = Spring(0, HUDModel.presenceSpring)
    public var width = Spring(0, HUDModel.widthSpring)
    public var work = Spring(0, HUDModel.mergeSpring)
    public var pulse = Spring(0, HUDModel.mergeSpring)
    public var collapse = Spring(0, HUDModel.mergeSpring)
    public var strands: [Strand] = (0..<HUDAnimation.defaultStreams).map { Strand("", $0) }
    public private(set) var skipped = false     // the transcript needed no correcting
    public private(set) var flash = 0.0         // the moment that was decided
    public private(set) var discarded = false   // the speaker called the dictation off
    public private(set) var edgePhase = 0.0     // where the light running round the edge has got to
    public private(set) var ripples: [Double] = []
    public private(set) var tokensSeen = false
    private var session: String?
    private var mergedPhase = 0.0
    private var pulseClock = 0.0
    private var correctingStarted = false

    public init() {}

    // MARK: - What the session reports

    public var visible: Bool { Self.visibleStatuses.contains(status) }

    public func setStatus(_ status: String) {
        if status == "recording" {
            // A new recording owns the pill, even if the last one is still fading out.
            session = nil
            setTranscript("")
            work.target = 0
            pulse.snap(0)
            correctingStarted = false
            skipped = false
            discarded = false
            flash = 0
            ripples = []
            tokensSeen = false
            collapse.snap(0)
            for i in strands.indices {
                strands[i].join.target = 0
                strands[i].presence.target = 1
            }
        } else if status == "processing" {
            work.target = 1
            // Nobody is reporting streams, so nothing will say when they're done.
            if session == nil { startCorrecting() }
        }
        self.status = status
    }

    public func streamsPlanned(session: String, streams: [String]) {
        self.session = session
        if strands.map(\.key) != streams {
            strands = streams.enumerated().map { Strand($1, $0) }
        }
    }

    public func streamLanded(session: String, stream: String, final: Bool) {
        guard let i = strandIndex(session, stream) else { return }
        strands[i].flash = 1
        strands[i].presence.target = 1
        if final { strands[i].join.target = 1 }
    }

    public func streamDropped(session: String, stream: String) {
        guard let i = strandIndex(session, stream) else { return }
        strands[i].presence.target = status == "processing" ? 0 : Self.strandDimmed
    }

    public func transcriptionDone(session: String) {
        if session == self.session { startCorrecting() }
    }

    /// A word just landed on screen: send a ripple down the line.
    public func tokenTyped(session: String) {
        guard session == self.session else { return }
        tokensSeen = true
        guard ripples.count < Self.maxRipples else { return }
        // The newest ripple is the one furthest left, so queue behind the smallest position;
        // otherwise a burst stacks into a single lump.
        let newest = ripples.min() ?? 1
        ripples.append(min(0, newest - Self.rippleSpacing))
    }

    /// Nothing will be typed: draw the line inward rather than fade it wide.
    public func dictationDiscarded(session: String) {
        guard session == self.session else { return }
        discarded = true
        collapse.target = 1
    }

    /// The transcript was typed as it stood: no pulse, one bright flash.
    public func correctionSkipped(session: String) {
        guard session == self.session else { return }
        skipped = true
        flash = 1
        pulse.target = 0
    }

    public func partialText(session: String, text: String) {
        if session == self.session { setTranscript(text) }
    }

    public func setTranscript(_ text: String) {
        transcript = text
        width.target = text.isEmpty ? 0 : 1
    }

    private func strandIndex(_ session: String, _ stream: String) -> Int? {
        guard session == self.session else { return nil }   // a finished session reporting late
        return strands.firstIndex { $0.key == stream }
    }

    private func startCorrecting() {
        // Streams that neither landed nor dropped were cancelled by consensus: the others
        // agreed for them, so they join too.
        for i in strands.indices where strands[i].presence.target > 0 {
            strands[i].presence.target = 1
            strands[i].join.target = 1
        }
        if !correctingStarted {
            correctingStarted = true
            pulse.target = 1
            pulseClock = -Self.pulseLeadIn
        }
    }

    // MARK: - Time

    public func step(_ dt: Double, level input: Double = 0) {
        let dt = max(0, min(dt, Self.maxStep))
        clock += dt
        presence.target = visible ? 1 : 0
        presence.step(dt); width.step(dt); work.step(dt); pulse.step(dt)
        level = HUDModel.smoothLevel(status == "recording" ? input : 0, level)

        // Each strand drifts at its own speed, faster while it transcribes; as it joins, its
        // speed converges and its phase is pulled onto the line.
        let busy = lerp(1, 1.5, clamp01(work.value))
        mergedPhase += Self.mergedSpeed * dt
        let fade = exp(-dt / Self.flashDecay)
        for i in strands.indices {
            strands[i].join.step(dt)
            strands[i].presence.step(dt)
            strands[i].flash *= fade
            let m = clamp01(strands[i].join.value)
            strands[i].phase += lerp(strands[i].speed * busy, Self.mergedSpeed, m) * dt
            strands[i].phase += wrapAngle(mergedPhase - strands[i].phase) * min(1, 8 * dt) * m
        }
        flash *= fade
        collapse.step(dt)
        // The edge light drifts while listening and runs while the pipeline works; it stops
        // when there is nothing left to wait for.
        let lapsPerSecond = skipped || discarded ? 0 : lerp(0.08, 0.55, clamp01(work.value))
        edgePhase = (edgePhase + lapsPerSecond * dt).truncatingRemainder(dividingBy: 1)
        // Ripples travel on their own; nothing moves while the stream is quiet.
        ripples = ripples.filter { $0 < 1.25 }.map { $0 + Self.pulseSpeed * dt }
        if correctingStarted && !skipped { pulseClock += dt }
    }

    /// Back to a blank slate once fully hidden.
    public func reset() {
        let keep = status
        let fresh = HUDAnimation()
        presence = fresh.presence; width = fresh.width; work = fresh.work; pulse = fresh.pulse
        collapse = fresh.collapse; strands = fresh.strands; transcript = ""; clock = 0; level = 0
        skipped = false; flash = 0; discarded = false; edgePhase = 0; ripples = []; tokensSeen = false
        session = nil; mergedPhase = 0; pulseClock = 0; correctingStarted = false
        status = keep
    }

    // MARK: - What the view draws

    /// Faded out with nothing left to show: the panel can be ordered out.
    public var hidden: Bool { !visible && presence.value < 0.01 }
    public var alpha: Double { clamp01(presence.value) }
    /// How far the dot has opened into a capsule. Unclamped on purpose: the presence spring's
    /// small overshoot is what makes the capsule land with a little give.
    public var morph: Double { presence.value }
    /// The strands and words arrive once the capsule has mostly opened.
    public var contentAlpha: Double { clamp01((presence.value - 0.35) / 0.55) * (1 - clamp01(collapse.value)) }
    /// How far below its resting place the capsule is, in points.
    public var rise: Double { (1 - min(presence.value, 1)) * 8 }
    public var correcting: Double { clamp01(pulse.value) }

    /// The light along the capsule's edge: steady (follows your voice while listening,
    /// flares when a dictation is typed without correction), where a brighter sweep has got
    /// to going round (0...1), and how bright that sweep is.
    public func edgeLight() -> (steady: Double, head: Double, sweep: Double) {
        let working = clamp01(work.value)
        let steady = lerp(0.02 + 0.5 * pow(level, 1.5), 0.05, working) + 0.9 * flash
        let sweep = skipped ? 0 : lerp(0.4, 0.9, working)
        let fade = 1 - clamp01(collapse.value)
        return (steady * fade, edgePhase, sweep * fade)
    }

    /// Where the correcting pulse is, as a fraction of the content: it starts off the left
    /// edge and exits off the right, so each pass fades in and out.
    public var pulsePosition: Double {
        let progress = (max(pulseClock, 0) / Self.pulsePeriod).truncatingRemainder(dividingBy: 1)
        return -0.2 + 1.4 * progress
    }

    /// Where the shimmer over the words sits: with the furthest-along ripple, so the light on
    /// the line and the shimmer over the words tell one story.
    public var pulseTrack: Double {
        tokensSeen ? (ripples.max() ?? -1) : pulsePosition
    }
}

// MARK: - The aurora

extension HUDModel {
    /// The curtains' colours: green, cyan, violet, and amber for a fourth stream.
    static let curtains: [SIMD3<Float>] = [SIMD3(0.30, 1.00, 0.62), SIMD3(0.22, 0.78, 1.00),
                                           SIMD3(0.70, 0.46, 1.00), SIMD3(0.95, 0.62, 0.35)]
    static let crown = SIMD3<Float>(0.95, 0.42, 0.85)       // tall rays go magenta at the top
    public static let settled = SIMD3<Float>(0.78, 1.00, 0.93) // the hem once the streams agree
    static let hemBelowCentre = 7.0     // pt: the hem sits low so the rays have room to climb
    static let rayGrain = 0.16          // rays per pt, roughly
    static let rippleReach = 11.0       // pt: how far along the hem one word's light spreads

    /// Smooth 1-D value noise in [-1, 1].
    static func noise1(_ x: Float, _ seed: Float) -> Float {
        let i = x.rounded(.down), f = x - i
        func hash(_ n: Float) -> Float {
            let v = sin((n + seed * 17.13) * 127.1 + 311.7) * 43758.5453
            return v - v.rounded(.down)
        }
        let a = hash(i), b = hash(i + 1)
        return (a + (b - a) * f * f * (3 - 2 * f)) * 2 - 1
    }

    /// The vertical striation of a curtain along x, drifting over time.
    static func rays(_ x: Float, _ t: Float, _ seed: Float) -> Float {
        var v: Float = 0, amp: Float = 0.5, freq: Float = 1
        for octave in 0..<3 {
            v += amp * noise1(x * freq + t * (0.6 + 0.3 * Float(octave)), seed + Float(octave))
            amp *= 0.5
            freq *= 2.1
        }
        return v
    }

    /// The aurora for this frame as premultiplied RGBA bytes, rows top to bottom.
    ///
    /// Each stream is a curtain of northern lights: a bright hem low in the capsule with rays
    /// climbing from it, taller when you are louder. `width` x `height` is the strip it lives
    /// in; `pad` of clear margin is added on every side so a blurred copy can glow into it.
    /// `track` is the length of the whole content (strip and words) that travelling light
    /// crosses, so a word's light runs along the hem and on into the words.
    public static func aurora(_ anim: HUDAnimation, width widthPt: Double, height heightPt: Double,
                              pad padPt: Double = 0, scale: Double = 1, track trackPt: Double? = nil)
        -> (pixels: [UInt8], width: Int, height: Int) {
        let w = max(1, Int((widthPt * scale).rounded())), h = max(1, Int((heightPt * scale).rounded()))
        let pad = Int((padPt * scale).rounded())
        let fullW = w + 2 * pad, fullH = h + 2 * pad
        var pixels = [UInt8](repeating: 0, count: fullW * fullH * 4)
        let contentGain = Float(clamp01(anim.contentAlpha * 1.2))
        guard contentGain > 0 else { return (pixels, fullW, fullH) }

        let xs = (0..<w).map { (Double($0) + 0.5) / scale }
        let ys = (0..<h).map { (Double($0) + 0.5) / scale }
        func smooth(_ t: Double) -> Double { let c = clamp01(t); return c * c * (3 - 2 * c) }
        let us = xs.map { $0 / max(1, widthPt) }
        let ends = us.map { smooth($0 / 0.18) * smooth((1 - $0) / 0.18) }
        let rest = heightPt / 2 + hemBelowCentre
        let work = clamp01(anim.work.value)
        // Once anything reports tokens the light follows them, and goes still when they stop
        // (a stalled stream should look stalled). Only an unreported correction sweeps.
        let track = trackPt ?? widthPt
        let (travelling, gain) = anim.tokensSeen ? (anim.ripples, 1.0) : ([anim.pulsePosition], anim.correcting)
        let ripple = xs.map { x in
            min(1, travelling.reduce(0) { $0 + exp(-pow((x - $1 * track) / rippleReach, 2)) }) * gain
        }
        let present = anim.strands.map { clamp01($0.presence.value) }
        let shown = max(1, present.filter { $0 > 0.05 }.count)
        let joined = zip(anim.strands, present).reduce(0) { $0 + clamp01($1.0.join.value) * $1.1 } / max(1e-3, present.reduce(0, +))
        let share = 1 / (1 + Double(shown - 1) * joined)

        var light = [SIMD3<Float>](repeating: .zero, count: w * h)
        for (i, strand) in anim.strands.enumerated() {
            let join = clamp01(strand.join.value), here = present[i]
            guard here >= 0.01 else { continue }
            let wander = 0.6 + 0.4 * sin(anim.clock * (1.1 + 0.5 * Double(i)) + 2 * Double(i))
            let amp = (2.5 + 7.5 * anim.level * wander) * (1 - join) + 0.8 * join
            let freq = [1.1, 1.6, 0.8, 1.3][i % 4]
            let baseBright = ((0.5 + 0.7 * anim.level + 1.1 * strand.flash) * (1 - work) + 0.8 * work) * here * share
            let curtain = curtains[i % 4]
            let settledMix = Float(0.8 * join)
            for col in 0..<w {
                let u = us[col], x = xs[col]
                var hem = rest + amp * sin(2 * .pi * freq * u + strand.phase * 0.35
                                           + 0.9 * sin(2 * .pi * 0.6 * u + anim.clock * 0.8 + Double(i)))
                hem = hem * (1 - join) + rest * join
                let r = Double(rays(Float(x * rayGrain + Double(i) * 9.7), Float(anim.clock * 0.9), Float(i)))
                let rayv = (0.55 + 0.45 * r) * (1 - join) + 0.9 * join
                let reach = (4 + 16 * anim.level + 9 * strand.flash) * (1 - 0.8 * join) + 3 * ripple[col] * join
                let bright = baseBright * (1 + 1.1 * ripple[col] * join) * ends[col]
                guard bright > 0 else { continue }
                for row in 0..<h {
                    let d = hem - ys[row]                            // > 0 above the hem
                    let shape = (d > 0 ? exp(-max(d, 0) / max(reach, 0.8)) * rayv : exp(-pow(min(d, 0) / 1.6, 2)))
                        + exp(-pow(d / 1.1, 2)) * (0.55 + 0.45 * join)
                    let tall = Float(clamp01(d / (reach * 2.2 + 1)))
                    var colour = curtain * (1 - 0.45 * tall) + crown * (0.45 * tall)
                    colour = colour * (1 - settledMix) + settled * settledMix
                    light[row * w + col] += colour * Float(shape * bright * 0.55)
                }
            }
        }
        let flashLight = Float(0.3 * anim.flash)
        for row in 0..<h {
            for col in 0..<w {
                var l = light[row * w + col] + settled * flashLight * Float(ends[col])
                // Saturate gently, never clip.
                l = SIMD3(1 - exp(-l.x * 1.6), 1 - exp(-l.y * 1.6), 1 - exp(-l.z * 1.6))
                let a = max(l.x, l.y, l.z)
                let rgba = [l.x, l.y, l.z, a].map { UInt8(max(0, min(1, $0 * contentGain)) * 255 + 0.5) }
                let offset = ((row + pad) * fullW + col + pad) * 4
                pixels[offset] = rgba[0]; pixels[offset + 1] = rgba[1]; pixels[offset + 2] = rgba[2]; pixels[offset + 3] = rgba[3]
            }
        }
        return (pixels, fullW, fullH)
    }
}
