import AppKit
import Foundation
import ImageIO
import MergeScribeCore
import MergeScribeMac
import UniformTypeIdentifiers

/// The scripted dictation scripts/preview_hud.py plays: three streams, a chunk landing while
/// you talk, release, results arriving one by one, one left behind, then a correction
/// streaming in bursts. Every state the HUD has, without a microphone or an API key.
enum HUDScript {
    static let session = "preview"
    static let streams = ["parakeet/MacBook Pro Microphone", "or-mai-transcribe-2/MacBook Pro Microphone",
                          "or-mai-transcribe-2/SoloCast"]
    static let length = 8.8

    /// Syllables at about 7Hz, phrases that swell and fade, a pause that ends a chunk.
    static func level(at t: Double) -> Double {
        if (t > 0.85 && t < 1.2) || t > 3.8 { return 0.03 }
        let syllables = abs(sin(t * 7.3)) * (0.55 + 0.45 * sin(t * 1.7 + 1.0))
        return min(1, 0.1 + 0.85 * syllables)
    }

    static var events: [(Double, (HUDAnimation) -> Void)] {
        var out: [(Double, (HUDAnimation) -> Void)] = [
            (0.0, { $0.setStatus("recording") }),
            (0.0, { $0.streamsPlanned(session: session, streams: streams) }),
            (1.25, { $0.streamLanded(session: session, stream: streams[0], final: false) }),
            (1.4, { $0.partialText(session: session, text: HUDModel.tailText("so I was thinking we should")) }),
            (1.6, { $0.streamLanded(session: session, stream: streams[1], final: false) }),
            (1.75, { $0.streamLanded(session: session, stream: streams[2], final: false) }),
            (2.8, { $0.partialText(session: session, text: HUDModel.tailText("so I was thinking we should merge all the streams into one")) }),
            (3.8, { $0.setStatus("processing") }),
            (4.1, { $0.streamLanded(session: session, stream: streams[0], final: true) }),
            (4.7, { $0.streamLanded(session: session, stream: streams[1], final: true) }),
            (5.5, { $0.streamDropped(session: session, stream: streams[2]) }),
            (5.5, { $0.transcriptionDone(session: session) }),
        ]
        // The correction streams in bursts, the way the model actually sends them.
        for at in [6.2, 6.55, 7.0, 7.3] {
            for k in 0..<4 { out.append((at + Double(k) * 0.045, { $0.tokenTyped(session: session) })) }
        }
        out.append((8.0, { $0.setStatus("idle") }))
        return out.sorted { $0.0 < $1.0 }
    }

    /// The same script, as calls on the live HUD.
    static var previewEvents: [(Double, (HUD) -> Void)] {
        var out: [(Double, (HUD) -> Void)] = [
            (0.0, { $0.setStatus("recording") }),
            (0.0, { $0.streamsPlanned(session: session, streams: streams) }),
            (1.25, { $0.streamLanded(session: session, stream: streams[0], final: false) }),
            (1.4, { $0.partialText(session: session, text: "so I was thinking we should") }),
            (1.6, { $0.streamLanded(session: session, stream: streams[1], final: false) }),
            (1.75, { $0.streamLanded(session: session, stream: streams[2], final: false) }),
            (2.8, { $0.partialText(session: session, text: "so I was thinking we should merge all the streams into one") }),
            (3.8, { $0.setStatus("processing") }),
            (4.1, { $0.streamLanded(session: session, stream: streams[0], final: true) }),
            (4.7, { $0.streamLanded(session: session, stream: streams[1], final: true) }),
            (5.5, { $0.streamDropped(session: session, stream: streams[2]) }),
            (5.5, { $0.transcriptionDone(session: session) }),
        ]
        for at in [6.2, 6.55, 7.0, 7.3] {
            for k in 0..<4 { out.append((at + Double(k) * 0.045, { $0.tokenTyped(session: session) })) }
        }
        out.append((8.0, { $0.setStatus("idle") }))
        return out
    }

    /// Frames at the given times, as PNGs, plus one sheet with them stacked.
    static func renderFrames(to directory: URL, at times: [Double]) {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let anim = HUDAnimation()
        var pending = events
        var t = 0.0
        var images: [CGImage] = []
        for target in times.sorted() {
            while t < target {
                while let next = pending.first, next.0 <= t { next.1(anim); pending.removeFirst() }
                anim.step(1.0 / 60, level: level(at: t))
                t += 1.0 / 60
            }
            if let image = HUDRenderer.snapshot(anim) {
                images.append(image)
                write(image, to: directory.appendingPathComponent(String(format: "hud_%04.1f.png", target)))
            }
        }
        if let first = images.first,
           let ctx = CGContext(data: nil, width: first.width, height: first.height * images.count, bitsPerComponent: 8,
                               bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                               bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) {
            for (i, image) in images.enumerated() {
                ctx.draw(image, in: CGRect(x: 0, y: (images.count - 1 - i) * first.height, width: first.width, height: first.height))
            }
            if let sheet = ctx.makeImage() { write(sheet, to: directory.appendingPathComponent("sheet.png")) }
        }
    }

    static func write(_ image: CGImage, to url: URL) {
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(dest, image, nil)
        CGImageDestinationFinalize(dest)
    }
}
