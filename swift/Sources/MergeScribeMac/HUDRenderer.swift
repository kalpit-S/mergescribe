import AppKit
import CoreImage
import Foundation
import MergeScribeCore

/// How the HUD is drawn, frame by frame, from the animation model. Used by the live panel,
/// which lays these out as layers, and by snapshot(), which composes them into an image.
///
/// The capsule is ink, not glass: a blurred desktop tints glass muddy (olive over a green
/// wallpaper, slate over blue) and greys the light inside. Near-black with a faint cool
/// falloff keeps depth and lets the aurora's colours carry.
public enum HUDRenderer {
    // Geometry. The pill stays compact until there is transcript to show: most dictations
    // are a single chunk, so nothing is available before release.
    public static let compactWidth = 150.0
    public static let wideWidth = 420.0
    public static let height = 40.0
    static let dot = 40.0                 // the capsule is born from, and returns to, a circle
    public static let bottomMargin = 130.0
    static let padding = 14.0
    static let textStrip = 112.0          // how much of the capsule the aurora keeps once words arrive
    static let textGap = 14.0
    static let inkTop = (0.075, 0.082, 0.11), inkBottom = (0.018, 0.02, 0.03)
    public static let auroraPad = 12.0    // clear margin round the aurora for its glow
    public static let auroraGlow = 5.0    // pt of blur on the glow copy
    public static let underBlur = 7.0     // the shadow and the edge light's halo, beneath the capsule
    /// The panel is a fixed transparent canvas larger than the pill, so the pill can grow,
    /// overshoot and glow without resizing the window every frame.
    public static let canvasMargin = 48.0
    public static var canvasSize: CGSize { CGSize(width: wideWidth + 2 * canvasMargin, height: height + 2 * canvasMargin) }

    static let srgb = CGColorSpace(name: CGColorSpace.sRGB)!

    /// The aurora's settled mint lifted toward white: light should glow, not paint.
    static func light(_ towardWhite: Double) -> (Double, Double, Double) {
        let s = HUDModel.settled
        return (Double(s.x) + (1 - Double(s.x)) * towardWhite, Double(s.y) + (1 - Double(s.y)) * towardWhite,
                Double(s.z) + (1 - Double(s.z)) * towardWhite)
    }

    /// Where the capsule is this frame: a dot that opens into it, and closes back.
    public static func capsuleRect(_ bounds: CGRect, _ anim: HUDAnimation) -> CGRect {
        let target = compactWidth + (wideWidth - compactWidth) * anim.width.value
        var width = max(dot, dot + (target - dot) * anim.morph)
        width += (dot - width) * 0.85 * max(0, min(1, anim.collapse.value))
        let cx = bounds.width / 2, cy = canvasMargin + height / 2 - anim.rise
        return CGRect(x: cx - width / 2, y: cy - height / 2, width: width, height: height)
    }

    public static func capsule(_ rect: CGRect, inset: Double = 0) -> CGPath {
        let r = rect.height / 2 - inset
        return CGPath(roundedRect: rect.insetBy(dx: inset, dy: inset), cornerWidth: r, cornerHeight: r, transform: nil)
    }

    /// The point a fraction t of the way round the capsule, clockwise from the top left.
    static func pointOnEdge(_ rect: CGRect, _ t: Double) -> CGPoint {
        let r = rect.height / 2
        let left = rect.minX + r, right = rect.maxX - r, cy = rect.minY + r
        let straight = max(0, right - left), arc = Double.pi * r
        var s = (t - t.rounded(.down)) * (2 * straight + 2 * arc)
        if s < straight { return CGPoint(x: left + s, y: cy + r) }
        s -= straight
        if s < arc { let a = .pi / 2 - s / r; return CGPoint(x: right + r * cos(a), y: cy + r * sin(a)) }
        s -= arc
        if s < straight { return CGPoint(x: right - s, y: cy - r) }
        let a = -.pi / 2 - (s - straight) / r
        return CGPoint(x: left + r * cos(a), y: cy + r * sin(a))
    }

    static func gradient(_ stops: [(Double, (Double, Double, Double, Double))]) -> CGGradient {
        CGGradient(colorsSpace: srgb, colors: stops.map { CGColor(srgbRed: $0.1.0, green: $0.1.1, blue: $0.1.2, alpha: $0.1.3) } as CFArray,
                   locations: stops.map { CGFloat($0.0) })!
    }

    static func clip(toStrokeOf path: CGPath, width: Double, in ctx: CGContext) {
        ctx.addPath(path)
        ctx.setLineWidth(width)
        ctx.replacePathWithStrokedPath()
        ctx.clip()
    }

    /// A band of light along the edge: steady, plus a sweep with a short tail.
    static func edgeLight(_ ctx: CGContext, _ rect: CGRect, _ anim: HUDAnimation, width: Double, reach: Double, strength: Double) {
        let (steady, head, sweep) = anim.edgeLight()
        let rgb = light(0.35)
        ctx.saveGState()
        clip(toStrokeOf: capsule(rect, inset: 0.5), width: width, in: ctx)
        ctx.setFillColor(CGColor(srgbRed: rgb.0, green: rgb.1, blue: rgb.2, alpha: min(1, max(0, steady * strength))))
        ctx.fill(ctx.boundingBoxOfClipPath)
        if sweep > 0.01 {
            let falloff = gradient([(0, (rgb.0, rgb.1, rgb.2, 1)), (0.35, (rgb.0, rgb.1, rgb.2, 0.35)), (1, (rgb.0, rgb.1, rgb.2, 0))])
            for (lag, weight) in [(0.0, 1.0), (0.035, 0.55), (0.07, 0.25)] {
                let p = pointOnEdge(rect, head - lag)
                ctx.setAlpha(min(1, max(0, sweep * strength * weight)))
                ctx.drawRadialGradient(falloff, startCenter: p, startRadius: 0, endCenter: p, endRadius: reach, options: [])
            }
        }
        ctx.restoreGState()
    }

    /// The aurora's span, the centre line, where the words begin, and the pulse.
    static func layout(_ rect: CGRect, _ anim: HUDAnimation) -> (midY: Double, left: Double, stripRight: Double, right: Double, pulseX: Double) {
        let midY = rect.midY
        var left = rect.minX + padding
        let right = rect.maxX - padding
        var stripRight = right + (left + textStrip - right) * anim.width.value
        // A discarded dictation draws itself in to a point instead of fading out at full width.
        let collapse = max(0, min(1, anim.collapse.value))
        if collapse > 0 {
            let centre = (left + stripRight) / 2
            left += (centre - left) * collapse
            stripRight += (centre - stripRight) * collapse
        }
        return (midY, left, stripRight, right, left + anim.pulseTrack * (right - left))
    }

    // MARK: - Layers, back to front

    /// Beneath the capsule, drawn at 1x and blurred: a shadow for depth, and the edge light's halo.
    public static func drawUnder(_ ctx: CGContext, _ bounds: CGRect, _ anim: HUDAnimation) {
        let rect = capsuleRect(bounds, anim)
        ctx.addPath(capsule(CGRect(x: rect.minX - 1, y: rect.minY - 7, width: rect.width + 2, height: rect.height)))
        ctx.setFillColor(CGColor(gray: 0, alpha: 0.34))
        ctx.fillPath()
        edgeLight(ctx, rect, anim, width: 6, reach: 72, strength: 0.8)
    }

    /// Crisp, beneath the aurora: the ink capsule, its rim, and the edge light.
    public static func drawBase(_ ctx: CGContext, _ bounds: CGRect, _ anim: HUDAnimation) {
        let rect = capsuleRect(bounds, anim)
        // A dark hairline just outside, so the capsule keeps its edge on a pale wallpaper.
        ctx.addPath(capsule(rect, inset: -0.35))
        ctx.setLineWidth(0.7)
        ctx.setStrokeColor(CGColor(gray: 0, alpha: 0.18))
        ctx.strokePath()

        ctx.saveGState()
        ctx.addPath(capsule(rect, inset: 0.3))
        ctx.clip()
        let ink = gradient([(0, (inkTop.0, inkTop.1, inkTop.2, 1)), (1, (inkBottom.0, inkBottom.1, inkBottom.2, 1))])
        ctx.drawLinearGradient(ink, start: CGPoint(x: 0, y: rect.maxY), end: CGPoint(x: 0, y: rect.minY), options: [])
        ctx.restoreGState()

        // The lit rim: bright along the top, a faint return along the bottom.
        ctx.saveGState()
        clip(toStrokeOf: capsule(rect, inset: 0.6), width: 1, in: ctx)
        let rim = gradient([(0, (1, 1, 1, 0.3)), (0.45, (1, 1, 1, 0.06)), (0.7, (1, 1, 1, 0.04)), (1, (1, 1, 1, 0.11))])
        ctx.drawLinearGradient(rim, start: CGPoint(x: 0, y: rect.maxY), end: CGPoint(x: 0, y: rect.minY), options: [])
        ctx.restoreGState()

        edgeLight(ctx, rect, anim, width: 1, reach: 34, strength: 0.55)
    }

    /// Crisp, above the aurora: the newest words. Older ones fade at the left; while the
    /// correction runs, a shimmer travels through them with the ripples on the line.
    public static func drawWords(_ ctx: CGContext, _ bounds: CGRect, _ anim: HUDAnimation) {
        guard !anim.transcript.isEmpty, anim.contentAlpha > 0.01 else { return }
        let rect = capsuleRect(bounds, anim)
        let (midY, _, stripRight, right, pulseX) = layout(rect, anim)
        let left = stripRight + textGap, room = right - left
        guard room >= 20 else { return }
        ctx.saveGState()
        ctx.addPath(capsule(rect, inset: 1.5))
        ctx.clip()
        let (line, trimmed) = fit(anim.transcript, room: room)
        let size = line.size()
        let m = anim.correcting
        let base = (0.86 + (0.5 - 0.86) * m) * max(0, min(1, anim.width.value)) * anim.contentAlpha
        let stops: [(Double, (Double, Double, Double, Double))] = (0...24).map { k in
            let u = Double(k) / 24, px = left + u * room
            var a = base + (1 - base) * m * exp(-pow((px - pulseX) / 30, 2)) * anim.contentAlpha
            if trimmed { a *= pow(max(0, min(1, (px - left) / 30)), 1.5) }
            return (u, (1, 1, 1, max(0, min(1, a))))
        }
        ctx.beginTransparencyLayer(auxiliaryInfo: nil)
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(cgContext: ctx, flipped: false)
        line.draw(at: CGPoint(x: left, y: midY - size.height / 2 + 0.5))
        NSGraphicsContext.restoreGraphicsState()
        ctx.setBlendMode(.destinationIn)
        ctx.drawLinearGradient(gradient(stops), start: CGPoint(x: left, y: 0), end: CGPoint(x: right, y: 0), options: [])
        ctx.endTransparencyLayer()
        ctx.restoreGState()
    }

    nonisolated(unsafe) private static var fitted: [String: (NSAttributedString, Bool)] = [:]

    /// The newest words that fit, trimmed whole from the front so a fade never cuts a letter.
    static func fit(_ text: String, room: Double) -> (NSAttributedString, Bool) {
        let key = "\(Int(room))|\(text)"
        if let hit = fitted[key] { return hit }
        let attributes: [NSAttributedString.Key: Any] = [.font: NSFont.systemFont(ofSize: 13, weight: .medium),
                                                         .foregroundColor: NSColor.white]
        var words = text.split(separator: " ").map(String.init), trimmed = false
        var line = NSAttributedString(string: words.joined(separator: " "), attributes: attributes)
        while words.count > 1 && line.size().width > room {
            words.removeFirst()
            trimmed = true
            line = NSAttributedString(string: words.joined(separator: " "), attributes: attributes)
        }
        if fitted.count > 64 { fitted.removeAll() }
        fitted[key] = (line, trimmed)
        return (line, trimmed)
    }

    /// Where this frame's aurora image goes (padded for its glow), the strip's length and the
    /// track its light travels; nil when there is nothing to draw.
    public static func auroraFrame(_ bounds: CGRect, _ anim: HUDAnimation) -> (frame: CGRect, span: Double, track: Double)? {
        let rect = capsuleRect(bounds, anim)
        let (_, left, stripRight, right, _) = layout(rect, anim)
        let span = stripRight - left
        guard span >= 2, anim.contentAlpha > 0.005 else { return nil }
        return (CGRect(x: left - auroraPad, y: rect.minY - auroraPad, width: span + 2 * auroraPad, height: rect.height + 2 * auroraPad),
                span, right - left)
    }

    /// This frame's aurora as an image, at 1x: it is a field of soft light, with no detail for
    /// Retina pixels to show, and the glow is a blurred copy composited on the GPU.
    public static func auroraImage(_ anim: HUDAnimation, span: Double, track: Double) -> CGImage? {
        let (pixels, w, h) = HUDModel.aurora(anim, width: span, height: height, pad: auroraPad, track: track)
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
        return CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4, space: srgb,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }

    // MARK: - Offscreen

    /// One frame composed into an image, over a backdrop, for previews and checks.
    public static func snapshot(_ anim: HUDAnimation, scale: Double = 2, backdrop: CGColor? = CGColor(srgbRed: 0.35, green: 0.5, blue: 0.75, alpha: 1)) -> CGImage? {
        let size = canvasSize
        let bounds = CGRect(origin: .zero, size: size)
        let w = Int(size.width * scale), h = Int(size.height * scale)
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: srgb,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.scaleBy(x: scale, y: scale)
        if let backdrop { ctx.setFillColor(backdrop); ctx.fill(bounds) }
        ctx.setAlpha(anim.alpha)
        ctx.beginTransparencyLayer(auxiliaryInfo: nil)
        if let under = layerImage(size: size, scale: 1, draw: { drawUnder($0, bounds, anim) }) {
            ctx.draw(blurred(under, radius: underBlur) ?? under, in: bounds)
        }
        drawBase(ctx, bounds, anim)
        if let (frame, span, track) = auroraFrame(bounds, anim), let image = auroraImage(anim, span: span, track: track) {
            ctx.saveGState()
            ctx.addPath(capsule(capsuleRect(bounds, anim), inset: 0.6))
            ctx.clip()
            if let glow = blurred(image, radius: auroraGlow) {
                ctx.setAlpha(0.9)
                ctx.draw(glow, in: frame)
                ctx.setAlpha(1)
            }
            ctx.draw(image, in: frame)
            ctx.restoreGState()
        }
        drawWords(ctx, bounds, anim)
        ctx.endTransparencyLayer()
        return ctx.makeImage()
    }

    static func layerImage(size: CGSize, scale: Double, draw: (CGContext) -> Void) -> CGImage? {
        guard let ctx = CGContext(data: nil, width: Int(size.width * scale), height: Int(size.height * scale), bitsPerComponent: 8,
                                  bytesPerRow: 0, space: srgb, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.scaleBy(x: scale, y: scale)
        draw(ctx)
        return ctx.makeImage()
    }

    nonisolated(unsafe) static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    /// A Gaussian blur of the same extent, as Core Image does for a layer's content filter.
    static func blurred(_ image: CGImage, radius: Double) -> CGImage? {
        let input = CIImage(cgImage: image)
        guard let filter = CIFilter(name: "CIGaussianBlur") else { return nil }
        filter.setValue(input.clampedToExtent(), forKey: kCIInputImageKey)
        filter.setValue(radius, forKey: kCIInputRadiusKey)
        guard let output = filter.outputImage?.cropped(to: input.extent) else { return nil }
        return ciContext.createCGImage(output, from: input.extent)
    }
}
