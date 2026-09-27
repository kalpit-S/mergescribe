import AppKit
import CoreImage
import Foundation
import MergeScribeCore
import QuartzCore

/// The floating recording HUD.
///
/// The menu bar sits at the top of the screen while your eyes are on the field you are
/// dictating into, so on its own it is no feedback at all. This puts the state where you
/// are looking.
///
/// Two hard constraints shape it. It must never take focus: dictation types into whatever
/// field is focused, so a panel that activated would break the whole app. Hence a
/// borderless non-activating panel, ordered front without activating, ignoring the mouse,
/// and staying visible while the app is inactive (which is always). And it must never block
/// or break recording: events hop to the main thread, and the audio thread only writes a
/// number the redraw timer reads.
public final class HUD: SessionObserver, @unchecked Sendable {
    private let levelSource: @Sendable () -> Double
    private let lock = NSLock()
    private var enabled: Bool
    private var lastToken = 0.0
    @MainActor private var panel: HUDPanel?

    public init(enabled: Bool = true, level: @escaping @Sendable () -> Double) {
        self.enabled = enabled
        levelSource = level
    }

    /// True when the HUD carries recording state, so the menu bar can stay quiet.
    public var isShowing: Bool { lock.withLock { enabled } }

    public func setEnabled(_ on: Bool) { lock.withLock { enabled = on } }

    /// The live panel's layers as an image, for checking the composition.
    @MainActor public func captureLayers() -> CGImage? { panel?.captureLayers() }

    /// "recording" or "processing" show it; anything else lets it spring away.
    public func setStatus(_ status: String) {
        guard isShowing else { return }
        DispatchQueue.main.async { [self] in
            MainActor.assumeIsolated {
                // Don't build a panel just to keep it hidden.
                if panel == nil {
                    guard HUDAnimation.visibleStatuses.contains(status) else { return }
                    panel = HUDPanel(level: levelSource)
                }
                panel?.setStatus(status)
            }
        }
    }

    private func forward(_ event: @escaping @MainActor (HUDAnimation) -> Void) {
        guard isShowing else { return }
        DispatchQueue.main.async { [self] in
            MainActor.assumeIsolated {
                if let panel { event(panel.animation) }   // nothing visible yet: status will bring it up
            }
        }
    }

    public func streamsPlanned(session: String, streams: [String]) { forward { $0.streamsPlanned(session: session, streams: streams) } }
    public func streamLanded(session: String, stream: String, final: Bool) { forward { $0.streamLanded(session: session, stream: stream, final: final) } }
    public func streamDropped(session: String, stream: String) { forward { $0.streamDropped(session: session, stream: stream) } }
    public func transcriptionDone(session: String) { forward { $0.transcriptionDone(session: session) } }
    public func correctionSkipped(session: String) { forward { $0.correctionSkipped(session: session) } }
    public func dictationDiscarded(session: String) { forward { $0.dictationDiscarded(session: session) } }
    public func partialText(session: String, text: String) {
        let tail = HUDModel.tailText(text)
        forward { $0.partialText(session: session, text: tail) }
    }

    /// Throttled: a burst of tokens can't outpace the screen.
    public func tokenTyped(session: String) {
        let now = HostClock.now
        let due: Bool = lock.withLock {
            guard now - lastToken >= 1.0 / 60 else { return false }
            lastToken = now
            return true
        }
        if due { forward { $0.tokenTyped(session: session) } }
    }
}

/// Draws a layer with a closure.
final class LayerPainter: NSObject, CALayerDelegate {
    let paint: (CGContext) -> Void
    init(_ paint: @escaping (CGContext) -> Void) { self.paint = paint }
    func draw(_ layer: CALayer, in ctx: CGContext) { paint(ctx) }
    func action(for layer: CALayer, forKey event: String) -> CAAction? { NSNull() }   // follow every frame, don't tween
}

/// The panel and its redraw timer. Main thread only.
@MainActor
final class HUDPanel {
    static let fps = 60.0
    /// Above normal windows and full-screen apps, below the screen saver.
    static let level = NSWindow.Level(rawValue: 25)

    let animation = HUDAnimation()
    private let levelSource: () -> Double
    private let panel: NSPanel
    private let canvas: CGRect
    private let under = CALayer(), base = CALayer(), inside = CALayer(), words = CALayer()
    private let auroraGlow = CALayer(), aurora = CALayer(), mask = CAShapeLayer()
    private var painters: [LayerPainter] = []
    private var timer: Timer?
    private var lastTick: Double?

    init(level: @escaping () -> Double) {
        levelSource = level
        canvas = CGRect(origin: .zero, size: HUDRenderer.canvasSize)
        panel = NSPanel(contentRect: Self.frame(), styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
        panel.level = Self.level
        panel.isOpaque = false
        panel.backgroundColor = .clear
        // The layers draw their own shadow and light; a window shadow would outline the canvas.
        panel.hasShadow = false
        panel.hidesOnDeactivate = false
        panel.ignoresMouseEvents = true
        panel.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary]
        panel.alphaValue = 0

        let view = NSView(frame: canvas)
        view.wantsLayer = true
        view.layerUsesCoreImageFilters = true
        let root = view.layer!
        let scale = NSScreen.main?.backingScaleFactor ?? 2
        for layer in [under, base, inside, words] {
            layer.frame = canvas
            root.addSublayer(layer)
        }
        // Beneath the capsule: drawn at 1x and softened by the GPU, which costs nothing per frame.
        under.contentsScale = 1
        under.filters = [Self.blur(HUDRenderer.underBlur)]
        let paintBase = LayerPainter { [unowned self] in HUDRenderer.drawBase($0, canvas, animation) }
        let paintWords = LayerPainter { [unowned self] in HUDRenderer.drawWords($0, canvas, animation) }
        painters = [paintBase, paintWords]
        base.delegate = paintBase
        words.delegate = paintWords
        base.contentsScale = scale
        words.contentsScale = scale
        // The aurora is kept inside the capsule by a mask that follows it.
        mask.frame = canvas
        inside.mask = mask
        auroraGlow.filters = [Self.blur(HUDRenderer.auroraGlow)]
        auroraGlow.opacity = 0.9
        for layer in [auroraGlow, aurora] {
            layer.contentsScale = 1
            layer.actions = ["contents": NSNull(), "bounds": NSNull(), "position": NSNull()]
            inside.addSublayer(layer)
        }
        panel.contentView = view
    }

    /// The panel's layers rendered into an image (without their blur filters): a check
    /// that the live composition matches what the renderer draws.
    func captureLayers() -> CGImage? {
        guard let root = panel.contentView?.layer else { return nil }
        let scale = 2.0
        guard let ctx = CGContext(data: nil, width: Int(canvas.width * scale), height: Int(canvas.height * scale),
                                  bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.setFillColor(CGColor(srgbRed: 0.35, green: 0.5, blue: 0.75, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: canvas.width * scale, height: canvas.height * scale))
        ctx.scaleBy(x: scale, y: scale)
        root.render(in: ctx)
        return ctx.makeImage()
    }

    static func blur(_ radius: Double) -> CIFilter {
        let filter = CIFilter(name: "CIGaussianBlur")!
        filter.setDefaults()
        filter.setValue(radius, forKey: kCIInputRadiusKey)
        return filter
    }

    /// Centred low on the screen with the mouse, where you are looking.
    static func frame() -> CGRect {
        let screen = NSScreen.screens.first { NSMouseInRect(NSEvent.mouseLocation, $0.frame, false) } ?? NSScreen.main ?? NSScreen.screens[0]
        let visible = screen.visibleFrame, size = HUDRenderer.canvasSize
        return CGRect(x: visible.midX - size.width / 2, y: visible.minY + HUDRenderer.bottomMargin - HUDRenderer.canvasMargin,
                      width: size.width, height: size.height)
    }

    func setStatus(_ status: String) {
        animation.setStatus(status)
        guard animation.visible else { return }   // hiding is left to the tick: the pill springs away
        if !panel.isVisible { panel.setFrame(Self.frame(), display: false) }   // the active screen may have changed
        panel.orderFrontRegardless()
        if timer == nil {
            lastTick = nil
            let timer = Timer(timeInterval: 1 / Self.fps, repeats: true) { [weak self] _ in
                MainActor.assumeIsolated { self?.tick() }
            }
            RunLoop.main.add(timer, forMode: .common)
            self.timer = timer
        }
    }

    private func tick() {
        let now = HostClock.now
        let dt = lastTick.map { now - $0 } ?? 1 / Self.fps
        lastTick = now
        animation.step(dt, level: animation.status == "recording" ? levelSource() : 0)
        if animation.hidden {
            panel.orderOut(nil)
            timer?.invalidate()
            timer = nil
            animation.reset()
            return
        }
        panel.alphaValue = animation.alpha
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        mask.path = HUDRenderer.capsule(HUDRenderer.capsuleRect(canvas, animation), inset: 0.6)
        under.contents = HUDRenderer.layerImage(size: canvas.size, scale: 1) { HUDRenderer.drawUnder($0, canvas, animation) }
        if let (frame, span, track) = HUDRenderer.auroraFrame(canvas, animation),
           let image = HUDRenderer.auroraImage(animation, span: span, track: track) {
            for layer in [auroraGlow, aurora] {
                layer.frame = frame
                layer.contents = image
            }
        } else {
            auroraGlow.contents = nil
            aurora.contents = nil
        }
        base.setNeedsDisplay()
        words.setNeedsDisplay()
        CATransaction.commit()
    }
}
