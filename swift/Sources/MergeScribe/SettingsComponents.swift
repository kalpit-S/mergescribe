import AppKit
import MergeScribeCore
import MergeScribeMac
import SwiftUI

/// A title with a detail line under it, as System Settings rows have.
struct RowLabel: View {
    let title: String
    var detail: String = ""
    init(_ title: String, _ detail: String = "") { self.title = title; self.detail = detail }
    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(title)
            if !detail.isEmpty { Text(detail).font(.caption).foregroundStyle(.secondary) }
        }
    }
}

struct Footer: View {
    let text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(text).font(.caption).foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// A big number with a label, for the top of a page.
struct StatTile: View {
    let title: String
    let value: String
    var detail: String = ""
    var symbol: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 5) {
                if !symbol.isEmpty { Image(systemName: symbol).font(.caption).foregroundStyle(.secondary) }
                Text(title).font(.caption).foregroundStyle(.secondary)
            }
            Text(value).font(.system(size: 22, weight: .semibold, design: .rounded)).monospacedDigit()
                .lineLimit(1).minimumScaleFactor(0.6)
            if !detail.isEmpty { Text(detail).font(.caption2).foregroundStyle(.tertiary).lineLimit(1) }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.primary.opacity(0.045)))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Color.primary.opacity(0.07)))
    }
}

/// A page that is a scrolling column of content rather than a form.
struct Page<Content: View>: View {
    let title: String
    var subtitle: String = ""
    @ViewBuilder let content: Content

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(title).font(.largeTitle.weight(.bold))
                    if !subtitle.isEmpty { Text(subtitle).foregroundStyle(.secondary) }
                }
                content
            }
            .padding(.horizontal, 28)
            .padding(.top, 36)
            .padding(.bottom, 28)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

/// A number that clamps what you type and shows what was kept.
struct NumberField: View {
    @Binding var value: Double
    let range: ClosedRange<Double>
    var places = 2
    @State private var text = ""
    @FocusState private var focused: Bool

    var body: some View {
        TextField("", text: $text)
            .labelsHidden()
            .frame(width: 72)
            .multilineTextAlignment(.trailing)
            .focused($focused)
            .onAppear { text = Self.trim(value, places) }
            .onSubmit(commit)
            .onChange(of: focused) { _, now in if !now { commit() } }
    }

    func commit() {
        guard let parsed = Double(text.trimmingCharacters(in: .whitespaces)) else { text = Self.trim(value, places); return }
        let clamped = min(max(parsed, range.lowerBound), range.upperBound)
        value = clamped
        text = Self.trim(clamped, places)
    }

    /// No trailing decimal zeros: 2.50 -> "2.5", 450 -> "450".
    static func trim(_ value: Double, _ places: Int) -> String {
        let text = String(format: "%.\(places)f", value)
        guard text.contains(".") else { return text }
        var out = text
        while out.hasSuffix("0") { out.removeLast() }
        if out.hasSuffix(".") { out.removeLast() }
        return out
    }
}

/// A plain-text editor that saves when editing ends. Smart quotes, dashes and text
/// replacement are off: they would silently rewrite prompts that forbid those characters.
struct PlainTextEditor: NSViewRepresentable {
    let text: String
    let onCommit: (String) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onCommit: onCommit) }

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSTextView.scrollableTextView()
        let view = scroll.documentView as! NSTextView
        view.string = text
        view.font = .systemFont(ofSize: 13)
        view.isRichText = false
        view.isAutomaticQuoteSubstitutionEnabled = false
        view.isAutomaticDashSubstitutionEnabled = false
        view.isAutomaticTextReplacementEnabled = false
        view.drawsBackground = false
        view.textContainerInset = NSSize(width: 0, height: 4)
        view.delegate = context.coordinator
        scroll.drawsBackground = false
        scroll.borderType = .noBorder
        scroll.autohidesScrollers = true
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        context.coordinator.onCommit = onCommit
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var onCommit: (String) -> Void
        init(onCommit: @escaping (String) -> Void) { self.onCommit = onCommit }
        func textDidEndEditing(_ notification: Notification) {
            if let view = notification.object as? NSTextView { onCommit(view.string) }
        }
    }
}

// MARK: - The HUD, playing

/// The recording HUD playing its scripted dictation on a loop, drawn by the same renderer
/// as the real one.
@MainActor
final class HUDPreviewPlayer: ObservableObject {
    @Published var frame: CGImage?
    private var animation = HUDAnimation()
    private var events = HUDScript.events
    private var clock = 0.0
    private var timer: Timer?

    func start() {
        guard timer == nil else { return }
        let timer = Timer(timeInterval: 1.0 / 30, repeats: true) { [weak self] timer in
            guard let self else { timer.invalidate(); return }    // the view went without stopping it
            MainActor.assumeIsolated { self.tick() }
        }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }

    private func tick() {
        for _ in 0..<2 {    // two model steps a frame: the spring model runs at 60 Hz
            while let next = events.first, next.0 <= clock { next.1(animation); events.removeFirst() }
            animation.step(1.0 / 60, level: HUDScript.level(at: clock))
            clock += 1.0 / 60
        }
        if clock > HUDScript.length + 0.6 {
            animation = HUDAnimation()
            events = HUDScript.events
            clock = 0
        }
        frame = HUDRenderer.snapshot(animation, scale: 2, backdrop: nil)
    }
}

struct HUDPreview: View {
    @StateObject private var player = HUDPreviewPlayer()

    var body: some View {
        ZStack {
            LinearGradient(colors: [Color(red: 0.18, green: 0.32, blue: 0.62), Color(red: 0.43, green: 0.28, blue: 0.62)],
                           startPoint: .topLeading, endPoint: .bottomTrailing)
            if let frame = player.frame {
                Image(decorative: frame, scale: 2).resizable().aspectRatio(contentMode: .fit)
            }
        }
        .frame(height: 150)
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .onAppear { player.start() }
        .onDisappear { player.stop() }
    }
}

// MARK: - The trigger key

/// Shows the trigger key; click it and press the key you want.
struct ShortcutRecorder: View {
    @Binding var key: String
    @State private var listening = false
    @State private var monitor: Any?

    static let names: [String: String] = [
        "cmd_r": "Right Command", "cmd_l": "Left Command", "alt_r": "Right Option", "alt_l": "Left Option",
        "ctrl_r": "Right Control", "ctrl_l": "Left Control", "shift_r": "Right Shift",
        "f13": "F13", "f14": "F14", "f15": "F15", "f16": "F16", "f17": "F17", "f18": "F18", "f19": "F19", "f20": "F20",
    ]

    var body: some View {
        Button {
            listening ? stopListening() : startListening()
        } label: {
            Text(listening ? "Press a key…" : Self.names[key] ?? key)
                .frame(minWidth: 130)
        }
        .onDisappear(perform: stopListening)
    }

    func startListening() {
        listening = true
        monitor = NSEvent.addLocalMonitorForEvents(matching: [.flagsChanged, .keyDown]) { event in
            if event.type == .keyDown && event.keyCode == 53 { stopListening(); return nil }   // Esc: never mind
            if let name = Self.names.keys.first(where: { Hotkey.keys[$0]?.code == event.keyCode }) {
                // Modifiers arrive on press and release; take the press.
                if event.type == .flagsChanged, let bit = Hotkey.keys[name]?.modifierBit,
                   UInt64(event.modifierFlags.rawValue) & bit == 0 { return nil }
                key = name
                stopListening()
                return nil
            }
            return event
        }
    }

    func stopListening() {
        listening = false
        if let monitor { NSEvent.removeMonitor(monitor) }
        monitor = nil
    }
}

// MARK: - The pipeline

/// One stage of the pipeline, with how long it takes.
struct Stage: View {
    let symbol: String
    let title: String
    let detail: String
    var tint: Color = .accentColor

    var body: some View {
        VStack(spacing: 5) {
            Image(systemName: symbol)
                .font(.system(size: 17, weight: .medium))
                .foregroundStyle(tint)
                .frame(width: 38, height: 38)
                .background(Circle().fill(tint.opacity(0.14)))
            Text(title).font(.caption.weight(.semibold)).multilineTextAlignment(.center).lineLimit(3)
            Text(detail).font(.caption2).foregroundStyle(.secondary).multilineTextAlignment(.center).lineLimit(2)
        }
        .frame(maxWidth: .infinity)
    }
}

struct StageArrow: View {
    var body: some View {
        Image(systemName: "chevron.right").font(.caption.weight(.bold)).foregroundStyle(.tertiary).padding(.bottom, 30)
    }
}

/// Milliseconds as a short duration: "420 ms", "1.2 s".
func duration(_ ms: Double?) -> String {
    guard let ms else { return "—" }
    return ms < 1000 ? "\(Int(ms.rounded())) ms" : String(format: "%.1f s", ms / 1000)
}
