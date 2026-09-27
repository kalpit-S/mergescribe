import AppKit
import ApplicationServices
import Foundation
import MergeScribeCore

/// Accessibility calls with a short timeout: a target app that has stalled must not hold up
/// output (the default is six seconds).
enum AX {
    static let timeout: Float = 0.5

    static func app(_ pid: pid_t) -> AXUIElement {
        let element = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(element, timeout)
        return element
    }

    static func get(_ element: AXUIElement, _ attribute: String) -> CFTypeRef? {
        var value: CFTypeRef?
        return AXUIElementCopyAttributeValue(element, attribute as CFString, &value) == .success ? value : nil
    }

    static func element(_ element: AXUIElement, _ attribute: String) -> AXUIElement? {
        guard let value = get(element, attribute), CFGetTypeID(value) == AXUIElementGetTypeID() else { return nil }
        return (value as! AXUIElement)
    }

    static func string(_ element: AXUIElement, _ attribute: String) -> String? {
        get(element, attribute) as? String
    }

    static func set(_ element: AXUIElement, _ attribute: String, _ value: CFTypeRef) {
        AXUIElementSetAttributeValue(element, attribute as CFString, value)
    }

    static func pid(_ element: AXUIElement) -> pid_t? {
        var pid: pid_t = 0
        return AXUIElementGetPid(element, &pid) == .success ? pid : nil
    }

    /// The element with keyboard focus, anywhere.
    static func focused() -> AXUIElement? {
        let system = AXUIElementCreateSystemWide()
        AXUIElementSetMessagingTimeout(system, timeout)
        return element(system, kAXFocusedUIElementAttribute)
    }

    /// Ask a Chromium or Electron app to build its accessibility tree. Chrome, Slack, Claude,
    /// ChatGPT and VS Code expose an empty tree until an assistive client asks, so their text
    /// fields read as nothing. Harmless for native apps.
    static func wake(_ pid: pid_t) {
        let app = app(pid)
        for attribute in ["AXManualAccessibility", "AXEnhancedUserInterface"] {
            set(app, attribute, kCFBooleanTrue)
        }
    }

    static var trusted: Bool { AXIsProcessTrusted() }
}

/// The frontmost app and window, where a dictation started, and the selection.
public final class MacFocus: Focus, @unchecked Sendable {
    /// Repeated reads within this long reuse the last answer.
    static let cacheSeconds = 0.3
    /// How long to wait for a window being brought back, then for it to restore its field.
    static let returnTimeout = 0.8
    static let returnSettle: useconds_t = 80_000

    private let cache = Locked<(at: Double, context: AppContext?)>((0, nil))
    private let source = CGEventSource(stateID: .privateState)

    public init() {}

    public func appContext() -> AppContext? {
        let now = HostClock.now
        if let cached = cache.read({ now - $0.at < Self.cacheSeconds ? $0.context : nil }) { return cached }
        guard let app = NSWorkspace.shared.frontmostApplication else { return nil }
        let element = AX.app(app.processIdentifier)
        var title = ""
        for attribute in [kAXFocusedWindowAttribute, kAXMainWindowAttribute] {
            if let window = AX.element(element, attribute), let t = AX.string(window, kAXTitleAttribute), !t.isEmpty {
                title = t
                break
            }
        }
        let context = AppContext(appName: app.localizedName ?? "", windowTitle: title, bundleId: app.bundleIdentifier ?? "")
        cache.update { $0 = (now, context) }
        return context
    }

    public func captureOrigin() -> Origin? {
        guard let app = NSWorkspace.shared.frontmostApplication else { return nil }
        let element = AX.app(app.processIdentifier)
        return Origin(pid: app.processIdentifier, handle: OriginHandle(app: element, window: AX.element(element, kAXFocusedWindowAttribute)))
    }

    public func atOrigin(_ origin: Origin) -> Bool {
        guard let app = NSWorkspace.shared.frontmostApplication, app.processIdentifier == origin.pid else { return false }
        guard let handle = origin.handle as? OriginHandle, let window = handle.window else { return true }
        // Two references to the same window compare equal.
        guard let focused = AX.element(handle.app, kAXFocusedWindowAttribute) else { return false }
        return CFEqual(focused, window)
    }

    /// Through accessibility rather than NSRunningApplication.activate, which a background
    /// app may not be allowed to use to take focus. Windows belong to the window server, so
    /// this works for Electron and browsers too, and the app puts the cursor back in the
    /// field it last had.
    public func returnTo(_ origin: Origin) -> Bool {
        if atOrigin(origin) { return true }
        guard let handle = origin.handle as? OriginHandle else { return false }
        AX.set(handle.app, kAXFrontmostAttribute, kCFBooleanTrue)
        if let window = handle.window {
            AX.set(window, kAXMainAttribute, kCFBooleanTrue)
            AXUIElementPerformAction(window, kAXRaiseAction as CFString)
        }
        let deadline = HostClock.now + Self.returnTimeout
        while HostClock.now < deadline {
            if atOrigin(origin) {
                usleep(Self.returnSettle)
                return true
            }
            usleep(20_000)
        }
        return false
    }

    /// The selected text, found by sending ⌘C and seeing whether the clipboard changed; the
    /// clipboard is put back as it was either way.
    public func selectedText() -> String? {
        Pasteboard.lock.withLock { probeSelection() }
    }

    private func probeSelection() -> String? {
        let board = NSPasteboard.general
        let saved = Self.snapshot(board)
        let before = board.changeCount
        guard let down = CGEvent(keyboardEventSource: source, virtualKey: 8, keyDown: true),
              let up = CGEvent(keyboardEventSource: source, virtualKey: 8, keyDown: false) else { return nil }
        down.flags = .maskCommand
        up.flags = .maskCommand
        down.post(tap: .cghidEventTap)
        up.post(tap: .cghidEventTap)
        // Most apps copy within a few milliseconds; nothing selected means nothing changes.
        var copied: String?
        let deadline = HostClock.now + 0.25
        while HostClock.now < deadline {
            usleep(10_000)
            if board.changeCount != before {
                usleep(10_000)          // let the app finish writing every type
                copied = board.string(forType: .string)
                break
            }
        }
        if board.changeCount != before { Self.restore(saved, to: board) }
        guard let copied, !copied.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        return copied
    }

    public func wakeFrontmostApp() {
        if let app = NSWorkspace.shared.frontmostApplication { AX.wake(app.processIdentifier) }
    }

    static func snapshot(_ board: NSPasteboard) -> [[NSPasteboard.PasteboardType: Data]] {
        (board.pasteboardItems ?? []).map { item in
            Dictionary(uniqueKeysWithValues: item.types.compactMap { type in item.data(forType: type).map { (type, $0) } })
        }
    }

    static func restore(_ items: [[NSPasteboard.PasteboardType: Data]], to board: NSPasteboard) {
        board.clearContents()
        let restored = items.map { saved -> NSPasteboardItem in
            let item = NSPasteboardItem()
            for (type, data) in saved { item.setData(data, forType: type) }
            return item
        }
        if !restored.isEmpty { board.writeObjects(restored) }
    }
}

/// A value that is safe to hand across threads though its type doesn't say so.
struct Unchecked<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}

/// The accessibility elements behind an Origin.
final class OriginHandle: @unchecked Sendable {
    let app: AXUIElement
    let window: AXUIElement?
    init(app: AXUIElement, window: AXUIElement?) { self.app = app; self.window = window }
}

/// After typing, watch the destination field and report what the user did to our text.
///
/// Right after typing, the field's whole value is snapshotted and our span located in it;
/// later reads keep only changes that land inside that span (see EditDiff).
public final class MacEditWatcher: EditWatching, @unchecked Sendable {
    static let checkDelays: [Double] = [8, 25, 60]
    /// Let the app finish processing our keystrokes before the baseline.
    static let baselineSettle = 0.4
    /// A changed field must hold still this long before it counts: a field still changing is
    /// someone mid-keystroke, and the 8s check once caught partial words.
    static let settleConfirm = 2.5
    /// A tree that was only just woken can hand back a placeholder element first.
    static let unreadableRetries = 3
    static let unreadableRetryDelay = 0.3

    public init() {}

    public func watch(typed: String, app: String,
                      onResult: @escaping @Sendable (EditDiff.Outcome, Double, String, String) -> Void) {
        guard !typed.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, AX.trusted else { return }
        guard let target = AX.focused() else { return }
        let pid = AX.pid(target)
        let tag = Log.tag
        // Accessibility elements can be used from any thread; they just aren't marked Sendable.
        let element = Unchecked(target)
        DispatchQueue.global(qos: .utility).async {
            Log.$tag.withValue(tag) {
                Self.run(target: element.value, pid: pid, typed: typed, app: app, onResult: onResult)
            }
        }
    }

    static func run(target: AXUIElement, pid: pid_t?, typed: String, app: String,
                    onResult: (EditDiff.Outcome, Double, String, String) -> Void) {
        var target = target
        Thread.sleep(forTimeInterval: baselineSettle)
        var baseline = AX.string(target, kAXValueAttribute)
        for _ in 0..<unreadableRetries where baseline == nil {
            Thread.sleep(forTimeInterval: unreadableRetryDelay)
            target = AX.focused() ?? target
            baseline = AX.string(target, kAXValueAttribute)
        }
        guard let baseline else {
            // Say which app, so the gap can be closed.
            let role = AX.string(target, kAXRoleAttribute) ?? "none"
            Log.line("[Feedback] Can't read the field in \(app.isEmpty ? "the focused app" : app) (role \(role))")
            onResult(.unreadable, 0, typed, "")
            return
        }
        guard let span = EditDiff.anchor(baseline, typed) else {
            let preview = String(baseline.trimmingCharacters(in: .whitespacesAndNewlines)
                .replacingOccurrences(of: "\n", with: " ").prefix(90))
            Log.line("[Feedback] Could not anchor output in field (field has \(baseline.count) chars: \"\(preview)\")")
            onResult(.unanchored, 0, typed, "")
            return
        }
        var elapsed = baselineSettle
        for delay in checkDelays {
            Thread.sleep(forTimeInterval: max(0, delay - elapsed))
            elapsed = delay
            var current: String?
            (target, current) = readField(target, pid)
            let (outcome, corrected) = EditDiff.classify(baseline, current, span)
            if outcome == .unchanged { continue }
            Thread.sleep(forTimeInterval: settleConfirm)
            elapsed += settleConfirm
            (target, current) = readField(target, pid)
            let again = EditDiff.classify(baseline, current, span)
            if again.0 != outcome || again.1 != corrected { continue }
            onResult(outcome, elapsed, typed, corrected)
            return
        }
        onResult(.unchanged, checkDelays.last!, typed, "")
    }

    /// Read the watched field; nil when it's gone.
    ///
    /// Some apps destroy the composer and build a new one when a message is sent. The saved
    /// element then goes stale, and whatever now has focus in the same app is its successor:
    /// after a send the fresh empty composer, after a mere re-render the edited field. Focus
    /// in another app means the field is truly gone.
    static func readField(_ target: AXUIElement, _ pid: pid_t?) -> (AXUIElement, String?) {
        if let value = AX.string(target, kAXValueAttribute) { return (target, value) }
        guard let pid, let successor = AX.focused(), AX.pid(successor) == pid,
              let value = AX.string(successor, kAXValueAttribute) else { return (target, nil) }
        return (successor, value)
    }
}
