import AppKit
import CoreGraphics
import Foundation
import MergeScribeCore
import UserNotifications

/// Typing, the clipboard, notifications and sounds.
public final class MacTextOutput: TextOutput, @unchecked Sendable {
    /// How many characters one key event carries. The unicode-string API drops characters
    /// when handed too much at once; 20 stays comfortably inside its limit.
    static let charactersPerEvent = 20
    static let pauseBetweenEvents: useconds_t = 2_000

    /// Our own event source, so a modifier the user is holding can't turn text into shortcuts.
    private let source = CGEventSource(stateID: .privateState)

    /// Type at the cursor with synthetic unicode key events. Called once per streamed token,
    /// so it must cost microseconds, not a process launch. Whatever can't be typed goes to
    /// the clipboard: retyping the whole text another way would repeat what already went in.
    /// Insert through accessibility where the field takes it (native text fields); keystrokes
    /// everywhere else.
    public let viaAccessibility: Locked<Bool>

    public init(viaAccessibility: Bool = true) { self.viaAccessibility = Locked(viaAccessibility) }

    public func type(_ text: String) {
        if viaAccessibility.current && AXInsert.insert(text) { return }
        let characters = Array(text)
        var start = 0
        while start < characters.count {
            let piece = String(characters[start..<min(start + Self.charactersPerEvent, characters.count)])
            // UTF-16 units, not characters: an emoji is two, and a count of one types half of it.
            let units = Array(piece.utf16)
            guard let down = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: true),
                  let up = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: false) else {
                handOver(String(characters[start...]), reason: "key events could not be created")
                return
            }
            for event in [down, up] {
                event.flags = []
                event.keyboardSetUnicodeString(stringLength: units.count, unicodeString: units)
                event.post(tap: .cghidEventTap)
            }
            start += Self.charactersPerEvent
            if start < characters.count { usleep(Self.pauseBetweenEvents) }
        }
    }

    private func handOver(_ rest: String, reason: String) {
        Log.line("type_text: \(reason); the rest is on the clipboard")
        copyToClipboard(rest)
        notify("Couldn't type everything - the rest is on the clipboard")
    }

    /// Straight to the pasteboard from whichever thread: hopping to the main thread and
    /// waiting could deadlock against a main thread that is itself waiting on the session.
    public func copyToClipboard(_ text: String) {
        guard !text.isEmpty else { return }
        Pasteboard.lock.withLock {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
        }
    }

    public func notify(_ message: String) {
        Notifications.post(message)
    }

    public func playBusySound() {
        DispatchQueue.main.async { NSSound(named: "Basso")?.play() }
    }
}

/// The pasteboard is shared by overlapping dictations: one may be handing text over while the
/// next probes for a selection with ⌘C. Each takes this lock, so the probe can't mistake the
/// hand-over for a selection, or put the old contents back over it.
enum Pasteboard {
    static let lock = NSLock()
}

/// macOS notifications. They need an app bundle; an unbundled development build logs instead.
public enum Notifications {
    static var available: Bool { Bundle.main.bundleIdentifier != nil && Bundle.main.bundleURL.pathExtension == "app" }

    public static func requestPermission() {
        guard available else { return }
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { _, _ in }
    }

    public static func post(_ message: String, title: String = "MergeScribe") {
        guard available else {
            Log.line("[Notify] \(title): \(message)")
            return
        }
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = message
        UNUserNotificationCenter.current().add(UNNotificationRequest(identifier: UUID().uuidString, content: content,
                                                                     trigger: nil))
    }
}

/// The trigger key and Shift+Esc, from a listen-only event tap.
///
/// Modifier keys arrive as flagsChanged events; which physical key moved is in the event's
/// keycode, and whether it is now down is in the device-dependent flag bits, so Right Command
/// is told apart from Left Command.
public final class Hotkey: @unchecked Sendable {
    public struct Key: Sendable, Equatable {
        public let code: CGKeyCode
        /// For a modifier: the device-dependent flag bit that says it is down.
        public let modifierBit: UInt64?
    }

    /// The trigger keys Settings offers, plus a few more a hand-edited file might name.
    public static let keys: [String: Key] = [
        "cmd_r": Key(code: 54, modifierBit: 0x10), "cmd_l": Key(code: 55, modifierBit: 0x08),
        "cmd": Key(code: 55, modifierBit: 0x08),
        "alt_r": Key(code: 61, modifierBit: 0x40), "alt_l": Key(code: 58, modifierBit: 0x20),
        "alt": Key(code: 58, modifierBit: 0x20),
        "ctrl_r": Key(code: 62, modifierBit: 0x2000), "ctrl_l": Key(code: 59, modifierBit: 0x01),
        "ctrl": Key(code: 59, modifierBit: 0x01),
        "shift_r": Key(code: 60, modifierBit: 0x04), "shift_l": Key(code: 56, modifierBit: 0x02),
        "f13": Key(code: 105, modifierBit: nil), "f14": Key(code: 107, modifierBit: nil),
        "f15": Key(code: 113, modifierBit: nil), "f16": Key(code: 106, modifierBit: nil),
        "f17": Key(code: 64, modifierBit: nil), "f18": Key(code: 79, modifierBit: nil),
        "f19": Key(code: 80, modifierBit: nil), "f20": Key(code: 90, modifierBit: nil),
    ]
    static let escape: CGKeyCode = 53
    static let shiftCodes: Set<CGKeyCode> = [56, 60]

    public var trigger: Key
    private let machine: InputStateMachine
    private var tap: CFMachPort?
    private var source: CFRunLoopSource?
    private var loop: CFRunLoop?

    public init?(trigger name: String, machine: InputStateMachine) {
        guard let key = Self.keys[name.lowercased()] else { return nil }
        trigger = key
        self.machine = machine
    }

    /// Start listening on the main run loop. False without Input Monitoring permission.
    public func start() -> Bool {
        let mask = (1 << CGEventType.keyDown.rawValue) | (1 << CGEventType.keyUp.rawValue)
            | (1 << CGEventType.flagsChanged.rawValue)
        let callback: CGEventTapCallBack = { _, type, event, refcon in
            let hotkey = Unmanaged<Hotkey>.fromOpaque(refcon!).takeUnretainedValue()
            hotkey.receive(type, event)
            return Unmanaged.passUnretained(event)
        }
        guard let tap = CGEvent.tapCreate(tap: .cgSessionEventTap, place: .headInsertEventTap, options: .listenOnly,
                                          eventsOfInterest: CGEventMask(mask), callback: callback,
                                          userInfo: Unmanaged.passUnretained(self).toOpaque()) else {
            return false
        }
        self.tap = tap
        let source = CFMachPortCreateRunLoopSource(nil, tap, 0)
        self.source = source
        // Its own thread and run loop: the tap must answer promptly even while the main thread
        // is busy (opening a microphone, say), or macOS switches it off and a release is lost.
        let ready = DispatchSemaphore(value: 0)
        let box = Unchecked((tap, source))
        let thread = Thread { [weak self] in
            let loop = CFRunLoopGetCurrent()
            self?.loop = loop
            CFRunLoopAddSource(loop, box.value.1, .commonModes)
            CGEvent.tapEnable(tap: box.value.0, enable: true)
            ready.signal()
            CFRunLoopRun()
        }
        thread.name = "MergeScribe hotkey"
        thread.qualityOfService = .userInteractive
        thread.start()
        ready.wait()
        return true
    }

    public func stop() {
        if let tap { CGEvent.tapEnable(tap: tap, enable: false) }
        if let loop {
            if let source { CFRunLoopRemoveSource(loop, source, .commonModes) }
            CFRunLoopStop(loop)
        }
        tap = nil
        source = nil
        loop = nil
    }

    /// In the tap's callback, which must return quickly: macOS switches off a tap that keeps
    /// it waiting, and starting a recording can take a moment. Read what is needed from the
    /// event and handle it just after.
    func receive(_ type: CGEventType, _ event: CGEvent) {
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            // macOS switched the tap off (it was slow, or secure input was on); switch it back on.
            if let tap { CGEvent.tapEnable(tap: tap, enable: true) }
            return
        }
        let code = CGKeyCode(event.getIntegerValueField(.keyboardEventKeycode))
        let flags = event.flags.rawValue
        DispatchQueue.main.async { [self] in handle(type, code: code, flags: flags) }
    }

    func handle(_ type: CGEventType, code: CGKeyCode, flags: UInt64) {
        switch type {
        case .flagsChanged:
            if Self.shiftCodes.contains(code) {
                machine.shift(down: flags & (0x02 | 0x04) != 0)
            }
            if code == trigger.code, let bit = trigger.modifierBit {
                if flags & bit != 0 { machine.triggerPressed() } else { machine.triggerReleased() }
            }
        case .keyDown:
            if code == trigger.code && trigger.modifierBit == nil { machine.triggerPressed() }
            if code == Self.escape { machine.escapePressed() }
        case .keyUp:
            if code == trigger.code && trigger.modifierBit == nil { machine.triggerReleased() }
        default:
            break
        }
    }
}
