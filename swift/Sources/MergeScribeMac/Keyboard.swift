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

    public init() {}

    /// Type at the cursor with synthetic unicode key events. Called once per streamed token,
    /// so it must cost microseconds, not a process launch. Whatever can't be typed goes to
    /// the clipboard: retyping the whole text another way would repeat what already went in.
    public func type(_ text: String) {
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

    public func copyToClipboard(_ text: String) {
        guard !text.isEmpty else { return }
        DispatchQueue.main.sync(execute: {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
        }, ifNotMain: true)
    }

    public func notify(_ message: String) {
        Notifications.post(message)
    }

    public func playBusySound() {
        DispatchQueue.main.async { NSSound(named: "Basso")?.play() }
    }
}

extension DispatchQueue {
    /// Run on the main queue, synchronously, without deadlocking when already on it.
    func sync(execute work: () -> Void, ifNotMain: Bool) {
        if Thread.isMainThread { work() } else { DispatchQueue.main.sync(execute: work) }
    }
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
            hotkey.handle(type, event)
            return Unmanaged.passUnretained(event)
        }
        guard let tap = CGEvent.tapCreate(tap: .cgSessionEventTap, place: .headInsertEventTap, options: .listenOnly,
                                          eventsOfInterest: CGEventMask(mask), callback: callback,
                                          userInfo: Unmanaged.passUnretained(self).toOpaque()) else {
            return false
        }
        self.tap = tap
        source = CFMachPortCreateRunLoopSource(nil, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
        return true
    }

    public func stop() {
        if let tap { CGEvent.tapEnable(tap: tap, enable: false) }
        if let source { CFRunLoopRemoveSource(CFRunLoopGetMain(), source, .commonModes) }
        tap = nil
        source = nil
    }

    func handle(_ type: CGEventType, _ event: CGEvent) {
        switch type {
        case .tapDisabledByTimeout, .tapDisabledByUserInput:
            // macOS switches a tap off when it is slow or on secure input; switch it back on.
            if let tap { CGEvent.tapEnable(tap: tap, enable: true) }
        case .flagsChanged:
            let code = CGKeyCode(event.getIntegerValueField(.keyboardEventKeycode))
            let flags = event.flags.rawValue
            if Self.shiftCodes.contains(code) {
                machine.shift(down: flags & (0x02 | 0x04) != 0)
            }
            if code == trigger.code, let bit = trigger.modifierBit {
                if flags & bit != 0 { machine.triggerPressed() } else { machine.triggerReleased() }
            }
        case .keyDown:
            let code = CGKeyCode(event.getIntegerValueField(.keyboardEventKeycode))
            if code == trigger.code && trigger.modifierBit == nil { machine.triggerPressed() }
            if code == Self.escape { machine.escapePressed() }
        case .keyUp:
            let code = CGKeyCode(event.getIntegerValueField(.keyboardEventKeycode))
            if code == trigger.code && trigger.modifierBit == nil { machine.triggerReleased() }
        default:
            break
        }
    }
}
