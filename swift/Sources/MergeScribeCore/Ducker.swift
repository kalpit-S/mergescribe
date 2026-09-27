import Foundation

/// The system's default output device, as far as turning it down goes.
public protocol OutputDevice: Sendable {
    func device() -> UInt32?
    /// Transport and data source, as four-character codes ("bltn", "ispk").
    func kind(_ device: UInt32) -> (transport: String, source: String)
    func volume(_ device: UInt32) -> Double?
    func setVolume(_ device: UInt32, _ value: Double)
}

/// Other audio turned down while dictating, when it plays out loud.
///
/// Speakers leak into the microphone and headphones don't, so while the key is held the
/// system output is lowered if it goes to speakers, and put back on release. It is put back
/// only if nobody changed it meanwhile: reaching for the volume mid-dictation is a choice.
public final class Ducker: @unchecked Sendable {
    public static let duckTo = 0.25     // of the volume it was at

    /// Transports that play into the room. Built-in output depends on its data source (the
    /// MacBook's speakers, not its headphone jack). Bluetooth and USB are left alone: usually
    /// headphones or a headset, and a guess either way.
    public static func playsOutLoud(transport: String, source: String) -> Bool {
        transport == "bltn" ? source == "ispk" : ["hdmi", "dprt", "airp"].contains(transport)
    }

    private let output: OutputDevice?
    private let lock = NSLock()
    private var saved: (device: UInt32, before: Double, quiet: Double)?

    public init(output: OutputDevice?) { self.output = output }

    public func duck() {
        lock.lock(); defer { lock.unlock() }
        guard let output, saved == nil, let device = output.device() else { return }
        let kind = output.kind(device)
        guard Self.playsOutLoud(transport: kind.transport, source: kind.source),
              let before = output.volume(device), before > 0 else { return }
        output.setVolume(device, before * Self.duckTo)
        // Read back what the device settled on: some step their volume.
        saved = (device, before, output.volume(device) ?? before * Self.duckTo)
    }

    public func restore() {
        lock.lock()
        let saved = self.saved
        self.saved = nil
        lock.unlock()
        guard let saved, let output, let now = output.volume(saved.device) else { return }
        if abs(now - saved.quiet) < 0.01 { output.setVolume(saved.device, saved.before) }
    }
}
