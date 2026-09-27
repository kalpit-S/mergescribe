import Foundation
import Testing
@testable import MergeScribeCore

/// A default output device: its kind, and a volume that may step.
final class FakeDevice: OutputDevice, @unchecked Sendable {
    var kindValue: (String, String)
    var level: Double
    let step: Double
    init(transport: String = "bltn", source: String = "ispk", volume: Double = 0.5, step: Double = 0) {
        kindValue = (transport, source); level = volume; self.step = step
    }
    func device() -> UInt32? { 76 }
    func kind(_ device: UInt32) -> (transport: String, source: String) { kindValue }
    func volume(_ device: UInt32) -> Double? { level }
    func setVolume(_ device: UInt32, _ value: Double) { level = step > 0 ? (value / step).rounded() * step : value }
}

@Suite struct DuckerTests {
    @Test(arguments: [("bltn", "ispk", true), ("bltn", "hdpn", false), ("hdmi", "", true), ("dprt", "", true),
                      ("airp", "", true), ("blue", "", false), ("usb ", "", false)])
    func whatCountsAsPlayingOutLoud(transport: String, source: String, loud: Bool) {
        #expect(Ducker.playsOutLoud(transport: transport, source: source) == loud)
    }

    @Test func speakersAreTurnedDownAndBackUp() {
        let device = FakeDevice(volume: 0.6)
        let ducker = Ducker(output: device)
        ducker.duck()
        #expect(abs(device.level - 0.6 * Ducker.duckTo) < 1e-9)
        ducker.restore()
        #expect(abs(device.level - 0.6) < 1e-9)
    }

    @Test func headphonesAreLeftAlone() {
        let device = FakeDevice(transport: "blue", volume: 0.6)
        Ducker(output: device).duck()
        #expect(device.level == 0.6)
    }

    @Test func aVolumeChangedMidDictationIsKept() {
        // Reaching for the volume while talking is a choice; restoring over it would undo it.
        let device = FakeDevice(volume: 0.6)
        let ducker = Ducker(output: device)
        ducker.duck()
        device.level = 0.9
        ducker.restore()
        #expect(device.level == 0.9)
    }

    @Test func aDeviceThatStepsItsVolumeStillRestores() {
        let device = FakeDevice(volume: 0.5, step: 1.0 / 16)
        let ducker = Ducker(output: device)
        ducker.duck()
        ducker.restore()
        #expect(device.level == 0.5)
    }

    @Test func duckingTwiceRestoresTheOriginal() {
        // A toggle-mode press while already ducked must not duck the ducked volume.
        let device = FakeDevice(volume: 0.8)
        let ducker = Ducker(output: device)
        ducker.duck(); ducker.duck(); ducker.restore()
        #expect(abs(device.level - 0.8) < 1e-9)
    }

    @Test func mutedOrMissingOutputIsANoOp() {
        let silent = FakeDevice(volume: 0)
        Ducker(output: silent).duck()
        #expect(silent.level == 0)
        Ducker(output: nil).restore()
    }
}
