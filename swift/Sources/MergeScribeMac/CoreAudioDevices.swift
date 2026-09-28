import CoreAudio
import Foundation
import MergeScribeCore

/// Audio devices through Core Audio's property API.
public enum CoreAudioDevices {
    public struct Device: Equatable, Sendable {
        public let id: AudioDeviceID
        public let name: String
        public let uid: String
        public let inputChannels: Int
    }

    static func address(_ selector: AudioObjectPropertySelector,
                        _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal,
                        _ element: AudioObjectPropertyElement = kAudioObjectPropertyElementMain) -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: element)
    }

    static func get<T>(_ object: AudioObjectID, _ address: AudioObjectPropertyAddress, _ initial: T) -> T? {
        var address = address
        var value = initial
        var size = UInt32(MemoryLayout<T>.size)
        let status = withUnsafeMutablePointer(to: &value) {
            AudioObjectGetPropertyData(object, &address, 0, nil, &size, $0)
        }
        return status == noErr ? value : nil
    }

    static func string(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
        var address = address(selector)
        var value: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        let status = AudioObjectGetPropertyData(object, &address, 0, nil, &size, &value)
        guard status == noErr, let value else { return nil }
        return value.takeRetainedValue() as String
    }

    /// The rate a device is running at; 0 when it can't be read.
    public static func nominalRate(_ device: AudioDeviceID) -> Double {
        get(device, address(kAudioDevicePropertyNominalSampleRate), Float64(0)) ?? 0
    }

    /// Every device that can record.
    public static func inputDevices() -> [Device] {
        var address = address(kAudioHardwarePropertyDevices)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size) == noErr else {
            return []
        }
        var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &ids) == noErr else {
            return []
        }
        return ids.compactMap { id in
            let channels = inputChannels(id)
            guard channels > 0 else { return nil }
            return Device(id: id, name: string(id, kAudioObjectPropertyName) ?? "Device \(id)",
                          uid: string(id, kAudioDevicePropertyDeviceUID) ?? "", inputChannels: channels)
        }
    }

    static func inputChannels(_ device: AudioDeviceID) -> Int {
        var address = address(kAudioDevicePropertyStreamConfiguration, kAudioObjectPropertyScopeInput)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(device, &address, 0, nil, &size) == noErr, size > 0 else { return 0 }
        let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment)
        defer { raw.deallocate() }
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, raw) == noErr else { return 0 }
        let list = UnsafeMutableAudioBufferListPointer(raw.assumingMemoryBound(to: AudioBufferList.self))
        return list.reduce(0) { $0 + Int($1.mNumberChannels) }
    }

    /// A configured mic name to a device: exact (ignoring case), then either name containing
    /// the other. Names change subtly across macOS versions and adapters.
    public static func find(_ name: String, in devices: [Device] = inputDevices()) -> Device? {
        let wanted = name.lowercased()
        return devices.first { $0.name.lowercased() == wanted }
            ?? devices.first { $0.name.lowercased().contains(wanted) }
            ?? devices.first { wanted.contains($0.name.lowercased()) }
    }
}

/// The default output device, for ducking.
public struct CoreAudioOutput: OutputDevice {
    public init() {}

    public func device() -> UInt32? {
        CoreAudioDevices.get(AudioObjectID(kAudioObjectSystemObject),
                             CoreAudioDevices.address(kAudioHardwarePropertyDefaultOutputDevice), AudioDeviceID(0))
    }

    public func kind(_ device: UInt32) -> (transport: String, source: String) {
        let transport = CoreAudioDevices.get(device, CoreAudioDevices.address(kAudioDevicePropertyTransportType), UInt32(0))
        let source = CoreAudioDevices.get(device, CoreAudioDevices.address(kAudioDevicePropertyDataSource,
                                                                           kAudioObjectPropertyScopeOutput), UInt32(0))
        return (transport.map(Self.fourCC) ?? "", source.map(Self.fourCC) ?? "")
    }

    public func volume(_ device: UInt32) -> Double? {
        CoreAudioDevices.get(device, Self.volumeAddress, Float32(0)).map(Double.init)
    }

    public func setVolume(_ device: UInt32, _ value: Double) {
        var address = Self.volumeAddress
        var level = Float32(max(0, min(1, value)))
        AudioObjectSetPropertyData(device, &address, 0, nil, UInt32(MemoryLayout<Float32>.size), &level)
    }

    /// The main volume, as the volume keys set it ('vmvc').
    static let volumeAddress = CoreAudioDevices.address(0x766D_7663, kAudioObjectPropertyScopeOutput)

    static func fourCC(_ code: UInt32) -> String {
        String(bytes: [24, 16, 8, 0].map { UInt8((code >> $0) & 0xFF) }, encoding: .isoLatin1) ?? ""
    }
}
