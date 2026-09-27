@preconcurrency import AVFoundation
import AudioToolbox
import Foundation
import MergeScribeCore

/// Seconds on the host clock, the one Core Audio timestamps buffers with.
public enum HostClock {
    private static let timebase: mach_timebase_info_data_t = {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return info
    }()

    public static func seconds(_ hostTime: UInt64) -> Double {
        Double(hostTime) * Double(timebase.numer) / Double(timebase.denom) / 1e9
    }

    public static var now: Double { seconds(mach_absolute_time()) }
}

/// One microphone: its own engine, its hardware format converted to 16 kHz mono, handed
/// on in 1024-sample blocks stamped with when they were heard.
final class MicCapture: @unchecked Sendable {
    let name: String
    let device: CoreAudioDevices.Device
    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private let target: AVAudioFormat
    private var pending: [Float] = []
    private var pendingStart: Double = 0     // when the first pending sample was heard
    private let onBlock: @Sendable ([Float], Double) -> Void

    init(name: String, device: CoreAudioDevices.Device, sampleRate: Int, onBlock: @escaping @Sendable ([Float], Double) -> Void) {
        self.name = name
        self.device = device
        self.onBlock = onBlock
        target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false)!
    }

    var isRunning: Bool { engine.isRunning }

    func start() throws {
        let input = engine.inputNode
        // The input node records the default device unless its I/O unit is pointed elsewhere.
        guard let unit = input.audioUnit else { throw CaptureError.noAudioUnit }
        var id = device.id
        let status = AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0,
                                          &id, UInt32(MemoryLayout<AudioDeviceID>.size))
        guard status == noErr else { throw CaptureError.deviceRefused(status) }
        let format = input.inputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else { throw CaptureError.noFormat }
        guard let converter = AVAudioConverter(from: format, to: target) else { throw CaptureError.noConverter }
        converter.downmix = true
        self.converter = converter

        let ratio = target.sampleRate / format.sampleRate
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, time in
            guard let self else { return }
            // The tap runs on AVAudioEngine's own thread, not the real-time one, so converting
            // and allocating here is safe.
            let heard = time.isHostTimeValid ? HostClock.seconds(time.hostTime) : HostClock.now
            let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 32
            guard let out = AVAudioPCMBuffer(pcmFormat: self.target, frameCapacity: capacity) else { return }
            var fed = false
            var error: NSError?
            converter.convert(to: out, error: &error) { _, status in
                if fed { status.pointee = .noDataNow; return nil }
                fed = true
                status.pointee = .haveData
                return buffer
            }
            guard error == nil, let data = out.floatChannelData?[0], out.frameLength > 0 else { return }
            self.append(Array(UnsafeBufferPointer(start: data, count: Int(out.frameLength))), heardAt: heard)
        }
        engine.prepare()
        try engine.start()
    }

    /// Collect converted audio into fixed blocks, each stamped with when its first sample was heard.
    private func append(_ samples: [Float], heardAt: Double) {
        if pending.isEmpty { pendingStart = heardAt }
        pending += samples
        let blockSeconds = Double(Chunker.blockSize) / target.sampleRate
        while pending.count >= Chunker.blockSize {
            let block = Array(pending.prefix(Chunker.blockSize))
            pending.removeFirst(Chunker.blockSize)
            onBlock(block, pendingStart + blockSeconds)
            pendingStart += blockSeconds
        }
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
    }

    enum CaptureError: Error { case noAudioUnit, deviceRefused(OSStatus), noFormat, noConverter }
}

/// Every enabled microphone, recorded in parallel into one Chunker.
///
/// Blocks from all mics are processed in order on one serial queue, so chunking decisions,
/// the start and stop of a recording, and handing chunks to the session never race: a
/// release waits for whatever was heard before it.
public final class AudioEngine: @unchecked Sendable {
    private let processing = DispatchQueue(label: "mergescribe.audio.processing", qos: .userInteractive)
    private let devices = DispatchQueue(label: "mergescribe.audio.devices")
    private var chunker: Chunker                  // processing queue only
    private var onChunk: ((AudioChunk) -> Void)?  // processing queue only
    private let captures = Locked<[String: MicCapture]>([:])
    private let recording = Locked(false)
    private let currentLevel = Locked(0.0)
    private var monitor: DispatchSourceTimer?
    private var wanted: [String] = []

    public init(settings: Settings) {
        chunker = Chunker(config: Chunker.Config(settings))
    }

    /// 0...1, the primary mic's level, for the HUD.
    public var level: Double { currentLevel.current }

    /// The mics currently delivering audio, in the order they were opened.
    public var activeMics: [String] { processing.sync { chunker.micNames } }

    /// Apply the enabled mics and speech settings. Idle only: changing mics mid-recording
    /// would make a session's audio inconsistent; the next recording picks the change up.
    @discardableResult
    public func apply(_ settings: Settings) -> [String] {
        guard !recording.current else { return activeMics }
        processing.sync { chunker.config = Chunker.Config(settings) }
        devices.sync { wanted = Array(NSOrderedSet(array: settings.enabledMics)) as? [String] ?? settings.enabledMics }
        syncDevices()
        return activeMics
    }

    /// Watch for mics coming and going every few seconds.
    public func startMonitoring() {
        let timer = DispatchSource.makeTimerSource(queue: devices)
        timer.schedule(deadline: .now() + 3, repeating: 3)
        timer.setEventHandler { [weak self] in self?.syncDevicesOnQueue() }
        timer.resume()
        monitor = timer
    }

    private func syncDevices() { devices.sync { syncDevicesOnQueue() } }

    /// Close mics that are gone or no longer wanted; open wanted ones that aren't running.
    private func syncDevicesOnQueue() {
        guard !recording.current else { return }
        let available = CoreAudioDevices.inputDevices()
        for (name, capture) in captures.current {
            let unwanted = !wanted.contains(name)
            let dead = !capture.isRunning || !available.contains { $0.id == capture.device.id }
            if unwanted || dead {
                Log.line(unwanted ? "[Audio] Mic disabled in settings: \(name)" : "[Audio] Mic disconnected: \(name)")
                capture.stop()
                captures.update { $0[name] = nil }
                processing.sync { chunker.removeMic(name); drainMessages() }
            }
        }
        for name in wanted where captures.read({ $0[name] == nil }) {
            guard let device = CoreAudioDevices.find(name, in: available) else { continue }
            let capture = MicCapture(name: name, device: device, sampleRate: 16000) { [weak self] block, heard in
                guard let self else { return }
                self.processing.async { self.feed(name, block, heard) }
            }
            do {
                processing.sync { chunker.addMic(name) }
                try capture.start()
                captures.update { $0[name] = capture }
                Log.line("[Audio] Mic connected: \(name) (\(device.name))")
            } catch {
                processing.sync { chunker.removeMic(name) }
                Log.line("[Audio] Failed to open \(name): \(error)")
            }
        }
    }

    // MARK: - Recording (any thread)

    /// Begin a recording. Chunks emitted on pauses go to `onChunk` on the audio queue.
    public func startRecording(onChunk: @escaping (AudioChunk) -> Void) {
        recording.update { $0 = true }
        let now = HostClock.now
        processing.sync {
            self.onChunk = onChunk
            chunker.startRecording(now: now)
            drainMessages()
        }
    }

    /// End the recording: the rest of the audio, after every block heard before now.
    public func stopRecording() -> FinalChunk {
        let final = processing.sync { () -> FinalChunk in
            onChunk = nil
            let final = chunker.stopRecording()
            drainMessages()
            return final
        }
        recording.update { $0 = false }
        return final
    }

    public func shutdown() {
        monitor?.cancel()
        for capture in captures.current.values { capture.stop() }
        captures.update { $0.removeAll() }
    }

    // MARK: - Audio queue

    private func feed(_ mic: String, _ block: [Float], _ heard: Double) {
        let chunk = chunker.feed(mic, block, now: heard)
        currentLevel.update { $0 = chunker.level }
        drainMessages()
        if let chunk, let onChunk { onChunk(chunk) }
    }

    private func drainMessages() {
        for message in chunker.messages { Log.line(message) }
        chunker.messages.removeAll()
    }
}
