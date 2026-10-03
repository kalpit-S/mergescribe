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

/// One microphone, recorded through its own input-only HAL unit and handed on as 16 kHz
/// mono in 1024-sample blocks, each stamped with when it was heard.
///
/// The unit is pointed at this device before it is initialized and has no output side, so the
/// system's default input and output never touch it. They move when headphones connect, and a
/// Bluetooth headset drops into its 16 kHz call mode as soon as anything opens its mic.
/// AVAudioEngine couldn't promise that: its input shares the engine's I/O unit with the default
/// output, and with a headset connected it failed to start (-10868) or ran silent.
///
/// Threads: prepare, start, pause, flush and stop come from one serial queue. The real-time
/// callback renders, mixes to mono, and copies into `ring`; the drain thread copies out,
/// converts to the target rate and cuts blocks. The two share `ringLock` only for the copies.
final class MicCapture: @unchecked Sendable {
    let name: String
    let device: CoreAudioDevices.Device
    private let target: AVAudioFormat
    private let onBlock: @Sendable ([Float], Double) -> Void

    // Built and torn down on the control queue, only while no callback or drain can run.
    private var unit: AudioUnit?
    private var deviceRate = 0.0
    private var channels = 1
    private var converter: AVAudioConverter?
    private var scratch: UnsafeMutablePointer<Float>?
    private var scratchFrames = 0

    private static let ringCapacity = 96_000          // mono samples: 2 s at 48 kHz
    private let ring = UnsafeMutablePointer<Float>.allocate(capacity: ringCapacity)
    private let taken = UnsafeMutablePointer<Float>.allocate(capacity: ringCapacity)   // the drain's copy
    private let ringLock: UnsafeMutablePointer<os_unfair_lock> = {
        let lock = UnsafeMutablePointer<os_unfair_lock>.allocate(capacity: 1)
        lock.initialize(to: os_unfair_lock())
        return lock
    }()
    // Under ringLock.
    private var written = 0           // samples ever written
    private var read = 0              // samples ever taken
    private var stampSample = 0       // the latest callback's first sample
    private var stampTime = 0.0       // and when it was heard
    private var restart = false       // a new start: the drain drops what it held
    private var generation = 0        // bumped when a unit goes, so its drain thread ends
    private var flushTo: Int?         // deliver everything written before this, part block included
    private var tooLong = false       // the device sent more than a slice holds

    private let wake = DispatchSemaphore(value: 0)
    private let flushed = DispatchSemaphore(value: 0)
    private var drained: DispatchSemaphore?   // signalled when the current drain thread has ended

    // Drain thread only.
    private var pending: [Float] = []
    private var pendingStart = 0.0

    init(name: String, device: CoreAudioDevices.Device, sampleRate: Int, onBlock: @escaping @Sendable ([Float], Double) -> Void) {
        self.name = name
        self.device = device
        self.onBlock = onBlock
        target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false)!
        _ = HostClock.now   // its timebase is read once, not on the real-time thread
    }

    deinit {
        tearDown()
        ring.deallocate()
        taken.deallocate()
        ringLock.deallocate()
    }

    var isRunning: Bool {
        guard let unit else { return false }
        var running: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        return AudioUnitGetProperty(unit, kAudioOutputUnitProperty_IsRunning, kAudioUnitScope_Global, 0, &running, &size) == noErr
            && running != 0
    }

    /// Ready to start at once: the unit made and pointed at the device, nothing heard yet.
    func prepare() throws {
        if unit != nil, CoreAudioDevices.nominalRate(device.id) == deviceRate { return }
        tearDown()
        try build()
    }

    func start() throws {
        if isRunning { return }
        locked { restart = true; read = written }
        do {
            try prepare()
            try check(AudioOutputUnitStart(unit!))
        } catch {
            // Whatever state the device left the unit in, a fresh one gets one more try.
            tearDown()
            do {
                try build()
                try check(AudioOutputUnitStart(unit!))
            } catch {
                tearDown()
                throw error
            }
        }
    }

    /// Hand on everything heard so far, the last part-block included, before returning; the
    /// blocks are delivered through onBlock first. Bounded: a stuck device can't hold it up.
    func flush() {
        guard unit != nil else { return }
        while flushed.wait(timeout: .now()) == .success {}   // a late signal from a flush that timed out
        locked { flushTo = written }
        wake.signal()
        _ = flushed.wait(timeout: .now() + 0.25)
    }

    /// Stop listening, keeping the unit for the next start.
    func pause() {
        if let unit { AudioOutputUnitStop(unit) }
    }

    func stop() {
        tearDown()
    }

    enum CaptureError: Error { case noAudioUnit, failed(OSStatus), noFormat, noConverter }

    private func check(_ status: OSStatus) throws {
        guard status == noErr else { throw CaptureError.failed(status) }
    }

    private func locked<T>(_ body: () -> T) -> T {
        os_unfair_lock_lock(ringLock)
        defer { os_unfair_lock_unlock(ringLock) }
        return body()
    }

    private func build() throws {
        var description = AudioComponentDescription(componentType: kAudioUnitType_Output, componentSubType: kAudioUnitSubType_HALOutput,
                                                    componentManufacturer: kAudioUnitManufacturer_Apple, componentFlags: 0, componentFlagsMask: 0)
        guard let component = AudioComponentFindNext(nil, &description) else { throw CaptureError.noAudioUnit }
        var made: AudioUnit?
        try check(AudioComponentInstanceNew(component, &made))
        guard let unit = made else { throw CaptureError.noAudioUnit }
        do {
            var on: UInt32 = 1, off: UInt32 = 0
            let flagSize = UInt32(MemoryLayout<UInt32>.size)
            try check(AudioUnitSetProperty(unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, &on, flagSize))
            try check(AudioUnitSetProperty(unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, &off, flagSize))
            var id = device.id
            try check(AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0,
                                           &id, UInt32(MemoryLayout<AudioDeviceID>.size)))
            // The device's own format; the unit hands it over at the same rate, as float.
            var hardware = AudioStreamBasicDescription()
            var formatSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
            try check(AudioUnitGetProperty(unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 1, &hardware, &formatSize))
            guard hardware.mSampleRate > 0, hardware.mChannelsPerFrame > 0 else { throw CaptureError.noFormat }
            let channels = Int(hardware.mChannelsPerFrame)
            var client = AudioStreamBasicDescription(
                mSampleRate: hardware.mSampleRate, mFormatID: kAudioFormatLinearPCM,
                mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
                mBytesPerPacket: UInt32(4 * channels), mFramesPerPacket: 1, mBytesPerFrame: UInt32(4 * channels),
                mChannelsPerFrame: UInt32(channels), mBitsPerChannel: 32, mReserved: 0)
            try check(AudioUnitSetProperty(unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1, &client, formatSize))
            // The largest slice the device may send, as the unit settles it; scratch holds one.
            var maxFrames: UInt32 = 8192
            AudioUnitSetProperty(unit, kAudioUnitProperty_MaximumFramesPerSlice, kAudioUnitScope_Global, 0, &maxFrames, flagSize)
            var sliceSize = flagSize
            AudioUnitGetProperty(unit, kAudioUnitProperty_MaximumFramesPerSlice, kAudioUnitScope_Global, 0, &maxFrames, &sliceSize)
            var callback = AURenderCallbackStruct(inputProc: MicCapture.inputCallback,
                                                  inputProcRefCon: Unmanaged.passUnretained(self).toOpaque())
            try check(AudioUnitSetProperty(unit, kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global, 0,
                                           &callback, UInt32(MemoryLayout<AURenderCallbackStruct>.size)))
            let mono = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: hardware.mSampleRate, channels: 1, interleaved: false)!
            guard let converter = AVAudioConverter(from: mono, to: target) else { throw CaptureError.noConverter }
            try check(AudioUnitInitialize(unit))
            self.channels = channels
            deviceRate = hardware.mSampleRate
            self.converter = converter
            scratchFrames = Int(max(maxFrames, 8192))
            scratch = .allocate(capacity: scratchFrames * channels)
            self.unit = unit
        } catch {
            AudioComponentInstanceDispose(unit)
            throw error
        }
        let mine = locked { generation }
        let done = DispatchSemaphore(value: 0)
        drained = done
        let thread = Thread { [weak self] in
            self?.drain(mine)
            done.signal()
        }
        thread.name = "mergescribe.mic.\(name)"
        thread.qualityOfService = .userInteractive
        thread.start()
    }

    private func tearDown() {
        guard let unit else { return }
        AudioOutputUnitStop(unit)      // returns once no callback is running
        AudioUnitUninitialize(unit)
        AudioComponentInstanceDispose(unit)
        self.unit = nil
        locked { generation += 1 }
        wake.signal()
        // One drain thread at a time: the next build must not find this one still converting.
        // It ends within the batch it is on.
        drained?.wait()
        drained = nil
        scratch?.deallocate()
        scratch = nil
    }

    private static let inputCallback: AURenderCallback = { refCon, flags, timestamp, bus, frames, _ in
        Unmanaged<MicCapture>.fromOpaque(refCon).takeUnretainedValue().render(flags, timestamp, bus, frames)
    }

    /// The real-time thread: no allocation; the lock is held only to copy into the ring.
    private func render(_ flags: UnsafeMutablePointer<AudioUnitRenderActionFlags>, _ timestamp: UnsafePointer<AudioTimeStamp>,
                        _ bus: UInt32, _ frames: UInt32) -> OSStatus {
        guard let unit, let scratch else { return noErr }
        let count = Int(frames), channels = self.channels
        guard count <= scratchFrames else {
            locked { tooLong = true }
            return noErr
        }
        var list = AudioBufferList(mNumberBuffers: 1, mBuffers: AudioBuffer(
            mNumberChannels: UInt32(channels), mDataByteSize: frames * UInt32(4 * channels), mData: UnsafeMutableRawPointer(scratch)))
        let status = AudioUnitRender(unit, flags, timestamp, bus, frames, &list)
        guard status == noErr else { return status }
        let heard = timestamp.pointee.mFlags.contains(.hostTimeValid) ? HostClock.seconds(timestamp.pointee.mHostTime) : HostClock.now
        if channels > 1 {   // mixed down in place: frame i's average lands in slot i
            let scale = 1 / Float(channels)
            for i in 0..<count {
                var sum: Float = 0
                for c in 0..<channels { sum += scratch[i * channels + c] }
                scratch[i] = sum * scale
            }
        }
        os_unfair_lock_lock(ringLock)
        stampSample = written
        stampTime = heard
        let at = written % Self.ringCapacity
        let first = min(count, Self.ringCapacity - at)
        (ring + at).update(from: scratch, count: first)
        if first < count { ring.update(from: scratch + first, count: count - first) }
        written += count
        if written - read > Self.ringCapacity { read = written - Self.ringCapacity }   // the drain fell 2 s behind
        os_unfair_lock_unlock(ringLock)
        wake.signal()
        return noErr
    }

    /// The drain thread: converts what was heard to the target rate and cuts it into blocks.
    private func drain(_ mine: Int) {
        while true {
            wake.wait()
            os_unfair_lock_lock(ringLock)
            if generation != mine { os_unfair_lock_unlock(ringLock); return }
            let fresh = restart
            restart = false
            let count = written - read
            let at = read % Self.ringCapacity
            let first = min(count, Self.ringCapacity - at)
            taken.update(from: ring + at, count: first)
            if first < count { (taken + first).update(from: ring, count: count - first) }
            let firstHeard = stampTime - Double(stampSample - read) / deviceRate
            read = written
            let flushing = flushTo.map { read >= $0 } ?? false
            if flushing { flushTo = nil }
            let overran = tooLong
            tooLong = false
            os_unfair_lock_unlock(ringLock)

            if overran { Log.line("[Audio] \(name) sent more audio in one go than expected; some was dropped") }
            if fresh {
                pending.removeAll()
                converter?.reset()
            }
            if count > 0, let converted = convert(count) { append(converted, heardAt: firstHeard) }
            if flushing {
                if !pending.isEmpty { onBlock(pending, pendingStart + Double(pending.count) / target.sampleRate) }
                pending.removeAll()
                flushed.signal()
            }
        }
    }

    private func convert(_ count: Int) -> [Float]? {
        guard let converter,
              let input = AVAudioPCMBuffer(pcmFormat: converter.inputFormat, frameCapacity: AVAudioFrameCount(count)) else { return nil }
        input.frameLength = AVAudioFrameCount(count)
        input.floatChannelData![0].update(from: taken, count: count)
        let capacity = AVAudioFrameCount(Double(count) * target.sampleRate / converter.inputFormat.sampleRate) + 32
        guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return nil }
        let fed = Locked(false)
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if fed.current { status.pointee = .noDataNow; return nil }
            fed.update { $0 = true }
            status.pointee = .haveData
            return input
        }
        guard error == nil, let data = out.floatChannelData?[0], out.frameLength > 0 else { return nil }
        return Array(UnsafeBufferPointer(start: data, count: Int(out.frameLength)))
    }

    /// Collect converted audio into fixed blocks, each stamped with when its last sample was heard.
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
    private var onBlock: (([Float]) -> Void)?     // processing queue only: the primary mic, while recording
    private let captures = Locked<[String: MicCapture]>([:])
    private let recording = Locked(false)
    private let currentLevel = Locked(0.0)
    private let sampleRate: Locked<Int>
    private var monitor: DispatchSourceTimer?
    private var wanted: [String] = []
    /// Microphones open between dictations (for pre-roll), or only while recording.
    private let keepOpen = Locked(false)
    /// When the key went down with the microphones closed, to log how long they took to open.
    private var openedAt: Double?          // processing queue only
    private var heardSinceOpen: Set<String> = []

    public init(settings: Settings) {
        chunker = Chunker(config: Chunker.Config(settings))
        sampleRate = Locked(settings.sampleRate)
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
        let wasOpen = keepOpen.current
        keepOpen.update { $0 = settings.keepMicOpen }
        if wasOpen && !settings.keepMicOpen {
            devices.sync { for capture in captures.current.values { capture.pause() } }
        }
        if settings.sampleRate != sampleRate.current {
            // Captures convert to one rate; a new rate means new captures.
            sampleRate.update { $0 = settings.sampleRate }
            devices.sync {
                for (name, capture) in captures.current {
                    capture.stop()
                    captures.update { $0[name] = nil }
                    processing.sync { chunker.removeMic(name) }
                }
            }
        }
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
        let shouldRun = keepOpen.current
        for (name, capture) in captures.current {
            let unwanted = !wanted.contains(name)
            let dead = (shouldRun && !capture.isRunning) || !available.contains { $0.id == capture.device.id }
            if unwanted || dead {
                Log.line(unwanted ? "[Audio] Mic disabled in settings: \(name)" : "[Audio] Mic disconnected: \(name)")
                capture.stop()
                captures.update { $0[name] = nil }
                processing.sync { chunker.removeMic(name); drainMessages() }
            }
        }
        for name in wanted where captures.read({ $0[name] == nil }) {
            guard let device = CoreAudioDevices.find(name, in: available) else { continue }
            let capture = MicCapture(name: name, device: device, sampleRate: sampleRate.current) { [weak self] block, heard in
                guard let self else { return }
                self.processing.async { self.feed(name, block, heard) }
            }
            do {
                processing.sync { chunker.addMic(name) }
                if shouldRun { try capture.start() } else { try capture.prepare() }
                captures.update { $0[name] = capture }
                Log.line("[Audio] Mic \(shouldRun ? "connected" : "ready"): \(name) (\(device.name))")
            } catch {
                processing.sync { chunker.removeMic(name) }
                Log.line("[Audio] Failed to open \(name): \(error)")
            }
        }
    }

    // MARK: - Recording (any thread)

    /// Begin a recording. Chunks emitted on pauses go to `onChunk` on the audio queue.
    public func startRecording(onChunk: @escaping (AudioChunk) -> Void, onBlock: (([Float]) -> Void)? = nil) {
        recording.update { $0 = true }
        let now = HostClock.now
        let open = keepOpen.current
        processing.sync {
            self.onChunk = onChunk
            self.onBlock = onBlock
            chunker.startRecording(now: now, preroll: open)
            // The pre-roll is part of the dictation for whoever follows it block by block too.
            if let onBlock, let primary = chunker.primaryMic {
                for block in chunker.held(primary) { onBlock(block) }
            }
            openedAt = open ? nil : now
            heardSinceOpen = []
            drainMessages()
        }
        guard !open else { return }
        // Opened now, off the main thread; blocks arrive on the audio queue as they are heard.
        devices.async { [self] in
            for (name, capture) in captures.current {
                do { try capture.start() } catch { Log.line("[Audio] Couldn't open \(name): \(error)") }
            }
        }
    }

    /// End the recording: the rest of the audio, after every block heard before now.
    public func stopRecording() -> FinalChunk {
        // What the mics heard up to the release, part-blocks included, reaches the audio queue
        // first; without this the last few tens of milliseconds were cut.
        devices.sync { for capture in captures.current.values { capture.flush() } }
        let final = processing.sync { () -> FinalChunk in
            onChunk = nil
            onBlock = nil
            let final = chunker.stopRecording()
            drainMessages()
            return final
        }
        recording.update { $0 = false }
        if !keepOpen.current {
            devices.async { [self] in for capture in captures.current.values { capture.pause() } }
        }
        return final
    }

    public func shutdown() {
        monitor?.cancel()
        devices.sync {
            for capture in captures.current.values { capture.stop() }
            captures.update { $0.removeAll() }
        }
    }

    // MARK: - Audio queue

    private func feed(_ mic: String, _ block: [Float], _ heard: Double) {
        // Mics opened on the press have heard nothing before it: an earlier block is the end
        // of the last dictation, still on its way.
        if chunker.isRecording, let opened = openedAt, heard - Double(block.count) / Double(sampleRate.current) < opened - 0.01 {
            return
        }
        if let opened = openedAt, !heardSinceOpen.contains(mic) {
            heardSinceOpen.insert(mic)
            // The block's stamp is its end; its start is when the microphone began hearing.
            let listening = heard - Double(block.count) / Double(sampleRate.current) - opened
            Log.line("[Audio] \(mic) listening \(Int((listening * 1000).rounded()))ms after the press")
        }
        let chunk = chunker.feed(mic, block, now: heard)
        if chunker.isRecording, mic == chunker.primaryMic, let onBlock { onBlock(block) }
        currentLevel.update { $0 = chunker.level }
        drainMessages()
        if let chunk, let onChunk { onChunk(chunk) }
    }

    private func drainMessages() {
        for message in chunker.messages { Log.line(message) }
        chunker.messages.removeAll()
    }
}
