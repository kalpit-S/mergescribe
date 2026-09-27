import Foundation

/// Something that runs an action later and can be called off.
public protocol Scheduling: Sendable {
    func schedule(after seconds: Double, _ action: @escaping @Sendable () -> Void) -> Cancellable
}

public protocol Cancellable: Sendable {
    func cancel()
}

/// The trigger key's presses and releases, as recording intents.
///
/// - Hold to record: press, talk, release.
/// - Double-tap to toggle: a second tap within the window keeps the same recording going
///   hands-free; the next press stops it.
/// - Shift+Esc always stops everything.
///
/// A short tap's stop waits out the double-tap window, so that a first tap doesn't start
/// processing (and block the second) before the second tap can arrive.
public final class InputStateMachine: @unchecked Sendable {
    public enum State: String, Sendable { case idle, recording, toggleRecording }

    public var onStart: (@Sendable () -> Void)?
    public var onStop: (@Sendable () -> Void)?
    public var onEmergencyReset: (@Sendable () -> Void)?

    public var doubleTapThreshold: Double
    public var toggleTimeout: Double

    private let scheduler: Scheduling
    private let clock: @Sendable () -> Double
    private let lock = NSRecursiveLock()
    public private(set) var state: State = .idle
    /// Shift was held as the recording started: edit the last dictation rather than add one.
    public private(set) var startedWithShift = false
    private var lastPress = 0.0
    private var pressStartedAt: Double?
    private var triggerDown = false
    private var shiftDown = false
    private var awaitingSecondTap = false
    private var pendingStop: Cancellable?
    private var toggleTimer: Cancellable?

    public init(doubleTapThreshold: Double = 0.45, toggleTimeout: Double = 600,
                scheduler: Scheduling, clock: @escaping @Sendable () -> Double) {
        self.doubleTapThreshold = doubleTapThreshold
        self.toggleTimeout = toggleTimeout
        self.scheduler = scheduler
        self.clock = clock
    }

    // MARK: - Key events

    public func shift(down: Bool) { lock.withLock { shiftDown = down } }

    public func escapePressed() {
        let reset: Bool = lock.withLock { shiftDown }
        if reset { emergencyReset() }
    }

    public func triggerPressed() {
        lock.lock(); defer { lock.unlock() }
        guard !triggerDown else { return }       // key repeat
        triggerDown = true
        let now = clock()
        pressStartedAt = now
        switch state {
        case .toggleRecording:
            stop()                                // in toggle mode, any press stops
        case .recording where awaitingSecondTap && now - lastPress < doubleTapThreshold:
            enterToggleMode()                     // the second tap keeps the same recording going
        case .idle:
            start()
        default:
            break
        }
        lastPress = now
    }

    public func triggerReleased() {
        lock.lock(); defer { lock.unlock() }
        triggerDown = false
        let now = clock()
        if state == .recording {
            let held = now - (pressStartedAt ?? now)
            if held >= doubleTapThreshold {
                stop()
            } else {
                armPendingStop(after: doubleTapThreshold - held)
            }
        }
        pressStartedAt = nil
    }

    // MARK: - Transitions (lock held)

    private func start() {
        startedWithShift = shiftDown
        awaitingSecondTap = false
        cancel(&pendingStop)
        state = .recording
        onStart?()
    }

    private func stop() {
        state = .idle
        awaitingSecondTap = false
        cancel(&pendingStop)
        cancel(&toggleTimer)
        onStop?()
    }

    private func enterToggleMode() {
        state = .toggleRecording
        awaitingSecondTap = false
        cancel(&pendingStop)
        cancel(&toggleTimer)
        // A safety net for a recording nobody stops.
        toggleTimer = scheduler.schedule(after: toggleTimeout) { [weak self] in
            guard let self else { return }
            self.lock.withLock {
                if self.state == .toggleRecording {
                    Log.line("Toggle mode timeout - stopping recording")
                    self.stop()
                }
            }
        }
    }

    private func armPendingStop(after seconds: Double) {
        awaitingSecondTap = true
        cancel(&pendingStop)
        pendingStop = scheduler.schedule(after: seconds) { [weak self] in
            guard let self else { return }
            self.lock.withLock {
                self.pendingStop = nil
                if self.state == .recording && self.awaitingSecondTap { self.stop() }
            }
        }
    }

    private func cancel(_ timer: inout Cancellable?) {
        timer?.cancel()
        timer = nil
    }

    private func emergencyReset() {
        lock.withLock {
            state = .idle
            triggerDown = false
            cancel(&toggleTimer)
            cancel(&pendingStop)
            awaitingSecondTap = false
            pressStartedAt = nil
        }
        if let onEmergencyReset { onEmergencyReset() } else { onStop?() }
        Log.line("Emergency reset triggered")
    }
}

/// Timers on a dispatch queue.
public struct DispatchScheduler: Scheduling {
    let queue: DispatchQueue
    public init(queue: DispatchQueue = .main) { self.queue = queue }

    public func schedule(after seconds: Double, _ action: @escaping @Sendable () -> Void) -> Cancellable {
        let item = DispatchWorkItem(block: action)
        queue.asyncAfter(deadline: .now() + seconds, execute: item)
        return DispatchCancellable(item: item)
    }
}

struct DispatchCancellable: Cancellable, @unchecked Sendable {
    let item: DispatchWorkItem
    func cancel() { item.cancel() }
}
