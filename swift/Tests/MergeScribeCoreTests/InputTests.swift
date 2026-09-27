import Foundation
import Testing
@testable import MergeScribeCore

/// A clock and timers the test moves by hand.
final class ManualTime: Scheduling, @unchecked Sendable {
    private let lock = NSLock()
    private(set) var now = 0.0
    private var timers: [(at: Double, id: Int, action: @Sendable () -> Void)] = []
    private var cancelled = Set<Int>()
    private var nextId = 0

    func schedule(after seconds: Double, _ action: @escaping @Sendable () -> Void) -> Cancellable {
        lock.withLock {
            nextId += 1
            timers.append((now + seconds, nextId, action))
            return Token(id: nextId, time: self)
        }
    }

    struct Token: Cancellable, @unchecked Sendable {
        let id: Int
        let time: ManualTime
        func cancel() { time.lock.withLock { _ = time.cancelled.insert(id) } }
    }

    /// Move the clock on, firing any timers that fall due.
    func advance(_ seconds: Double) {
        let target = lock.withLock { now + seconds }
        while true {
            let due: (at: Double, id: Int, action: @Sendable () -> Void)? = lock.withLock {
                guard let next = timers.filter({ $0.at <= target && !cancelled.contains($0.id) }).min(by: { $0.at < $1.at }) else {
                    return nil
                }
                timers.removeAll { $0.id == next.id }
                now = next.at
                return next
            }
            guard let due else { break }
            due.action()
        }
        lock.withLock { now = target }
    }
}

private final class Events: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [String] = []
    func add(_ e: String) { lock.withLock { items.append(e) } }
    var all: [String] { lock.withLock { items } }
}

private func machine(threshold: Double, toggleTimeout: Double = 5) -> (InputStateMachine, ManualTime, Events) {
    let time = ManualTime(), events = Events()
    let m = InputStateMachine(doubleTapThreshold: threshold, toggleTimeout: toggleTimeout, scheduler: time, clock: { time.now })
    m.onStart = { events.add("start") }
    m.onStop = { events.add("stop") }
    return (m, time, events)
}

@Suite struct InputTests {
    @Test func aShortTapStopsAfterTheDoubleTapWindow() {
        let (m, time, events) = machine(threshold: 0.08)
        m.triggerPressed(); time.advance(0.01); m.triggerReleased()
        #expect(events.all == ["start"] && m.state == .recording)
        time.advance(0.10)
        #expect(events.all == ["start", "stop"] && m.state == .idle)
    }

    @Test func aHoldStopsOnRelease() {
        let (m, time, events) = machine(threshold: 0.05)
        m.triggerPressed(); time.advance(0.08); m.triggerReleased()
        #expect(events.all == ["start", "stop"] && m.state == .idle)
    }

    @Test func aDoubleTapTogglesAndTheThirdPressStops() {
        let (m, time, events) = machine(threshold: 0.5)
        m.triggerPressed(); time.advance(0.01); m.triggerReleased()
        time.advance(0.02)
        m.triggerPressed(); m.triggerReleased()
        #expect(events.all == ["start"], "no stop between the taps: the same recording continues")
        #expect(m.state == .toggleRecording)
        m.triggerPressed()
        #expect(events.all == ["start", "stop"] && m.state == .idle)
        time.advance(0.6)
        #expect(events.all == ["start", "stop"], "no delayed stop fires later")
    }

    @Test func toggleModeTimesOut() {
        let (m, time, events) = machine(threshold: 0.5, toggleTimeout: 10)
        m.triggerPressed(); m.triggerReleased(); time.advance(0.1)
        m.triggerPressed(); m.triggerReleased()
        time.advance(10.5)
        #expect(events.all == ["start", "stop"] && m.state == .idle)
    }

    @Test func keyRepeatIsIgnored() {
        let (m, _, events) = machine(threshold: 0.45)
        m.triggerPressed(); m.triggerPressed(); m.triggerPressed()
        #expect(events.all == ["start"])
    }

    @Test func shiftHeldAtThePressIsRemembered() {
        // Shift + trigger edits the last dictation instead of adding one.
        let (m, time, _) = machine(threshold: 0.05)
        m.shift(down: true); m.triggerPressed(); m.shift(down: false)
        #expect(m.startedWithShift)
        time.advance(0.1); m.triggerReleased()
        m.triggerPressed()
        #expect(!m.startedWithShift)
    }

    @Test func shiftEscapeStopsEverything() {
        let (m, time, events) = machine(threshold: 0.5)
        m.triggerPressed(); m.triggerReleased(); time.advance(0.1)
        m.triggerPressed(); m.triggerReleased()
        m.escapePressed()
        #expect(m.state == .toggleRecording, "escape alone does nothing")
        m.shift(down: true); m.escapePressed(); m.shift(down: false)
        #expect(m.state == .idle && events.all == ["start", "stop"])
        time.advance(20)
        #expect(events.all == ["start", "stop"])
    }
}
