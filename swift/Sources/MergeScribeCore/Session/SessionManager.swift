import Foundation

/// Starts sessions, one recording at a time.
///
/// A session holds the microphone only while it captures. Finalization (correction and
/// typing) runs long after the mic is free, and output is kept in order by OutputOrder, so
/// a new recording may begin while the previous one finishes. Gating on the whole session
/// made the app deaf for its ~2s tail.
public final class SessionManager: @unchecked Sendable {
    public let services: SessionServices
    public let history = TranscriptionHistory()
    let outputOrder = OutputOrder()
    let outputLock = NSLock()
    private let lock = NSLock()
    private var active: Session?
    /// Called when a session finishes, with whether it was the newest one: only the newest
    /// may put the status back to idle.
    public var onSessionComplete: (@Sendable (Session, _ newest: Bool) -> Void)?

    public init(services: SessionServices) { self.services = services }

    public var activeSession: Session? { lock.withLock { active } }

    /// True while a session still holds the microphone.
    public var isBusy: Bool { lock.withLock { active?.isCapturing ?? false } }

    /// A new session, or nil (with the busy sound) while one is still recording.
    public func startSession(settings: Settings, recognizers: [Recognizer]) -> Session? {
        let session: Session? = lock.withLock {
            if active?.isCapturing == true { return nil }
            let session = Session(settings: settings, recognizers: recognizers, services: services, history: history,
                                  outputOrder: outputOrder, outputLock: outputLock)
            active = session
            return session
        }
        guard let session else {
            services.output.playBusySound()
            return nil
        }
        session.onComplete = { [weak self] finished in self?.completed(finished) }
        return session
    }

    func completed(_ session: Session) {
        let newest: Bool = lock.withLock {
            let newest = active == nil || active === session
            if active === session { active = nil }
            return newest
        }
        onSessionComplete?(session, newest)
    }
}
