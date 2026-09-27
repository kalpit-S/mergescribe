import AppKit
import Foundation
import MergeScribeCore
import MergeScribeMac

/// The running app: settings, audio, the hotkey, sessions, and the menu bar.
@MainActor
final class App: NSObject, NSApplicationDelegate {
    enum Status: String { case idle, recording, processing, error }

    let paths = AppPaths()
    var settings = Settings()
    let http = URLSessionHTTPClient()
    let vocabulary = VocabularyStore()
    lazy var metrics = MetricsWriter(url: paths.metricsFile)
    var audio: AudioEngine!
    var manager: SessionManager!
    var input: InputStateMachine!
    var hotkey: Hotkey?
    var current: Session?
    let ducker = Ducker(output: CoreAudioOutput())
    var statusItem: NSStatusItem!
    var statusLine: NSMenuItem!

    /// The recognizers the settings ask for. Built per recording, so a change in Settings
    /// applies to the next dictation and never swaps a running session's set under it.
    nonisolated static func recognizers(for settings: Settings, http: HTTPClient, vocabulary: VocabularyStore) -> [Recognizer] {
        var out: [Recognizer] = []
        for name in settings.enabledProviders where name != "parakeet" {
            Log.line("  Unknown or unconfigured provider: \(name)")
        }
        if settings.enabledProviders.contains("parakeet") {
            Log.line("  Parakeet (local) isn't in the Swift build yet; using the cloud recognizers")
        }
        guard !settings.openrouterApiKey.isEmpty else { return out }
        let client = TranscriptionClient(http: http)
        let learn = settings.learnVocabulary
        for model in settings.openrouterSttModels {
            out.append(OpenRouterRecognizer(model: model, key: settings.openrouterApiKey, client: client,
                                            sampleRate: settings.sampleRate,
                                            keyterms: { learn ? vocabulary.knownTerms() : [] }))
        }
        return out
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        Log.line("MergeScribe \(MergeScribe.version) starting...")
        settings = Settings.load(paths: paths)
        Log.line("  Enabled mics: \(settings.enabledMics)")
        Log.line("  Transcription: \(settings.openrouterSttModels)")
        Notifications.requestPermission()

        let training = settings.trainingEnabled ? TrainingWriter(directory: paths.trainingDir, sampleRate: settings.sampleRate) : nil
        let services = SessionServices(
            focus: MacFocus(), output: MacTextOutput(),
            corrector: CorrectionClient(http: http, vocabulary: vocabulary), judge: JudgeClient(http: http),
            editWatcher: MacEditWatcher(), metrics: metrics, training: training,
            corpus: CorrectionsCorpus(url: paths.correctionsFile), observer: nil)
        manager = SessionManager(services: services)
        manager.onSessionComplete = { [weak self] _, newest in
            guard newest else { return }
            DispatchQueue.main.async { self?.setStatus(.idle) }
        }

        audio = AudioEngine(settings: settings)
        Log.line("  Active mics: \(audio.apply(settings))")
        audio.startMonitoring()

        input = InputStateMachine(doubleTapThreshold: settings.doubleTapThreshold,
                                  toggleTimeout: Double(settings.toggleModeTimeout),
                                  scheduler: DispatchScheduler(queue: .main), clock: { HostClock.now })
        input.onStart = { [weak self] in MainActor.assumeIsolated { self?.startRecording() } }
        input.onStop = { [weak self] in MainActor.assumeIsolated { self?.stopRecording() } }
        startHotkey()

        buildMenu()
        setStatus(.idle)
        Log.line("Ready! Press \(settings.triggerKey) to record.")
    }

    func startHotkey() {
        hotkey?.stop()
        guard let key = Hotkey(trigger: settings.triggerKey, machine: input) else {
            Log.line("Unknown trigger key \(settings.triggerKey)")
            return
        }
        if key.start() {
            hotkey = key
        } else {
            Log.line("Can't listen for the trigger key: allow MergeScribe in System Settings > Privacy & Security > Input Monitoring")
            Notifications.post("Allow MergeScribe under Input Monitoring to use the trigger key")
            setStatus(.error)
        }
    }

    // MARK: - Recording

    func startRecording() {
        if manager.isBusy {
            _ = manager.startSession(settings: settings, recognizers: [])     // plays the busy sound
            return
        }
        settings = Settings.load(paths: paths)
        input.doubleTapThreshold = settings.doubleTapThreshold
        input.toggleTimeout = Double(settings.toggleModeTimeout)
        let mics = audio.apply(settings)
        let recognizers = Self.recognizers(for: settings, http: http, vocabulary: vocabulary)
        guard !mics.isEmpty else { return showError("No active microphones") }
        guard !recognizers.isEmpty else { return showError("No transcription providers") }
        guard let session = manager.startSession(settings: settings, recognizers: recognizers) else { return }
        current = session
        // Capture first, then let the session gather context: the press-to-record gap is
        // what the user feels.
        audio.startRecording(onChunk: { session.chunkReady($0) })
        setStatus(.recording)
        if settings.duckWhileRecording { ducker.duck() }
        session.start(mics: mics)
        metrics.log("recording_started", ["session_id": .string(session.id.uuidString.lowercased())])
    }

    func stopRecording() {
        guard let session = current else { return }
        let final = audio.stopRecording()
        ducker.restore()
        setStatus(.processing)
        session.finalize(final)
        current = nil
        metrics.log("recording_stopped", ["session_id": .string(session.id.uuidString.lowercased())])
    }

    func showError(_ message: String) {
        setStatus(.error)
        Notifications.post(message, title: "MergeScribe: Error")
        Log.line("[Error] \(message)")
    }

    // MARK: - Menu bar

    func buildMenu() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        let menu = NSMenu()
        statusLine = NSMenuItem(title: "Status: Idle", action: nil, keyEquivalent: "")
        menu.addItem(statusLine)
        menu.addItem(.separator())
        menu.addItem(withTitle: "Open Log", action: #selector(openLog), keyEquivalent: "").target = self
        menu.addItem(.separator())
        menu.addItem(withTitle: "Quit MergeScribe", action: #selector(quit), keyEquivalent: "q").target = self
        statusItem.menu = menu
    }

    func setStatus(_ status: Status) {
        let symbols: [Status: String] = [.idle: "waveform", .recording: "waveform.circle.fill",
                                         .processing: "ellipsis.circle", .error: "exclamationmark.triangle"]
        let image = NSImage(systemSymbolName: symbols[status]!, accessibilityDescription: "MergeScribe")?
            .withSymbolConfiguration(.init(pointSize: 14, weight: .regular))
        image?.isTemplate = true
        statusItem?.button?.image = image
        statusLine?.title = "Status: \(status.rawValue.capitalized)"
    }

    @objc func openLog() { NSWorkspace.shared.open(paths.logFile) }

    @objc func quit() { NSApp.terminate(nil) }

    func applicationWillTerminate(_ notification: Notification) {
        Log.line("\nShutting down...")
        hotkey?.stop()
        ducker.restore()          // quitting mid-dictation must not leave the volume down
        audio?.shutdown()
        metrics.flush()
        Log.line("Goodbye!")
    }
}
