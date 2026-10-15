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
    var live: LiveWords?
    var stream: AssemblyAIStream?
    /// The latest dictations, newest first: for the menu, and for editing the last one.
    var recent: [(text: String, app: String, window: String, at: Date, inPlace: Bool)] = []
    var recentMenu: NSMenu!
    let ducker = Ducker(output: CoreAudioOutput())
    var statusItem: NSStatusItem!
    var statusLine: NSMenuItem!
    var hud: HUD!
    let textOutput = MacTextOutput()
    lazy var settingsWindow = SettingsWindowController(http: http)

    /// The recognizers the settings ask for. Built per recording, so a change in Settings
    /// applies to the next dictation and never swaps a running session's set under it.
    nonisolated static func recognizers(for settings: Settings, http: HTTPClient, vocabulary: VocabularyStore) -> [Recognizer] {
        var out: [Recognizer] = []
        for name in settings.enabledProviders where name != "parakeet" && name != AppleProvider.name {
            Log.line("  Unknown or unconfigured provider: \(name)")
        }
        if settings.enabledProviders.contains("parakeet") {
            Log.line("  Parakeet (local) isn't in the Swift build yet; using the cloud recognizers")
        }
        let learn = settings.learnVocabulary
        if settings.enabledProviders.contains(AppleProvider.name), #available(macOS 26.0, *) {
            out.append(AppleRecognizer(sampleRate: settings.sampleRate, vocabulary: { learn ? vocabulary.knownTerms() : [] }))
        }
        guard !settings.openrouterApiKey.isEmpty else { return out }
        let client = TranscriptionClient(http: http)
        for model in settings.openrouterSttModels {
            out.append(OpenRouterRecognizer(model: model, key: settings.openrouterApiKey, client: client,
                                            sampleRate: settings.sampleRate,
                                            keyterms: { learn ? vocabulary.knownTerms() : [] }))
        }
        return out
    }

    /// Opened again (Spotlight, Finder, `open`) while running: Settings, since the menu bar icon
    /// can be hidden when the menu bar is full.
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        openSettings()
        return false
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        Log.line("MergeScribe \(MergeScribe.version) starting...")
        DistributedNotificationCenter.default().addObserver(forName: SingleInstance.showSettings, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.openSettings() }
        }
        settings = Settings.load(paths: paths)
        Log.line("  Enabled mics: \(settings.enabledMics)")
        Log.line("  Transcription: \(settings.openrouterSttModels)")
        Notifications.requestPermission()

        audio = AudioEngine(settings: settings)
        // The HUD reads the live mic level straight off the audio engine.
        let engine = audio!
        hud = HUD(enabled: settings.hudEnabled, level: { engine.level })

        textOutput.viaAccessibility.update { $0 = settings.insertViaAccessibility }
        let training = settings.trainingEnabled ? TrainingWriter(directory: paths.trainingDir, sampleRate: settings.sampleRate) : nil
        let services = SessionServices(
            focus: MacFocus(), output: textOutput,
            corrector: CorrectionClient(http: http, vocabulary: vocabulary), judge: JudgeClient(http: http),
            editWatcher: MacEditWatcher(), metrics: metrics, training: training,
            corpus: CorrectionsCorpus(url: paths.correctionsFile), observer: hud)
        manager = SessionManager(services: services)
        manager.onSessionComplete = { [weak self] session, newest in
            let (text, method, _) = session.outcome
            let context = session.context
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    guard let self else { return }
                    if !text.isEmpty { self.remember(text, context: context, inPlace: method != "clipboard") }
                    if newest { self.setStatus(.idle) }
                }
            }
        }

        Log.line("  Active mics: \(audio.apply(settings))")
        audio.startMonitoring()

        hud.setLiveWords(settings.liveWords)
        if settings.liveWords, #available(macOS 26.0, *) { LiveTranscriber.warmUp(sampleRate: settings.sampleRate) }

        input = InputStateMachine(doubleTapThreshold: settings.doubleTapThreshold,
                                  toggleTimeout: Double(settings.toggleModeTimeout),
                                  scheduler: DispatchScheduler(queue: .main), clock: { HostClock.now })
        input.onStart = { [weak self] in MainActor.assumeIsolated { self?.startRecording() } }
        input.onStop = { [weak self] in MainActor.assumeIsolated { self?.stopRecording() } }
        startHotkey()

        buildMenu()
        setStatus(.idle)
        NotificationCenter.default.addObserver(forName: .settingsChanged, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.settingsDidChange() }
        }
        Log.line("Ready! Press \(settings.triggerKey) to record.")
    }

    func startHotkey() {
        hotkey?.stop()
        guard let key = Hotkey(trigger: settings.triggerKey, secondary: settings.settingsShortcut,
                               onSecondary: { [weak self] in self?.openSettings() }, machine: input) else {
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

    /// Settings saved a change: apply what can be applied now. The rest (models, prompts)
    /// is read afresh by the next recording anyway.
    func settingsDidChange() {
        let before = settings
        settings = Settings.load(paths: paths, log: { _ in })
        hud.setEnabled(settings.hudEnabled)
        hud.setLiveWords(settings.liveWords && settings.hudEnabled)
        textOutput.viaAccessibility.update { $0 = settings.insertViaAccessibility }
        if settings.triggerKey != before.triggerKey || settings.settingsShortcut != before.settingsShortcut { startHotkey() }
        if settings.enabledMics != before.enabledMics, current == nil { audio.apply(settings) }
        setStatus(current == nil ? .idle : .recording)
    }

    // MARK: - Recording

    func startRecording() {
        if manager.isBusy {
            _ = manager.startSession(settings: settings, recognizers: [])     // plays the busy sound
            return
        }
        settings = Settings.load(paths: paths)
        hud.setEnabled(settings.hudEnabled)
        hud.setLiveWords(settings.liveWords && settings.hudEnabled)
        input.doubleTapThreshold = settings.doubleTapThreshold
        input.toggleTimeout = Double(settings.toggleModeTimeout)
        let mics = audio.apply(settings)
        var recognizers = Self.recognizers(for: settings, http: http, vocabulary: vocabulary)
        var stream: AssemblyAIStream?
        if settings.streaming {
            let learn = settings.learnVocabulary, vocabulary = vocabulary
            let opened = AssemblyAIStream(key: settings.assemblyaiApiKey, model: settings.streamingModel,
                                          sampleRate: settings.sampleRate, keyterms: learn ? vocabulary.knownTerms() : [])
            // If the stream fails or stalls, batch Universal transcribes the whole recording.
            let fallback = settings.openrouterApiKey.isEmpty ? nil : OpenRouterRecognizer(
                model: "assemblyai/universal-3-5-pro", key: settings.openrouterApiKey, client: TranscriptionClient(http: http),
                sampleRate: settings.sampleRate, keyterms: { learn ? vocabulary.knownTerms() : [] })
            recognizers.insert(StreamingRecognizer(stream: opened, fallback: fallback), at: 0)
            stream = opened
        }
        guard !mics.isEmpty else { return showError("No active microphones") }
        guard !recognizers.isEmpty else {
            return showError(settings.openrouterApiKey.isEmpty
                ? "Add an OpenRouter key in Settings > Models to transcribe"
                : "Turn on a transcription model in Settings > Models (Parakeet isn't in this build yet)")
        }
        // Shift + trigger: select the last dictation first, and what is said becomes an edit
        // instruction for it (the selection sends the session into edit mode).
        if input.startedWithShift {
            let focus = MacFocus()
            guard let last = recent.first(where: { $0.inPlace }), Date().timeIntervalSince(last.at) < 600,
                  focus.appContext()?.appName == last.app, focus.selectBeforeCaret(last.text) else {
                NSSound(named: "Funk")?.play()
                Log.line("[Edit] No recent dictation at the cursor to edit")
                return
            }
            Log.line("[Edit] Selected the last dictation to edit it")
        }
        guard let session = manager.startSession(settings: settings, recognizers: recognizers) else { return }
        current = session
        // Connecting takes ~0.4 s; what the mics hear meanwhile is held and sent when it opens.
        stream?.start()
        self.stream = stream
        // Capture first, then let the session gather context: the press-to-record gap is
        // what the user feels.
        var words: LiveWords?
        if settings.liveWords && settings.hudEnabled {
            let started = LiveWords()
            let hud = hud!, sid = session.id.uuidString.lowercased()
            started.start(sampleRate: settings.sampleRate, vocabulary: settings.learnVocabulary ? vocabulary.knownTerms() : []) { text in
                hud.liveText(session: sid, text: text)
            }
            words = started
        }
        live = words
        let onBlock: (([Float]) -> Void)? = words == nil && stream == nil ? nil : { block in
            words?.append(block)
            stream?.append(block)
        }
        audio.startRecording(onChunk: { session.chunkReady($0) }, onBlock: onBlock)
        setStatus(.recording)
        if settings.duckWhileRecording { ducker.duck() }
        session.start(mics: mics)
        metrics.log("recording_started", ["session_id": .string(session.id.uuidString.lowercased())])
    }

    func stopRecording() {
        guard let session = current else { return }
        let final = audio.stopRecording()
        stream?.endOfAudio()      // after every block heard before the release
        stream = nil
        live?.stop()
        live = nil
        ducker.restore()
        setStatus(.processing)
        session.finalize(final)
        current = nil
        metrics.log("recording_stopped", ["session_id": .string(session.id.uuidString.lowercased())])
    }

    func remember(_ text: String, context: AppContext?, inPlace: Bool) {
        recent.insert((text.trimmingCharacters(in: .whitespacesAndNewlines), context?.appName ?? "", context?.windowTitle ?? "",
                       Date(), inPlace), at: 0)
        if recent.count > 10 { recent.removeLast(recent.count - 10) }
        rebuildRecentMenu()
    }

    /// Recent dictations: click one to copy it again.
    func rebuildRecentMenu() {
        recentMenu.removeAllItems()
        if recent.isEmpty {
            recentMenu.addItem(withTitle: "Nothing yet", action: nil, keyEquivalent: "")
            return
        }
        let relative = RelativeDateTimeFormatter()
        relative.unitsStyle = .short
        for (index, entry) in recent.enumerated() {
            let title = entry.text.count > 60 ? String(entry.text.prefix(57)) + "…" : entry.text
            let item = NSMenuItem(title: title, action: #selector(copyRecent(_:)), keyEquivalent: "")
            item.target = self
            item.tag = index
            item.toolTip = entry.text
            let detail = [entry.app, relative.localizedString(for: entry.at, relativeTo: Date())].filter { !$0.isEmpty }
            if #available(macOS 14.4, *) { item.subtitle = detail.joined(separator: " · ") }
            recentMenu.addItem(item)
        }
        recentMenu.addItem(.separator())
        recentMenu.addItem(withTitle: "Shift + \(Self.keyName(settings.triggerKey)) edits the last one", action: nil, keyEquivalent: "")
    }

    @objc func copyRecent(_ item: NSMenuItem) {
        guard item.tag < recent.count else { return }
        textOutput.copyToClipboard(recent[item.tag].text)
    }

    static func keyName(_ key: String) -> String {
        ["cmd_r": "Right Command", "alt_r": "Right Option", "alt_l": "Left Option", "ctrl_r": "Right Control"][key] ?? key.uppercased()
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
        let recentItem = NSMenuItem(title: "Recent Dictations", action: nil, keyEquivalent: "")
        recentMenu = NSMenu(title: "Recent Dictations")
        recentItem.submenu = recentMenu
        menu.addItem(recentItem)
        rebuildRecentMenu()
        menu.addItem(.separator())
        menu.addItem(withTitle: "Settings…", action: #selector(openSettings), keyEquivalent: ",").target = self
        menu.addItem(withTitle: "Open Log", action: #selector(openLog), keyEquivalent: "").target = self
        menu.addItem(.separator())
        menu.addItem(withTitle: "Quit MergeScribe", action: #selector(quit), keyEquivalent: "q").target = self
        statusItem.menu = menu
    }

    /// Show recording state in exactly one place: the HUD sits where you are looking, so while
    /// it is on the menu bar icon stays put; if it is off, the icon takes the job back.
    func setStatus(_ status: Status) {
        hud?.setStatus(status.rawValue)
        let status = hud?.isShowing == true && status != .error ? .idle : status
        let symbols: [Status: String] = [.idle: "waveform", .recording: "waveform.circle.fill",
                                         .processing: "ellipsis.circle", .error: "exclamationmark.triangle"]
        let image = NSImage(systemSymbolName: symbols[status]!, accessibilityDescription: "MergeScribe")?
            .withSymbolConfiguration(.init(pointSize: 14, weight: .regular))
        image?.isTemplate = true
        statusItem?.button?.image = image
        statusLine?.title = "Status: \(status.rawValue.capitalized)"
    }

    @objc func openSettings() { settingsWindow.present() }

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
