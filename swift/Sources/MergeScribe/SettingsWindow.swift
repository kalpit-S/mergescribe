import AppKit
import MergeScribeCore
import MergeScribeMac
import SwiftUI

/// SwiftUI has a Settings scene; this is ours.
typealias AppSettings = MergeScribeCore.Settings

extension Notification.Name {
    /// Posted after Settings writes a change, so the running app can pick it up.
    static let settingsChanged = Notification.Name("MergeScribeSettingsChanged")
}

/// What the Settings window edits and shows. Every control saves as it changes, so there is
/// no Save button to forget; the numbers come from the metrics log and load off the main thread.
@MainActor
final class SettingsModel: ObservableObject {
    let paths: AppPaths
    let file: SettingsFile
    let http: HTTPClient
    @Published var settings = AppSettings()
    @Published var apiKey = ""
    @Published var assemblyKey = ""
    @Published var keyStatus: (text: String, ok: Bool?) = ("", nil)
    @Published var listed: [(id: String, name: String)] = []
    @Published var catalogue: [(id: String, name: String)]?
    @Published var usage: Stats.Usage?
    @Published var report: Insights.Report?
    @Published var mics: [String] = []
    @Published var learned: [Vocabulary.Correction] = []
    @Published var supplied: [String] = []
    @Published var appleAvailable = false

    init(paths: AppPaths = AppPaths(), http: HTTPClient) {
        self.paths = paths
        self.http = http
        file = SettingsFile(url: paths.settingsFile)
        reload()
    }

    /// Everything from disk, so a reopened window never shows stale values.
    func reload() {
        settings = AppSettings.load(paths: paths, log: { _ in })
        let env = EnvFile.read(paths.envFile)
        apiKey = env["OPENROUTER_API_KEY"] ?? ""
        assemblyKey = env["ASSEMBLYAI_API_KEY"] ?? ""
        let raw = file.load()
        listed = (raw["openrouter_stt_listed"] as? [[Any]] ?? []).compactMap { pair in
            guard pair.count == 2, let id = pair[0] as? String, let name = pair[1] as? String else { return nil }
            return (id, name)
        }
        let connected = CoreAudioDevices.inputDevices().map(\.name)
        mics = connected + settings.enabledMics.filter { !connected.contains($0) }
        let store = VocabularyStore(paths: paths)
        learned = store.learnedCorrections()
        supplied = store.suppliedTerms()
        validateKey()
        if #available(macOS 26.0, *) {
            Task { appleAvailable = await AppleRecognizer.available() }
        }
    }

    func save(_ values: [String: Any]) {
        do {
            try file.save(values)
            NotificationCenter.default.post(name: .settingsChanged, object: nil)
        } catch {
            Log.line("[Settings] Could not save: \(error)")
        }
    }

    func remove(_ keys: [String]) {
        try? file.remove(keys)
        NotificationCenter.default.post(name: .settingsChanged, object: nil)
    }

    /// A control bound to one setting, saved as it changes.
    func bind<T: Equatable>(_ key: String, _ path: WritableKeyPath<AppSettings, T>, _ encode: @escaping (T) -> Any = { $0 }) -> Binding<T> {
        Binding(get: { self.settings[keyPath: path] }, set: { value in
            guard self.settings[keyPath: path] != value else { return }
            self.settings[keyPath: path] = value
            self.save([key: encode(value)])
        })
    }

    // MARK: - The key

    func commitKey(_ key: String) {
        let key = key.trimmingCharacters(in: .whitespaces)
        guard key != apiKey else { return }
        try? EnvFile.write(["OPENROUTER_API_KEY": key], to: paths.envFile)
        apiKey = key
        settings.openrouterApiKey = key
        NotificationCenter.default.post(name: .settingsChanged, object: nil)
        validateKey()
    }

    func commitAssemblyKey(_ key: String) {
        let key = key.trimmingCharacters(in: .whitespaces)
        guard key != assemblyKey else { return }
        try? EnvFile.write(["ASSEMBLYAI_API_KEY": key], to: paths.envFile)
        assemblyKey = key
        settings.assemblyaiApiKey = key
        NotificationCenter.default.post(name: .settingsChanged, object: nil)
    }

    func validateKey() {
        guard !apiKey.isEmpty else { keyStatus = ("Not set", nil); return }
        keyStatus = ("Checking…", nil)
        let key = apiKey, http = http
        Task {
            let check = await Catalogue.validateKey(http: http, key: key)
            keyStatus = check.valid ? ("Connected · \(check.latencyMs ?? 0) ms", true) : (check.error ?? "Invalid key", false)
        }
    }

    // MARK: - Recognizers

    /// The current models, then any added from the catalogue (kept listed when switched off),
    /// then anything enabled by hand in the file.
    var shownModels: [(id: String, name: String)] {
        var shown = Catalogue.transcriptionModels
        for model in listed + settings.openrouterSttModels.map({ ($0, $0) }) where !shown.contains(where: { $0.id == model.0 }) {
            shown.append(model)
        }
        return shown
    }

    func modelEnabled(_ id: String) -> Binding<Bool> {
        Binding(get: { self.settings.openrouterSttModels.contains(id) }, set: { on in
            var models = self.settings.openrouterSttModels.filter { $0 != id }
            if on { models.append(id) }
            self.settings.openrouterSttModels = models
            self.save(["openrouter_stt_models": models])
        })
    }

    func providerEnabled(_ name: String) -> Binding<Bool> {
        Binding(get: { self.settings.enabledProviders.contains(name) }, set: { on in
            var providers = self.settings.enabledProviders.filter { $0 != name }
            if on { providers.append(name) }
            self.settings.enabledProviders = providers
            self.save(["enabled_providers": providers])
        })
    }

    func loadCatalogue() {
        guard catalogue == nil else { return }
        let http = http, key = apiKey
        Task { catalogue = await Catalogue.fetchTranscriptionModels(http: http, key: key) }
    }

    func addModel(_ id: String) {
        guard !id.isEmpty else { return }
        let name = catalogue?.first { $0.id == id }?.name ?? id
        listed.append((id, name))
        settings.openrouterSttModels.append(id)
        save(["openrouter_stt_listed": listed.map { [$0.id, $0.name] }, "openrouter_stt_models": settings.openrouterSttModels])
    }

    /// What the log says about a recognizer, by the name its results carry.
    func insight(for provider: String) -> Insights.Recognizer? {
        report?.recognizers.first { $0.name == provider }
    }

    /// The recognizers the next dictation will use, by display name.
    var activeRecognizers: [String] {
        var names = settings.openrouterSttModels.map { id in shownModels.first { $0.id == id }?.name ?? id }
        if settings.enabledProviders.contains(AppleProvider.name) { names.append("Apple") }
        if settings.streamingEnabled && !assemblyKey.isEmpty { names.insert("Universal 3.6, streaming", at: 0) }
        return names
    }

    // MARK: - Mics

    func micEnabled(_ mic: String) -> Binding<Bool> {
        Binding(get: { self.settings.enabledMics.contains(mic) }, set: { on in
            // In the order the mics are listed, so the primary is predictable.
            let enabled = self.mics.filter { $0 == mic ? on : self.settings.enabledMics.contains($0) }
            self.settings.enabledMics = enabled
            self.save(["enabled_mics": enabled])
        })
    }

    // MARK: - Vocabulary

    /// vocabulary.txt, one term per line, most important first. Comment lines in the file are
    /// kept, at the top.
    func setSupplied(_ terms: [String]) {
        var seen = Set<String>()
        let terms = terms.map { $0.collapsingWhitespace() }.filter { !$0.isEmpty && seen.insert($0.lowercased()).inserted }
        let existing = (try? String(contentsOf: paths.vocabularyFile, encoding: .utf8)) ?? ""
        var comments = existing.split(whereSeparator: \.isNewline).filter { $0.trimmingCharacters(in: .whitespaces).hasPrefix("#") }
            .map(String.init)
        if comments.isEmpty {
            comments = ["# Words and names you use, most important first. MergeScribe primes the recognizers",
                        "# and the correction model with them."]
        }
        do {
            try FileManager.default.createDirectory(at: paths.dataDir, withIntermediateDirectories: true)
            try Data(((comments + terms).joined(separator: "\n") + "\n").utf8).write(to: paths.vocabularyFile, options: .atomic)
            supplied = terms
        } catch {
            Log.line("[Settings] Could not save the vocabulary: \(error)")
        }
    }

    /// Stop using a learned correction (the corpus of corrections is kept).
    func forget(_ correction: Vocabulary.Correction) {
        let store = VocabularyStore(paths: paths)
        do {
            try store.forget(correction.after)
            learned = store.learnedCorrections()
        } catch {
            Log.line("[Settings] Could not forget a correction: \(error)")
        }
    }

    // MARK: - Numbers

    private var numbersFrom: Date?

    /// The numbers, again only if the log has changed since they were read.
    func refreshNumbers() {
        let modified = (try? FileManager.default.attributesOfItem(atPath: paths.metricsFile.path))?[.modificationDate] as? Date
        guard modified != numbersFrom else { return }
        numbersFrom = modified
        loadNumbers()
    }

    func loadNumbers() {
        let paths = paths
        Task.detached {
            let usage = Stats.usage(metrics: paths.metricsFile, training: paths.trainingDir)
            let report = Insights.report(metrics: paths.metricsFile)
            await MainActor.run {
                self.usage = usage
                self.report = report
            }
        }
    }
}

// MARK: - The window

enum SettingsSection: String, CaseIterable, Identifiable {
    case overview = "Overview", recording = "Recording", transcription = "Transcription", correction = "Correction"
    case vocabulary = "Vocabulary", history = "History", stats = "Stats", advanced = "Advanced"
    var id: String { rawValue }
    var symbol: String {
        switch self {
        case .overview: "sparkles"
        case .recording: "waveform"
        case .transcription: "text.bubble"
        case .correction: "wand.and.stars"
        case .vocabulary: "character.book.closed"
        case .history: "clock.arrow.circlepath"
        case .stats: "chart.bar.xaxis"
        case .advanced: "slider.horizontal.3"
        }
    }
}

/// Which section is showing; outside the view so it can be driven for captures.
@MainActor
final class SettingsNavigation: ObservableObject {
    @Published var section: SettingsSection? = .overview
}

struct SettingsView: View {
    @ObservedObject var model: SettingsModel
    @ObservedObject var navigation: SettingsNavigation

    var body: some View {
        NavigationSplitView {
            List(selection: $navigation.section) {
                row(.overview)
                Section("Dictation") {
                    ForEach([SettingsSection.recording, .transcription, .correction, .vocabulary]) { row($0) }
                }
                Section("Your Dictations") {
                    ForEach([SettingsSection.history, .stats]) { row($0) }
                }
                Section { row(.advanced) }
            }
            .navigationSplitViewColumnWidth(min: 190, ideal: 200)
        } detail: {
            Group {
                switch navigation.section ?? .overview {
                case .overview: OverviewPane(model: model, navigation: navigation)
                case .recording: RecordingPane(model: model)
                case .transcription: TranscriptionPane(model: model)
                case .correction: CorrectionPane(model: model)
                case .vocabulary: VocabularyPane(model: model)
                case .history: HistoryPane(model: model)
                case .stats: StatsPane(model: model)
                case .advanced: AdvancedPane(model: model)
                }
            }
            .frame(minWidth: 540)
        }
        .onAppear { model.refreshNumbers() }
        .onChange(of: navigation.section) { _, _ in model.refreshNumbers() }
    }

    func row(_ section: SettingsSection) -> some View {
        Label(section.rawValue, systemImage: section.symbol).tag(section)
    }
}

/// Owns the settings window. A menu-bar app's windows can't become key, so text couldn't be
/// typed into them: the app becomes a regular one while Settings is open, with an Edit menu
/// so ⌘C, ⌘V and ⌘A work.
@MainActor
final class SettingsWindowController: NSObject, NSWindowDelegate {
    let model: SettingsModel
    let navigation = SettingsNavigation()
    private var window: NSWindow?

    init(http: HTTPClient) { model = SettingsModel(http: http) }

    func present(section: SettingsSection? = nil, activate: Bool = true) {
        if let section { navigation.section = section }
        if window == nil {
            let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 880, height: 680),
                                  styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
                                  backing: .buffered, defer: false)
            window.title = "MergeScribe Settings"
            window.titleVisibility = .hidden
            window.titlebarAppearsTransparent = true
            window.contentMinSize = NSSize(width: 780, height: 540)
            window.isReleasedWhenClosed = false
            window.delegate = self
            window.contentView = NSHostingView(rootView: SettingsView(model: model, navigation: navigation))
            window.center()
            self.window = window
        } else if let window, !window.isVisible {
            model.reload()
            window.contentView = NSHostingView(rootView: SettingsView(model: model, navigation: navigation))
        }
        guard activate else { window?.orderFrontRegardless(); return }   // a preview that leaves focus alone
        NSApp.setActivationPolicy(.regular)
        ensureEditMenu()
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate()
    }

    func windowWillClose(_ notification: Notification) {
        // Ending editing fires the fields' commits, so a half-typed field still saves.
        window?.makeFirstResponder(nil)
        NSApp.setActivationPolicy(.accessory)
        // Then the views go, once those commits have run, so nothing (the HUD preview's
        // timer, an audio player) keeps going behind a closed window.
        DispatchQueue.main.async { [weak self] in
            MainActor.assumeIsolated {
                guard let window = self?.window, !window.isVisible else { return }
                window.contentView = NSView()
            }
        }
    }

    func ensureEditMenu() {
        let main = NSApp.mainMenu ?? NSMenu(title: "Main")
        if NSApp.mainMenu == nil { NSApp.mainMenu = main }
        guard main.item(withTitle: "Edit") == nil else { return }
        if main.numberOfItems == 0 {
            let appItem = NSMenuItem(), appMenu = NSMenu(title: "MergeScribe")
            appMenu.addItem(withTitle: "Close Window", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
            appItem.submenu = appMenu
            main.addItem(appItem)
        }
        let editItem = NSMenuItem(), edit = NSMenu(title: "Edit")
        edit.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        edit.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
        edit.addItem(.separator())
        edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = edit
        main.addItem(editItem)
    }
}
