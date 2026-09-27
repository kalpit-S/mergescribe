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

/// What the Settings window edits: settings.json and the key in .env. Every control saves as
/// it changes, so there is no Save button to forget.
@MainActor
final class SettingsModel: ObservableObject {
    let paths: AppPaths
    let file: SettingsFile
    let http: HTTPClient
    @Published var settings = AppSettings()
    @Published var apiKey = ""
    @Published var keyStatus: (text: String, ok: Bool?) = ("", nil)
    @Published var listed: [(id: String, name: String)] = []
    @Published var catalogue: [(id: String, name: String)]?
    @Published var usage: Stats.Usage?
    @Published var mics: [String] = []
    @Published var learned: [String] = []

    init(paths: AppPaths = AppPaths(), http: HTTPClient) {
        self.paths = paths
        self.http = http
        file = SettingsFile(url: paths.settingsFile)
        reload()
    }

    /// Everything from disk, so a reopened window never shows stale values.
    func reload() {
        settings = AppSettings.load(paths: paths, log: { _ in })
        apiKey = EnvFile.read(paths.envFile)["OPENROUTER_API_KEY"] ?? ""
        let raw = file.load()
        listed = (raw["openrouter_stt_listed"] as? [[Any]] ?? []).compactMap { pair in
            guard pair.count == 2, let id = pair[0] as? String, let name = pair[1] as? String else { return nil }
            return (id, name)
        }
        let connected = CoreAudioDevices.inputDevices().map(\.name)
        mics = connected + settings.enabledMics.filter { !connected.contains($0) }
        learned = VocabularyStore(paths: paths).learnedCorrections().map(\.after)
        validateKey()
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

    func validateKey() {
        guard !apiKey.isEmpty else { keyStatus = ("Not set", nil); return }
        keyStatus = ("Checking…", nil)
        let key = apiKey, http = http
        Task {
            let check = await Catalogue.validateKey(http: http, key: key)
            keyStatus = check.valid ? ("Connected · \(check.latencyMs ?? 0) ms", true) : (check.error ?? "Invalid key", false)
        }
    }

    // MARK: - Transcription models

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

    // MARK: - Mics

    func micEnabled(_ mic: String) -> Binding<Bool> {
        Binding(get: { self.settings.enabledMics.contains(mic) }, set: { on in
            // In the order the mics are listed, so the primary is predictable.
            let enabled = self.mics.filter { $0 == mic ? on : self.settings.enabledMics.contains($0) }
            self.settings.enabledMics = enabled
            self.save(["enabled_mics": enabled])
        })
    }

    func loadStats() {
        let paths = paths
        Task.detached {
            let usage = Stats.usage(metrics: paths.metricsFile, training: paths.trainingDir)
            await MainActor.run { self.usage = usage }
        }
    }
}

// MARK: - The window

enum SettingsSection: String, CaseIterable, Identifiable {
    case recording = "Recording", models = "Models", instructions = "Instructions", advanced = "Advanced", stats = "Stats"
    var id: String { rawValue }
    var symbol: String {
        switch self {
        case .recording: "waveform"
        case .models: "cpu"
        case .instructions: "text.quote"
        case .advanced: "slider.horizontal.3"
        case .stats: "chart.bar.xaxis"
        }
    }
}

struct SettingsView: View {
    @ObservedObject var model: SettingsModel
    @ObservedObject var navigation: SettingsNavigation
    private var section: Binding<SettingsSection?> { $navigation.section }

    var body: some View {
        NavigationSplitView {
            List(SettingsSection.allCases, selection: section) { section in
                Label(section.rawValue, systemImage: section.symbol).tag(section)
            }
            .navigationSplitViewColumnWidth(200)
        } detail: {
            switch navigation.section ?? .recording {
            case .recording: RecordingPane(model: model)
            case .models: ModelsPane(model: model)
            case .instructions: InstructionsPane(model: model)
            case .advanced: AdvancedPane(model: model)
            case .stats: StatsPane(model: model)
            }
        }
    }
}

/// Which section is showing; outside the view so it can be driven for captures.
@MainActor
final class SettingsNavigation: ObservableObject {
    @Published var section: SettingsSection? = .recording
}

/// A title with a detail line under it, as System Settings rows have.
struct RowLabel: View {
    let title: String
    var detail: String = ""
    init(_ title: String, _ detail: String = "") { self.title = title; self.detail = detail }
    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(title)
            if !detail.isEmpty { Text(detail).font(.caption).foregroundStyle(.secondary) }
        }
    }
}

/// A number that clamps what you type and shows what was kept.
struct NumberField: View {
    @Binding var value: Double
    let range: ClosedRange<Double>
    var places = 2
    @State private var text = ""
    @FocusState private var focused: Bool

    var body: some View {
        TextField("", text: $text)
            .labelsHidden()
            .frame(width: 72)
            .multilineTextAlignment(.trailing)
            .focused($focused)
            .onAppear { text = Self.trim(value, places) }
            .onSubmit(commit)
            .onChange(of: focused) { _, now in if !now { commit() } }
    }

    func commit() {
        guard let parsed = Double(text.trimmingCharacters(in: .whitespaces)) else { text = Self.trim(value, places); return }
        let clamped = min(max(parsed, range.lowerBound), range.upperBound)
        value = clamped
        text = Self.trim(clamped, places)
    }

    /// No trailing decimal zeros: 2.50 -> "2.5", 450 -> "450".
    static func trim(_ value: Double, _ places: Int) -> String {
        let text = String(format: "%.\(places)f", value)
        guard text.contains(".") else { return text }
        var out = text
        while out.hasSuffix("0") { out.removeLast() }
        if out.hasSuffix(".") { out.removeLast() }
        return out
    }
}

struct RecordingPane: View {
    @ObservedObject var model: SettingsModel
    static let triggerKeys = [("cmd_r", "Right Command"), ("alt_r", "Right Option"), ("alt_l", "Left Option"),
                              ("ctrl_r", "Right Control"), ("f17", "F17"), ("f18", "F18")]

    var body: some View {
        Form {
            Section {
                if model.mics.isEmpty { Text("No microphones found") }
                ForEach(model.mics, id: \.self) { mic in Toggle(mic, isOn: model.micEnabled(mic)) }
            } header: { Text("Microphones") } footer: { Footer("Every enabled microphone is recorded in parallel.") }

            Section("Shortcut") {
                Picker(selection: model.bind("trigger_key", \.triggerKey)) {
                    ForEach(Self.triggerKeys + (Self.triggerKeys.contains { $0.0 == model.settings.triggerKey } ? []
                                                : [(model.settings.triggerKey, model.settings.triggerKey)]), id: \.0) {
                        Text($0.1).tag($0.0)
                    }
                } label: { RowLabel("Trigger key", "Hold to record, double-tap to toggle") }
                LabeledContent {
                    NumberField(value: Binding(get: { model.settings.doubleTapThreshold * 1000 },
                                               set: { model.bind("double_tap_threshold", \.doubleTapThreshold).wrappedValue = $0 / 1000 }),
                                range: 150...1000, places: 0)
                } label: { RowLabel("Double-tap window", "Milliseconds") }
            }

            Section("While Recording") {
                Toggle(isOn: model.bind("hud_enabled", \.hudEnabled)) { RowLabel("Recording HUD", "Mic level, live transcript and progress") }
                Toggle(isOn: model.bind("space_between_dictations", \.spaceBetweenDictations)) { RowLabel("Space between dictations", "When continuing in the same app") }
                Toggle(isOn: model.bind("duck_while_recording", \.duckWhileRecording)) { RowLabel("Quieter audio", "Turns other sound down while you talk, through speakers only") }
            }

            Section {
                Toggle(isOn: model.bind("chunk_on_silence", \.chunkOnSilence)) { RowLabel("Split on silence", "Transcribe while you're still talking") }
                Toggle(isOn: model.bind("adaptive_threshold_enabled", \.adaptiveThresholdEnabled)) { RowLabel("Adaptive threshold", "Follows the room's noise floor") }
                LabeledContent {
                    NumberField(value: model.bind("silence_threshold", \.silenceThreshold), range: 0.2...10)
                } label: { RowLabel("Split after silence", "Seconds") }
                LabeledContent {
                    NumberField(value: model.bind("speech_headroom_db", \.speechHeadroomDb), range: 1...30, places: 1)
                } label: { RowLabel("Speech threshold", "Decibels above the noise floor") }
            } header: { Text("Speech Detection") } footer: { Footer("Raise the threshold in noisy rooms; lower it if quiet speech gets missed.") }
        }
        .formStyle(.grouped)
        .navigationTitle("Recording")
    }
}

struct Footer: View {
    let text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(text).font(.caption).foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct ModelsPane: View {
    @ObservedObject var model: SettingsModel
    @State private var keyDraft = ""
    @State private var custom = false
    @State private var customModel = ""
    @State private var providerOrder = ""
    static let efforts = [("", "Auto"), ("none", "None"), ("minimal", "Minimal"), ("low", "Low"), ("medium", "Medium"),
                          ("high", "High"), ("xhigh", "XHigh")]

    var body: some View {
        Form {
            Section {
                LabeledContent {
                    SecureField("API key", text: $keyDraft, prompt: Text("sk-or-…"))
                        .labelsHidden()
                        .frame(width: 260)
                        .onSubmit { model.commitKey(keyDraft) }
                } label: {
                    VStack(alignment: .leading, spacing: 1) {
                        Text("API key")
                        Text(model.keyStatus.text).font(.caption)
                            .foregroundStyle(model.keyStatus.ok == true ? .green : model.keyStatus.ok == false ? .red : .secondary)
                    }
                }
            } header: { Text("OpenRouter") } footer: { Footer("Used for cloud transcription and for correction. Press Return to save the key.") }

            Section {
                Toggle(isOn: .constant(false)) { RowLabel("Parakeet", "Local, on this Mac: not in the Swift build yet") }.disabled(true)
                ForEach(model.shownModels, id: \.id) { entry in
                    Toggle(isOn: model.modelEnabled(entry.id)) {
                        RowLabel(entry.name, model.apiKey.isEmpty ? "Needs an OpenRouter key" : "")
                    }.disabled(model.apiKey.isEmpty)
                }
                Picker(selection: Binding(get: { "" }, set: { model.addModel($0) })) {
                    Text(model.catalogue == nil ? "Loading…" : addable.isEmpty ? (model.catalogue!.isEmpty ? "Couldn't reach OpenRouter" : "All added") : "Choose…").tag("")
                    ForEach(addable, id: \.id) { Text($0.name).tag($0.id) }
                } label: { RowLabel("Add a model", "From OpenRouter's speech-to-text models") }
                .disabled(model.apiKey.isEmpty)
            } header: { Text("Transcription") } footer: { Footer("Everything enabled runs in parallel; the correction model reconciles them.") }

            Section {
                Picker("Model", selection: Binding(
                    get: { custom ? "__custom__" : model.settings.openrouterCorrectionModel },
                    set: { choice in
                        custom = choice == "__custom__"
                        if !custom { model.bind("openrouter_correction_model", \.openrouterCorrectionModel).wrappedValue = choice }
                    })) {
                    ForEach(Catalogue.correctionModels, id: \.id) { Text($0.name).tag($0.id) }
                    Divider()
                    Text("Custom…").tag("__custom__")
                }
                if custom {
                    LabeledContent("Model ID") {
                        TextField("Model ID", text: $customModel, prompt: Text("vendor/model")).labelsHidden().frame(width: 240)
                            .onSubmit {
                                let id = customModel.trimmingCharacters(in: .whitespaces)
                                if !id.isEmpty { model.bind("openrouter_correction_model", \.openrouterCorrectionModel).wrappedValue = id }
                            }
                    }
                }
                Picker("Reasoning", selection: model.bind("openrouter_correction_reasoning_effort", \.openrouterCorrectionReasoningEffort)) {
                    ForEach(Self.efforts, id: \.0) { Text($0.1).tag($0.0) }
                }
            } header: { Text("Correction") } footer: {
                Footer(Catalogue.routingStatus(key: model.apiKey, model: model.settings.openrouterCorrectionModel,
                                               providerOrder: model.settings.openrouterCorrectionProviderOrder,
                                               effort: model.settings.openrouterCorrectionReasoningEffort))
            }

            Section("Provider Routing") {
                LabeledContent {
                    TextField("Provider order", text: $providerOrder, prompt: Text("e.g. together")).labelsHidden().frame(width: 200)
                        .onSubmit {
                            model.bind("openrouter_correction_provider_order", \.openrouterCorrectionProviderOrder).wrappedValue =
                                Catalogue.parseModelIds(providerOrder)
                        }
                } label: { RowLabel("Provider order", "Preferred OpenRouter providers") }
                Toggle("Allow fallbacks", isOn: model.bind("openrouter_correction_allow_fallbacks", \.openrouterCorrectionAllowFallbacks))
            }
        }
        .formStyle(.grouped)
        .navigationTitle("Models")
        .onAppear {
            keyDraft = model.apiKey
            custom = !Catalogue.correctionModels.contains { $0.id == model.settings.openrouterCorrectionModel }
            customModel = custom ? model.settings.openrouterCorrectionModel : ""
            providerOrder = model.settings.openrouterCorrectionProviderOrder.joined(separator: " ")
            model.loadCatalogue()
        }
    }

    var addable: [(id: String, name: String)] {
        (model.catalogue ?? []).filter { entry in !model.shownModels.contains { $0.id == entry.id } }
    }
}

struct InstructionsPane: View {
    @ObservedObject var model: SettingsModel

    var body: some View {
        Form {
            Section {
                PlainTextEditor(text: model.settings.customInstructions) { text in
                    model.bind("custom_instructions", \.customInstructions).wrappedValue = text
                }
                .frame(height: 170)
            } header: { Text("About You") } footer: {
                Footer("Who you are and the names you use. The model also sees the active app, so app-specific rules work here too.")
            }
            Section {
                Toggle(isOn: model.bind("learn_vocabulary", \.learnVocabulary)) {
                    RowLabel("Learn from corrections", "A word you correct in two separate dictations is shown to the model")
                }
            } header: { Text("Learning") } footer: {
                Footer(model.learned.isEmpty ? "Nothing learned yet." : "Learned so far: " + model.learned.joined(separator: ", "))
            }
        }
        .formStyle(.grouped)
        .navigationTitle("Instructions")
    }
}

struct AdvancedPane: View {
    @ObservedObject var model: SettingsModel
    @State private var generation = 0      // bumped by a reset, to refresh the editors

    var body: some View {
        Form {
            promptSection("Correction Prompt", key: "system_prompt", path: \.systemPrompt,
                          fallback: CorrectionPrompt.defaultSystem, height: 230,
                          footer: "How transcripts are cleaned up before they're typed.")
            promptSection("Editing Prompt", key: "editing_prompt", path: \.editingPrompt,
                          fallback: CorrectionPrompt.defaultEditing, height: 80,
                          footer: "Used when you select text and speak an instruction.")
        }
        .formStyle(.grouped)
        .navigationTitle("Advanced")
    }

    /// Only a genuine override is kept in the file, so the built-in default can evolve.
    func promptSection(_ title: String, key: String, path: WritableKeyPath<AppSettings, String>, fallback: String,
                       height: CGFloat, footer: String) -> some View {
        Section {
            PlainTextEditor(text: model.settings[keyPath: path].isEmpty ? fallback : model.settings[keyPath: path]) { text in
                if text.trimmingCharacters(in: .whitespacesAndNewlines) == fallback.trimmingCharacters(in: .whitespacesAndNewlines) {
                    model.settings[keyPath: path] = ""
                    model.remove([key])
                } else {
                    model.bind(key, path).wrappedValue = text
                }
            }
            .id("\(key)-\(generation)")
            .frame(height: height)
            HStack {
                Spacer()
                Button("Reset to Default") {
                    model.settings[keyPath: path] = ""
                    model.remove([key])
                    generation += 1
                }.controlSize(.small)
            }
        } header: { Text(title) } footer: { Footer(footer) }
    }
}

struct StatsPane: View {
    @ObservedObject var model: SettingsModel

    var body: some View {
        Form {
            Section("Dictated") {
                LabeledContent("Today", value: amount("today"))
                LabeledContent("Past 7 days", value: amount("week"))
                LabeledContent("All time", value: amount("all"))
            }
            Section {
                Bars(values: model.usage?.dailyWords ?? []).frame(height: 64).padding(.vertical, 6)
            } footer: { Footer("Words per day over the past two weeks; today is the brightest.") }
            Section {
                LabeledContent { Text(wpm(model.usage?.speakingWpm)) } label: { RowLabel("Speaking pace", "Words per minute while the key is held") }
                LabeledContent { Text(wpm(model.usage?.overallWpm)) } label: { RowLabel("End to end", "From pressing the key to the text being typed") }
                LabeledContent {
                    Text(model.usage?.afterRelease.map { String(format: "%.1f s · %.1f s", $0.median, $0.slow) } ?? "—")
                } label: { RowLabel("After you let go", "Until the text is done; median and slowest tenth") }
            } header: { Text("Speed") } footer: { Footer("Medians over the past week, for dictations of five words or more.") }
            Section {
                LabeledContent("Past 7 days", value: model.usage.map { usage in
                    usage.apps.isEmpty ? "—" : usage.apps.map { "\($0.app) \(Int(($0.share * 100).rounded()))%" }.joined(separator: "  ·  ")
                } ?? "…")
            } header: { Text("Where it goes") } footer: { Footer("The apps dictation went into.") }
        }
        .formStyle(.grouped)
        .navigationTitle("Stats")
        .onAppear { model.loadStats() }
    }

    func amount(_ period: String) -> String {
        guard let usage = model.usage else { return "…" }
        let words = usage.words[period] ?? 0, count = usage.dictations[period] ?? 0
        return "\(words.formatted()) words · \(count.formatted()) dictation\(count == 1 ? "" : "s")"
    }

    func wpm(_ value: Double?) -> String { value.map { "\(Int($0.rounded())) wpm" } ?? "—" }
}

/// Words per day as rounded bars in the accent colour; today is the last.
struct Bars: View {
    let values: [Int]
    var body: some View {
        GeometryReader { geo in
            let top = Double(max(values.max() ?? 1, 1)), gap = 6.0
            let width = values.isEmpty ? 0 : (geo.size.width - gap * Double(values.count - 1)) / Double(values.count)
            HStack(alignment: .bottom, spacing: gap) {
                ForEach(Array(values.enumerated()), id: \.offset) { index, value in
                    RoundedRectangle(cornerRadius: 3)
                        .fill(Color.accentColor.opacity(index == values.count - 1 ? 1 : 0.55))
                        .frame(width: max(0, width), height: max(2, geo.size.height * Double(value) / top))
                }
            }
            .frame(maxHeight: .infinity, alignment: .bottom)
        }
    }
}

/// A plain-text editor that saves when editing ends. Smart quotes, dashes and text
/// replacement are off: they would silently rewrite prompts that forbid those characters.
struct PlainTextEditor: NSViewRepresentable {
    let text: String
    let onCommit: (String) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onCommit: onCommit) }

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSTextView.scrollableTextView()
        let view = scroll.documentView as! NSTextView
        view.string = text
        view.font = .systemFont(ofSize: 13)
        view.isRichText = false
        view.isAutomaticQuoteSubstitutionEnabled = false
        view.isAutomaticDashSubstitutionEnabled = false
        view.isAutomaticTextReplacementEnabled = false
        view.drawsBackground = false
        view.textContainerInset = NSSize(width: 0, height: 4)
        view.delegate = context.coordinator
        scroll.drawsBackground = false
        scroll.borderType = .noBorder
        scroll.autohidesScrollers = true
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        context.coordinator.onCommit = onCommit
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var onCommit: (String) -> Void
        init(onCommit: @escaping (String) -> Void) { self.onCommit = onCommit }
        func textDidEndEditing(_ notification: Notification) {
            if let view = notification.object as? NSTextView { onCommit(view.string) }
        }
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

    /// Each section of the open window as a PNG (for checking layout; they show personal
    /// settings, so they stay on this Mac).
    func captureSections(to directory: URL) {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var remaining = SettingsSection.allCases
        func next() {
            guard let section = remaining.first else { exit(0) }
            remaining.removeFirst()
            navigation.section = section
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [self] in
                MainActor.assumeIsolated {
                    if let view = window?.contentView, let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) {
                        view.cacheDisplay(in: view.bounds, to: rep)
                        try? rep.representation(using: .png, properties: [:])?
                            .write(to: directory.appendingPathComponent("\(section.rawValue).png"))
                    }
                    next()
                }
            }
        }
        next()
    }

    func present(activate: Bool = true) {
        if window == nil {
            let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 780, height: 620),
                                  styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
                                  backing: .buffered, defer: false)
            window.title = "MergeScribe Settings"
            window.titleVisibility = .hidden
            window.titlebarAppearsTransparent = true
            window.contentMinSize = NSSize(width: 680, height: 480)
            window.isReleasedWhenClosed = false
            window.delegate = self
            window.contentView = NSHostingView(rootView: SettingsView(model: model, navigation: navigation))
            window.center()
            self.window = window
        } else if window?.isVisible == false {
            model.reload()
        }
        guard activate else {
            window?.orderBack(nil)       // visible for a capture, without taking focus from anything
            return
        }
        NSApp.setActivationPolicy(.regular)
        ensureEditMenu()
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate()
    }

    func windowWillClose(_ notification: Notification) {
        // Ending editing fires the fields' commits, so a half-typed field still saves.
        window?.makeFirstResponder(nil)
        NSApp.setActivationPolicy(.accessory)
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
