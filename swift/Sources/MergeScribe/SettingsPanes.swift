import AppKit
import AVFoundation
import Charts
import MergeScribeCore
import MergeScribeMac
import SwiftUI

// MARK: - Overview

struct OverviewPane: View {
    @ObservedObject var model: SettingsModel
    @ObservedObject var navigation: SettingsNavigation

    var body: some View {
        Page(title: "MergeScribe", subtitle: "Hold \(ShortcutRecorder.names[model.settings.triggerKey] ?? model.settings.triggerKey), talk, let go.") {
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 12), count: 4), spacing: 12) {
                StatTile(title: "Today", value: words(model.usage?.words["today"]), detail: count(model.usage?.dictations["today"]),
                         symbol: "sun.max")
                StatTile(title: "Past 7 days", value: words(model.usage?.words["week"]), detail: count(model.usage?.dictations["week"]),
                         symbol: "calendar")
                StatTile(title: "You speak at", value: model.usage?.speakingWpm.map { "\(Int($0.rounded())) wpm" } ?? "—",
                         detail: "while the key is held", symbol: "waveform")
                StatTile(title: "After you let go", value: model.usage?.afterRelease.map { String(format: "%.1f s", $0.median) } ?? "—",
                         detail: "median, until it's typed", symbol: "bolt")
            }

            GroupBox {
                HStack(alignment: .top, spacing: 4) {
                    Stage(symbol: "mic", title: model.settings.enabledMics.first ?? "No microphone",
                          detail: model.settings.keepMicOpen ? "always listening" : "opens on the key")
                    StageArrow()
                    Stage(symbol: "text.bubble", title: recognizersTitle, detail: "fastest " + duration(fastestRecognizer))
                    StageArrow()
                    if model.settings.judgeEnabled {
                        Stage(symbol: "checkmark.seal", title: "Jev", detail: judgeLine, tint: .green)
                        StageArrow()
                    }
                    Stage(symbol: "wand.and.stars", title: correctionName,
                          detail: duration(model.report?.corrections.first { $0.model == model.settings.openrouterCorrectionModel }?.medianMs),
                          tint: .purple)
                    StageArrow()
                    Stage(symbol: "keyboard", title: "Typed where you started",
                          detail: model.settings.insertViaAccessibility ? "or inserted, in native apps" : "as keystrokes", tint: .orange)
                }
                .padding(.vertical, 8)
            } label: {
                Text("How a dictation goes").font(.headline)
            }

            if let history = model.report?.history.prefix(4), !history.isEmpty {
                GroupBox {
                    VStack(alignment: .leading, spacing: 10) {
                        ForEach(Array(history)) { entry in
                            HStack(alignment: .firstTextBaseline) {
                                Text(entry.text).lineLimit(1)
                                Spacer()
                                Text(entry.app).foregroundStyle(.secondary).font(.caption)
                                Text(entry.date, format: .relative(presentation: .named, unitsStyle: .abbreviated))
                                    .foregroundStyle(.tertiary).font(.caption)
                            }
                        }
                        Button("All dictations") { navigation.section = .history }.buttonStyle(.link)
                    }
                    .padding(6)
                } label: {
                    Text("Recently").font(.headline)
                }
            }
        }
    }

    var recognizersTitle: String {
        let names = model.activeRecognizers
        return names.count > 2 ? "\(names.count) recognizers" : names.isEmpty ? "No recognizer" : names.joined(separator: ", ")
    }

    /// The quickest of the recognizers switched on now.
    var fastestRecognizer: Double? {
        var names = Set(model.settings.openrouterSttModels.map(TranscriptionClient.providerName))
        if model.settings.enabledProviders.contains(AppleProvider.name) { names.insert(AppleProvider.name) }
        if model.settings.streamingEnabled { names.insert("assemblyai-" + model.settings.streamingModel.replacingOccurrences(of: ".", with: "-")) }
        return model.report?.recognizers.filter { names.contains($0.name) && $0.answers > 0 }.compactMap(\.medianMs).min()
    }

    var correctionName: String {
        Catalogue.correctionModels.first { $0.id == model.settings.openrouterCorrectionModel }?.name ?? model.settings.openrouterCorrectionModel
    }

    var judgeLine: String {
        guard let judge = model.report?.judge, judge.verdicts > 0 else { return "skips correction when it isn't needed" }
        return "nothing to fix in \(Int((Double(judge.clean) / Double(judge.verdicts) * 100).rounded()))%"
    }

    func words(_ value: Int?) -> String { value.map { "\($0.formatted()) words" } ?? "…" }
    func count(_ value: Int?) -> String { value.map { "\($0.formatted()) dictation\($0 == 1 ? "" : "s")" } ?? "" }
}

// MARK: - Recording

struct RecordingPane: View {
    @ObservedObject var model: SettingsModel

    var body: some View {
        Form {
            Section {
                HUDPreview()
                    .listRowInsets(EdgeInsets(top: 8, leading: 8, bottom: 8, trailing: 8))
                Toggle(isOn: model.bind("hud_enabled", \.hudEnabled)) {
                    RowLabel("Recording HUD", "Where you're looking: your voice, the recognizers, then the words landing")
                }
                Toggle(isOn: model.bind("live_words", \.liveWords)) {
                    RowLabel("Live words", model.appleAvailable
                             ? "Apple's on-device recognizer shows what you're saying as you say it"
                             : "Needs Apple's on-device speech model (macOS 26)")
                }
                .disabled(!model.settings.hudEnabled || !model.appleAvailable)
            } header: { Text("While You Talk") }

            Section {
                LabeledContent {
                    ShortcutRecorder(key: model.bind("trigger_key", \.triggerKey))
                } label: {
                    RowLabel("Trigger key", "Hold to record, double-tap to keep recording hands-free; with Shift, edit the last dictation")
                }
                LabeledContent {
                    NumberField(value: Binding(get: { model.settings.doubleTapThreshold * 1000 },
                                               set: { model.bind("double_tap_threshold", \.doubleTapThreshold).wrappedValue = $0 / 1000 }),
                                range: 150...1000, places: 0)
                } label: { RowLabel("Double-tap window", "Milliseconds") }
            } header: { Text("Shortcut") }

            Section {
                if model.mics.isEmpty { Text("No microphones found") }
                ForEach(model.mics, id: \.self) { mic in Toggle(mic, isOn: model.micEnabled(mic)) }
                Toggle(isOn: model.bind("keep_mic_open", \.keepMicOpen)) {
                    RowLabel("Keep the microphone on", "Catches a word said as the key goes down; the mic indicator stays on")
                }
            } header: { Text("Microphones") } footer: { Footer("Every enabled microphone is recorded in parallel; the first is the one chunks are cut on.") }

            Section("Typing") {
                Toggle(isOn: model.bind("space_between_dictations", \.spaceBetweenDictations)) {
                    RowLabel("Space between dictations", "When you carry on in the same place")
                }
                Toggle(isOn: model.bind("insert_via_accessibility", \.insertViaAccessibility)) {
                    RowLabel("Insert text directly", "In native apps: exact and instant. Browsers and Electron apps get keystrokes")
                }
                Toggle(isOn: model.bind("duck_while_recording", \.duckWhileRecording)) {
                    RowLabel("Quieter audio", "Turns other sound down while you talk, through speakers only")
                }
            }
        }
        .formStyle(.grouped)
    }
}

// MARK: - Transcription

struct TranscriptionPane: View {
    @ObservedObject var model: SettingsModel
    @State private var keyDraft = ""
    @FocusState private var keyFocused: Bool
    @State private var assemblyDraft = ""
    @FocusState private var assemblyFocused: Bool

    var body: some View {
        Form {
            Section {
                LabeledContent {
                    SecureField("API key", text: $keyDraft, prompt: Text("sk-or-…")).labelsHidden().frame(width: 260)
                        .focused($keyFocused)
                        .onSubmit { model.commitKey(keyDraft) }
                        .onChange(of: keyFocused) { _, now in if !now { model.commitKey(keyDraft) } }
                } label: {
                    VStack(alignment: .leading, spacing: 1) {
                        Text("OpenRouter key")
                        Text(model.keyStatus.text).font(.caption)
                            .foregroundStyle(model.keyStatus.ok == true ? .green : model.keyStatus.ok == false ? .red : .secondary)
                    }
                }
            } footer: { Footer("Used for cloud transcription and for correction.") }

            Section {
                Toggle(isOn: model.bind("assemblyai_streaming", \.streamingEnabled)) {
                    RowLabel("Stream while you talk", model.assemblyKey.isEmpty
                             ? "Needs an AssemblyAI key"
                             : "Universal 3.6 Pro hears you as you speak; the text is ready about 0.15 s after you stop")
                }
                .disabled(model.assemblyKey.isEmpty)
                LabeledContent {
                    SecureField("AssemblyAI key", text: $assemblyDraft, prompt: Text("AssemblyAI key")).labelsHidden().frame(width: 260)
                        .focused($assemblyFocused)
                        .onSubmit { model.commitAssemblyKey(assemblyDraft) }
                        .onChange(of: assemblyFocused) { _, now in if !now { model.commitAssemblyKey(assemblyDraft) } }
                } label: { RowLabel("AssemblyAI key", "For streaming only") }
            } header: { Text("Streaming") } footer: {
                Footer("While streaming, the recognizers below run only if switched on, on the whole recording. "
                       + "If the stream fails, batch Universal through OpenRouter transcribes it instead.")
            }

            Section {
                ForEach(model.shownModels, id: \.id) { entry in
                    Toggle(isOn: model.modelEnabled(entry.id)) {
                        RowLabel(entry.name, model.apiKey.isEmpty ? "Needs an OpenRouter key" : line(for: TranscriptionClient.providerName(entry.id)))
                    }.disabled(model.apiKey.isEmpty)
                }
                Toggle(isOn: model.providerEnabled(AppleProvider.name)) {
                    RowLabel("Apple, on this Mac", model.appleAvailable
                             ? "Free and offline; less accurate than the cloud ones on your dictations. " + line(for: AppleProvider.name)
                             : "Needs macOS 26 and its English speech model")
                }.disabled(!model.appleAvailable)
                Picker(selection: Binding(get: { "" }, set: { model.addModel($0) })) {
                    Text(model.catalogue == nil ? "Loading…" : addable.isEmpty ? (model.catalogue!.isEmpty ? "Couldn't reach OpenRouter" : "All added") : "Choose…").tag("")
                    ForEach(addable, id: \.id) { Text($0.name).tag($0.id) }
                } label: { RowLabel("Add a model", "From OpenRouter's speech-to-text models") }
                .disabled(model.apiKey.isEmpty)
            } header: { Text("Recognizers") } footer: {
                Footer("Everything on runs in parallel on every stretch of speech; the correction model reconciles them. Times are medians over the past week.")
            }

            Section {
                LabeledContent {
                    NumberField(value: Binding(get: { Double(model.settings.providerDeadlineMinMs) },
                                               set: { model.bind("provider_deadline_min_ms", \.providerDeadlineMinMs).wrappedValue = Int($0) }),
                                range: 0...30000, places: 0)
                } label: { RowLabel("Wait at least", "Milliseconds before leaving a slow recognizer behind") }
                LabeledContent {
                    NumberField(value: model.bind("provider_deadline_multiplier", \.providerDeadlineMultiplier), range: 0...10)
                } label: { RowLabel("And up to", "Times the fastest one's time on top (0 waits for all)") }
            } header: { Text("Waiting") } footer: {
                Footer("Once the fastest answers with words, the others get this long. When two agree word for word, nothing more is waited for.")
            }
        }
        .formStyle(.grouped)
        .onAppear {
            keyDraft = model.apiKey
            assemblyDraft = model.assemblyKey
            model.loadCatalogue()
        }
        .onDisappear {
            model.commitKey(keyDraft)
            model.commitAssemblyKey(assemblyDraft)
        }
    }

    func line(for provider: String) -> String {
        guard let r = model.insight(for: provider), r.answers > 0 else { return "" }
        var parts = ["\(duration(r.medianMs)) typical", "\(duration(r.p90Ms)) slow"]
        if r.leftBehind > 0 { parts.append("left behind \(r.leftBehind)×") }
        return parts.joined(separator: " · ")
    }

    var addable: [(id: String, name: String)] {
        (model.catalogue ?? []).filter { entry in !model.shownModels.contains { $0.id == entry.id } }
    }
}

// MARK: - Correction

struct CorrectionPane: View {
    @ObservedObject var model: SettingsModel
    @State private var custom = false
    @State private var customModel = ""
    @State private var providerOrder = ""
    @FocusState private var focused: String?
    static let efforts = [("", "Auto"), ("none", "None"), ("minimal", "Minimal"), ("low", "Low"), ("medium", "Medium"),
                          ("high", "High"), ("xhigh", "XHigh")]

    var body: some View {
        Form {
            Section {
                Picker(selection: Binding(
                    get: { custom ? "__custom__" : model.settings.openrouterCorrectionModel },
                    set: { choice in
                        custom = choice == "__custom__"
                        if !custom { model.bind("openrouter_correction_model", \.openrouterCorrectionModel).wrappedValue = choice }
                    })) {
                    ForEach(Catalogue.correctionModels, id: \.id) { Text($0.name).tag($0.id) }
                    Divider()
                    Text("Custom…").tag("__custom__")
                } label: { RowLabel("Model", modelLine) }
                if custom {
                    LabeledContent("Model ID") {
                        TextField("Model ID", text: $customModel, prompt: Text("vendor/model")).labelsHidden().frame(width: 240)
                            .focused($focused, equals: "model")
                            .onSubmit(commitModel)
                    }
                }
                Picker("Reasoning", selection: model.bind("openrouter_correction_reasoning_effort", \.openrouterCorrectionReasoningEffort)) {
                    ForEach(Self.efforts, id: \.0) { Text($0.1).tag($0.0) }
                }
            } header: { Text("Correction") } footer: {
                Footer("Filler and false starts go, mishearings are fixed, and the text is punctuated, in the words you used. "
                       + Catalogue.routingStatus(key: model.apiKey, model: model.settings.openrouterCorrectionModel,
                                                 providerOrder: model.settings.openrouterCorrectionProviderOrder,
                                                 effort: model.settings.openrouterCorrectionReasoningEffort))
            }

            Section {
                Toggle(isOn: model.bind("judge_enabled", \.judgeEnabled)) {
                    RowLabel("Skip correction when it isn't needed", judgeLine)
                }
            } header: { Text("Jev") } footer: {
                Footer("A classifier is asked, alongside the correction, whether the transcript already reads as you meant it. "
                       + "If it answers first and says yes, the transcript is typed at once. Your Instructions are checked too.")
            }

            Section("Provider Routing") {
                LabeledContent {
                    TextField("Provider order", text: $providerOrder, prompt: Text("e.g. openai/fast")).labelsHidden().frame(width: 200)
                        .focused($focused, equals: "order")
                        .onSubmit(commitOrder)
                } label: { RowLabel("Provider order", "Preferred OpenRouter providers") }
                Toggle("Allow fallbacks", isOn: model.bind("openrouter_correction_allow_fallbacks", \.openrouterCorrectionAllowFallbacks))
            }

            Section {
                PlainTextEditor(text: model.settings.customInstructions) { text in
                    model.bind("custom_instructions", \.customInstructions).wrappedValue = text
                }
                .frame(height: 170)
            } header: { Text("Instructions") } footer: {
                Footer("Who you are, the names you use, and how you like to write. The model sees the app and window you're in, so rules like \"lowercase in Slack\" work.")
            }
        }
        .formStyle(.grouped)
        .onChange(of: focused) { was, _ in
            if was == "model" { commitModel() }
            if was == "order" { commitOrder() }
        }
        .onDisappear {
            commitModel()
            commitOrder()
        }
        .onAppear {
            custom = !Catalogue.correctionModels.contains { $0.id == model.settings.openrouterCorrectionModel }
            customModel = custom ? model.settings.openrouterCorrectionModel : ""
            providerOrder = model.settings.openrouterCorrectionProviderOrder.joined(separator: " ")
        }
    }

    func commitModel() {
        let id = customModel.trimmingCharacters(in: .whitespaces)
        if custom, !id.isEmpty { model.bind("openrouter_correction_model", \.openrouterCorrectionModel).wrappedValue = id }
    }

    func commitOrder() {
        model.bind("openrouter_correction_provider_order", \.openrouterCorrectionProviderOrder).wrappedValue =
            Catalogue.parseModelIds(providerOrder)
    }

    var modelLine: String {
        guard let c = model.report?.corrections.first(where: { $0.model == model.settings.openrouterCorrectionModel }), c.count > 0 else { return "" }
        return "\(duration(c.medianMs)) typical, \(duration(c.p90Ms)) slow, over \(c.count) corrections this week"
    }

    var judgeLine: String {
        guard let judge = model.report?.judge, judge.verdicts > 0 else { return "Answers in about a quarter of a second" }
        let share = Int((Double(judge.clean) / Double(judge.verdicts) * 100).rounded())
        return "Found nothing to fix in \(share)% of \(judge.verdicts) this week; answers in \(duration(judge.medianMs))"
    }
}

// MARK: - Vocabulary

struct VocabularyPane: View {
    @ObservedObject var model: SettingsModel
    @State private var newTerm = ""

    var body: some View {
        Form {
            Section {
                Toggle(isOn: model.bind("learn_vocabulary", \.learnVocabulary)) {
                    RowLabel("Use your vocabulary", "Your words below, and corrections you've made the same way twice")
                }
                if model.supplied.isEmpty {
                    Text("None yet").foregroundStyle(.secondary)
                } else {
                FlowLayout(spacing: 6) {
                    ForEach(model.supplied, id: \.self) { term in
                        HStack(spacing: 4) {
                            Text(term)
                            Button {
                                model.setSupplied(model.supplied.filter { $0 != term })
                            } label: { Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary) }
                            .buttonStyle(.plain)
                        }
                        .padding(.horizontal, 9).padding(.vertical, 4)
                        .background(Capsule().fill(Color.accentColor.opacity(0.12)))
                    }
                }
                .padding(.vertical, 4)
                }
                HStack {
                    TextField("Add a word or name", text: $newTerm, prompt: Text("Add a word or name")).labelsHidden()
                        .onSubmit(add)
                    Button("Add", action: add).disabled(newTerm.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            } header: { Text("Your Words") } footer: {
                Footer("Names, products and jargon you say. The recognizers that take a vocabulary are primed with them, and the correction model is told how you spell them. Kept in ~/.mergescribe/vocabulary.txt.")
            }

            Section {
                if model.learned.isEmpty {
                    Text("Nothing learned yet").foregroundStyle(.secondary)
                }
                ForEach(model.learned, id: \.after) { correction in
                    LabeledContent {
                        HStack(spacing: 10) {
                            Text("\(correction.count) dictations").foregroundStyle(.secondary).font(.caption)
                            Button { model.forget(correction) } label: {
                                Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                            .help("Forget this one")
                        }
                    } label: {
                        HStack(spacing: 6) {
                            Text(correction.before.joined(separator: ", ")).foregroundStyle(.secondary).strikethrough()
                            Image(systemName: "arrow.right").font(.caption).foregroundStyle(.tertiary)
                            Text(correction.after).fontWeight(.medium)
                        }
                    }
                }
            } header: { Text("Learned") } footer: {
                Footer("A word you fix the same way after two separate dictations, shown to the correction model as evidence. Everyday words like \"it\" come from rewording and aren't passed to the recognizers. Needs Watch for your edits, in Advanced.")
            }
        }
        .formStyle(.grouped)
    }

    func add() {
        let terms = newTerm.split(separator: ",").map { String($0) }
        model.setSupplied(model.supplied + terms)
        newTerm = ""
    }
}

/// Lays children out in rows, wrapping, like words in a paragraph.
struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? 400
        var x = 0.0, y = 0.0, row = 0.0
        for view in subviews {
            let size = view.sizeThatFits(.unspecified)
            if x + size.width > width, x > 0 { x = 0; y += row + spacing; row = 0 }
            x += size.width + spacing
            row = max(row, size.height)
        }
        return CGSize(width: width, height: y + row)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, row = 0.0
        for view in subviews {
            let size = view.sizeThatFits(.unspecified)
            if x + size.width > bounds.maxX, x > bounds.minX { x = bounds.minX; y += row + spacing; row = 0 }
            view.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            row = max(row, size.height)
        }
    }
}

// MARK: - History

struct HistoryPane: View {
    @ObservedObject var model: SettingsModel
    @State private var search = ""
    @State private var selection: String?
    @State private var detail: Insights.Detail?
    @State private var player: AVAudioPlayer?
    @State private var playing = false

    var body: some View {
        HSplitView {
            VStack(spacing: 0) {
                TextField("Search", text: $search, prompt: Text("Search dictations"))
                    .textFieldStyle(.roundedBorder)
                    .padding(.horizontal, 10).padding(.top, 10).padding(.bottom, 6)
                List(selection: $selection) {
                    ForEach(filtered) { entry in
                        VStack(alignment: .leading, spacing: 3) {
                            Text(entry.text).lineLimit(2)
                            HStack(spacing: 6) {
                                Text(entry.app.isEmpty ? "—" : entry.app)
                                Text("·")
                                Text(entry.date, format: .dateTime.month(.abbreviated).day().hour().minute())
                                Text("·")
                                Text("\(entry.words) words")
                            }
                            .font(.caption).foregroundStyle(.secondary)
                        }
                        .padding(.vertical, 2)
                        .tag(entry.id)
                    }
                }
            }
            .frame(minWidth: 280)

            Group {
                if let entry = selected {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 14) {
                            Text(detail?.typed ?? entry.text + (entry.clipped ? "…" : "")).textSelection(.enabled).font(.title3)
                            HStack {
                                Button("Copy") { MacTextOutput().copyToClipboard(detail?.typed ?? entry.text) }
                                if let audio = detail?.audio {
                                    Button(playing ? "Stop" : "Play recording") { toggle(audio) }
                                }
                            }
                            if let detail {
                                Text(methodLine(detail)).font(.caption).foregroundStyle(.secondary)
                                Divider()
                                Text("What each recognizer heard").font(.headline)
                                ForEach(Array(detail.transcripts.enumerated()), id: \.offset) { _, t in
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text("\(recognizerName(t.provider)) · \(duration(Double(t.latencyMs)))").font(.caption).foregroundStyle(.secondary)
                                        Text(t.text.isEmpty ? "(nothing)" : t.text).textSelection(.enabled)
                                    }
                                }
                            } else {
                                if entry.clipped {
                                    Text("The log kept only the start of this one.").font(.caption).foregroundStyle(.secondary)
                                }
                                Text("Turn on Keep recordings in Advanced to keep what each recognizer heard.")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        .padding(20)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                } else {
                    Text(model.report == nil ? "Loading…" : "Choose a dictation").foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .frame(minWidth: 280)
        }
        .onChange(of: selection) { _, _ in
            stopPlaying()
            detail = nil
            guard let entry = selected else { return }
            Task.detached {
                let found = Insights.detail(entry)
                await MainActor.run { if selection == entry.id { detail = found } }
            }
        }
        .onDisappear(perform: stopPlaying)
    }

    var filtered: [Insights.Entry] {
        let all = model.report?.history ?? []
        guard !search.isEmpty else { return all }
        return all.filter { $0.text.localizedCaseInsensitiveContains(search) || $0.app.localizedCaseInsensitiveContains(search) }
    }

    var selected: Insights.Entry? { model.report?.history.first { $0.id == selection } }

    func methodLine(_ d: Insights.Detail) -> String {
        switch d.method {
        case "judged": return "Typed as heard: Jev found nothing to correct"
        case "streamed": return "Corrected by \(d.correctionModel ?? "the correction model") and streamed in"
        case "clipboard": return "Put on the clipboard: the window had gone"
        case "typed": return "Typed as heard"
        default: return d.method
        }
    }

    func recognizerName(_ provider: String) -> String {
        if provider == AppleProvider.name { return "Apple" }
        return model.shownModels.first { TranscriptionClient.providerName($0.id) == provider }?.name ?? provider
    }

    func toggle(_ url: URL) {
        if playing { stopPlaying(); return }
        guard let player = try? AVAudioPlayer(contentsOf: url), player.play() else { return }
        self.player = player
        playing = true
        DispatchQueue.main.asyncAfter(deadline: .now() + player.duration + 0.1) {
            if self.player === player { stopPlaying() }
        }
    }

    func stopPlaying() {
        player?.stop()
        player = nil
        playing = false
    }
}

// MARK: - Stats

struct StatsPane: View {
    @ObservedObject var model: SettingsModel

    struct Day: Identifiable { let id: Int; let date: Date; let words: Int }

    var body: some View {
        Page(title: "Stats", subtitle: "What dictation has added up to.") {
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 12), count: 3), spacing: 12) {
                StatTile(title: "Today", value: amount("today"), detail: dictations("today"))
                StatTile(title: "Past 7 days", value: amount("week"), detail: dictations("week"))
                StatTile(title: "All time", value: amount("all"), detail: dictations("all"))
                StatTile(title: "Speaking pace", value: wpm(model.usage?.speakingWpm), detail: "while the key is held")
                StatTile(title: "End to end", value: wpm(model.usage?.overallWpm), detail: "press to typed")
                StatTile(title: "After you let go", value: model.usage?.afterRelease.map { String(format: "%.1f s", $0.median) } ?? "—",
                         detail: model.usage?.afterRelease.map { String(format: "slowest tenth %.1f s", $0.slow) } ?? "")
            }

            GroupBox {
                Chart(days) { day in
                    BarMark(x: .value("Day", day.date, unit: .day), y: .value("Words", day.words))
                        .foregroundStyle(day.id == days.count - 1 ? Color.accentColor : Color.accentColor.opacity(0.5))
                        .cornerRadius(3)
                }
                .chartXAxis { AxisMarks(values: .stride(by: .day, count: 2)) { _ in AxisValueLabel(format: .dateTime.weekday(.abbreviated)) } }
                .frame(height: 170)
                .padding(8)
            } label: { Text("Words per day").font(.headline) }

            if let apps = model.usage?.apps, !apps.isEmpty {
                GroupBox {
                    Chart(apps, id: \.app) { entry in
                        BarMark(x: .value("Share", entry.share), y: .value("App", entry.app))
                            .annotation(position: .trailing) { Text("\(Int((entry.share * 100).rounded()))%").font(.caption) }
                    }
                    .chartXAxis(.hidden)
                    .frame(height: CGFloat(apps.count) * 34 + 10)
                    .padding(8)
                } label: { Text("Where it goes, past 7 days").font(.headline) }
            }
        }
    }

    var days: [Day] {
        let values = model.usage?.dailyWords ?? []
        let today = Calendar.current.startOfDay(for: Date())
        return values.enumerated().map { index, words in
            Day(id: index, date: Calendar.current.date(byAdding: .day, value: index - (values.count - 1), to: today)!, words: words)
        }
    }

    func amount(_ period: String) -> String { model.usage.map { "\(($0.words[period] ?? 0).formatted()) words" } ?? "…" }
    func dictations(_ period: String) -> String {
        guard let count = model.usage?.dictations[period] else { return "" }
        return "\(count.formatted()) dictation\(count == 1 ? "" : "s")"
    }
    func wpm(_ value: Double?) -> String { value.map { "\(Int($0.rounded())) wpm" } ?? "—" }
}

// MARK: - Advanced

struct AdvancedPane: View {
    @ObservedObject var model: SettingsModel
    @State private var generation = 0      // bumped by a reset, to refresh the editors

    var body: some View {
        Form {
            Section {
                Toggle(isOn: model.bind("chunk_on_silence", \.chunkOnSilence)) { RowLabel("Split on silence", "Transcribe while you're still talking") }
                Toggle(isOn: model.bind("adaptive_threshold_enabled", \.adaptiveThresholdEnabled)) { RowLabel("Adaptive threshold", "Follows each mic's noise floor") }
                LabeledContent {
                    NumberField(value: model.bind("silence_threshold", \.silenceThreshold), range: 0.2...10)
                } label: { RowLabel("Split after silence", "Seconds") }
                LabeledContent {
                    NumberField(value: model.bind("speech_headroom_db", \.speechHeadroomDb), range: 1...30, places: 1)
                } label: { RowLabel("Speech threshold", "Decibels above the noise floor") }
            } header: { Text("Speech Detection") } footer: { Footer("Raise the threshold in noisy rooms; lower it if quiet speech gets missed.") }

            promptSection("Correction Prompt", key: "system_prompt", path: \.systemPrompt, fallback: CorrectionPrompt.defaultSystem,
                          height: 230, footer: "How transcripts are cleaned up before they're typed.")
            promptSection("Editing Prompt", key: "editing_prompt", path: \.editingPrompt, fallback: CorrectionPrompt.defaultEditing,
                          height: 80, footer: "Used when you select text and speak an instruction, or edit the last dictation with Shift.")

            Section {
                Toggle(isOn: model.bind("edit_feedback_enabled", \.editFeedbackEnabled)) {
                    RowLabel("Watch for your edits", "What you change after a dictation is how vocabulary is learned")
                }
                Toggle(isOn: model.bind("training_enabled", \.trainingEnabled)) {
                    RowLabel("Keep recordings", "Audio and what became of it, for History and for fine-tuning")
                }
                HStack {
                    Button("Show Data Folder") { NSWorkspace.shared.open(model.paths.dataDir) }
                    Button("Open Log") { NSWorkspace.shared.open(model.paths.logFile) }
                }
            } header: { Text("Your Data") } footer: { Footer("Everything stays on this Mac, in ~/.mergescribe.") }
        }
        .formStyle(.grouped)
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
