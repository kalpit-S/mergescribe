import Foundation

/// Where MergeScribe keeps its files. Everything lives under ~/.mergescribe, in the
/// same layout the Python app used, so its scripts keep reading the same files.
public struct AppPaths: Sendable, Equatable {
    public var dataDir: URL

    public init(dataDir: URL = FileManager.default.homeDirectoryForCurrentUser
                    .appendingPathComponent(".mergescribe", isDirectory: true)) {
        self.dataDir = dataDir
    }

    public var settingsFile: URL { dataDir.appendingPathComponent("settings.json") }
    public var envFile: URL { dataDir.appendingPathComponent(".env") }
    public var metricsFile: URL { dataDir.appendingPathComponent("metrics.jsonl") }
    public var correctionsFile: URL { dataDir.appendingPathComponent("corrections.jsonl") }
    public var vocabularyFile: URL { dataDir.appendingPathComponent("vocabulary.txt") }
    public var trainingDir: URL { dataDir.appendingPathComponent("training", isDirectory: true) }
    public var logFile: URL { dataDir.appendingPathComponent("logs/mergescribe.log") }
}

/// Every setting the app reads, with its default.
///
/// One definition: the defaults are the property initialisers, and `fields` maps each
/// property to its settings.json key. A session copies the whole struct when it starts,
/// so a change made in Settings mid-dictation applies from the next one.
public struct Settings: Sendable, Equatable {
    // Audio
    public var enabledMics: [String] = []
    /// One second is enough to catch the start of speech.
    public var prerollSeconds: Double = 1.0
    /// Sits just above the p90 of measured pause length (1.09s), so chunks break at
    /// sentence boundaries without cutting the frequent sub-second thinking pauses.
    public var silenceThreshold: Double = 1.2
    public var sampleRate: Int = 16000
    /// Keep the microphones open between dictations. Pre-roll (the second before the press)
    /// needs it; off, a microphone opens when the key goes down and the indicator goes away.
    public var keepMicOpen = false
    public var chunkOnSilence = true
    public var adaptiveThresholdEnabled = true
    public var speechHeadroomDb: Double = 12.0
    public var speechHysteresisDb: Double = 3.0
    public var noiseFloorPercentile: Int = 10

    // Input
    public var triggerKey = "alt_r"
    public var doubleTapThreshold: Double = 0.45
    public var toggleModeTimeout: Int = 600

    // Recognizers
    public var enabledProviders: [String] = ["parakeet"]
    public var openrouterSttModels: [String] = []

    // Processing
    public var consensusThreshold: Int = 2
    /// Skip correction when a classifier says the transcript needs none. Runs beside the
    /// correction request, so a slow answer costs nothing.
    public var judgeEnabled = false
    public var judgeModel = "~typesafe/jev-latest"
    public var judgeTimeoutMs: Int = 900
    /// Once the fastest recognizer has answered, the rest get this much longer (1.0 = twice
    /// the fastest one's time) before being left behind, but never less than
    /// `providerDeadlineMinMs`. The floor is deliberately high: cloud recognizers are often
    /// slower and more accurate, and a low floor systematically drops them.
    public var providerDeadlineMultiplier: Double = 1.0
    public var providerDeadlineMinMs: Int = 2000

    // Correction
    public var openrouterCorrectionModel = "openai/gpt-6-luna"
    public var openrouterCorrectionProviderOrder: [String] = []
    public var openrouterCorrectionAllowFallbacks = true
    public var openrouterCorrectionReasoningEffort = ""
    public var customInstructions = ""
    public var systemPrompt = ""
    public var editingPrompt = ""

    // Data kept on this Mac
    public var trainingEnabled = false
    /// Watch the destination field after typing and record what the user changed.
    public var editFeedbackEnabled = true
    /// Terms corrected the same way in two separate dictations are offered to the
    /// correction model as evidence.
    public var learnVocabulary = true

    // While recording
    public var hudEnabled = true
    /// Apple's on-device recognizer listens along and puts your words on the HUD as you say
    /// them. Display only: what gets typed still comes from the ensemble and the correction.
    public var liveWords = true
    /// Lower the system output while recording, when it plays through speakers.
    public var duckWhileRecording = true
    /// A leading space when continuing a recent dictation into the same destination.
    public var spaceBetweenDictations = true
    /// Put text into native text fields through accessibility (exact and instant) instead of
    /// synthetic keystrokes. Web pages and Electron apps always get keystrokes.
    public var insertViaAccessibility = true

    /// From ~/.mergescribe/.env or the environment; never written to settings.json.
    public var openrouterApiKey = ""

    public init() {}
}

// MARK: - Reading the files

extension Settings {
    /// Defaults, then the .env files and environment, then settings.json.
    public static func load(paths: AppPaths = AppPaths(),
                            environment: [String: String] = ProcessInfo.processInfo.environment,
                            currentDirectory: URL? = URL(fileURLWithPath: FileManager.default.currentDirectoryPath),
                            log: (String) -> Void = { print($0) }) -> Settings {
        var settings = Settings()
        // The project-root .env was the Python app's first place to look; it only
        // matters when run from a checkout.
        var envFiles = [paths.envFile]
        if let currentDirectory { envFiles.insert(currentDirectory.appendingPathComponent(".env"), at: 0) }
        for file in envFiles {
            if let key = EnvFile.read(file)["OPENROUTER_API_KEY"] { settings.openrouterApiKey = key }
        }
        if let key = environment["OPENROUTER_API_KEY"] { settings.openrouterApiKey = key }

        guard let data = try? Data(contentsOf: paths.settingsFile) else { return settings }
        guard let object = try? JSONSerialization.jsonObject(with: data),
              let values = object as? [String: Any] else {
            log("Error loading \(paths.settingsFile.lastPathComponent): not a JSON object")
            return settings
        }
        settings.apply(values, log: log)
        return settings
    }

    /// Apply values from settings.json, ignoring any that can't be read as their type.
    public mutating func apply(_ values: [String: Any], log: (String) -> Void = { print($0) }) {
        for field in Settings.fields {
            guard let raw = values[field.key] else { continue }
            if !field.apply(&self, raw) {
                log("Ignoring \(field.key)=\(raw) in settings.json: expected \(field.kind)")
            }
        }
    }

    /// The values as settings.json holds them (the API key is not among them).
    public var jsonValues: [String: Any] {
        Dictionary(uniqueKeysWithValues: Settings.fields.map { ($0.key, $0.value(self)) })
    }
}

/// One settings.json key and the property it sets.
struct SettingsField {
    let key: String
    let kind: String
    let apply: (inout Settings, Any) -> Bool
    let value: (Settings) -> Any

    static func bool(_ key: String, _ path: WritableKeyPath<Settings, Bool>) -> SettingsField {
        SettingsField(key: key, kind: "bool", apply: { settings, raw in
            guard let v = Lenient.bool(raw) else { return false }
            settings[keyPath: path] = v
            return true
        }, value: { $0[keyPath: path] })
    }

    static func int(_ key: String, _ path: WritableKeyPath<Settings, Int>) -> SettingsField {
        SettingsField(key: key, kind: "int", apply: { settings, raw in
            guard let v = Lenient.int(raw) else { return false }
            settings[keyPath: path] = v
            return true
        }, value: { $0[keyPath: path] })
    }

    static func double(_ key: String, _ path: WritableKeyPath<Settings, Double>) -> SettingsField {
        SettingsField(key: key, kind: "float", apply: { settings, raw in
            guard let v = Lenient.double(raw) else { return false }
            settings[keyPath: path] = v
            return true
        }, value: { $0[keyPath: path] })
    }

    static func string(_ key: String, _ path: WritableKeyPath<Settings, String>) -> SettingsField {
        SettingsField(key: key, kind: "str", apply: { settings, raw in
            guard let v = Lenient.string(raw) else { return false }
            settings[keyPath: path] = v
            return true
        }, value: { $0[keyPath: path] })
    }

    static func strings(_ key: String, _ path: WritableKeyPath<Settings, [String]>) -> SettingsField {
        SettingsField(key: key, kind: "list", apply: { settings, raw in
            guard let v = Lenient.strings(raw) else { return false }
            settings[keyPath: path] = v
            return true
        }, value: { $0[keyPath: path] })
    }
}

extension Settings {
    nonisolated(unsafe) static let fields: [SettingsField] = [
        .strings("enabled_mics", \.enabledMics),
        .double("preroll_seconds", \.prerollSeconds),
        .double("silence_threshold", \.silenceThreshold),
        .int("sample_rate", \.sampleRate),
        .bool("keep_mic_open", \.keepMicOpen),
        .bool("chunk_on_silence", \.chunkOnSilence),
        .bool("adaptive_threshold_enabled", \.adaptiveThresholdEnabled),
        .double("speech_headroom_db", \.speechHeadroomDb),
        .double("speech_hysteresis_db", \.speechHysteresisDb),
        .int("noise_floor_percentile", \.noiseFloorPercentile),
        .string("trigger_key", \.triggerKey),
        .double("double_tap_threshold", \.doubleTapThreshold),
        .int("toggle_mode_timeout", \.toggleModeTimeout),
        .strings("enabled_providers", \.enabledProviders),
        .int("consensus_threshold", \.consensusThreshold),
        .bool("judge_enabled", \.judgeEnabled),
        .string("judge_model", \.judgeModel),
        .int("judge_timeout_ms", \.judgeTimeoutMs),
        .double("provider_deadline_multiplier", \.providerDeadlineMultiplier),
        .int("provider_deadline_min_ms", \.providerDeadlineMinMs),
        .strings("openrouter_stt_models", \.openrouterSttModels),
        .string("openrouter_correction_model", \.openrouterCorrectionModel),
        .strings("openrouter_correction_provider_order", \.openrouterCorrectionProviderOrder),
        .bool("openrouter_correction_allow_fallbacks", \.openrouterCorrectionAllowFallbacks),
        .string("openrouter_correction_reasoning_effort", \.openrouterCorrectionReasoningEffort),
        .string("custom_instructions", \.customInstructions),
        .string("system_prompt", \.systemPrompt),
        .string("editing_prompt", \.editingPrompt),
        .bool("training_enabled", \.trainingEnabled),
        .bool("edit_feedback_enabled", \.editFeedbackEnabled),
        .bool("learn_vocabulary", \.learnVocabulary),
        .bool("hud_enabled", \.hudEnabled),
        .bool("live_words", \.liveWords),
        .bool("duck_while_recording", \.duckWhileRecording),
        .bool("space_between_dictations", \.spaceBetweenDictations),
        .bool("insert_via_accessibility", \.insertViaAccessibility),
    ]
}

/// Reading hand-edited JSON values as the type a setting has.
///
/// A plain cast turns a hand-edited file into nonsense: "false" is not true, and a string
/// where a list belongs must not become a list of its letters. Anything that can't sensibly
/// be the type is refused, and the setting keeps its previous value.
public enum Lenient {
    static func isBool(_ raw: Any) -> Bool {
        guard let number = raw as? NSNumber else { return false }
        return CFGetTypeID(number) == CFBooleanGetTypeID()
    }

    public static func bool(_ raw: Any) -> Bool? {
        if let number = raw as? NSNumber { return number.doubleValue != 0 }
        if let text = raw as? String {
            switch text.trimmingCharacters(in: .whitespaces).lowercased() {
            case "true", "yes", "on", "1": return true
            case "false", "no", "off", "0": return false
            default: return nil
            }
        }
        return nil
    }

    public static func int(_ raw: Any) -> Int? {
        if let number = raw as? NSNumber {
            let value = number.doubleValue
            guard value.isFinite, abs(value) < 9e15 else { return nil }
            return Int(value)       // truncates toward zero, as Python's int() does
        }
        if let text = raw as? String {
            return Int(text.trimmingCharacters(in: .whitespaces))
        }
        return nil
    }

    public static func double(_ raw: Any) -> Double? {
        if let number = raw as? NSNumber { return number.doubleValue }
        if let text = raw as? String {
            return Double(text.trimmingCharacters(in: .whitespaces))
        }
        return nil
    }

    public static func string(_ raw: Any) -> String? {
        if let text = raw as? String { return text }
        if isBool(raw) { return nil }
        if let number = raw as? NSNumber { return number.stringValue }
        return nil
    }

    /// A list of strings; anything in it that is not a string is dropped.
    public static func strings(_ raw: Any) -> [String]? {
        guard let items = raw as? [Any] else { return nil }
        return items.compactMap { $0 as? String }
    }
}

// MARK: - Writing the files

/// settings.json as a dictionary: the layer the Settings window writes through.
///
/// Saves merge into what is there rather than replace it, so one section can't drop
/// settings another owns, and every write replaces the file in one step.
public struct SettingsFile: Sendable {
    public let url: URL

    public init(url: URL = AppPaths().settingsFile) { self.url = url }

    /// The file's contents, or [:] when it is missing or unreadable.
    public func load() -> [String: Any] {
        guard let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return [:]
        }
        return object
    }

    public enum SaveError: Error, Equatable {
        /// Merging into nothing would write back only the new keys and lose every other
        /// setting, so the file is left for a person to look at.
        case unreadable(String)
    }

    /// Merge `values` into the file.
    public func save(_ values: [String: Any]) throws {
        var existing: [String: Any] = [:]
        if FileManager.default.fileExists(atPath: url.path) {
            do {
                let data = try Data(contentsOf: url)
                guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    throw SaveError.unreadable("not a JSON object")
                }
                existing = object
            } catch let error as SaveError {
                throw error
            } catch {
                throw SaveError.unreadable(error.localizedDescription)
            }
        }
        existing.merge(values) { _, new in new }
        try write(existing)
    }

    /// Delete keys. Saving merges, so without this a prompt reset to its default would keep
    /// the old override on disk forever.
    public func remove(_ keys: [String]) throws {
        guard let data = try? Data(contentsOf: url),
              var existing = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }
        let before = existing.count
        for key in keys { existing.removeValue(forKey: key) }
        if existing.count != before { try write(existing) }
    }

    private func write(_ values: [String: Any]) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: values,
                                              options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        try AtomicFile.write(data, to: url)
    }
}

/// ~/.mergescribe/.env: KEY=value lines. Only the keys MergeScribe owns are touched.
public enum EnvFile {
    public static func read(_ url: URL) -> [String: String] {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [:] }
        var values: [String: String] = [:]
        for raw in text.split(whereSeparator: \.isNewline) {
            let line = raw.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty, !line.hasPrefix("#"), let eq = line.firstIndex(of: "=") else { continue }
            let key = line[..<eq].trimmingCharacters(in: .whitespaces)
            let value = line[line.index(after: eq)...].trimmingCharacters(in: .whitespaces)
                .trimmingCharacters(in: CharacterSet(charactersIn: "'\""))
            values[key] = value
        }
        return values
    }

    /// Set these keys, keeping every other line. An empty value removes the key.
    public static func write(_ keys: [String: String], to url: URL) throws {
        var kept: [String] = []
        if let text = try? String(contentsOf: url, encoding: .utf8) {
            for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
                let key = line.contains("=")
                    ? line.split(separator: "=", maxSplits: 1)[0].trimmingCharacters(in: .whitespaces) : ""
                if keys[key] == nil { kept.append(String(line)) }
            }
            while kept.last?.isEmpty == true { kept.removeLast() }
        }
        for key in keys.keys.sorted() where !(keys[key] ?? "").isEmpty {
            kept.append("\(key)=\(keys[key]!)")
        }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        // A secret: readable by this user only.
        try AtomicFile.write(Data((kept.joined(separator: "\n") + "\n").utf8), to: url, permissions: 0o600)
    }
}

/// Replace a file in one step, so a crash mid-write can't leave half of it.
public enum AtomicFile {
    public static func write(_ data: Data, to url: URL, permissions: Int? = nil) throws {
        let temp = url.deletingLastPathComponent()
            .appendingPathComponent(".\(url.lastPathComponent).\(UUID().uuidString).tmp")
        do {
            try data.write(to: temp)
            if let permissions {
                try FileManager.default.setAttributes([.posixPermissions: permissions], ofItemAtPath: temp.path)
            }
            // rename(2) replaces the destination atomically, whether or not it exists.
            guard rename(temp.path, url.path) == 0 else {
                throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: url.path,
                                                               NSUnderlyingErrorKey: POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)])
            }
        } catch {
            try? FileManager.default.removeItem(at: temp)
            throw error
        }
    }
}
