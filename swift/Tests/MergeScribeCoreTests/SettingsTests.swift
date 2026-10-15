import Foundation
import Testing
@testable import MergeScribeCore

/// A throwaway ~/.mergescribe for one test.
struct TempHome {
    let root: URL
    let paths: AppPaths

    init() {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("ms-\(UUID().uuidString)")
        paths = AppPaths(dataDir: root.appendingPathComponent(".mergescribe"))
        try? FileManager.default.createDirectory(at: paths.dataDir, withIntermediateDirectories: true)
    }

    func write(_ name: String, _ text: String) {
        try! Data(text.utf8).write(to: paths.dataDir.appendingPathComponent(name))
    }

    func read(_ name: String) -> String {
        (try? String(contentsOf: paths.dataDir.appendingPathComponent(name), encoding: .utf8)) ?? ""
    }

    func json(_ name: String) -> [String: Any] {
        let data = (try? Data(contentsOf: paths.dataDir.appendingPathComponent(name))) ?? Data()
        return (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    }
}

@Suite struct SettingsFileTests {
    @Test func saveMergesRatherThanReplacing() throws {
        // Saving one section must not drop settings owned by another.
        let home = TempHome()
        home.write("settings.json", #"{"keep_me": 1, "trigger_key": "alt_r"}"#)
        let file = SettingsFile(url: home.paths.settingsFile)
        try file.save(["trigger_key": "cmd_r"])
        let result = file.load()
        #expect(result["keep_me"] as? Int == 1, "an untouched setting was dropped")
        #expect(result["trigger_key"] as? String == "cmd_r")
    }

    @Test func loadSurvivesACorruptFile() {
        let home = TempHome()
        home.write("settings.json", "{not json")
        #expect(SettingsFile(url: home.paths.settingsFile).load().isEmpty)
    }

    @Test func anUnreadableFileIsNotReplacedByAFragment() {
        // A save that can't read the file must not write back only its own keys.
        let home = TempHome()
        let cut = #"{"trigger_key": "cmd_r", "enabled_mics": ["Mic""#     // cut off mid-write
        home.write("settings.json", cut)
        #expect(throws: SettingsFile.SaveError.self) {
            try SettingsFile(url: home.paths.settingsFile).save(["hud_enabled": false])
        }
        #expect(home.read("settings.json") == cut)                        // left for a human to see
        let leftovers = try? FileManager.default.contentsOfDirectory(atPath: home.paths.dataDir.path)
            .filter { $0.hasSuffix(".tmp") }
        #expect(leftovers?.isEmpty == true)
    }

    @Test func savesAreWholeFiles() throws {
        let home = TempHome()
        let file = SettingsFile(url: home.paths.settingsFile)
        try file.save(["a": 1])
        try file.save(["b": 2])
        #expect(file.load() as NSDictionary == ["a": 1, "b": 2] as NSDictionary)
    }

    @Test func removesOnlyTheNamedKeys() throws {
        // Resetting a prompt must actually delete the override, not leave it on disk.
        let home = TempHome()
        home.write("settings.json", #"{"system_prompt": "old override", "trigger_key": "cmd_r"}"#)
        let file = SettingsFile(url: home.paths.settingsFile)
        try file.remove(["system_prompt"])
        let result = file.load()
        #expect(result["system_prompt"] == nil)
        #expect(result["trigger_key"] as? String == "cmd_r")
    }

    @Test func removingFromAMissingFileOrKeyIsHarmless() throws {
        let home = TempHome()
        let file = SettingsFile(url: home.paths.settingsFile)
        try file.remove(["anything"])              // no file at all
        home.write("settings.json", #"{"a": 1}"#)
        try file.remove(["not_there"])
        #expect(file.load() as NSDictionary == ["a": 1] as NSDictionary)
    }
}

@Suite struct EnvFileTests {
    @Test func writingAKeyPreservesUnrelatedLines() throws {
        // The .env may hold other people's variables; we only own our key.
        let home = TempHome()
        home.write(".env", "SOMETHING_ELSE=keepme\nOPENROUTER_API_KEY=old\n")
        try EnvFile.write(["OPENROUTER_API_KEY": "new"], to: home.paths.envFile)
        let written = home.read(".env")
        #expect(written.contains("SOMETHING_ELSE=keepme"))
        #expect(written.contains("OPENROUTER_API_KEY=new"))
        #expect(!written.contains("old"))
    }

    @Test func anEmptyKeyIsNotWritten() throws {
        let home = TempHome()
        home.write(".env", "OPENROUTER_API_KEY=existing\n")
        try EnvFile.write(["OPENROUTER_API_KEY": ""], to: home.paths.envFile)
        #expect(!home.read(".env").contains("OPENROUTER_API_KEY"))
    }

    @Test func theKeyFileIsPrivate() throws {
        let home = TempHome()
        try EnvFile.write(["OPENROUTER_API_KEY": "sk-or-test"], to: home.paths.envFile)
        let attributes = try FileManager.default.attributesOfItem(atPath: home.paths.envFile.path)
        #expect((attributes[.posixPermissions] as? Int) == 0o600)
    }

    @Test func quotesCommentsAndBlankLinesAreHandled() {
        let home = TempHome()
        home.write(".env", "# comment\n\n OPENROUTER_API_KEY = 'sk-or-quoted' \nNOEQUALS\n")
        #expect(EnvFile.read(home.paths.envFile)["OPENROUTER_API_KEY"] == "sk-or-quoted")
    }
}

@Suite struct SettingsLoadTests {
    @Test func defaultsWithoutAnyFiles() {
        let home = TempHome()
        let settings = Settings.load(paths: home.paths, environment: [:], currentDirectory: nil)
        #expect(settings == Settings())
        #expect(settings.openrouterCorrectionModel == "openai/gpt-6-luna")
        #expect(settings.judgeEnabled == false)
        #expect(settings.settingsShortcut == "", "off until the person picks one: a good choice depends on what else is bound")
    }

    @Test func theSettingsShortcutIsJustAnotherStringField() {
        let home = TempHome()
        home.write("settings.json", #"{"settings_shortcut": "ctrl_l"}"#)
        #expect(Settings.load(paths: home.paths, environment: [:], currentDirectory: nil).settingsShortcut == "ctrl_l")
    }

    @Test func handEditedValuesDoNotTurnIntoNonsense() {
        // "false" is not true, and a string where a list belongs is not a list of letters.
        let home = TempHome()
        home.write("settings.json", #"{"hud_enabled": "false", "training_enabled": "true", "enabled_mics": "MacBook Pro Microphone"}"#)
        var ignored: [String] = []
        let settings = Settings.load(paths: home.paths, environment: [:], currentDirectory: nil,
                                     log: { ignored.append($0) })
        #expect(settings.hudEnabled == false)
        #expect(settings.trainingEnabled == true)
        #expect(settings.enabledMics == [])
        #expect(ignored.count == 1 && ignored[0].contains("enabled_mics"))
    }

    @Test func numbersAreReadTheWayPythonReadThem() {
        let home = TempHome()
        home.write("settings.json", #"{"judge_timeout_ms": 750.9, "silence_threshold": "1.5", "consensus_threshold": true, "double_tap_threshold": 1, "judge_model": 5, "sample_rate": "16k"}"#)
        let settings = Settings.load(paths: home.paths, environment: [:], currentDirectory: nil, log: { _ in })
        #expect(settings.judgeTimeoutMs == 750)       // int() truncates
        #expect(settings.silenceThreshold == 1.5)
        #expect(settings.consensusThreshold == 1)
        #expect(settings.doubleTapThreshold == 1.0)
        #expect(settings.judgeModel == "5")
        #expect(settings.sampleRate == 16000)         // unreadable: the default stands
    }

    @Test func theKeyComesFromTheEnvFileAndTheEnvironmentWins() {
        let home = TempHome()
        home.write(".env", "OPENROUTER_API_KEY=from-file\n")
        #expect(Settings.load(paths: home.paths, environment: [:], currentDirectory: nil).openrouterApiKey == "from-file")
        #expect(Settings.load(paths: home.paths, environment: ["OPENROUTER_API_KEY": "from-env"],
                              currentDirectory: nil).openrouterApiKey == "from-env")
    }

    @Test func everyFieldRoundTripsThroughItsKey() {
        var changed = Settings()
        changed.enabledMics = ["A", "B"]
        changed.prerollSeconds = 0.5
        changed.triggerKey = "cmd_r"
        changed.toggleModeTimeout = 30
        changed.judgeEnabled = true
        changed.openrouterCorrectionProviderOrder = ["openai/fast"]
        changed.customInstructions = "lowercase in Slack"
        changed.duckWhileRecording = false
        var copy = Settings()
        copy.apply(changed.jsonValues)
        #expect(copy == changed)
        #expect(Set(changed.jsonValues.keys).count == Settings.fields.count, "two fields share a key")
    }
}
