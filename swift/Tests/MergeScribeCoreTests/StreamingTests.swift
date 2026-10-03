import Foundation
import Testing
@testable import MergeScribeCore

@Suite struct StreamingTests {
    private func turn(_ order: Int, _ text: String, endOfTurn: Bool, formatted: Bool) -> JSON {
        ["type": "Turn", "turn_order": .number(Double(order)), "transcript": .string(text),
         "end_of_turn": .bool(endOfTurn), "turn_is_formatted": .bool(formatted)]
    }

    @Test func aTurnIsFinalOnlyOnceFormatted() {
        var turns = StreamTurns()
        turns.update(turn(0, "ship it friday", endOfTurn: false, formatted: false))
        #expect(!turns.isComplete)
        turns.update(turn(0, "ship it friday", endOfTurn: true, formatted: false))
        #expect(!turns.isComplete, "the formatted version is still to come")
        turns.update(turn(0, "Ship it Friday.", endOfTurn: true, formatted: true))
        #expect(turns.isComplete && turns.text == "Ship it Friday.")
    }

    @Test func turnsJoinInOrderAndTheEmptyNextTurnDoesNotHoldUpTheEnd() {
        var turns = StreamTurns()
        turns.update(turn(1, "Then the rest.", endOfTurn: true, formatted: true))
        turns.update(turn(0, "First part.", endOfTurn: true, formatted: true))
        turns.update(turn(2, "", endOfTurn: false, formatted: false))   // opened when turn 1 ended
        #expect(turns.isComplete)
        #expect(turns.text == "First part. Then the rest.")
        turns.update(turn(2, "and one more", endOfTurn: false, formatted: false))
        #expect(!turns.isComplete, "a turn with words isn't final yet")
    }

    @Test func theReleaseIsAnsweredBeforeTheTranscriptCounts() {
        func settled(_ turns: StreamTurns, final: Bool = false, grace: Bool = false, endpoint: Bool = true) -> Bool {
            AssemblyAIStream.settled(failed: false, endpointSent: endpoint, turns: turns, finalSinceEndpoint: final, graceOver: grace)
        }
        var none = StreamTurns()
        // A one-second dictation released before any of it came back: wait for the final.
        #expect(!settled(none, grace: true), "settling here typed nothing for a real dictation")
        none.update(turn(0, "", endOfTurn: false, formatted: false))
        #expect(!settled(none, grace: true))

        var ended = StreamTurns()
        ended.update(turn(0, "Ship it.", endOfTurn: true, formatted: true))
        #expect(!settled(ended, endpoint: false), "not before the release")
        #expect(!settled(ended), "a turn already over gets a moment for another")
        #expect(settled(ended, grace: true))
        #expect(settled(ended, final: true), "the reply to the release settles at once")

        var more = ended
        more.update(turn(1, "and then", endOfTurn: false, formatted: false))
        #expect(!settled(more, grace: true), "speech still coming in")
        #expect(AssemblyAIStream.settled(failed: true, endpointSent: false, turns: none, finalSinceEndpoint: false, graceOver: false))
    }

    @Test func framesAreAtLeastFiftyMilliseconds() {
        #expect(AssemblyAIStream.frame([Float](repeating: 0.1, count: 300), sampleRate: 16000).count == 800 * 2)
        #expect(AssemblyAIStream.frame([Float](repeating: 0.1, count: 1024), sampleRate: 16000).count == 1024 * 2)
    }

    @Test func audioIsSixteenBitLittleEndianClipped() {
        let data = AssemblyAIStream.pcm16([0, 1, -1, 2, 0.5])
        let values = stride(from: 0, to: data.count, by: 2).map { Int16(littleEndian: data[$0..<$0 + 2].withUnsafeBytes { $0.loadUnaligned(as: Int16.self) }) }
        #expect(values == [0, 32767, -32767, 32767, 16383])
    }

    @Test func aStreamedDictationIsNotCutIntoChunks() {
        var settings = Settings()
        #expect(Chunker.Config(settings).chunkOnSilence)
        settings.streamingEnabled = true
        #expect(Chunker.Config(settings).chunkOnSilence, "no key, no stream: chunks as before")
        settings.assemblyaiApiKey = "k"
        #expect(!Chunker.Config(settings).chunkOnSilence)
    }

    @Test func theKeyComesFromTheEnvFileAndStreamingFromSettings() {
        let home = TempHome()
        home.write(".env", "OPENROUTER_API_KEY=or\nASSEMBLYAI_API_KEY=aai\n")
        home.write("settings.json", #"{"assemblyai_streaming": true}"#)
        let settings = Settings.load(paths: home.paths, environment: [:], currentDirectory: nil, log: { _ in })
        #expect(settings.assemblyaiApiKey == "aai" && settings.openrouterApiKey == "or" && settings.streaming)
    }

    @Test func keytermsAreCappedAtTheAPIsLimit() {
        let stream = AssemblyAIStream(key: "k", model: "universal-3-6-pro", sampleRate: 16000,
                                      keyterms: (0..<150).map { "Term\($0)" })
        #expect(stream.name == "assemblyai-universal-3-6-pro")
        let url = Mirror(reflecting: stream).children.first { $0.label == "request" }.flatMap { ($0.value as? URLRequest)?.url }
        let terms = URLComponents(url: url!, resolvingAgainstBaseURL: false)?.queryItems?.first { $0.name == "keyterms_prompt" }?.value
        let decoded = (try? JSONSerialization.jsonObject(with: Data((terms ?? "").utf8))) as? [String]
        #expect(decoded?.count == 100)
    }
}
