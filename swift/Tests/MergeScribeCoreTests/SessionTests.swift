import Foundation
import Testing
@testable import MergeScribeCore

@Suite struct AggregationTests {
    @Test func consensusFirstThenTheLongestTranscript() {
        let s = Rig().session()
        s.state.update {
            $0.chunkResults = [
                (1, [result("Hello world", "p1"), result("Hello world", "p2")], "Hello world"),
                (2, [result("How are you", "p1"), result("How you are", "p2"), result("How are you doing", "p3")], nil),
            ]
        }
        let (texts, all) = s.aggregate()
        #expect(texts == ["Hello world", "How are you doing"])
        #expect(all.count == 5)
    }

    @Test func anEmptyChunkIsIgnored() {
        let s = Rig().session()
        s.chunkReady(AudioChunk([("mic1", [])]))
        #expect(s.state.read(\.chunksSubmitted) == 0)
    }
}

@Suite struct SessionManagerTests {
    private func manager() -> (SessionManager, Rig) {
        let rig = Rig()
        return (SessionManager(services: rig.services), rig)
    }

    @Test func startsAndTracksTheActiveSession() {
        let (m, _) = manager()
        let s = m.startSession(settings: Settings(), recognizers: [])
        #expect(s != nil && m.activeSession === s)
    }

    @Test func aSecondRecordingIsRejectedWhileTheFirstHoldsTheMic() {
        let (m, rig) = manager()
        let first = m.startSession(settings: Settings(), recognizers: [])!
        first.start(mics: ["mic1"])
        #expect(m.isBusy)
        #expect(m.startSession(settings: Settings(), recognizers: []) == nil)
        #expect(rig.output.busy.count == 1)
    }

    @Test func aFinalizingSessionDoesNotBlockANewRecording() {
        // The mic is free at key release; the ~2s correction and typing tail must not deafen the app.
        let (m, rig) = manager()
        let first = m.startSession(settings: Settings(), recognizers: [])!
        first.start(mics: ["mic1"])
        first.state.update { $0.isCapturing = false }
        #expect(!m.isBusy)
        let second = m.startSession(settings: Settings(), recognizers: [])
        #expect(second != nil && second !== first)
        #expect(rig.output.busy.count == 0)
    }

    @Test func completionClearsTheActiveSessionAndSaysWhichWasNewest() async {
        let (m, _) = manager()
        let newest = Recorder<Bool>()
        m.onSessionComplete = { _, isNewest in newest.add(isNewest) }
        let first = m.startSession(settings: Settings(), recognizers: [])!
        first.finalize(FinalChunk(chunk: AudioChunk(), hasSpeech: false))
        #expect(await eventually { m.activeSession == nil })
        #expect(newest.all == [true])
    }
}

@Suite struct WaitingForRecognizerTests {
    private func run(_ recognizers: [Recognizer], multiplier: Double, minMs: Int, threshold: Int = 99,
                     observer: Bool = false) async -> (Session, Rig, Double) {
        var rig = Rig()
        rig.settings.consensusThreshold = threshold
        rig.settings.providerDeadlineMultiplier = multiplier
        rig.settings.providerDeadlineMinMs = minMs
        rig.recognizers = recognizers
        let s = rig.session()
        let started = Date()
        await s.transcribe(speech(0.1))
        return (s, rig, Date().timeIntervalSince(started))
    }

    private func providers(_ s: Session) -> [String] {
        s.state.read { $0.chunkResults[0].results.map(\.provider) }.sorted()
    }

    @Test func theDeadlineAbandonsAStraggler() async {
        // The chunk should not pay for the slowest recognizer's tail.
        let (s, rig, elapsed) = await run([FakeRecognizer("fast", text: "fast"), FakeRecognizer("slow", delay: 3, text: "slow")],
                                          multiplier: 1, minMs: 120)
        #expect(providers(s) == ["fast"])
        #expect(elapsed < 1.0, "waited \(elapsed)s for an abandoned recognizer")
        #expect(rig.observer.events.all.contains("dropped slow/mic1"))
        #expect(rig.metrics.named("provider_deadline").first?["abandoned"] == ["slow/mic1"])
    }

    @Test func multiplierZeroWaitsForEveryone() async {
        let (s, _, _) = await run([FakeRecognizer("fast", text: "x"), FakeRecognizer("slow", delay: 0.35, text: "y")],
                                  multiplier: 0, minMs: 120)
        #expect(providers(s) == ["fast", "slow"])
    }

    @Test func theMinimumGraceProtectsANormallySlowRecognizer() async {
        let (s, _, _) = await run([FakeRecognizer("fast", text: "x"), FakeRecognizer("slow", delay: 0.3, text: "y")],
                                  multiplier: 1, minMs: 1500)
        #expect(providers(s) == ["fast", "slow"])
    }

    @Test func anEmptyAnswerDoesNotStartTheClock() async {
        // A recognizer that heard nothing is no evidence the rest are late.
        let (s, _, _) = await run([FakeRecognizer("blank"), FakeRecognizer("slow", delay: 0.6, text: "ship it friday")],
                                  multiplier: 1, minMs: 300)
        #expect(s.state.read { $0.chunkResults[0].results.map(\.text) }.contains("ship it friday"))
    }

    @Test func theHardLimitHoldsAfterAFirstAnswer() async {
        var rig = Rig()
        rig.settings.consensusThreshold = 99
        rig.settings.providerDeadlineMultiplier = 10
        rig.settings.providerDeadlineMinMs = 0
        rig.recognizers = [FakeRecognizer("fast", delay: 0.2, text: "ship it"), FakeRecognizer("slow", delay: 1.5, text: "ship it friday")]
        let s = rig.session()
        s.chunkHardTimeout = 0.5
        let started = Date()
        await s.transcribe(speech(0.1))
        #expect(Date().timeIntervalSince(started) < 1.0)
    }

    @Test func consensusStopsWaitingAndIsRecorded() async {
        let (s, rig, elapsed) = await run([FakeRecognizer("p1", text: "Hello world"), FakeRecognizer("p2", text: "Hello world."),
                                           FakeRecognizer("slow", delay: 2, text: "x")], multiplier: 0, minMs: 0, threshold: 2)
        #expect(s.state.read { $0.chunkResults[0].consensus } == "Hello world")
        #expect(elapsed < 1.0)
        #expect(!rig.observer.events.all.contains("dropped slow/mic1"), "the others agreed for it")
        #expect(rig.metrics.named("consensus").first?["matching_count"] == 2)
    }

    @Test func noConsensusKeepsEveryResult() async {
        let (s, _, _) = await run([FakeRecognizer("p1", text: "Hello world"), FakeRecognizer("p2", text: "Hi there")],
                                  multiplier: 1, minMs: 900, threshold: 2)
        #expect(s.state.read { $0.chunkResults[0].consensus } == nil)
        #expect(providers(s).count == 2)
    }

    @Test func aFailingRecognizerIsDroppedNotFatal() async {
        let (s, rig, _) = await run([FakeRecognizer("bad", fails: true), FakeRecognizer("good", text: "ok")], multiplier: 1, minMs: 100)
        #expect(providers(s) == ["good"])
        #expect(rig.observer.events.all.contains("dropped bad/mic1"))
    }

    @Test func aSerializedRecognizerRunsOnThePrimaryMicOnly() async {
        var rig = Rig()
        rig.settings.enabledMics = ["mic_a", "mic_b"]
        let calls = Recorder<(String, Int)>()
        rig.recognizers = [FakeRecognizer("local", text: "hi", singleInstance: true, calls: calls),
                           FakeRecognizer("cloud", text: "hi", calls: calls)]
        let s = rig.session()
        await s.transcribe(AudioChunk([("mic_a", [0.1, 0.2]), ("mic_b", [0.1, 0.2])]))
        #expect(calls.count == 3)
        let plan = s.streamPlan(["mic_a", "mic_b"]).map { "\($0.recognizer.name)/\($0.mic)" }.sorted()
        #expect(plan == ["cloud/mic_a", "cloud/mic_b", "local/mic_a"])
    }

    @Test func thePrimaryMicFollowsTheEnabledOrder() {
        var rig = Rig()
        rig.settings.enabledMics = ["mic_b", "mic_a"]
        rig.recognizers = [FakeRecognizer("local", singleInstance: true), FakeRecognizer("cloud")]
        let s = rig.session()
        let plan = { (mics: [String]) in s.streamPlan(mics).map { "\($0.recognizer.name)/\($0.mic)" }.sorted() }
        #expect(plan(["mic_a", "mic_b"]) == ["cloud/mic_a", "cloud/mic_b", "local/mic_b"])
        // An unplugged primary falls through to whatever produced audio.
        #expect(plan(["mic_a"]) == ["cloud/mic_a", "local/mic_a"])
    }

    @Test func startAnnouncesTheStreamsThatWillRun() {
        var rig = Rig()
        rig.recognizers = [FakeRecognizer("fast"), FakeRecognizer("slow")]
        rig.session().start(mics: ["mic1"])
        #expect(rig.observer.events.all.first == "planned fast/mic1,slow/mic1")
    }

    @Test func theFinalChunkIsMarkedAndFollowedByDone() async {
        var rig = Rig()
        rig.settings.providerDeadlineMinMs = 1500
        rig.recognizers = [FakeRecognizer("fast", text: "a"), FakeRecognizer("slow", delay: 0.05, text: "b")]
        let s = rig.session()
        await s.transcribeRemaining(speech(0.1))
        let events = rig.observer.events.all
        #expect(events.contains("landed fast/mic1 final") && events.contains("landed slow/mic1 final"))
        #expect(events.last == "done")
    }
}

@Suite struct PartsInOrderTests {
    @Test func anEarlierPartThatFinishesLastStillComesFirst() async {
        // The 30s part is still transcribing when the short tail comes back.
        var rig = Rig()
        rig.settings.consensusThreshold = 99
        rig.settings.providerDeadlineMinMs = 2000
        rig.recognizers = [FakeRecognizer("p", delay: { $0[0] == 1 ? 0.3 : 0 }, text: { "part \(Int($0[0]))" })]
        let s = rig.session()
        s.chunkReady(AudioChunk([("mic1", Array(repeating: 1, count: 1600))]))
        await s.transcribeRemaining(AudioChunk([("mic1", Array(repeating: 2, count: 1600))]))
        let (texts, results) = s.aggregate()
        #expect(texts == ["part 1", "part 2"])
        #expect(results.map(\.text) == ["part 1", "part 2"])
    }
}

@Suite struct FinalizeTests {
    private func finalize(_ s: Session, _ final: FinalChunk) async {
        await s.finalizeInBackground(final)
    }

    @Test func aStrayTapIsDiscarded() async {
        var rig = Rig()
        let calls = Recorder<(String, Int)>()
        rig.recognizers = [FakeRecognizer("p", text: "yeah", calls: calls)]
        let s = rig.session()
        let completed = Recorder<Bool>()
        s.onComplete = { _ in completed.add(true) }
        await finalize(s, FinalChunk(chunk: speech(0.2, value: 0), hasSpeech: false))
        #expect(rig.metrics.named("session_discarded").first?["reason"] == "no_speech")
        #expect(calls.count == 0 && rig.corrector.calls.count == 0)
        #expect(completed.count == 1, "on_complete must fire or the manager stays wedged")
    }

    @Test func speechStillBeingTranscribedSurvivesAQuietRelease() async {
        // The log sequence behind this: a 30s chunk emitted, release 0.4s later on silence.
        var rig = Rig()
        let calls = Recorder<(String, Int)>()
        rig.recognizers = [FakeRecognizer("p", delay: 0.3, text: "ship the fix today", calls: calls)]
        rig.judge = FakeJudge(nil)
        rig.corrector.tokens = ["Ship the fix today."]
        let s = rig.session()
        s.chunkReady(speech())
        await finalize(s, FinalChunk(chunk: speech(0.4, value: 0), hasSpeech: false))
        #expect(rig.metrics.named("session_discarded").isEmpty, "in-flight speech was thrown away")
        #expect(calls.count == 1, "the silent tail is not sent to the recognizers")
        #expect(rig.corrector.calls.all.first?.first?.text == "ship the fix today")
        #expect(rig.output.typed == "Ship the fix today.")
    }

    @Test func agreedTextStillGoesThroughCorrection() async {
        // Consensus is not a licence to type: "um, ship it" can be agreed on too.
        var rig = Rig()
        rig.settings.consensusThreshold = 2
        rig.recognizers = [FakeRecognizer("p1", text: "um, ship it"), FakeRecognizer("p2", text: "um ship it")]
        rig.judge = FakeJudge(nil)
        rig.corrector.tokens = ["Ship it."]
        let s = rig.session()
        await finalize(s, FinalChunk(chunk: speech(), hasSpeech: true))
        #expect(rig.corrector.calls.count == 1)
        #expect(rig.output.typed == "Ship it.")
    }

    @Test func aShortCommandWaitsForTheSelectionBeforeDeciding() async {
        // Selection capture (a synthetic copy) can outlast a fast transcription.
        var rig = Rig()
        rig.recognizers = [FakeRecognizer("p", text: "make this more formal")]
        rig.focus.selection = "the selected paragraph"
        rig.focus.captureDelay = 0.3
        rig.corrector.editResult = "The selected paragraph, formally."
        let s = rig.session()
        s.start(mics: ["mic1"])
        await finalize(s, FinalChunk(chunk: speech(), hasSpeech: true))
        #expect(rig.corrector.edits.all.map(\.1) == ["make this more formal"])
        #expect(rig.corrector.edits.all.map(\.0) == ["the selected paragraph"])
        #expect(rig.corrector.calls.count == 0)
        #expect(rig.output.typed == "The selected paragraph, formally.")
    }

    @Test func aFailedCorrectionTypesTheRawTranscript() async {
        var rig = Rig()
        rig.recognizers = [FakeRecognizer("p", text: "ship it friday")]
        rig.judge = FakeJudge(nil)
        rig.corrector.fails = true
        let s = rig.session()
        await finalize(s, FinalChunk(chunk: speech(), hasSpeech: true))
        #expect(rig.output.typed == "ship it friday")
        #expect(s.state.read(\.outputMethod) == "typed")
    }

    @Test func completionIsLoggedWithTheWords() async {
        var rig = Rig()
        rig.recognizers = [FakeRecognizer("p", text: "ship it friday")]
        rig.judge = FakeJudge(nil)
        rig.corrector.tokens = ["Ship it Friday."]
        let s = rig.session()
        await finalize(s, FinalChunk(chunk: speech(), hasSpeech: true))
        let complete = rig.metrics.named("session_complete").first
        #expect(complete?["words"] == 3 && complete?["app"] == "Editor")
        #expect(complete?["session_id"]?.string == s.sid)
    }
}

@Suite struct StreamCorrectionTests {
    private func stream(_ rig: Rig, _ tokens: [String], answer: String? = nil) async -> (Session, String?) {
        rig.corrector.tokens = tokens
        rig.corrector.result = answer
        let s = rig.session()
        let out = await s.streamCorrection([result("x")])
        return (s, out)
    }

    private var noJudge: Rig {
        var rig = Rig()
        rig.judge = FakeJudge(nil)
        return rig
    }

    @Test func typesTokensAndReturnsThem() async {
        let rig = noJudge
        let (s, out) = await stream(rig, ["Hello ", "world", "!"])
        #expect(out == "Hello world!" && rig.output.typed == "Hello world!")
        #expect(s.state.read(\.outputMethod) == "streamed")
    }

    @Test func inventedLineBreaksAreFlattened() async {
        // A typed newline is a Return keystroke, which sends the message in chat apps.
        let rig = noJudge
        let (_, out) = await stream(rig, ["Okay cool.", "\n\n", "Also, one more thing."], answer: "")
        #expect(out == "Okay cool. Also, one more thing.")
        #expect(!rig.output.typedPieces.all.contains { $0.contains("\n") })
    }

    @Test func continuingARecentDictationLeadsWithASpace() async {
        var rig = noJudge
        rig.settings.spaceBetweenDictations = true
        rig.history.add("earlier words", destination: "Editor: t")
        let (_, out) = await stream(rig, ["Ship it."])
        #expect(out == " Ship it.")
    }

    @Test func aFreshDictationHasNoStraySpace() async {
        var rig = noJudge
        rig.settings.spaceBetweenDictations = true
        let (_, out) = await stream(rig, ["Ship it."])
        #expect(out == "Ship it.")
    }

    @Test func staleOrElsewhereIsNotContinued() {
        var rig = noJudge
        rig.settings.spaceBetweenDictations = true
        rig.history.add("long ago", destination: "Editor: t")
        rig.history.backdateLast(by: 600)
        #expect(!rig.session().continuesPrevious())
        rig.history.add("elsewhere", destination: "Some Other App")
        #expect(!rig.session().continuesPrevious())
    }

    @Test func aFailedCorrectionLeavesNoStraySpaceAndReportsEmpty() async {
        var rig = noJudge
        rig.settings.spaceBetweenDictations = true
        rig.history.add("earlier", destination: "Editor: t")
        rig.corrector.fails = true
        let s = rig.session()
        #expect(s.continuesPrevious())
        #expect(await s.streamCorrection([result("x")]) == "")
        #expect(rig.output.typedPieces.count == 0)
    }

    @Test func theMarkerTypesNothingAndDoesNotFallBack() async {
        // Falling back would type the raw transcript, "scratch all that" included.
        var rig = noJudge
        rig.corrector.tokens = ["[no", "thing]"]
        let s = rig.session()
        await s.correctAndOutput([result("raw words")], combined: "raw words")
        #expect(rig.output.typedPieces.count == 0)
        #expect(rig.metrics.named("session_discarded").first?["reason"] == "called_off")
        #expect(rig.observer.events.all.contains("discarded"))
        _ = rig.settings
    }

    @Test func calledOffLeavesNoStraySeparator() async {
        var rig = noJudge
        rig.settings.spaceBetweenDictations = true
        rig.history.add("earlier", destination: "Editor: t")
        let (_, out) = await stream(rig, ["[nothing]"])
        #expect(out == nil && rig.output.typedPieces.count == 0)
    }

    @Test func bracketedTextThatIsNotTheMarkerStillTypes() async {
        let rig = noJudge
        let (_, out) = await stream(rig, ["[WIP] ", "ship it"])
        #expect(out == "[WIP] ship it" && rig.output.typed == "[WIP] ship it")
    }

    @Test func historyRecordsTheRawTranscriptNotTheCorrection() async {
        let rig = noJudge
        rig.corrector.tokens = ["Polished output."]
        await rig.session().correctAndOutput([result("um raw")], combined: "um raw")
        let recent = rig.history.last()?.text ?? ""
        #expect(recent == "um raw")
    }
}

@Suite struct BackToWhereItStartedTests {
    private func rig(here: [Bool], comesBack: Bool) -> Rig {
        var rig = Rig()
        rig.judge = FakeJudge(nil)
        rig.focus.origin = Origin(pid: 1, handle: nil)
        rig.focus.here = here
        rig.focus.comesBack = comesBack
        return rig
    }

    private func session(_ rig: Rig) -> Session {
        let s = rig.session()
        s.state.update { $0.origin = rig.focus.origin }
        return s
    }

    @Test func stillThereItJustTypes() async {
        let rig = rig(here: [true], comesBack: true)
        await session(rig).outputAsync("Ship it Friday.")
        #expect(rig.output.typed == "Ship it Friday." && rig.focus.returned.count == 0 && rig.output.clipboard.count == 0)
    }

    @Test func movedOnItBringsTheWindowBackAndTypesThere() async {
        let rig = rig(here: [false, true], comesBack: true)
        await session(rig).outputAsync("Ship it Friday.")
        #expect(rig.focus.returned.count == 1 && rig.output.typed == "Ship it Friday." && rig.output.clipboard.count == 0)
    }

    @Test func aWindowThatCannotComeBackGetsTheClipboard() async {
        // Closed since: nothing to type into, so nothing is typed anywhere else.
        let rig = rig(here: [false], comesBack: false)
        let s = session(rig)
        await s.outputAsync("Ship it Friday.")
        #expect(rig.output.typedPieces.count == 0 && rig.output.clipboard.all == ["Ship it Friday."])
        #expect(s.state.read(\.outputMethod) == "clipboard")
    }

    @Test func theCorrectionStreamsIntoTheWindowItBroughtBack() async {
        let rig = rig(here: [false, true], comesBack: true)
        rig.corrector.tokens = ["Ship it."]
        await session(rig).correctAndOutput([result("ship it")], combined: "ship it")
        #expect(rig.output.typed == "Ship it." && rig.output.clipboard.count == 0)
    }

    @Test func correctionGoesToTheClipboardWhenTheWindowIsGone() async {
        let rig = rig(here: [false], comesBack: false)
        rig.corrector.tokens = []
        rig.corrector.result = "corrected text"
        let s = session(rig)
        await s.correctAndOutput([result("raw")], combined: "raw")
        #expect(rig.output.clipboard.all == ["corrected text"] && rig.output.typedPieces.count == 0)
        #expect(s.state.read(\.outputMethod) == "clipboard")
    }

    @Test func theClipboardPathHonoursTheMarker() async {
        let rig = rig(here: [false], comesBack: false)
        rig.corrector.result = "[nothing]"
        await session(rig).correctAndOutput([result("raw")], combined: "raw")
        #expect(rig.output.clipboard.count == 0)
        #expect(rig.metrics.named("session_discarded").first?["reason"] == "called_off")
    }

    @Test func withoutAnOriginTheSameAppIsEnough() async {
        var rig = Rig()
        rig.focus.current = AppContext(appName: "Other", windowTitle: "w", bundleId: "com.other")
        let s = rig.session()
        _ = rig.focus.appContext()      // the first read is the start's
        await s.outputAsync("Hello")
        #expect(rig.output.clipboard.all == ["Hello"] && rig.output.notifications.count == 1)
    }

    @Test func movingToAnotherWindowStopsTypingAndHandsOverTheRest() async {
        let rig = rig(here: [true, false], comesBack: true)
        rig.corrector.tokens = ["Ship ", "it ", "Friday."]
        rig.corrector.tokenGap = 0.12        // past the 100ms between focus checks
        let s = session(rig)
        _ = await s.streamCorrection([result("ship it friday")])
        #expect(rig.output.typed == "Ship ")
        #expect(rig.output.clipboard.all == ["it Friday."])
        #expect(s.state.read(\.partialOutput))
    }

    @Test func aCorrectionCutOffAfterTypingPutsTheWholeTextOnTheClipboard() async {
        let rig = rig(here: [true], comesBack: true)
        rig.corrector.tokens = ["Ship it"]
        rig.corrector.interrupted = CorrectionInterrupted(shown: "Ship it", complete: "We ship Friday.")
        let out = await session(rig).streamCorrection([result("ship it friday")])
        #expect(out == "Ship it" && rig.output.typed == "Ship it")
        #expect(rig.output.clipboard.all == ["We ship Friday."])
    }
}

@Suite struct JudgeRaceTests {
    private func race(_ verdict: Judge.Verdict?, correctionDelay: Double = 0, judgeDelay: Double = 0,
                      tokens: [String] = ["corrected text"], timeoutMs: Int = 900) async -> (Session, Rig, String?) {
        var rig = Rig()
        rig.settings.judgeTimeoutMs = timeoutMs
        rig.judge = FakeJudge(verdict, delay: judgeDelay)
        rig.corrector.tokens = tokens
        rig.corrector.delay = correctionDelay
        let s = rig.session()
        let out = await s.streamCorrection([result("raw words")])
        return (s, rig, out)
    }

    @Test func aCleanVerdictTypesTheTranscriptAndDropsTheCorrection() async {
        let (s, rig, out) = await race(FakeJudge.clean("raw words"), correctionDelay: 0.4)
        #expect(out == "raw words" && rig.output.typed == "raw words")
        #expect(s.state.read(\.outputMethod) == "judged")
        #expect(rig.observer.events.all.contains("skipped"))
    }

    @Test func aDirtyVerdictLeavesTheCorrectionAlone() async {
        var dirty = FakeJudge.clean("raw words")
        dirty.clean = false
        let (_, rig, out) = await race(dirty)
        #expect(out == "corrected text" && rig.output.typed == "corrected text")
    }

    @Test func noJudgeMeansTheCorrection() async {
        let (_, rig, out) = await race(nil)
        #expect(out == "corrected text" && rig.output.typed == "corrected text")
    }

    @Test func aLateVerdictNeverTypesOverTheCorrection() async {
        // The correction got there first; two copies must not land.
        let (_, rig, out) = await race(FakeJudge.clean("raw words", latency: 800), judgeDelay: 0.25)
        #expect(out == "corrected text" && rig.output.typed == "corrected text")
    }

    @Test func theVerdictIsLoggedEitherWay() async {
        let (_, rig, _) = await race(FakeJudge.clean("raw words", latency: 210), correctionDelay: 0.3)
        let judged = rig.metrics.named("judge").first
        #expect(judged?["clean"] == true && judged?["latency_ms"] == 210)
    }

    @Test func aHangingJudgeDoesNotHoldUpTyping() async {
        // A judge that never answers must not add its timeout to the dictation.
        let started = Date()
        let (_, rig, _) = await race(FakeJudge.clean("raw words"), judgeDelay: 3, timeoutMs: 5000)
        #expect(rig.output.typed == "corrected text")
        #expect(Date().timeIntervalSince(started) < 1.0, "typing waited on the judge")
    }

    @Test func aMultiChunkDictationNeverReachesTheJudge() async {
        // Otherwise a clean verdict would type one chunk and drop the others.
        var rig = Rig()
        rig.judge = FakeJudge(FakeJudge.clean("chunk 0"))
        rig.corrector.tokens = ["All three chunks."]
        let s = rig.session()
        s.state.update { st in
            for i in 0..<3 { st.chunkResults.append((i + 1, [result("chunk \(i)", chunk: i + 1)], nil)) }
        }
        let out = await s.streamCorrection([result("chunk 0"), result("chunk 1"), result("chunk 2")])
        #expect(rig.judge.asked.count == 0)
        #expect(out == "All three chunks.")
    }
}

@Suite struct EditWatchRowTests {
    private func edited(_ method: String, model: String?) async -> [String: JSON] {
        var rig = Rig()
        rig.settings.editFeedbackEnabled = true
        let home = TempHome()
        rig.corpusURL = home.paths.correctionsFile
        let s = rig.session()
        s.state.update {
            $0.outputMethod = method
            if let model {
                $0.llmResult = LLMCorrectionResult(text: "t", provider: "openrouter", model: model, inputTokensEst: 1, latencyMs: 1)
            }
        }
        s.startEditWatch("ship it friday")
        rig.watcher.callbacks.all[0](.edited, 8, "ship it friday", "ship it Friday")
        let line = home.read("corrections.jsonl").split(separator: "\n").first.map(String.init) ?? "{}"
        return (try? JSONDecoder().decode(JSON.self, from: Data(line.utf8)))?.object ?? [:]
    }

    @Test func aFixedCorrectionRecordsTheModelThatWroteIt() async {
        let row = await edited("streamed", model: "openai/gpt-6-luna")
        #expect(row["output_method"] == "streamed" && row["correction_model"] == "openai/gpt-6-luna")
        #expect(row["corrected"] == "ship it Friday")
    }

    @Test func aFixedRawTranscriptNamesNoCorrectionModel() async {
        // Typed by the judge: whatever was wrong came from a recognizer.
        let row = await edited("judged", model: "openai/gpt-6-luna")
        #expect(row["output_method"] == "judged" && row["correction_model"] == "")
    }
}

@Suite struct OutputOrderTests {
    @Test func aLaterDictationWaitsForTheEarlierOne() async {
        let order = OutputOrder()
        let events = Recorder<String>()
        let first = order.take(), second = order.take()
        let later = Task { await order.wait(second); events.add("second types") }
        try? await Task.sleep(for: .milliseconds(100))
        events.add("first types")
        order.done(first)
        await later.value
        #expect(events.all == ["first types", "second types"])
    }

    @Test func finishingOutOfOrderDoesNotSkipATurn() async {
        let order = OutputOrder(patience: 0.2)
        let a = order.take(), b = order.take(), c = order.take()
        order.done(b)                        // b gave up waiting and finished early
        var started = Date()
        await order.wait(c)                  // a is still going: c must still wait
        #expect(Date().timeIntervalSince(started) >= 0.15)
        order.done(a)
        started = Date()
        await order.wait(c)
        #expect(Date().timeIntervalSince(started) < 0.05)
    }

    @Test func aSessionThatFailsStillGivesUpItsTurn() async {
        let rig = Rig()
        let s = rig.session()
        s.finalize(FinalChunk(chunk: AudioChunk(), hasSpeech: false))
        #expect(await eventually { !s.isCapturing && rig.metrics.named("session_complete").count == 1 })
        let started = Date()
        await rig.order.wait(rig.order.take())
        #expect(Date().timeIntervalSince(started) < 0.5)
    }
}
