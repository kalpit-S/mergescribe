import Testing
@testable import MergeScribeCore

@Test func versionIsSet() {
    #expect(!MergeScribe.version.isEmpty)
}
