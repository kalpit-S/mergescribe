import AppKit

/// One MergeScribe at a time: two would both hear the trigger key and type every dictation
/// twice. Opening it again, from Spotlight or a terminal, brings up Settings in the one that is
/// running instead, which keeps Settings reachable when the menu bar is too full to show the icon.
enum SingleInstance {
    static let showSettings = Notification.Name("io.github.kalpit-s.mergescribe.showSettings")

    /// Another copy already running: the same bundle, or the same executable outside one.
    static func other() -> NSRunningApplication? {
        let me = ProcessInfo.processInfo.processIdentifier
        let bundle = Bundle.main.bundleIdentifier
        let executable = Bundle.main.executableURL?.resolvingSymlinksInPath()
        return NSWorkspace.shared.runningApplications.first { app in
            guard app.processIdentifier != me else { return false }
            if let bundle, app.bundleIdentifier == bundle { return true }
            return executable != nil && app.executableURL?.resolvingSymlinksInPath() == executable
        }
    }

    static func askToShowSettings() {
        DistributedNotificationCenter.default().postNotificationName(showSettings, object: nil, userInfo: nil,
                                                                     deliverImmediately: true)
    }
}
