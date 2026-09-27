import Foundation
import MergeScribeCore

/// Console output mirrored to ~/.mergescribe/logs/mergescribe.log, so a session can be
/// inspected later. Starts fresh past 5 MB, keeping the previous file as .log.old.
enum LogFile {
    static let maxBytes = 5_000_000

    static func start(_ url: URL = AppPaths().logFile) {
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if let size = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int, size > maxBytes {
            let old = url.deletingPathExtension().appendingPathExtension("log.old")
            try? FileManager.default.removeItem(at: old)
            try? FileManager.default.moveItem(at: url, to: old)
        }
        if !FileManager.default.fileExists(atPath: url.path) { FileManager.default.createFile(atPath: url.path, contents: nil) }
        let handle = try? FileHandle(forWritingTo: url)
        _ = try? handle?.seekToEnd()
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        let queue = DispatchQueue(label: "mergescribe.log")
        let file = Box(handle)
        let write: @Sendable (String) -> Void = { line in
            queue.async {
                print(line)
                file.value?.write(Data((line + "\n").utf8))
            }
        }
        write("\n===== session start \(formatter.string(from: Date())) (Swift \(MergeScribe.version)) =====")
        Log.setSink(write)
    }
}

final class Box<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}
