import Foundation

/// Fixtures written by scripts/swift_goldens.py: what the Python code returned.
enum Golden {
    static func load(_ name: String) -> JSON {
        let url = Bundle.module.url(forResource: "Goldens/\(name)", withExtension: "json")!
        return try! JSONDecoder().decode(JSON.self, from: Data(contentsOf: url))
    }
}

@testable import MergeScribeCore
