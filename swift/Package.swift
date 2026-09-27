// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MergeScribe",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "MergeScribe", targets: ["MergeScribe"]),
    ],
    targets: [
        // Foundation only: everything that can be tested without a screen.
        .target(name: "MergeScribeCore"),
        // AppKit, Core Audio, accessibility and CGEvent implementations of Core's protocols.
        .target(name: "MergeScribeMac", dependencies: ["MergeScribeCore"]),
        .executableTarget(name: "MergeScribe", dependencies: ["MergeScribeCore", "MergeScribeMac"]),
        .testTarget(name: "MergeScribeCoreTests", dependencies: ["MergeScribeCore"],
                    resources: [.copy("Goldens")]),
    ]
)
