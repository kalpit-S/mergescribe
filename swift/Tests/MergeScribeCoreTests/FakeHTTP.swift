import Foundation
@testable import MergeScribeCore

/// Scripted replies for the OpenRouter clients, and a record of what they sent.
final class FakeHTTP: HTTPClient, @unchecked Sendable {
    enum Reply {
        case data(Int, String)
        /// Lines, then optionally hang (a stall) or fail after them.
        case lines(Int, [String], hang: Bool = false, fail: URLError? = nil)
        case error(URLError)
    }

    private let lock = NSLock()
    private var replies: [Reply]
    private(set) var sent: [URLRequest] = []

    init(_ replies: [Reply]) { self.replies = replies }

    var bodies: [JSON] {
        lock.withLock { sent.compactMap { $0.httpBody.flatMap { try? JSONDecoder().decode(JSON.self, from: $0) } } }
    }

    private func next(_ request: URLRequest) -> Reply {
        lock.withLock {
            sent.append(request)
            return replies.isEmpty ? .error(URLError(.cannotConnectToHost)) : replies.removeFirst()
        }
    }

    private static func response(_ request: URLRequest, _ status: Int) -> HTTPURLResponse {
        HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil)!
    }

    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        switch next(request) {
        case .data(let status, let body): return (Data(body.utf8), Self.response(request, status))
        case .lines(let status, let lines, _, _): return (Data(lines.joined(separator: "\n").utf8), Self.response(request, status))
        case .error(let error): throw error
        }
    }

    func lines(for request: URLRequest) async throws -> (HTTPURLResponse, AsyncThrowingStream<String, Error>) {
        switch next(request) {
        case .data(let status, let body):
            return (Self.response(request, status), AsyncThrowingStream { $0.yield(body); $0.finish() })
        case .lines(let status, let lines, let hang, let fail):
            let stream = AsyncThrowingStream<String, Error> { continuation in
                let task = Task {
                    for line in lines { continuation.yield(line) }
                    if hang { try? await Task.sleep(for: .seconds(30)) }
                    if let fail { continuation.finish(throwing: fail) } else { continuation.finish() }
                }
                continuation.onTermination = { _ in task.cancel() }
            }
            return (Self.response(request, status), stream)
        case .error(let error): throw error
        }
    }
}

/// A server-sent event carrying one piece of a correction.
func sse(_ content: String) -> String {
    "data: " + JSON.object(["id": "gen-1", "model": "openai/gpt-6-luna", "provider": "OpenAI",
                            "choices": [["delta": ["content": .string(content)]]]]).line()
}
