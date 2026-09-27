import Foundation

/// The network, as the OpenRouter clients use it: whole replies and streams of lines.
/// Tests stand in for it with scripted replies.
public protocol HTTPClient: Sendable {
    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse)
    /// The status as soon as headers arrive, then the body line by line.
    func lines(for request: URLRequest) async throws -> (HTTPURLResponse, AsyncThrowingStream<String, Error>)
}

public struct HTTPError: Error, CustomStringConvertible {
    public let status: Int
    public let body: String
    public var description: String { "HTTP \(status) \(body.prefix(120))" }
}

/// URLSession, with one shared connection pool (a warm connection saves ~70-100ms a request).
public final class URLSessionHTTPClient: HTTPClient {
    private let session: URLSession

    public init(configuration: URLSessionConfiguration = .default) {
        configuration.httpMaximumConnectionsPerHost = 8
        session = URLSession(configuration: configuration)
    }

    public func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw URLError(.badServerResponse) }
        return (data, http)
    }

    public func lines(for request: URLRequest) async throws -> (HTTPURLResponse, AsyncThrowingStream<String, Error>) {
        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse else { throw URLError(.badServerResponse) }
        let stream = AsyncThrowingStream<String, Error> { continuation in
            let reader = Task {
                do {
                    for try await line in bytes.lines { continuation.yield(line) }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in reader.cancel() }
        }
        return (http, stream)
    }
}

/// OpenRouter's endpoints and the headers every request carries.
public enum OpenRouterAPI {
    public static let base = URL(string: "https://openrouter.ai/api/v1")!

    public static func request(_ url: URL, key: String, json: JSON? = nil, timeout: TimeInterval) -> URLRequest {
        var request = URLRequest(url: url, timeoutInterval: timeout)
        request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        request.setValue("https://github.com/kalpit-S/mergescribe", forHTTPHeaderField: "HTTP-Referer")
        request.setValue("MergeScribe", forHTTPHeaderField: "X-Title")
        if let json {
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.withoutEscapingSlashes]
            request.httpBody = try? encoder.encode(json)
        }
        return request
    }
}

extension AsyncThrowingStream where Element == String, Failure == Error {
    /// Everything left in the stream, joined: an error reply's body.
    func collect(limit: Int = 4000) async -> String {
        var out = ""
        do {
            for try await line in self {
                out += line + "\n"
                if out.count > limit { break }
            }
        } catch {}
        return out
    }
}
