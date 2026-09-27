import Foundation

/// Cloud speech-to-text through OpenRouter.
///
/// Every OpenRouter transcription model is served by /audio/transcriptions (the catalogue is
/// /models?output_modalities=transcription), so any of them works without a table of which
/// model takes which endpoint.
public final class TranscriptionClient: Sendable {
    static let timeout: TimeInterval = 30

    let http: HTTPClient
    /// A fresh connection for the retry: a pooled connection the server already dropped
    /// fails with an SSL or "connection lost" error, and the same pool would hand it back.
    let freshHTTP: @Sendable () -> HTTPClient

    public init(http: HTTPClient, freshHTTP: @escaping @Sendable () -> HTTPClient = {
        URLSessionHTTPClient(configuration: .ephemeral)
    }) {
        self.http = http
        self.freshHTTP = freshHTTP
    }

    /// The recognizer name the rest of the app knows this model by: "or-<model slug>".
    public static func providerName(_ model: String) -> String {
        "or-" + model.replacingOccurrences(of: "/", with: "-").replacingOccurrences(of: ".", with: "-")
            .replacingOccurrences(of: ":", with: "-")
    }

    /// What the model heard, annotations stripped. Throws on failure.
    ///
    /// keyterms: words the speaker is known to use, for models that can be primed with them
    /// (AssemblyAI's keyterms_prompt).
    public func transcribe(wav: Data, model: String, key: String, keyterms: [String] = []) async throws -> String {
        var body: [String: JSON] = [
            "model": .string(model),
            "input_audio": ["data": .string(wav.base64EncodedString()), "format": "wav"],
        ]
        if model.hasPrefix("assemblyai/") && !keyterms.isEmpty {
            body["provider"] = ["options": ["assemblyai": ["keyterms_prompt": JSON(keyterms)]]]
        }
        let request = OpenRouterAPI.request(OpenRouterAPI.base.appendingPathComponent("audio/transcriptions"),
                                            key: key, json: .object(body), timeout: Self.timeout)
        var client = http
        for attempt in 1...2 {
            do {
                let (data, response) = try await client.data(for: request)
                guard response.statusCode == 200 else {
                    throw HTTPError(status: response.statusCode, body: String(decoding: data.prefix(300), as: UTF8.self))
                }
                let text = (try? JSONDecoder().decode(JSON.self, from: data))?["text"]?.string ?? ""
                return stripAnnotations(text)
            } catch let error as URLError where attempt == 1 && Self.isTransport(error) {
                Log.line("[\(Self.providerName(model))] Transport error, retrying with a fresh connection: \(error.localizedDescription)")
                try await Task.sleep(for: .milliseconds(250))
                client = freshHTTP()
            }
        }
        throw URLError(.unknown)
    }

    static func isTransport(_ error: URLError) -> Bool {
        [.networkConnectionLost, .secureConnectionFailed, .timedOut, .cannotConnectToHost,
         .notConnectedToInternet, .dnsLookupFailed, .cannotFindHost, .badServerResponse].contains(error.code)
    }
}

/// 16-bit PCM WAV, mono: what the recognizers are sent and the training capture keeps.
public enum WAV {
    public static func encode(_ samples: [Float], sampleRate: Int) -> Data {
        var pcm = Data(capacity: samples.count * 2)
        for sample in samples {
            // Clipped: a value past full scale would otherwise wrap round to the other extreme.
            let value = Int16((max(-1, min(1, sample)) * 32767).rounded(.towardZero))
            withUnsafeBytes(of: value.littleEndian) { pcm.append(contentsOf: $0) }
        }
        var header = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { header.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { header.append(contentsOf: $0) } }
        header.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + pcm.count))
        header.append(contentsOf: Array("WAVE".utf8))
        header.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(1)
        u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 2)); u16(2); u16(16)
        header.append(contentsOf: Array("data".utf8)); u32(UInt32(pcm.count))
        return header + pcm
    }

    /// Samples and sample rate from a 16-bit PCM WAV (the format encode writes), mixed to mono.
    public static func decode(_ data: Data) -> (samples: [Float], sampleRate: Int)? {
        let bytes = [UInt8](data)
        guard bytes.count >= 12, String(decoding: bytes[0..<4], as: UTF8.self) == "RIFF",
              String(decoding: bytes[8..<12], as: UTF8.self) == "WAVE" else { return nil }
        func u16(_ i: Int) -> Int { Int(bytes[i]) | Int(bytes[i + 1]) << 8 }
        func u32(_ i: Int) -> Int { u16(i) | u16(i + 2) << 16 }
        var offset = 12, channels = 1, rate = 16000, bits = 16
        while offset + 8 <= bytes.count {
            let id = String(decoding: bytes[offset..<(offset + 4)], as: UTF8.self), size = u32(offset + 4)
            let body = offset + 8
            if id == "fmt " {
                channels = u16(body + 2); rate = u32(body + 4); bits = u16(body + 14)
            } else if id == "data" {
                guard bits == 16 else { return nil }
                let end = min(bytes.count, body + size)
                var samples: [Float] = []
                samples.reserveCapacity((end - body) / 2 / max(1, channels))
                var i = body
                while i + 2 * channels <= end {
                    var sum: Float = 0
                    for c in 0..<channels { sum += Float(Int16(bitPattern: UInt16(u16(i + 2 * c)))) / 32768 }
                    samples.append(sum / Float(channels))
                    i += 2 * channels
                }
                return (samples, rate)
            }
            offset = body + size + (size % 2)
        }
        return nil
    }
}
