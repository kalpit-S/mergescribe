import Foundation

/// A JSON value, for the parts of the log and the APIs whose shape is open-ended.
public enum JSON: Sendable, Equatable, Codable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSON])
    case object([String: JSON])

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let v = try? c.decode(Bool.self) { self = .bool(v) }
        else if let v = try? c.decode(Double.self) { self = .number(v) }
        else if let v = try? c.decode(String.self) { self = .string(v) }
        else if let v = try? c.decode([JSON].self) { self = .array(v) }
        else { self = .object(try c.decode([String: JSON].self)) }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let v): try c.encode(v)
        case .number(let v):
            // Whole numbers are written as integers, as Python writes an int.
            if v.rounded() == v, abs(v) < 9e15 { try c.encode(Int(v)) } else { try c.encode(v) }
        case .string(let v): try c.encode(v)
        case .array(let v): try c.encode(v)
        case .object(let v): try c.encode(v)
        }
    }

    /// From what JSONSerialization hands back.
    public init(any value: Any?) {
        switch value {
        case nil, is NSNull: self = .null
        case let number as NSNumber:
            self = Lenient.isBool(number) ? .bool(number.boolValue) : .number(number.doubleValue)
        case let text as String: self = .string(text)
        case let items as [Any]: self = .array(items.map { JSON(any: $0) })
        case let dict as [String: Any]: self = .object(dict.mapValues { JSON(any: $0) })
        default: self = .string(String(describing: value!))
        }
    }

    public subscript(key: String) -> JSON? {
        if case .object(let dict) = self { return dict[key] }
        return nil
    }

    public var string: String? { if case .string(let v) = self { return v }; return nil }
    public var double: Double? { if case .number(let v) = self { return v }; return nil }
    public var int: Int? { double.map { Int($0) } }
    public var bool: Bool? { if case .bool(let v) = self { return v }; return nil }
    public var array: [JSON]? { if case .array(let v) = self { return v }; return nil }
    public var object: [String: JSON]? { if case .object(let v) = self { return v }; return nil }
    public var isNull: Bool { self == .null }
}

extension JSON: ExpressibleByStringLiteral, ExpressibleByIntegerLiteral, ExpressibleByFloatLiteral,
                ExpressibleByBooleanLiteral, ExpressibleByArrayLiteral, ExpressibleByDictionaryLiteral,
                ExpressibleByNilLiteral {
    public init(stringLiteral value: String) { self = .string(value) }
    public init(integerLiteral value: Int) { self = .number(Double(value)) }
    public init(floatLiteral value: Double) { self = .number(value) }
    public init(booleanLiteral value: Bool) { self = .bool(value) }
    public init(arrayLiteral elements: JSON...) { self = .array(elements) }
    public init(dictionaryLiteral elements: (String, JSON)...) {
        self = .object(Dictionary(elements, uniquingKeysWith: { _, new in new }))
    }
    public init(nilLiteral: ()) { self = .null }
}

extension JSON {
    public init(_ value: String?) { self = value.map { .string($0) } ?? .null }
    public init(_ value: Int?) { self = value.map { .number(Double($0)) } ?? .null }
    public init(_ value: Double?) { self = value.map { .number($0) } ?? .null }
    public init(_ value: Bool?) { self = value.map { .bool($0) } ?? .null }
    public init(_ value: [String]) { self = .array(value.map { .string($0) }) }

    /// One line of JSON, keys in a stable order.
    public func line() -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return String(decoding: (try? encoder.encode(self)) ?? Data("null".utf8), as: UTF8.self)
    }
}
