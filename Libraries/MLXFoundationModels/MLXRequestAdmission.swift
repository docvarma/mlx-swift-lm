// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation

/// Non-sensitive metrics for the exact request prepared by the MLX executor.
/// Prompt text, schemas, tools, labels, and media bytes are intentionally absent.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
public struct MLXRequestMetrics: Equatable, Sendable {
    public let inputTokenCount: Int
    public let reservedOutputTokenCount: Int
    public let attachmentCount: Int

    public init(
        inputTokenCount: Int,
        reservedOutputTokenCount: Int,
        attachmentCount: Int
    ) {
        self.inputTokenCount = inputTokenCount
        self.reservedOutputTokenCount = reservedOutputTokenCount
        self.attachmentCount = attachmentCount
    }
}

/// An async admission decision made after MLX has rendered the exact `LMInput`
/// and immediately before a generation phase starts.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
public struct MLXRequestAdmission: Hashable, Sendable {
    private let id: UUID
    private let body: @Sendable (MLXRequestMetrics) async throws -> Void

    public init(
        _ body: @Sendable @escaping (MLXRequestMetrics) async throws -> Void
    ) {
        self.id = UUID()
        self.body = body
    }

    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.id == rhs.id
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }

    func callAsFunction(_ metrics: MLXRequestMetrics) async throws {
        try Task.checkCancellation()
        try await body(metrics)
        try Task.checkCancellation()
    }

    static func perform<Result>(
        metrics: MLXRequestMetrics,
        admission: MLXRequestAdmission?,
        operation: () async throws -> Result
    ) async throws -> Result {
        if let admission {
            try await admission(metrics)
        } else {
            try Task.checkCancellation()
        }
        return try await operation()
    }
}

#endif
#endif
