// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import Testing

@testable import MLXFoundationModels

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private final class SeedRecorder: @unchecked Sendable, Hashable {
    private let lock = NSLock()
    private var storage: UInt64?

    static func == (lhs: SeedRecorder, rhs: SeedRecorder) -> Bool {
        lhs === rhs
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self))
    }

    func record(_ seed: UInt64?) {
        lock.lock()
        storage = seed
        lock.unlock()
    }

    var seed: UInt64? {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct RecordingLanguageModel: LanguageModel {
    typealias Executor = RecordingLanguageModelExecutor

    let recorder: SeedRecorder

    var capabilities: LanguageModelCapabilities {
        LanguageModelCapabilities([])
    }

    var executorConfiguration: SeedRecorder {
        recorder
    }
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct RecordingLanguageModelExecutor: LanguageModelExecutor {
    typealias Configuration = SeedRecorder
    typealias Model = RecordingLanguageModel

    let recorder: SeedRecorder

    init(configuration: SeedRecorder) throws {
        recorder = configuration
    }

    func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: RecordingLanguageModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        let seed: UInt64?
        switch request.generationOptions.samplingMode?.kind {
        case .some(.randomTopK(_, let value)):
            seed = value
        default:
            seed = nil
        }
        recorder.record(seed)
        await channel.send(
            .response(action: .appendText("ok", tokenCount: 1)))
    }
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct UsageForwardingConfiguration: Hashable, Sendable {
    let emitsResponseText: Bool
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct UsageForwardingLanguageModel: LanguageModel {
    typealias Executor = UsageForwardingLanguageModelExecutor

    let configuration: UsageForwardingConfiguration

    var capabilities: LanguageModelCapabilities {
        LanguageModelCapabilities([.reasoning])
    }

    var executorConfiguration: UsageForwardingConfiguration {
        configuration
    }
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct UsageForwardingLanguageModelExecutor: LanguageModelExecutor {
    typealias Configuration = UsageForwardingConfiguration
    typealias Model = UsageForwardingLanguageModel

    let configuration: UsageForwardingConfiguration

    init(configuration: UsageForwardingConfiguration) throws {
        self.configuration = configuration
    }

    func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: UsageForwardingLanguageModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        let responseEntryID = UUID().uuidString
        if configuration.emitsResponseText {
            await MLXLanguageModel.Executor.emit(
                text: "ok",
                entryID: responseEntryID,
                destination: .response,
                into: channel)
        } else {
            await MLXLanguageModel.Executor.emit(
                text: "private reasoning",
                entryID: UUID().uuidString,
                destination: .reasoning,
                into: channel)
            await MLXLanguageModel.Executor.establishEmptyResponseEntry(
                entryID: responseEntryID,
                into: channel)
            await MLXLanguageModel.Executor.emitMetadata(
                ["incompleteOutput": true],
                entryID: responseEntryID,
                into: channel)
        }
        await MLXLanguageModel.Executor.emitUsage(
            input: .init(totalTokenCount: 7, cachedTokenCount: 0),
            output: .init(totalTokenCount: 3, reasoningTokenCount: 3),
            entryID: responseEntryID,
            into: channel)
    }
}

@Suite("GenerationOptions forwarding")
struct GenerationOptionsForwardingTests {
    @Test("LanguageModelSession forwards an exact UInt64 seed to the executor")
    func seedReachesExecutorWithoutNarrowing() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let recorder = SeedRecorder()
        let session = LanguageModelSession(
            model: RecordingLanguageModel(recorder: recorder))

        _ = try await session.respond(
            to: "Record the generation options.",
            options: GenerationOptions(
                samplingMode: .random(top: 40, seed: UInt64.max),
                maximumResponseTokens: 1))

        #expect(recorder.seed == UInt64.max)
    }
}

@Suite("Usage forwarding")
struct UsageForwardingTests {
    @Test(arguments: [true, false])
    func usageReachesConsumerWithOrWithoutPublicResponseText(
        emitsResponseText: Bool
    ) async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let session = LanguageModelSession(
            model: UsageForwardingLanguageModel(
                configuration: UsageForwardingConfiguration(
                    emitsResponseText: emitsResponseText)))

        var finalInputTokenCount = 0
        var finalOutputTokenCount = 0
        for try await snapshot in session.streamResponse(to: "Report usage.") {
            finalInputTokenCount = snapshot.usage.input.totalTokenCount
            finalOutputTokenCount = snapshot.usage.output.totalTokenCount
        }

        #expect(finalInputTokenCount == 7)
        #expect(finalOutputTokenCount == 3)
    }
}

#endif
