// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXFoundationModels

/// No-model coverage for the adapter's prompt preparation paths. The scripted
/// processor records the context passed to the tokenizer boundary. The
/// allowed-tool/schema case uses a tiny deterministic model to complete the
/// native allowed response before stopping at the second schema preparation.
@Suite("Reasoning prompt forwarding")
struct ReasoningPromptForwardingTests {

    @Test func unconstrainedMapsLightToLow() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let probe = PromptProbe()
        let request = makeExecutorRequest(
            transcript: Self.transcript(),
            contextOptions: ContextOptions(reasoningLevel: .light))

        let snapshots = try await probe.respond(to: request)
        #expect(snapshots.last?.effort == "low")
        #expect(snapshots.last?.toolCount == 0)
    }

    @Test func schemaMapsModerateToMedium() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let probe = PromptProbe()
        let request = makeExecutorRequest(
            transcript: Self.transcript(),
            schema: String.generationSchema,
            contextOptions: ContextOptions(reasoningLevel: .moderate))

        let snapshots = try await probe.respond(to: request)
        #expect(snapshots.last?.effort == "medium")
        #expect(snapshots.last?.toolCount == 0)
    }

    @Test func toolsMapDeepToHigh() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let probe = PromptProbe()
        let request = makeExecutorRequest(
            transcript: Self.transcript(),
            enabledTools: [Self.tool()],
            contextOptions: ContextOptions(reasoningLevel: .deep))

        let snapshots = try await probe.respond(to: request)
        #expect(snapshots.last?.effort == "high")
        #expect(snapshots.last?.toolCount == 1)
    }

    @Test func allowedToolToSchemaForwardsEffortAfterNativeResponse() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let probe = PromptProbe(stopAfterPreparation: 4)
        let request = makeExecutorRequest(
            transcript: Self.transcript(),
            enabledTools: [Self.tool()],
            schema: String.generationSchema,
            generationOptions: GenerationOptions(
                samplingMode: .greedy, maximumResponseTokens: 4),
            contextOptions: ContextOptions(reasoningLevel: .light))

        let snapshots = try await probe.respond(to: request)
        let generation = probe.generation

        // Preparation sequence: baseline, allowed tools, schema baseline, and
        // the schema reasoning prompt. The final prompt is the distinct
        // allowed-tool-to-schema branch: it has no tools and retains "low".
        #expect(snapshots.count == 4)
        #expect(snapshots[0].effort == nil)
        #expect(snapshots[0].toolCount == 0)
        #expect(snapshots[1].effort == "low")
        #expect(snapshots[1].toolCount == 1)
        #expect(snapshots[2].effort == nil)
        #expect(snapshots[2].toolCount == 0)
        #expect(snapshots[3].effort == "low")
        #expect(snapshots[3].toolCount == 0)

        // The fake model seeds a normal "hello" token and then emits EOS;
        // the executor therefore completes native allowed generation without
        // a tool call before it enters the schema branch.
        #expect(generation.producedNormalResponse)
        #expect(generation.toolCallCount == 0)
    }

    @Test func effortCustomLevelFailsClosed() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let probe = PromptProbe()
        let request = makeExecutorRequest(
            transcript: Self.transcript(),
            contextOptions: ContextOptions(reasoningLevel: .custom("no_think")))

        do {
            _ = try await probe.respond(to: request)
            Issue.record("Expected custom effort level to be rejected")
        } catch LanguageModelError.unsupportedCapability(let capability) {
            #expect(capability.capability == .reasoning)
        }
        #expect(probe.snapshots.count == 1)
        #expect(probe.snapshots[0].effort == nil)
    }

    @Test func templateFlagPreservesNoThink() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let probe = PromptProbe(config: .thinkTagsWithEnableThinking)
        let request = makeExecutorRequest(
            transcript: Self.transcript(),
            contextOptions: ContextOptions(reasoningLevel: .custom("no_think")))

        let snapshots = try await probe.respond(to: request)
        #expect(snapshots.last?.thinking == false)
        #expect(snapshots.last?.effort == nil)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func transcript() -> Transcript {
        Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: "hello"))]))
        ])
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func tool() -> Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: "weather",
            description: "Returns weather",
            parameters: String.generationSchema)
    }
}

private struct PromptSnapshot: Sendable {
    let effort: String?
    let thinking: Bool?
    let toolCount: Int
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private final class PromptProbeGeneration: @unchecked Sendable {
    private let lock = NSLock()
    private var normalResponse = false
    private var toolCalls = 0

    func markNormalResponse() {
        lock.withLock { normalResponse = true }
    }

    func append(_ event: MLXLanguageModel.Executor.GenerationEvent) {
        lock.withLock {
            switch event {
            case .toolCall:
                toolCalls += 1
            default:
                break
            }
        }
    }

    var producedNormalResponse: Bool {
        lock.withLock { normalResponse }
    }

    var toolCallCount: Int {
        lock.withLock { toolCalls }
    }
}

private struct PromptProbeStop: Error {}

private final class PromptProbeProcessor: UserInputProcessor, @unchecked Sendable {
    private let lock = NSLock()
    private let stopAfterPreparation: Int
    private var recorded: [PromptSnapshot] = []

    init(stopAfterPreparation: Int) {
        self.stopAfterPreparation = stopAfterPreparation
    }

    var snapshots: [PromptSnapshot] {
        lock.withLock { recorded }
    }

    func prepare(input: UserInput) async throws -> LMInput {
        let snapshot = PromptSnapshot(
            effort: input.additionalContext?["reasoning_effort"] as? String,
            thinking: input.additionalContext?["enable_thinking"] as? Bool,
            toolCount: input.tools?.count ?? 0)
        lock.withLock { recorded.append(snapshot) }

        // Stop at the first strategy-specific prompt by default. The
        // allowed-tool/schema case raises this threshold so the scripted
        // generation can complete before the schema re-prompt is recorded.
        if snapshots.count == stopAfterPreparation {
            throw PromptProbeStop()
        }
        return LMInput(tokens: MLXArray([Int32(0)]))
    }
}

private struct PromptProbeTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [0] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.compactMap { tokenID in
            switch tokenID {
            case 1: return "hello"
            case 2: return skipSpecialTokens ? nil : ""
            default: return nil
            }
        }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    var eosTokenId: Int? { 2 }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [0]
    }
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private final class PromptProbeModel: Module, MLXLMCommon.LanguageModel, @unchecked Sendable {
    private let generation: PromptProbeGeneration

    init(generation: PromptProbeGeneration) {
        self.generation = generation
        super.init()
    }

    override init() {
        self.generation = PromptProbeGeneration()
        super.init()
    }

    private static func logits(selecting tokenID: Int) -> MLXArray {
        var values = [Float](repeating: -100, count: 3)
        values[tokenID] = 100
        return MLXArray(values, [1, 1, 3])
    }

    func prepare(
        _ input: LMInput,
        cache: [KVCache],
        state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        .logits(LMOutput(logits: Self.logits(selecting: 1)))
    }

    func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        generation.markNormalResponse()
        return LMOutput(logits: Self.logits(selecting: 2))
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] { [] }
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private final class PromptProbe: @unchecked Sendable {
    private let processor: PromptProbeProcessor
    private let generationRecorder: PromptProbeGeneration
    private let model: MLXLanguageModel

    init(
        config: ReasoningConfig = .harmonyChannels,
        stopAfterPreparation: Int = 2
    ) {
        let processor = PromptProbeProcessor(stopAfterPreparation: stopAfterPreparation)
        self.processor = processor
        let generation = PromptProbeGeneration()
        self.generationRecorder = generation

        var configuration = ModelConfiguration(
            id: "prompt-probe-\(UUID().uuidString)",
            toolCallFormat: .json,
            reasoningConfig: config)
        configuration.stopStrings = []
        self.model = MLXLanguageModel(
            configuration: configuration,
            capabilities: [.guidedGeneration, .toolCalling, .reasoning],
            weightsLocation: { _ in URL(fileURLWithPath: "/tmp") },
            load: { configuration, _ in
                ModelContainer(
                    context: ModelContext(
                        configuration: configuration,
                        model: PromptProbeModel(generation: generation),
                        processor: processor,
                        tokenizer: PromptProbeTokenizer()))
            })
    }

    var snapshots: [PromptSnapshot] { processor.snapshots }
    var generation: PromptProbeGeneration { generationRecorder }

    func respond(
        to request: LanguageModelExecutorGenerationRequest
    ) async throws -> [PromptSnapshot] {
        let executor = try MLXLanguageModel.Executor(
            configuration: .init(modelID: model.modelID))
        let channel = LanguageModelExecutorGenerationChannel()
        let drain = Task<Void, Never> {
            do { for try await _ in channel {} } catch {}
        }
        defer { drain.cancel() }

        do {
            try await MLXLanguageModel.Executor.$generationObserver.withValue(
                { self.generationRecorder.append($0) },
                operation: {
                    try await executor.respond(to: request, model: model, streamingInto: channel)
                })
            Issue.record("Expected the prompt probe to stop before generation")
        } catch is PromptProbeStop {
            // Expected: the processor has already captured the relevant prompt.
        }
        return snapshots
    }
}

#endif
