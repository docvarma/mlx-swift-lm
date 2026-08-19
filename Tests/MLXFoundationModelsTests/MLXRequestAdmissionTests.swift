// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

@Suite("MLX request admission")
struct MLXRequestAdmissionTests {
    private enum ProbeError: Error {
        case rejected
    }

    private actor GenerationProbe {
        private(set) var starts = 0

        func start() {
            starts += 1
        }
    }

    @Test("rejection prevents generation")
    func rejectionPreventsGeneration() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let probe = GenerationProbe()
        let metrics = MLXRequestMetrics(
            inputTokenCount: 41,
            reservedOutputTokenCount: 17,
            attachmentCount: 1)
        let admission = MLXRequestAdmission { received in
            #expect(received == metrics)
            throw ProbeError.rejected
        }

        await #expect(throws: ProbeError.self) {
            _ = try await MLXRequestAdmission.perform(
                metrics: metrics,
                admission: admission
            ) {
                await probe.start()
            }
        }
        #expect(await probe.starts == 0)
    }

    @Test("cancellation propagates while admission is running")
    func cancellationDuringAdmission() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let (started, startedContinuation) = AsyncStream<Void>.makeStream()
        let probe = GenerationProbe()
        let task = Task {
            try await MLXRequestAdmission.perform(
                metrics: .init(
                    inputTokenCount: 1,
                    reservedOutputTokenCount: 1,
                    attachmentCount: 0),
                admission: MLXRequestAdmission { _ in
                    startedContinuation.yield()
                    startedContinuation.finish()
                    try await Task.sleep(for: .seconds(60))
                }
            ) {
                await probe.start()
            }
        }

        for await _ in started { break }
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(await probe.starts == 0)
    }

    @Test("cancellation propagates after the engine starts")
    func cancellationDuringGeneration() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let (started, startedContinuation) = AsyncStream<Void>.makeStream()
        let task = Task {
            try await MLXRequestAdmission.perform(
                metrics: .init(
                    inputTokenCount: 1,
                    reservedOutputTokenCount: 1,
                    attachmentCount: 0),
                admission: MLXRequestAdmission { _ in }
            ) {
                startedContinuation.yield()
                startedContinuation.finish()
                try await Task.sleep(for: .seconds(60))
            }
        }

        for await _ in started { break }
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
    }

    @Test("metrics use the exact prepared LMInput sequence length")
    func metricsMatchPreparedInput() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let preparedSequenceLength = 5
        let metrics = MLXLanguageModel.Executor.metrics(
            inputTokenCount: preparedSequenceLength,
            reservedOutputTokenCount: 29,
            attachmentCount: 2)

        #expect(metrics.inputTokenCount == preparedSequenceLength)
        #expect(metrics.reservedOutputTokenCount == 29)
        #expect(metrics.attachmentCount == 2)
    }

    @Test("admission does not alter caller-declared capabilities")
    func capabilitiesRemainIndependent() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let capabilities: [LanguageModelCapabilities.Capability] = [
            .vision, .guidedGeneration, .toolCalling, .reasoning,
        ]
        let model = MLXLanguageModel(
            configuration: ModelConfiguration(id: "synthetic/combined"),
            capabilities: capabilities,
            requestAdmission: MLXRequestAdmission { _ in },
            weightsLocation: { _ in URL(fileURLWithPath: "/tmp") },
            load: stubLoad())

        for capability in capabilities {
            #expect(model.capabilities.contains(capability))
        }
    }
}

#endif
