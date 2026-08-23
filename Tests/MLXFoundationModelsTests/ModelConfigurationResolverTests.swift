// Copyright © 2025 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import Testing

@testable import MLXFoundationModels
import MLXLMCommon

@Suite
struct ModelConfigurationResolverTests {

    // MARK: - ModelDescriptor

    @Test func descriptorStoresInspectionInputs() {
        let configData = Data(#"{"model_type":"qwen3"}"#.utf8)
        let descriptor = ModelDescriptor(
            modelType: "qwen3",
            modelId: "mlx-community/Qwen3-4B-4bit",
            configData: configData,
            tokenizer: ByteTokenizer())
        #expect(descriptor.modelType == "qwen3")
        #expect(descriptor.modelId == "mlx-community/Qwen3-4B-4bit")
        #expect(descriptor.configData == configData)
    }

    // MARK: - DefaultConfigurationResolver pass-through

    private func descriptor(
        modelType: String = "qwen3",
        modelId: String = "mlx-community/Qwen3-4B-4bit"
    ) -> ModelDescriptor {
        ModelDescriptor(
            modelType: modelType, modelId: modelId,
            configData: nil, tokenizer: ByteTokenizer())
    }

    private func sampleConfiguration() -> ModelConfiguration {
        ModelConfiguration(
            directory: URL(fileURLWithPath: "/tmp/model"),
            extraEOSTokens: ["<|im_end|>"],
            eosTokenIds: [151645],
            toolCallFormat: .json,
            reasoningConfig: ReasoningConfig(
                startDelimiter: "<think>", endDelimiter: "</think>",
                promptStrategy: .alwaysOn))
    }

    @Test func defaultResolverReturnsConfigurationUnchanged() {
        let config = sampleConfiguration()
        let resolved = DefaultConfigurationResolver().resolve(config, for: descriptor())
        #expect(resolved.reasoningConfig == config.reasoningConfig)
        #expect(resolved.toolCallFormat == config.toolCallFormat)
        #expect(resolved.extraEOSTokens == config.extraEOSTokens)
        #expect(resolved.eosTokenIds == config.eosTokenIds)
    }

    /// `.default` must resolve where the parameter type is `any
    /// ModelConfigurationResolver`, proving the `where Self ==
    /// DefaultConfigurationResolver` extension is wired correctly.
    @Test func dotDefaultResolvesAsExistential() {
        func accept(_ resolver: any ModelConfigurationResolver) -> ModelConfiguration {
            resolver.resolve(sampleConfiguration(), for: descriptor())
        }
        let viaSugar = accept(.default)
        let viaDirect = accept(DefaultConfigurationResolver())
        #expect(viaSugar.reasoningConfig == viaDirect.reasoningConfig)
    }

    /// A custom resolver patches one field and leaves the rest at baseline.
    @Test func customResolverPatchesReasoningDelimiterOnly() {
        struct DelimiterResolver: ModelConfigurationResolver {
            func resolve(
                _ configuration: ModelConfiguration, for descriptor: ModelDescriptor
            ) -> ModelConfiguration {
                var c = configuration
                c.reasoningConfig?.startDelimiter = "<reason>"
                return c
            }
        }
        let baseline = sampleConfiguration()
        let patched = DelimiterResolver().resolve(baseline, for: descriptor())
        #expect(patched.reasoningConfig?.startDelimiter == "<reason>")
        #expect(patched.reasoningConfig?.endDelimiter == baseline.reasoningConfig?.endDelimiter)
        #expect(patched.toolCallFormat == baseline.toolCallFormat)
        #expect(patched.extraEOSTokens == baseline.extraEOSTokens)
    }

    /// The Foundation Models bridge consumes this resolved value on every path
    /// (allowed/required tools, schema generation, and reasoning). This pins the
    /// public downstream seam used to select a full response protocol such as
    /// GPT-OSS Harmony without adding application-specific model checks.
    @Test func customResolverSelectsPublicModelProtocol() {
        struct HarmonyResolver: ModelConfigurationResolver {
            func resolve(
                _ configuration: ModelConfiguration, for descriptor: ModelDescriptor
            ) -> ModelConfiguration {
                var configuration = configuration
                configuration.toolCallFormat = .gptOSS
                configuration.reasoningConfig = .harmonyChannels
                return configuration
            }
        }

        let resolved = HarmonyResolver().resolve(
            sampleConfiguration(),
            for: descriptor(modelType: "gpt_oss", modelId: "example/gpt-oss"))

        #expect(resolved.toolCallFormat == .gptOSS)
        #expect(resolved.reasoningConfig == .harmonyChannels)
    }

}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
