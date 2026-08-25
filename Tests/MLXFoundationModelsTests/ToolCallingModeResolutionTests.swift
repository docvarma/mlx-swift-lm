// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import FoundationModels
import MLXLMCommon
import Testing
@testable import MLXFoundationModels

@Suite
struct ToolCallingModeResolutionTests {
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func tool(named name: String) -> Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: name,
            description: "Test tool",
            parameters: String.generationSchema)
    }

    @Test func nilDefaultsToAllowed() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let mode = ToolCallingModeResolution.resolve(nil)
        #expect(mode == GenerationOptions.ToolCallingMode.allowed)
        #expect(ToolCallingModeResolution.usesAllowedBehavior(mode))
        #expect(
            try ToolCallingModeResolution.enabledToolDefinitions(
                for: mode, from: [tool(named: "real")]
            ).count == 1)
    }

    @Test func preservesAllDocumentedModes() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let modes: [GenerationOptions.ToolCallingMode] = [
            .allowed, .required, .disallowed,
        ]
        for mode in modes {
            #expect(ToolCallingModeResolution.resolve(mode) == mode)
        }
    }

    @Test func requiredAllowsResponseAfterDynamicToolsAreExhausted() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let definitions = try ToolCallingModeResolution.enabledToolDefinitions(
            for: .required, from: [])
        #expect(definitions.isEmpty)
    }

    @Test func requiredAllowsSchemaAfterDynamicToolsAreExhausted() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let definitions = try ToolCallingModeResolution.enabledToolDefinitions(
            for: .required,
            from: [],
            responseSchemaPresent: true)
        #expect(definitions.isEmpty)
    }

    @Test func disallowedDropsEvenManuallyEnabledDefinitions() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let definitions = try ToolCallingModeResolution.enabledToolDefinitions(
            for: .disallowed, from: [tool(named: "must_not_render")])
        #expect(definitions.isEmpty)
    }

    @Test func requiredPreservesEnabledToolDefinitions() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let definitions = try ToolCallingModeResolution.enabledToolDefinitions(
            for: .required,
            from: [tool(named: "first"), tool(named: "second")])
        #expect(definitions.map(\.name) == ["first", "second"])
        #expect(!ToolCallingModeResolution.usesAllowedBehavior(.required))
        #expect(!ToolCallingModeResolution.usesAllowedBehavior(.disallowed))
    }

    @Test func toolReasoningSelectionIsFamilySpecific() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let qwen = ReasoningConfig(
            startDelimiter: "<think>", endDelimiter: "</think>",
            promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true))
        let onyx = ReasoningConfig(
            startDelimiter: "to=self<|message|>", endDelimiter: "<|eom|>",
            promptStrategy: .none)
        let harmony = ReasoningConfig.harmonyChannels

        #expect(
            MLXLanguageModel.Executor.toolReasoningConfig(
                declared: true, config: qwen, format: .json, thinkingEnabled: true) == qwen)
        #expect(
            MLXLanguageModel.Executor.toolReasoningConfig(
                declared: true, config: onyx, format: .atem, thinkingEnabled: true) == onyx)
        #expect(
            MLXLanguageModel.Executor.toolReasoningConfig(
                declared: true, config: onyx, format: .json, thinkingEnabled: true) == nil)
        #expect(
            MLXLanguageModel.Executor.toolReasoningConfig(
                declared: true, config: onyx, format: .atem, thinkingEnabled: false) == nil)
        #expect(
            MLXLanguageModel.Executor.toolReasoningConfig(
                declared: true, config: harmony, format: .gptOSS, thinkingEnabled: true) == nil)
    }

    @Test func effortReasoningMapsFoundationLevels() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let config = ReasoningConfig.harmonyChannels

        #expect(
            try MLXLanguageModel.Executor.reasoningPromptContext(
                config: config, level: nil
            ).reasoningEffort == .medium)
        #expect(
            try MLXLanguageModel.Executor.reasoningPromptContext(
                config: config, level: .light
            ).reasoningEffort == .low)
        #expect(
            try MLXLanguageModel.Executor.reasoningPromptContext(
                config: config, level: .moderate
            ).reasoningEffort == .medium)
        #expect(
            try MLXLanguageModel.Executor.reasoningPromptContext(
                config: config, level: .deep
            ).reasoningEffort == .high)
    }

    @Test(arguments: ["no_think", "low", "medium", "high", "future_effort"])
    func effortReasoningRejectsCustomLevels(_ customLevel: String) throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        do {
            _ = try MLXLanguageModel.Executor.reasoningPromptContext(
                config: .harmonyChannels, level: .custom(customLevel))
            Issue.record("Expected custom effort level to be rejected")
        } catch LanguageModelError.unsupportedCapability(let capability) {
            #expect(capability.debugDescription.contains("Custom reasoning levels"))
        }
    }

    @Test func effortContextBuildsSharedAdditionalContext() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let context = try MLXLanguageModel.Executor.reasoningPromptContext(
            config: .harmonyChannels, level: .light)
        let additionalContext = try MLXLanguageModel.Executor.reasoningPromptAdditionalContext(
            config: .harmonyChannels,
            thinkingEnabled: context.thinkingEnabled,
            reasoningEffort: context.reasoningEffort)

        #expect(additionalContext?["reasoning_effort"] as? String == "low")
        #expect(additionalContext?["enable_thinking"] == nil)
    }

    @Test func templateFlagRetainsNoThinkConvention() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let config = ReasoningConfig.thinkTagsWithEnableThinking

        let disabled = try MLXLanguageModel.Executor.reasoningPromptContext(
            config: config, level: .custom("no_think"))
        #expect(disabled.thinkingEnabled == false)
        #expect(disabled.reasoningEffort == nil)

        let enabled = try MLXLanguageModel.Executor.reasoningPromptContext(
            config: config, level: .custom("other"))
        #expect(enabled.thinkingEnabled == true)
        #expect(enabled.reasoningEffort == nil)
    }
}

#endif
