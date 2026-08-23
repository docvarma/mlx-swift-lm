// Copyright © 2026 Apple Inc.

import Testing

@testable import MLXLMCommon

struct ReasoningGuidedResponseDecoderTests {
    @Test("Decodes primed Qwen reasoning and final JSON")
    func primedReasoning() throws {
        let tokenizer = ReasoningGuidedTokenizer(map: [
            1: "reason", 2: "ing", 3: "</think>", 4: "\n\n", 5: #"{"ok":true}"#,
        ])
        let config = ReasoningConfig(
            startDelimiter: "<think>",
            endDelimiter: "</think>",
            promptStrategy: .alwaysOn)
        let response = try #require(
            ReasoningGuidedResponseDecoder.decode(
                rawText: "reasoning</think>\n\n{\"ok\":true}",
                tokenIDs: [1, 2, 3, 4, 5],
                config: config,
                primedInside: true,
                responseSeparator: "\n\n",
                tokenizer: tokenizer))

        #expect(response.reasoningText == "reasoning")
        #expect(response.responseText == #"{"ok":true}"#)
        #expect(response.reasoningTokenCount == 3)
    }

    @Test("Rejects a response without a reasoning close")
    func missingClose() {
        let tokenizer = ReasoningGuidedTokenizer(map: [1: "reason"])
        let config = ReasoningConfig(
            startDelimiter: "<think>",
            endDelimiter: "</think>",
            promptStrategy: .alwaysOn)
        #expect(
            ReasoningGuidedResponseDecoder.decode(
                rawText: "reason",
                tokenIDs: [1],
                config: config,
                primedInside: true,
                responseSeparator: "\n\n",
                tokenizer: tokenizer) == nil)
    }
}

private struct ReasoningGuidedTokenizer: Tokenizer {
    let map: [Int: String]
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.compactMap { map[$0] }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? {
        map.first { $0.value == token }?.key
    }
    func convertIdToToken(_ id: Int) -> String? { map[id] }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
