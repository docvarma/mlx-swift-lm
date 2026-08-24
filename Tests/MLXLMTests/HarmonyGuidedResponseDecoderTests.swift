// Copyright © 2026 Apple Inc.

import Testing

@testable import MLXLMCommon

struct HarmonyGuidedResponseDecoderTests {
    @Test("Decodes analysis and schema payload without framing")
    func analysisAndFinal() throws {
        let pieces = [
            "<|channel|>", "analysis", "<|message|>", "reason", "ing", "<|end|>",
            "<|start|>", "assistant",
            "<|channel|>", "final ", "<|constrain|>", "json", "<|message|>",
            #"{"answer":"yes"}"#,
        ]
        let tokenizer = GuidedResponseTokenizer(tokens: pieces + guidedControlTokens)
        let tokenIDs = try pieces.map { try #require(tokenizer.convertTokenToId($0)) }
        let response = try #require(
            HarmonyGuidedResponseDecoder.decode(
                tokenIDs: tokenIDs,
                grammarTerminated: true,
                tokenizer: tokenizer))

        #expect(response.reasoningText == "reasoning")
        #expect(response.responseText == #"{"answer":"yes"}"#)
        #expect(response.reasoningTokenCount == 2)
    }

    @Test("Accepts final-only response")
    func finalOnly() throws {
        let pieces = [
            "<|channel|>", "final ", "<|constrain|>", "json", "<|message|>",
            #"{"answer":"yes"}"#, "<|end|>",
        ]
        let tokenizer = GuidedResponseTokenizer(tokens: pieces + guidedControlTokens)
        let tokenIDs = try pieces.map { try #require(tokenizer.convertTokenToId($0)) }
        let response = try #require(
            HarmonyGuidedResponseDecoder.decode(
                tokenIDs: tokenIDs,
                grammarTerminated: true,
                tokenizer: tokenizer))

        #expect(response.reasoningText.isEmpty)
        #expect(response.responseText == #"{"answer":"yes"}"#)
        #expect(response.reasoningTokenCount == 0)
    }

    @Test("Rejects truncated final frame")
    func truncatedFinal() throws {
        let pieces = [
            "<|channel|>", "final ", "<|constrain|>", "json", "<|message|>", "{",
        ]
        let tokenizer = GuidedResponseTokenizer(tokens: pieces + guidedControlTokens)
        let tokenIDs = try pieces.map { try #require(tokenizer.convertTokenToId($0)) }
        #expect(
            HarmonyGuidedResponseDecoder.decode(
                tokenIDs: tokenIDs,
                grammarTerminated: false,
                tokenizer: tokenizer) == nil)
    }

    @Test("Decodes schema payload committed by the sampled stop token")
    func finalPayloadWithSampledStopToken() throws {
        // The final frame's <|end|> arrives only as the sampled stop token,
        // not inside the generated token stream.
        let pieces = [
            "<|channel|>", "final ", "<|constrain|>", "json", "<|message|>",
            #"{"answer":"yes"}"#,
        ]
        let tokenizer = GuidedResponseTokenizer(tokens: pieces + guidedControlTokens)
        let tokenIDs = try pieces.map { try #require(tokenizer.convertTokenToId($0)) }
        let endTokenID = try #require(tokenizer.convertTokenToId("<|end|>"))
        let response = try #require(
            HarmonyGuidedResponseDecoder.decode(
                tokenIDs: tokenIDs,
                sampledStopTokenID: endTokenID,
                grammarTerminated: true,
                tokenizer: tokenizer))

        #expect(response.reasoningText.isEmpty)
        #expect(response.responseText == #"{"answer":"yes"}"#)
        #expect(response.reasoningTokenCount == 0)
    }

    @Test("Decodes a required Harmony tool call after analysis")
    func requiredToolCall() throws {
        let bufferedPieces = [
            "<|channel|>", "analysis", "<|message|>", "inspect", "<|end|>",
            "<|start|>", "assistant",
            "<|channel|>", "commentary to=functions.read_source ",
            "<|constrain|>", "json", "<|message|>",
            #"{"sourceID":"source-1"}"#,
        ]
        let tokenizer = GuidedResponseTokenizer(tokens: bufferedPieces + guidedControlTokens)
        let tokenIDs = try bufferedPieces.map { try #require(tokenizer.convertTokenToId($0)) }
        let callTokenID = try #require(tokenizer.convertTokenToId("<|call|>"))
        let call = try #require(
            HarmonyGuidedResponseDecoder.decodeToolCall(
                tokenIDs: tokenIDs,
                sampledStopTokenID: callTokenID,
                grammarTerminated: true,
                allowedToolNames: ["read_source"],
                tokenizer: tokenizer))

        #expect(call.reasoningText == "inspect")
        #expect(call.reasoningTokenCount == 1)
        #expect(call.name == "read_source")
        #expect(call.argumentsText == #"{"sourceID":"source-1"}"#)

        #expect(
            HarmonyGuidedResponseDecoder.decodeToolCall(
                tokenIDs: tokenIDs,
                sampledStopTokenID: nil,
                grammarTerminated: true,
                allowedToolNames: ["read_source"],
                tokenizer: tokenizer) == nil)
        #expect(
            HarmonyGuidedResponseDecoder.decodeToolCall(
                tokenIDs: tokenIDs,
                sampledStopTokenID: tokenizer.convertTokenToId("<|return|>"),
                grammarTerminated: true,
                allowedToolNames: ["read_source"],
                tokenizer: tokenizer) == nil)
    }

    @Test("Rejects undeclared or incomplete Harmony tool calls")
    func rejectsInvalidToolCalls() throws {
        let pieces = [
            "<|channel|>", "commentary to=functions.secret ",
            "<|constrain|>", "json", "<|message|>", "{}", "<|call|>",
        ]
        let tokenizer = GuidedResponseTokenizer(tokens: pieces + guidedControlTokens)
        let tokenIDs = try pieces.map { try #require(tokenizer.convertTokenToId($0)) }

        #expect(
            HarmonyGuidedResponseDecoder.decodeToolCall(
                tokenIDs: tokenIDs,
                grammarTerminated: true,
                allowedToolNames: ["read_source"],
                tokenizer: tokenizer) == nil)
        #expect(
            HarmonyGuidedResponseDecoder.decodeToolCall(
                tokenIDs: Array(tokenIDs.dropLast()),
                grammarTerminated: false,
                allowedToolNames: ["secret"],
                tokenizer: tokenizer) == nil)
    }
}

private let guidedControlTokens = [
    "<|start|>", "<|channel|>", "<|message|>", "<|end|>",
    "<|call|>", "<|return|>", "<|constrain|>",
]

private struct GuidedResponseTokenizer: Tokenizer {
    private let tokenToID: [String: Int]
    private let idToToken: [Int: String]

    init(tokens: [String]) {
        var tokenToID: [String: Int] = [:]
        var idToToken: [Int: String] = [:]
        for token in tokens where tokenToID[token] == nil {
            let id = tokenToID.count
            tokenToID[token] = id
            idToToken[id] = token
        }
        self.tokenToID = tokenToID
        self.idToToken = idToToken
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.compactMap { idToToken[$0] }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { tokenToID[token] }
    func convertIdToToken(_ id: Int) -> String? { idToToken[id] }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
