// Copyright © 2026 Apple Inc.

import Foundation

/// A completed GPT-OSS schema response decoded from its authoritative token
/// stream. Harmony framing is removed; only the analysis payload and final JSON
/// payload are exposed to Foundation Models.
package struct HarmonyGuidedResponse: Sendable, Equatable {
    package let reasoningText: String
    package let responseText: String
    package let reasoningTokenCount: Int
}

/// A completed, schema-validated Harmony function call. The provider still
/// checks the JSON payload before emitting it across Foundation Models.
package struct HarmonyGuidedToolCall: Sendable, Equatable {
    package let reasoningText: String
    package let reasoningTokenCount: Int
    package let name: String
    package let argumentsText: String
}

/// Offline decoder for an xgrammar-constrained Harmony response.
package enum HarmonyGuidedResponseDecoder {
    package static func decode(
        tokenIDs: [Int],
        sampledStopTokenID _: Int? = nil,
        grammarTerminated: Bool,
        tokenizer: any Tokenizer
    ) -> HarmonyGuidedResponse? {
        guard grammarTerminated,
            var parser = HarmonyFrameParser(tokenizer: tokenizer)
        else {
            return nil
        }

        var reasoningTokenIDs: [Int] = []
        var responseTokenIDs: [Int] = []
        var sawInvalidTerminator = false
        var sawFinalCompletion = false

        for token in tokenIDs {
            for step in parser.push(token) {
                switch step {
                case .payload(let header, let payloadToken):
                    switch header.channel {
                    case .analysis:
                        reasoningTokenIDs.append(payloadToken)
                    case .final:
                        responseTokenIDs.append(payloadToken)
                    case .commentary, .other:
                        break
                    }
                case .closed(let frame):
                    switch frame.header.channel {
                    case .analysis:
                        if frame.terminator != .end { sawInvalidTerminator = true }
                    case .final:
                        if frame.terminator == .return || frame.terminator == .end {
                            sawFinalCompletion = true
                        } else {
                            sawInvalidTerminator = true
                        }
                    case .commentary, .other:
                        sawInvalidTerminator = true
                    }
                case .consumed:
                    break
                }
            }
        }
        for step in parser.finish() {
            guard case .closed(let frame) = step else { continue }
            if frame.header.channel == .final, frame.terminator == .incomplete {
                // XGrammar has already accepted the JSON schema. A separately
                // sampled stop token is transport metadata, not part of the
                // constrained response, and some Harmony checkpoints select
                // `<|call|>` here even though the completed frame is `final`.
                sawFinalCompletion = true
            } else {
                sawInvalidTerminator = true
            }
        }

        guard sawFinalCompletion, !sawInvalidTerminator else { return nil }
        return HarmonyGuidedResponse(
            reasoningText: tokenizer.decode(
                tokenIds: reasoningTokenIDs,
                skipSpecialTokens: false),
            responseText: tokenizer.decode(
                tokenIds: responseTokenIDs,
                skipSpecialTokens: false),
            reasoningTokenCount: reasoningTokenIDs.count)
    }

    package static func decodeToolCall(
        tokenIDs: [Int],
        sampledStopTokenID: Int? = nil,
        grammarTerminated: Bool,
        allowedToolNames: Set<String>,
        tokenizer: any Tokenizer
    ) -> HarmonyGuidedToolCall? {
        guard grammarTerminated,
            var parser = HarmonyFrameParser(tokenizer: tokenizer)
        else {
            return nil
        }

        var reasoningTokenIDs: [Int] = []
        var selectedName: String?
        var argumentsTokenIDs: [Int]?
        var invalid = false

        func inspect(_ step: HarmonyParseStep) {
            guard case .closed(let frame) = step else { return }
            switch frame.header.channel {
            case .analysis:
                if frame.terminator != .end {
                    invalid = true
                }
            case .commentary:
                guard frame.terminator == .call,
                    selectedName == nil,
                    let recipient = frame.header.recipient,
                    recipient.hasPrefix("functions.")
                else {
                    invalid = true
                    return
                }
                let name = String(recipient.dropFirst("functions.".count))
                guard allowedToolNames.contains(name) else {
                    invalid = true
                    return
                }
                selectedName = name
                argumentsTokenIDs = frame.payloadTokenIds
            case .final, .other:
                invalid = true
            }
        }

        for token in tokenIDs {
            for step in parser.push(token) {
                if case .payload(let header, let payloadToken) = step,
                    header.channel == .analysis
                {
                    reasoningTokenIDs.append(payloadToken)
                }
                inspect(step)
            }
        }
        if let sampledStopTokenID {
            for step in parser.push(sampledStopTokenID) {
                inspect(step)
            }
        }
        for step in parser.finish() {
            inspect(step)
        }

        guard !invalid,
            let name = selectedName,
            let argumentsTokenIDs
        else {
            return nil
        }
        let argumentsText = tokenizer.decode(
            tokenIds: argumentsTokenIDs,
            skipSpecialTokens: false
        ).trimmingCharacters(in: .whitespacesAndNewlines)
        return HarmonyGuidedToolCall(
            reasoningText: tokenizer.decode(
                tokenIds: reasoningTokenIDs,
                skipSpecialTokens: false),
            reasoningTokenCount: reasoningTokenIDs.count,
            name: name,
            argumentsText: argumentsText)
    }
}
