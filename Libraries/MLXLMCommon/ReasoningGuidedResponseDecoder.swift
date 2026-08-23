// Copyright © 2026 Apple Inc.

package struct ReasoningGuidedResponse: Sendable, Equatable {
    package let reasoningText: String
    package let responseText: String
    package let reasoningTokenCount: Int
}

/// Decodes a completed delimiter-based reasoning + schema response while
/// retaining exact token accounting for the reasoning phase.
package enum ReasoningGuidedResponseDecoder {
    package static func decode(
        rawText: String,
        tokenIDs: [Int],
        config: ReasoningConfig,
        primedInside: Bool,
        responseSeparator: String,
        tokenizer: any Tokenizer
    ) -> ReasoningGuidedResponse? {
        guard let boundary = rawText.range(of: config.endDelimiter) else {
            return nil
        }
        var response = String(rawText[boundary.upperBound...])
        guard response.hasPrefix(responseSeparator) else { return nil }
        response.removeFirst(responseSeparator.count)

        var collector = ReasoningTokenCollector(
            config: config,
            primedInside: primedInside,
            tokenizer: tokenizer)
        var reasoning = ""
        for token in tokenIDs {
            for segment in collector.ingest(token) {
                if case .reasoning(let text) = segment { reasoning += text }
            }
            if collector.shouldStopAfterReasoning { break }
        }
        guard collector.shouldStopAfterReasoning else { return nil }
        return ReasoningGuidedResponse(
            reasoningText: reasoning,
            responseText: response,
            reasoningTokenCount: collector.reasoningTokenIDs.count)
    }
}
