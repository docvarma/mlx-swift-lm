// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon

/// Utility that identifies JSON-closing tokens in a tokenizer's vocabulary
/// and produces a logit bias array.
public enum ClosingTokenBias {

    // MARK: - Constants

    private static let stopBias: Float = 300.0
    private static let structuralCloseBias: Float = 200.0
    private static let numericCloseBias: Float = 100.0

    private static let structuralCloseCharacters: Set<String> = ["\"", "}", "]"]
    private static let numericCloseCharacters: Set<String> = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    ]

    // MARK: - Public API

    /// Returns an MLXArray of shape [vocabSize]. Closing tokens get a large
    /// positive value (tiered by priority), all others get 0.0.
    ///
    /// Stop (+300): EOS token
    /// Structural close (+200): `"`, `}`, `]`
    /// Numeric close (+100): single digits `0`-`9`
    ///
    /// Structural closes outrank digits so a model inside a bounded JSON
    /// string closes the string instead of filling the remaining budget with
    /// numeric text. Digits remain biased when the grammar masks structural
    /// tokens out for an integer value.
    public static func compute(tokenizer: any Tokenizer, eosTokenId: Int?) -> MLXArray {
        // Discover vocab size by scanning token IDs
        var vocabSize = 0
        while tokenizer.convertIdToToken(vocabSize) != nil {
            vocabSize += 1
            if vocabSize > 500_000 { break }
        }

        var biases = [Float](repeating: 0.0, count: vocabSize)

        for id in 0 ..< vocabSize {
            if let token = tokenizer.convertIdToToken(id) {
                if structuralCloseCharacters.contains(token) {
                    biases[id] = structuralCloseBias
                } else if numericCloseCharacters.contains(token) {
                    biases[id] = numericCloseBias
                }
            }
        }

        // Stop bias applied last so it overrides any ordinary token class.
        if let eos = eosTokenId, eos >= 0, eos < vocabSize {
            biases[eos] = stopBias
        }

        return MLXArray(biases)
    }
}
