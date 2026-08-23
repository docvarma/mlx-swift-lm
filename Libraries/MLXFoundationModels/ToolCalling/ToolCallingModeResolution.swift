// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import FoundationModels

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
enum ToolCallingModeResolution {
    static func resolve(
        _ mode: GenerationOptions.ToolCallingMode?
    ) -> GenerationOptions.ToolCallingMode {
        mode ?? .allowed
    }

    static func usesAllowedBehavior(
        _ mode: GenerationOptions.ToolCallingMode
    ) -> Bool {
        switch mode.kind {
        case .allowed:
            return true
        case .required, .disallowed:
            return false
        @unknown default:
            return true
        }
    }

    static func enabledToolDefinitions(
        for mode: GenerationOptions.ToolCallingMode,
        from definitions: [Transcript.ToolDefinition],
        responseSchemaPresent: Bool = false
    ) throws -> [Transcript.ToolDefinition] {
        if usesAllowedBehavior(mode) {
            return definitions
        }
        if mode.kind == .disallowed {
            return []
        }
        // A DynamicProfile may consume its final one-shot tool while the
        // enclosing response still carries `.required`. An empty current
        // surface means the session has reached its response-only round; it is
        // not a provider configuration error.
        return definitions
    }
}

#endif
#endif
