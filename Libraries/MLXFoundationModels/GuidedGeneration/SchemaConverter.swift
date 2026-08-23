// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import os
import FoundationModels
import MLXLMCommon

/// Converts FoundationModels.GenerationSchema to a JSON string for xgrammar.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
enum SchemaConverter {
    private static let logger = Logger(
        subsystem: "com.apple.FoundationModels-MLX",
        category: "SchemaConverter"
    )

    /// Encodes a GenerationSchema to a standard JSON Schema string.
    ///
    /// `GenerationSchema` is itself `Codable`, and its `encode(to:)` internally
    /// calls `jsonSchema()` and encodes the resulting JSON Schema structure.
    /// So `JSONEncoder().encode(schema)` produces the same JSON bytes as
    /// `JSONEncoder().encode(schema.jsonSchema())` would, without needing
    /// to import the framework that owns the `JSONSchema` type.
    static func encodeToJSON(_ schema: GenerationSchema) throws -> String {
        let data = try JSONEncoder().encode(schema)
        guard let jsonString = String(data: data, encoding: .utf8) else {
            throw SchemaConversionError.encodingFailed
        }
        logger.debug("Schema JSON (\(data.count) bytes)")
        return jsonString
    }

    /// Adds the named response format GPT-OSS expects in its developer
    /// message. The same name appears in the Harmony final-frame
    /// `<|constrain|>json` header enforced by ``encodeHarmonyResponseGrammar``.
    static func harmonyResponseFormatInstruction(schemaJSON: String) throws -> String {
        guard let data = schemaJSON.data(using: .utf8) else {
            throw SchemaConversionError.encodingFailed
        }
        _ = try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        return "# Response Formats\n\n## json\n\n\(schemaJSON)"
    }

    /// Builds the established one-pass GPT-OSS structured-output grammar:
    /// optional unconstrained Harmony analysis, followed by a required final
    /// frame whose payload alone is constrained to the response JSON schema.
    ///
    /// The model remains inside its trained Harmony protocol and xgrammar
    /// governs the complete assistant turn; a bare JSON grammar at token zero
    /// would incorrectly suppress the analysis/final channel headers.
    static func encodeHarmonyResponseGrammar(schemaJSON: String) throws -> String {
        guard let data = schemaJSON.data(using: .utf8) else {
            throw SchemaConversionError.encodingFailed
        }
        let schema = try JSONSerialization.jsonObject(
            with: data, options: [.fragmentsAllowed])

        let final: [String: Any] = [
            "type": "tag",
            "begin": "<|channel|>final <|constrain|>json<|message|>",
            "content": [
                "type": "json_schema",
                "json_schema": schema,
            ],
            // XGrammar owns schema completion; Harmony stop tokens remain
            // sampler-owned and must not be required behind the JSON grammar.
            "end": ["<|end|>", ""],
        ]
        let analysis: [String: Any] = [
            "type": "tag",
            "begin": "<|channel|>analysis<|message|>",
            "content": [
                "type": "any_text",
                "excludes": [] as [String],
            ],
            "end": ["<|end|>"],
        ]
        let structuralTag: [String: Any] = [
            "type": "structural_tag",
            "format": [
                "type": "or",
                "elements": [
                    [
                        "type": "sequence",
                        "elements": [
                            analysis,
                            [
                                "type": "const_string",
                                "value": "<|start|>assistant",
                            ],
                            final,
                        ] as [Any],
                    ],
                    final,
                ] as [Any],
            ] as [String: Any],
        ]

        let encoded = try JSONSerialization.data(withJSONObject: structuralTag)
        guard let result = String(data: encoded, encoding: .utf8) else {
            throw SchemaConversionError.encodingFailed
        }
        logger.debug("Harmony response structural-tag JSON (\(encoded.count) bytes)")
        return result
    }

    /// Builds a required, single-call Harmony grammar for GPT-OSS. The tool
    /// recipient commits the selected function before xgrammar opens that
    /// function's argument schema. Optional analysis remains unconstrained,
    /// while the tool call stays in the model's native commentary channel.
    static func encodeHarmonyToolCallingGrammar(
        tools: [Transcript.ToolDefinition]
    ) throws -> String {
        guard !tools.isEmpty else {
            throw SchemaConversionError.noTools
        }

        let encoder = JSONEncoder()
        let toolTags: [[String: Any]] = try tools.map { tool in
            let paramsData = try encoder.encode(tool.parameters)
            let params = try JSONSerialization.jsonObject(with: paramsData)
            let content: [String: Any] = [
                "type": "json_schema",
                "json_schema": params,
            ]

            return [
                "type": "tag",
                "begin":
                    "<|channel|>commentary to=functions.\(tool.name)<|constrain|>json<|message|>",
                "content": content,
                "end": "<|call|>",
            ]
        }
        let toolChoice: [String: Any] = [
            "type": "or",
            "elements": toolTags,
        ]
        let analysis: [String: Any] = [
            "type": "tag",
            "begin": "<|channel|>analysis<|message|>",
            "content": [
                "type": "any_text",
                "excludes": [] as [String],
            ],
            "end": "<|end|>",
        ]
        let structuralTag: [String: Any] = [
            "type": "structural_tag",
            "format": [
                "type": "or",
                "elements": [
                    [
                        "type": "sequence",
                        "elements": [
                            analysis,
                            [
                                "type": "const_string",
                                "value": "<|start|>assistant",
                            ],
                            toolChoice,
                        ] as [Any],
                    ],
                    toolChoice,
                ] as [Any],
            ] as [String: Any],
        ]

        let encoded = try JSONSerialization.data(withJSONObject: structuralTag)
        guard let result = String(data: encoded, encoding: .utf8) else {
            throw SchemaConversionError.encodingFailed
        }
        logger.debug("Harmony tool structural-tag JSON (\(encoded.count) bytes)")
        return result
    }

    /// Builds the required-tool grammar owned by a public MLX model-family
    /// convention. Every supported ``ToolCallFormat`` stays in its native wire
    /// protocol; the application never selects tags or argument syntax itself.
    static func encodeRequiredToolCallingGrammar(
        tools: [Transcript.ToolDefinition],
        format: ToolCallFormat
    ) throws -> String {
        switch format {
        case .json:
            try encodeToolCallingGrammar(tools: tools)
        case .gptOSS:
            try encodeHarmonyToolCallingGrammar(tools: tools)
        case .mistral:
            try encodeJSONArgumentToolGrammar(
                tools: tools,
                begin: { "[TOOL_CALLS]\($0)[ARGS]" },
                end: "")
        case .llama3:
            try encodeJSONArgumentToolGrammar(
                tools: tools,
                begin: { "<|python_tag|>{\"name\": \"\($0)\", \"parameters\": " },
                end: "}")
        case .kimiK2:
            try encodeJSONArgumentToolGrammar(
                tools: tools,
                begin: {
                    "<|tool_calls_section_begin|><|tool_call_begin|>functions.\($0):0<|tool_call_argument_begin|>"
                },
                end: "<|tool_call_end|><|tool_calls_section_end|>")
        case .xmlFunction, .qwen35:
            try encodeXMLParameterToolGrammar(
                tools: tools,
                begin: { "<tool_call><function=\($0)>" },
                end: "</function></tool_call>")
        case .glm4:
            try encodeTaggedParameterToolGrammar(
                tools: tools,
                outerBegin: { "<tool_call>\n\($0)" },
                outerEnd: "\n</tool_call>",
                parameterBegin: { "<arg_key>\($0)</arg_key><arg_value>" },
                parameterEnd: "</arg_value>")
        case .atem:
            try encodeATEMToolGrammar(tools: tools)
        case .minimaxM2:
            try encodeTaggedParameterToolGrammar(
                tools: tools,
                outerBegin: { "<minimax:tool_call><invoke name=\"\($0)\">" },
                outerEnd: "</invoke></minimax:tool_call>",
                parameterBegin: { "<parameter name=\"\($0)\">" },
                parameterEnd: "</parameter>")
        case .gemma:
            try encodeGemmaToolGrammar(
                tools: tools,
                startTag: "<start_function_call>",
                endTag: "<end_function_call>",
                escapeMarker: "<escape>")
        case .gemma4:
            try encodeGemmaToolGrammar(
                tools: tools,
                startTag: "<|tool_call>",
                endTag: "<tool_call|>",
                escapeMarker: "<|\"|>")
        case .lfm2:
            try encodeLFMToolGrammar(tools: tools)
        }
    }

    /// Native formats with a fixed function prefix followed by JSON arguments.
    private static func encodeJSONArgumentToolGrammar(
        tools: [Transcript.ToolDefinition],
        begin: (String) -> String,
        end: String
    ) throws -> String {
        try encodeStructuralToolChoice(
            tools: tools,
            makeTag: { tool, parameters in
                [
                    "type": "tag",
                    "begin": begin(tool.name),
                    "content": [
                        "type": "json_schema",
                        "json_schema": parameters,
                    ],
                    "end": end,
                ]
            })
    }

    /// Qwen/Qwen-derived XML functions use xgrammar's schema-aware public XML
    /// parameter format rather than treating their parameter bodies as JSON.
    private static func encodeXMLParameterToolGrammar(
        tools: [Transcript.ToolDefinition],
        begin: (String) -> String,
        end: String
    ) throws -> String {
        try encodeStructuralToolChoice(
            tools: tools,
            makeTag: { tool, parameters in
                [
                    "type": "tag",
                    "begin": begin(tool.name),
                    "content": [
                        "type": "qwen_xml_parameter",
                        "json_schema": parameters,
                    ],
                    "end": end,
                ]
            })
    }

    /// Native XML-like formats whose parameter values are not JSON strings.
    /// Each parameter remains schema-authorized by name and is parsed back
    /// through the format's public `ToolCallParser` before execution.
    private static func encodeTaggedParameterToolGrammar(
        tools: [Transcript.ToolDefinition],
        outerBegin: (String) -> String,
        outerEnd: String,
        parameterBegin: (String) -> String,
        parameterEnd: String
    ) throws -> String {
        try encodeStructuralToolChoice(
            tools: tools,
            makeTag: { tool, parameters in
                let parameterTags = nativeParameterElements(
                    parameters: parameters,
                    parameterBegin: parameterBegin,
                    parameterEnd: parameterEnd)
                return [
                    "type": "tag",
                    "begin": outerBegin(tool.name),
                    "content": sequenceOrEmpty(parameterTags),
                    "end": outerEnd,
                ]
            })
    }

    /// Muse Glimmer tool calls are ATEM payloads inside the Onyx response
    /// protocol. The chat template primes the first generation after
    /// `<|start|>assistant`; a generation following a private-reasoning frame
    /// starts a fresh assistant frame. Accept precisely those two native entry
    /// states and require the tool commit token in both.
    private static func encodeATEMToolGrammar(
        tools: [Transcript.ToolDefinition]
    ) throws -> String {
        try encodeStructuralToolChoices(
            tools: tools,
            makeTags: { tool, parameters in
                let parameterTags = nativeParameterElements(
                    parameters: parameters,
                    parameterBegin: { "<atem:parameter name=\"\($0)\">" },
                    parameterEnd: "</atem:parameter>")
                let payloadBegin =
                    "<atem:function_calls><atem:invoke name=\"\(tool.name)\">"
                let payloadEnd = "</atem:invoke></atem:function_calls><|eot|>"
                return [
                    [
                        "type": "tag",
                        "begin": " to=\(tool.name)<|message|>\(payloadBegin)",
                        "content": sequenceOrEmpty(parameterTags),
                        "end": payloadEnd,
                    ],
                    [
                        "type": "tag",
                        "begin": "<|start|>assistant to=\(tool.name)<|message|>\(payloadBegin)",
                        "content": sequenceOrEmpty(parameterTags),
                        "end": payloadEnd,
                    ],
                ]
            })
    }

    private static func encodeGemmaToolGrammar(
        tools: [Transcript.ToolDefinition],
        startTag: String,
        endTag: String,
        escapeMarker: String
    ) throws -> String {
        try encodeStructuralToolChoice(
            tools: tools,
            makeTag: { tool, parameters in
                let required = requiredPropertyNames(in: parameters)
                let properties = parameters["properties"] as? [String: Any] ?? [:]
                let values: [[String: Any]] = required.enumerated().map { index, name in
                    let separator = index == 0 ? "" : ","
                    let schema = propertySchema(named: name, in: parameters, properties: properties)
                    if isStringOnlySchema(schema) {
                        return [
                            "type": "tag",
                            "begin": "\(separator)\(name):\(escapeMarker)",
                            "content": [
                                "type": "any_text"
                            ],
                            "end": escapeMarker,
                        ]
                    }
                    return [
                        "type": "tag",
                        "begin": "\(separator)\(name):",
                        "content": [
                            "type": "json_schema",
                            "json_schema": schema,
                        ],
                        "end": "",
                    ]
                }
                return [
                    "type": "tag",
                    "begin": "\(startTag)call:\(tool.name){",
                    "content": sequenceOrEmpty(values),
                    "end": "}\(endTag)",
                ]
            })
    }

    private static func encodeLFMToolGrammar(
        tools: [Transcript.ToolDefinition]
    ) throws -> String {
        try encodeStructuralToolChoice(
            tools: tools,
            makeTag: { tool, parameters in
                let required = requiredPropertyNames(in: parameters)
                let properties = parameters["properties"] as? [String: Any] ?? [:]
                let values: [[String: Any]] = required.enumerated().map { index, name in
                    let separator = index == 0 ? "" : ", "
                    let schema = propertySchema(named: name, in: parameters, properties: properties)
                    if isStringOnlySchema(schema) {
                        return [
                            "type": "tag",
                            "begin": "\(separator)\(name)='",
                            "content": [
                                "type": "any_text"
                            ],
                            "end": "'",
                        ]
                    }
                    return [
                        "type": "tag",
                        "begin": "\(separator)\(name)=",
                        "content": [
                            "type": "json_schema",
                            "json_schema": schema,
                        ],
                        "end": "",
                    ]
                }
                return [
                    "type": "tag",
                    "begin": "<|tool_call_start|>[\(tool.name)(",
                    "content": sequenceOrEmpty(values),
                    "end": ")]<|tool_call_end|>",
                ]
            })
    }

    private static func encodeStructuralToolChoice(
        tools: [Transcript.ToolDefinition],
        makeTag: (Transcript.ToolDefinition, [String: Any]) throws -> [String: Any]
    ) throws -> String {
        try encodeStructuralToolChoices(
            tools: tools,
            makeTags: { tool, parameters in [try makeTag(tool, parameters)] })
    }

    private static func encodeStructuralToolChoices(
        tools: [Transcript.ToolDefinition],
        makeTags: (Transcript.ToolDefinition, [String: Any]) throws -> [[String: Any]]
    ) throws -> String {
        guard !tools.isEmpty else { throw SchemaConversionError.noTools }
        let encoder = JSONEncoder()
        let tags = try tools.flatMap { tool in
            let data = try encoder.encode(tool.parameters)
            guard let parameters = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { throw SchemaConversionError.encodingFailed }
            return try makeTags(tool, parameters)
        }
        let structuralTag: [String: Any] = [
            "type": "structural_tag",
            "format": [
                "type": "or",
                "elements": tags,
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: structuralTag)
        guard let result = String(data: data, encoding: .utf8) else {
            throw SchemaConversionError.encodingFailed
        }
        logger.debug(
            "Native tool structural-tag JSON (\(data.count) bytes, \(tools.count) tools, \(tags.count) variants)"
        )
        return result
    }

    private static func sequenceOrEmpty(_ elements: [[String: Any]]) -> [String: Any] {
        guard !elements.isEmpty else {
            return ["type": "const_string", "value": ""]
        }
        return ["type": "sequence", "elements": elements]
    }

    private static func optional(_ element: [String: Any]) -> [String: Any] {
        [
            "type": "or",
            "elements": [
                element,
                ["type": "const_string", "value": ""],
            ],
        ]
    }

    private static func nativeParameterElements(
        parameters: [String: Any],
        parameterBegin: (String) -> String,
        parameterEnd: String
    ) -> [[String: Any]] {
        let required = Set(requiredPropertyNames(in: parameters))
        let properties = parameters["properties"] as? [String: Any] ?? [:]
        return properties.keys.sorted().map { name in
            let schema = propertySchema(
                named: name, in: parameters, properties: properties)
            let content: [String: Any]
            if isStringOnlySchema(schema) {
                content = ["type": "any_text"]
            } else {
                content = ["type": "json_schema", "json_schema": schema]
            }
            let tag: [String: Any] = [
                "type": "tag",
                "begin": parameterBegin(name),
                "content": content,
                "end": parameterEnd,
            ]
            return required.contains(name) ? tag : optional(tag)
        }
    }

    private static func requiredPropertyNames(in schema: [String: Any]) -> [String] {
        (schema["required"] as? [String] ?? []).sorted()
    }

    private static func propertySchema(
        named name: String,
        in root: [String: Any],
        properties: [String: Any]
    ) -> [String: Any] {
        var schema = properties[name] as? [String: Any] ?? [:]
        if let definitions = root["$defs"] {
            schema["$defs"] = definitions
        }
        return schema
    }

    private static func isStringOnlySchema(_ schema: [String: Any]) -> Bool {
        if schema["type"] as? String == "string" { return true }
        guard let types = schema["type"] as? [String] else { return false }
        return Set(types).subtracting(["null"]) == ["string"]
    }

    /// Muse Glimmer structured responses are JSON payloads inside an Onyx
    /// assistant-to-user frame. As with required tools, support both the
    /// initially primed assistant header and a fresh frame following private
    /// reasoning, and require the native `<|eot|>` commit.
    static func encodeOnyxResponseGrammar(schemaJSON: String) throws -> String {
        guard let data = schemaJSON.data(using: .utf8) else {
            throw SchemaConversionError.encodingFailed
        }
        let schema = try JSONSerialization.jsonObject(
            with: data, options: [.fragmentsAllowed])
        let variants: [[String: Any]] = [
            [
                "type": "tag",
                "begin": " to=user<|message|>",
                "content": ["type": "json_schema", "json_schema": schema],
                "end": "<|eot|>",
            ],
            [
                "type": "tag",
                "begin": "<|start|>assistant to=user<|message|>",
                "content": ["type": "json_schema", "json_schema": schema],
                "end": "<|eot|>",
            ],
        ]
        let structuralTag: [String: Any] = [
            "type": "structural_tag",
            "format": ["type": "or", "elements": variants],
        ]
        let encoded = try JSONSerialization.data(withJSONObject: structuralTag)
        guard let result = String(data: encoded, encoding: .utf8) else {
            throw SchemaConversionError.encodingFailed
        }
        logger.debug("Onyx response structural-tag JSON (\(encoded.count) bytes)")
        return result
    }

    /// Builds Qwen3's native thinking-to-schema sequence. Its chat template
    /// normally primes the assistant inside `<think>`, so the reasoning tag has
    /// an empty `begin`; callers may supply the explicit opening marker for a
    /// non-primed compatible prompt.
    static func encodeQwen3ResponseGrammar(
        schemaJSON: String,
        primedInsideReasoning: Bool
    ) throws -> String {
        guard let data = schemaJSON.data(using: .utf8) else {
            throw SchemaConversionError.encodingFailed
        }
        let schema = try JSONSerialization.jsonObject(
            with: data, options: [.fragmentsAllowed])
        let structuralTag: [String: Any] = [
            "type": "structural_tag",
            "format": [
                "type": "sequence",
                "elements": [
                    [
                        "type": "tag",
                        "begin": primedInsideReasoning ? "" : "<think>",
                        "content": [
                            "type": "any_text",
                            "excludes": [] as [String],
                        ],
                        "end": "</think>",
                    ],
                    [
                        "type": "const_string",
                        "value": "\n\n",
                    ],
                    [
                        "type": "json_schema",
                        "json_schema": schema,
                    ],
                ] as [Any],
            ] as [String: Any],
        ]
        let encoded = try JSONSerialization.data(withJSONObject: structuralTag)
        guard let result = String(data: encoded, encoding: .utf8) else {
            throw SchemaConversionError.encodingFailed
        }
        logger.debug("Qwen3 response structural-tag JSON (\(encoded.count) bytes)")
        return result
    }

    /// Builds the JSON Schema describing the tool-calling envelope itself:
    /// a `oneOf` over each supplied tool's `{name, arguments}` shape.
    ///
    /// Shape:
    /// ```
    /// {
    ///   "oneOf": [
    ///     {
    ///       "type": "object",
    ///       "required": ["name", "arguments"],
    ///       "additionalProperties": false,
    ///       "properties": {
    ///         "name": {"const": "<tool name>"},
    ///         "arguments": <tool's parameters schema>
    ///       }
    ///     },
    ///     ...
    ///   ],
    ///   "$defs": { "<tool name>__<def name>": ... }
    /// }
    /// ```
    ///
    /// If a tool's parameters schema carries `$defs` (named sub-schemas such
    /// as nested `@Generable` types), they are hoisted to the envelope root
    /// under per-tool namespaced keys, with that tool's `$ref`s rewritten to
    /// match — JSON Pointers resolve from the document root, so defs left
    /// nested inside `arguments` would leave every ref dangling.
    ///
    /// This is the *inner* schema -- it describes one tool call JSON object.
    /// For end-to-end grammar generation that also encodes the model's native
    /// tool-call wrapper (e.g. Qwen's `<tool_call>...</tool_call>`), see
    /// `encodeToolCallingGrammar(tools:)`.
    ///
    /// Requires a non-empty tool list.
    static func encodeToolCallingEnvelopeJSON(
        tools: [Transcript.ToolDefinition]
    ) throws -> String {
        let envelope = try toolCallingEnvelopeObject(tools: tools)
        let data = try JSONSerialization.data(withJSONObject: envelope)
        guard let jsonString = String(data: data, encoding: .utf8) else {
            throw SchemaConversionError.encodingFailed
        }
        logger.debug(
            "Tool-calling envelope JSON (\(data.count) bytes, \(tools.count) tools)")
        return jsonString
    }

    /// Builds an xgrammar structural-tag JSON that constrains the model
    /// to emit a tool call either wrapped in Qwen-style
    /// `<tool_call>...</tool_call>` delimiters or as bare JSON.
    ///
    /// Each tool is its own `tag` whose `begin` is the literal prefix
    /// `{"name": "<tool>", "arguments": ` and whose `content` is that
    /// tool's parameter schema, closed by an `end` of `}`. Structural-tag
    /// shape:
    /// ```json
    /// {
    ///   "type": "structural_tag",
    ///   "format": {
    ///     "type": "or",
    ///     "elements": [
    ///       {
    ///         "type": "tag",
    ///         "begin": "<tool_call>\n",
    ///         "content": <per-tool or>,
    ///         "end": ["\n</tool_call>"]
    ///       },
    ///       <per-tool or>
    ///     ]
    ///   }
    /// }
    /// ```
    /// where `<per-tool or>` is an `or` over one `tag` per tool:
    /// ```json
    /// {
    ///   "type": "tag",
    ///   "begin": "{\"name\": \"set_flashlight\", \"arguments\": ",
    ///   "content": { "type": "json_schema", "json_schema": <tool params> },
    ///   "end": ["}"]
    /// }
    /// ```
    ///
    /// **Why per-tool tags instead of one `oneOf` json_schema.** The
    /// earlier shape embedded a single `{oneOf: [{name, arguments}, …]}`
    /// json_schema in each arm. The structural-tag path compiles that
    /// embedded schema with xgrammar's default (non-strict) property
    /// ordering, so greedy decoding could open `"arguments"` before
    /// `"name"` and dive into an unbounded free-text field before ever
    /// committing to a tool -- producing a nameless, unparseable buffer
    /// that ran the token budget dry (observed: Qwen filling `response`
    /// with `"1234567890…"`). Making the name a literal tag prefix forces
    /// the model to commit to a specific tool first, then fill only that
    /// tool's arguments. It also removes the JSON whitespace wiggle room
    /// around the `name`/`arguments` keys that open-source models tend to
    /// exploit into long whitespace runs.
    ///
    /// Accepting both wrapped and bare arms lets the model stay in its
    /// trained distribution -- Qwen-family models overwhelmingly prefer
    /// the wrapped form; the bare arm is a defensive fallback for models
    /// trained on raw JSON.
    ///
    /// **Why structural tag over hand-rolled GBNF.** Each tool's
    /// `arguments` is a JSON object whose shape depends on the tool's
    /// `parameters` schema. Emitting GBNF for it would require a
    /// Swift-side JSON-schema-to-GBNF compiler -- reinventing what
    /// xgrammar's `Grammar::FromJSONSchema` already does in C++.
    /// Structural tag composes the fixed dispatch prefix with the
    /// per-tool json_schema and lets xgrammar compile the embedded
    /// schema the same way the plain `jsonSchema:` path does.
    ///
    /// Requires a non-empty tool list.
    static func encodeToolCallingGrammar(
        tools: [Transcript.ToolDefinition]
    ) throws -> String {
        guard !tools.isEmpty else {
            throw SchemaConversionError.noTools
        }

        let encoder = JSONEncoder()
        // One tag per tool. `begin` fixes `{"name": "<tool>", "arguments": `
        // so the tool name is committed before the arguments schema opens;
        // `content` is the tool's parameter schema; `end` closes the object.
        let toolTags: [[String: Any]] = try tools.map { tool in
            let paramsData = try encoder.encode(tool.parameters)
            let paramsAny = try JSONSerialization.jsonObject(with: paramsData)
            let nameData = try JSONSerialization.data(
                withJSONObject: tool.name, options: [.fragmentsAllowed])
            let nameJSON = String(data: nameData, encoding: .utf8) ?? "\"\(tool.name)\""
            return [
                "type": "tag",
                "begin": "{\"name\": \(nameJSON), \"arguments\": ",
                "content": [
                    "type": "json_schema",
                    "json_schema": paramsAny,
                ],
                "end": ["}"],
            ]
        }
        let perToolOr: [String: Any] = [
            "type": "or",
            "elements": toolTags,
        ]

        let structuralTag: [String: Any] = [
            "type": "structural_tag",
            "format": [
                "type": "or",
                "elements": [
                    [
                        "type": "tag",
                        "begin": "<tool_call>\n",
                        "content": perToolOr,
                        "end": ["\n</tool_call>"],
                    ],
                    perToolOr,
                ] as [Any],
            ] as [String: Any],
        ]

        let data = try JSONSerialization.data(withJSONObject: structuralTag)
        guard let jsonString = String(data: data, encoding: .utf8) else {
            throw SchemaConversionError.encodingFailed
        }
        logger.debug(
            "Tool-calling structural-tag JSON (\(data.count) bytes, \(tools.count) tools)"
        )
        return jsonString
    }

    private static func toolCallingEnvelopeObject(
        tools: [Transcript.ToolDefinition]
    ) throws -> [String: Any] {
        guard !tools.isEmpty else {
            throw SchemaConversionError.noTools
        }

        let encoder = JSONEncoder()
        // `GenerationSchema` serializes named sub-schemas (e.g. a nested
        // `@Generable` type, or a named `DynamicGenerationSchema`) as
        // root-level `$defs` plus root-anchored `"$ref": "#/$defs/..."`
        // pointers. Embedding a tool's schema as a nested object under
        // `oneOf[i].properties.arguments` buries its `$defs` inside
        // `arguments` while the refs stay anchored to the document root —
        // and xgrammar resolves JSON Pointers from the document root, so
        // every ref dangles and grammar compilation hard-fails
        // ("Cannot find field $defs in {\"oneOf\": ...",
        // json_schema_converter.cc). Hoist each tool's `$defs` to the
        // envelope root instead, namespacing keys per tool
        // (`<tool>__<def>`) so same-named defs across tools cannot collide.
        var hoistedDefs: [String: Any] = [:]
        let oneOf: [[String: Any]] = try tools.map { tool in
            // Round-trip the tool's parameters through JSONSerialization so we
            // can embed it as a nested object in the envelope we assemble via
            // JSONSerialization.data(withJSONObject:). Cheap: schemas are small.
            let paramsData = try encoder.encode(tool.parameters)
            var paramsAny = rewriteDefsRefs(
                in: try JSONSerialization.jsonObject(with: paramsData),
                toolName: tool.name
            )
            if var paramsObj = paramsAny as? [String: Any] {
                if let defs = paramsObj.removeValue(forKey: "$defs") as? [String: Any] {
                    for (key, value) in defs {
                        hoistedDefs["\(tool.name)__\(key)"] = value
                    }
                }
                paramsAny = paramsObj
            }
            return [
                "type": "object",
                "required": ["name", "arguments"],
                "additionalProperties": false,
                "properties": [
                    "name": ["const": tool.name],
                    "arguments": paramsAny,
                ],
            ]
        }
        var envelope: [String: Any] = ["oneOf": oneOf]
        if !hoistedDefs.isEmpty {
            envelope["$defs"] = hoistedDefs
        }
        return envelope
    }

    /// Rewrites every `"$ref": "#/$defs/<name>"` in a parsed schema tree to
    /// the per-tool namespaced key (`#/$defs/<tool>__<name>`).
    ///
    /// The rewrite is structure-aware: only the string value directly under a
    /// `$ref` key is touched, so other strings that merely mention the
    /// pointer text — a `description`, `const`, `enum` entry, `default`, or
    /// `pattern` containing "#/$defs/" — survive verbatim. Runs before the
    /// `$defs` are hoisted out, and recurses through the whole tree because
    /// refs can appear anywhere, including inside other `$defs` bodies.
    private static func rewriteDefsRefs(in value: Any, toolName: String) -> Any {
        switch value {
        case let object as [String: Any]:
            var result: [String: Any] = [:]
            result.reserveCapacity(object.count)
            for (key, nested) in object {
                if key == "$ref", let ref = nested as? String, ref.hasPrefix("#/$defs/") {
                    result[key] = "#/$defs/\(toolName)__" + ref.dropFirst("#/$defs/".count)
                } else {
                    result[key] = rewriteDefsRefs(in: nested, toolName: toolName)
                }
            }
            return result
        case let array as [Any]:
            return array.map { rewriteDefsRefs(in: $0, toolName: toolName) }
        default:
            return value
        }
    }

    enum SchemaConversionError: Error {
        case encodingFailed
        case noTools
    }
}

#endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
