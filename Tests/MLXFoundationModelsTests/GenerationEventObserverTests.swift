// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import Testing

@testable import MLXFoundationModels

/// Proves the behavior-neutral observation hook: when an observer is attached
/// via the task-local, each emit helper both sends to the channel (drained and
/// discarded here) and hands the observer a readable GenerationEvent mirror.
@Suite("GenerationEvent observer")
struct GenerationEventObserverTests {

    /// Runs `body` with an attached observer and a drained channel, returning
    /// every mirrored event the observer received.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func capture(
        _ body: (LanguageModelExecutorGenerationChannel) async -> Void
    ) async -> [MLXLanguageModel.Executor.GenerationEvent] {
        let channel = LanguageModelExecutorGenerationChannel()
        let drain = Task<Void, Never> {
            do { for try await _ in channel {} } catch {}
        }
        let box = EventBox()
        await MLXLanguageModel.Executor.$generationObserver.withValue(
            { box.append($0) },
            operation: { await body(channel) })
        drain.cancel()
        return box.events
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private final class EventBox: @unchecked Sendable {
        private let lock = NSLock()
        private var storage: [MLXLanguageModel.Executor.GenerationEvent] = []
        func append(_ e: MLXLanguageModel.Executor.GenerationEvent) {
            lock.lock()
            storage.append(e)
            lock.unlock()
        }
        var events: [MLXLanguageModel.Executor.GenerationEvent] {
            lock.lock()
            defer { lock.unlock() }
            return storage
        }
    }

    @Test("appendText is mirrored with destination and entryID")
    func mirrorsText() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let events = await capture { channel in
            await MLXLanguageModel.Executor.emit(
                text: "hi", entryID: "e1", destination: .response, into: channel)
        }
        #expect(events.count == 1)
        guard case .appendText(let text, "e1", .response) = events.first else {
            Issue.record("expected .appendText mirror, got \(String(describing: events.first))")
            return
        }
        #expect(text == "hi")
    }

    @Test("toolCall is mirrored with name and arguments")
    func mirrorsToolCall() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let events = await capture { channel in
            await MLXLanguageModel.Executor.emitToolCall(
                id: "id1", name: "get_weather", arguments: "{\"a\":1}",
                entryID: "tc1", into: channel)
        }
        guard case .toolCall(_, let name, let arguments) = events.first else {
            Issue.record("expected .toolCall mirror, got \(String(describing: events.first))")
            return
        }
        #expect(name == "get_weather")
        #expect(arguments == "{\"a\":1}")
    }

    @Test("schema output is delivered as one complete response event")
    func buffersSchemaOutput() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let json = #"{"schemaVersion":1,"claims":[],"kind":"note-draft@1.0.0"}"#
        let events = await capture { channel in
            await MLXLanguageModel.Executor.emitCompletedSchemaText(
                json,
                entryID: "schema-entry",
                into: channel
            )
        }
        let textEvents = events.compactMap { event -> String? in
            guard case .appendText(let text, "schema-entry", .response) = event else {
                return nil
            }
            return text
        }
        #expect(textEvents == [json])
    }

    @Test("usage emission adds prior internal generation work")
    func accumulatesPriorGenerationUsage() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let events = await capture { channel in
            await MLXLanguageModel.Executor.emitUsage(
                input: .init(totalTokenCount: 7, cachedTokenCount: 2),
                output: .init(totalTokenCount: 3, reasoningTokenCount: 1),
                priorUsage: .init(
                    inputTotalTokenCount: 11,
                    inputCachedTokenCount: 4,
                    outputTotalTokenCount: 5,
                    outputReasoningTokenCount: 2),
                entryID: "usage-entry",
                into: channel)
        }

        guard case .updateUsage(let input, let output, "usage-entry") = events.first else {
            Issue.record("expected cumulative .updateUsage mirror")
            return
        }
        #expect(input.totalTokenCount == 18)
        #expect(input.cachedTokenCount == 6)
        #expect(output.totalTokenCount == 8)
        #expect(output.reasoningTokenCount == 3)
    }

    @Test("no observer attached means no crash and events are simply sent")
    func noObserverIsSafe() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // Not inside withValue: generationObserver is nil (shipping behavior).
        let channel = LanguageModelExecutorGenerationChannel()
        let drain = Task<Void, Never> { do { for try await _ in channel {} } catch {} }
        await MLXLanguageModel.Executor.emit(
            text: "x", entryID: nil, destination: .reasoning, into: channel)
        drain.cancel()
        // Reaching here without trapping is the assertion.
    }
}

#endif
