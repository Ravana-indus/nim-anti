# OpenAI-Compatible Endpoints Design

## Goal

Add OpenAI-compatible API support on top of the existing Anthropic-style execution path so OpenAI clients can access the proxy with:

- `POST /v1/chat/completions`
- `POST /v1/responses`
- streaming support on both endpoints
- function tools/tool calls only
- per-request model selection

The existing Anthropic API surface and local tool execution behavior should remain intact.

## Problem

The service currently exposes an Anthropic-style `POST /v1/messages` interface backed by a single internal execution path. That works for Anthropic-compatible clients, but OpenAI-compatible clients cannot use the proxy directly.

Two compatibility requirements shape this change:

1. OpenAI clients need protocol-compatible request and streaming response shapes.
2. The current local tool execution path must remain the single source of truth rather than being reimplemented separately for OpenAI endpoints.

If OpenAI support is added by forking execution logic, tool behavior, error handling, and streaming semantics will drift.

## Constraints

- Preserve existing Anthropic endpoint behavior.
- Reuse the current provider and local tool execution path.
- Support only function tools/tool calls for the first pass.
- Respect the client-supplied `model` per request.
- Continue using configured model fallback behind the requested model unless a stricter mode is added later.
- Keep unsupported OpenAI-only features explicit rather than silently pretending they work.

## Chosen Approach

### 1. Keep the current Anthropic request as the internal canonical format

`MessagesRequest` remains the core request object for execution. New OpenAI-compatible routes translate into that format before calling the provider.

Why:

- The current system already knows how to do request validation, local tool execution, upstream provider calls, streaming, and response conversion.
- Reusing one execution core minimizes behavioral drift.

### 2. Add protocol adapter layers, not a second execution pipeline

Introduce OpenAI-specific adapters in both directions:

- request adapters:
  - OpenAI chat completions -> internal `MessagesRequest`
  - OpenAI responses -> internal `MessagesRequest`
- response adapters:
  - internal result -> chat completions response / chunks
  - internal result -> responses response / SSE events

Why:

- It isolates protocol translation from execution behavior.
- It makes OpenAI compatibility additive rather than invasive.

### 3. Support per-request model override as the primary model for that call

When a client sends `model`, that value becomes the primary target model for that request. Existing fallback behavior still applies behind it.

Why:

- Clients expect model selection to be request-scoped.
- It preserves current resilience behavior without introducing a new global switch.

### 4. Support function tools only

OpenAI tool definitions are accepted only when their type is function-oriented. Hosted tools such as code interpreter or file search are rejected with `400`.

Why:

- Function tools map cleanly onto the current local tool execution flow.
- Hosted tools would require different runtime semantics and should not be faked.

## Route Surface

### `POST /v1/chat/completions`

Supported:

- non-streaming responses
- streaming responses
- `messages`
- `model`
- function `tools`
- tool calls in assistant output
- follow-up messages that include tool results

Not supported in first pass:

- hosted tools
- multimodal features not already supported by the internal request model
- OpenAI fields with no safe internal equivalent

### `POST /v1/responses`

Supported:

- non-streaming responses
- streaming responses
- `input`
- `model`
- function `tools`
- tool call outputs sent back in follow-up requests

Not supported in first pass:

- hosted tools
- OpenAI-specific advanced response features with no internal equivalent

## Internal Architecture

### Request normalization

Create an OpenAI adapter module that:

- validates supported OpenAI payload fields
- converts OpenAI messages or response input items into internal Anthropic-style message content
- maps OpenAI function tools into internal tool definitions
- carries the request `model` through unchanged as the internal primary model for that request

The current Anthropic route remains unchanged except for possible extraction of shared execution helpers.

### Shared execution core

Factor the route logic so all three public endpoints can reuse a common path for:

- request logging
- fast-path optimizations that remain semantically valid
- provider invocation
- token estimation for streaming setup
- error mapping

This shared core should accept a normalized internal request and return either:

- a final internal response object, or
- a stream of internal events suitable for endpoint-specific formatting

### Response formatting

Create separate OpenAI response formatter modules for:

- chat completions JSON
- chat completions streamed chunks
- responses JSON
- responses streamed events

These formatters should operate on normalized internal outputs rather than reaching into provider-specific response details directly.

## Data Mapping

### Chat completions -> internal request

Map:

- `model` -> internal request model
- `messages` -> internal messages
- `tools` -> internal tool schema
- `stream` -> internal stream flag

Tool-related messages must normalize into the same internal conversational shape already expected by the current tool execution path.

### Responses -> internal request

Map:

- `model` -> internal request model
- `input` items -> internal messages/content blocks
- `tools` -> internal tool schema
- `stream` -> internal stream flag

For follow-up tool execution cycles, tool outputs supplied by the client must be transformed into the internal message/content form that the existing execution path already understands.

### Internal output -> chat completions

Return:

- `choices`
- `message`
- `tool_calls` where applicable
- usage fields where available

Streaming should emit delta chunks that reflect text and tool-call argument growth in OpenAI-compatible form.

### Internal output -> responses

Return:

- top-level response metadata
- output items for text and tool calls
- usage fields where available

Streaming should emit protocol-correct response events and a clear terminal event.

## Streaming Design

Streaming support is required for both OpenAI-compatible endpoints.

### Chat completions streaming

Emit `chat.completion.chunk`-style SSE data with:

- assistant role initialization
- incremental text deltas
- incremental tool-call deltas
- a final chunk with finish reason
- terminal `[DONE]`

### Responses streaming

Emit OpenAI-compatible response SSE events that represent:

- response creation/start
- output text deltas
- tool call creation / argument deltas
- completion events

The exact event names should be implemented consistently from one internal event stream adapter rather than hand-built in route functions.

### Internal event strategy

The current provider emits Anthropic-style SSE. For OpenAI endpoints, add a translation layer that consumes normalized internal events and formats them into OpenAI-compatible streamed output.

This translation layer should be endpoint-specific but share as much parsing logic as possible.

## Error Handling

### Unsupported features

Return `400` with precise messages for:

- hosted tool types
- unsupported payload structures
- fields that materially change behavior but have no implementation

### Safe ignored fields

If an OpenAI field is non-semantic and safe to ignore, it may be ignored. This should be conservative and documented in code comments/tests.

### Provider and upstream failures

Continue to use the current provider error mapping path so authentication, rate limiting, overload, and timeout behavior remain consistent across Anthropic and OpenAI routes.

## Model Selection Semantics

For all supported endpoints:

- the client-supplied `model` is the primary upstream model for that request
- the existing configured fallback chain remains active behind that model
- no global active model mutation should occur for a per-request override

This keeps request handling stateless and avoids cross-request leakage.

## Non-Goals

- supporting hosted OpenAI tools
- redesigning the provider abstraction
- replacing the Anthropic-style internal canonical request in this change
- implementing every OpenAI field or compatibility edge case
- changing current Anthropic endpoint semantics beyond extracting shared helpers

## Implementation Plan

1. Add OpenAI request/response models for `chat.completions` and `responses`.
2. Add failing tests for non-streaming and streaming route behavior on both endpoints.
3. Extract shared internal execution helpers from the Anthropic route.
4. Implement request adapters from OpenAI payloads into the internal request shape.
5. Implement non-streaming response adapters for both endpoint families.
6. Implement streaming event adapters for both endpoint families.
7. Add explicit validation for unsupported tool types and unsupported payload fields.
8. Update root or discovery metadata only if needed, without changing existing Anthropic clients.
9. Document new endpoints and per-request model behavior in `README.md`.

## Test Strategy

Add focused coverage for:

- chat completions non-streaming success
- chat completions streaming text
- chat completions streaming tool calls
- responses non-streaming success
- responses streaming text
- responses streaming tool calls
- per-request model override propagation
- follow-up tool result submission mapping
- unsupported hosted tools returning `400`
- provider error mapping consistency on new endpoints

Prefer adapter-level tests for shape conversion and route-level tests for protocol correctness.

## Expected Outcome

After this change, OpenAI-compatible clients should be able to call the proxy through either `chat.completions` or `responses`, stream text and function tool calls, and choose the model per request, while the service continues to use the existing local tool execution path and resilience behavior under the hood.
