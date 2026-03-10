# Anthropic Stream Compatibility Design

## Goal

Fix Claude Code interoperability issues by making the streaming API output stricter Anthropic SSE and by preventing malformed streamed tool-use blocks for code-editing tools.

## Problem

Two user-visible failures point at protocol edges rather than model quality:

1. Some clients report they "can't decode the response".
2. Code-editing tools intermittently fail with "Error editing file".

Current investigation found two concrete risks in the proxy:

- The SSE builder appends a raw `[DONE]` trailer even though the stream is otherwise Anthropic-style SSE.
- Streamed tool-use blocks can start before the real tool name is known, which allows placeholder or partial tool names to reach the client.

## Constraints

- Keep the fix small and protocol-focused.
- Preserve existing message/text/thinking behavior.
- Add regression coverage before production changes.
- Avoid speculative refactors outside the streaming compatibility path.

## Chosen Approach

### 1. Strict Anthropic stream termination

End streams with `message_stop` and connection close only. Remove the OpenAI-style `[DONE]` trailer from Anthropic SSE output.

Why:

- It is the clearest protocol mismatch in the current implementation.
- It aligns the stream with Anthropic clients instead of hybrid behavior.

### 2. Buffered tool-use start

Do not emit `content_block_start` for a streamed tool call until the actual tool name is available. Buffer early argument fragments and flush them only after the tool block starts with the correct name.

Why:

- Claude Code dispatch depends on the tool name being correct.
- Starting a block as `"tool_call"` or a partial name like `"Ed"` can break edit/write tool execution.

### 3. Test-first regression coverage

Add focused tests for:

- clean Anthropic stream termination without `[DONE]`
- split tool names not emitting partial names
- argument chunks arriving before the tool name

## Non-Goals

- Reworking unrelated CLI session parsing
- Changing non-streaming response conversion beyond what tests require
- Large protocol/parser rewrites

## Expected Outcome

Clients that consume Anthropic SSE should stop failing at stream decode boundaries, and code-editing agents should receive stable tool-use payloads with the correct tool names.
