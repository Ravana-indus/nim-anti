# Anthropic Stream Compatibility Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the proxy emit Anthropic-compatible streaming responses and avoid malformed streamed tool-use blocks that break code-editing agents.

**Architecture:** Keep the change local to the SSE builder and streamed tool-call path in the NVIDIA NIM provider. First lock the desired behavior with focused regression tests, then implement the smallest code changes needed to satisfy them.

**Tech Stack:** Python 3.10+, FastAPI, pytest, NVIDIA NIM provider, Anthropic-style SSE

---

### Task 1: Remove the OpenAI-style stream trailer

**Files:**
- Modify: `providers/nvidia_nim/utils/sse_builder.py`
- Modify: `providers/nvidia_nim/client.py`
- Test: `tests/test_sse_builder.py`
- Test: `tests/test_streaming_errors.py`

**Step 1: Write the failing test**

Add assertions that Anthropic streaming ends at `message_stop` and does not append `[DONE]`.

**Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_sse_builder.py tests/test_streaming_errors.py`
Expected: failures on the new `[DONE]` assertions.

**Step 3: Write minimal implementation**

- Remove the raw `[DONE]` trailer from the SSE builder.
- Stop yielding the final trailer event from `providers/nvidia_nim/client.py`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest -q tests/test_sse_builder.py tests/test_streaming_errors.py`
Expected: PASS

**Step 5: Commit**

```bash
git add providers/nvidia_nim/utils/sse_builder.py providers/nvidia_nim/client.py tests/test_sse_builder.py tests/test_streaming_errors.py
git commit -m "fix: remove openai done trailer from anthropic streams"
```

### Task 2: Prevent partial or placeholder tool names

**Files:**
- Modify: `providers/nvidia_nim/utils/sse_builder.py`
- Modify: `providers/nvidia_nim/client.py`
- Test: `tests/test_streaming_errors.py`

**Step 1: Write the failing test**

Add tests covering:

- streamed tool name arriving in multiple chunks
- arguments arriving before the tool name

The assertions should prove that the emitted `tool_use` block starts only once and uses the final tool name rather than a placeholder or partial name.

**Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/test_streaming_errors.py`
Expected: failures showing the current implementation emits an early or incorrect tool name.

**Step 3: Write minimal implementation**

- Buffer tool name fragments until a usable name exists.
- Buffer argument fragments received before the tool block is started.
- Start the tool block once the real name is available, then flush buffered argument fragments in order.
- Keep the existing `Task` interception behavior, but only after a full JSON payload is available.

**Step 4: Run test to verify it passes**

Run: `uv run pytest -q tests/test_streaming_errors.py`
Expected: PASS

**Step 5: Commit**

```bash
git add providers/nvidia_nim/utils/sse_builder.py providers/nvidia_nim/client.py tests/test_streaming_errors.py
git commit -m "fix: buffer streamed tool calls until tool names are stable"
```

### Task 3: Run focused verification

**Files:**
- Test: `tests/test_sse_builder.py`
- Test: `tests/test_streaming_errors.py`
- Test: `tests/test_response_conversion.py`
- Test: `tests/test_nvidia_nim.py`

**Step 1: Run the focused regression suite**

Run: `uv run pytest -q tests/test_sse_builder.py tests/test_streaming_errors.py tests/test_response_conversion.py tests/test_nvidia_nim.py`
Expected: PASS

**Step 2: Sanity-check for protocol regressions**

Inspect any failures for:

- missing `message_stop`
- reordered tool deltas
- broken `Task` interception

**Step 3: Commit**

```bash
git add -A
git commit -m "test: verify anthropic stream compatibility regressions"
```
