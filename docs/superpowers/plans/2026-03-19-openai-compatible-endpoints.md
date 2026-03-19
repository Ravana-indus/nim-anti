# OpenAI-Compatible Endpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OpenAI-compatible `chat.completions` and `responses` endpoints, including streaming, function tool calls, and per-request model selection, while reusing the existing Anthropic-style execution path.

**Architecture:** Keep `api.models.anthropic.MessagesRequest` as the internal canonical request. Add OpenAI protocol adapters that normalize requests into that shape, extract shared execution helpers from the current Anthropic route, and format internal responses/streams back into OpenAI-compatible JSON and SSE output. Avoid a second execution pipeline so local tool behavior, error mapping, and model fallback stay consistent.

**Tech Stack:** FastAPI, Pydantic, pytest, httpx/openai-compatible upstream provider, SSE translation helpers

---

## File Structure

### Existing files to modify

- `api/routes.py`
  Add `POST /v1/chat/completions` and `POST /v1/responses`, and extract shared route execution helpers so Anthropic and OpenAI endpoints share one execution path.
- `api/models/responses.py`
  Keep Anthropic response models here if needed, but avoid overloading this file with OpenAI request models.
- `api/request_utils.py`
  Reuse only generic helpers; do not make OpenAI route behavior depend on Anthropic-specific shortcuts unless explicitly valid.
- `README.md`
  Document the new endpoints, per-request `model`, streaming, and function-tool-only support.
- `tests/test_api.py`
  Keep the existing Anthropic route tests stable and add only minimal shared-route regressions if needed.

### New files to create

- `api/models/openai.py`
  Pydantic request/response models for the supported subset of `chat.completions` and `responses`.
- `api/openai_adapters.py`
  Conversion helpers from OpenAI payloads to internal `MessagesRequest`, plus non-streaming response conversion helpers back to OpenAI shapes.
- `api/openai_streaming.py`
  Translation of normalized internal/Anthropic SSE events into OpenAI-compatible `chat.completion.chunk` and `responses` SSE event streams.
- `api/execution.py`
  Shared request execution helpers extracted from `api/routes.py` so all endpoint families can reuse provider invocation, token estimation, and error handling.
- `tests/test_openai_adapters.py`
  Focused unit tests for request normalization and non-streaming response formatting.
- `tests/test_openai_streaming.py`
  Focused tests for SSE translation from Anthropic-style events to OpenAI-style events.
- `tests/test_openai_routes.py`
  Route-level tests for both new endpoints, including model override and unsupported tool errors.

### Boundary rules

- `api/models/openai.py` defines only supported OpenAI protocol shapes. Do not leak route logic into model validators.
- `api/openai_adapters.py` owns all protocol mapping between OpenAI and internal request/response structures.
- `api/openai_streaming.py` owns all OpenAI SSE formatting. Do not hand-build OpenAI chunks inside route functions.
- `api/execution.py` owns shared provider execution flow. Route functions should stay thin.

## Task 1: Create OpenAI Request Models And Failing Validation Tests

**Files:**
- Create: `api/models/openai.py`
- Create: `tests/test_openai_adapters.py`

- [ ] **Step 1: Write the failing test for supported chat completions payloads**

```python
def test_chat_completion_request_accepts_function_tools():
    request = ChatCompletionsRequest.model_validate(
        {
            "model": "moonshotai/kimi-k2.5",
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Find data",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "stream": True,
        }
    )

    assert request.model == "moonshotai/kimi-k2.5"
    assert request.tools[0].function.name == "lookup"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_adapters.py::test_chat_completion_request_accepts_function_tools -v`
Expected: FAIL with import error or missing `ChatCompletionsRequest`

- [ ] **Step 3: Add request models for the supported OpenAI subset**

Create `api/models/openai.py` with focused models for:

```python
class OpenAIFunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class OpenAITool(BaseModel):
    type: Literal["function"]
    function: OpenAIFunctionDefinition


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    tools: list[OpenAITool] | None = None
    stream: bool = False


class ResponsesRequest(BaseModel):
    model: str
    input: str | list[OpenAIResponseInputItem]
    tools: list[OpenAITool] | None = None
    stream: bool = False
```

- [ ] **Step 4: Add failing validation tests for unsupported tool types**

```python
def test_responses_request_rejects_hosted_tools():
    with pytest.raises(ValidationError):
        ResponsesRequest.model_validate(
            {
                "model": "moonshotai/kimi-k2.5",
                "input": "ping",
                "tools": [{"type": "file_search"}],
            }
        )
```

- [ ] **Step 5: Run the new model tests**

Run: `uv run pytest tests/test_openai_adapters.py -v`
Expected: PASS for supported payloads, PASS for unsupported-tool rejection

- [ ] **Step 6: Commit**

```bash
git add api/models/openai.py tests/test_openai_adapters.py
git commit -m "feat: add OpenAI request models"
```

## Task 2: Normalize OpenAI Requests Into Internal MessagesRequest

**Files:**
- Modify: `api/models/openai.py`
- Create: `api/openai_adapters.py`
- Modify: `tests/test_openai_adapters.py`

- [ ] **Step 1: Write the failing test for chat completions normalization**

```python
def test_chat_completion_maps_to_internal_messages_request():
    request = ChatCompletionsRequest.model_validate(
        {
            "model": "moonshotai/kimi-k2.5",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                    },
                }
            ],
        }
    )

    normalized = chat_completions_to_messages_request(request)

    assert normalized.model == "moonshotai/kimi-k2.5"
    assert normalized.messages[0].role == "user"
    assert normalized.tools[0].name == "lookup"
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `uv run pytest tests/test_openai_adapters.py::test_chat_completion_maps_to_internal_messages_request -v`
Expected: FAIL because `chat_completions_to_messages_request` does not exist

- [ ] **Step 3: Implement normalization helpers**

Add conversion helpers in `api/openai_adapters.py`:

```python
def chat_completions_to_messages_request(request: ChatCompletionsRequest) -> MessagesRequest:
    return MessagesRequest(
        model=request.model,
        messages=[_map_chat_message(message) for message in request.messages],
        tools=[_map_tool(tool) for tool in request.tools] if request.tools else None,
        stream=request.stream,
    )


def responses_to_messages_request(request: ResponsesRequest) -> MessagesRequest:
    return MessagesRequest(
        model=request.model,
        messages=_map_responses_input(request.input),
        tools=[_map_tool(tool) for tool in request.tools] if request.tools else None,
        stream=request.stream,
    )
```

- [ ] **Step 4: Add failing tests for tool results and per-request model propagation**

```python
def test_responses_tool_output_maps_to_tool_result_block():
    request = ResponsesRequest.model_validate(
        {
            "model": "minimaxai/minimax-m2.5",
            "input": [
                {"type": "function_call_output", "call_id": "call_1", "output": "{\"ok\":true}"}
            ],
        }
    )

    normalized = responses_to_messages_request(request)

    block = normalized.messages[0].content[0]
    assert block.type == "tool_result"
    assert block.tool_use_id == "call_1"
    assert normalized.model == "minimaxai/minimax-m2.5"
```

- [ ] **Step 5: Run all adapter normalization tests**

Run: `uv run pytest tests/test_openai_adapters.py -v`
Expected: PASS for chat and responses normalization, PASS for tool-result mapping, PASS for per-request model propagation

- [ ] **Step 6: Commit**

```bash
git add api/openai_adapters.py api/models/openai.py tests/test_openai_adapters.py
git commit -m "feat: add OpenAI request normalization"
```

## Task 3: Extract Shared Execution Helpers From The Anthropic Route

**Files:**
- Create: `api/execution.py`
- Modify: `api/routes.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_routes_optimizations.py`

- [ ] **Step 1: Write the failing test for shared non-stream execution**

Add a test that proves the Anthropic route still calls the provider through the extracted helper and preserves current response conversion behavior.

```python
def test_create_message_non_stream_uses_shared_executor(client):
    payload = {
        "model": "claude-3-sonnet",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
        "stream": False,
    }

    response = client.post("/v1/messages", json=payload)

    assert response.status_code == 200
    assert response.json()["type"] == "message"
```

- [ ] **Step 2: Run the targeted route tests to establish baseline**

Run: `uv run pytest tests/test_api.py tests/test_routes_optimizations.py -v`
Expected: PASS before refactor

- [ ] **Step 3: Extract shared execution functions**

Create `api/execution.py` and move reusable logic into helpers such as:

```python
async def execute_non_stream(
    request_data: MessagesRequest,
    provider: BaseProvider,
) -> dict:
    response_json = await provider.complete(request_data)
    return response_json


def stream_internal_events(
    request_data: MessagesRequest,
    provider: BaseProvider,
) -> AsyncIterator[str]:
    input_tokens = _fast_token_estimate(
        request_data.messages, request_data.system, request_data.tools
    )
    return provider.stream_response(request_data, input_tokens=input_tokens)
```

Keep Anthropic-only shortcuts in `api/routes.py` unless they are explicitly protocol-agnostic.

- [ ] **Step 4: Update `api/routes.py` to use the shared execution helper without changing endpoint behavior**

- [ ] **Step 5: Re-run the existing Anthropic route tests**

Run: `uv run pytest tests/test_api.py tests/test_routes_optimizations.py -v`
Expected: PASS with no behavior change

- [ ] **Step 6: Commit**

```bash
git add api/execution.py api/routes.py tests/test_api.py tests/test_routes_optimizations.py
git commit -m "refactor: extract shared API execution helpers"
```

## Task 4: Add Non-Streaming OpenAI Route Support

**Files:**
- Modify: `api/routes.py`
- Modify: `api/openai_adapters.py`
- Modify: `api/models/openai.py`
- Create: `tests/test_openai_routes.py`

- [ ] **Step 1: Write failing route tests for chat completions and responses**

```python
def test_chat_completions_non_stream_returns_openai_shape(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "moonshotai/kimi-k2.5",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"


def test_responses_non_stream_returns_openai_shape(client):
    response = client.post(
        "/v1/responses",
        json={"model": "moonshotai/kimi-k2.5", "input": "Hello", "stream": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response"
    assert "output" in body
```

- [ ] **Step 2: Run the targeted route tests to verify they fail**

Run: `uv run pytest tests/test_openai_routes.py::test_chat_completions_non_stream_returns_openai_shape tests/test_openai_routes.py::test_responses_non_stream_returns_openai_shape -v`
Expected: FAIL with `404` or missing route/formatter behavior

- [ ] **Step 3: Implement non-streaming OpenAI response formatters**

Add helpers to `api/openai_adapters.py`:

```python
def messages_response_to_chat_completion(response: dict) -> dict:
    return {
        "id": response["id"],
        "object": "chat.completion",
        "model": response["model"],
        "choices": [_messages_response_choice(response)],
        "usage": _openai_usage(response.get("usage", {})),
    }


def messages_response_to_response_api(response: dict) -> dict:
    return {
        "id": response["id"],
        "object": "response",
        "model": response["model"],
        "output": _response_output_items(response),
        "usage": _responses_usage(response.get("usage", {})),
    }
```

- [ ] **Step 4: Add the two new POST routes and wire them through normalization + shared execution**

- [ ] **Step 5: Add route tests for function tool calls and per-request model override**

```python
def test_chat_completions_preserves_requested_model(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "minimaxai/minimax-m2.5",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert mock_provider.complete.call_args.args[0].model == "minimaxai/minimax-m2.5"
```

- [ ] **Step 6: Run the full non-streaming OpenAI route suite**

Run: `uv run pytest tests/test_openai_routes.py tests/test_openai_adapters.py -v`
Expected: PASS for both endpoints, PASS for model override, PASS for tool-call output formatting

- [ ] **Step 7: Commit**

```bash
git add api/routes.py api/openai_adapters.py api/models/openai.py tests/test_openai_routes.py tests/test_openai_adapters.py
git commit -m "feat: add non-streaming OpenAI endpoints"
```

## Task 5: Translate Anthropic SSE Into OpenAI-Compatible Streaming

**Files:**
- Create: `api/openai_streaming.py`
- Modify: `api/routes.py`
- Create: `tests/test_openai_streaming.py`
- Modify: `tests/test_openai_routes.py`

- [ ] **Step 1: Write failing tests for chat-completions SSE translation**

```python
def test_chat_completion_stream_translates_text_and_done():
    anthropic_events = [
        'event: message_start\\ndata: {"type":"message_start","message":{"id":"msg_1","model":"m","role":"assistant","content":[],"usage":{"input_tokens":1,"output_tokens":1}}}\\n\\n',
        'event: content_block_start\\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\\n\\n',
        'event: content_block_delta\\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\\n\\n',
        'event: message_delta\\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}\\n\\n',
        'event: message_stop\\ndata: {"type":"message_stop"}\\n\\n',
    ]

    chunks = list(iter_openai_chat_chunks(iter(anthropic_events)))

    assert any('"object":"chat.completion.chunk"' in chunk for chunk in chunks)
    assert chunks[-1].strip() == "data: [DONE]"
```

- [ ] **Step 2: Run the streaming tests to verify they fail**

Run: `uv run pytest tests/test_openai_streaming.py::test_chat_completion_stream_translates_text_and_done -v`
Expected: FAIL because the streaming translator does not exist

- [ ] **Step 3: Implement a streaming translator module**

Create `api/openai_streaming.py` with helpers such as:

```python
def iter_openai_chat_chunks(events: Iterable[str]) -> Iterator[str]:
    state = OpenAIChatStreamState()
    for event in events:
        parsed = _parse_anthropic_sse(event)
        yield from _chat_chunks_from_event(parsed, state)
    yield "data: [DONE]\\n\\n"


def iter_openai_response_events(events: Iterable[str]) -> Iterator[str]:
    state = OpenAIResponsesStreamState()
    for event in events:
        parsed = _parse_anthropic_sse(event)
        yield from _responses_events_from_parsed(parsed, state)
```

- [ ] **Step 4: Add failing tests for streamed tool-call deltas and responses SSE**

```python
def test_responses_stream_translates_tool_call_arguments():
    anthropic_events = [
        'event: content_block_start\\ndata: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tool_1","name":"lookup","input":{}}}\\n\\n',
        'event: content_block_delta\\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"q\\":\\"abc\\"}"}}\\n\\n',
    ]

    events = list(iter_openai_response_events(iter(anthropic_events)))

    assert any("response.function_call_arguments.delta" in event for event in events)
```

- [ ] **Step 5: Wire streaming routes through the translator**

In `api/routes.py`, OpenAI routes should:

- normalize the OpenAI request
- obtain the shared internal stream iterator
- pass that iterator into the correct OpenAI SSE translator
- return `StreamingResponse(..., media_type="text/event-stream")`

- [ ] **Step 6: Add end-to-end streaming route tests**

Run: `uv run pytest tests/test_openai_streaming.py tests/test_openai_routes.py -v`
Expected: PASS for chat-completions text streaming, PASS for tool-call delta streaming, PASS for responses SSE terminal events

- [ ] **Step 7: Commit**

```bash
git add api/openai_streaming.py api/routes.py tests/test_openai_streaming.py tests/test_openai_routes.py
git commit -m "feat: add OpenAI streaming adapters"
```

## Task 6: Enforce Unsupported-Feature Errors And Regression Coverage

**Files:**
- Modify: `api/models/openai.py`
- Modify: `api/openai_adapters.py`
- Modify: `tests/test_openai_routes.py`
- Modify: `tests/test_openai_adapters.py`

- [ ] **Step 1: Write failing tests for unsupported hosted tools**

```python
def test_chat_completions_rejects_non_function_tools(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "moonshotai/kimi-k2.5",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "file_search"}],
        },
    )

    assert response.status_code == 400
    assert "function tools" in response.json()["detail"].lower()
```

- [ ] **Step 2: Run the targeted error-handling tests**

Run: `uv run pytest tests/test_openai_routes.py::test_chat_completions_rejects_non_function_tools -v`
Expected: FAIL until explicit `400` behavior is implemented

- [ ] **Step 3: Implement explicit unsupported-feature handling**

Use a dedicated validator or route-level guard so the behavior is deterministic and the error text is precise. Do not silently ignore unsupported tool types or payload structures.

- [ ] **Step 4: Add provider error mapping regression tests for the new routes**

```python
def test_responses_route_maps_rate_limit_error(client):
    mock_provider.complete.side_effect = RateLimitError("Too Many Requests")
    response = client.post(
        "/v1/responses",
        json={"model": "moonshotai/kimi-k2.5", "input": "hello"},
    )

    assert response.status_code == 429
    assert response.json()["error"]["type"] == "rate_limit_error"
```

- [ ] **Step 5: Run the full OpenAI compatibility regression suite**

Run: `uv run pytest tests/test_openai_adapters.py tests/test_openai_routes.py tests/test_openai_streaming.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add api/models/openai.py api/openai_adapters.py tests/test_openai_routes.py tests/test_openai_adapters.py
git commit -m "test: cover OpenAI compatibility edge cases"
```

## Task 7: Document The New Endpoints And Run End-To-End Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the failing docs-minded verification checklist**

Before editing docs, capture the exact behaviors that must already work:

- `POST /v1/chat/completions` non-stream
- `POST /v1/chat/completions` stream
- `POST /v1/responses` non-stream
- `POST /v1/responses` stream
- per-request `model`
- function tools only

- [ ] **Step 2: Update README with concrete examples**

Add examples like:

```bash
curl http://localhost:8082/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "moonshotai/kimi-k2.5",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

and:

```bash
curl http://localhost:8082/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "minimaxai/minimax-m2.5",
    "input": "hello",
    "tools": [{
      "type": "function",
      "function": {
        "name": "lookup",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
      }
    }]
  }'
```

- [ ] **Step 3: Run the full test suite for touched areas**

Run: `uv run pytest tests/test_api.py tests/test_routes_optimizations.py tests/test_openai_adapters.py tests/test_openai_routes.py tests/test_openai_streaming.py tests/test_response_conversion.py tests/test_sse_builder.py -v`
Expected: PASS

- [ ] **Step 4: Run one manual smoke check if the app can be started locally**

Run:

```bash
uv run uvicorn server:app --host 127.0.0.1 --port 8082
```

Then in another shell:

```bash
curl -N http://127.0.0.1:8082/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"moonshotai/kimi-k2.5","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

Expected: SSE chunks followed by `data: [DONE]`

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: add OpenAI-compatible endpoint usage"
```

## Risks To Watch During Execution

- `MessagesRequest` currently performs model normalization for Anthropic aliases. Ensure OpenAI per-request models are passed through as intended and not accidentally rewritten to the server default.
- OpenAI `responses` input items can be more flexible than this first-pass implementation. Keep the supported subset explicit and reject ambiguous structures early.
- Anthropic SSE and OpenAI SSE have different lifecycle semantics. Keep translation stateful and tested rather than doing string replacement.
- Fast-path request shortcuts in `api/routes.py` are written for Anthropic messages. Do not accidentally apply them to OpenAI endpoints unless the semantics are identical.

## Verification Checklist

- Anthropic `/v1/messages` remains unchanged.
- OpenAI `/v1/chat/completions` works for non-stream and stream.
- OpenAI `/v1/responses` works for non-stream and stream.
- Function tool calls round-trip through the existing local tool path.
- Unsupported hosted tools return `400`.
- Request `model` becomes the per-request upstream target without mutating global model state.
