import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch
import httpx
import openai
from providers.nvidia_nim import NvidiaNimProvider
from providers.exceptions import APIError
from providers.model_utils import resolve_model_alias
from config.settings import DEFAULT_NIM_MODEL_FALLBACK_ORDER


# Mock data classes
class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockTool:
    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.input_schema = input_schema


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "test-model"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = ["STOP"]
        self.tools = []
        self.extra_body = {}
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_timeout_error() -> openai.APITimeoutError:
    request = httpx.Request("POST", "https://test.api.nvidia.com/v1/chat/completions")
    return openai.APITimeoutError(request=request)


# ---------- Raw httpx mock helpers ----------

def _make_sse_lines(chunks: list[dict]) -> list[str]:
    """Convert a list of OpenAI chunk dicts into SSE lines."""
    lines = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}")
    lines.append("data: [DONE]")
    return lines


def _mock_httpx_stream(lines: list[str], status_code: int = 200):
    """Create a mock async context manager that mimics httpx.stream()."""
    class _MockResponse:
        def __init__(self):
            self.status_code = status_code

        async def aiter_lines(self):
            for line in lines:
                yield line

        async def aread(self):
            return b""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    return _MockResponse()


def _mock_httpx_stream_error(status_code: int, body: str = "error"):
    """Create a mock stream that returns an error status code."""
    class _MockResponse:
        def __init__(self):
            self.status_code = status_code

        async def aread(self):
            return body.encode()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    return _MockResponse()


def _mock_httpx_post(response_json: dict, status_code: int = 200):
    """Create a mock httpx post response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.content = json.dumps(response_json).encode()
    mock_resp.text = json.dumps(response_json)
    return mock_resp


def _make_text_chunk(text: str, finish_reason=None, usage=None):
    """Build a minimal OpenAI streaming chunk dict."""
    chunk = {
        "choices": [{
            "delta": {"content": text, "role": "assistant"},
            "finish_reason": finish_reason,
        }],
    }
    if usage:
        chunk["usage"] = usage
    return chunk


def _make_reasoning_chunk(reasoning: str, finish_reason=None):
    """Build an OpenAI streaming chunk with reasoning_content."""
    return {
        "choices": [{
            "delta": {"reasoning_content": reasoning, "role": "assistant"},
            "finish_reason": finish_reason,
        }],
    }


def _make_tool_chunk(tool_id: str, name: str, arguments: str, index: int = 0, finish_reason=None):
    """Build an OpenAI streaming chunk with a tool call."""
    return {
        "choices": [{
            "delta": {
                "role": "assistant",
                "tool_calls": [{
                    "index": index,
                    "id": tool_id,
                    "function": {"name": name, "arguments": arguments},
                }],
            },
            "finish_reason": finish_reason,
        }],
    }


# ---------- Fixtures ----------

@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""
    with patch("providers.nvidia_nim.client.GlobalRateLimiter") as mock:
        instance = mock.get_instance.return_value
        instance.wait_if_blocked = AsyncMock(return_value=False)
        yield instance


@pytest.mark.asyncio
async def test_init(provider_config):
    """Test provider initialization."""
    with patch("providers.nvidia_nim.client.AsyncOpenAI") as mock_openai:
        provider = NvidiaNimProvider(provider_config)
        assert provider._api_key == "test_key"
        assert provider._base_url == "https://test.api.nvidia.com/v1"
        mock_openai.assert_called_once()


@pytest.mark.asyncio
async def test_build_request_body(nim_provider):
    """Test request body construction."""
    req = MockRequest()
    body = nim_provider._build_request_body(req, stream=True)

    assert body["model"] == "test-model"
    assert body["temperature"] == 0.5
    assert len(body["messages"]) == 2  # System + User
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "System prompt"

    assert "extra_body" in body
    assert "thinking" in body["extra_body"]
    assert body["extra_body"]["thinking"]["type"] == "enabled"


@pytest.mark.asyncio
async def test_build_request_body_applies_hard_max_tokens(nim_provider):
    """Request max_tokens should be clamped by hard_max_tokens."""
    req = MockRequest(max_tokens=50000)
    body = nim_provider._build_request_body(req, stream=False)
    assert body["max_tokens"] == nim_provider._nim_settings.hard_max_tokens


@pytest.mark.asyncio
async def test_stream_response_text(nim_provider):
    """Test streaming text response via raw httpx."""
    req = MockRequest()

    chunks = [
        _make_text_chunk("Hello"),
        _make_text_chunk(" World", finish_reason="stop", usage={"completion_tokens": 10}),
    ]
    sse_lines = _make_sse_lines(chunks)
    mock_stream = _mock_httpx_stream(sse_lines)

    with patch.object(nim_provider._http_client, "stream", return_value=mock_stream):
        events = []
        async for event in nim_provider.stream_response(req):
            events.append(event)

        assert len(events) > 0
        assert "event: message_start" in events[0]

        text_content = ""
        for e in events:
            if "event: content_block_delta" in e and '"text_delta"' in e:
                for line in e.splitlines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "delta" in data and "text" in data["delta"]:
                            text_content += data["delta"]["text"]

        assert "Hello World" in text_content


@pytest.mark.asyncio
async def test_stream_response_thinking_reasoning_content(nim_provider):
    """Test streaming with native reasoning_content via raw httpx."""
    req = MockRequest()

    chunks = [
        _make_reasoning_chunk("Thinking..."),
        _make_text_chunk("Answer", finish_reason="stop"),
    ]
    sse_lines = _make_sse_lines(chunks)
    mock_stream = _mock_httpx_stream(sse_lines)

    with patch.object(nim_provider._http_client, "stream", return_value=mock_stream):
        events = []
        async for event in nim_provider.stream_response(req):
            events.append(event)

        # Check for thinking_delta
        found_thinking = False
        for e in events:
            if "event: content_block_delta" in e and '"thinking_delta"' in e:
                if "Thinking..." in e:
                    found_thinking = True
        assert found_thinking


@pytest.mark.asyncio
async def test_complete_success(nim_provider):
    """Test successful completion via raw httpx."""
    req = MockRequest()

    response_json = {
        "id": "test_id",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello world"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    mock_resp = _mock_httpx_post(response_json)

    with patch.object(nim_provider._http_client, "post", new_callable=AsyncMock, return_value=mock_resp):
        result = await nim_provider.complete(req)
        assert result["id"] == "test_id"
        assert result["choices"][0]["message"]["content"] == "Hello world"


@pytest.mark.asyncio
async def test_complete_error_handling(nim_provider):
    """Test error handling on completion via raw httpx."""
    req = MockRequest()

    # Simulate a 500 error from upstream
    mock_resp = _mock_httpx_post({"error": "API Error"}, status_code=500)

    with patch.object(nim_provider._http_client, "post", new_callable=AsyncMock, return_value=mock_resp):
        with pytest.raises(Exception) as exc:
            await nim_provider.complete(req)
        assert "500" in str(exc.value)


@pytest.mark.asyncio
async def test_complete_model_fallback_chain_uses_next_model(nim_provider):
    """If first model fails, provider should try next fallback model."""
    req = MockRequest()

    # First call: 500 error, second call: success
    fail_resp = _mock_httpx_post({"error": "bad model"}, status_code=500)
    ok_resp = _mock_httpx_post({"id": "ok", "choices": []})

    call_count = 0
    called_models = []

    async def mock_post(url, content=None, headers=None, **kwargs):
        nonlocal call_count
        call_count += 1
        if content:
            body = json.loads(content)
            called_models.append(body.get("model"))
        if call_count == 1:
            return fail_resp
        return ok_resp

    with (
        patch.object(nim_provider._http_client, "post", side_effect=mock_post),
        patch.object(nim_provider, "_candidate_models", return_value=["model-a", "model-b"]),
    ):
        result = await nim_provider.complete(req)

    assert result["id"] == "ok"
    assert call_count == 2
    assert called_models[0] == "model-a"
    assert called_models[1] == "model-b"
    assert nim_provider._sticky_model == "model-b"


@pytest.mark.asyncio
async def test_complete_bad_request_fails_fast_without_fallback(nim_provider):
    """400/422 invalid request errors should fail fast to avoid long retry chains."""
    req = MockRequest()

    fail_resp = _mock_httpx_post({"error": "invalid request payload"}, status_code=400)
    call_count = 0

    async def mock_post(url, content=None, headers=None, **kwargs):
        nonlocal call_count
        call_count += 1
        return fail_resp

    with (
        patch.object(nim_provider._http_client, "post", side_effect=mock_post),
        patch.object(nim_provider, "_candidate_models", return_value=["model-a", "model-b"]),
    ):
        with pytest.raises(Exception):
            await nim_provider.complete(req)

    # Should not keep trying fallback models for the same invalid payload.
    assert call_count == 1


def test_candidate_models_prefers_sticky_working_model(nim_provider):
    with patch(
        "providers.nvidia_nim.client.get_model_fallback_chain",
        return_value=["model-a", "model-b", "model-c"],
    ):
        nim_provider._sticky_model = "model-b"
        assert nim_provider._candidate_models("model-a") == [
            "model-b",
            "model-a",
            "model-c",
        ]


def test_model_alias_normalizes_leading_slash_for_requested_models():
    assert (
        resolve_model_alias("/nvidia/nemotron-3-super-120b-a12b")
        == "nvidia/nemotron-3-super-120b-a12b"
    )
    assert (
        resolve_model_alias("/qwen/qwen3.5-122b-a10b")
        == "qwen/qwen3.5-122b-a10b"
    )


def test_default_fallback_order_includes_requested_models():
    assert "nvidia/nemotron-3-super-120b-a12b" in DEFAULT_NIM_MODEL_FALLBACK_ORDER
    assert "qwen/qwen3.5-122b-a10b" in DEFAULT_NIM_MODEL_FALLBACK_ORDER


@pytest.mark.asyncio
async def test_complete_timeout_opens_model_circuit_breaker_and_uses_fallback(
    provider_config,
):
    req = MockRequest()
    timeout_provider = NvidiaNimProvider(
        provider_config.model_copy(update={"circuit_breaker_threshold": 1})
    )

    ok_resp = _mock_httpx_post({"id": "ok", "choices": []})
    call_count = 0
    called_models = []

    async def mock_post(url, content=None, headers=None, **kwargs):
        nonlocal call_count
        call_count += 1
        if content:
            body = json.loads(content)
            called_models.append(body.get("model"))
        if call_count == 1:
            raise _make_timeout_error()
        return ok_resp

    with (
        patch.object(timeout_provider._http_client, "post", side_effect=mock_post),
        patch.object(
            timeout_provider, "_candidate_models", return_value=["model-a", "model-b"]
        ),
    ):
        result = await timeout_provider.complete(req)

    assert result["id"] == "ok"
    assert call_count == 2
    assert timeout_provider._get_model_cb("model-a").is_open
    assert timeout_provider._sticky_model == "model-b"


@pytest.mark.asyncio
async def test_stream_timeout_opens_model_circuit_breaker_and_uses_fallback(
    provider_config,
):
    req = MockRequest()
    timeout_provider = NvidiaNimProvider(
        provider_config.model_copy(update={"circuit_breaker_threshold": 1})
    )

    chunks = [_make_text_chunk("Recovered", finish_reason="stop")]
    sse_lines = _make_sse_lines(chunks)
    ok_stream = _mock_httpx_stream(sse_lines)
    call_count = 0

    def mock_stream_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _make_timeout_error()
        return ok_stream

    with (
        patch.object(timeout_provider._http_client, "stream", side_effect=mock_stream_call),
        patch.object(
            timeout_provider, "_candidate_models", return_value=["model-a", "model-b"]
        ),
    ):
        events = [event async for event in timeout_provider.stream_response(req)]

    assert call_count == 2
    assert timeout_provider._get_model_cb("model-a").is_open
    assert timeout_provider._sticky_model == "model-b"
    assert any("Recovered" in event for event in events)


@pytest.mark.asyncio
async def test_tool_call_stream(nim_provider):
    """Test streaming tool calls via raw httpx."""
    req = MockRequest()

    chunks = [
        _make_tool_chunk("call_1", "search", '{"q": "test"}'),
        _make_text_chunk("", finish_reason="stop"),
    ]
    sse_lines = _make_sse_lines(chunks)
    mock_stream = _mock_httpx_stream(sse_lines)

    with patch.object(nim_provider._http_client, "stream", return_value=mock_stream):
        events = []
        async for event in nim_provider.stream_response(req):
            events.append(event)

        starts = [
            e for e in events if "event: content_block_start" in e and '"tool_use"' in e
        ]
        assert len(starts) == 1
        assert "search" in starts[0]
