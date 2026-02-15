import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch
from providers.nvidia_nim import NvidiaNimProvider
from providers.exceptions import APIError


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
    """Test streaming text response."""
    req = MockRequest()

    # Create mock chunks
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [
        MagicMock(
            delta=MagicMock(content="Hello", reasoning_content=""), finish_reason=None
        )
    ]
    mock_chunk1.usage = None

    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [
        MagicMock(
            delta=MagicMock(content=" World", reasoning_content=""),
            finish_reason="stop",
        )
    ]
    mock_chunk2.usage = MagicMock(completion_tokens=10)

    async def mock_stream():
        yield mock_chunk1
        yield mock_chunk2

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

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
    """Test streaming with native reasoning_content."""
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content="Thinking..."),
            finish_reason=None,
        )
    ]
    mock_chunk.usage = None

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

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
    """Test successful completion."""
    req = MockRequest()

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "id": "test_id",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello world"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_response

        result = await nim_provider.complete(req)
        assert result["id"] == "test_id"
        assert result["choices"][0]["message"]["content"] == "Hello world"


@pytest.mark.asyncio
async def test_complete_error_handling(nim_provider):
    """Test error handling on completion."""
    req = MockRequest()

    import openai

    with patch.object(
        nim_provider._client.chat.completions,
        "create",
        side_effect=openai.APIError("API Error", request=MagicMock(), body=None),
    ):
        with pytest.raises(APIError) as exc:
            await nim_provider.complete(req)
        assert "API Error" in str(exc.value)


@pytest.mark.asyncio
async def test_complete_model_fallback_chain_uses_next_model(nim_provider):
    """If first model fails, provider should try next fallback model."""
    req = MockRequest()

    class _MockFailure(Exception):
        status_code = 500

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {"id": "ok", "choices": []}

    with (
        patch.object(
            nim_provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[_MockFailure("bad model"), mock_response],
        ) as mock_create,
        patch.object(
            nim_provider, "_candidate_models", return_value=["model-a", "model-b"]
        ),
    ):
        result = await nim_provider.complete(req)

    assert result["id"] == "ok"
    assert mock_create.await_count == 2
    first_call = mock_create.await_args_list[0].kwargs
    second_call = mock_create.await_args_list[1].kwargs
    assert first_call["model"] == "model-a"
    assert second_call["model"] == "model-b"
    assert nim_provider._sticky_model == "model-b"


@pytest.mark.asyncio
async def test_complete_bad_request_fails_fast_without_fallback(nim_provider):
    """400/422 invalid request errors should fail fast to avoid long retry chains."""
    req = MockRequest()

    class _BadRequest(Exception):
        status_code = 400

    with (
        patch.object(
            nim_provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=_BadRequest("invalid request payload"),
        ) as mock_create,
        patch.object(
            nim_provider, "_candidate_models", return_value=["model-a", "model-b"]
        ),
    ):
        with pytest.raises(_BadRequest):
            await nim_provider.complete(req)

    # Should not keep trying fallback models for the same invalid payload.
    assert mock_create.await_count == 1


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


@pytest.mark.asyncio
async def test_tool_call_stream(nim_provider):
    """Test streaming tool calls."""
    req = MockRequest()

    # Mock tool call delta
    mock_tc = MagicMock()
    mock_tc.index = 0
    mock_tc.id = "call_1"
    mock_tc.function.name = "search"
    mock_tc.function.arguments = '{"q": "test"}'

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content="", tool_calls=[mock_tc]),
            finish_reason=None,
        )
    ]
    mock_chunk.usage = None

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = []
        async for event in nim_provider.stream_response(req):
            events.append(event)

        starts = [
            e for e in events if "event: content_block_start" in e and '"tool_use"' in e
        ]
        assert len(starts) == 1
        assert "search" in starts[0]
