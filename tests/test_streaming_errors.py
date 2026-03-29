"""Tests for streaming error handling in providers/nvidia_nim/client.py."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from providers.nvidia_nim import NvidiaNimProvider
from providers.base import ProviderConfig
from config.nim import NimSettings


# ---------- Raw httpx mock helpers ----------

def _make_sse_lines(chunks: list[dict]) -> list[str]:
    """Convert chunk dicts into SSE lines."""
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


def _mock_httpx_stream_with_mid_error(lines_before_error: list[str], error: Exception):
    """Create a mock stream that yields some lines then raises an error."""
    class _MockResponse:
        def __init__(self):
            self.status_code = 200

        async def aiter_lines(self):
            for line in lines_before_error:
                yield line
            raise error

        async def aread(self):
            return b""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    return _MockResponse()


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


def _make_empty_chunk(finish_reason="stop"):
    """Build a chunk with no content."""
    return {
        "choices": [{
            "delta": {"role": "assistant"},
            "finish_reason": finish_reason,
        }],
    }


# ---------- Provider/Request helpers ----------

def _make_provider():
    """Create a provider instance for testing."""
    config = ProviderConfig(
        api_key="test_key",
        base_url="https://test.api.nvidia.com/v1",
        rate_limit=10,
        rate_window=60,
        nim_settings=NimSettings(),
    )
    return NvidiaNimProvider(config)


def _make_request(model="test-model", stream=True):
    """Create a mock request with all fields build_request_body needs."""
    req = MagicMock()
    req.model = model
    req.stream = stream
    req.messages = []
    req.system = None
    req.tools = None
    req.tool_choice = None
    req.metadata = None
    req.max_tokens = 4096
    req.temperature = None
    req.top_p = None
    req.top_k = None
    req.stop_sequences = None
    req.extra_body = None
    req.thinking = None
    return req


async def _collect_stream(provider, request):
    """Collect all SSE events from a stream."""
    events = []
    async for event in provider.stream_response(request):
        events.append(event)
    return events


class TestStreamingExceptionHandling:
    """Tests for error paths during stream_response."""

    @pytest.mark.asyncio
    async def test_api_error_emits_sse_error_event(self):
        """When API raises during streaming, SSE error event is emitted."""
        provider = _make_provider()
        request = _make_request()

        def mock_stream_raise(*args, **kwargs):
            raise RuntimeError("API failed")

        with patch.object(
            provider._http_client,
            "stream",
            side_effect=mock_stream_raise,
        ):
            with patch.object(
                provider._global_rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ):
                events = await _collect_stream(provider, request)

        # Should have message_start, error text block, close blocks, message_delta, message_stop
        event_text = "".join(events)
        assert "message_start" in event_text
        assert "API failed" in event_text
        assert "message_stop" in event_text
        assert "[DONE]" not in event_text

    @pytest.mark.asyncio
    async def test_error_after_partial_content(self):
        """Error after partial content: blocks closed, error emitted."""
        provider = _make_provider()
        request = _make_request()

        partial_lines = [f"data: {json.dumps(_make_text_chunk('Hello '))}"]
        mock_stream = _mock_httpx_stream_with_mid_error(
            partial_lines, RuntimeError("Connection lost")
        )

        with patch.object(
            provider._http_client,
            "stream",
            return_value=mock_stream,
        ):
            with patch.object(
                provider._global_rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ):
                events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert "Hello" in event_text
        assert "Connection lost" in event_text
        assert "message_stop" in event_text

    @pytest.mark.asyncio
    async def test_empty_response_gets_space(self):
        """Empty response with no text/tools gets a single space text block."""
        provider = _make_provider()
        request = _make_request()

        chunks = [_make_empty_chunk()]
        sse_lines = _make_sse_lines(chunks)
        mock_stream = _mock_httpx_stream(sse_lines)

        with patch.object(
            provider._http_client,
            "stream",
            return_value=mock_stream,
        ):
            with patch.object(
                provider._global_rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ):
                events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert '"text_delta"' in event_text
        assert "message_stop" in event_text

    @pytest.mark.asyncio
    async def test_stream_with_thinking_content(self):
        """Thinking content via think tags is emitted as thinking blocks."""
        provider = _make_provider()
        request = _make_request()

        chunks = [
            _make_text_chunk("<think>reasoning</think>answer"),
            _make_empty_chunk(),
        ]
        sse_lines = _make_sse_lines(chunks)
        mock_stream = _mock_httpx_stream(sse_lines)

        with patch.object(
            provider._http_client,
            "stream",
            return_value=mock_stream,
        ):
            with patch.object(
                provider._global_rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ):
                events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert "thinking" in event_text
        assert "reasoning" in event_text
        assert "answer" in event_text

    @pytest.mark.asyncio
    async def test_stream_with_reasoning_content_field(self):
        """reasoning_content delta field is emitted as thinking block."""
        provider = _make_provider()
        request = _make_request()

        chunks = [
            _make_reasoning_chunk("I think..."),
            _make_text_chunk("The answer"),
            _make_empty_chunk(),
        ]
        sse_lines = _make_sse_lines(chunks)
        mock_stream = _mock_httpx_stream(sse_lines)

        with patch.object(
            provider._http_client,
            "stream",
            return_value=mock_stream,
        ):
            with patch.object(
                provider._global_rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=False,
            ):
                events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert "thinking_delta" in event_text
        assert "I think..." in event_text
        assert "The answer" in event_text

    @pytest.mark.asyncio
    async def test_stream_rate_limited_shows_notice(self):
        """When globally rate limited, a notice is shown before stream starts."""
        provider = _make_provider()
        request = _make_request()

        chunks = [
            _make_text_chunk("Response"),
            _make_empty_chunk(),
        ]
        sse_lines = _make_sse_lines(chunks)
        mock_stream = _mock_httpx_stream(sse_lines)

        with patch.object(
            provider._http_client,
            "stream",
            return_value=mock_stream,
        ):
            with patch.object(
                provider._global_rate_limiter,
                "wait_if_blocked",
                new_callable=AsyncMock,
                return_value=True,
            ):
                events = await _collect_stream(provider, request)

        event_text = "".join(events)
        assert "rate limit" in event_text.lower()
        assert "Response" in event_text


class TestProcessToolCall:
    """Tests for _process_tool_call method."""

    def test_tool_call_with_id(self):
        """Tool call with id starts a tool block."""
        provider = _make_provider()
        from providers.nvidia_nim.utils import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
        tc = {
            "index": 0,
            "id": "call_123",
            "function": {"name": "search", "arguments": '{"q": "test"}'},
        }
        events = list(provider._process_tool_call(tc, sse))
        event_text = "".join(events)
        assert "tool_use" in event_text
        assert "search" in event_text
        assert "call_123" in event_text

    def test_tool_call_without_id_generates_uuid(self):
        """Tool call without id generates a uuid-based id."""
        provider = _make_provider()
        from providers.nvidia_nim.utils import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
        tc = {
            "index": 0,
            "id": None,
            "function": {"name": "test", "arguments": "{}"},
        }
        events = list(provider._process_tool_call(tc, sse))
        event_text = "".join(events)
        assert "tool_" in event_text

    def test_task_tool_forces_background_false(self):
        """Task tool with run_in_background=true is forced to false."""
        provider = _make_provider()
        from providers.nvidia_nim.utils import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
        args = json.dumps({"run_in_background": True, "prompt": "test"})
        tc = {
            "index": 0,
            "id": "call_task",
            "function": {"name": "Task", "arguments": args},
        }
        events = list(provider._process_tool_call(tc, sse))
        event_text = "".join(events)
        # The intercepted args should have run_in_background=false
        assert "false" in event_text.lower()

    def test_task_tool_invalid_json_logs_warning(self):
        """Invalid JSON args for Task tool doesn't crash."""
        provider = _make_provider()
        from providers.nvidia_nim.utils import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
        tc = {
            "index": 0,
            "id": "call_task2",
            "function": {"name": "Task", "arguments": "not json"},
        }
        assert list(provider._process_tool_call(tc, sse)) == []
        flushed = "".join(provider._flush_pending_tool_calls(sse))
        assert '"name":"Task"' in flushed
        assert "not json" in flushed

    def test_negative_tool_index_fallback(self):
        """tc_index < 0 uses len(tool_indices) as fallback."""
        provider = _make_provider()
        from providers.nvidia_nim.utils import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
        tc = {
            "index": -1,
            "id": "call_neg",
            "function": {"name": "test", "arguments": "{}"},
        }
        events = list(provider._process_tool_call(tc, sse))
        # Should not crash, should still emit events
        assert len(events) > 0

    def test_tool_args_emitted_as_delta(self):
        """Arguments are emitted as input_json_delta events."""
        provider = _make_provider()
        from providers.nvidia_nim.utils import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")
        tc = {
            "index": 0,
            "id": "call_args",
            "function": {"name": "grep", "arguments": '{"pattern": "test"}'},
        }
        events = list(provider._process_tool_call(tc, sse))
        event_text = "".join(events)
        assert "input_json_delta" in event_text

    def test_tool_name_split_is_buffered_until_arguments_arrive(self):
        """Split tool names should not emit partial tool_use names."""
        provider = _make_provider()
        from providers.nvidia_nim.utils import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")

        first = {
            "index": 0,
            "id": "call_edit",
            "function": {"name": "Ed", "arguments": ""},
        }
        second = {
            "index": 0,
            "id": "call_edit",
            "function": {"name": "it", "arguments": '{"file_path":"a.txt"}'},
        }

        first_events = list(provider._process_tool_call(first, sse))
        second_events = list(provider._process_tool_call(second, sse))

        assert first_events == []
        event_text = "".join(second_events)
        assert '"name":"Edit"' in event_text
        assert '"name":"Ed"' not in event_text

    def test_tool_arguments_before_name_flush_when_name_arrives_late(self):
        """Buffered arguments should flush once the tool name is finally known."""
        provider = _make_provider()
        from providers.nvidia_nim.utils import SSEBuilder

        sse = SSEBuilder("msg_test", "test-model")

        args_first = {
            "index": 0,
            "id": "call_write",
            "function": {"name": None, "arguments": '{"file_path":"a.txt"}'},
        }
        name_late = {
            "index": 0,
            "id": "call_write",
            "function": {"name": "Write", "arguments": ""},
        }

        assert list(provider._process_tool_call(args_first, sse)) == []
        assert list(provider._process_tool_call(name_late, sse)) == []

        flushed = "".join(provider._flush_pending_tool_calls(sse))
        assert '"name":"Write"' in flushed
        assert '\\"file_path\\":\\"a.txt\\"' in flushed
