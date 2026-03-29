import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from providers.nvidia_nim.client import NvidiaNimProvider

def create_mock_config():
    config = MagicMock()
    config.api_key = "test-key"
    config.api_keys = ["test-key"]
    config.base_url = "http://test"
    config.rate_limit = 10
    config.rate_window = 60
    config.max_in_flight = 5
    config.key_cooldown_sec = 60
    config.circuit_breaker_threshold = 5
    config.circuit_breaker_recovery = 30.0
    config.max_connections = 100
    config.max_keepalive_connections = 20
    
    config.nim_settings = MagicMock()
    config.nim_settings.max_tokens = 4096
    config.nim_settings.hard_max_tokens = 8192
    config.nim_settings.temperature = 0.7
    config.nim_settings.top_p = 1.0
    config.nim_settings.frequency_penalty = 0.0
    config.nim_settings.presence_penalty = 0.0
    config.nim_settings.seed = None
    config.nim_settings.stop = []
    config.nim_settings.reasoning_effort = "medium"
    return config

@pytest.mark.asyncio
async def test_response_caching():
    """Verify that repeat completion requests use the cache."""
    config = create_mock_config()
    provider = NvidiaNimProvider(config)
    
    # Mock HTTP client
    provider._http_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'{"id": "test-id", "choices": [{"message": {"content": "hello"}}]}'
    provider._http_client.post.return_value = mock_resp
    
    request = MagicMock()
    request.model = "test-model"
    msg = MagicMock()
    msg.role = "user"
    msg.content = "hi"
    request.messages = [msg]
    request.max_tokens = 100
    request.temperature = 0.7
    request.top_p = 1.0
    request.stop_sequences = []
    request.tools = []
    request.thinking = None
    
    # First call - should hit the network
    resp1 = await provider.complete(request)
    assert provider._http_client.post.call_count == 1
    
    # Second call - same request, should hit the cache
    resp2 = await provider.complete(request)
    assert provider._http_client.post.call_count == 1  # Still 1
    assert resp2 == resp1
    
    # Check cache size limit
    provider._max_cache_size = 1
    request2 = MagicMock()
    request2.model = "test-model"
    msg2 = MagicMock()
    msg2.role = "user"
    msg2.content = "different"
    request2.messages = [msg2]
    request2.max_tokens = 100
    request2.temperature = 0.7
    request2.top_p = 1.0
    request2.stop_sequences = []
    request2.tools = []
    request2.thinking = None
    
    await provider.complete(request2)
    assert provider._http_client.post.call_count == 2
    
    # Original request should now be evicted
    await provider.complete(request)
    assert provider._http_client.post.call_count == 3

@pytest.mark.asyncio
async def test_stream_options_in_raw_body():
    """Verify that stream_options={"include_usage": True} is added to the request."""
    config = create_mock_config()
    provider = NvidiaNimProvider(config)
    provider._http_client = MagicMock()
    
    # Mock stream context manager
    mock_stream = AsyncMock()
    mock_stream.__aenter__.return_value = MagicMock()
    mock_stream.__aenter__.return_value.status_code = 200
    mock_stream.__aenter__.return_value.aiter_lines.return_value.__aiter__.return_value = []
    provider._http_client.stream.return_value = mock_stream
    
    request = MagicMock()
    request.model = "test-model"
    msg = MagicMock()
    msg.role = "user"
    msg.content = "hi"
    request.messages = [msg]
    request.max_tokens = 100
    request.temperature = 0.7
    request.top_p = 1.0
    request.stop_sequences = []
    request.tools = []
    request.thinking = None
    
    # Consume the stream
    async for _ in provider.stream_response(request):
        pass
        
    # Verify the body sent to httpx.stream
    args, kwargs = provider._http_client.stream.call_args
    sent_body = json.loads(kwargs["content"])
    
    assert sent_body["stream_options"] == {"include_usage": True}
    assert sent_body["stream"] is True
