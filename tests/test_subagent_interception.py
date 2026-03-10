import json
import pytest
from providers.nvidia_nim.utils import SSEBuilder
from providers.nvidia_nim import NvidiaNimProvider
from providers.base import ProviderConfig


@pytest.mark.asyncio
async def test_task_tool_interception():
    # Setup provider
    config = ProviderConfig(api_key="test")
    provider = NvidiaNimProvider(config)

    sse = SSEBuilder("msg_test", "test-model")

    # Tool call data (Task tool)
    tc = {
        "index": 0,
        "id": "tool_123",
        "function": {
            "name": "Task",
            "arguments": json.dumps(
                {
                    "description": "test task",
                    "prompt": "do something",
                    "run_in_background": True,
                }
            ),
        },
    }

    events = []
    for event in provider._process_tool_call(tc, sse):
        events.append(event)

    event_text = "".join(events)
    assert "input_json_delta" in event_text
    assert '\\"run_in_background\\":false' in event_text.lower()
