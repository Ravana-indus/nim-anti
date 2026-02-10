import pytest
from unittest.mock import patch
from api.models.anthropic import MessagesRequest, TokenCountRequest, Message


def test_messages_request_map_model_claude_to_default():
    with patch(
        "api.models.anthropic.get_active_model",
        return_value="target-model-from-settings",
    ):
        request = MessagesRequest(
            model="claude-3-opus",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )

        assert request.model == "target-model-from-settings"
        assert request.original_model == "claude-3-opus"


def test_messages_request_map_model_non_claude_unchanged():
    with patch(
        "api.models.anthropic.get_active_model",
        return_value="target-model-from-settings",
    ):
        request = MessagesRequest(
            model="gpt-4",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )

        # normalize_model_name returns original if not Claude
        assert request.model == "gpt-4"


def test_messages_request_map_model_with_provider_prefix():
    with patch(
        "api.models.anthropic.get_active_model",
        return_value="target-model-from-settings",
    ):
        request = MessagesRequest(
            model="anthropic/claude-3-haiku",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )

        assert request.model == "target-model-from-settings"


def test_token_count_request_model_validation():
    with patch(
        "api.models.anthropic.get_active_model",
        return_value="target-model-from-settings",
    ):
        request = TokenCountRequest(
            model="claude-3-sonnet", messages=[Message(role="user", content="hello")]
        )

        assert request.model == "target-model-from-settings"


def test_messages_request_model_mapping_logs():
    with (
        patch(
            "api.models.anthropic.get_active_model",
            return_value="target-model-from-settings",
        ),
        patch("api.models.anthropic.logger.debug") as mock_log,
    ):
        MessagesRequest(
            model="claude-2.1",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )

        mock_log.assert_called()
        args = mock_log.call_args[0][0]
        assert "MODEL MAPPING" in args
        assert "claude-2.1" in args
        assert "target-model-from-settings" in args
