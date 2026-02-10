"""Request builder for NVIDIA NIM provider."""

from typing import Any, Dict

from config.nim import NimSettings
from .utils.message_converter import AnthropicToOpenAIConverter


def _set_if_not_none(body: Dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        body[key] = value


def _set_extra(
    extra_body: Dict[str, Any], key: str, value: Any, ignore_value: Any = None
) -> None:
    if key in extra_body:
        return
    if value is None:
        return
    if ignore_value is not None and value == ignore_value:
        return
    extra_body[key] = value


def _convert_tool_choice(tool_choice: Any) -> Any:
    """Convert Anthropic tool_choice to OpenAI format.

    Anthropic: {"type": "auto", "disable_parallel_tool_use": true}
    OpenAI: "auto" or {"type": "function", "function": {"name": "..."}}
    """
    if not tool_choice:
        return None

    # Try to convert to dict if it's a Pydantic model or similar
    data = tool_choice
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    elif hasattr(data, "to_dict"):
        data = data.to_dict()

    if isinstance(data, dict):
        tool_type = data.get("type")

        if tool_type in ("auto", "required", "none"):
            return tool_type
        elif tool_type == "function":
            # OpenAI function format
            fn = data.get("function", {})
            fn_name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
            if fn_name:
                return {"type": "function", "function": {"name": fn_name}}

    # If already a string, pass through
    if isinstance(data, str):
        return data

    # Fallback: return as string
    return "auto"


def build_request_body(
    request_data: Any, nim: NimSettings, stream: bool = False
) -> dict:
    """Build OpenAI-format request body from Anthropic request."""
    messages = AnthropicToOpenAIConverter.convert_messages(request_data.messages)

    # Add system prompt
    system = getattr(request_data, "system", None)
    if system:
        system_msg = AnthropicToOpenAIConverter.convert_system_prompt(system)
        if system_msg:
            messages.insert(0, system_msg)

    body: Dict[str, Any] = {
        "model": request_data.model,
        "messages": messages,
    }

    # max_tokens with optional cap
    max_tokens = getattr(request_data, "max_tokens", None)
    if max_tokens is None:
        max_tokens = nim.max_tokens
    elif nim.max_tokens:
        max_tokens = min(max_tokens, nim.max_tokens)
    _set_if_not_none(body, "max_tokens", max_tokens)

    req_temperature = getattr(request_data, "temperature", None)
    temperature = req_temperature if req_temperature is not None else nim.temperature
    _set_if_not_none(body, "temperature", temperature)

    req_top_p = getattr(request_data, "top_p", None)
    top_p = req_top_p if req_top_p is not None else nim.top_p
    _set_if_not_none(body, "top_p", top_p)

    stop_sequences = getattr(request_data, "stop_sequences", None)
    if stop_sequences:
        body["stop"] = stop_sequences
    elif nim.stop:
        body["stop"] = nim.stop

    tools = getattr(request_data, "tools", None)
    if tools:
        body["tools"] = AnthropicToOpenAIConverter.convert_tools(tools)
        tool_choice = getattr(request_data, "tool_choice", None)
        if tool_choice:
            body["tool_choice"] = _convert_tool_choice(tool_choice)
        else:
            body["tool_choice"] = "auto"

    if nim.presence_penalty != 0.0:
        body["presence_penalty"] = nim.presence_penalty
    if nim.frequency_penalty != 0.0:
        body["frequency_penalty"] = nim.frequency_penalty
    if nim.seed is not None:
        body["seed"] = nim.seed

    # Many NIM models don't support parallel_tool_calls or fail with it
    # Only set if explicitly requested and model is known to support it
    # body["parallel_tool_calls"] = nim.parallel_tool_calls

    # Handle non-standard parameters via extra_body
    extra_body: Dict[str, Any] = {}
    
    thinking = getattr(request_data, "thinking", None)
    if thinking and getattr(thinking, "enabled", True):
        extra_body["thinking"] = {"type": "enabled"}
        # Preserve richer reasoning fields for models that commonly support them.
        if "kimi" in body["model"].lower() or "thinking" in body["model"].lower():
            extra_body["reasoning_split"] = True
            extra_body["chat_template_kwargs"] = {
                "thinking": True,
                "reasoning_split": True,
                "clear_thinking": False,
            }
            extra_body["include_reasoning"] = True
            extra_body["reasoning_effort"] = nim.reasoning_effort

    if extra_body:
        body["extra_body"] = extra_body

    return body
